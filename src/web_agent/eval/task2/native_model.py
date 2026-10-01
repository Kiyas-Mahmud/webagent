"""Browser Use's real custom-LLM interface, with local-only inference transport."""
import asyncio
import json
from pathlib import Path
import re
import time
from web_agent.eval.task1.core import write_new

# Identical to upstream Browser Use 0.13.10 llm/ollama/chat.py (_JSON_FENCE_RE):
# only a fence enclosing the entire response is removed; the body is never edited.
_JSON_FENCE_RE = re.compile(r'\A```[ \t]*(?:json)?[ \t]*\r?\n(?P<body>.*?)\r?\n?```[ \t]*\Z', re.IGNORECASE | re.DOTALL)


def unwrap_native_fence(content):
    text = content.strip()
    match = _JSON_FENCE_RE.fullmatch(text)
    return match.group('body').strip() if match else text


def strict_object(raw):
    def unique(pairs):
        result = {}
        for key,value in pairs:
            if key in result:raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def invalid(_):raise ValueError('Nonfinite JSON value')
    result = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)
    if not isinstance(result,dict):raise ValueError('Expected one bare JSON object')
    return result


class LocalInternVL:
    model = 'InternVL3.5-8B-HF-local-base'
    provider = 'task2-local'
    name = model
    model_name = model
    _verified_api_keys = True

    def __init__(self, worker, output, budget, label=None):
        if label:
            self.model = self.name = self.model_name = label
        self.worker = worker
        self.output = Path(output)
        self.budget = budget
        self.calls = 0
        self.advice = None
        self.memory_context = []
        self.memory_kind = 'training_labels'
        self.memory_exposures = 0
        self.infrastructure_error = None
        self.validation_feedback = None
        # v3: the previous step's assessment runs while Browser Use captures the next
        # page, and is awaited here, so the actor sees the page at the baseline's moment.
        self.pending = None
        # v3: async callable(completion) -> reason string if the proposal repeats an
        # action the assessor judged failed, else None. It blocks; it never substitutes.
        self.guard = None
        self.guard_regenerations = 2
        self.guard_blocks = 0
        self.guard_overrides = 0
        # v3.1: async callable(advice, followup) -> (context_text, [{'label','action','then'?}])
        # built from page facts after a confident failure. The actor answers with an
        # option number; the chosen action goes through Browser Use's own executor.
        self.options_provider = None
        self.followup = None
        self.option_choices = 0

    async def choose(self, prefix, messages, context, options):
        """Ask the actor to pick one recovery option by number; None means 'decide myself'."""
        image = None
        for message in reversed(messages):
            content = message.model_dump(mode='json', exclude_none=True).get('content')
            parts = [p for p in content if isinstance(p, dict) and p.get('type') == 'image_url'] if isinstance(content, list) else []
            if parts:
                image = parts[-1]
                break
        lines = '\n'.join(f'{k}. {o["label"]}' for k, o in enumerate(options, 1))
        text = (context + '\n\nRecovery options for the next action:\n' + lines +
                '\n0. None of these; I will choose my own action.\n'
                'Reply with only the number of the best option.')
        request = {'messages': [{'role': 'user', 'content': ([image] if image else []) + [{'type': 'text', 'text': text}]}],
                   'output_schema': None, 'advice': None, 'plain': True}
        if not self.budget.available():
            return None
        self.budget.charge('actor_choice')
        result = await self.generate(request)
        match = re.search(r'\d+', result['raw_response'])
        number = int(match.group()) if match else 0
        picked = options[number-1] if 1 <= number <= len(options) else None
        write_new(str(prefix)+'-choice.json', {'context': context, 'options': options, 'raw_response': result['raw_response'],
                                               'chosen': number, 'picked': picked})
        return picked

    async def ready(self):
        if self.pending is not None:
            pending, self.pending = self.pending, None
            await pending

    async def generate(self, request):
        try:
            return await asyncio.to_thread(self.worker.call,'generate',request=request)
        except Exception as exc:
            self.infrastructure_error=str(exc)
            raise

    def parse(self, raw, output_format):
        if not output_format:
            return raw
        value = strict_object(unwrap_native_fence(raw))
        # Preserve Browser Use's proposal contract. Its own
        # max_actions_per_step=1 selects the first action before execution,
        # and its own empty-action retry/noop handles an empty list.
        if not isinstance(value.get('action'),list):
            raise ValueError('A native action list is required')
        return output_format.model_validate(value, strict=True)

    async def ainvoke(self, messages, output_format=None, **kwargs):
        from browser_use.llm.views import ChatInvokeCompletion
        await self.ready()
        self.budget.charge('actor')
        self.calls += 1
        prefix = self.output / f'actor-{self.calls:04d}'
        request = {'messages':[m.model_dump(mode='json', exclude_none=True) for m in messages],
                   'output_schema':output_format.model_json_schema() if output_format else None,
                   'advice':self.advice, 'previous_proposal_rejection':self.validation_feedback}
        write_new(str(prefix)+'-started.json', {'request':request, 'started_at':time.time()})
        started = time.monotonic()
        if output_format and self.options_provider:
            offer = await self.options_provider(self.advice, self.followup)
            self.followup = None
            if offer:
                picked = await self.choose(prefix, messages, *offer)
                if picked is not None:
                    self.option_choices += 1
                    self.followup = picked.get('then')
                    completion = output_format.model_validate({
                        'evaluation_previous_goal': 'The failure monitor reported that the last action made no progress.',
                        'memory': 'Chose recovery option: ' + picked['label'],
                        'next_goal': picked['label'], 'action': [picked['action']]}, strict=True)
                    write_new(str(prefix)+'-parsed.json', {'status':'valid','recovery_option':picked,
                        'native_proposal':completion.model_dump(mode='json',exclude_none=True)})
                    self.validation_feedback = None
                    return ChatInvokeCompletion(completion=completion, usage=None, stop_reason='end_turn')
        try:
            result = await self.generate(request)
            result['latency_seconds'] = time.monotonic()-started
            write_new(str(prefix)+'-raw.json', result)
            if not self.budget.available(0):
                raise ValueError('Episode deadline reached before browser execution')
            note = isinstance(self.advice, dict) and self.advice.get('schema') == 'task2.recovery_note.v3'
            if self.memory_context and not note:
                # Complete B-style raw decision is durable before memory changes generation.
                # Empty admission follows exactly the original request and model result.
                self.budget.charge('memory_actor')
                if self.memory_kind == 'experience':
                    context = dict(retrieved_past_experiences=self.memory_context,
                        memory_instruction='Past failure-recovery experiences of this agent on similar pages. Outcomes are learned assessments, not guaranteed. Reuse what worked, avoid what did not, and ground your action in the current page.')
                else:
                    context = dict(retrieved_training_examples=self.memory_context,
                        memory_instruction='Advisory training labels only; missing values/reflections are unknown. Ground your action in the current page.')
                memory_request=dict(request, advice=dict(request['advice'] or {}, **context))
                write_new(str(prefix)+'-memory-started.json', {'request':memory_request,'started_at':time.time()})
                result=await self.generate(memory_request)
                write_new(str(prefix)+'-memory-raw.json',result)
                self.memory_exposures+=1
                if not self.budget.available(0):
                    raise ValueError('Episode deadline reached before browser execution')
            completion = self.parse(result['raw_response'], output_format)
            for attempt in range(self.guard_regenerations if self.guard and output_format else 0):
                reason = await self.guard(completion)
                if reason is None:
                    break
                self.guard_blocks += 1
                write_new(str(prefix)+f'-guard-{attempt}.json', {'blocked_proposal':completion.model_dump(mode='json',exclude_none=True),
                                                                 'reason':reason})
                if not self.budget.available():
                    break
                self.budget.charge('actor_guard')
                result = await self.generate(dict(request, previous_proposal_rejection={'stage':'recovery_guard','reason':reason}))
                write_new(str(prefix)+f'-guard-{attempt}-raw.json', result)
                completion = self.parse(result['raw_response'], output_format)
            else:
                if self.guard and output_format and self.guard_regenerations and await self.guard(completion):
                    # Still the blocked action: let it run rather than turn it into a failure.
                    self.guard_overrides += 1
            write_new(str(prefix)+'-parsed.json', {'status':'valid',
                'native_proposal':completion.model_dump(mode='json',exclude_none=True) if hasattr(completion,'model_dump') else completion})
            self.validation_feedback = None
            return ChatInvokeCompletion(completion=completion, usage=None,
                stop_reason='max_tokens' if result['limit_hit'] else 'end_turn')
        except Exception as exc:
            write_new(str(prefix)+'-error.json', {'error_type':type(exc).__name__, 'error':str(exc)})
            if not self.infrastructure_error:
                self.validation_feedback = {'stage':'structured_output_validation',
                    'proposal_id':prefix.name,'error_type':type(exc).__name__,
                    'reason':str(exc), 'rejected_response':locals().get('result',{}).get('raw_response')}
            raise

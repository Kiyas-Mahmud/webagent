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

    async def generate(self, request):
        try:
            return await asyncio.to_thread(self.worker.call,'generate',request=request)
        except Exception as exc:
            self.infrastructure_error=str(exc)
            raise

    async def ainvoke(self, messages, output_format=None, **kwargs):
        from browser_use.llm.views import ChatInvokeCompletion
        self.budget.charge('actor')
        self.calls += 1
        prefix = self.output / f'actor-{self.calls:04d}'
        request = {'messages':[m.model_dump(mode='json', exclude_none=True) for m in messages],
                   'output_schema':output_format.model_json_schema() if output_format else None,
                   'advice':self.advice, 'previous_proposal_rejection':self.validation_feedback}
        write_new(str(prefix)+'-started.json', {'request':request, 'started_at':time.time()})
        started = time.monotonic()
        try:
            result = await self.generate(request)
            result['latency_seconds'] = time.monotonic()-started
            write_new(str(prefix)+'-raw.json', result)
            if not self.budget.available(0):
                raise ValueError('Episode deadline reached before browser execution')
            if self.memory_context:
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
            raw = result['raw_response']
            if output_format:
                value = strict_object(unwrap_native_fence(raw))
                # Preserve Browser Use's proposal contract. Its own
                # max_actions_per_step=1 selects the first action before execution,
                # and its own empty-action retry/noop handles an empty list.
                if not isinstance(value.get('action'),list):
                    raise ValueError('A native action list is required')
                completion = output_format.model_validate(value, strict=True)
            else:
                completion = raw
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

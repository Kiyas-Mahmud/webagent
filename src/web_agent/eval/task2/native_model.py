"""Browser Use's real custom-LLM interface, with local-only inference transport."""
import asyncio
import json
from pathlib import Path
import time
from web_agent.eval.task1.core import write_new


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

    def __init__(self, worker, output, budget):
        self.worker = worker
        self.output = Path(output)
        self.budget = budget
        self.calls = 0
        self.advice = None
        self.memory_context = []
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
                memory_request=dict(request, advice=dict(request['advice'] or {},
                    retrieved_training_examples=self.memory_context,
                    memory_instruction='Advisory training labels only; missing values/reflections are unknown. Ground your action in the current page.'))
                write_new(str(prefix)+'-memory-started.json', {'request':memory_request,'started_at':time.time()})
                result=await self.generate(memory_request)
                write_new(str(prefix)+'-memory-raw.json',result)
                self.memory_exposures+=1
                if not self.budget.available(0):
                    raise ValueError('Episode deadline reached before browser execution')
            raw = result['raw_response']
            if output_format:
                value = strict_object(raw)
                # Preserve Browser Use's proposal contract. Its own
                # max_actions_per_step=1 selects the first action before execution.
                if not isinstance(value.get('action'),list) or not value['action']:
                    raise ValueError('A nonempty native action list is required')
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

"""Langfuse tracing: every LLM call as a generation (prompt, reply, tokens, latency), grouped under the request
that caused it, with automatic scores. Off unless LANGFUSE_PUBLIC_KEY is set (./run.sh api sets it)."""
import os
from contextlib import contextmanager, nullcontext


class LangfuseTracer:
    def __init__(self):
        from langfuse import Langfuse

        self.client = Langfuse()

    @contextmanager
    def generation(self, model: str, messages: list, options: dict):
        with self.client.start_as_current_observation(
            name="ollama.chat", as_type="generation", model=model, input=messages, model_parameters=options
        ) as gen:
            yield lambda reply: gen.update(
                output=reply["message"]["content"],
                usage_details={"input": reply.get("prompt_eval_count", 0), "output": reply.get("eval_count", 0)},
            )

    def span(self, name: str, **input):
        return self.client.start_as_current_observation(name=name, as_type="chain", input=input)

    def score(self, name: str, value: float, comment: str | None = None):
        self.client.score_current_trace(name=name, value=value, comment=comment)


class NoTracer:
    def span(self, name, **input):
        return nullcontext(_Quiet())

    def score(self, name, value, comment=None):
        pass


class _Quiet:
    def update(self, **kw):
        pass


def tracer():
    if not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        return NoTracer()
    try:
        return LangfuseTracer()
    except ImportError:  # the mlops dependency group isn't installed
        return NoTracer()

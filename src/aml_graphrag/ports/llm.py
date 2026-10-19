from typing import Protocol


class LLM(Protocol):
    name: str

    def json(self, prompt: str, schema: dict, system: str | None = None) -> dict: ...

    def text(self, prompt: str, system: str | None = None) -> str: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

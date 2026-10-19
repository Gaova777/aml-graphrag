import json

import httpx


class OllamaLLM:
    def __init__(self, model: str = "qwen3:4b-instruct", url: str = "http://localhost:11434",
                 num_ctx: int = 10240, temperature: float = 0.0, tracer=None):
        self.name = model
        self.url = url
        self.options = {"num_ctx": num_ctx, "temperature": temperature}
        self.client = httpx.Client(timeout=600)
        self.tracer = tracer

    def json(self, prompt: str, schema: dict, system: str | None = None) -> dict:
        return json.loads(self._chat(prompt, system, schema))

    def text(self, prompt: str, system: str | None = None) -> str:
        return self._chat(prompt, system, None)

    def _chat(self, prompt, system, schema):
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {"model": self.name, "messages": messages, "stream": False, "think": False, "options": self.options}
        if schema:
            body["format"] = schema
        resp = self.client.post(f"{self.url}/api/chat", json=body)
        resp.raise_for_status()
        reply = resp.json()
        if self.tracer:
            self.tracer.generation(self.name, messages, reply)
        return reply["message"]["content"]


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", url: str = "http://localhost:11434"):
        self.model = model
        self.url = url
        self.client = httpx.Client(timeout=300)

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.post(f"{self.url}/api/embed", json={"model": self.model, "input": texts})
        resp.raise_for_status()
        return resp.json()["embeddings"]

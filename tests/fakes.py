class FakeLLM:
    """Replies from a list, in order, and remembers the prompts it got."""

    name = "fake"

    def __init__(self, json_replies=(), text_replies=()):
        self.json_replies = list(json_replies)
        self.text_replies = list(text_replies)
        self.prompts = []

    def json(self, prompt, schema, system=None):
        self.prompts.append(prompt)
        return self.json_replies.pop(0)

    def text(self, prompt, system=None):
        self.prompts.append(prompt)
        return self.text_replies.pop(0)


class FakeEmbedder:
    """Bag of letters: texts that share letters are close."""

    def embed(self, texts):
        return [[t.lower().count(c) + 1e-3 for c in "abcdefghijklmnopqrstuvwxyz"] for t in texts]

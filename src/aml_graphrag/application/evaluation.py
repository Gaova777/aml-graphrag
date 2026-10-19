"""The paper's evaluation (section 4.1): questions from users and tasks, and an LLM judge comparing two answers."""
from aml_graphrag.ports.llm import LLM

CRITERIA = {
    "comprehensiveness": "How much detail does the answer provide to cover all aspects and details of the question?",
    "diversity": "How varied and rich is the answer in providing different perspectives and insights on the question?",
    "empowerment": "How well does the answer help the reader understand and make informed judgements about the topic?",
    "directness": "How specifically and clearly does the answer address the question?",
}

USERS_SCHEMA = {
    "type": "object",
    "properties": {"users": {"type": "array", "items": {"type": "object", "properties": {
        "user": {"type": "string"},
        "tasks": {"type": "array", "items": {"type": "string"}},
    }, "required": ["user", "tasks"]}}},
    "required": ["users"],
}

USERS_PROMPT = """Here is a description of a dataset: {description}

Name {n_users} potential users of this dataset (each one a short job description) and, for each user, {n_tasks} tasks
they would do with it. Tasks should need an understanding of the whole dataset, not a single fact."""

QUESTIONS_SCHEMA = {
    "type": "object",
    "properties": {"questions": {"type": "array", "items": {"type": "string"}}},
    "required": ["questions"],
}

QUESTIONS_PROMPT = """Here is a description of a dataset: {description}

User: {user}
Task: {task}

Write {n} questions this user would ask to do this task. Each question must need an understanding of the whole
dataset (themes, patterns, comparisons across many documents), not a fact that one sentence could answer.
Do not name specific people or cases unless the description does."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {name: {"type": "object", "properties": {
        "reasoning": {"type": "string"}, "winner": {"type": "integer", "enum": [0, 1, 2]},
    }, "required": ["reasoning", "winner"]} for name in CRITERIA},
    "required": list(CRITERIA),
}

JUDGE_PROMPT = """You are judging two answers to the same question. For each criterion, explain briefly and say which
answer is better: 1, 2, or 0 if they are equally good. Judge the content, not the length alone.

Criteria:
{criteria}

Question: {question}

Answer 1:
{a}

Answer 2:
{b}"""


def users_and_tasks(llm: LLM, description: str, n_users: int, n_tasks: int) -> list[dict]:
    return llm.json(USERS_PROMPT.format(description=description, n_users=n_users, n_tasks=n_tasks), USERS_SCHEMA)["users"]


def questions_for(llm: LLM, description: str, user: str, task: str, n: int) -> list[str]:
    reply = llm.json(QUESTIONS_PROMPT.format(description=description, user=user, task=task, n=n), QUESTIONS_SCHEMA)
    return reply["questions"][:n]


def judge(llm: LLM, question: str, a: str, b: str) -> dict:
    """{criterion: 1 | 2 | 0} for answers a (1) and b (2)."""
    criteria = "\n".join(f"- {name}: {text}" for name, text in CRITERIA.items())
    reply = llm.json(JUDGE_PROMPT.format(criteria=criteria, question=question, a=a, b=b), JUDGE_SCHEMA)
    return {name: reply[name]["winner"] for name in CRITERIA}

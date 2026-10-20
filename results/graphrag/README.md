# Reproducing GraphRAG on a laptop-sized corpus

Edge et al. (2024) compare their global search over community summaries against plain vector RAG on two corpora of
about one million tokens, judged head to head by an LLM. Here I repeat that comparison: the same pipeline and the
same evaluation, rebuilt from the paper, at a scale that fits a 4B model on an 8 GB GPU.

```
uv run scripts/make_questions.py     # questions.json
uv run scripts/answer.py             # answers.jsonl
uv run scripts/judge.py              # judgements_qwen3_8b.jsonl, win_rates_qwen3_8b.csv
```

## Setup, next to the paper's

| | Edge et al. 2024 | here |
|---|---|---|
| corpus | podcast transcripts (~1 M tokens), news (~1.7 M) | 106 Wikipedia articles on money laundering, ~204 k tokens (first 2,400 tokens of each) |
| chunks | 600 tokens | 600 tokens, 100 overlap: 434 chunks |
| model | GPT-4-turbo | qwen3:4b-instruct, local |
| graph | entities, relationships, claims | entities and relationships (no claims, no gleaning): 8,136 entities, 9,200 relationships, 78% in the main component |
| communities | hierarchical Leiden, C0 to C3 | hierarchical Leiden (max size 10), C0: 51 and C1: 503 |
| context window | 8k tokens | 3k tokens for every condition |
| questions | 125 per corpus (5 users × 5 tasks × 5) | 42 (5 users, their tasks, 2 per task), from a description of the corpus only |
| judge | the same LLM, 5 repetitions | a different model (qwen3:8b), each pair judged in **both orders** |

## Results

Win rate of the row's method over the column's, averaged over questions (a win counts 1, a tie 0.5), each
comparison judged once with each answer first ([`win_rates_qwen3_8b.csv`](win_rates_qwen3_8b.csv)):

| | questions | comprehensiveness | diversity | empowerment | directness | verdict flips with order |
|---|---:|---:|---:|---:|---:|---:|
| **C0 vs vector RAG** | 42 | **77%** | **89%** | **82%** | 74% | 22% |
| C1 vs vector RAG | 12 | 96% | 100% | 100% | 100% | 2% |
| C1 vs C0 | 12 | 62% | 71% | 62% | 83% | 52% |

The paper reports 72–83% comprehensiveness and 62–82% diversity for GraphRAG over vector RAG. **The main result
holds**: summaries of the whole corpus beat the passages closest to the question on global questions.

Cost per answer (medians; three requests shared the GPU, so seconds are inflated equally):

| | LLM calls | seconds | words |
|---|---:|---:|---:|
| vector RAG | 1 | 18 | 352 |
| C0 (51 root communities) | 13 | 256 | 701 |
| C1 (503 communities) | 89 | 1,581 | 844 |

C1 costs seven times what C0 costs and buys little: against C0 the judge changes its verdict half of the time when
only the order changes, so it can barely tell them apart. That matches the paper's practical advice: **the root
level gives most of the benefit for a fraction of the tokens.** Because of that cost, C1 was run on the first 12
questions only.

## Why vector RAG loses, and how much of it is the judge

- **Half of the vector RAG answers (21 of 42) start by saying the passages don't contain enough to answer.** That is
  the paper's argument in one number: for "how do techniques differ across banks, exchanges and cartels?", the eight
  most similar chunks are about one case each. Only 3 of the 42 C0 answers say that.
- On the 21 questions vector RAG did answer, C0 still wins: comprehensiveness 65%, diversity 88%,
  empowerment 76%, directness 60%.
- **Directness doesn't behave like in the paper.** There, vector RAG was the most direct, a sign that the judge isn't
  just rewarding length. Here C0 wins directness too (60% even when vector RAG answered), and GraphRAG's answers are
  twice as long. So part of these wins is probably a judge that likes longer, more complete-looking answers. Judging in
  both orders removes the position bias (the verdict flipped 22% of the time for C0 vs vector RAG) but not a length bias.

## Caveats

- One corpus, ~5 times smaller than the paper's smallest. Vector RAG's 3k-token context covers about 1.5% of it;
  with a bigger corpus that share shrinks and GraphRAG's advantage should grow, but I didn't test it.
- Questions and answers come from the same 4B model, and only one judge. 42 questions is enough to see a 77% win rate
  but not to separate close conditions like C1 vs C0.
- No claims extraction and no gleaning (the paper's extra extraction passes), to keep extraction at about 1.7 hours (plus about 1.2 hours of community reports).
- Like the paper, this measures how answers *look* to a judge, not whether they are true. That's the next part:
  [results/analyst](../analyst).

"""Download the Wikipedia articles listed in data/corpus/articles.txt.

Texts go to data/corpus/raw/ (not committed); the manifest with each revision id is committed, so the
exact same corpus can be fetched again with --pinned.

usage: uv run scripts/fetch_corpus.py
"""
import argparse
import json
import time
from pathlib import Path

from aml_graphrag.adapters.wikipedia import Wikipedia


def main(args):
    titles = [t.strip() for t in args.list.read_text().splitlines() if t.strip() and not t.startswith("#")]
    args.raw.mkdir(parents=True, exist_ok=True)
    wiki = Wikipedia()
    manifest = []
    for title in titles:
        page = wiki.article(title)
        (args.raw / f"{page['revision']}.txt").write_text(page["text"])
        manifest.append({k: page[k] for k in ("title", "revision", "url")} | {"words": len(page["text"].split())})
        print(f"  {page['title']:55} {manifest[-1]['words']:>6} words")
        time.sleep(1)
    args.manifest.write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(manifest)} articles, {sum(m['words'] for m in manifest):,} words")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", type=Path, default=Path("data/corpus/articles.txt"))
    parser.add_argument("--raw", type=Path, default=Path("data/corpus/raw"))
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus/manifest.json"))
    main(parser.parse_args())

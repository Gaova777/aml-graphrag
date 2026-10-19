import time

import httpx

API = "https://en.wikipedia.org/w/api.php"
# Wikipedia throttles anonymous clients that don't say who they are
USER_AGENT = "aml-graphrag/0.1 (https://github.com/Gaova777/aml-graphrag)"


class Wikipedia:
    def __init__(self, client: httpx.Client | None = None, retries: int = 4):
        self.client = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30)
        self.retries = retries

    def article(self, title: str) -> dict:
        """Plain text of the current revision, following redirects."""
        params = {
            "action": "query", "format": "json", "formatversion": 2, "redirects": 1, "titles": title,
            "prop": "extracts|revisions|info", "explaintext": 1, "rvprop": "ids", "inprop": "url",
        }
        page = self._get(params)["query"]["pages"][0]
        if page.get("missing"):
            raise LookupError(f"no Wikipedia article called {title!r}")
        return {
            "title": page["title"],
            "revision": page["revisions"][0]["revid"],
            "url": page["fullurl"],
            "text": page["extract"],
        }

    def _get(self, params: dict) -> dict:
        for attempt in range(self.retries):
            resp = self.client.get(API, params=params)
            if resp.status_code != 429:
                resp.raise_for_status()
                return resp.json()
            time.sleep(2 ** attempt * 5)
        resp.raise_for_status()

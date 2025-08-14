from __future__ import annotations

from typing import List
import json
import urllib.parse
import urllib.request

from retrieval.base import RetrievalDoc


WIKI_API = "https://en.wikipedia.org/w/rest.php/v1/search/page"


def search_wikipedia(query: str, k: int = 3) -> List[RetrievalDoc]:
	params = urllib.parse.urlencode({"q": query, "limit": k})
	url = f"{WIKI_API}?{params}"
	req = urllib.request.Request(url, headers={"User-Agent": "VEGETA/0.1"})
	with urllib.request.urlopen(req, timeout=10) as resp:
		data = json.loads(resp.read().decode("utf-8"))
	items = data.get("pages", []) if isinstance(data, dict) else []
	docs: List[RetrievalDoc] = []
	for it in items:
		title = it.get("title")
		snippet = it.get("excerpt")
		url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}" if title else None
		text = f"{title}\n{snippet}" if snippet else (title or "")
		meta = {"source": "wikipedia", "title": title, "url": url}
		docs.append(RetrievalDoc(text=text, metadata=meta))
	return docs



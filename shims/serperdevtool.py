# shims/serperdevtool.py
import os
import requests
from typing import Any, Dict, Optional
from crewai.tools import BaseTool

class SerperDevTool(BaseTool):
    name = "SerperDevTool"
    description = "Google search via Serper.dev. Returns JSON like crewai_tools’ SerperDevTool."

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        if not self.api_key:
            raise RuntimeError("SERPER_API_KEY not set")

    # CrewAI BaseTool calls run() which delegates to _run(); keep both for compatibility
    def run(self, *, search_query: str, **kwargs) -> Dict[str, Any]:
        return self._run(search_query=search_query, **kwargs)

    def _run(self, search_query: str, **kwargs) -> Dict[str, Any]:
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        payload = {"q": search_query}
        # pass-through optional params if you want (hl, gl, num, etc.)
        payload.update({k: v for k, v in kwargs.items() if k in {"hl", "gl", "num"}})
        resp = requests.post(url, headers=headers, json=payload, timeout=20)
        resp.raise_for_status()
        return resp.json()

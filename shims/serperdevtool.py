import os
import requests
from typing import Optional, Any, Dict

class SerperDevTool:
    name: str = "SerperDevTool"
    description: str = "Search via Serper (Google) API; returns JSON."

    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None):
        self.api_key = api_key or os.getenv("SERPER_API_KEY")
        self.endpoint = endpoint or "https://google.serper.dev/search"

    def run(self, *, search_query: str, **kwargs: Any) -> Dict[str, Any]:
        if not self.api_key:
            return {"organic": []}
        payload: Dict[str, Any] = {"q": search_query}
        for k, v in kwargs.items():
            if v is not None:
                payload[k] = v
        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}
        resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()

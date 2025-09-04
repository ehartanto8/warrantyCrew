import os
import inspect
import json
import requests
from dotenv import load_dotenv
from crewai import Agent, Task, Crew

# env
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

try:
    # prefer real package
    from crewai_tools import SerperDevTool
    _HAS_CREWAI_TOOLS = True
except Exception:
    _HAS_CREWAI_TOOLS = False

    # Minimal inline fallback if crewai_tools isn't available
    class SerperDevTool:
        def __init__(self):
            key = os.getenv("SERPER_API_KEY")
            if not key:
                raise RuntimeError("SERPER_API_KEY is required")
            self._key = key

        # Accept BOTH, normalize to q
        def run(self, query = None, search_query = None, **kwargs):
            q = search_query or query
            if not q:
                raise RuntimeError("SEARCH_QUERY is required")
            r = requests.post(
                "https://google.serper.dev/search",
                headers = {"X-API-KEY": self._key, "Content-Type": "application/json"},
                json = {"q": q},
                timeout = 20,
            )
            r.raise_for_status()
            return r.json()

# VM Support Site search tool (used directly, not passed into Agent.tools)
class KBSearchTool(SerperDevTool):
    def _call_serper(self, q, **kwargs):
        # If using the real crewai_tools impl, try both param names
        if _HAS_CREWAI_TOOLS:
            try:
                sig = inspect.signature(super().run)
                if "search_query" in sig.parameters:
                    return super().run(search_query = q, **kwargs)
                if "query" in sig.parameters:
                    return super().run(query = q, **kwargs)
            except Exception:
                pass

        # Fallback: direct HTTP
        key = os.getenv("SERPER_API_KEY")
        if not key:
            raise RuntimeError("SERPER_API_KEY is required for Serper")
        r = requests.post(
            "https://google.serper.dev/search",
            headers = {"X-API-KEY": key, "Content-Type": "application/json"},
            json = {"q": q},
            timeout = 20,
        )
        r.raise_for_status()
        return r.json()

    # Accept BOTH `query` and `search_query`
    def run(self, query = None, search_query = None, **kwargs):
        q = search_query or query
        if not q:
            raise ValueError("query is required")

        kb_query = f"site:support.vanmetrehomes.com {q}"
        if os.getenv("DEBUG_TOOLS") == "1":
            print(f"[KBSearchTool] q={q} kb_query={kb_query}")

        kb_results = self._call_serper(kb_query, **kwargs)

        organic = (kb_results or {}).get("organic") or []
        if organic:
            kb_results["organic"] = sorted(organic, key = lambda r: r.get("position", float("inf")))
            return kb_results

        if os.getenv("DEBUG_TOOLS") == "1":
            print("[KBSearchTool] no KB hits, falling back to broad web query")
        return self._call_serper(q, **kwargs)


# Optional local PDF/doc embeddings search
try:
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma

    class DocumentSearchTool:
        def __init__(self, docs_path = "./data"):
            self.vectorstore = None
            if not os.path.isdir(docs_path):
                return
            loader = DirectoryLoader(docs_path, glob = "**/*.pdf")
            docs = loader.load()
            if not docs:
                return
            self.vectorstore = Chroma.from_documents(docs, embedding = OpenAIEmbeddings())

        def run(self, query: str, k: int = 4):
            if not getattr(self, "vectorstore", None):
                return []
            results = self.vectorstore.similarity_search(query, k = k)
            return [{"page_content": d.page_content, "metadata": d.metadata} for d in results]
except ImportError:
    DocumentSearchTool = None


def _format_kb_block(kb_json) -> str:
    organic = (kb_json or {}).get("organic") or []
    lines = []
    for item in organic[:5]:
        title   = (item.get("title") or "").strip()
        link    = (item.get("link") or item.get("url") or "").strip()
        snippet = (item.get("snippet") or item.get("description") or "").strip()
        if title or link or snippet:
            lines.append(f"- {title}\n  {snippet}\n  Source: {link}")
    return "\n".join(lines) if lines else "No KB results found."


def _format_doc_block(doc_json_list) -> str:
    if not doc_json_list:
        return "No document results found."
    lines = []
    for d in doc_json_list[:4]:
        meta    = d.get("metadata") or {}
        source  = meta.get("source") or meta.get("file_path") or ""
        page    = meta.get("page", "")
        snippet = (d.get("page_content") or "").strip().replace("\n", " ")
        if len(snippet) > 400:
            snippet = snippet[:400] + "..."
        src_str = f"{source}" + (f" (page {page})" if page != "" else "")
        lines.append(f"- {snippet}\n  Source: {src_str}")
    return "\n".join(lines)


# Agent wrapper used by orchestrator
class HomeownerHelpAgent:
    def __init__(self):
        self.kb_tool = KBSearchTool()
        self.doc_tool = None

        if DocumentSearchTool:
            try:
                tmp = DocumentSearchTool()
                if getattr(tmp, "vectorstore", None):
                    self.doc_tool = tmp
            except Exception:
                self.doc_tool = None

        # NOTE: We DO NOT pass tools to the Agent to avoid BaseTool validation errors.
        self.agent = Agent(
            role = "Homeowner Support Assistant",
            goal = "Answer homeowner troubleshooting questions using the support knowledge base and optional uploaded docs.",
            backstory = "You help homeowners by finding relevant KB pages and turning them into clear, actionable steps with citations.",
            verbose = False,
        )

    def run(self, question: str) -> str:
        # 1) Search KB (direct call, not via Agent.tools)
        kb_json   = {}
        try:
            kb_json = self.kb_tool.run(search_query = question)
        except Exception as e:
            kb_json = {"error": str(e)}

        # 2) Optional doc search
        docs_json = []
        if self.doc_tool:
            try:
                docs_json = self.doc_tool.run(question)
            except Exception as e:
                docs_json = [{"page_content": f"[Doc search error: {e}]", "metadata": {}}]

        kb_block   = _format_kb_block(kb_json)
        doc_block  = _format_doc_block(docs_json)

        # 3) Summarization task—feed the results directly
        desc  = (
            f"User question: '{question}'\n\n"
            f"Search results from support.vanmetrehomes.com:\n{kb_block}\n\n"
        )
        if self.doc_tool:
            desc += f"Uploaded document snippets:\n{doc_block}\n\n"

        desc += (
            "Write a clear, numbered, step-by-step guide that answers the user's question. "
            "When a step is derived from a specific source, append '(Source: URL or file)' at the end of that step. "
            "Keep the steps concise and homeowner-friendly."
        )

        task_summarize = Task(
            description = desc,
            expected_output = f"A numbered list of steps answering: '{question}', with source URLs where applicable.",
            agent = self.agent,
            verbose = True,
        )

        crew = Crew(agents = [self.agent], tasks = [task_summarize], verbose = True)
        return crew.kickoff()

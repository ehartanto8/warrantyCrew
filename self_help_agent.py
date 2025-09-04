import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
import json
import requests
from langchain_core.tools import Tool as LCTool

# env
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

try:
    from crewai_tools import SerperDevTool            # prefer real package
except Exception:
    # Minimal inline fallback if crewai_tools drags embedchain
    class SerperDevTool:
        def __init__(self):
            key = os.getenv("SERPER_API_KEY")
            if not key:
                raise RuntimeError("SERPER_API_KEY is required")
            self._key = key

        def run(self, *, search_query: str, **kwargs):
            r = requests.post(
                "https://google.serper.dev/search",
                headers={"X-API-KEY": self._key, "Content-Type": "application/json"},
                json={"q": search_query},
                timeout=20,
            )
            r.raise_for_status()
            return r.json()

# VM Support Site
class KBSearchTool(SerperDevTool):
    def run(self, *, search_query: str, **kwargs):
        kb_query = f"site:support.vanmetrehomes.com {search_query}"
        kb_results = super().run(query=kb_query, **kwargs)

        organic = (kb_results or {}).get("organic", [])

        if organic:
            kb_results["organic"] = sorted(organic, key=lambda r: r.get("position", float('inf')))
            return kb_results
        else:
            return super().run(query=search_query, **kwargs)

# KB file uploads
try:
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma

    class DocumentSearchTool:
        def __init__(self, docs_path="./data"):
            self.vectorstore = None
            if not os.path.isdir(docs_path):
                return
            loader = DirectoryLoader(docs_path, glob="**/*.pdf")
            docs = loader.load()
            if not docs:
                return
            self.vectorstore = Chroma.from_documents(docs, embedding=OpenAIEmbeddings())

        def run(self, query: str, k: int = 4):
            if not getattr(self, "vectorstore", None):
                return []
            results = self.vectorstore.similarity_search(query, k=k)
            return [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in results]
except ImportError:
    DocumentSearchTool = None

# Agent
class HomeownerHelpAgent:
    def __init__(self):
        self.kb_tool = KBSearchTool()
        self.tools = [
            LCTool(
                name="KBSearchTool",
                description="Search Van Metre support site via Serper; input is a query string.",
                func=lambda q: json.dumps(self.kb_tool.run(search_query=q)),
            )
        ]

        if DocumentSearchTool:
            try:
                self.doc_tool = DocumentSearchTool()
                if self.doc_tool.vectorstore:
                    self.tools.append(
                        LCTool(
                            name = "DocumentSearchTool",
                            description = "Search uploaded PDFs; input is a query string.",
                            func = lambda q: json.dumps(self.doc_tool.run(q)),
                        )
                    )
            except Exception:
                pass

    def run(self, question: str) -> str:
        task_search_web = Task(
            description = f'Use KBSearchTool to search for: "{question}"',
            expected_output = "A JSON array of web KB snippets with title, link, and snippet.",
            agent = self.agent,
            verbose = False
        )
        tasks = [task_search_web]

        if DocumentSearchTool and getattr(self, 'doc_tool', None) and getattr(self.doc_tool, "vectorstore", None):
            task_search_doc = Task(
                description = f'Use DocumentSearchTool to search uploaded documents for: "{question}"',
                expected_output = "A JSON array of document snippets with page_content and metadata.",
                agent = self.agent,
                verbose = False
            )
            tasks.append(task_search_doc)

        desc = "Here are the results from support.vanmetrehomes.com:\n{task_search_web}"
        if DocumentSearchTool and getattr(self, 'doc_tool', None) and getattr(self.doc_tool, "vectorstore", None):
            desc += "\n{task_search_doc}"
        desc += (
            f"\n\nWrite a clear, numbered, step-by-step guide for '{question}', "
            "including the source URL after each step in parentheses."
        )

        task_summarize = Task(
            description = desc,
            expected_output = f"A numbered list of steps answering: '{question}', with URL citations.",
            agent = self.agent,
            verbose = True,
            context = tasks
        )
        tasks.append(task_summarize)

        crew = Crew(agents = [self.agent], tasks = tasks, verbose = True)
        return crew.kickoff()

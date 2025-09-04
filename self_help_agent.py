import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew

# env
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

try:
    from crewai_tools import SerperDevTool               # preferred (if it doesn't drag embedchain in)
except Exception:
    try:
        from crewai_tools.tools import SerperDevTool      # older path
    except Exception:
        from shims.serperdevtool import SerperDevTool     # fallback: our lightweight local shim

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
        self.tools = [self.kb_tool]

        if DocumentSearchTool:
            try:
                self.doc_tool = DocumentSearchTool()
                if getattr(self.doc_tool, "vectorstore", None):
                    self.tools.append(self.doc_tool)
            except Exception:
                pass

        self.agent = Agent(
            role="Homeowner Knowledge Assistant",
            goal="Answer homeowner warranty/maintenance questions using Van Metre's support site and uploaded documents",
            backstory=(
                "Use KBSearchTool to fetch relevant support articles first. "
                "If a question requires document context, use DocumentSearchTool. "
                "Finally, synthesize all results into a concise, step-by-step answer, citing URLs."
            ),
            tools=self.tools,
            verbose=True
        )

    def run(self, question: str) -> str:
        task_search_web = Task(
            description=f'Use KBSearchTool to search for: "{question}"',
            expected_output="A JSON array of web KB snippets with title, link, and snippet.",
            agent=self.agent,
            verbose=False
        )
        tasks = [task_search_web]

        if DocumentSearchTool and getattr(self, 'doc_tool', None) and getattr(self.doc_tool, "vectorstore", None):
            task_search_doc = Task(
                description=f'Use DocumentSearchTool to search uploaded documents for: "{question}"',
                expected_output="A JSON array of document snippets with page_content and metadata.",
                agent=self.agent,
                verbose=False
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
            description=desc,
            expected_output=f"A numbered list of steps answering: '{question}', with URL citations.",
            agent=self.agent,
            verbose=True
        )
        tasks.append(task_summarize)

        crew = Crew(agents=[self.agent], tasks=tasks, verbose=True)
        return crew.kickoff()

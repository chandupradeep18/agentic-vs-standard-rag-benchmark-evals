import uuid
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

NO_DOCS_SENTINEL = "NO_RELEVANT_DOCUMENTS"

REFUSAL_MESSAGE = (
    "I can only answer questions based on the indexed documents. "
    "I could not find relevant information for that in the knowledge base."
)

SYSTEM_PROMPT = (
    "You are a strict document-only RAG assistant.\n"
    "Rules:\n"
    "1. You MUST call the `rag_retriever` tool for every user question.\n"
    "2. Call `rag_retriever` at most ONCE (or at most TWICE if rephrasing). Never loop.\n"
    "3. Answer ONLY using facts explicitly present in the retrieved documents.\n"
    "4. NEVER use outside knowledge, general advice, or external links.\n"
    f"5. If the tool returns '{NO_DOCS_SENTINEL}' or the retrieved text does not "
    f"answer the question, immediately reply with ONLY this exact sentence: {REFUSAL_MESSAGE}"
)


class Rag_bot:
    def __init__(self, client, store):
        self.client = client
        self.store = store
        self.tools = self._build_tools()
        self.agent = create_agent(
            model=self.client,
            tools=self.tools,
            system_prompt=SYSTEM_PROMPT,
            checkpointer=InMemorySaver(),
        )

    def _build_tools(self):
        @tool
        def rag_retriever(query: str, top_k: int = 5) -> str:
            """
            Search the local knowledge base for relevant information.
            Use this tool when answering questions about documents
            stored in the vector database.
            Args:
                query: The question or search phrase.
                top_k: Number of relevant document chunks to retrieve.
            Returns:
                Relevant document contents with their source metadata.
            """
            results = self.store.query(query, top_k)
            if not results:
                return NO_DOCS_SENTINEL
            context = []
            for index, result in enumerate(results, start=1):
                content = result.get("content", "")
                metadata = result.get("metadata", {})
                source = metadata.get("source", "Unknown source")
                context.append(
                    f"Document {index}\n"
                    f"Source: {source}\n"
                    f"Content: {content}"
                )
            return "\n\n".join(context)

        return [rag_retriever]

    def ask(self, question: str) -> dict:
        """Stateless single-turn call for automated evaluation."""
        config = {
            "configurable": {"thread_id": f"eval-{uuid.uuid4().hex}"},
            "recursion_limit": 10,
        }
        try:
            response = self.agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config=config,
            )
        except GraphRecursionError:
            return {"answer": REFUSAL_MESSAGE, "contexts": []}

        messages = response["messages"]
        tool_outputs = [
            m.content for m in messages if isinstance(m, ToolMessage)
        ]
        valid_contexts = [
            out for out in tool_outputs if out != NO_DOCS_SENTINEL
        ]

        if not valid_contexts:
            return {"answer": REFUSAL_MESSAGE, "contexts": []}

        return {
            "answer": messages[-1].content,
            "contexts": valid_contexts,
        }

    def chat(self):
        print("\n RAG Chatbot Started!")
        print("Type 'exit' or 'quit' to stop.\n")
        config = {"configurable": {"thread_id": "rag-chat-session"}, "recursion_limit": 10}
        while True:
            query = input("You: ").strip()
            if not query or query in ["exit", "quit", "q"]:
                print("Chatbot: Goodbye!")
                break
            try:
                response = self.agent.invoke(
                    {"messages": [{"role": "user", "content": query}]},
                    config=config,
                )
                print(f"\nAssistant: {response['messages'][-1].content}\n")
            except GraphRecursionError:
                print(f"\nAssistant: {REFUSAL_MESSAGE}\n")
            except Exception as e:
                print(f"Error: {e}")
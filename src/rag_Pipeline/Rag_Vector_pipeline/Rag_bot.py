import os
from openai import OpenAI
from .VectorStore import QdrantVectorStore

REFUSAL_MESSAGE = (
    "I don't have enough context or information in the documents to answer this question."
)

SYSTEM_PROMPT = (
    "You are a strict document-only RAG assistant. "
    "Answer questions based ONLY on the provided context. "
    f"If the context does not contain the answer, reply ONLY with: {REFUSAL_MESSAGE} "
    "Do not make up information or use outside knowledge."
)

class StandardRAGPipeline:
    def __init__(
        self,
        store: QdrantVectorStore,
        model: str = "nvidia/nemotron-3.5-lightning-30b-a3b",
        api_key: str | None = None,
    ):
        self.store = store
        self.model = model
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key or os.getenv("NVIDIA_API_KEY"),
        )

    def ask(self, question: str, top_k: int = 5, min_score: float = 0.35) -> dict:
        retrieved = self.store.query(question, top_k=top_k, min_score=min_score)
        if not retrieved:
            return {"answer": REFUSAL_MESSAGE, "contexts": []}

        context_blocks = [
            f"\n{doc['content']}" for doc in retrieved
        ]
        context_text = "\n\n---\n\n".join(context_blocks)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}"},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.2,
            max_tokens=2000,
        )
        return {
            "answer": response.choices[0].message.content,
            "contexts": [doc["content"] for doc in retrieved],
        }

    def chat(self):
        print("\nStandard RAG (Qdrant + NVIDIA) Started!")
        print("Type 'exit' or 'quit' to stop.\n")
        while True:
            query = input("You: ").strip()
            if not query or query in ["exit", "quit", "q"]:
                print("Chatbot: Goodbye!")
                break
            try:
                result = self.ask(query)
                print(f"\nAssistant: {result['answer']}\n")
            except Exception as e:
                print(f"Error: {e}")

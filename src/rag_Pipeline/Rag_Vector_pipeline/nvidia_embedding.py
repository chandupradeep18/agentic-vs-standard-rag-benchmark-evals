import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import numpy as np
from .data_loader import load_all_documents,chunk_documents


class NvidiaEmbedder:
    def __init__(self, model: str = "nvidia/nemotron-3.5-embed-8b",):
        self.model = model
        self.client = OpenAI(base_url="https://integrate.api.nvidia.com/v1",api_key=os.getenv("NVIDIA_API_KEY"))

    def embed_passages(self, chunks: list[dict], batch_size: int = 32)->np.ndarray:
        if not chunks:
            return np.empty((0, 0), dtype=np.float32)
        all_embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = [c["text"] for c in chunks[i:i + batch_size]]
            response = self.client.embeddings.create(
            model=self.model,
            input=batch,
            encoding_format="float",
            extra_body={"input_type": "passage", "truncate": "END"},
            )
            data = sorted(response.data, key=lambda d: d.index)
            all_embeddings.extend(d.embedding for d in data)
            print(f"Embedding Completed : {len(all_embeddings)} / {len(chunks)}")
        return np.array(all_embeddings, dtype=np.float32)

    def embed_query(self, query: str)->np.ndarray:
        if not query:
            return np.array([])
        response = self.client.embeddings.create(
            model=self.model,
            input=query,
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "END"},
        )
        return np.array([item.embedding for item in response.data], dtype=np.float32)
    
if __name__=="__main__":
    nvidia=NvidiaEmbedder()
    load_docus=load_all_documents('D:/Github/AI-Engineering/Agentic AI/LangChainc/src/rag_pipeline/data')
    print(len(load_docus))
    if load_docus:
        chunks=chunk_documents(load_docus)
        embeddings=nvidia.embed_passages([chunks[-1],chunks[-2]])
        print(nvidia.embed_query('Hi This Pradeep'))
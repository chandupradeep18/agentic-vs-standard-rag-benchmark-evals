import os
import uuid
from pathlib import Path
from typing import List,Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from .nvidia_embedding import NvidiaEmbedder
from .data_loader import chunk_documents

DEFAULT_PERSIST_DIR = Path(__file__).resolve().parent / "qdrant_db"
class QdrantVectorStore:
    def __init__(self,chunk_size: int = 1000, chunk_overlap: int = 200,
                 collection_name: str = "company_docs"):
        self.persist_dir = DEFAULT_PERSIST_DIR
        self.collection_name=collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.client = None
        self.embedder = NvidiaEmbedder()
        self._init_store()

    def _init_store(self):
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            self.client = QdrantClient(path=str(self.persist_dir))
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise
    def count(self) -> int:
        if not self.client.collection_exists(self.collection_name):
            return 0
        return self.client.count(collection_name=self.collection_name, exact=True).count

    def _ensure_collection(self, vector_size: int):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            print(
                f"[INFO] Created Qdrant collection '{self.collection_name}' "
                f"(dim={vector_size}, distance=COSINE)"
            )
    def build_from_documents(self, documents: list[dict]):
        if not documents:
            print("[WARNING] No documents provided to build vector store.")
            return
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        chunks=chunk_documents(documents,self.chunk_size,self.chunk_overlap)
        if not chunks:
            print("[WARNING] No chunks generated from documents.")
            return
        embeddings=self.embedder.embed_passages(chunks)
        if embeddings.size==0:
            print("[WARNING] No Embedding Are Generated For chunked documents.")
            return
        self._ensure_collection(embeddings.shape[1])
        doc_emd=self.add_embeddings(chunks,embeddings)
        self.save(doc_emd)

    def add_embeddings(self,documents:List[Any],embeddings:np.ndarray)->dict:
        print(f"Adding {len(documents)} documents to vector store...")
        ids ,metadatas,documents_text ,embeddings_list = [],[],[],[]
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = str(uuid.uuid4())
            ids.append(doc_id)
            metadatas.append({
            "source": doc["source"],
            "doc_index": i,
            "content_length": len(doc["text"]),})
            documents_text.append(doc['text'])
            embeddings_list.append(embedding.tolist())
        return {"Id":ids,'metadatas':metadatas,'documents_text':documents_text,
                        'embeddings_list':embeddings_list}

    def save(self,doc_emd:dict, batch_size: int = 5000):
        if not doc_emd['Id'] or not doc_emd['metadatas'] or not doc_emd['documents_text'] or not doc_emd['embeddings_list']:
            return 
        try:
            total = len(doc_emd["Id"])
            for start in range(0, total, batch_size):
                end = start + batch_size
                points = [
                    PointStruct(id=doc_id,vector=vec,payload={"content": text, **meta},)
                    for doc_id, vec, text, meta in zip(
                        doc_emd["Id"][start:end],
                        doc_emd["embeddings_list"][start:end],
                        doc_emd["documents_text"][start:end],
                        doc_emd["metadatas"][start:end],
                    )
                ]
                self.client.upsert(collection_name=self.collection_name, points=points)
            print(f"Successfully added {len(doc_emd['documents_text'])} documents to vector store")
            print(f"Total documents in collection: {self.count()}")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

    def query(self, query: str, top_k: int = 5,min_score: float = 0.0) -> list[dict]:
        total_docs = self.count()
        if total_docs == 0:
            print("Collection is empty. No documents found.")
            return []
        query_embedding = self.embedder.embed_query(query)
        if query_embedding.size == 0:
            return []
        try:
            hits = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding[0].tolist(),
                limit=min(top_k, total_docs),
                score_threshold=min_score,
                with_payload=True,
            ).points
            retrieved_docs = []
            for i, hit in enumerate(hits):
                payload = dict(hit.payload or {})
                content = payload.pop("content", "")
                similarity_score = float(hit.score)
                retrieved_docs.append(
                    {
                        "id": str(hit.id),
                        "content": content,
                        "metadata": payload,
                        "similarity_score": similarity_score,
                        "distance": 1.0 - similarity_score,
                        "rank": i + 1,
                    }
                )

            if retrieved_docs:
                print(f"Retrieved {len(retrieved_docs)} documents")
            else:
                print("No documents found")
            return retrieved_docs
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
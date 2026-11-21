import os,chromadb
import numpy as np
from typing import List, Any
import uuid
from .embedding import EmbeddingPipeline
from pathlib import Path
DEFAULT_PERSIST_DIR = Path(__file__).resolve().parent / "vector"

class VectorStore:
    def __init__(self, 
                embedding_model: str = "all-MiniLM-L6-v2", 
                chunk_size: int = 1000, 
                chunk_overlap: int = 200,
                collection_name="file_documents"):
        self.persist_dir = DEFAULT_PERSIST_DIR
        self.collection_name=collection_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.client=None
        self.collection=None
        self.emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self._init_store()
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def _init_store(self):
        try:
            os.makedirs(self.persist_dir,exist_ok=True)
            self.client=chromadb.PersistentClient(path=str(self.persist_dir))
            self.collection=self.client.get_or_create_collection(name=self.collection_name,metadata={'description':"PDF Document"})
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def build_from_documents(self, documents: List[Any]):
        if not documents:
            print("[WARNING] No documents provided to build vector store.")
            return
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        chunks=self.emb_pipe.chuck_documents(documents)
        if not chunks:
            print("[WARNING] No chunks generated from documents.")
            return
        embeddings=self.emb_pipe.convert_chunks_to_embedding(chunks)
        if embeddings.size==0:
            print("[WARNING] No Embedding Are Generated For chunked documents.")
            return
        doc_emd=self.add_embeddings(chunks,embeddings)
        self.save(doc_emd)

    def add_embeddings(self,documents:List[Any],embeddings:np.ndarray)->dict:
        print(f"Adding {len(documents)} documents to vector store...")
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            metadata = dict(doc.metadata)
            metadata['doc_index']=i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            documents_text.append(doc.page_content)
            embeddings_list.append(embedding.tolist())
        return {"Id":ids,'metadatas':metadatas,'documents_text':documents_text,
                'embeddings_list':embeddings_list}

    def save(self,doc_emd:dict, batch_size: int = 5000):
        if not doc_emd['Id'] or not doc_emd['metadatas'] or not doc_emd['documents_text'] or not doc_emd['embeddings_list']:
            return 
        try:
            max_batch = getattr(self.client, "get_max_batch_size", lambda: 5000)()
            step = min(batch_size, max_batch)
            total = len(doc_emd["Id"])
            for start in range(0, total, step):
                end = start + step
                self.collection.add(
                        ids=doc_emd['Id'][start:end],
                        embeddings=doc_emd['embeddings_list'][start:end],
                        metadatas=doc_emd['metadatas'][start:end],
                        documents=doc_emd['documents_text'][start:end]
                    )
            print(f"Successfully added {len(doc_emd['documents_text'])} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise
    def query(self,query:str,top_k:int=5)-> list[dict]:
        count=self.collection.count()
        if count==0:
            print("Collection is empty. No documents found.")
            return []
        query_embedding=self.emb_pipe.convert_chunks_to_embedding_query([query])
        try:
            results=self.collection.query(query_embeddings=query_embedding.tolist(),n_results=min(top_k,count))
            retrieved_docs = []            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = 1 - distance
                    retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                    })
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")
            return retrieved_docs
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
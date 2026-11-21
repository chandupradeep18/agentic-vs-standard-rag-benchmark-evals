import os
from src.rag_Pipeline.Agentic_Rag.VectorStore import VectorStore
from src.rag_Pipeline.Agentic_Rag.data_loader import load_all_documents
from src.rag_Pipeline.Agentic_Rag.Rag_bot import Rag_bot
from langchain_groq import ChatGroq
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv
load_dotenv()
groq_client=ChatGroq(model='qwen/qwen3.8-27b',api_key=os.getenv("GROQ_API_KEY"))
nvidia_client=ChatNVIDIA(model="nvidia/nemotron-3.5-lightning-30b-a3b",api_key=os.getenv("NVIDIA_API_KEY"))
from langchain_core.messages import HumanMessage, SystemMessage,ToolMessage
from langchain.tools import tool

from src.rag_Pipeline.Rag_Vector_pipeline.VectorStore import QdrantVectorStore
from src.rag_Pipeline.Rag_Vector_pipeline.data_loader import load_all_documents as rag_load_all_Document
from src.rag_Pipeline.Rag_Vector_pipeline.Rag_bot import StandardRAGPipeline

class Agentic_Rag:
    def __init__(self,client,dir_path,store_name):
        self.store=None
        self.client=client
        self.dir_path=dir_path
        self.store_name=store_name
        self._load_data()
        self.rag_bot=Rag_bot(self.client,self.store)
        
    def _load_data(self):
        self.store = VectorStore(collection_name=self.store_name)
        if self.store.collection.count() > 0:
            print(f"[INFO] Using existing collection '{self.store_name}' "
            f"({self.store.collection.count()} chunks). Skipping indexing.")
            return
        docs = load_all_documents(self.dir_path)
        if not docs:
            print(f"[WARNING] No documents to index from '{self.dir_path}'.")
            return
        self.store.build_from_documents(docs)

    def run(self):
        self.rag_bot.chat()


class Rag_chat_bot:
    def __init__(self,dir_path,store_name):
        self.store=None
        self.dir_path=dir_path
        self.store_name=store_name
        self._load_data()
        self.rag_bot=StandardRAGPipeline(self.store)
    def _load_data(self):
        self.store = QdrantVectorStore(collection_name=self.store_name)
        if self.store.count()>0:
            print(f"[INFO] Using existing collection '{self.store_name}' "
            f"({self.store.count()} chunks). Skipping indexing.")
            return
        docs = rag_load_all_Document(self.dir_path)
        if not docs:
            print(f"[WARNING] No documents to index from '{self.dir_path}'.")
            return
        self.store.build_from_documents(docs)
    def run(self):
        self.rag_bot.chat()


    
if __name__=="__main__":
    #app=Agentic_Rag(groq_client,"D:/Github/AI-Engineering/Agentic AI/LangChainc/src/rag_pipeline/data","File_Document")
    #app.run()
    app=Rag_chat_bot("D:/Github/AI-Engineering/Agentic AI/LangChainc/src/rag_pipeline/data","File_Document")
    try:
        app.run()
    finally:
        app.store.client.close()
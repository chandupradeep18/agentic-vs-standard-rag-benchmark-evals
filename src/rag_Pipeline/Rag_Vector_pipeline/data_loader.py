import csv
import json
from pathlib import Path
from pypdf import PdfReader
from docx import Document as DocxDocument

def load_all_documents(files_dir: str) -> list[dict]:
    documents=[]
    files_dir_path = Path(files_dir)
    if not files_dir_path.exists() or not files_dir_path.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {files_dir_path}")
    all_document=files_dir_path.rglob("*")
    for file_path in all_document:
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        source = str(file_path)
        try:
            if ext == ".pdf":
                reader = PdfReader(str(file_path))
                for page_num, page in enumerate(reader.pages, start=1):
                    text = (page.extract_text() or "").strip()
                    if text:
                        documents.append({"text": text, "source": f"{source} (p.{page_num})"})

            elif ext in (".txt", ".md"):
                text = file_path.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    documents.append({"text": text, "source": source})

            elif ext == ".docx":
                doc = DocxDocument(str(file_path))
                text = "\n\n".join(p.text.strip() for p in doc.paragraphs if p.text.strip())
                if text:
                    documents.append({"text": text, "source": source})

            elif ext == ".csv":
                with file_path.open("r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row_text = "\n".join(f"{k}: {v}" for k, v in row.items() if v)
                        if row_text.strip():
                            #pass
                            documents.append({"text": row_text, "source": source})

            elif ext == ".json":
                data = json.loads(file_path.read_text(encoding="utf-8"))
                text = json.dumps(data, indent=2, ensure_ascii=False)
                if text.strip():
                    #pass
                    documents.append({"text": text, "source": source})
        except Exception as e:
            print(f"[ERROR] Failed to load {file_path}: {e}")
    print(f"[INFO] Loaded {len(documents)} raw documents from {files_dir_path}")
    return documents
def chunk_documents(documents: list[dict],chunk_size: int = 1000,chunk_overlap: int = 200,) -> list[dict]:
    """Pure-Python paragraph + sliding-window chunker."""
    chunks = []
    step = max(1, chunk_size - chunk_overlap)

    for doc in documents:
        text = doc["text"].strip()
        source = doc["source"]
        if not text:
            continue

        # Fast path for short docs (e.g., CSV rows)
        if len(text) <= chunk_size:
            chunks.append({"text": text, "source": source})
            continue

        # Split by paragraphs, then group or slice to chunk_size
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        current = ""
        for para in paragraphs:
            if len(para) > chunk_size:
                if current:
                    chunks.append({"text": current, "source": source})
                    current = ""
                for start in range(0, len(para), step):
                    piece = para[start : start + chunk_size].strip()
                    if piece:
                        chunks.append({"text": piece, "source": source})
            elif len(current) + len(para) + 2 <= chunk_size:
                current = f"{current}\n\n{para}".strip()
            else:
                if current:
                    chunks.append({"text": current, "source": source})
                current = para

        if current:
            chunks.append({"text": current, "source": source})

    print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

if __name__=="__main__":
    load_docus=load_all_documents('D:/Github/AI-Engineering/Agentic AI/LangChainc/src/rag_pipeline/data')
    print(len(load_docus))
    if load_docus:
        chunks=chunk_documents(load_docus)
        print(len(chunks))
from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader,Docx2txtLoader,JSONLoader

def load_all_documents(files_dir:str)->list[Any]:
    documents=[]
    files_dir_path = Path(files_dir)
    if not files_dir_path.exists() or not files_dir_path.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {files_dir_path}")
    #pdf FIle
    pdf_files=list(files_dir_path.glob('**/*.pdf'))
    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: {pdf_file}")
        try:
            loader=PyPDFLoader(str(pdf_file))
            loaded=loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

    #Txt FIle
    text_files=list(files_dir_path.glob('**/*.txt'))
    for text_file in text_files:
        print(f"[DEBUG] Loading Text: {text_file}")
        try:
            loader=TextLoader(str(text_file),encoding="utf-8", autodetect_encoding=True)
            loaded=loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Text docs from {text_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Text {text_file}: {e}")

    # Markdown Files (.md)
    for md_file in files_dir_path.glob("**/*.md"):
        print(f"[DEBUG] Loading Markdown: {md_file}")
        try:
            loaded = TextLoader(str(md_file), encoding="utf-8").load()
            print(f"[DEBUG] Loaded {len(loaded)} Markdown docs from {md_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Markdown {md_file}: {e}")

    # Word Files (.docx)
    for docx_file in files_dir_path.glob("**/*.docx"):
        print(f"[DEBUG] Loading Word: {docx_file}")
        try:
            loaded = Docx2txtLoader(str(docx_file)).load()
            print(f"[DEBUG] Loaded {len(loaded)} Word docs from {docx_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Word {docx_file}: {e}")

   #csv FIle
    csv_files=list(files_dir_path.glob('**/*.csv'))
    for csv_file in csv_files:
        print(f"[DEBUG] Loading CSV: {csv_file}")
        try:
            loader=CSVLoader(str(csv_file), encoding="utf-8-sig",csv_args={"delimiter": ",","quotechar": '"',"doublequote": True})
            loaded=loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} CSV docs from {csv_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load CSV {csv_file}: {e}")

    #Json FIle
    json_files=list(files_dir_path.glob('**/*.json'))
    for json_file in json_files:
        print(f"[DEBUG] Loading JSON: {json_file}")
        try:
            loader=JSONLoader(file_path=str(json_file),jq_schema=".",text_content=False)
            loaded=loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Json from {json_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load JSON {json_file}: {e}")

    if not documents:
        print(f"[WARNING] No supported files found in: {files_dir_path}")
    return documents
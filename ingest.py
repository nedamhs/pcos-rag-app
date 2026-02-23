import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# Config
CORPUS_DIR = "PCOS_corpus"
CHROMA_DIR = "chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def ingest_pdfs():
    """Load PDFs, chunk them, create embeddings, store in Chroma db"""

    # Load all PDFs
    documents = []
    for filename in os.listdir(CORPUS_DIR):
        if filename.endswith('.pdf'):
            filepath = os.path.join(CORPUS_DIR, filename)
            loader = PyPDFLoader(filepath) #load pdf
            docs = loader.load()           #
            documents.extend(docs)         

    print(f"Loaded {len(documents)} pages from PDFs")
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    
    # Create embeddings and store in Chroma
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents=chunks,
                                        embedding=embeddings,
                                        persist_directory=CHROMA_DIR)
                                        
    print(f"Ingestion complete! Vector DB saved to {CHROMA_DIR}")

if __name__ == "__main__":
    ingest_pdfs()
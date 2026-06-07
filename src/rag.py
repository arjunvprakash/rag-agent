from langchain.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader


# Flag that controls reindexing
REINDEX=False

## Embedding
PDF_PATH = "data/data.pdf"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "rag_agent"
EMBED_MODEL="mxbai-embed-large"
BASE_URL="http://127.0.0.1:11434"


vectorstore = None

def init_vectorstore():
    global vectorstore
    if vectorstore == None:
        embeddings = OllamaEmbeddings(
            model=EMBED_MODEL,
            base_url=BASE_URL
        )
        vectorstore = Chroma(
            embedding_function=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_DIR,
        )
    return vectorstore

def load_retriever(reindex:bool):
    global vectorstore
    if vectorstore == None:
        vectorstore = init_vectorstore()

    if reindex:
        print("Reindexing vector store....")
        loader = PyPDFLoader(PDF_PATH, extract_images=False)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        doc_splits = splitter.split_documents(docs)

        vectorstore.add_documents(doc_splits)
        print("PDF indexed and persisted to Chroma.")
    else:
        print("Loaded existing Chroma index from disk.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    # retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 4, "score_threshold": 0.8})
    return retriever


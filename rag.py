import bs4
import requests
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os
from langchain_community.document_loaders import PyPDFLoader


# Flag that controls reindexing
REINDEX=False

PDF_PATH = "data/data.pdf"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "rag_agent"

load_dotenv()

# # Below is a minimal helper for demonstration purposes.
# def load_web_page(url: str, bs_kwargs: dict | None = None) -> list[Document]:
#     response = requests.get(url)
#     response.raise_for_status()
#     soup = bs4.BeautifulSoup(response.text, "html.parser", **(bs_kwargs or {}))
#     return [Document(page_content=soup.get_text(), metadata={"source": url})]


# urls = [
#     "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
#     "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
#     "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
# ]

# docs = [load_web_page(url) for url in urls]

# docs_list = [item for sublist in docs for item in sublist]

# text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=100, chunk_overlap=50
# )
# doc_splits = text_splitter.split_documents(docs_list)


embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://127.0.0.1:11434"
)

vectorstore = Chroma(
    embedding_function=embeddings,
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR,
)

reindex_prompt = input("Reindex? yes/no: ")
REINDEX=(reindex_prompt.strip().lower() == "yes")
if REINDEX:
    loader = PyPDFLoader(PDF_PATH)
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

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
llm = ChatOllama(model="llama3.2:1b", base_url="http://127.0.0.1:11434")

# @tool
# def retrieve_blog_posts(query: str) -> str:
#     """Search and return information about Lilian Weng blog posts."""
#     docs = retriever.invoke(query)
#     return "\n\n".join([doc.page_content for doc in docs])

# retriever_tool = retrieve_blog_posts

# retriever_tool.invoke({"query": "types of reward hacking"})

while True:
    question = input("Question: ").strip()
    if question.lower() in {"exit", "quit"}:
        break

    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
Answer the question using only the context below.
If the answer is not in the context, say you could not find it in the PDF.

Context:
{context}

Question:
{question}
"""
    response = llm.invoke(prompt)
    print("\nAnswer:\n", response.content)
    print("\n" + "-" * 60)
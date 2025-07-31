import json
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm

# Load your parsed JSON data
def load_onepiece_json(json_path: str) -> list[Document]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for entry in tqdm(data):
        content = entry.get("content", "").strip()
        title = entry.get("title", "unknown")
        if content:
            metadata = {
                "source": title
            }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

def chunking(json_path: str):
    docs = load_onepiece_json(json_path)
    #Splitting the documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    docs = splitter.split_documents(docs)
    return docs

def embedding(json_path: str):
    docs = chunking(json_path)
    # Initialize the embedder
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # Save to Chroma vectorstore locally
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_function,
        persist_directory="./chroma_db"
    )
    return vectorstore

if __name__ == "__main__":
    embedding_function = embedding("onepiece_sample.json")

    #Check if it works:
    vector_store = Chroma(
        persist_directory = "./chroma_db",
        embedding_function=embedding_function
    )
    results = vector_store.similarity_search(
        "King of the pirates and the world's strongest swordsman",
        k=5
    )
    for res in results:
        print(f"* {res.page_content[:100]}... [{res.metadata}] -> ID: {id(res)}")

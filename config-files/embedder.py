import json
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_onepiece_json(json_path: str) -> list[Document]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for entry in data:
        content = entry.get("content", "").strip()
        title = entry.get("title", "unknown")
        if content:
            docs.append(Document(page_content=content, metadata={"source": title}))

    return docs

documents = load_onepiece_json("onepiece_clean.json")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = splitter.split_documents(documents)

## Vector Embedding And Vector Store
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma.from_documents(documents,embeddings, collection_name="one_piece_wiki", persist_directory="..\chroma_db")
db.persist()

query = "What episode did oden die?"
retireved_results=db.similarity_search(query)
print(retireved_results)

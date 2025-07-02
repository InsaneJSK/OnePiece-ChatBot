import json
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm

# Load your parsed JSON data
def load_onepiece_json(json_path: str) -> list[Document]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for entry in tqdm(data):
        content = entry.get("content", "").strip()
        title = entry.get("title", "unknown")
        headings = entry.get("headings", [])
        categories = entry.get("categories", [])
        if content:
            metadata = {
                "source": title,
                "headings": ", ".join(headings) if isinstance(headings, list) else str(headings),
                "categories": ", ".join(categories) if isinstance(categories, list) else str(categories),
            }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

documents = load_onepiece_json("onepiece_structured.json")

#Splitting the documents
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = splitter.split_documents(documents)
print(f"Number of documents: {len(documents)}")

# Initialize the embedder
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedder initilized successfully")

# Save to Chroma vectorstore locally
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_function,
    persist_directory=".\\chroma_db"
)

print(f"✅ Indexed {len(documents)} documents to Chroma.")


#Check if it works:
vector_store = Chroma(
    persist_directory = "../chroma_db",
    embedding_function=embedding_function
)
results = vector_store.similarity_search(
    "What do you know about the ancients",
    k=5
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}] -> ID: {id(res)}")
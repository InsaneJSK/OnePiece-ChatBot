import json
from tqdm import tqdm
import os
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

# ----------- Load & Split JSON -------------
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
total_chunks = len(documents)
print(f"📄 Total chunks after splitting: {total_chunks}")

# ----------- Embedding Model -------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------- Qdrant Setup -------------
qdrant_url = os.getenv("qdrant_url")
qdrant_api_key = os.getenv("qdrant_api_key")
collection_name = "one_piece_wiki"

client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)

# ----------- Resume Upload -------------
current_uploaded = client.count(collection_name=collection_name).count
print(f"🔄 Resuming from index: {current_uploaded}")

batch_size = 100
print(f"🚀 Uploading remaining {total_chunks - current_uploaded} chunks in batches of {batch_size}...")

# ✅ Proper initialization with `embeddings` param
db = Qdrant(
    client=client,
    collection_name=collection_name,
    embeddings=embeddings,
)

for i in tqdm(range(current_uploaded, total_chunks, batch_size), desc="Resuming upload to Qdrant"):
    batch = documents[i:i + batch_size]
    texts = [doc.page_content for doc in batch]
    metadatas = [doc.metadata for doc in batch]

    try:
        db.add_texts(texts, metadatas=metadatas)
    except Exception as e:
        print(f"⚠️ Upload failed at chunk index {i}: {e}")
        break

# ----------- Final Count Check -------------
new_total = client.count(collection_name=collection_name).count
print(f"✅ Upload resumed and completed! Total points now in collection: {new_total}")

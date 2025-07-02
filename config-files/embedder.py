import json
from tqdm import tqdm
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import os
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
        headings = entry.get("headings", [])
        categories = entry.get("")
        if content:
            docs.append(Document(page_content=content, metadata={"source": title, "headings": headings, "categories": categories}))
    return docs

documents = load_onepiece_json("onepiece_clean.json")

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = splitter.split_documents(documents)
print(f"Total chunks after splitting: {len(documents)}")

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

# ❗ Wipe and recreate collection
client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

# ----------- Batching Upload -------------
batch_size = 100
print(f"🚀 Uploading in batches of {batch_size}...")

for i in tqdm(range(0, len(documents), batch_size), desc="Uploading to Qdrant"):
    batch = documents[i:i + batch_size]
    Qdrant.from_documents(
        documents=batch,
        embedding=embeddings,
        url=qdrant_url,
        api_key=qdrant_api_key,
        collection_name=collection_name,
    )

# ----------- Final Count Check -------------
collection_info = client.get_collection(collection_name)
total_uploaded = collection_info.points_count
print(f"✅ Upload complete! Total points in collection: {total_uploaded}")

from langchain_qdrant import QdrantVectorStore as Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

load_dotenv()

# ----------- Embedding Model -------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------- Qdrant Client Setup -------------
qdrant_url = os.getenv("qdrant_url")
qdrant_api_key = os.getenv("qdrant_api_key")
collection_name = "one_piece_wiki"

client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
)

# ----------- Load VectorStore from Qdrant -------------
db = Qdrant(
    client=client,
    collection_name=collection_name,
    embedding=embedding,
)

# ----------- Perform Search -------------
query = "What episode did Oden die?"
retrieved_results = db.similarity_search(query)

for i, doc in enumerate(retrieved_results, 1):
    print(doc.page_content)

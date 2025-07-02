import json
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm

with open("unique_categories.json", "r", encoding="utf-8") as f:
    cats = json.load(f)

print(len(cats))
print(len(list(set(cats))))

documents = [Document(cat) for cat in tqdm(cats)]
print(f"Number of documents: {len(documents)}")

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedder initialized successfully")

vectorstore = Chroma(
    embedding_function=embedding_function,
    persist_directory="..\\chroma_db_cats"
)

BATCH_SIZE = len(cats)

# Process in batches
for i in tqdm(range(0, len(cats), BATCH_SIZE), desc="Indexing"):
    batch = cats[i:i+BATCH_SIZE]
    documents = [Document(page_content=cat) for cat in batch]
    vectorstore.add_documents(documents)

print(f"✅ Indexed {len(cats)} documents into Chroma successfully.")

#Check if it works:
vector_store = Chroma(
    persist_directory = "../chroma_db_cats",
    embedding_function=embedding_function
)
results = vector_store.similarity_search(
    "What do you know about the ancients",
    k=5
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}] -> ID: {id(res)}")

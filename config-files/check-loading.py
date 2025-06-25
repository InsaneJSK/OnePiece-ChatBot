from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = Chroma(
    collection_name="one_piece_wiki",
    persist_directory="..\chroma_db",
    embedding_function=embedding
)

query = "What episode did oden die?"
retireved_results=db.similarity_search(query)
print(retireved_results)
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from embedder import embedding
from dotenv import load_dotenv
load_dotenv()

def load_vector_db():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    try:
        vector_store = Chroma(
            persist_directory = "./chroma_db",
            embedding_function=embedding
        )
        print("Found existing embeddings!")
        return vector_store
    except:
        path = "onepiece_sample.json"
        print(f"Embedding {path}")
        vector_store = embedding(path)

def initialize_llm():
    llm = ChatGroq(model_name="llama-3.3-70b-versatile")
    prompt=ChatPromptTemplate.from_template(
    """
    You are a helpful assistant who knows everything there is to know about the one piece anime and manga!
    Answer the questions based on the provided context only. Please provide the most accurate responses for the question.
    If sufficient information about a question is not provided, say you don't know the answer.
    Don't mention that you were provided a context.
    <context>
    {context}
    <context>
    Question: {input}

    """
    )
    db = load_vector_db()
    retriever = db.as_retriever(
        search_kwargs={"k": 5},
        search_type="mmr"
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    return retrieval_chain

if __name__ == "__main__":
    retrieval_chain = initialize_llm()
    user_input:str = ""
    while True:
        user_input = input("User: ")
        if user_input != 'q':
            try:
                response = retrieval_chain.invoke({
                    "input": user_input,
                })
                answer = response["answer"]
                print(f"Assistant: {answer}")
            except Exception as e:
                print(f"Error: {e}")
                continue
        else:
            break

        # # Uncomment if you wish to see the sources
        # for doc in response.get("context", []):
        #     print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        #     print(doc.page_content[:100])

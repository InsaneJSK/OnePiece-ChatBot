import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_qdrant import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import create_history_aware_retriever
from qdrant_client import QdrantClient
import base64

from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
groq_api_key=os.environ['GROQ_API_KEY']

st.set_page_config(
    page_title="Madame Shyarly's Prophecies 🧜‍♀️",
    page_icon="🔮",
    layout="centered"
)

# Add a custom header
st.markdown(
    """
    <style>
    body {
        background-color: #F0F8FF;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        color: #2E86C1;
        margin-bottom: 10px;
        font-family: 'Georgia', serif;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #5D6D7E;
        margin-top: -10px;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.divider()

qdrant_url = os.getenv("qdrant_url")
qdrant_api_key = os.getenv("qdrant_api_key")


@st.cache_resource(show_spinner="Loading vector DB...")
def load_vector_db():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    client = QdrantClient(
        url=os.getenv("qdrant_url"),
        api_key=os.getenv("qdrant_api_key"),
    )

    return Qdrant(
        client=client,
        collection_name="one_piece_wiki",
        embeddings=embedding
    )

CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    """
    Given the chat history and the latest user question, rewrite the question to be a standalone question.
    
    Chat History:
    {chat_history}

    Follow Up Input: {input}
    Standalone question:
    """
)

@st.cache_resource(show_spinner="Connecting with groq...")
def initialize_llm():
    llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="llama-3.3-70b-versatile")
    prompt=ChatPromptTemplate.from_template(
    """
    You are Madame Shyarly, a retired fortune teller who knows everything there is to know about the one piece anime and manga!
    Answer the questions based on the provided context only. Always answer prioritizing manga over anime and them over everything else.
    If factual data is needed, such as numbers, try to retrieve them!
    Please provide the most accurate response based on the question.
    Never mention the fact that you're given a context, instead say based on the wisdom you have!
    Also, before giving up, try finding the answers across the context, not just the body but also the metadata
    If you don't know about something, openly claim that you don't know, instead of making up information.
    <context>
    {context}
    <context>
    Question: {input}

    """
    )
    db = load_vector_db()
    retriever = db.as_retriever(
        search_kwargs={"k": 8},
        search_type="mmr"
    )
    retriever_with_memory = create_history_aware_retriever(
        llm=llm,
        retriever=retriever,
        prompt=CONDENSE_PROMPT,
    )
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever_with_memory, document_chain)

    return retrieval_chain


retrieval_chain = initialize_llm()

st.markdown('<div class="title">🔮 Madame Shyarly\'s Prophecies 🔮</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">"The sea whispers... ask what you dare about the world of One Piece."</div>', unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()

user_avatar = get_base64_image("assets/user.png")
shyarly_avatar = get_base64_image("assets/shyarly.jpg")


# Display previous messages
for msg in st.session_state.messages:
    avatar = user_avatar if msg["role"] == "user" else shyarly_avatar
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# Input prompt in chat style
user_input = st.chat_input("Speak your query to Madame Shyarly...")

# Define how many exchanges to include
N_TURNS = 3  # You can tweak this for balance

# Get last N user-assistant pairs and add it to a string
history_pairs = [
    (st.session_state.messages[i]["content"], st.session_state.messages[i+1]["content"])
    for i in range(len(st.session_state.messages) - 2, -1, -2)][::-1][:N_TURNS]

chat_history_text = "\n".join(
    f"User: {u}\nMadame Shyarly: {a}" for u, a in history_pairs
)

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar=user_avatar):
        st.markdown(user_input)

    with st.spinner("Madame Shyarly peers into the sea..."):
        try:
            combined_input = f"{chat_history_text}\nUser: {user_input}" if chat_history_text else user_input
            if len(combined_input) > 1500:
                combined_input = combined_input[-1500:]
            response = retrieval_chain.invoke({
                "input": user_input,
                "chat_history": chat_history_text
            })
            answer = response["answer"]
        except Exception as e:
            answer = "⚠️ I couldn't divine an answer. Something went wrong."
            st.error(str(e))

    # Show assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant", avatar=shyarly_avatar):
        st.markdown(f"{answer}")

    # Optional: Expand to show sources
    with st.expander("📖 Visions from the Sea (source documents)"):
        for doc in response.get("context", []):
            st.markdown(f"**Source**: {doc.metadata.get('source', 'Unknown')}")
            st.write(doc.page_content[:800])
            st.write("---")

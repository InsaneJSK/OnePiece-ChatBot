import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore as Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
import base64

#Updated imports
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


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
        check_compatibility=False
    )

    return Qdrant(
        client=client,
        collection_name="one_piece_wiki",
        embedding=embedding
    )

CONDENSE_PROMPT = ChatPromptTemplate.from_template(
    """
    Given the chat history and the latest user question, rewrite the question to be a standalone question.
    
    Chat History:
    {chat_history}

    Follow Up Input: {question}
    Standalone question:
    """
) #removes ambiguity by changing {input} to {question}

def make_hashable(obj):
    if isinstance(obj, dict):
        return tuple((k, make_hashable(v)) for k, v in sorted(obj.items()))
    elif isinstance(obj, list):
        return tuple(make_hashable(v) for v in obj)
    else:
        return obj

@st.cache_resource(show_spinner="Connecting with groq...")
def initialize_llm():
    llm_main = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
    llm_light = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")
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
    def base_retrieve(query: str):
        return db.similarity_search(
            query,
            k=8
        )

    MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template("""
    Generate 3 diverse search queries for the following question.
    Question: {question}
    """)

    query_expansion = (
        MULTI_QUERY_PROMPT
        | llm_light
        | StrOutputParser()
        | RunnableLambda(lambda x: [q.strip() for q in x.split("\n") if q.strip()])
    )

    def retrieve_many(queries):
        all_docs = []
        seen = set()

        for q in queries:
            docs = base_retrieve(q)
            for d in docs:
                key = (d.page_content, make_hashable(d.metadata))
                if key not in seen:
                    seen.add(key)
                    all_docs.append(d)

        return all_docs
    
    retrieval_parallel = RunnableParallel(
        original=RunnablePassthrough(),
        expanded=RunnableLambda(lambda q: query_expansion.invoke({"question": q}))
    )

    retrieval_runnable = (
        retrieval_parallel
        | RunnableLambda(
            lambda x: retrieve_many([x["original"]] + x["expanded"])
        )
    )

    rewrite_chain = (
    CONDENSE_PROMPT
    | llm_light
    | StrOutputParser()
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    document_chain = (
        {
            "context": lambda x: format_docs(x["docs"]),
            "input": lambda x: x["input"]
        }
        | prompt
        | llm_main
        | StrOutputParser()
    )

    retrieve = RunnableLambda(lambda inputs: {
        "docs": (
            retrieval_runnable.invoke(
                rewrite_chain.invoke({
                    "chat_history": inputs["chat_history"],
                    "question": inputs["input"]
                }) if inputs["chat_history"] else inputs["input"]
            )
        ),
        "input": inputs["input"]
    })


    final_chain = (
        retrieve
        | document_chain
    )

    return final_chain

final_chain = initialize_llm()
st.divider()
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
            answer = final_chain.invoke({
                "input": user_input,
                "chat_history": chat_history_text
            })
        except Exception as e:
            answer = "⚠️ I couldn't divine an answer. Something went wrong."
            st.error(str(e))

    # Show assistant message
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant", avatar=shyarly_avatar):
        st.markdown(f"{answer}")


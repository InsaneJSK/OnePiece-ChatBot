# 🔮 Madame Shyarly's Prophecies

## _"The sea whispers... ask what you dare about the world of One Piece."_

An intelligent Q&A chatbot built with RAG (Retrieval-Augmented Generation) that answers One Piece-related questions as Madame Shyarly 🧜‍♀️.

## [Try It!!](https://onepiece-chatbot.streamlit.app/)

---

## 📜 Overview

This project is a Streamlit-based app that lets fans explore the lore of _One Piece_ using natural language. It combines:

- 💬 Conversational memory
- 🔍 Contextual retrieval from a Qdrant vector database
- 🧠 Query rephrasing using LLMs (Multi-query retrieval)
- 🧙‍♀️ Persona-based prompt engineering for Madame Shyarly
- 🌐 Self-hosted avatars and a custom-styled chat interface

---

## 🚀 Features

- **Ask anything** about _One Piece_ characters, plot, lore, Devil Fruits, etc.
- Uses **LangChain** and **Groq**'s LLMs for blazing-fast generation.
- **Multi-query Retrieval**: Automatically expands your question to fetch more diverse and relevant results.
- **History-aware Retrieval**: Handles follow-up questions naturally by rephrasing them into standalone ones.
- Streamlit UI with **chat-like avatars** and **source expansions**.

---

## 🧱 Tech Stack

| Layer      | Tech                                         |
|------------|----------------------------------------------|
| Frontend   | Streamlit + HTML/CSS                         |
| Backend    | LangChain, Groq API                          |
| Embeddings | `all-MiniLM-L6-v2` from HuggingFace          |
| Vector DB  | Qdrant (cloud-hosted)                        |
| Models     | `llama3-8b-instant` (retriever) + `llama3-70b` (answering) |

---

## ⚙️ Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/onepiece-chatbot.git
cd onepiece-chatbot
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Set your environment variables in a `.env` file as `.env.dist`

```bash
GROQ_API_KEY=your_groq_api_key
qdrant_url=https://your-qdrant-instance-url
qdrant_api_key=your_qdrant_api_key
```

### 4. Run the app

```bash
streamlit run app.py
```

## 🧪 Example Queries

- Who is Shanks?

- What is Luffy’s bounty after Dressrosa?

- How did Ace die?

- What is Nami’s backstory?

- Who are the Gorosei?

## 🧠 Future Ideas

- TTS add-on
- refined answers
- multiple data entry such as reddits, library-of-ohara

## 🖼️ Avatar Support

Chat avatars are embedded using base64-encoded image data to support Streamlit Cloud restrictions. Make sure your images are placed inside the assets/ folder and are in .png or .jpg format.

## 👑 Credits

- Built with ❤️ by [Jaspreet Singh](https://github.com/InsaneJSK)

- Powered by Groq, LangChain, and Qdrant

- One Piece © Eiichiro Oda / Shueisha

## 📜 License

This project is for educational and fan purposes only. All One Piece intellectual property belongs to its rightful owners.

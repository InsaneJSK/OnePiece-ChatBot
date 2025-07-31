# RAG-Barebones

## Article: [Build a Wiki Chatbot Using RAG — for Free (ft. My One Piece Bot)](https://medium.com/p/36cdff9005d7)

## Setup

1. Clone the repo and change directory to RAG-barebones

    ```bash
    cd .\RAG-barebones\ #For windows
    ```

2. (Optional) Create Virtual Environment

    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3. Install the dependencies

    ```bash
    pip install -r .\requirements.txt
    ```

4. Final preparations

    - (For Groq) Setting up `.env`
        - create a new file `.env`
        - copy the contents of `.env.dist` and add in your groq api key

    Don't have a key? [Get yours here](https://console.groq.com/keys) after sign-up.

    - (For Ollama) Pull `llama3`

        ```bash
        ollama pull llama3 #or 'smol' for lower-end PCs
        ```

    Don't have Ollama installed? [Download it here](https://ollama.com/download/windows)

5. Try it!

    ```bash
    python .\main-groq.py #or .\main-ollama.py
    ```

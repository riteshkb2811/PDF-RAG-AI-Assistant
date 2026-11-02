# 🤖 PDF RAG Assistant

An interactive,  **Agentic Retrieval-Augmented Generation (Agentic-RAG)** application built with **Streamlit**, **LangChain / LangGraph**, **Groq**, and **Google Gemini Embeddings**.

Upload any PDF document dynamically, and the agent retrieves relevant context using vector similarity search to answer your questions accurately.

---

## ✨ Key Features

- **📄 Dynamic PDF Ingestion:** Upload any PDF via the sidebar (`➕ Upload a New PDF`), and the system automatically parses and embeds the document on the fly.
- **⚡ Ultra-Fast Inference:** Powered by Groq (`openai/gpt-oss-20b`) for rapid response generation.
- **🔍 Semantic Search:** Embeds document chunks using Google's `gemini-embedding-2` model and stores them in an `InMemoryVectorStore`.
- **🛠️ Agentic RAG Architecture:** Built with LangGraph's `create_agent` using a custom `@tool retrieve_data` that the LLM invokes whenever document context is required.
- **🧠 Multi-Turn Memory:** Utilizes LangGraph `InMemorySaver` checkpointer with session thread IDs to maintain conversational context.
- **🎨 Sleek Streamlit UI:** Fixed bottom search bar (`st.chat_input`), scrolling chat history, and sidebar document management.

---

## 🏗️ Architecture Flow

```
[ User Uploads PDF ]
        │
        ▼
[ PyPDFLoader ] ──► [ RecursiveCharacterTextSplitter (1000/200) ]
                            │
                            ▼
           [ GoogleGenerativeAIEmbeddings ]
                            │
                            ▼
              [ InMemoryVectorStore ]
                            ▲
                            │ similarity_search(k=3)
                            ▼
                [ @tool: retrieve_data ]
                            ▲
                            │ calls tool when needed
                            ▼
            [ LangGraph Agent (Groq LLM) ]
                            ▲
                            │
                   [ InMemorySaver ]
                    (thread_id session)
                            ▲
                            │
                 [ Streamlit Web UI ]
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- A [Groq API Key](https://console.groq.com/)
- A [Google Gemini API Key](https://aistudio.google.com/)

### 2. Installation

Clone this repository and navigate to the project directory:

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

Set up and activate a virtual environment:

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```bash
pip install streamlit langchain-groq langchain-google-genai langchain-community langchain-text-splitters langgraph pypdf python-dotenv
```

### 3. Environment Variables

Create a `.env` file in the root of your project:

```env
Gemini_KEY=your_google_gemini_api_key_here
GroqKEY=your_groq_api_key_here
```

---

## 💻 Running the Application

Launch the Streamlit app:

```bash
streamlit run pdfRAGproj.py
```

Your browser will automatically open at `http://localhost:8501`.

1. Click **➕ Upload a New PDF** in the sidebar to upload a document.
2. Ask questions about the document in the chat box at the bottom!

---

## 📂 Project Structure

```
├── pdfRAGproj.py          # Main Streamlit application & Agentic RAG pipeline
├── README.md              # Project documentation
├── .env                   # Environment variables (API keys)
└── requirement.txt        # Python package dependencies
```

---

## 📜 License
This project is open-source and licensed under the [MIT License](LICENSE).


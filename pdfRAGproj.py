import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

# ── Streamlit UI Setup ──
st.set_page_config(page_title="Medical Report Assistant", page_icon="🤖", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent1" not in st.session_state:
    st.session_state.agent1 = None
if "current_file" not in st.session_state:
    st.session_state.current_file = None

# ── Function containing YOUR EXACT BACKEND CODE ──
def init_backend(path1):
    load_dotenv()

    # Build path relative to file or project
    loader = PyPDFLoader(path1)
    doc = loader.load()

    # Split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(documents=doc)

    # Embeddings
    key1 = os.getenv("Gemini_KEY")
    embedd = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2",
        google_api_key=key1
    )

    # Vector Store
    vector_store = InMemoryVectorStore.from_documents(
        embedding=embedd,
        documents=split_docs
    )

    # LLM
    key2 = os.getenv("GroqKEY")
    llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=key2)

    @tool
    def retrieve_data(query: str) -> str:
        """Search and retrieve relevant context and details from the medical report based on a user query."""
        docs = vector_store.similarity_search(query, k=3)
        content = ""
        for d in docs:
            content += d.page_content + "\n\n"
        return content

    prompt1 = (
        "You are a helpful medical assistant that answers questions using retrieved context from uploaded documents. "
        "Always use the 'retrieve_data' tool to look up facts and data before answering."
    )

    mem = InMemorySaver()

    agent1 = create_agent(
        model=llm,
        tools=[retrieve_data],
        system_prompt=prompt1,
        checkpointer=mem
    )
    return agent1

# Default document loading on first launch
default_path = os.path.join(os.path.dirname(__file__), "medical_report.pdf")
if st.session_state.agent1 is None and os.path.exists(default_path):
    with st.spinner("Indexing default medical report..."):
        st.session_state.agent1 = init_backend(default_path)
        st.session_state.current_file = "medical_report.pdf"

# ── Sidebar for PDF Upload (➕) ──
with st.sidebar:
    st.header("📄 Document Management")
    uploaded_file = st.file_uploader("➕ Upload a New PDF", type=["pdf"], help="Upload any PDF to query it")
    if uploaded_file is not None and uploaded_file.name != st.session_state.current_file:
        with st.spinner(f"Ingesting & indexing {uploaded_file.name}..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_pdf_path = tmp_file.name

            st.session_state.agent1 = init_backend(temp_pdf_path)
            st.session_state.current_file = uploaded_file.name
            st.session_state.messages = []
            st.success(f"Loaded {uploaded_file.name} successfully!")
            st.rerun()

    st.divider()
    active_name = st.session_state.current_file or "No document loaded"
    st.info(f"📌 Active Document:\n**{active_name}**")

# ── Header ──
st.title("🤖 PDF RAG Assistant")

# ── Display previous chat messages (Top area) ──
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ── Search Input (Fixed strictly at the bottom) ──
query = st.chat_input("Ask a question about the medical report...")

if query:
    if st.session_state.agent1 is None:
        st.warning("Please upload a PDF document first!")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user", avatar="👤"):
            st.markdown(query)

        # Agent invocation
        config = {"configurable": {"thread_id": "medical_chat_session"}}
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Analyzing document..."):
                res = st.session_state.agent1.invoke(
                    {"messages": [{"role": "user", "content": query}]},
                    config
                )
                answer = res["messages"][-1].content
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

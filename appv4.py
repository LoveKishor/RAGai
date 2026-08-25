import streamlit as st
import os
import hashlib
import time
from dotenv import load_dotenv

# ------------------------------
# 1. PAGE CONFIG – MUST BE FIRST
# ------------------------------
st.set_page_config(page_title="EV-Assistant", layout="wide")

# ------------------------------
# 2. IMPORTS
# ------------------------------
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from pinecone import Pinecone, ServerlessSpec

# ------------------------------
# 3. HELPER: Get keys from Secrets or .env
# ------------------------------
def get_secret_or_env(key_name):
    try:
        value = st.secrets.get(key_name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(key_name)

def sanitize_api_key(key):
    if not key:
        return ""
    return key.strip().encode('ascii', 'ignore').decode('ascii')

# ------------------------------
# 4. LOAD ENVIRONMENT VARIABLES (local)
# ------------------------------
load_dotenv()

# ------------------------------
# 5. SESSION STATE INITIALISATION
# ------------------------------
if "pinecone_api_key" not in st.session_state:
    pinecone_key = get_secret_or_env("PINECONE_API_KEY")
    if not pinecone_key:
        st.error("🚨 PINECONE_API_KEY not found in Streamlit Secrets or .env.")
        st.stop()
    st.session_state.pinecone_api_key = sanitize_api_key(pinecone_key)

if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""
if "user_provided_groq_key" not in st.session_state:
    st.session_state.user_provided_groq_key = ""

if "upload_authorized" not in st.session_state:
    st.session_state.upload_authorized = False
if "feedback_authorized" not in st.session_state:
    st.session_state.feedback_authorized = False
if "feedback_enabled" not in st.session_state:
    st.session_state.feedback_enabled = False
if "messages" not in st.session_state:
    st.session_state.messages = []

INDEX_NAME = "self-learning-rag"
os.environ["PINECONE_DISABLE_DEPRECATED_PLUGIN_CHECK"] = "true"

# ------------------------------
# 6. FUNCTION DEFINITIONS
# ------------------------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_llm(api_key):
    if not api_key:
        st.error("Groq API key is missing.")
        st.stop()
    safe_key = sanitize_api_key(api_key)
    if not safe_key:
        st.error("Invalid Groq API key (empty after sanitization).")
        st.stop()
    return ChatGroq(
        temperature=0.3,
        groq_api_key=safe_key,
        model="llama-3.3-70b-instruct"   # ✅ Updated to current stable model
    )

@st.cache_resource
def get_cross_encoder():
    from sentence_transformers import CrossEncoder
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def get_pinecone():
    if not st.session_state.pinecone_api_key:
        st.error("Pinecone API key is missing.")
        st.stop()
    return Pinecone(api_key=st.session_state.pinecone_api_key)

def ensure_index():
    pc = get_pinecone()
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

def generate_document_id(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def ingest_pdfs(uploaded_files, namespace="pdfs"):
    embeddings = get_embeddings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    all_chunks = []
    all_ids = []
    all_metadatas = []
    for file in uploaded_files:
        with open(file.name, "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(file.name)
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            chunk_id = generate_document_id(chunk.page_content)
            all_chunks.append(chunk.page_content)
            all_ids.append(chunk_id)
            all_metadatas.append({"source": file.name})
        os.remove(file.name)
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=namespace,
        pinecone_api_key=st.session_state.pinecone_api_key
    )
    vector_store.add_texts(
        texts=all_chunks,
        metadatas=all_metadatas,
        ids=all_ids
    )
    return len(all_chunks)

def hyde_retrieve(question, k=15):
    llm = get_llm(st.session_state.groq_api_key)
    hyde_prompt = PromptTemplate.from_template(
        "Write a concise passage that answers the following question. "
        "Do not say 'I don't know'. Just write a factual passage:\n\nQuestion: {question}\n\nPassage:"
    )
    chain = hyde_prompt | llm
    hypothetical_answer = chain.invoke({"question": question}).content
    embeddings = get_embeddings()
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace="pdfs",
        pinecone_api_key=st.session_state.pinecone_api_key
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    documents = retriever.invoke(hypothetical_answer)
    return documents

def rerank_documents(query, documents, top_k=4):
    if not documents:
        return []
    cross_encoder = get_cross_encoder()
    pairs = [[query, doc.page_content] for doc in documents]
    scores = cross_encoder.predict(pairs)
    doc_score_pairs = list(zip(documents, scores))
    doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in doc_score_pairs[:top_k]]

def retrieve_memory(question, k=2):
    embeddings = get_embeddings()
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace="memory",
        pinecone_api_key=st.session_state.pinecone_api_key
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(question)

def save_to_memory(question, answer):
    embeddings = get_embeddings()
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace="memory",
        pinecone_api_key=st.session_state.pinecone_api_key
    )
    text = f"Question: {question}\nAnswer: {answer}"
    doc_id = generate_document_id(text)
    vector_store.add_texts(
        texts=[text],
        metadatas=[{"type": "user_feedback"}],
        ids=[doc_id]
    )

def generate_final_answer(question, context_docs, memory_docs):
    llm = get_llm(st.session_state.groq_api_key)
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    memory_text = ""
    if memory_docs:
        memory_text = "## 💡 Past Lessons for similar questions:\n" + "\n\n".join([doc.page_content for doc in memory_docs])
    prompt_template = PromptTemplate.from_template(
        """
        You are an expert assistant. Answer the question ONLY based on the provided context.
        
        {memory_text}
        
        ### Relevant Context from Documents:
        {context}
        
        ### Question:
        {question}
        
        If Past Lessons are relevant, apply their corrections. Be concise and factual.
        Answer:
        """
    )
    chain = prompt_template | llm
    response = chain.invoke({
        "memory_text": memory_text,
        "context": context_text,
        "question": question
    })
    return response.content

# ------------------------------
# 7. STREAMLIT UI
# ------------------------------
st.title("🧠 Welcome to the AI WORLD, This is your EV assistant")

with st.sidebar:
    st.divider()
    st.header("🤖 pass Key")
    
    is_admin = st.session_state.upload_authorized or st.session_state.feedback_authorized
    
    if is_admin:
        admin_groq_key = get_secret_or_env("GROQ_API_KEY")
        if admin_groq_key:
            safe_key = sanitize_api_key(admin_groq_key)
            st.session_state.groq_api_key = safe_key
            st.success("✅ Using Admin's Groq API Key (from Secrets or .env)")
            st.caption(f"Key starts with: {safe_key[:3]}...")
        else:
            st.error("❌ GROQ_API_KEY not found in Secrets or .env.")
    else:
        st.info("🔑 Enter pass Key to chat")
        user_key = st.text_input(
            "Your Key",
            type="password",
            value=st.session_state.user_provided_groq_key,
            key="groq_user_input",
        )
        if user_key:
            cleaned_key = sanitize_api_key(user_key)
            if cleaned_key:
                st.session_state.user_provided_groq_key = cleaned_key
                st.session_state.groq_api_key = cleaned_key
                st.success("✅key set!")
                st.caption(f"Key starts with: {cleaned_key[:3]}...")
            else:
                st.error("❌ Invalid key (empty after cleaning).")
    
    st.divider()
    st.header("🔒 Upload Protection")
    upload_password = st.text_input("Enter password to upload PDFs", type="password")
    if st.button("🔓 Verify Upload Password"):
        correct_password = get_secret_or_env("UPLOAD_PASSWORD") or "admin123"
        if upload_password == correct_password:
            st.session_state.upload_authorized = True
            st.success("✅ Access granted! You can now upload PDFs.")
            admin_groq_key = get_secret_or_env("GROQ_API_KEY")
            if admin_groq_key:
                st.session_state.groq_api_key = sanitize_api_key(admin_groq_key)
                st.success("✅ Admin Groq key loaded!")
            st.rerun()
        else:
            st.session_state.upload_authorized = False
            st.error("❌ Wrong password!")
    
    st.divider()
    st.header("👍 Feedback Mode")
    st.caption("Enable or disable user feedback (thumbs up/down).")
    feedback_password = st.text_input("Enter admin password to change feedback mode", type="password", key="feedback_password_input")
    if st.button("🔑 Authorize Feedback Settings"):
        correct_password = get_secret_or_env("UPLOAD_PASSWORD") or "admin123"
        if feedback_password == correct_password:
            st.session_state.feedback_authorized = True
            st.success("✅ Authorized! You can now toggle feedback.")
            admin_groq_key = get_secret_or_env("GROQ_API_KEY")
            if admin_groq_key:
                st.session_state.groq_api_key = sanitize_api_key(admin_groq_key)
                st.success("✅ Admin Groq key loaded!")
            st.rerun()
        else:
            st.session_state.feedback_authorized = False
            st.error("❌ Wrong password!")
    
    feedback_toggle = st.checkbox(
        "Allow User Feedback (👍/👎)",
        value=st.session_state.feedback_enabled,
        disabled=not st.session_state.feedback_authorized,
        key="feedback_checkbox"
    )
    if st.session_state.feedback_authorized:
        st.session_state.feedback_enabled = feedback_toggle
        if feedback_toggle:
            st.success("🟢 Feedback is ON")
        else:
            st.info("🔴 Feedback is OFF")
    
    st.divider()
    if st.session_state.upload_authorized:
        st.header("📤 Update Knowledge Base")
        uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        if uploaded_files and st.button("🚀 Add to Vector DB"):
            with st.spinner(f"Processing {len(uploaded_files)} PDF(s)..."):
                try:
                    num_chunks = ingest_pdfs(uploaded_files)
                    st.success(f"✅ Uploaded {num_chunks} chunks to Pinecone.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    else:
        st.info("🔒 Upload area is locked. Enter the upload password above.")

st.subheader("💬 Ask anything")

try:
    ensure_index()
except Exception as e:
    st.error(f"❌ Failed to connect to Pinecone: {e}")
    st.stop()

if not st.session_state.groq_api_key:
    st.warning("⚠️ Please enter your pass key in the sidebar to start chatting.")
    st.stop()

st.caption(f"🔑 Chat mode active")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 thinking..."):
            retrieved_docs = hyde_retrieve(prompt, k=15)
        with st.spinner("🔄 analysing..."):
            top_docs = rerank_documents(prompt, retrieved_docs, top_k=4)
        with st.spinner("🧠 Recalling past lessons..."):
            memory_docs = retrieve_memory(prompt, k=2)
        with st.spinner("💬 getting answer for you..."):
            answer = generate_final_answer(prompt, top_docs, memory_docs)
            st.markdown(answer)
        
        if st.session_state.feedback_enabled:
            col1, col2, col3 = st.columns([1, 1, 8])
            with col1:
                if st.button("👍 Good", key=f"up_{time.time()}"):
                    save_to_memory(prompt, answer)
                    st.toast("✅ Memorized this good Q&A!")
            with col2:
                if st.button("👎 Wrong", key=f"down_{time.time()}"):
                    st.session_state["correction_prompt"] = prompt
                    st.session_state["waiting_for_correction"] = True
                    st.rerun()
        else:
            if st.session_state.upload_authorized or st.session_state.feedback_authorized:
                st.caption("chat only mode")
    
    st.session_state.messages.append({"role": "assistant", "content": answer})

if st.session_state.get("waiting_for_correction", False):
    with st.chat_message("assistant"):
        st.warning("🤔 I got it wrong. What is the correct answer?")
        correction = st.text_input("Enter the correct answer:", key="correction_input")
        if st.button("Submit Correction"):
            if correction.strip():
                if st.session_state.feedback_enabled:
                    save_to_memory(st.session_state.correction_prompt, correction)
                    st.success("✅ Correction saved! I will use this for future similar questions.")
                else:
                    st.warning("⚠️ Feedback is OFF. Correction not saved.")
                st.session_state["waiting_for_correction"] = False
                st.rerun()
            else:
                st.error("Please enter a valid correction.")
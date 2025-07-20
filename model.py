import os
import streamlit as st
from pathlib import Path
import subprocess
import uuid
import json
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA

# Paths
DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = "./data"

# Create required directories
os.makedirs(DATA_PATH, exist_ok=True)

def init_state():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.citations = []
    st.session_state.trace = {}
    st.session_state.chat_enabled = False  # Ensure chat functionality is disabled initially
    st.session_state.questions = ""  # Initialize for pasting questions
    st.session_state.script_output = ""  # To store the script output

# General page configuration
st.set_page_config(page_title="Document Chatbot", page_icon="\U0001F4C4", layout="wide")
st.title("Document Chatbot")

# Initialize session state
if len(st.session_state.items()) == 0:
    init_state()

# Sidebar for upload and ingestion
with st.sidebar:
    st.title("PDF Document Processor")

    # Reset session button
    if st.button("Reset Session"):
        init_state()

    # Check if the vector store exists
    if not st.session_state.get("chat_enabled", False):
        st.write("## Step 1: Upload PDF Documents")

        # File uploader
        uploaded_files = st.file_uploader(
            "Upload your PDF files here:",
            type="pdf",
            accept_multiple_files=True,
            help="You can upload multiple PDF files, up to 20MB each."
        )

        if uploaded_files:
            # Save uploaded files
            saved_files = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(DATA_PATH, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
                saved_files.append(uploaded_file.name)

            st.success(f"\U0001F4CE {len(saved_files)} PDF file(s) uploaded:")
            for fname in saved_files:
                st.write(f"- {fname}")

            # Provide an action to process files
            if st.button("Ingest Documents"):
                st.info("Document ingestion process initiated...")
                try:
                    # Run the ingest.py script
                    subprocess.run(["python", "ingest.py"], check=True)
                    st.success("Documents have been ingested successfully!")
                    st.session_state["chat_enabled"] = True  # Enable chat functionality
                except subprocess.CalledProcessError as e:
                    st.error(f"Error during ingestion: {e}")
    else:
        st.session_state["chat_enabled"] = True

# Define prompt and retrieval chain
custom_prompt_template = """
Use the following pieces of information to answer the user's question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    prompt = PromptTemplate(
        template=custom_prompt_template,
        input_variables=['context', 'question']
    )
    return prompt

def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def load_llm():
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    db = FAISS.load_local(
        DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
    )
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa

# Add new functionality for bulk questions
st.header("Bulk Question Processing")

# Input text box for pasting questions
questions = st.text_area(
    "Paste your questions here:",
    height=200,
    value=st.session_state.get("questions", ""),
    key="questions_input"
)

# Run scripts button
if st.button("Generate form"):
    st.info("Running scripts...")
    try:
        # Run the first script and pass the text as an argument
        subprocess.run(
            ["python", "questions_to_csv.py", "--text", questions], 
            check=True
        )
        
        # Run the second script and capture output
        result = subprocess.run(
            ["python", "create_form.py"], 
            capture_output=True, text=True, check=True
        )
        st.session_state["script_output"] = result.stdout

        st.success("Scripts ran successfully!")
    except subprocess.CalledProcessError as e:
        st.error(f"Error running scripts: {e}")


if st.session_state.get("script_output"):
    st.subheader("Script Output")
    st.text(st.session_state["script_output"])

if st.session_state.get("chat_enabled", False):
    st.write("You can now interact with the document bot.")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    # Chat input box
    if user_query := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("...")

            qa_model = qa_bot()
            response = qa_model({"query": user_query})

            if "result" in response:
                output_text = response['result']
                st.session_state.messages.append({"role": "assistant", "content": output_text})
                placeholder.markdown(output_text, unsafe_allow_html=True)

                # Handle citations if available
                if "source_documents" in response:
                    citations = response["source_documents"]
                    st.session_state.citations = citations
                    with st.sidebar:
                        st.subheader("Citations")
                        for doc in citations:
                            st.text(f"Source: {doc.metadata.get('source', 'Unknown')}")

            else:
                placeholder.markdown("No answer available.")
                st.session_state.messages.append({"role": "assistant", "content": "No answer available."})
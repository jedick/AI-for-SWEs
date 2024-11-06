import os
import gradio as gr
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

local_ok = True
# The try block lets this app work without installing a local model
try:
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except ImportError:
    local_ok = False

import fitz  # PyMuPDF
import sqlite3
from datetime import datetime
import threading
import uuid

# Thread-local storage for database connections
local = threading.local()

# Function to get a thread-local database connection
def get_db_connection():
    if not hasattr(local, "db_conn"):
        local.db_conn = sqlite3.connect('qa_traces.db', check_same_thread=False)
        local.db_conn.execute('''CREATE TABLE IF NOT EXISTS conversations
                                 (id TEXT PRIMARY KEY, timestamp TEXT, filename TEXT, model TEXT)''')
        local.db_conn.execute('''CREATE TABLE IF NOT EXISTS messages
                                 (id TEXT PRIMARY KEY, conversation_id TEXT, 
                                  timestamp TEXT, role TEXT, content TEXT,
                                  FOREIGN KEY(conversation_id) REFERENCES conversations(id))''')
        local.db_conn.commit()
    return local.db_conn

# Function to start a new conversation
def start_conversation(pdf_file):
    conn = get_db_connection()
    c = conn.cursor()
    conversation_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    filename = os.path.basename(pdf_file)
    modelstring = Settings.llm.model
    c.execute("INSERT INTO conversations VALUES (?, ?, ?, ?)", (conversation_id, timestamp, filename, modelstring))
    conn.commit()
    return conversation_id

# Function to log a message in a conversation
def log_message(conversation_id, role, content):
    conn = get_db_connection()
    c = conn.cursor()
    message_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO messages VALUES (?, ?, ?, ?, ?)", 
              (message_id, conversation_id, timestamp, role, content))
    conn.commit()

# Function to extract text from the PDF
def extract_text_from_pdf(pdf_file_bytes):
    pdf_doc = fitz.open(stream=pdf_file_bytes, filetype="pdf")
    text = ""
    for page_num in range(pdf_doc.page_count):
        page = pdf_doc.load_page(page_num)
        text += page.get_text("text")
    return text

# Function to process the uploaded PDF and create an index
def process_pdf(pdf_file_bytes):
    extracted_text = extract_text_from_pdf(pdf_file_bytes)
    document = Document(text=extracted_text)
    index = VectorStoreIndex.from_documents([document])
    return index

# Function to handle conversation, with option for model choice and logging traces
def query_pdf(pdf_file, query, history, conversation_id, model_choice):
    if pdf_file is None:
        return [("Please upload a PDF.", "")], history, conversation_id
    if not query.strip():
        return [("Please enter a valid query.", "")], history, conversation_id

    # Choose between local (Ollama) or OpenAI model
    if model_choice == "Local (Ollama)":
        # Use Ollama and local embedding model
        Settings.llm = Ollama(model="llama3.2", request_timeout=60.0)
        Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    elif model_choice == "OpenAI":
        # Use OpenAI's LLM and embedding model
        Settings.llm = OpenAI(model = "gpt-4o-mini")
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # Get PDF file as binary
    with open(pdf_file, mode="rb") as f:
        pdf_file_bytes = f.read()
    
    # Start a new conversation if there isn't one
    # This is done after the model setting so that the model setting can be logged
    if conversation_id is None:
        conversation_id = start_conversation(pdf_file)
    
    try:
        # Process the PDF file bytes directly
        index = process_pdf(pdf_file_bytes)
        # Set up the query engine with the selected LLM
        query_engine = index.as_query_engine()
        
        # Add previous conversation to the query for context
        conversation = ""
        for h in history:
            conversation += f"User: {h[0]}\nAssistant: {h[1]}\n"
        conversation += f"User: {query}\n"

        # Log the user's query
        log_message(conversation_id, "user", query)
        
        # Query the index using the user's question with context
        response = query_engine.query(conversation)
        
        # Log the assistant's response
        log_message(conversation_id, "assistant", response.response)
        
        # Update the conversation history with a tuple (user's query, model's response)
        history.append((query, response.response))
        # Return the updated history (list of tuples) and the conversation ID
        return history, history, conversation_id
    except Exception as e:
        error_message = str(e)
        # Log the error
        log_message(conversation_id, "system", f"Error: {error_message}")
        return [("An error occurred", error_message)], history, conversation_id

# Gradio interface setup
with gr.Blocks() as app:
    # Get filepath so that the filename can be logged; we'll read the file as binary later
    pdf_file = gr.File(label="Upload PDF", type="filepath")
    query_input = gr.Textbox(label="Ask a question about the PDF")
    if local_ok:
      model_choice = gr.Radio(label="Select Model", choices=["Local (Ollama)", "OpenAI"], value="Local (Ollama)")
    else:
      model_choice = gr.Radio(label="Select Model - Local (Ollama) is not available!", choices=["OpenAI"], value="OpenAI")
    output = gr.Chatbot(label="Conversation")
    history_state = gr.State([])  # Store conversation history
    conversation_id_state = gr.State(None)  # Store conversation ID
    
    query_button = gr.Button("Submit")
    query_button.click(fn=query_pdf, 
                       inputs=[pdf_file, query_input, history_state, conversation_id_state, model_choice], 
                       outputs=[output, history_state, conversation_id_state])

app.launch()

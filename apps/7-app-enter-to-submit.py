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
        local.db_conn.execute('''CREATE TABLE IF NOT EXISTS feedback
                                 (id TEXT PRIMARY KEY, message_id TEXT, feedback INTEGER, 
                                  timestamp TEXT, FOREIGN KEY(message_id) REFERENCES messages(id))''')
        local.db_conn.commit()
    return local.db_conn

# Call this function on app launch to ensure the database is created upfront
get_db_connection()

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
    print(f"New conversation started with ID: {conversation_id}")
    return conversation_id

# Function to log a message in a conversation
def log_message(conversation_id, role, content):
    conn = get_db_connection()
    c = conn.cursor()
    message_id = str(uuid.uuid4())  # Generate a new message ID
    timestamp = datetime.now().isoformat()  # Get the current timestamp
    c.execute("INSERT INTO messages VALUES (?, ?, ?, ?, ?)", 
              (message_id, conversation_id, timestamp, role, content))
    conn.commit()
    return message_id  # Return the message ID properly

# Function to log feedback (thumbs-up or thumbs-down)
def log_feedback(message_id, feedback_value):
    conn = get_db_connection()
    c = conn.cursor()
    feedback_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO feedback VALUES (?, ?, ?, ?)", 
              (feedback_id, message_id, feedback_value, timestamp))
    conn.commit()
    print(f"Feedback logged: {feedback_id} | Message ID: {message_id} | Feedback: {feedback_value}")

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

# Complete query_pdf function with proper logging of messages
def query_pdf(pdf_file, query, messages, history, conversation_id, model_choice, message_id_state):
    if pdf_file is None:
        messages.append({"role":"assistant", "content":"Please upload a PDF."})
        return messages, history, conversation_id, message_id_state
    if not query.strip():
        messages.append({"role":"assistant", "content":"Please enter a valid query."})
        return messages, history, conversation_id, message_id_state

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
    if conversation_id is None:
        conversation_id = start_conversation(pdf_file)

    try:
        # Process the PDF and create an index
        index = process_pdf(pdf_file_bytes)
        query_engine = index.as_query_engine()

        # Construct the conversation string
        conversation = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in history])
        conversation += f"\nUser: {query}\n"

        # Log user's query and update state
        user_message_id = log_message(conversation_id, "user", query)
        # Query the index
        response = query_engine.query(conversation)
        # Log assistant's response and update state
        assistant_message_id = log_message(conversation_id, "assistant", response.response)

        # Update conversation history
        history.append((query, response.response))
        # Update chatbot messages
        messages.append({"role":"user", "content":query})
        messages.append({"role":"assistant", "content":response.response})
    except Exception as e:
        error_message = str(e)
        log_message(conversation_id, "system", f"Error: {error_message}")
        messages.append({"role":"assistant", "content":"An error occurred: "+error_message})

    return messages, history, conversation_id, assistant_message_id

# Function to handle thumbs-up feedback
def handle_thumbs_up(message_id):
    if message_id:
        log_feedback(message_id, 1)  # Log thumbs-up as 1
    return "Feedback logged: 👍"

# Function to handle thumbs-down feedback
def handle_thumbs_down(message_id):
    if message_id:
        log_feedback(message_id, 0)  # Log thumbs-down as 0
    return "Feedback logged: 👎"

# Gradio interface setup
with gr.Blocks() as app:
    # Get filepath so that the filename can be logged; we'll read the file as binary later
    pdf_file = gr.File(label="Upload PDF", type="filepath")
    # This holds the error messages and chat history to display in chatbot
    messages = gr.Chatbot(label="Conversation", type="messages")
    # This holds the chat history for providing context to the conversation
    history = gr.State([])
    conversation_id_state = gr.State(None)  # Store conversation ID
    message_id_state = gr.State(None)  # Store message ID for feedback

    query_input = gr.Textbox(label="Ask a question about the PDF")
    if local_ok:
        model_choice = gr.Radio(label="Select Model", choices=["Local (Ollama)", "OpenAI"], value="Local (Ollama)")
    else:
        model_choice = gr.Radio(label="Select Model - Local (Ollama) is not available!", choices=["OpenAI"], value="OpenAI")
    query_button = gr.Button("Submit")

    # Feedback message output
    feedback_message = gr.Textbox(label="Feedback Status", interactive=False)

    # Feedback buttons
    with gr.Row():
        thumbs_up_button = gr.Button("👍")
        thumbs_down_button = gr.Button("👎")

    # Hit Enter or press submit button to submit, then clear query input field
    # https://www.gradio.app/guides/blocks-and-event-listeners
    def clear_query():
        return ""

    gr.on(
        triggers=[query_input.submit, query_button.click],
        fn=query_pdf,
        inputs=[pdf_file, query_input, messages, history, conversation_id_state, model_choice, message_id_state],
        outputs=[messages, history, conversation_id_state, message_id_state],
    ).then(clear_query, outputs=[query_input])


    # Connect feedback buttons to logging functions
    thumbs_up_button.click(fn=handle_thumbs_up, inputs=[message_id_state], outputs=feedback_message)
    thumbs_down_button.click(fn=handle_thumbs_down, inputs=[message_id_state], outputs=feedback_message)

app.launch()

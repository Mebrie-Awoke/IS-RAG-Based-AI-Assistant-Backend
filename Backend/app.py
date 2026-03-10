from flask import Flask, render_template, jsonify, request
from src.document_ingestion import download_hugging_face_embeddings
from langchain_chroma import Chroma  # Changed from langchain_pinecone
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

load_dotenv()

# Removed PINECONE_API_KEY since we're using ChromaDB
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

os.environ["GROQ_API_KEY"] = GROQ_API_KEY

embeddings = download_hugging_face_embeddings()

# Define the persistent directory for ChromaDB
persist_directory = "./chroma_db"  # Local folder to store the vector database

# Check if the ChromaDB exists and load it
if os.path.exists(persist_directory):
    # Load existing ChromaDB
    docsearch = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    print("Loaded existing ChromaDB from", persist_directory)
else:
    # If no existing database, you'll need to create one
    # This part would typically be in a separate script for ingesting documents
    print("No existing ChromaDB found. Please run ingestion script first.")
    # For now, create an empty one (you'll need to populate it)
    docsearch = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Using Groq as before
chatModel = ChatGroq(
    model="llama-3.3-70b-versatile",  # Latest Llama model
    temperature=0.1,
    max_tokens=1024,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Frontend', 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Frontend', 'static'))

app = Flask(__name__, 
            template_folder=template_dir,
            static_folder=static_dir)

@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False) 
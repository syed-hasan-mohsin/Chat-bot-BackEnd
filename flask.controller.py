import json
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import SVMRetriever
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from flask import Flask, jsonify, request
import google.generativeai as palm
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter 
from langchain.document_loaders import UnstructuredPDFLoader  
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


import shutil



import chromadb
import tiktoken
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set the upload folder path
UPLOAD_FOLDER = 'uploads'
KNOWLEDGE_BASE = 'knowledge_base'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.environ["OPENAI_API_KEY"] = "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"
# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/api/upload', methods=['POST'])
def upload_file():

    file = request.files['files']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = file.filename
        upload_file_path = os.path.join(UPLOAD_FOLDER, filename)
        knowledge_base_file_path = os.path.join(KNOWLEDGE_BASE, filename)
        if os.path.exists(knowledge_base_file_path):
            return (f"File {filename} already exists in knowledge base")
        else:
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            loader = PyPDFDirectoryLoader("./uploads")
            data = loader.load()
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            all_splits = text_splitter.split_documents(data)
            persist_directory = "chroma_db"
            global vectorstore
            vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(), persist_directory=persist_directory)
            shutil.move(upload_file_path, knowledge_base_file_path)
            return jsonify({'message': 'File successfully uploaded'})

@app.route('/api/ask', methods=['GET'])
def get_data():

    loader = PyPDFDirectoryLoader("./uploads")
    data = loader.load()
    question = request.args['question']
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    all_splits = text_splitter.split_documents(data)
    persist_directory = "chroma_db"

    print(vectorstore)
    model_name = "gpt-3.5-turbo"
    os.environ["GOOGLE_API_KEY"] = 'AIzaSyBMGQqaazkwud84u7bH8N1aQMiV42zuoO0'
    palm.configure(api_key=os.environ["GOOGLE_API_KEY"])
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
    memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=True
    )
    result = conversation_chain({'question': request.args['question']})
    print(result['answer'])
    # chain = load_qa_chain(llm, chain_type="stuff",verbose=True)
    # response = chain.run(input_documents=docs, question=question)
    return jsonify(result['answer'])

@app.route('/api/getAllPDF', methods=['GET'])
def get_pdfs():
    pdfs = []
    for file in os.listdir('./knowledge_base'):
        if file.endswith('.pdf'):
            pdfs.append({
                'name': file,
                'size': os.path.getsize(os.path.join(KNOWLEDGE_BASE, file))
            })
    return jsonify(pdfs)

if __name__ == '__main__':
    app.run(debug=True)


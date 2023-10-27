from flask import Flask, jsonify, request
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import SVMRetriever
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import chromadb
import tiktoken
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set the upload folder path
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.environ["OPENAI_API_KEY"] = "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"
# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    print(request)

    file = request.files['files']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'message': 'File successfully uploaded'})

@app.route('/api/ask', methods=['GET'])
def get_data():
    loader = PyPDFDirectoryLoader("./uploads")
    data = loader.load()
    print("DATA",data)
    question = request.args['question']
    text_splitter = RecursiveCharacterTextSplitter()
    all_splits = text_splitter.split_documents(data)
    persist_directory = "chroma_db"
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings(), persist_directory=persist_directory)
    docs = vectorstore.similarity_search(question)
    print(docs)
    model_name = "gpt-3.5-turbo"
    llm = ChatOpenAI(model_name=model_name)
    chain = load_qa_chain(llm, chain_type="stuff",verbose=True)
    response = chain.run(input_documents=docs, question=question)
    print(response)
    return jsonify(response)

@app.route('/api/test', methods=['GET'])
def test():
    return "API WORKS"

if __name__ == '__main__':
    app.run(debug=True)

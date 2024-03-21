# Importing required libraries
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from flask import Flask, request, jsonify

app = Flask(__name__)

# Define the files and their access permissions
class FileAccessControl:
    def __init__(self):
        # Define files and their access permissions in a dictionary
        self.files = {
            "file1.txt": {"user_1", "user_2"},
            "file2.txt": {"user_1", "user_2"},
            # More files with corresponding access permissions
        }

    def get_accessible_files(self, user):
        # Retrieve list of files accessible to the user
        accessible_files = []
        for file_name, access_list in self.files.items():
            if user in access_list:
                accessible_files.append(path+"/"+file_name)
        return accessible_files


# Function reads text files in the path
def readFiles(path):
    # Load text files from a directory using multithreading
    loader = DirectoryLoader(path, glob="**/*.txt", loader_cls=TextLoader, use_multithreading=True, show_progress=True)
    docs = loader.load()
    return docs


# Convert text to chunks
def textToChunks(docs):
    # Initialize RecursiveCharacterTextSplitter
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    # Split documents into chunks
    chunks = r_splitter.split_documents(docs)
    return chunks


# Converts chunks to embeddings and stores in vectorDB
def createStoreEmbeddings(chunks):
    # Initialize OpenAIEmbeddings
    openai_embeddings = OpenAIEmbeddings(openai_api_key="YOUR_API_KEY")  # Pass the API key
    vector_index = FAISS.from_documents(chunks, openai_embeddings)
    vector_index.save_local("doc_index_store")

    # Load the vector index
    vectordb = FAISS.load_local("doc_index_store", openai_embeddings)
    return vectordb

def loadLlm():
    # Initialize ChatOpenAI for language model
    llm = ChatOpenAI(temperature=0, openai_api_key="YOUR_API_KEY")
    return llm


def promptTemplate(question):
    # Create PromptTemplate from the template
    template = f"""Answer the below question. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. 
    Question: {question}
    Helpful Answer:"""
    # Create PromptTemplate from the template
    qa_chain_prompt = PromptTemplate.from_template(template)
    return qa_chain_prompt


# Flask API call to handle user query
@app.route('/query', methods=['POST'])
def llmQA():
    """This function takes the user query through POST request
    build prompt template, Checks for user file permission and retrieves
    the chunks from the docs that user have
    access to by utilizing the filter function
    and generates response using OpenAI LLM
    and returns the response and source docs in JSON format.
    """
    if request.method == 'POST':
        user = request.form.get('user')
        query = request.form.get('query')
        # Create prompt template from the query
        qa_chain_prompt = promptTemplate(query)
        access_control = FileAccessControl()
        files = access_control.get_accessible_files(user)
        # Create RetrievalQA chain for question answering
        chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vector_db.as_retriever(filter={'source': files}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": qa_chain_prompt}
        )
        # Get result from the QA chain
        result = chain({"query": query})
        return jsonify({'result': result["result"], 'source_docs': result["source_documents"]})


if __name__ == '__main__':
    """Reads files, converts text to chunks, 
    creates embeddings, loads LLM, 
    and runs the Flask app."""
    path = 'sample_project_text_files'
    # Read text files from the given path
    docs = readFiles(path)
    # Convert text documents to chunks
    chunks = textToChunks(docs)
    # Create embeddings from chunks and store in vector store
    vector_db = createStoreEmbeddings(chunks)
    # Load the language model
    llm = loadLlm()
    # Run the Flask app
    app.run(debug=True, host="0.0.0.0")
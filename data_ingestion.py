from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers  
from langchain.chains import RetrievalQA
import chainlit as cl



# Paths to data and FAISS database
data_path = "data/"  # Directory containing PDF files for processing
DB_FAISS_PATH = "vector_stores/db_faiss"  # Path where the FAISS index will be saved

# Function to create vector database
def create_vector_db():

    # Load PDF documents from the 'data' directory
    loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()  # List of loaded documents
    
    # Split the text of the documents into chunks for better processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)  # List of text chunks

    # Create embeddings using a pre-trained sentence-transformers model
    embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',  # Model for generating embeddings
        model_kwargs={'device': 'cpu'}  # Use CPU for computation
    )
    
    # Create a FAISS index from the document chunks and their embeddings
    db = FAISS.from_documents(texts, embeddings)
    
    # Save the FAISS index to the specified path for future use
    db.save_local(DB_FAISS_PATH)

# Run the vector database creation function if this script is executed directly
if __name__ == '__main__':
    create_vector_db()

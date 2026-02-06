from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
load_dotenv() 

# Load the PDF document
file_path ="./data/PDF-Guide-Node-Andrew-Mead-v3.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

# Split the document into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
splits= text_splitter.split_documents(docs)

# Create embeddings for the document chunks
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Store the embeddings in a vector store
vector_store = FAISS.from_documents(splits, embeddings)
vector_store.save_local("Autonomous Chat-with-PDF Agent/vectorstore_db")

print("✅ PDF Indexed Successfully! (Ready to Answer)")
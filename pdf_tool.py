import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.tools import create_retriever_tool

# Create a retriever tool for the PDF document
def create_pdf_retriever_tool():
    current_dir= os.path.dirname(os.path.abspath(__file__))
    env_db_path = os.getenv("VECTORSTORE_PATH")
    base_db_path = env_db_path or os.path.join(current_dir, "vectorstore_db")
    candidate_paths = [
        base_db_path,
        os.path.join(base_db_path, "vectorstore_db"),
    ]

    db_path = None
    for path in candidate_paths:
        if os.path.exists(os.path.join(path, "index.faiss")) and os.path.exists(os.path.join(path, "index.pkl")):
            db_path = path
            break

    if not db_path:
        raise FileNotFoundError(
            "Vector Database not found. Set VECTORSTORE_PATH or run indexing to create index.faiss/index.pkl."
        )

    print(f"📂 Loading Vector DB from: {db_path}")
    
#Load Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    #convert text to vector

    #Load the vector store from the local directory
    vector_store = FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True 
    )

     #Create a retriever tool using the loaded vector store
    retriver=vector_store.as_retriever(search_kwargs={"k": 3})

    #tool me pack kro
    tool = create_retriever_tool(
        retriever=retriver,
        name="PDF_Retriever",
        description="node_expert"
        "Use this tool to answer questions about node from the uploaded PDF."
        "Strict PDF Expert: Use this tool for Node.js questions. Answer ONLY using the context provided by this tool. If the information is not in the tool, say 'I don't know'. Do not add external information."
    )
    return tool

print("✅ PDF Retriever Tool Created Successfully!")

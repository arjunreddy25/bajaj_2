import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec # Import the main Pinecone client
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration ---
PINECONE_INDEX_NAME = "bajaj-hackathon-index"
SOURCE_DOCUMENTS_DIR = "E:/bajaj_2/data"

# The embedding dimension for the Gemini model 'models/embedding-001' is 768
GEMINI_EMBEDDING_DIMENSION = 768

def load_all_documents(directory_path: str):
    """Loads all supported documents (.docx, .pdf) from a given directory."""
    # (This function remains the same)
    document_loaders = []
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if filename.endswith(".docx"):
            document_loaders.append(Docx2txtLoader(filepath))
        elif filename.endswith(".pdf"):
            document_loaders.append(PyPDFLoader(filepath))
    docs = []
    for loader in document_loaders:
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {loader.file_path}: {e}")
    print(f"Successfully loaded {len(docs)} document pages.")
    return docs

def chunk_documents_into_clauses(docs: list):
    """Splits the loaded documents into smaller, semantically meaningful chunks."""
    # (This function remains the same)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150,
        length_function=len
    )
    chunked_docs = text_splitter.split_documents(docs)
    print(f"Split documents into {len(chunked_docs)} chunks.")
    return chunked_docs

def run_indexing_pipeline():
    """
    The main function that executes the end-to-end indexing process.
    This version includes logic to create the Pinecone index if it doesn't exist.
    """
    google_api_key = os.getenv("GOOGLE_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not google_api_key or not pinecone_api_key:
        raise ValueError("GOOGLE_API_KEY and PINECONE_API_KEY must be set in the .env file")

    print("--- Starting Document Indexing Pipeline ---")

    # --- NEW: Initialize Pinecone Client and Create Index if it Doesn't Exist ---
    print("\nStep 1: Initializing Pinecone client...")
    pc = Pinecone(api_key=pinecone_api_key)

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        print(f"Index '{PINECONE_INDEX_NAME}' not found. Creating a new one...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=GEMINI_EMBEDDING_DIMENSION,  # Critical: Must match your embedding model's dimension
            metric="cosine",  # 'cosine' is standard for semantic similarity
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1' # You can choose a different region
            )
        )
        # Wait for the index to be ready
        while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
            print("Waiting for index to be ready...")
            time.sleep(1)
        print("Index created successfully!")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists. Proceeding to add documents.")
    # --------------------------------------------------------------------------

    print("\nStep 2: Loading documents from source directory...")
    documents = load_all_documents(SOURCE_DOCUMENTS_DIR)
    if not documents:
        print("No documents found. Exiting.")
        return

    print("\nStep 3: Chunking documents into clauses...")
    chunked_documents = chunk_documents_into_clauses(documents)

    print("\nStep 4: Initializing Gemini embedding model...")
    gemini_embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=google_api_key
    )

    print(f"\nStep 5: Uploading documents and embeddings to Pinecone index '{PINECONE_INDEX_NAME}'...")
    # Now this command will succeed because the index is guaranteed to exist.
    PineconeVectorStore.from_documents(
        documents=chunked_documents,
        embedding=gemini_embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    
    print("\n--- ✅ Indexing Pipeline Complete ---")
    print("Your documents are now vectorized and stored in Pinecone, ready for querying.")

if __name__ == "__main__":
    run_indexing_pipeline()

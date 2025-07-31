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
    
    print("\n--- Indexing Pipeline Complete ---")
    print("Your documents are now vectorized and stored in Pinecone, ready for querying.")

if __name__ == "__main__":
    run_indexing_pipeline()


# import os
# import time
# import re
# import numpy as np
# from typing import List, Dict, Any
# from dotenv import load_dotenv
# from pinecone import Pinecone, ServerlessSpec
# from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
# from langchain.schema import Document
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import threading

# load_dotenv()

# PINECONE_INDEX_NAME = "bajaj-hackathon-index"
# SOURCE_DOCUMENTS_DIR = "E:/bajaj_2/data"
# GEMINI_EMBEDDING_DIMENSION = 768

# class UltraFastIndexing:
#     def __init__(self):
#         """Initialize ultra-fast indexing system"""
#         print("⚡ Initializing Ultra-Fast Indexing System...")
        
#         self.embedding_model = GoogleGenerativeAIEmbeddings(
#             model="models/embedding-001", 
#             google_api_key=os.getenv("GOOGLE_API_KEY"),
#             # Optimize for speed
#             request_options={
#                 "timeout": 30,
#                 "retry": {"max_retries": 2}
#             }
#         )
        
#         self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
#         self.setup_pinecone_fast()
        
#     def setup_pinecone_fast(self):
#         """Fast Pinecone setup"""
#         try:
#             # Check if index exists quickly
#             existing_indexes = self.pinecone_client.list_indexes().names()
            
#             if PINECONE_INDEX_NAME not in existing_indexes:
#                 print(f"⚡ Creating index {PINECONE_INDEX_NAME}...")
#                 self.pinecone_client.create_index(
#                     name=PINECONE_INDEX_NAME,
#                     dimension=GEMINI_EMBEDDING_DIMENSION,
#                     metric="cosine",
#                     spec=ServerlessSpec(cloud='aws', region='us-east-1')
#                 )
                
#                 # Quick ready check
#                 max_wait = 30  # Maximum 30 seconds
#                 wait_time = 0
#                 while wait_time < max_wait:
#                     try:
#                         status = self.pinecone_client.describe_index(PINECONE_INDEX_NAME).status
#                         if status.get('ready', False):
#                             break
#                     except:
#                         pass
#                     time.sleep(2)
#                     wait_time += 2
                    
#             print("✅ Pinecone ready!")
            
#         except Exception as e:
#             print(f"Pinecone setup error: {e}")

#     def fast_document_loading(self, directory_path: str) -> List[Document]:
#         """Ultra-fast parallel document loading"""
#         print("⚡ Fast loading documents...")
        
#         # Get all files first
#         files = []
#         for filename in os.listdir(directory_path):
#             if filename.endswith(('.pdf', '.docx')):
#                 files.append(os.path.join(directory_path, filename))
        
#         if not files:
#             return []
        
#         documents = []
        
#         def load_single_file(filepath):
#             try:
#                 if filepath.endswith('.pdf'):
#                     loader = PyPDFLoader(filepath)
#                 else:
#                     loader = Docx2txtLoader(filepath)
#                 return loader.load()
#             except Exception as e:
#                 print(f"Error loading {filepath}: {e}")
#                 return []
        
#         # Parallel loading
#         with ThreadPoolExecutor(max_workers=min(4, len(files))) as executor:
#             futures = {executor.submit(load_single_file, filepath): filepath for filepath in files}
            
#             for future in as_completed(futures):
#                 docs = future.result()
#                 documents.extend(docs)
        
#         print(f"⚡ Loaded {len(documents)} pages in parallel")
#         return documents

#     def lightning_chunking(self, documents: List[Document]) -> List[Document]:
#         """Lightning-fast chunking with minimal processing"""
#         print("⚡ Lightning chunking...")
        
#         chunks = []
        
#         for doc_idx, doc in enumerate(documents):
#             text = doc.page_content
            
#             # Super simple but effective chunking
#             # Split by double newlines first (paragraphs)
#             paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
#             for para_idx, paragraph in enumerate(paragraphs):
#                 if len(paragraph) < 50:  # Skip tiny chunks
#                     continue
                
#                 if len(paragraph) <= 600:  # Perfect size
#                     chunks.append(Document(
#                         page_content=paragraph,
#                         metadata={
#                             **doc.metadata,
#                             'chunk_id': f"fast_{doc_idx}_{para_idx}",
#                             'word_count': len(paragraph.split())
#                         }
#                     ))
#                 else:
#                     # Quick sentence split for long paragraphs
#                     sentences = re.split(r'(?<=[.!?])\s+', paragraph)
#                     current_chunk = ""
                    
#                     for sentence in sentences:
#                         if len(current_chunk + sentence) <= 600:
#                             current_chunk += sentence + " "
#                         else:
#                             if current_chunk.strip():
#                                 chunks.append(Document(
#                                     page_content=current_chunk.strip(),
#                                     metadata={
#                                         **doc.metadata,
#                                         'chunk_id': f"fast_{doc_idx}_{para_idx}_{len(chunks)}",
#                                         'word_count': len(current_chunk.split())
#                                     }
#                                 ))
#                             current_chunk = sentence + " "
                    
#                     if current_chunk.strip():
#                         chunks.append(Document(
#                             page_content=current_chunk.strip(),
#                             metadata={
#                                 **doc.metadata,
#                                 'chunk_id': f"fast_{doc_idx}_{para_idx}_{len(chunks)}",
#                                 'word_count': len(current_chunk.split())
#                             }
#                         ))
        
#         print(f"⚡ Created {len(chunks)} chunks instantly")
#         return chunks

#     def turbo_embedding_and_storage(self, chunks: List[Document]):
#         """Turbo-charged embedding creation and storage"""
#         print("⚡ Turbo embedding and storage...")
        
#         try:
#             # Direct PineconeVectorStore creation - fastest method
#             start_embed_time = time.time()
            
#             # Use the most optimized approach
#             vectorstore = PineconeVectorStore.from_documents(
#                 documents=chunks,
#                 embedding=self.embedding_model,
#                 index_name=PINECONE_INDEX_NAME,
#                 batch_size=100  # Larger batches for speed
#             )
            
#             embed_time = time.time() - start_embed_time
#             print(f"⚡ Embedding + Storage completed in {embed_time:.2f} seconds")
            
#         except Exception as e:
#             print(f"Error in embedding/storage: {e}")
#             # Fallback method
#             self.fallback_embedding_storage(chunks)

#     def fallback_embedding_storage(self, chunks: List[Document]):
#         """Fallback faster method if main fails"""
#         print("⚡ Using fallback storage method...")
        
#         try:
#             # Batch process embeddings
#             texts = [chunk.page_content for chunk in chunks]
            
#             # Create embeddings in batches
#             batch_size = 50
#             all_embeddings = []
            
#             for i in range(0, len(texts), batch_size):
#                 batch_texts = texts[i:i+batch_size]
#                 batch_embeddings = self.embedding_model.embed_documents(batch_texts)
#                 all_embeddings.extend(batch_embeddings)
#                 print(f"⚡ Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
#             # Quick storage
#             index = self.pinecone_client.Index(PINECONE_INDEX_NAME)
            
#             # Prepare vectors
#             vectors = []
#             for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
#                 vectors.append({
#                     "id": f"fast_{i}",
#                     "values": embedding,
#                     "metadata": {
#                         "text": chunk.page_content[:500],  # Limit metadata size
#                         "word_count": chunk.metadata.get('word_count', 0)
#                     }
#                 })
            
#             # Batch upsert
#             upsert_batch_size = 100
#             for i in range(0, len(vectors), upsert_batch_size):
#                 batch = vectors[i:i+upsert_batch_size]
#                 index.upsert(vectors=batch)
            
#             print("✅ Fallback storage completed")
            
#         except Exception as e:
#             print(f"Fallback storage error: {e}")

#     def run_ultra_fast_pipeline(self, directory_path: str = None):
#         """Ultra-fast pipeline - target under 30 seconds"""
#         start_time = time.time()
        
#         print("🚀 ULTRA-FAST INDEXING PIPELINE")
#         print("Target: Under 30 seconds total")
#         print("=" * 50)
        
#         try:
#             # Step 1: Fast document loading (5-8 seconds)
#             step_start = time.time()
#             source_dir = directory_path or SOURCE_DOCUMENTS_DIR
#             documents = self.fast_document_loading(source_dir)
            
#             if not documents:
#                 return {"error": "No documents found", "total_time": 0}
            
#             step_time = time.time() - step_start
#             print(f"⚡ Step 1 completed in {step_time:.2f}s")
            
#             # Step 2: Lightning chunking (2-3 seconds)
#             step_start = time.time()
#             chunks = self.lightning_chunking(documents)
#             step_time = time.time() - step_start
#             print(f"⚡ Step 2 completed in {step_time:.2f}s")
            
#             # Step 3: Turbo embedding and storage (15-20 seconds)
#             step_start = time.time()
#             self.turbo_embedding_and_storage(chunks)
#             step_time = time.time() - step_start
#             print(f"⚡ Step 3 completed in {step_time:.2f}s")
            
#             total_time = time.time() - start_time
            
#             result = {
#                 "total_documents": len(documents),
#                 "total_chunks": len(chunks),
#                 "total_time": round(total_time, 2),
#                 "target_achieved": total_time < 30,
#                 "status": "ultra_fast_success"
#             }
            
#             print(f"\n🎉 ULTRA-FAST PIPELINE COMPLETE!")
#             print(f"⚡ Total time: {total_time:.2f} seconds")
#             print(f"🎯 Target achieved: {'✅ YES' if result['target_achieved'] else '❌ NO'}")
#             print(f"📊 Processed: {len(documents)} docs → {len(chunks)} chunks")
            
#             return result
            
#         except Exception as e:
#             total_time = time.time() - start_time
#             print(f"❌ Error in ultra-fast pipeline: {e}")
#             return {
#                 "error": str(e),
#                 "total_time": round(total_time, 2),
#                 "status": "error"
#             }

# def main():
#     """Main ultra-fast execution"""
#     # Quick API key check
#     if not all([os.getenv("GOOGLE_API_KEY"), os.getenv("PINECONE_API_KEY")]):
#         raise ValueError("❌ API keys missing!")
    
#     pipeline = UltraFastIndexing()
#     result = pipeline.run_ultra_fast_pipeline()
    
#     return result

# if __name__ == "__main__":
#     result = main()
#     print(f"\n📈 Final result: {result}")

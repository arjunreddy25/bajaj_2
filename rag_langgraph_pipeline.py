#latest version

import os
import json
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from functools import lru_cache

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration & Initialization ---
PINECONE_INDEX_NAME = "bajaj-hackathon-index"

class OptimizedRAGSystem:
    def __init__(self):
        """Initialize the RAG system with optimized configurations."""
        # Use faster model with optimized settings
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite", 
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_output_tokens=2048,  # Reduced for faster response
            timeout=15  # Add timeout
        )
        
        # Initialize embedding model
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Initialize vector store
        self.vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME, 
            embedding=self.embedding_model
        )
        
        # Cache for retrieved documents to avoid redundant searches
        self.document_cache = {}
        
        # Pre-compiled prompts for efficiency
        self.setup_prompts()
    
    def setup_prompts(self):
        """Pre-compile prompts for better performance."""
        self.unified_prompt = ChatPromptTemplate.from_template(
            """You are an insurance policy expert. Answer the question directly based on the policy clauses provided.

Question: {query}

Policy Clauses:
{context}

Provide a direct, concise answer in 1-2 sentences. Include specific numbers, timeframes, and conditions mentioned in the clauses.""")
    
    @lru_cache(maxsize=50)
    def get_cached_documents(self, query_hash: str, query: str) -> List[str]:
        """Cache retrieved documents to avoid redundant API calls."""
        if query_hash in self.document_cache:
            return self.document_cache[query_hash]
        
        # Retrieve documents with reduced count for speed
        retrieved_docs = self.vectorstore.similarity_search(query, k=3)
        doc_texts = [doc.page_content for doc in retrieved_docs]
        
        # Cache the results
        self.document_cache[query_hash] = doc_texts
        return doc_texts
    
    def process_single_query(self, query: str) -> str:
        """Process a single query efficiently and return direct answer."""
        try:
            # Create a hash for caching
            query_hash = str(hash(query.lower().strip()))
            
            # Get relevant documents (cached if available)
            relevant_docs = self.get_cached_documents(query_hash, query)
            context = "\n\n".join(relevant_docs)
            
            # Single LLM call for direct answer
            result = self.llm.invoke(
                self.unified_prompt.format_messages(
                    query=query,
                    context=context
                )
            )
            
            # Extract clean answer
            answer = result.content.strip()
            return answer
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}"
    
    def process_multiple_queries_parallel(self, queries: List[str]) -> List[str]:
        """Process multiple queries in parallel for optimal performance."""
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=min(len(queries), 8)) as executor:
            # Submit all queries for parallel processing and maintain order
            futures = [executor.submit(self.process_single_query, query) for query in queries]
            
            results = []
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=20)  # Reduced timeout
                    results.append(result)
                    print(f"Completed query {i+1}/{len(queries)}")
                except Exception as e:
                    print(f"Error in query {i+1}: {str(e)}")
                    results.append(f"Error processing query: {str(e)}")
        
        return results
    
    def process_batch_optimized(self, queries: List[str]) -> Dict[str, Any]:
        """
        Optimized batch processing that returns the required format.
        """
        start_time = time.time()
        
        # Process all queries directly in parallel (no grouping for speed)
        print("Processing queries in parallel...")
        all_results = self.process_multiple_queries_parallel(queries)
        
        total_time = time.time() - start_time
        
        return {
            "answers": all_results,
            "processing_time": round(total_time, 2),
            "target_achieved": total_time < 30
        }
    
    def group_similar_queries(self, queries: List[str]) -> List[List[str]]:
        """Group similar queries to maximize cache efficiency."""
        # Simple grouping by keywords - can be enhanced with embedding similarity
        groups = []
        processed = set()
        
        for i, query in enumerate(queries):
            if i in processed:
                continue
                
            current_group = [query]
            processed.add(i)
            
            # Find similar queries (simple keyword-based approach)
            query_words = set(query.lower().split())
            
            for j, other_query in enumerate(queries[i+1:], i+1):
                if j in processed:
                    continue
                    
                other_words = set(other_query.lower().split())
                similarity = len(query_words.intersection(other_words)) / len(query_words.union(other_words))
                
                if similarity > 0.3:  # Threshold for grouping
                    current_group.append(other_query)
                    processed.add(j)
            
            groups.append(current_group)
        
        return groups


# --- Usage Functions ---

def process_sample_queries():
    """Process the sample queries provided and return in required format."""
    sample_queries = [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
    
    # Initialize the optimized RAG system
    rag_system = OptimizedRAGSystem()
    
    print("Starting Optimized RAG Processing...")
    print(f"Processing {len(sample_queries)} queries")
    print("Target: <30 seconds total")
    print("-" * 60)
    
    # Process all queries
    results = rag_system.process_batch_optimized(sample_queries)
    
    # Display results and print answers
    print(f"\nPROCESSING COMPLETE!")
    print(f"Total Time: {results['processing_time']} seconds")
    print(f"Target achieved: {results['target_achieved']}")
    
    print("\n" + "="*60)
    print("ANSWERS:")
    print("="*60)
    
    # for i, answer in enumerate(results["answers"], 1):
    #     print(f"{i}. {answer}")
    #     print("-" * 50)
    
    # # Return in required format
    return {
        "answers": results["answers"]
    }


def process_custom_queries(queries: List[str]) -> Dict[str, Any]:
    """Process custom list of queries and return in required format."""
    if not queries:
        raise ValueError("No queries provided")
    
    rag_system = OptimizedRAGSystem()
    results = rag_system.process_batch_optimized(queries)
    
    return {
        "answers": results["answers"]
    }


if __name__ == "__main__":
    # Ensure API keys are available
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        raise ValueError("API keys not found. Please create a .env file with GOOGLE_API_KEY and PINECONE_API_KEY.")
    
    # Process sample queries
    results = process_sample_queries()
    print(json.dumps(results, indent=2))
    
    # You can also process custom queries like this:
    # custom_queries = ["Your custom query here"]
    # results = process_custom_queries(custom_queries)
    # print(json.dumps(results, indent=2))







#multiple answers



import os
import json
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langgraph.graph import StateGraph, END

# --- Load Environment Variables ---
load_dotenv()

# --- Configuration & Initialization ---
PINECONE_INDEX_NAME = "bajaj-hackathon-index"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0, 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
vectorstore = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME, 
    embedding=embedding_model
)
# Use a retriever that returns scores
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold", 
    search_kwargs={'k': 7, 'score_threshold': 0.7}
)

# --- LangGraph State Definition ---
class RAGState(TypedDict):
    """
    Represents the state of our RAG workflow. It's passed between nodes.
    """
    original_query: str
    parsed_details: dict
    # We will now store documents along with their retrieval scores
    retrieved_documents: List[Dict[str, Any]] 
    final_decision: dict

# --- Node Functions for the Graph ---

def parse_query_node(state: RAGState):
    """Node 1: Parses the user's query into a structured dictionary."""
    print("--- (1) Parsing Query ---")
    query = state["original_query"]
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_template(
        "You are an expert at parsing insurance claim queries. "
        "Extract key details from the query into a JSON object. "
        "Query: '{query}'\n\n{format_instructions}"
    )
    chain = prompt | llm | parser
    parsed_details = chain.invoke({
        "query": query,
        "format_instructions": parser.get_format_instructions()
    })
    return {"parsed_details": parsed_details}

def retrieve_clauses_node(state: RAGState):
    """
    Node 2: Retrieves relevant policy clauses and their confidence scores.
    """
    print("--- (2) Retrieving Clauses with Scores ---")
    parsed_details = state["parsed_details"]
    search_query = json.dumps(parsed_details)
    
    # Use 'similarity_search_with_relevance_scores' to get a normalized score (0 to 1)
    # This method returns a list of (Document, score) tuples.
    scored_docs = vectorstore.similarity_search_with_relevance_scores(search_query, k=7)
    
    # Format the retrieved documents for easier use in the next steps
    retrieved_documents = [
        {"content": doc.page_content, "retrieval_score": round(score, 2)} 
        for doc, score in scored_docs
    ]
    
    # We can filter out low-confidence documents if needed
    retrieved_documents = [doc for doc in retrieved_documents if doc["retrieval_score"] > 0.7]
    print(f"Retrieved {len(retrieved_documents)} relevant documents.")
    
    # Print the retrieved documents
    print("\n" + "="*60)
    print("RETRIEVED DOCUMENTS:")
    print("="*60)
    for i, doc in enumerate(retrieved_documents, 1):
        print(f"\nDocument {i} (Score: {doc['retrieval_score']}):")
        print("-" * 40)
        print(doc['content'])
        print("-" * 40)
    
    return {"retrieved_documents": retrieved_documents}

def evaluate_decision_node(state: RAGState):
    """
    Node 3: The core reasoning engine. Makes a decision and evaluates its own confidence.
    """
    print("--- (3) Evaluating Decision with Confidence ---")
    parsed_details = state["parsed_details"]
    
    # Format the context to include the retrieval scores
    context_clauses = ""
    for doc in state["retrieved_documents"]:
        context_clauses += f"Clause (Retrieval Confidence: {doc['retrieval_score']}):\n{doc['content']}\n---\n"
    
    parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_template(
        """You are a senior insurance claim adjudicator. Your task is to make a decision based *strictly* on the provided policy clauses.

        **Claim Details:** 
        {details}

        **Relevant Policy Clauses (with retrieval confidence scores):**
        ---
        {context}
        ---

        **Your Task:**
        1.  Analyze the claim against the provided clauses. Pay more attention to clauses with higher retrieval confidence.
        2.  Determine the decision: "Approved", "Rejected", or "Needs More Information".
        3.  Determine the payout amount if applicable and calculable.
        4.  Provide a clear justification, mapping your decision to the specific clauses.
        5.  **Crucially, provide a 'decision_confidence' score (from 0.0 to 1.0) representing your certainty in the final decision based *only* on the provided information. 1.0 means absolute certainty, 0.5 means it's a toss-up.**
        
        **Return a JSON object with the specified structure:**
        {{
            "decision": "...",
            "decision_confidence": 0.9,
            "amount": "...",
            "justification": [
                {{
                    "clause": "The specific text of the clause used.",
                    "retrieval_score": 0.85,
                    "reasoning": "How this clause led to your decision."
                }}
            ]
        }}
        
        {format_instructions}
        """
    )
    
    chain = prompt | llm | parser
    decision_output = chain.invoke({
        "details": json.dumps(parsed_details),
        "context": context_clauses,
        "format_instructions": parser.get_format_instructions(),
    })
    
    return {"final_decision": decision_output}

# --- Graph Definition and Execution (Remains the same) ---

def build_and_run_graph(query: str):
    workflow = StateGraph(RAGState)
    workflow.add_node("parse_query", parse_query_node)
    workflow.add_node("retrieve_clauses", retrieve_clauses_node)
    workflow.add_node("evaluate_decision", evaluate_decision_node)
    workflow.set_entry_point("parse_query")
    workflow.add_edge("parse_query", "retrieve_clauses")
    workflow.add_edge("retrieve_clauses", "evaluate_decision")
    workflow.add_edge("evaluate_decision", END)
    app = workflow.compile()
    inputs = {"original_query": query}
    final_state = app.invoke(inputs)
    print("\n" + "="*50)
    print("✅ RAG PIPELINE COMPLETE. FINAL DECISION:")
    print("="*50)
    print(json.dumps(final_state['final_decision'], indent=2))
    
    # Also print the retrieved documents from the final state
    print("\n" + "="*50)
    print("📄 RETRIEVED DOCUMENTS (Final State):")
    print("="*50)
    for i, doc in enumerate(final_state['retrieved_documents'], 1):
        print(f"\nDocument {i} (Score: {doc['retrieval_score']}):")
        print("-" * 40)
        print(doc['content'])
        print("-" * 40)

if __name__ == "__main__":
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        raise ValueError("API keys not found. Please create a .env file with GOOGLE_API_KEY and PINECONE_API_KEY.")
    sample_query = "How does the policy define a 'Hospital'?"
    build_and_run_graph(sample_query)










#single answer




# import os
# import json
# from dotenv import load_dotenv
# from typing import TypedDict, List
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
# from langgraph.graph import StateGraph, END

# # --- Load Environment Variables ---
# load_dotenv()

# # --- Configuration & Initialization ---
# PINECONE_INDEX_NAME = "bajaj-hackathon-index"

# # Initialize the LLM for generation and parsing tasks
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash", 
#     temperature=0, 
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )

# # Initialize the embedding model for document retrieval
# embedding_model = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001", 
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )

# # Initialize the Pinecone vector store retriever
# vectorstore = PineconeVectorStore(
#     index_name=PINECONE_INDEX_NAME, 
#     embedding=embedding_model
# )
# retriever = vectorstore.as_retriever(search_kwargs={'k': 7}) # Retrieve top 5 relevant chunks

# # --- LangGraph State Definition ---
# class RAGState(TypedDict):
#     """
#     Represents the state of our RAG workflow. It's passed between nodes.
#     """
#     original_query: str
#     parsed_details: dict
#     retrieved_clauses: List[str]
#     final_decision: dict

# # --- Node Functions for the Graph ---

# def parse_query_node(state: RAGState):
#     """
#     Node 1: Parses the user's natural language query into a structured dictionary.
#     This structured data is easier for the subsequent nodes to work with.
#     """
#     print("--- (1) Parsing Query ---")
#     query = state["original_query"]
    
#     parser = JsonOutputParser()
#     prompt = ChatPromptTemplate.from_template(
#         "You are an expert at parsing insurance claim queries. "
#         "Extract key details from the query into a JSON object. "
#         "Query: '{query}'\n\n{format_instructions}"
#     )
#     chain = prompt | llm | parser
#     parsed_details = chain.invoke({
#         "query": query,
#         "format_instructions": parser.get_format_instructions()
#     })
#     return {"parsed_details": parsed_details}

# def retrieve_clauses_node(state: RAGState):
#     """
#     Node 2: Retrieves relevant policy clauses from Pinecone based on the query.
#     This is the "Retrieval" step of RAG.
#     """
#     print("--- (2) Retrieving Clauses ---")
#     parsed_details = state["parsed_details"]
    
#     # Create a descriptive query for better semantic search results
#     search_query = json.dumps(parsed_details)
    
#     retrieved_docs = retriever.invoke(search_query)
#     doc_texts = [doc.page_content for doc in retrieved_docs]
    
#     return {"retrieved_clauses": doc_texts}

# def evaluate_decision_node(state: RAGState):
#     """
#     Node 3: The core reasoning engine. It makes a decision based on the retrieved clauses.
#     This is the "Augmented Generation" step of RAG.
#     """
#     print("--- (3) Evaluating Decision ---")
#     parsed_details = state["parsed_details"]
#     context_clauses = "\n---\n".join(state["retrieved_clauses"])
    
#     parser = JsonOutputParser()
#     prompt = ChatPromptTemplate.from_template(
#         """You are a senior insurance claim adjudicator. Your task is to make a decision based *strictly* on the provided policy clauses. Do not use any external knowledge.

#         **Claim Details:** 
#         {details}

#         **Relevant Policy Clauses:**
#         ---
#         {context}
#         ---

#         **Your Task:**
#         1. Analyze the claim against the provided clauses.
#         2. Determine the decision: "Approved", "Rejected", or "Needs More Information".
#         3. Determine the payout amount if applicable and calculable from the clauses.
#         4. Provide a clear justification, mapping each part of your decision directly to the specific clause(s) that support it.

#         **Return a JSON object with the specified structure:**
#         {format_instructions}
#         """
#     )
    
#     chain = prompt | llm | parser
#     decision_output = chain.invoke({
#         "details": json.dumps(parsed_details),
#         "context": context_clauses,
#         "format_instructions": parser.get_format_instructions(),
#     })
    
#     return {"final_decision": decision_output}

# # --- Graph Definition and Execution ---

# def build_and_run_graph(query: str):
#     """
#     Defines the graph structure using LangGraph, compiles it, and runs the query.
#     """
#     # Define the state machine graph
#     workflow = StateGraph(RAGState)

#     # Add the nodes to the graph
#     workflow.add_node("parse_query", parse_query_node)
#     workflow.add_node("retrieve_clauses", retrieve_clauses_node)
#     workflow.add_node("evaluate_decision", evaluate_decision_node)

#     # Define the edges that connect the nodes, setting the workflow
#     workflow.set_entry_point("parse_query")
#     workflow.add_edge("parse_query", "retrieve_clauses")
#     workflow.add_edge("retrieve_clauses", "evaluate_decision")
#     workflow.add_edge("evaluate_decision", END) # The last node
    
#     # Compile the graph into a runnable application
#     app = workflow.compile()

#     # Run the query through the graph
#     inputs = {"original_query": query}
#     final_state = app.invoke(inputs)
    
#     # Print the final, structured result
#     print("\n" + "="*50)
#     print(" RAG PIPELINE COMPLETE. FINAL DECISION:")
#     print("="*50)
#     print(json.dumps(final_state['final_decision'], indent=2))

# if __name__ == "__main__":
#     # Ensure your API keys are available
#     if not os.getenv("GOOGLE_API_KEY") or not os.getenv("PINECONE_API_KEY"):
#         raise ValueError("API keys not found. Please create a .env file with GOOGLE_API_KEY and PINECONE_API_KEY.")

#     # --- Sample Query ---
#     # This query should be rejected based on the 24-month waiting period clause.
#     sample_query = "How does the policy define a 'Hospital'?"
#     build_and_run_graph(sample_query)
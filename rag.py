import os
import json
import time
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from sklearn.metrics.pairwise import cosine_similarity
from functools import lru_cache

load_dotenv()

PINECONE_INDEX_NAME = "bajaj-hackathon-index"

class AdaptiveRAGSystem:
    def __init__(self):
        """Initialize adaptive RAG system that learns from data"""
        print("🧠 Initializing Adaptive RAG System...")
        
        # Initialize models
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            max_output_tokens=1024,
            timeout=15
        )
        
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        self.vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME, 
            embedding=self.embedding_model
        )
        
        # Load learned patterns (NO HARDCODING)
        self.learned_patterns = self.load_learned_patterns()
        self.query_embeddings_cache = {}
        
        self.setup_adaptive_prompts()
        
        print("✅ Adaptive RAG System ready!")

    def load_learned_patterns(self) -> Dict:
        """Load patterns learned during indexing"""
        try:
            patterns_file = os.path.join(os.path.dirname(__file__), 'learned_patterns.pkl')
            if os.path.exists(patterns_file):
                with open(patterns_file, 'rb') as f:
                    patterns = pickle.load(f)
                    print(f"📚 Loaded {len(patterns.get('topics', {}))} topics and {len(patterns.get('semantic_clusters', {}))} clusters")
                    return patterns
        except Exception as e:
            print(f"Could not load learned patterns: {e}")
        
        print("⚠️ No learned patterns found - using adaptive mode")
        return {'topics': {}, 'semantic_clusters': {}, 'structural_patterns': []}

    def setup_adaptive_prompts(self):
        """Setup prompts that adapt to discovered content"""
        self.system_prompt = """You are an expert document analyst. Answer questions directly based on the provided content.

Guidelines:
1. Provide direct, accurate answers based only on the provided content
2. Include specific details, numbers, and conditions mentioned
3. If information is not in the provided content, state "This information is not available in the provided content"
4. Be concise but comprehensive in your response
"""

        self.unified_prompt = ChatPromptTemplate.from_template(
            """{system_prompt}

Question: {query}

Relevant Content:
{context}

Answer based on the content above:"""
        )

    def adaptive_query_expansion(self, query: str) -> List[str]:
        """Expand queries based on learned semantic patterns"""
        expanded_queries = [query]
        
        # Method 1: Use learned topic keywords
        topic_expansions = self.expand_using_topics(query)
        expanded_queries.extend(topic_expansions)
        
        # Method 2: Use learned cluster patterns
        cluster_expansions = self.expand_using_clusters(query)
        expanded_queries.extend(cluster_expansions)
        
        # Method 3: Semantic similarity based expansion
        semantic_expansions = self.expand_using_semantics(query)
        expanded_queries.extend(semantic_expansions)
        
        # Remove duplicates and limit
        unique_queries = list(dict.fromkeys(expanded_queries))
        return unique_queries[:5]

    def expand_using_topics(self, query: str) -> List[str]:
        """Expand query using discovered topics"""
        expansions = []
        topics = self.learned_patterns.get('topics', {})
        
        query_words = set(query.lower().split())
        
        for topic_id, topic_info in topics.items():
            topic_words = set(topic_info.get('words', []))
            
            # If query overlaps with topic words, use related words for expansion
            if query_words.intersection(topic_words):
                for word in topic_words:
                    if word not in query.lower():
                        expanded_query = f"{query} {word}"
                        expansions.append(expanded_query)
                        if len(expansions) >= 2:  # Limit expansions
                            break
        
        return expansions[:2]

    def expand_using_clusters(self, query: str) -> List[str]:
        """Expand query using discovered semantic clusters"""
        expansions = []
        clusters = self.learned_patterns.get('semantic_clusters', {})
        
        query_words = set(query.lower().split())
        
        for cluster_id, cluster_info in clusters.items():
            cluster_terms = set(cluster_info.get('characteristic_terms', []))
            
            # If query overlaps with cluster terms
            if query_words.intersection(cluster_terms):
                for term in cluster_terms:
                    if term not in query.lower() and len(term.split()) <= 2:
                        expanded_query = f"{query} {term}"
                        expansions.append(expanded_query)
                        if len(expansions) >= 2:
                            break
        
        return expansions[:2]

    def expand_using_semantics(self, query: str) -> List[str]:
        """Expand query using semantic transformations"""
        expansions = []
        
        # Learn common query patterns from the system
        transformations = [
            (r'what is', 'define'),
            (r'how much', 'amount of'),
            (r'when', 'time for'),
            (r'does.*cover', 'coverage for'),
            (r'what.*period', 'duration of')
        ]
        
        import re
        for pattern, replacement in transformations:
            if re.search(pattern, query.lower()):
                expanded = re.sub(pattern, replacement, query.lower())
                if expanded != query.lower():
                    expansions.append(expanded)
        
        return expansions[:2]

    def adaptive_search_strategy(self, query: str, k: int = 6) -> List[Dict]:
        """Adaptive search that uses learned patterns"""
        all_results = []
        
        # Step 1: Direct vector search
        vector_results = self.vector_search(query, k=k//2)
        all_results.extend(vector_results)
        
        # Step 2: Expanded query search
        expanded_queries = self.adaptive_query_expansion(query)
        for exp_query in expanded_queries[:2]:
            exp_results = self.vector_search(exp_query, k=2)
            all_results.extend(exp_results)
        
        # Step 3: Topic-based search
        topic_results = self.topic_based_search(query, k=2)
        all_results.extend(topic_results)
        
        # Step 4: Cluster-based search
        cluster_results = self.cluster_based_search(query, k=2)
        all_results.extend(cluster_results)
        
        # Step 5: Intelligent ranking and deduplication
        unique_results = self.intelligent_ranking(all_results, query)
        
        return unique_results[:k]

    def vector_search(self, query: str, k: int) -> List[Dict]:
        """Standard vector search"""
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return [{'content': doc.page_content, 'metadata': doc.metadata, 'search_type': 'vector'} 
                    for doc in docs]
        except Exception as e:
            print(f"Vector search error: {e}")
            return []

    def topic_based_search(self, query: str, k: int) -> List[Dict]:
        """Search based on discovered topics"""
        try:
            # Find most relevant topic for query
            relevant_topic = self.find_relevant_topic(query)
            
            if relevant_topic:
                # Search with topic filter
                docs = self.vectorstore.similarity_search(
                    query,
                    k=k,
                    filter={'discovered_topic': relevant_topic}
                )
                return [{'content': doc.page_content, 'metadata': doc.metadata, 'search_type': 'topic'} 
                        for doc in docs]
        except Exception as e:
            print(f"Topic search error: {e}")
        
        return []

    def cluster_based_search(self, query: str, k: int) -> List[Dict]:
        """Search based on discovered semantic clusters"""
        try:
            # Find most relevant cluster for query
            relevant_cluster = self.find_relevant_cluster(query)
            
            if relevant_cluster:
                # Search with cluster filter
                docs = self.vectorstore.similarity_search(
                    query,
                    k=k,
                    filter={'semantic_cluster': relevant_cluster}
                )
                return [{'content': doc.page_content, 'metadata': doc.metadata, 'search_type': 'cluster'} 
                        for doc in docs]
        except Exception as e:
            print(f"Cluster search error: {e}")
        
        return []

    def find_relevant_topic(self, query: str) -> Optional[str]:
        """Find most relevant topic for query"""
        topics = self.learned_patterns.get('topics', {})
        query_words = set(query.lower().split())
        
        best_topic = None
        best_overlap = 0
        
        for topic_id, topic_info in topics.items():
            topic_words = set(topic_info.get('words', []))
            overlap = len(query_words.intersection(topic_words))
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_topic = topic_id
        
        return best_topic if best_overlap > 0 else None

    def find_relevant_cluster(self, query: str) -> Optional[str]:
        """Find most relevant semantic cluster for query"""
        clusters = self.learned_patterns.get('semantic_clusters', {})
        query_words = set(query.lower().split())
        
        best_cluster = None
        best_overlap = 0
        
        for cluster_id, cluster_info in clusters.items():
            cluster_terms = set(cluster_info.get('characteristic_terms', []))
            overlap = len(query_words.intersection(cluster_terms))
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_cluster = cluster_id
        
        return best_cluster if best_overlap > 0 else None

    def intelligent_ranking(self, results: List[Dict], query: str) -> List[Dict]:
        """Rank results using multiple adaptive criteria"""
        if not results:
            return []
        
        # Remove duplicates
        seen_content = set()
        unique_results = []
        
        for result in results:
            content_hash = hash(result['content'][:200])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        # Score results adaptively
        query_words = set(query.lower().split())
        
        def adaptive_score(result):
            content = result['content'].lower()
            content_words = set(content.split())
            
            # Base similarity score
            word_overlap = len(query_words.intersection(content_words))
            word_score = word_overlap / max(len(query_words), 1)
            
            # Metadata-based scoring
            metadata = result.get('metadata', {})
            
            # Prefer content with numbers (often contains specific info)
            number_bonus = 0.1 if metadata.get('has_numbers', False) else 0
            
            # Prefer content from relevant topics/clusters
            topic_bonus = 0.2 if 'topic' in result.get('search_type', '') else 0
            cluster_bonus = 0.2 if 'cluster' in result.get('search_type', '') else 0
            
            # Prefer appropriate content length
            word_count = metadata.get('word_count', len(content.split()))
            length_score = 1.0 if 50 <= word_count <= 300 else 0.8
            
            total_score = word_score + number_bonus + topic_bonus + cluster_bonus
            return total_score * length_score
        
        # Sort by adaptive score
        unique_results.sort(key=adaptive_score, reverse=True)
        
        return unique_results

    @lru_cache(maxsize=100)
    def cached_adaptive_search(self, query_hash: str, query: str) -> List[str]:
        """Cache adaptive search results"""
        search_results = self.adaptive_search_strategy(query, k=5)
        return [result['content'] for result in search_results]

    def process_single_query_adaptive(self, query: str) -> str:
        """Process query with full adaptive system"""
        try:
            # Create cache key
            query_hash = str(hash(query.lower().strip()))
            
            # Get results using adaptive search
            relevant_docs = self.cached_adaptive_search(query_hash, query)
            
            if not relevant_docs:
                return "No relevant information found in the documents."
            
            # Create context
            context = "\n\n---\n\n".join(relevant_docs)
            
            # Generate response
            result = self.llm.invoke(
                self.unified_prompt.format_messages(
                    system_prompt=self.system_prompt,
                    query=query,
                    context=context
                )
            )
            
            return result.content.strip()
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            return f"Error processing query: {str(e)}"

    def process_multiple_queries_adaptive(self, queries: List[str]) -> List[str]:
        """Process multiple queries with adaptive system"""
        with ThreadPoolExecutor(max_workers=min(len(queries), 6)) as executor:
            futures = [executor.submit(self.process_single_query_adaptive, query) for query in queries]
            
            results = []
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=25)
                    results.append(result)
                    print(f"✅ Completed adaptive query {i+1}/{len(queries)}")
                except Exception as e:
                    print(f"❌ Error in query {i+1}: {str(e)}")
                    results.append(f"Error processing query: {str(e)}")
        
        return results

    def process_batch_adaptive(self, queries: List[str]) -> Dict[str, Any]:
        """Process batch with full adaptive system"""
        start_time = time.time()
        
        print(f"🤖 Processing {len(queries)} queries with Adaptive RAG")
        print("Using machine learning patterns discovered from YOUR data")
        print("-" * 60)
        
        results = self.process_multiple_queries_adaptive(queries)
        
        total_time = time.time() - start_time
        
        return {
            "answers": results,
            "processing_time": round(total_time, 2),
            "learned_topics": len(self.learned_patterns.get('topics', {})),
            "learned_clusters": len(self.learned_patterns.get('semantic_clusters', {})),
            "target_achieved": total_time < 25
        }

# Sample queries
SAMPLE_QUERIES = [
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

def process_sample_queries() -> Dict[str, Any]:
    """Process sample queries with adaptive system"""
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("PINECONE_API_KEY"):
        raise ValueError("API keys required")
    
    rag_system = AdaptiveRAGSystem()
    results = rag_system.process_batch_adaptive(SAMPLE_QUERIES)
    
    print(f"\n{'='*60}")
    print("🎯 ADAPTIVE RAG PROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"⏱️  Total Time: {results['processing_time']} seconds")
    print(f"🧠 Learned Topics: {results['learned_topics']}")
    print(f"🔗 Learned Clusters: {results['learned_clusters']}")
    print(f"🎯 Target (<25s): {'✅ ACHIEVED' if results['target_achieved'] else '❌ MISSED'}")
    
    return {"answers": results["answers"]}

def process_custom_queries(queries: List[str]) -> Dict[str, Any]:
    """Process any custom queries adaptively"""
    if not queries:
        raise ValueError("No queries provided")
    
    rag_system = AdaptiveRAGSystem()
    results = rag_system.process_batch_adaptive(queries)
    
    return {"answers": results["answers"]}

if __name__ == "__main__":
    results = process_sample_queries()
    
    print(f"\n{'='*60}")
    print("📋 ADAPTIVE ANSWERS:")
    print(f"{'='*60}")
    
    for i, answer in enumerate(results["answers"], 1):
        print(f"\n{i}. {answer}")
        print("-" * 50)

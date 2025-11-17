import os 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List, Any 

class FaissSearch:
    """FAISS-based vector search implementation"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimension: int = 384):
            """
            Initialize FAISS search with embedding model
            
            Args:
                model_name: HuggingFace model name for embeddings
                dimension: Embedding dimension (384 for all-MiniLM-L6-v2)
            """
            self.model_name = model_name
            self.dimension = dimension
            
            # Initialize embedding model
            self.embed_model = HuggingFaceEmbedding(model_name=model_name)
            Settings.embed_model = self.embed_model
            
            # Initialize FAISS index (L2 distance)
            self.faiss_index = faiss.IndexFlatL2(dimension)
            
            # Create FAISS vector store
            self.vector_store = FaissVectorStore(faiss_index=self.faiss_index)
            
            # Storage context
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # Index will be created after ingesting documents
            self.index = None
            
            # Store document chunks for reference
            self.documents = []
        
    def load_document(self, file_path: str) -> List[Document]:
        """
        Load and parse document from file
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of Document objects
        """
        print(f"Loading document from: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split into chunks (simple paragraph-based splitting)
        chunks = self._chunk_text(text, chunk_size=500, overlap=50)
        
        # Create Document objects
        documents = [
            Document(text=chunk, metadata={"source": file_path, "chunk_id": i})
            for i, chunk in enumerate(chunks)
        ]
        
        print(f"Created {len(documents)} document chunks")
        self.documents = documents
        
        return documents
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text
            chunk_size: Maximum characters per chunk
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        # Convert chunk_size to approximate word count
        avg_word_length = 5
        words_per_chunk = chunk_size // avg_word_length
        overlap_words = overlap // avg_word_length
        
        for i in range(0, len(words), words_per_chunk - overlap_words):
            chunk = ' '.join(words[i:i + words_per_chunk])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def build_index(self, documents: List[Document] = None):
        """
        Build FAISS index from documents
        
        Args:
            documents: List of Document objects (uses self.documents if None)
        """
        if documents is None:
            documents = self.documents
            
        if not documents:
            raise ValueError("No documents to index. Load documents first.")
        
        print(f"Building FAISS index with {len(documents)} documents...")
        
        # Create index from documents
        self.index = VectorStoreIndex.from_documents(
            documents,
            storage_context=self.storage_context,
            show_progress=True
        )
        
        print(f"Index built successfully. Total vectors: {self.faiss_index.ntotal}")
        
    def query(self, query_text: str, top_k: int = 5):
        """
        Query the FAISS index
        
        Args:
            query_text: Query string
            top_k: Number of top results to return
            
        Returns:
            List of results with text and metadata
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        print(f"\nQuerying: '{query_text}'")
        
        # # Create query engine
        # query_engine = self.index.as_query_engine(similarity_top_k=top_k)
        
        # # Execute query
        # response = query_engine.query(query_text)

        # just retrieve chunks - no LLM required 
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query_text)
        
        # Extract results
        results = []
        for node in nodes:
            results.append({
                "text": node.node.text,
                "score": node.score,
                "metadata": node.node.metadata
            })
        
        return results
    
    def save_index(self, persist_dir: str = "./faiss_storage"):
        """
        Save FAISS index to disk
        
        Args:
            persist_dir: Directory to save index
        """
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        
        print(f"Saving index to {persist_dir}...")
        self.index.storage_context.persist(persist_dir=persist_dir)
        print("Index saved successfully.")
        
    def load_index(self, persist_dir: str = "./faiss_storage"):
        """
        Load FAISS index from disk
        
        Args:
            persist_dir: Directory containing saved index
        """
        print(f"Loading index from {persist_dir}...")
        
        # Rebuild storage context with vector store
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store,
            persist_dir=persist_dir
        )
        
        # Load index
        self.index = load_index_from_storage(storage_context=storage_context)
        print("Index loaded successfully.")
        
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for a text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as numpy array
        """
        embedding = self.embed_model.get_text_embedding(text)
        return np.array(embedding)
    
    def similarity_search(self, query_text: str, k: int = 5) -> List[tuple]:
        """
        Perform raw FAISS similarity search
        
        Args:
            query_text: Query string
            k: Number of results
            
        Returns:
            List of (distance, document_index) tuples
        """
        # Get query embedding
        query_embedding = self.get_embedding(query_text)
        query_vector = np.array([query_embedding]).astype('float32')
        
        # Search FAISS index
        distances, indices = self.faiss_index.search(query_vector, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                results.append({
                    "distance": float(dist),
                    "document": self.documents[idx].text,
                    "metadata": self.documents[idx].metadata
                })
        
        return results

def main():
    """Local FAISS implementation of vector store"""
    faiss_instance = FaissSearch(
        model_name="all-MiniLM-L6-v2",  # 384 dimensions
        dimension=384
    )

    sample_file = "mock_data.txt"

    # Load and process document
    documents = faiss_instance.load_document(sample_file)

    # Build FAISS index
    faiss_instance.build_index(documents)

    queries = [
        "Who is Bob Cratchet?"
    ]

    print("\n" + "="*80)
    print("QUERYING THE INDEX")
    print("="*80)

    for query in queries:
        results = faiss_instance.query(query, top_k=3)
        
        print(f"\nQuery: {query}")
        print("\nTop matching chunks:")
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.4f}")
            print(f"   Text: {result['text'][:200]}...")
            print(f"   Metadata: {result['metadata']}")
    
    print("\n" + "="*80)
    print("SAVING INDEX")
    print("="*80)
    faiss_instance.save_index("./faiss_storage")
    print("Index saved successfully!")

if __name__ == "__main__":
    main()


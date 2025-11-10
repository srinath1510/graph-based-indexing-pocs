import os 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as numpy
import faiss
from sentence_transformers import SentenceTransformer
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from typing import List, Any 

def main():
    """Local FAISS implementation of vector store"""
    faiss_instance = FaissSearch(
        model_name = "all-MiniLM-L6-v2" # 384 dimensions
    )

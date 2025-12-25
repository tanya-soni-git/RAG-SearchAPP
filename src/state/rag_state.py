"""RAG state definition for LangGraph"""

from typing import List
import uuid
from pydantic import BaseModel
#from langchain.schema import Document
from langchain_core.documents import Document
class RAGState(BaseModel):
    """State object for RAG workflow"""
    
    question: str
    retrieved_docs: List[Document] = []
    answer: str = ""
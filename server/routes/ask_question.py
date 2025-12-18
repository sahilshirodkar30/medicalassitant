from fastapi import APIRouter, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import Field
import os


from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


from modules.llm import get_llm_chain
from modules.query_handlers import query_chain
from logger import logger


router = APIRouter()


# -----------------------------
# GLOBAL SINGLETONS (CRITICAL)
# -----------------------------
_embed_model: SentenceTransformer | None = None
_pinecone_index = None




def get_vector_resources():
"""Lazy-load heavy resources (Render-safe)"""
global _embed_model, _pinecone_index


if _embed_model is None:
logger.info("Loading SentenceTransformer model")
_embed_model = SentenceTransformer("all-mpnet-base-v2")


if _pinecone_index is None:
logger.info("Connecting to Pinecone")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
_pinecone_index = pc.Index(os.environ["PINECONE_INDEX_NAME"])


return _embed_model, _pinecone_index




# -----------------------------
# Simple Retriever (DEFINED ONCE)
# -----------------------------
class SimpleRetriever(BaseRetriever):
tags: Optional[List[str]] = Field(default_factory=list)
metadata: Optional[dict] = Field(default_factory=dict)


def __init__(self, documents: List[Document]):
super().__init__()
self._docs = documents


def _get_relevant_documents(self, query: str) -> List[Document]:
return self._docs




# -----------------------------
# ASK ENDPOINT

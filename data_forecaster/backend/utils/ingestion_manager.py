import os
import pandas as pd
from typing import Tuple, List
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader
)
from langchain_core.documents import Document
from core.logging_config import get_logger

logger = get_logger(__name__)

def load_file_to_dataframe(file_path: str) -> pd.DataFrame:
    """
    Standardizes the ingestion of CSV and Excel files for the forecasting pipeline.
    """
    ext = os.path.splitext(file_path)[1].lower().lstrip('.')
    try:
        if ext == 'csv':
            return pd.read_csv(file_path)
        elif ext in ['xls', 'xlsx']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format for forecasting: {ext}")
    except Exception as e:
        logger.error(f"Failed to load dataframe from {file_path}: {e}")
        raise

def load_document_to_rag(file_path: str) -> List[Document]:
    """
    Parses various enterprise formats into LangChain Documents for the RAG Knowledge Base.
    """
    ext = os.path.splitext(file_path)[1].lower().lstrip('.')
    try:
        if ext == 'pdf':
            loader = PyPDFLoader(file_path)
        elif ext == 'txt':
            loader = TextLoader(file_path)
        elif ext == 'csv':
            loader = CSVLoader(file_path)
        elif ext in ['xls', 'xlsx']:
            loader = UnstructuredExcelLoader(file_path)
        else:
            logger.warning(f"Unsupported format {ext} for RAG. Falling back to text.")
            loader = TextLoader(file_path)
        
        return loader.load()
    except Exception as e:
        logger.error(f"Failed to process document {file_path} for RAG: {e}")
        return []

def batch_ingest_directory(directory_path: str) -> List[Document]:
    """
    Iterates through a directory to ingest all enterprise documents.
    """
    import os
    all_docs = []
    for filename in os.listdir(directory_path):
        f_path = os.path.join(directory_path, filename)
        if os.path.isfile(f_path):
            docs = load_document_to_rag(f_path)
            all_docs.extend(docs)
    logger.info(f"Batch ingestion complete: {len(all_docs)} document segments processed.")
    return all_docs
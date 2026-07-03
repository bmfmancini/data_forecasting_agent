"""Chat service for the Data Forecaster backend.

Contains :func:`chat_with_data` and :func:`chat_general` for the
data-explorer chat functionality.  Extracted from ``orchestrator.py``
to separate chat concerns from pipeline orchestration.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from core.llm_factory import get_llm
from core.logging_config import get_logger
from prompts.orchestrator_prompt import ORCHESTRATOR_CHAT_PROMPT
from services.rag_service import get_rag_kb

logger = get_logger(__name__)


def chat_with_data(
    query: str, df: pd.DataFrame, file_id: str, chroma_persist_dir: str
) -> dict[str, Any]:
    """Handle a data-explorer chat query about a specific uploaded dataset.

    Retrieves analysis memory from RAG, prepares a data summary, and
    invokes the LLM with the orchestrator chat prompt.

    Args:
        query:              The user's natural-language question.
        df:                 The uploaded DataFrame.
        file_id:            Identifier for the uploaded file.
        chroma_persist_dir: Path to the ChromaDB persistence directory.

    Returns:
        A dict with ``answer``, and optionally ``visualization_data``
        and ``visualization_type`` if the LLM produced a chart config.
    """
    logger.info("Data Explorer query for file_id=%s: %s", file_id, query)

    # 1. Retrieve Analysis Memory from RAG
    rag_kb = get_rag_kb(chroma_persist_dir)
    memory_context = ""
    try:
        chunks = rag_kb.retrieve(query, k=3)
        memory_context = "\n".join(chunks)
    except Exception as exc:
        logger.warning("RAG retrieval failed for chat: %s", exc)

    # 2. Prepare Data Summary for the LLM
    stats_dict = df.describe().iloc[:, :20].to_dict()

    # ── Enhanced Data Visibility ─────────────────────────────────────────────
    data_snapshots = ""
    try:
        date_cols = df.select_dtypes(
            include=["datetime", "datetimetz"]
        ).columns.tolist()
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if date_cols and num_cols:
            main_date = date_cols[0]
            val_col = num_cols[0]

            top_5 = df.nlargest(5, val_col)[[main_date, val_col]].to_string(
                index=False
            )
            bottom_5 = df.nsmallest(5, val_col)[[main_date, val_col]].to_string(
                index=False
            )

            data_snapshots = (
                f"\nHIGHEST 5 RECORDS (Sorted by {val_col}):\n{top_5}\n"
            )
            data_snapshots += (
                f"\nLOWEST 5 RECORDS (Sorted by {val_col}):\n{bottom_5}\n"
            )
    except Exception as exc:
        logger.warning("Failed to generate data highlights for chat: %s", exc)

    data_summary = f"""
    Dataset Stats:
    - Rows: {len(df)}
    - Statistical Summary: {stats_dict}
    {data_snapshots}
    """

    # 3. Setup LLM based on configuration
    llm = get_llm(temperature=0)

    prompt = ORCHESTRATOR_CHAT_PROMPT

    try:
        chain = prompt | llm
        response = chain.invoke(
            {
                "analysis_context": memory_context,
                "data_summary": data_summary,
                "query": query,
            }
        )

        content = response.content

        # Handle potential JSON visualization responses from LLM
        if "visualization_type" in content:
            clean_content = re.sub(r"```json\s*|\s*```", "", content).strip()
            try:
                return json.loads(clean_content)
            except json.JSONDecodeError:
                pass

        # Try to parse any JSON visualization configuration from the response
        json_pattern = r"\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}"
        matches = re.findall(json_pattern, content, re.DOTALL)

        for match in matches:
            try:
                potential_config = json.loads(match)
                if isinstance(potential_config, dict) and (
                    "data" in potential_config
                    or "type" in potential_config
                    or "layout" in potential_config
                ):
                    clean_answer = content.replace(match, "").strip()
                    return {
                        "answer": clean_answer,
                        "visualization_data": potential_config,
                        "visualization_type": "dynamic",
                    }
            except json.JSONDecodeError:
                continue

        return {"answer": content}

    except Exception:
        logger.exception("LLM Chat failed")
        return {
            "answer": (
                "I encountered an error while processing your request. "
                "Please try again later."
            )
        }


def chat_general(query: str, chroma_persist_dir: str) -> dict[str, Any]:
    """Handle general questions using only the RAG knowledge base.

    No dataset is required — the LLM answers based on forecasting
    documentation stored in ChromaDB.

    Args:
        query:              The user's natural-language question.
        chroma_persist_dir: Path to the ChromaDB persistence directory.

    Returns:
        A dict with an ``answer`` key.
    """
    logger.info("General chat query: %s", query)

    # 1. Retrieve relevant information from RAG
    rag_kb = get_rag_kb(chroma_persist_dir)
    memory_context = ""
    try:
        chunks = rag_kb.retrieve(query, k=5)
        memory_context = "\n".join(chunks)
    except Exception as exc:
        logger.warning("RAG retrieval failed for general chat: %s", exc)

    # 2. Setup LLM based on configuration
    llm = get_llm(temperature=0)

    general_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert in time series forecasting and data analysis.
         Use the provided context to answer questions about time series forecasting concepts, methodologies, and best practices.
         If the context doesn't contain enough information to fully answer the question, provide the best answer you can based on your general knowledge.

         Context from documentation:
         {context}

         Answer the question in a clear, concise, and helpful manner.""",
            ),
            ("human", "{query}"),
        ]
    )

    try:
        chain = general_prompt | llm
        response = chain.invoke({"context": memory_context, "query": query})

        content = response.content
        return {"answer": content}

    except Exception:
        logger.exception("LLM General Chat failed")
        return {
            "answer": (
                "I encountered an error while processing your request. "
                "Please try again later."
            )
        }
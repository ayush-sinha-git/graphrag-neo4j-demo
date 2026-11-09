"""Corpus chunking, LLM graph extraction, Neo4j ingestion, and vector indexing."""

from __future__ import annotations

import hashlib
import os
from typing import Any

import httpx
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_neo4j import LLMGraphTransformer
except ImportError:
    from langchain_experimental.graph_transformers import LLMGraphTransformer

from dummy_data import CORPUS

VECTOR_INDEX_NAME = "graphrag_document_index"

# Corporate SSL-inspecting proxies (e.g. Cloudflare Gateway) present a
# self-signed cert that the openai SDK's bundled certifi roots don't trust,
# even though the OS keychain does and setting SSL_CERT_FILE is not reliably
# honored by the SDK's httpx client. Passing an explicit httpx.Client with
# verify=<combined bundle> is the dependable fix. See .certs/README (export
# command) for how combined_ca_bundle.pem was generated.
_CA_BUNDLE = os.path.join(os.path.dirname(__file__), ".certs", "combined_ca_bundle.pem")


def _http_client() -> httpx.Client | None:
    return httpx.Client(verify=_CA_BUNDLE) if os.path.exists(_CA_BUNDLE) else None


def build_llm(api_key: str) -> ChatOpenAI:
    """Create the configured OpenRouter chat model."""
    if not api_key.strip():
        raise ValueError("An OpenRouter API key is required.")
    return ChatOpenAI(
        model="stealth/crouton-alpha",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "GraphRAG Local Demo",
        },
        temperature=0,
        http_client=_http_client(),
    )


def _chunk_corpus() -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=950, chunk_overlap=120)
    chunks = splitter.create_documents(
        CORPUS,
        metadatas=[{"source": f"corpus-{i + 1}"} for i in range(len(CORPUS))],
    )
    for index, chunk in enumerate(chunks):
        digest = hashlib.sha256(f"{index}:{chunk.page_content}".encode("utf-8")).hexdigest()
        chunk.metadata["id"] = digest
    return chunks


def build_graph(
    *,
    uri: str,
    username: str,
    password: str,
    database: str,
    api_key: str,
    embeddings: Any,
) -> dict[str, int]:
    """Extract entities and relationships, then index matching source chunks."""
    chunks = _chunk_corpus()
    llm = build_llm(api_key)
    graph = Neo4jGraph(url=uri, username=username, password=password, database=database)
    try:
        transformer = LLMGraphTransformer(llm=llm)
        graph_documents = transformer.convert_to_graph_documents(chunks)

        # Preserve stable chunk IDs so vector hits and graph mentions share nodes.
        for graph_document, chunk in zip(graph_documents, chunks, strict=True):
            if graph_document.source is None:
                graph_document.source = chunk
            graph_document.source.metadata["id"] = chunk.metadata["id"]

        graph.add_graph_documents(graph_documents, include_source=True)
        Neo4jVector.from_documents(
            documents=chunks,
            embedding=embeddings,
            url=uri,
            username=username,
            password=password,
            database=database,
            index_name=VECTOR_INDEX_NAME,
            ids=[chunk.metadata["id"] for chunk in chunks],
            node_label="Document",
            embedding_node_property="embedding",
            text_node_property="text",
        )

        _ensure_document_mentions(graph, graph_documents, chunks)
        return {"chunks": len(chunks), "graph_documents": len(graph_documents)}
    finally:
        graph.close()


def _ensure_document_mentions(
    graph: Neo4jGraph, graph_documents: list[Any], chunks: list[Document]
) -> None:
    """Create canonical Document nodes and connect extracted entities to them.

    LLMGraphTransformer gives each extracted node an `id` that may just be the
    entity's display name (not a separate name property), so matching tries both
    entity.id and entity.name against the same key list to catch either shape.
    """
    for graph_document, chunk in zip(graph_documents, chunks, strict=True):
        entity_ids = [str(node.id) for node in graph_document.nodes if node.id is not None]
        entity_names = [
            str(node.properties.get("name"))
            for node in graph_document.nodes
            if node.properties and node.properties.get("name")
        ]
        graph.query(
            """
            MERGE (doc:Document {id: $id})
            SET doc.text = $text, doc.source = $source
            WITH doc
            UNWIND $entity_ids + $entity_names AS entity_key
            MATCH (entity)
            WHERE entity <> doc
              AND (toString(entity.id) = entity_key OR toString(entity.name) = entity_key)
              AND NOT entity:Document
            MERGE (doc)-[:MENTIONS]->(entity)
            """,
            {
                "id": chunk.metadata["id"],
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", ""),
                "entity_ids": entity_ids,
                "entity_names": entity_names,
            },
        )


def get_embeddings(api_key: str) -> Any:
    """Load the cloud-hosted NVIDIA Nemotron embedding model via API."""
    return OpenAIEmbeddings(
        model="nvidia/nemotron-3.5-embed-1b:free",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=api_key,
        # Disable tiktoken-based length checks — OpenRouter validates context length
        # server-side, and tiktoken cannot download its encoding file through the
        # GS corporate SSL inspection proxy.
        tiktoken_enabled=False,
        check_embedding_ctx_length=False,
        http_client=_http_client(),
        # OpenRouter's free Nemotron embedding backend 502s ("service inference
        # failed") on any multi-input batch; it only accepts one text per request.
        chunk_size=1,
    )

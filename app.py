"""Streamlit interface for the local Neo4j GraphRAG demo."""

from __future__ import annotations

import os
import uuid

_project_dir = os.path.dirname(__file__)

# Must be set before any langchain/tiktoken import so tiktoken reads from the
# local cache instead of hitting openaipublic.blob.core.windows.net (blocked
# by corporate SSL inspection on this network).
_tiktoken_cache = os.path.join(_project_dir, ".tiktoken_cache")
os.makedirs(_tiktoken_cache, exist_ok=True)
os.environ.setdefault("TIKTOKEN_CACHE_DIR", _tiktoken_cache)

# The corporate network's SSL-inspecting proxy (Cloudflare Gateway) presents a
# self-signed cert that Python's bundled certifi roots don't trust, even though
# the OS keychain does. Point every HTTP client (httpx, requests, urllib3) at a
# combined bundle = certifi roots + the corporate gateway CA, generated via:
#   security find-certificate -a -c "Gateway CA - Cloudflare Managed G1" \
#     -p /Library/Keychains/System.keychain > .certs/cloudflare_gateway_ca.pem
_ca_bundle = os.path.join(_project_dir, ".certs", "combined_ca_bundle.pem")
if os.path.exists(_ca_bundle):
    os.environ.setdefault("SSL_CERT_FILE", _ca_bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _ca_bundle)
    os.environ.setdefault("CURL_CA_BUNDLE", _ca_bundle)

import streamlit as st
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_neo4j import Neo4jGraph

from graph_builder import build_graph, build_llm, get_embeddings
from workflow import create_workflow

load_dotenv()

STRATEGIES = {
    "Hybrid (Vector + Graph Traversal)": "hybrid",
    "Text2Cypher": "text2cypher",
    "Entity-to-Document Linking": "entity_to_document",
}

st.set_page_config(page_title="GraphRAG + Neo4j", page_icon="🕸️", layout="wide")
st.title("GraphRAG Local Demo")
st.caption("Explore a synthetic corporate intelligence graph with three retrieval strategies.")

defaults = {
    "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "neo4j_username": os.getenv("NEO4J_USERNAME", "neo4j"),
    "neo4j_password": os.getenv("NEO4J_PASSWORD", "password123"),
    "neo4j_database": os.getenv("NEO4J_DATABASE", "neo4j"),
    "openrouter_api_key": os.getenv("OPENROUTER_API_KEY", ""),
}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("thread_id", str(uuid.uuid4()))
st.session_state.setdefault("last_result", None)
st.session_state.setdefault("last_question", "")
st.session_state.setdefault("runtime", None)

with st.sidebar:
    st.header("Connections")
    with st.form("connection_form"):
        uri = st.text_input("NEO4J_URI", value=st.session_state.neo4j_uri)
        username = st.text_input("NEO4J_USERNAME", value=st.session_state.neo4j_username)
        password = st.text_input("NEO4J_PASSWORD", value=st.session_state.neo4j_password, type="password")
        database = st.text_input("NEO4J_DATABASE", value=st.session_state.neo4j_database)
        api_key = st.text_input(
            "OPENROUTER_API_KEY",
            value=st.session_state.openrouter_api_key,
            type="password",
        )
        save_connection = st.form_submit_button("Save connection settings")
    if save_connection:
        st.session_state.neo4j_uri = uri.strip()
        st.session_state.neo4j_username = username.strip()
        st.session_state.neo4j_password = password
        st.session_state.neo4j_database = database.strip()
        st.session_state.openrouter_api_key = api_key.strip()
        st.success("Connection settings saved for this Streamlit session.")

    selected_label = st.selectbox("Retrieval strategy", list(STRATEGIES))
    populate = st.button("Populate DB & Build Graph", use_container_width=True)
    reset = st.button("Reset / Clear Database", use_container_width=True)
    st.caption("Local Neo4j: http://localhost:7474")

connection = {
    "uri": st.session_state.neo4j_uri,
    "username": st.session_state.neo4j_username,
    "password": st.session_state.neo4j_password,
    "database": st.session_state.neo4j_database,
}

if populate:
    try:
        with st.spinner("Extracting entities, building graph, and indexing chunks..."):
            embeddings = get_embeddings(st.session_state.openrouter_api_key)
            stats = build_graph(
                **connection,
                api_key=st.session_state.openrouter_api_key,
                embeddings=embeddings,
            )
        st.success(
            f"Graph built: {stats['chunks']} document chunks and "
            f"{stats['graph_documents']} extracted graph documents."
        )
    except Exception as error:
        st.error(f"Could not build the graph: {type(error).__name__}: {error}")

if reset:
    try:
        driver = GraphDatabase.driver(
            st.session_state.neo4j_uri,
            auth=(st.session_state.neo4j_username, st.session_state.neo4j_password),
        )
        try:
            with driver.session(database=st.session_state.neo4j_database) as session:
                session.execute_write(lambda tx: tx.run("MATCH (node) DETACH DELETE node").consume())
        finally:
            driver.close()
        st.session_state.last_result = None
        st.success("All Neo4j nodes and relationships were removed. Index definitions are retained.")
    except Exception as error:
        st.error(f"Could not clear the database: {type(error).__name__}: {error}")

question = st.chat_input("Ask about companies, people, products, deals, or regulatory inquiries")
if question:
    st.session_state.last_question = question
    st.session_state.last_result = None
    try:
        with st.spinner("Retrieving evidence and generating an answer..."):
            # Rebuild the driver/graph/workflow only when connection settings actually
            # change, so each question reuses the same Neo4j connection and LangGraph
            # checkpointer instead of reconnecting on every chat turn.
            signature = (
                st.session_state.neo4j_uri,
                st.session_state.neo4j_username,
                st.session_state.neo4j_password,
                st.session_state.neo4j_database,
                st.session_state.openrouter_api_key,
            )
            runtime = st.session_state.runtime
            if runtime is None or runtime["signature"] != signature:
                if runtime is not None:
                    runtime["graph"].close()
                    runtime["driver"].close()
                driver = GraphDatabase.driver(
                    st.session_state.neo4j_uri,
                    auth=(st.session_state.neo4j_username, st.session_state.neo4j_password),
                )
                graph = None
                try:
                    graph = Neo4jGraph(
                        url=st.session_state.neo4j_uri,
                        username=st.session_state.neo4j_username,
                        password=st.session_state.neo4j_password,
                        database=st.session_state.neo4j_database,
                    )
                    llm = build_llm(st.session_state.openrouter_api_key)
                    runtime = {
                        "signature": signature,
                        "driver": driver,
                        "graph": graph,
                        "workflow": create_workflow(
                            driver=driver,
                            database=st.session_state.neo4j_database,
                            graph=graph,
                            embeddings=get_embeddings(st.session_state.openrouter_api_key),
                            llm=llm,
                        ),
                    }
                    st.session_state.runtime = runtime
                except Exception:
                    if graph is not None:
                        graph.close()
                    driver.close()
                    raise
            result = runtime["workflow"].invoke(
                {
                    "user_query": question,
                    "selected_strategy": STRATEGIES[selected_label],
                    "retrieved_context": "",
                    "cypher_query": "",
                    "final_answer": "",
                    "error": "",
                    "trace": [],
                },
                config={"configurable": {"thread_id": st.session_state.thread_id}},
            )
            st.session_state.last_result = result
    except Exception as error:
        st.error(f"Request failed: {type(error).__name__}: {error}")

result = st.session_state.last_result
if result:
    if result.get("error"):
        st.error(result["error"])
    tabs = st.tabs(["Final Answer", "Retrieved Context & Cypher", "LangGraph Trace"])
    with tabs[0]:
        st.subheader(st.session_state.last_question)
        st.markdown(result.get("final_answer", "No answer was produced."))
    with tabs[1]:
        metadata = result.get("retrieval_metadata", {})
        if result.get("cypher_query"):
            st.markdown("**Cypher**")
            st.code(result["cypher_query"], language="cypher")
        st.markdown("**Retrieved context**")
        st.text(result.get("retrieved_context") or "No context was retrieved.")
        st.markdown("**Retrieval metadata**")
        st.json(metadata)
    with tabs[2]:
        st.json(result.get("trace", []))
        st.caption("Node timings are in milliseconds; retrieval and generation are traced separately.")
else:
    st.info("Save your connection settings, populate the database, then ask a question.")

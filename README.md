# GraphRAG + Neo4j Local Demo

A local GraphRAG (Graph Retrieval-Augmented Generation) demo that combines a Neo4j
knowledge graph with vector similarity search to answer questions over a synthetic
corporate-intelligence corpus. Built with LangChain, LangGraph, Neo4j, and Streamlit,
using OpenRouter as the LLM/embeddings provider.

It ships with three independent, swappable retrieval strategies so you can compare
how graph-aware retrieval differs from plain vector search and from LLM-generated
Cypher queries, all on the same data and the same questions.

## How it works

```
dummy_data.py (synthetic corpus)
       │ chunk + embed
       ▼
Neo4j vector index  ◄────────────────────────┐
       │ LLMGraphTransformer extracts          │
       │ entities & relationships              │
       ▼                                       │
Neo4j graph (entities + MENTIONS edges)        │
       │                                       │
       ▼                                       │
retrievers.py (3 strategies) ──────────────────┘
       │
       ▼
workflow.py (LangGraph state machine) → LLM answer
       │
       ▼
app.py (Streamlit UI)
```

1. **`dummy_data.py`** — a small synthetic corpus about a fictional corporate
   ecosystem (a company acquisition, a product rivalry, an FTC inquiry, an investor
   conflict of interest). Deliberately self-contained so the demo has no external
   data dependency.
2. **`graph_builder.py`** — chunks the corpus, extracts an entity/relationship graph
   via an LLM (`LLMGraphTransformer`), writes it to Neo4j, builds a vector index over
   the chunks, and links each chunk (`Document` node) to the entities it mentions via
   `MENTIONS` edges. That `MENTIONS` edge is what lets vector search "jump" into the
   graph.
3. **`retrievers.py`** — three independent retrieval strategies (see below), each
   returning `{context, metadata}` for the LLM to answer from.
4. **`workflow.py`** — a LangGraph state machine: `route → retrieve_<strategy> →
   generate_answer`, with per-node timing captured in a trace.
5. **`app.py`** — a Streamlit chat UI: connect to Neo4j, populate the graph, pick a
   strategy, ask questions, and inspect the retrieved context, generated Cypher, and
   LangGraph trace.

### Retrieval strategies

| Strategy | What it does |
|---|---|
| **Hybrid (Vector + Graph Traversal)** | Vector-search the question against chunk embeddings, then traverse 1–2 hops out from each hit's mentioned entities to pull in connected graph facts. |
| **Text2Cypher** | Asks the LLM to write a single read-only Cypher query against the live graph schema, validates it (see **Security note** below), runs it, and retries once with the DB error fed back if it fails. |
| **Entity-to-Document Linking** | Asks the LLM to extract key entities/keywords from the question, then finds `Document` chunks that `MENTIONS` a matching entity. |

## Prerequisites

- Python 3.11+ (3.14 confirmed working)
- Docker (for Neo4j)
- An [OpenRouter](https://openrouter.ai) API key

## Setup

```bash
# 1. Start Neo4j
docker compose up -d

# 2. Create and activate a virtualenv
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# edit .env and set OPENROUTER_API_KEY

# 5. Run the app
streamlit run app.py
```

Then, in the browser UI:
1. Confirm/save your connection settings in the sidebar.
2. Click **"Populate DB & Build Graph"** (one-time; re-run after "Reset" or corpus
   changes).
3. Pick a retrieval strategy and ask a question in the chat box.
4. Inspect the **Final Answer**, **Retrieved Context & Cypher**, and **LangGraph
   Trace** tabs to see what each strategy actually retrieved.

Neo4j Browser is available at http://localhost:7474 (user `neo4j`, password from
`docker-compose.yml` / `.env`).

## Models

- **Chat**: `stealth/crouton-alpha` via OpenRouter
- **Embeddings**: `nvidia/nemotron-3.5-embed-1b:free` via OpenRouter

Both are free-tier models and can be swapped for any OpenRouter chat/embedding
model in `graph_builder.py` (`build_llm`, `get_embeddings`). If you swap in a
different embedding model, check whether it supports batched (multi-input)
embedding requests — see the **Known quirks** section below.

## Known quirks / gotchas

These were discovered the hard way while getting this running behind a corporate
network, and are left as code comments at their exact location — this section is
just an index into them.

- **tiktoken download blocked** (`app.py`): `tiktoken` tries to download its
  `cl100k_base` encoding from `openaipublic.blob.core.windows.net` on first use.
  Behind an SSL-inspecting corporate proxy this fails with a cert verification
  error. Fixed by pointing `TIKTOKEN_CACHE_DIR` at a local, pre-populated cache
  (`.tiktoken_cache/`) before any LangChain import.
- **Corporate SSL inspection breaks the OpenAI SDK's own HTTP client**
  (`graph_builder.py`): a corporate SSL-inspecting proxy (e.g. a Cloudflare
  Gateway) presents a cert that Python's bundled `certifi` roots don't trust, even
  though the OS trusts it. Setting `SSL_CERT_FILE` is *not* reliably honored by
  the installed `openai`/`httpx` client version, so the fix passes an explicit
  `httpx.Client(verify=<combined CA bundle>)` into both `ChatOpenAI` and
  `OpenAIEmbeddings`. See `.certs/` below for how to generate that bundle on a
  machine that needs it.
- **Free embedding model rejects batched requests** (`graph_builder.py`):
  `nvidia/nemotron-3.5-embed-1b:free` on OpenRouter returns
  `502 service inference failed` for any multi-input embedding request — it only
  accepts one input string per call. `OpenAIEmbeddings` batches by default, so
  `get_embeddings` forces `chunk_size=1`. This made the root cause look like a
  dead/invalid model when it was actually a batching limitation.
- **Text2Cypher's query validator is a filter, not a sandbox** (`retrievers.py`):
  `_sanitize_cypher` blocks obvious mutating keywords and multi-statement queries
  as a practical guardrail, but it's a regex denylist, not true isolation. For
  anything beyond a local demo, also run the Neo4j connection as a read-only
  database user.

### If you're not behind a corporate SSL-inspecting proxy

You likely don't need the `.tiktoken_cache/` or `.certs/` workarounds at all —
`TIKTOKEN_CACHE_DIR`/`SSL_CERT_FILE` setup in `app.py` and the `http_client` wiring
in `graph_builder.py` are no-ops when `.certs/combined_ca_bundle.pem` doesn't
exist, and tiktoken will just download normally.

### Regenerating `.certs/combined_ca_bundle.pem`

This file is **not committed** (see `.gitignore`) because it's machine-specific.
If you hit SSL certificate errors on a new machine behind the same kind of proxy:

```bash
mkdir -p .certs

# 1. Find your proxy's root CA name in the OS keychain (macOS example):
security find-certificate -a /Library/Keychains/System.keychain | grep -i "labl"

# 2. Export it (replace the name with what you found):
security find-certificate -a -c "<your proxy CA name>" -p \
  /Library/Keychains/System.keychain > .certs/proxy_ca.pem

# 3. Combine it with certifi's default roots:
python3 -c "
import certifi, shutil
shutil.copyfile(certifi.where(), '.certs/combined_ca_bundle.pem')
with open('.certs/proxy_ca.pem') as gw, open('.certs/combined_ca_bundle.pem', 'a') as out:
    out.write('\n' + gw.read())
"
```

## Project layout

```
app.py               Streamlit UI and session/connection management
graph_builder.py     Corpus chunking, LLM graph extraction, Neo4j ingestion, vector indexing
retrievers.py        Hybrid, Text2Cypher, and entity-to-document retrieval strategies
workflow.py          LangGraph state machine wiring retrieval to answer generation
dummy_data.py        Synthetic corpus
docker-compose.yml   Local Neo4j 5 Community + APOC
requirements.txt     Python dependencies
.env.example         Template for required environment variables
```

## Security notes

- `.env` (real credentials) and `.certs/` (machine-specific CA material) are
  git-ignored — never commit them.
- The Text2Cypher strategy sends LLM-generated Cypher straight to the database;
  see the gotcha above before using this pattern against a non-throwaway database.
- This demo uses a hardcoded local dev Neo4j password by default
  (`docker-compose.yml`) — change it before exposing Neo4j beyond localhost.

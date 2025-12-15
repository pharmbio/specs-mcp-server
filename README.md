Specs MCP Server
================

An MCP server built with [FastMCP](https://pypi.org/project/fastmcp/) that serves similarity search and analytics over a compound dataset (`compound_aggregate_with_annotations.csv`). The server exposes tools for compound-to-compound and MOA-to-MOA similarity, gene co-occurrence, disease rollups, and related lookups.


Repository layout
-----------------
- `main.py` — MCP server definition and all registered tools.
- `data/compound_aggregate_with_annotations.csv` — PCA embeddings (150 dims) plus annotations per compound.
- `requirements.txt` — Python dependencies.
- `Dockerfile` — container recipe for running the server.


Prerequisites
-------------
- Python 3.10+ recommended.
- `pip` for dependency installation.
- ~40 MB free disk for the dataset CSV that ships with the repo.


Setup
-----
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


Running the MCP server locally
------------------------------
By default the server uses FastMCP's `streamable-http` transport on `0.0.0.0:8000`.

```bash
source .venv/bin/activate
python main.py
```

Environment overrides:
- `MCP_TRANSPORT` — `streamable-http` (default) or `http`.
- `HOST` — bind address (default `0.0.0.0`).
- `PORT` — listen port (default `8000`).

Example (explicit HTTP on localhost):
```bash
MCP_TRANSPORT=http HOST=127.0.0.1 PORT=8000 python main.py
```

After the server starts, point your MCP-compatible client at the URL it prints (e.g., `http://localhost:8000`).


Remote connection
-----------------
If you host this server remotely, point clients to the deployed base URL:
- Remote MCP endpoint: **<add-remote-url-here>**


Using from an MCP client
------------------------
Most MCP clients take a command plus transport config. Example JSON snippet for a local launch:
```json
{
  "mcpServers": {
    "specs-mcp": {
      "command": ["python", "main.py"],
      "transport": {
        "type": "streamable-http",
        "port": 8000
      }
    }
  }
}
```
Adjust the transport type/port to match your env vars or remote URL.


Available tools (high level)
----------------------------
- `query_compound_similarity` — nearest neighbors for a compound by PCA cosine similarity.
- `query_moa_similarity` — intra-MOA stats plus closest non-MOA compounds.
- `list_available_compounds` / `list_available_moas` — discovery helpers with optional filters.
- `get_compound_details`, `compare_two_compounds`, `compound_metadata` — metadata and pairwise comparison utilities.
- `search_by_target_gene`, `gene_compounds`, `gene_context`, `compound_targets` — gene-centric queries and overlaps.
- `compound_neighbors` — scoped neighbors by MOA, disease, shared genes, or all.
- `moa_summary`, `moa_relations`, `disease_summary`, `disease_relations_by_phase`, `phase_summary_for_moa`, `moa_summary_for_phase`, `disease_summary_for_moa`, `moa_summary_for_disease` — rollups and cross-group relations.
- `genes_for_group`, `top_viability_in_group` — gene and viability summaries for MOAs or phases.
- `list_compounds_by_moa`, `match_closest_compounds`, `match_closest_moa` — fuzzy matching and quick lists.
- `reload_dataset` — reloads the CSV into memory after you update it.

All tools operate on the CSV loaded at startup; call `reload_dataset` after replacing `compound_aggregate_with_annotations.csv` to refresh the in-memory cache.


Docker
------
Build and run with Docker:
```bash
docker build -t specs-mcp .
docker run --rm -p 8000:8000 specs-mcp
```
Override `MCP_TRANSPORT`, `HOST`, or `PORT` via `-e` flags if needed.

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

Remote connection
-----------------
If you host this server remotely, point clients to the deployed base URL:
- Remote MCP endpoint: **https://specs-mcp.serve.scilifelab.se/mcp**

Running the MCP server locally with Docker
------
Build and run with Docker:
```bash
docker build -t specs-mcp .
docker run --rm -p 8000:8000 specs-mcp
```

After the server starts, point your MCP-compatible client at the URL it prints (e.g., `http://localhost:8000`).


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




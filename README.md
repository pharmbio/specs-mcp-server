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


Available tools
---------------
- `query_compound_similarity` — global embedding neighbors for a compound.
- `compound_neighbors` — scoped embedding neighbors by MOA/disease/shared_genes (no global mode).
- `compound_metadata` — core fields plus optional disease-matched neighbors.
- `get_compound_details` — detailed metadata for one compound.
- `list_available_compounds` — list compounds with optional substring filter.
- `match_closest_compounds` — fuzzy-match free text to compound names.
- `compare_two_compounds` — pairwise similarity and shared annotations.
- `compound_targets` — gene-overlap neighbors for a compound.
- `gene_compounds` — compounds targeting a specific gene.
- `gene_context` — co-genes, MOAs, diseases for a gene.
- `search_by_target_gene` — quick gene-to-compound lookup.
- `query_moa_similarity` — intra-MOA stats and nearest compounds outside the MOA.
- `list_available_moas` — list MOAs with optional filters.
- `match_closest_moa` — fuzzy-match free text to MOA strings.
- `list_compounds_by_moa` — compounds within a MOA.
- `moa_summary` — MOA stats plus optional compound rows.
- `moa_relations` — nearest MOA centroids to a given MOA.
- `phase_summary_for_moa` — clinical phase distribution within an MOA.
- `disease_summary_for_moa` — disease distribution within an MOA.
- `moa_summary_for_phase` — MOA distribution within a clinical phase.
- `top_viability_in_group` — viability ranking within an MOA or phase.
- `disease_summary` — disease-level stats plus optional compounds.
- `disease_relations_by_phase` — top compounds in a phase and disease mix.
- `moa_summary_for_disease` — MOA distribution within a disease area.
- `genes_for_group` — top genes within an MOA or disease group.
- `compound_neighbors_shared_genes` — see `compound_neighbors` (scope shared_genes).
- `compound_neighbors_disease` — see `compound_neighbors` (scope disease).
- `compound_neighbors_moa` — see `compound_neighbors` (scope moa).
- `top_viability_in_group_moa` — see `top_viability_in_group` (group_type moa).
- `list_available_compounds_filtered` — see `list_available_compounds` with filter.
- `list_available_moas_filtered` — see `list_available_moas` with filter.
- `get_dataset_statistics` — dataset summary stats.
- `reload_dataset` — reload the CSV into memory after updates.

All tools operate on the CSV loaded at startup; call `reload_dataset` after replacing `compound_aggregate_with_annotations.csv` to refresh the in-memory cache.



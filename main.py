from __future__ import annotations

from difflib import SequenceMatcher
from dataclasses import dataclass, asdict
from typing import Optional
from collections import Counter
import os

import numpy as np
import pandas as pd
from fastmcp import FastMCP
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# DATA LOADING AND CONFIGURATION
# =============================================================================

DATASET_PATH = "data/compound_aggregate_with_annotations.csv"
PCA_COLUMNS = [f"X_pca_harmony_PC{i}" for i in range(1, 151)]
ANNOTATION_COLUMNS = [
    "batch_id",
    "compound_name",
    "iupac_name",
    "smiles",
    "mechanism_of_action",
    "target_genes",
    "disease_area",
    "clinical_phase",
    "compound_concentration",
    "cell_viability_pct",
    "grit_score",
]


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def get_embedding_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[PCA_COLUMNS].values


def compute_all_pairwise_similarities(df: pd.DataFrame) -> np.ndarray:
    embeddings = get_embedding_matrix(df)
    return cosine_similarity(embeddings)


# =============================================================================
# DATA CACHE HELPERS
# =============================================================================

_df_cache: Optional[pd.DataFrame] = None


def _get_df() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        _df_cache = load_dataset()
    return _df_cache


def _reload_df() -> pd.DataFrame:
    global _df_cache
    _df_cache = load_dataset()
    return _df_cache


# MCP server instance
mcp = FastMCP("specs-mcp")


# =============================================================================
# SECTION 1: SINGLE COMPOUND QUERY
# =============================================================================

@dataclass
class CompoundSimilarityResult:
    query_compound: dict
    similar_compounds: list[dict]
    query_found: bool
    error_message: Optional[str] = None


@mcp.tool()
def query_compound_similarity(
    compound_name: str,
    top_n: int = 20,
) -> dict:
    """
    Global embedding similarity: top-N compounds most similar to the given compound
    by PCA cosine similarity with **no filters or scoping**. Use this for all-compound
    nearest neighbors; use compound_neighbors for scoped neighbors.

    Args:
        compound_name: Partial or full compound name (case-insensitive contains match).
        top_n: Number of similar compounds to return, excluding the query compound itself.

    Returns:
        Dict with the matched query compound metadata, ranked list of similar compounds,
        and a query_found flag (or an error message if not found).
    """
    df = _get_df()

    names = df["compound_name"].fillna("").str.lower().str.strip()
    query_lower = compound_name.lower().strip()
    exact_mask = names == query_lower
    if exact_mask.any():
        matches = df[exact_mask]
    else:
        contains_mask = names.str.contains(query_lower, regex=False)
        matches = df[contains_mask]

    if len(matches) == 0:
        return asdict(
            CompoundSimilarityResult(
                query_compound={},
                similar_compounds=[],
                query_found=False,
                error_message=(
                    f"Compound '{compound_name}' not found in dataset. "
                    f"Try a different name or check spelling."
                ),
            )
        )

    query_idx = matches.index[0]
    query_row = df.loc[query_idx]
    query_compound_info = {
        "batch_id": query_row.get("batch_id"),
        "compound_name": query_row.get("compound_name"),
        "mechanism_of_action": query_row.get("mechanism_of_action"),
        "target_genes": query_row.get("target_genes"),
        "disease_area": query_row.get("disease_area"),
        "clinical_phase": query_row.get("clinical_phase"),
        "smiles": query_row.get("smiles"),
        "cell_viability_pct": round(float(query_row.get("cell_viability_pct")), 2)
        if pd.notna(query_row.get("cell_viability_pct"))
        else None,
        "grit_score": round(float(query_row.get("grit_score")), 4)
        if pd.notna(query_row.get("grit_score"))
        else None,
    }

    query_embedding = df.loc[query_idx, PCA_COLUMNS].values.reshape(1, -1)
    all_embeddings = get_embedding_matrix(df)
    similarities = cosine_similarity(query_embedding, all_embeddings)[0]

    similarity_df = pd.DataFrame({"index": range(len(similarities)), "similarity": similarities})
    similarity_df = similarity_df[similarity_df["index"] != query_idx]
    similarity_df = similarity_df.sort_values("similarity", ascending=False).head(top_n)

    query_genes = set()
    if pd.notna(query_row.get("target_genes")):
        query_genes = set(str(query_row["target_genes"]).split("|"))

    similar_compounds = []
    for _, sim_row in similarity_df.iterrows():
        idx = int(sim_row["index"])
        compound_row = df.iloc[idx]

        compound_genes = set()
        if pd.notna(compound_row.get("target_genes")):
            compound_genes = set(str(compound_row["target_genes"]).split("|"))
        shared_genes = list(query_genes & compound_genes)

        similar_compounds.append(
            {
                "rank": len(similar_compounds) + 1,
                "similarity_score": round(float(sim_row["similarity"]), 4),
                "batch_id": compound_row.get("batch_id"),
                "compound_name": compound_row.get("compound_name"),
                "mechanism_of_action": compound_row.get("mechanism_of_action"),
                "target_genes": compound_row.get("target_genes"),
                "shared_target_genes": shared_genes if shared_genes else None,
                "disease_area": compound_row.get("disease_area"),
                "clinical_phase": compound_row.get("clinical_phase"),
                "smiles": compound_row.get("smiles"),
                "cell_viability_pct": round(float(compound_row.get("cell_viability_pct")), 2)
                if pd.notna(compound_row.get("cell_viability_pct"))
                else None,
                "grit_score": round(float(compound_row.get("grit_score")), 4)
                if pd.notna(compound_row.get("grit_score"))
                else None,
            }
        )

    return asdict(
        CompoundSimilarityResult(
            query_compound=query_compound_info,
            similar_compounds=similar_compounds,
            query_found=True,
        )
    )


@mcp.tool()
def get_compound_details(
    compound_name: str,
) -> dict:
    """
    Look up detailed metadata for a single compound.

    Args:
        compound_name: Partial or full compound name (case-insensitive contains match).

    Returns:
        A dict of compound metadata fields or an error message if not found.
    """
    df = _get_df()

    mask = df["compound_name"].fillna("").str.lower().str.contains(
        compound_name.lower(), regex=False
    )
    matches = df[mask]

    if len(matches) == 0:
        return {"error": f"Compound '{compound_name}' not found"}

    row = matches.iloc[0]
    return {
        "batch_id": row.get("batch_id"),
        "compound_name": row.get("compound_name"),
        "iupac_name": row.get("iupac_name"),
        "smiles": row.get("smiles"),
        "mechanism_of_action": row.get("mechanism_of_action"),
        "target_genes": row.get("target_genes"),
        "disease_area": row.get("disease_area"),
        "clinical_phase": row.get("clinical_phase"),
        "cell_viability_pct": round(float(row.get("cell_viability_pct")), 2)
        if pd.notna(row.get("cell_viability_pct"))
        else None,
        "grit_score": round(float(row.get("grit_score")), 4)
        if pd.notna(row.get("grit_score"))
        else None,
    }


@mcp.tool()
def list_available_compounds(
    search_term: Optional[str] = None,
    limit: int = 50,
) -> list[str]:
    """
    List compound names in the dataset with optional substring filtering.

    Args:
        search_term: Optional case-insensitive substring to filter names.
        limit: Maximum number of names to return.

    Returns:
        Sorted list of compound names.
    """
    df = _get_df()

    compounds = df["compound_name"].dropna().unique()
    if search_term:
        compounds = [c for c in compounds if search_term.lower() in c.lower()]
    return sorted(compounds)[:limit]


# =============================================================================
# SECTION 2: MOA-BASED QUERY
# =============================================================================

@dataclass
class MOASimilarityResult:
    moa: str
    compounds_in_moa: list[dict]
    intra_class_statistics: dict
    intra_class_similarity_ranking: list[dict]
    similar_compounds_other_moas: list[dict]
    moa_found: bool
    error_message: Optional[str] = None


@mcp.tool()
def query_moa_similarity(
    moa: str,
    top_n_other_moas: int = 20,
) -> dict:
    """
    MOA-centric analysis: summarize intra-class similarity and return the closest
    compounds that sit **outside** the MOA (via centroid embedding similarity). Use
    this when the intent is “find near neighbors not in this MOA.”

    Args:
        moa: Partial or full MOA string to match (case-insensitive contains).
        top_n_other_moas: Number of nearest compounds outside the MOA to return.

    Returns:
        A dict with compounds in the MOA, intra-class similarity stats and ranking,
        and the top similar compounds from other MOAs (or an error message).
    """
    df = _get_df()

    mask = df["mechanism_of_action"].fillna("").str.lower().str.contains(
        moa.lower(), regex=False
    )
    moa_compounds = df[mask]

    if len(moa_compounds) == 0:
        return asdict(
            MOASimilarityResult(
                moa=moa,
                compounds_in_moa=[],
                intra_class_statistics={},
                intra_class_similarity_ranking=[],
                similar_compounds_other_moas=[],
                moa_found=False,
                error_message=(
                    f"MOA '{moa}' not found in dataset. "
                    f"Use list_available_moas() to see available options."
                ),
            )
        )

    matched_moa = moa_compounds["mechanism_of_action"].mode().iloc[0]
    compounds_in_moa = []
    for _, row in moa_compounds.iterrows():
        compounds_in_moa.append(
            {
                "batch_id": row.get("batch_id"),
                "compound_name": row.get("compound_name"),
                "target_genes": row.get("target_genes"),
                "disease_area": row.get("disease_area"),
                "clinical_phase": row.get("clinical_phase"),
                "cell_viability_pct": round(float(row.get("cell_viability_pct")), 2)
                if pd.notna(row.get("cell_viability_pct"))
                else None,
                "grit_score": round(float(row.get("grit_score")), 4)
                if pd.notna(row.get("grit_score"))
                else None,
            }
        )

    moa_indices = moa_compounds.index.tolist()
    moa_embeddings = moa_compounds[PCA_COLUMNS].values
    intra_similarities = []
    intra_class_ranking = []

    if len(moa_indices) > 1:
        intra_sim_matrix = cosine_similarity(moa_embeddings)
        for i in range(len(moa_indices)):
            for j in range(i + 1, len(moa_indices)):
                sim = intra_sim_matrix[i, j]
                intra_similarities.append(sim)
                intra_class_ranking.append(
                    {
                        "compound_1": moa_compounds.iloc[i]["compound_name"],
                        "compound_2": moa_compounds.iloc[j]["compound_name"],
                        "similarity_score": round(float(sim), 4),
                    }
                )

        intra_class_ranking.sort(key=lambda x: x["similarity_score"], reverse=True)
        intra_similarities_arr = np.array(intra_similarities)
        mean_sim = float(np.mean(intra_similarities_arr))
        std_sim = float(np.std(intra_similarities_arr))
        min_sim = float(np.min(intra_similarities_arr))
        max_sim = float(np.max(intra_similarities_arr))

        avg_sim_per_compound = []
        for i in range(len(moa_indices)):
            row_sims = [intra_sim_matrix[i, j] for j in range(len(moa_indices)) if i != j]
            avg_sim_per_compound.append(
                {
                    "compound_name": moa_compounds.iloc[i]["compound_name"],
                    "avg_similarity_to_class": round(float(np.mean(row_sims)), 4),
                }
            )

        avg_sim_per_compound.sort(key=lambda x: x["avg_similarity_to_class"])
        outlier_threshold = mean_sim - 1.5 * std_sim
        outliers = [
            c for c in avg_sim_per_compound if c["avg_similarity_to_class"] < outlier_threshold
        ]

        intra_class_statistics = {
            "num_compounds": len(moa_indices),
            "num_pairwise_comparisons": len(intra_similarities),
            "mean_similarity": round(mean_sim, 4),
            "std_similarity": round(std_sim, 4),
            "min_similarity": round(min_sim, 4),
            "max_similarity": round(max_sim, 4),
            "variance": round(float(np.var(intra_similarities_arr)), 4),
            "outlier_threshold": round(outlier_threshold, 4),
            "outliers": outliers,
            "per_compound_avg_similarity": avg_sim_per_compound,
        }
    else:
        intra_class_statistics = {
            "num_compounds": 1,
            "num_pairwise_comparisons": 0,
            "note": "Only one compound in this MOA class, cannot compute intra-class similarity",
        }

    # Compare against compounds whose MOA is different from the matched_moa (not substring-based)
    other_compounds = df[df["mechanism_of_action"] != matched_moa]
    if len(other_compounds) > 0 and len(moa_compounds) > 0:
        moa_centroid = moa_embeddings.mean(axis=0).reshape(1, -1)
        other_embeddings = other_compounds[PCA_COLUMNS].values

        centroid_similarities = cosine_similarity(moa_centroid, other_embeddings)[0]
        all_inter_sims = cosine_similarity(moa_embeddings, other_embeddings)
        max_sim_to_moa = all_inter_sims.max(axis=0)

        inter_results = []
        for idx, (_, row) in enumerate(other_compounds.iterrows()):
            inter_results.append(
                {
                    "batch_id": row.get("batch_id"),
                    "compound_name": row.get("compound_name"),
                    "mechanism_of_action": row.get("mechanism_of_action"),
                    "target_genes": row.get("target_genes"),
                    "disease_area": row.get("disease_area"),
                    "clinical_phase": row.get("clinical_phase"),
                    "cell_viability_pct": round(float(row.get("cell_viability_pct")), 2)
                    if pd.notna(row.get("cell_viability_pct"))
                    else None,
                    "grit_score": round(float(row.get("grit_score")), 4)
                    if pd.notna(row.get("grit_score"))
                    else None,
                    "similarity_to_moa_centroid": round(float(centroid_similarities[idx]), 4),
                    "max_similarity_to_moa_member": round(float(max_sim_to_moa[idx]), 4),
                }
            )

        inter_results.sort(key=lambda x: x["similarity_to_moa_centroid"], reverse=True)
        similar_compounds_other_moas = inter_results[:top_n_other_moas]
    else:
        similar_compounds_other_moas = []

    return asdict(
        MOASimilarityResult(
            moa=matched_moa,
            compounds_in_moa=compounds_in_moa,
            intra_class_statistics=intra_class_statistics,
            intra_class_similarity_ranking=intra_class_ranking,
            similar_compounds_other_moas=similar_compounds_other_moas,
            moa_found=True,
        )
    )


@mcp.tool()
def list_available_moas(
    search_term: Optional[str] = None,
    min_compounds: int = 1,
    limit: int = 50,
) -> list[dict]:
    """
    List MOAs present in the dataset with optional filters.

    Args:
        search_term: Optional case-insensitive substring filter.
        min_compounds: Minimum compound count required to include an MOA.
        limit: Maximum number of MOAs to return.

    Returns:
        List of dicts with MOA name and compound_count.
    """
    df = _get_df()

    moa_counts = df["mechanism_of_action"].value_counts()
    results = []
    for moa_value, count in moa_counts.items():
        if pd.isna(moa_value):
            continue
        if count < min_compounds:
            continue
        if search_term and search_term.lower() not in moa_value.lower():
            continue
        results.append({"moa": moa_value, "compound_count": int(count)})
    return results[:limit]


def get_moa_compound_list(
    moa: str,
    df: Optional[pd.DataFrame] = None,
) -> list[dict]:
    if df is None:
        df = _get_df()

    mask = df["mechanism_of_action"].fillna("").str.lower().str.contains(
        moa.lower(), regex=False
    )
    matches = df[mask]

    results = []
    for _, row in matches.iterrows():
        results.append(
            {
                "batch_id": row.get("batch_id"),
                "compound_name": row.get("compound_name"),
                "mechanism_of_action": row.get("mechanism_of_action"),
                "target_genes": row.get("target_genes"),
                "disease_area": row.get("disease_area"),
                "clinical_phase": row.get("clinical_phase"),
                "smiles": row.get("smiles"),
                "cell_viability_pct": round(float(row.get("cell_viability_pct")), 2)
                if pd.notna(row.get("cell_viability_pct"))
                else None,
                "grit_score": round(float(row.get("grit_score")), 4)
                if pd.notna(row.get("grit_score"))
                else None,
            }
        )
    return results


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@mcp.tool()
def get_dataset_statistics() -> dict:
    """
    Return high-level dataset counts and dimension info for the loaded CSV.

    Returns:
        A dict with totals (compounds, MOAs, disease areas), clinical phase distribution,
        and PCA dimension count.
    """
    df = _get_df()

    clinical_phase_counts = {
        phase: int(count) for phase, count in df["clinical_phase"].value_counts().items()
    }

    return {
        "total_compounds": int(len(df)),
        "unique_compound_names": int(df["compound_name"].nunique()),
        "unique_moas": int(df["mechanism_of_action"].nunique()),
        "compounds_with_moa": int(df["mechanism_of_action"].notna().sum()),
        "compounds_with_target_genes": int(df["target_genes"].notna().sum()),
        "unique_disease_areas": int(df["disease_area"].nunique()),
        "clinical_phase_distribution": clinical_phase_counts,
        "pca_dimensions": len(PCA_COLUMNS),
    }


@mcp.tool()
def search_by_target_gene(
    gene: str,
    top_n: int = 50,
) -> list[dict]:
    """
    Find compounds whose target_genes include the given gene symbol.

    Args:
        gene: Gene symbol (case-insensitive; compared as uppercase).
        top_n: Maximum number of compounds to return.

    Returns:
        List of compound records matching the gene.
    """
    df = _get_df()

    mask = df["target_genes"].fillna("").str.contains(gene.upper(), regex=False)
    matches = df[mask]

    results = []
    for _, row in matches.head(top_n).iterrows():
        results.append(
            {
                "batch_id": row.get("batch_id"),
                "compound_name": row.get("compound_name"),
                "mechanism_of_action": row.get("mechanism_of_action"),
                "target_genes": row.get("target_genes"),
                "disease_area": row.get("disease_area"),
                "clinical_phase": row.get("clinical_phase"),
                "cell_viability_pct": round(float(row.get("cell_viability_pct")), 2)
                if pd.notna(row.get("cell_viability_pct"))
                else None,
                "grit_score": round(float(row.get("grit_score")), 4)
                if pd.notna(row.get("grit_score"))
                else None,
            }
        )
    return results


@mcp.tool()
def compare_two_compounds(
    compound_1: str,
    compound_2: str,
) -> dict:
    """
    Compare two compounds by name and report cosine similarity plus shared annotations.

    Args:
        compound_1: First compound name (case-insensitive contains match).
        compound_2: Second compound name (case-insensitive contains match).

    Returns:
        A dict with metadata for both compounds, similarity score, shared genes,
        and boolean flags for same MOA/disease (or an error if not found).
    """
    df = _get_df()

    row1 = _find_compound_row(df, compound_1)
    row2 = _find_compound_row(df, compound_2)
    if row1 is None:
        return {"error": f"Compound '{compound_1}' not found"}
    if row2 is None:
        return {"error": f"Compound '{compound_2}' not found"}

    emb1 = row1[PCA_COLUMNS].values.reshape(1, -1)
    emb2 = row2[PCA_COLUMNS].values.reshape(1, -1)
    similarity = cosine_similarity(emb1, emb2)[0, 0]

    genes1 = set(str(row1["target_genes"]).split("|")) if pd.notna(row1["target_genes"]) else set()
    genes2 = set(str(row2["target_genes"]).split("|")) if pd.notna(row2["target_genes"]) else set()
    shared_genes = list(genes1 & genes2)

    return {
        "compound_1": {
            "name": row1["compound_name"],
            "moa": row1["mechanism_of_action"],
            "target_genes": row1["target_genes"],
            "disease_area": row1["disease_area"],
            "clinical_phase": row1["clinical_phase"],
            "cell_viability_pct": round(float(row1["cell_viability_pct"]), 2)
            if pd.notna(row1["cell_viability_pct"])
            else None,
            "grit_score": round(float(row1["grit_score"]), 4)
            if pd.notna(row1["grit_score"])
            else None,
        },
        "compound_2": {
            "name": row2["compound_name"],
            "moa": row2["mechanism_of_action"],
            "target_genes": row2["target_genes"],
            "disease_area": row2["disease_area"],
            "clinical_phase": row2["clinical_phase"],
            "cell_viability_pct": round(float(row2["cell_viability_pct"]), 2)
            if pd.notna(row2["cell_viability_pct"])
            else None,
            "grit_score": round(float(row2["grit_score"]), 4)
            if pd.notna(row2["grit_score"])
            else None,
        },
        "similarity_score": round(float(similarity), 4),
        "same_moa": row1["mechanism_of_action"] == row2["mechanism_of_action"],
        "shared_target_genes": shared_genes if shared_genes else None,
        "same_disease_area": row1["disease_area"] == row2["disease_area"],
    }


# =============================================================================
# FUZZY MATCH HELPERS
# =============================================================================

def _fuzzy_rank_names(query: str, candidates: list[str], top_n: int) -> list[dict]:
    """Return candidates ranked by similarity to the query."""
    q = query.strip().lower()
    scored = []
    for name in candidates:
        score = SequenceMatcher(None, q, name.lower()).ratio()
        scored.append({"name": name, "similarity_score": round(float(score), 4)})
    scored.sort(key=lambda x: x["similarity_score"], reverse=True)
    return scored[:top_n]


@mcp.tool()
def match_closest_compounds(
    query: str,
    top_n: int = 5,
    min_score: float = 0.6,
) -> dict:
    """
    Fuzzy-match a query string to compound names using SequenceMatcher similarity.

    Args:
        query: Free-text query to match against compound names.
        top_n: Number of candidates to return.
        min_score: Threshold to flag low-confidence matches.

    Returns:
        A dict with the best match, ranked candidates, and any low-confidence note.
    """
    df = _get_df()

    compound_names = sorted(df["compound_name"].dropna().unique().tolist())
    if not compound_names:
        return {"error": "No compound names available in dataset"}

    ranked = _fuzzy_rank_names(query, compound_names, top_n)
    best_match = ranked[0] if ranked else None
    result = {
        "query": query,
        "best_match": best_match,
        "candidates": ranked,
    }
    if not best_match:
        result["note"] = "No candidates available."
    elif best_match["similarity_score"] < min_score:
        result["note"] = (
            f"Low-confidence match: best score {best_match['similarity_score']} "
            f"is below threshold {min_score}"
        )
    return result


@mcp.tool()
def match_closest_moa(
    query: str,
    top_n: int = 5,
    min_score: float = 0.6,
) -> dict:
    """
    Fuzzy-match a query string to MOA values using SequenceMatcher similarity.

    Args:
        query: Free-text query to match against MOA strings.
        top_n: Number of candidates to return.
        min_score: Threshold to flag low-confidence matches.

    Returns:
        A dict with the best match, ranked candidates, and any low-confidence note.
    """
    df = _get_df()

    moa_names = sorted(df["mechanism_of_action"].dropna().unique().tolist())
    if not moa_names:
        return {"error": "No MOA values available in dataset"}

    ranked = _fuzzy_rank_names(query, moa_names, top_n)
    best_match = ranked[0] if ranked else None
    result = {
        "query": query,
        "best_match": best_match,
        "candidates": ranked,
    }
    if not best_match:
        result["note"] = "No candidates available."
    elif best_match["similarity_score"] < min_score:
        result["note"] = (
            f"Low-confidence match: best score {best_match['similarity_score']} "
            f"is below threshold {min_score}"
        )
    return result


@mcp.tool()
def list_compounds_by_moa(
    moa: str,
) -> dict:
    """
    List compounds that belong to a given MOA.

    Args:
        moa: Partial or full MOA string (case-insensitive contains match).

    Returns:
        Dict with the matched MOA and sorted list of compound names (or an error).
    """
    df = _get_df()

    mask = df["mechanism_of_action"].fillna("").str.lower().str.contains(
        moa.lower(), regex=False
    )
    matches = df[mask]
    if len(matches) == 0:
        return {"error": f"MOA '{moa}' not found"}

    compounds = matches["compound_name"].dropna().unique()
    return {
        "query": moa,
        "matched_moa": matches.iloc[0]["mechanism_of_action"],
        "compounds": sorted(compounds.tolist()),
    }


# =============================================================================
# RELATIONAL ANALYTICS HELPERS
# =============================================================================

def _split_genes(val: Optional[str]) -> list[str]:
    if pd.isna(val):
        return []
    if not isinstance(val, str):
        return []
    genes = [g.strip() for g in val.split("|") if g.strip()]
    # Normalize to upper-case to make comparisons consistent while preserving display order
    return [g.upper() for g in genes]


def _find_compound_row(df: pd.DataFrame, compound_name: str) -> Optional[pd.Series]:
    """Find a compound row preferring exact (case-insensitive) name match, else substring."""
    q = compound_name.strip().lower()
    names = df["compound_name"].fillna("").str.lower().str.strip()
    exact_mask = names == q
    if exact_mask.any():
        return df[exact_mask].iloc[0]

    contains_mask = names.str.contains(q, regex=False)
    matches = df[contains_mask]
    if len(matches) == 0:
        return None
    return matches.iloc[0]


def _viability_stats(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) == 0:
        return {}
    return {
        "min": round(float(s.min()), 2),
        "max": round(float(s.max()), 2),
        "mean": round(float(s.mean()), 2),
        "median": round(float(s.median()), 2),
    }


def _grit_stats(series: pd.Series) -> dict:
    s = series.dropna()
    if len(s) == 0:
        return {}
    return {
        "min": round(float(s.min()), 4),
        "max": round(float(s.max()), 4),
        "mean": round(float(s.mean()), 4),
        "median": round(float(s.median()), 4),
    }


def _counter_to_list(counter: dict, top_n: Optional[int] = None) -> list[dict]:
    items = counter.most_common(top_n) if isinstance(counter, Counter) else []
    results = []
    total = sum(counter.values()) if isinstance(counter, Counter) else 0
    for key, count in items:
        entry = {"value": key, "count": int(count)}
        if total > 0:
            entry["percent"] = round(float(count) / float(total), 4)
        results.append(entry)
    return results


def _collect_gene_counts(sub_df: pd.DataFrame) -> Counter:
    gene_counter: Counter = Counter()
    for genes_str in sub_df["target_genes"]:
        for gene in _split_genes(genes_str):
            gene_counter[gene] += 1
    return gene_counter


def _top_compound_rows(rows: pd.DataFrame, limit: int) -> list[dict]:
    output = []
    for _, row in rows.head(limit).iterrows():
        output.append(
            {
                "batch_id": row.get("batch_id"),
                "compound_name": row.get("compound_name"),
                "mechanism_of_action": row.get("mechanism_of_action"),
                "disease_area": row.get("disease_area"),
                "clinical_phase": row.get("clinical_phase"),
                "target_genes": row.get("target_genes"),
                "cell_viability_pct": round(float(row.get("cell_viability_pct")), 2)
                if pd.notna(row.get("cell_viability_pct"))
                else None,
                "grit_score": round(float(row.get("grit_score")), 4)
                if pd.notna(row.get("grit_score"))
                else None,
            }
        )
    return output


# =============================================================================
# RELATIONAL ANALYTICS CORE FUNCTIONS
# =============================================================================

@mcp.tool()
def gene_compounds(gene: str, limit: int = 50) -> dict:
    """
    List compounds that target a specific gene symbol.

    Args:
        gene: Gene symbol (compared in uppercase).
        limit: Maximum number of compounds to return.

    Returns:
        Dict with the gene, total count, and compound details up to the limit.
    """
    df = _get_df()

    query_gene = gene.upper().strip()
    if not query_gene:
        return {"error": "Gene must be a non-empty string"}

    matches = df[df["target_genes"].notna()].copy()
    matches = matches[matches["target_genes"].apply(lambda x: query_gene in _split_genes(x))]

    results = _top_compound_rows(matches, limit)
    return {
        "gene": query_gene,
        "compound_count": int(len(matches)),
        "compounds": results,
    }


@mcp.tool()
def gene_context(
    gene: str,
    top_genes: int = 50,
    top_moas: int = 20,
    top_diseases: int = 20,
) -> dict:
    """
    Summarize the context for a gene: co-occurring genes, MOAs, and diseases.

    Args:
        gene: Gene symbol (compared in uppercase).
        top_genes: Number of top co-genes to include.
        top_moas: Number of top MOAs to include.
        top_diseases: Number of top disease areas to include.

    Returns:
        Dict with counts and ranked co-gene/MOA/disease stats for the gene.
    """
    df = _get_df()

    query_gene = gene.upper().strip()
    if not query_gene:
        return {"error": "Gene must be a non-empty string"}

    matches = df[df["target_genes"].notna()].copy()
    matches = matches[matches["target_genes"].apply(lambda x: query_gene in _split_genes(x))]
    if len(matches) == 0:
        return {
            "gene": query_gene,
            "compound_count": 0,
            "top_co_genes": [],
            "top_moas": [],
            "top_diseases": [],
        }

    co_gene_counter: Counter = Counter()
    for genes_str in matches["target_genes"]:
        genes = _split_genes(genes_str)
        for g in genes:
            if g != query_gene:
                co_gene_counter[g] += 1

    moa_counter: Counter = Counter()
    moa_examples: dict = {}
    for _, row in matches.iterrows():
        moa = row.get("mechanism_of_action")
        if pd.isna(moa):
            continue
        moa_counter[moa] += 1
        moa_examples.setdefault(moa, []).append(row.get("compound_name"))

    disease_counter: Counter = Counter()
    disease_examples: dict = {}
    for _, row in matches.iterrows():
        dis = row.get("disease_area")
        if pd.isna(dis):
            continue
        disease_counter[dis] += 1
        disease_examples.setdefault(dis, []).append(row.get("compound_name"))

    top_moas_list = []
    for moa, count in moa_counter.most_common(top_moas):
        top_moas_list.append(
            {
                "moa": moa,
                "count": int(count),
                "sample_compounds": moa_examples.get(moa, [])[:3],
            }
        )

    top_diseases_list = []
    for dis, count in disease_counter.most_common(top_diseases):
        top_diseases_list.append(
            {
                "disease_area": dis,
                "count": int(count),
                "sample_compounds": disease_examples.get(dis, [])[:3],
            }
        )

    return {
        "gene": query_gene,
        "compound_count": int(len(matches)),
        "top_co_genes": _counter_to_list(co_gene_counter, top_genes),
        "top_moas": top_moas_list,
        "top_diseases": top_diseases_list,
    }


@mcp.tool()
def compound_targets(
    compound_name: str,
) -> dict:
    """
    Gene-overlap neighbors: retrieve target genes for a compound and find other compounds
    that share those genes, prioritized by shared count then embedding similarity. Use
    this when gene overlap is primary; use query_compound_similarity for global embedding
    neighbors or compound_neighbors for scoped embedding neighbors.

    Args:
        compound_name: Partial or full compound name (case-insensitive contains match).

    Returns:
        Dict with the compound name, its target genes, and compounds sharing genes
        (ranked by shared count then embedding similarity). Error returned if not found.
    """
    df = _get_df()

    row = _find_compound_row(df, compound_name)
    if row is None:
        return {"error": f"Compound '{compound_name}' not found"}

    query_idx = row.name
    query_genes = _split_genes(row.get("target_genes"))
    result = {
        "compound_name": row.get("compound_name"),
        "target_genes": query_genes,
    }

    if not query_genes:
        result["note"] = "No target genes available for this compound."
        return result

    other = df.drop(index=query_idx)
    overlaps = []
    query_genes_set = set(query_genes)
    for idx, other_row in other.iterrows():
        other_genes = set(_split_genes(other_row.get("target_genes")))
        shared = list(query_genes_set & other_genes)
        if not shared:
            continue
        overlaps.append(
            {
                "_idx": idx,
                "compound_name": other_row.get("compound_name"),
                "mechanism_of_action": other_row.get("mechanism_of_action"),
                "disease_area": other_row.get("disease_area"),
                "clinical_phase": other_row.get("clinical_phase"),
                "shared_genes": shared,
                "shared_gene_count": len(shared),
            }
        )

    if len(overlaps) > 0:
        query_emb = row[PCA_COLUMNS].values.reshape(1, -1)
        for comp in overlaps:
            emb = df.loc[comp["_idx"], PCA_COLUMNS].values.reshape(1, -1)
            sim = float(cosine_similarity(query_emb, emb)[0, 0])
            comp["similarity_score"] = round(sim, 4)
            comp.pop("_idx", None)

        overlaps.sort(
            key=lambda x: (
                -x["shared_gene_count"],
                -x.get("similarity_score", 0.0),
                x["compound_name"] or "",
            )
        )

    result["compounds_sharing_genes"] = overlaps
    return result


@mcp.tool()
def compound_neighbors(
    compound_name: str,
    scope: str = "auto",
    filter_value: Optional[str] = None,
    top_n: int = 20,
) -> dict:
    """
    Scoped embedding neighbors: find the nearest compounds for a query **within a specific scope**.
    This tool never returns global neighbors; use query_compound_similarity for unscoped search.

    Args:
        compound_name: Partial or full compound name (case-insensitive contains match).
        scope: 'moa', 'disease', 'shared_genes', or 'auto' (default). 'auto' picks the first
               available among moa -> disease -> shared_genes based on the query compound.
        filter_value: Optional override value for MOA/disease scopes instead of the query compound's value.
        top_n: Number of neighbors to return.

    Returns:
        Dict with the query compound, scope, and a ranked neighbor list (or an error).
    """
    df = _get_df()

    row = _find_compound_row(df, compound_name)
    if row is None:
        return {"error": f"Compound '{compound_name}' not found"}

    query_idx = row.name
    query_emb = row[PCA_COLUMNS].values.reshape(1, -1)

    scope = scope.lower() if isinstance(scope, str) else None
    valid_scopes = {"moa", "disease", "shared_genes", "auto"}
    if scope not in valid_scopes:
        return {"error": f"Invalid scope '{scope}'. Use one of {valid_scopes}."}

    if scope == "auto":
        if pd.notna(row.get("mechanism_of_action")) and str(row.get("mechanism_of_action")).strip():
            scope = "moa"
        elif pd.notna(row.get("disease_area")) and str(row.get("disease_area")).strip():
            scope = "disease"
        elif _split_genes(row.get("target_genes")):
            scope = "shared_genes"
        else:
            return {
                "compound_name": row.get("compound_name"),
                "scope": scope,
                "neighbors": [],
                "note": "No MOA, disease, or target gene information available to scope neighbors.",
            }

    candidates = df.drop(index=query_idx)

    if scope == "moa":
        target_moa = filter_value or row.get("mechanism_of_action")
        if pd.isna(target_moa) or not target_moa:
            return {"error": "No MOA available for the query compound; provide filter_value."}
        candidates = candidates[
            candidates["mechanism_of_action"].fillna("").str.lower().str.contains(
                str(target_moa).lower(), regex=False
            )
        ]
    elif scope == "disease":
        target_disease = filter_value or row.get("disease_area")
        if pd.isna(target_disease) or not target_disease:
            return {"error": "No disease area available for the query compound; provide filter_value."}
        candidates = candidates[
            candidates["disease_area"].fillna("").str.lower().str.contains(
                str(target_disease).lower(), regex=False
            )
        ]
    elif scope == "shared_genes":
        query_genes = set(_split_genes(row.get("target_genes")))
        if not query_genes:
            return {"error": "Query compound has no target genes to filter by shared_genes scope."}
        candidates = candidates[
            candidates["target_genes"].apply(
                lambda x: len(query_genes & set(_split_genes(x))) > 0
            )
        ]

    if len(candidates) == 0:
        return {
            "compound_name": row.get("compound_name"),
            "scope": scope,
            "neighbors": [],
            "note": "No candidate compounds found for the given scope/filter.",
        }

    candidate_embeddings = candidates[PCA_COLUMNS].values
    sims = cosine_similarity(query_emb, candidate_embeddings)[0]

    query_genes_set = set(_split_genes(row.get("target_genes")))
    neighbors = []
    for sim_value, (_, cand_row) in zip(sims, candidates.iterrows()):
        shared_genes = list(query_genes_set & set(_split_genes(cand_row.get("target_genes"))))
        neighbors.append(
            {
                "compound_name": cand_row.get("compound_name"),
                "mechanism_of_action": cand_row.get("mechanism_of_action"),
                "disease_area": cand_row.get("disease_area"),
                "clinical_phase": cand_row.get("clinical_phase"),
                "similarity_score": round(float(sim_value), 4),
                "shared_genes": shared_genes if shared_genes else None,
            }
        )

    if scope == "shared_genes":
        neighbors.sort(
            key=lambda x: (
                -(len(x.get("shared_genes") or [])),
                -x.get("similarity_score", 0.0),
                str(x.get("compound_name") or ""),
            )
        )
    else:
        neighbors.sort(
            key=lambda x: (
                -x.get("similarity_score", 0.0),
                str(x.get("compound_name") or ""),
            )
        )

    return {
        "compound_name": row.get("compound_name"),
        "scope": scope,
        "neighbors": neighbors[:top_n],
    }


@mcp.tool()
def compound_metadata(
    compound_name: str,
    include_disease_neighbors: bool = False,
    neighbors_top_n: int = 20,
) -> dict:
    """
    Fetch core metadata for a compound and optionally include disease-matched neighbors.

    Args:
        compound_name: Partial or full compound name (case-insensitive contains match).
        include_disease_neighbors: Whether to append neighbors sharing the disease_area.
        neighbors_top_n: Number of neighbors to include when requested.

    Returns:
        Dict with compound metadata and optional disease neighbors (or an error).
    """
    df = _get_df()

    row = _find_compound_row(df, compound_name)
    if row is None:
        return {"error": f"Compound '{compound_name}' not found"}

    result = {
        "compound_name": row.get("compound_name"),
        "mechanism_of_action": row.get("mechanism_of_action"),
        "disease_area": row.get("disease_area"),
        "clinical_phase": row.get("clinical_phase"),
        "target_genes": _split_genes(row.get("target_genes")),
        "cell_viability_pct": round(float(row.get("cell_viability_pct")), 2)
        if pd.notna(row.get("cell_viability_pct"))
        else None,
        "grit_score": round(float(row.get("grit_score")), 4)
        if pd.notna(row.get("grit_score"))
        else None,
    }

    if include_disease_neighbors and pd.notna(row.get("disease_area")):
        neighbors = compound_neighbors(
            compound_name=row.get("compound_name"),
            scope="disease",
            filter_value=row.get("disease_area"),
            top_n=neighbors_top_n,
        )
        if "neighbors" in neighbors:
            result["disease_neighbors"] = neighbors["neighbors"]

    return result


@mcp.tool()
def moa_summary(
    moa: str,
    include_compounds: bool = False,
    limit: int = 100,
) -> dict:
    """
    Summarize a mechanism of action with gene/disease/phase distributions and stats.

    Args:
        moa: Partial or full MOA string (case-insensitive contains match).
        include_compounds: Whether to include per-compound rows.
        limit: Max number of compounds to include when requested.

    Returns:
        Dict with counts, viability/grit stats, top genes, disease/phase distribution,
        and optional compound list (or an error).
    """
    df = _get_df()

    mask = df["mechanism_of_action"].fillna("").str.lower().str.contains(
        moa.lower(), regex=False
    )
    subset = df[mask]
    if len(subset) == 0:
        return {"error": f"MOA '{moa}' not found"}

    matched_moa = subset["mechanism_of_action"].mode().iloc[0]
    gene_counter = _collect_gene_counts(subset)
    disease_counter = Counter(subset["disease_area"].dropna())
    phase_counter = Counter(subset["clinical_phase"].dropna())

    result = {
        "moa": matched_moa,
        "compound_count": int(len(subset)),
        "cell_viability_stats": _viability_stats(subset["cell_viability_pct"]),
        "grit_score_stats": _grit_stats(subset["grit_score"]),
        "top_genes": _counter_to_list(gene_counter),
        "disease_distribution": _counter_to_list(disease_counter),
        "clinical_phase_distribution": _counter_to_list(phase_counter),
    }

    if include_compounds:
        result["compounds"] = _top_compound_rows(subset, limit)

    return result


@mcp.tool()
def moa_relations(
    moa: str,
    top_n: int = 20,
) -> dict:
    """
    Compute similarity between the given MOA centroid and other MOA centroids.

    Args:
        moa: Partial or full MOA string (case-insensitive contains match).
        top_n: Number of nearest MOA neighbors to return.

    Returns:
        Dict with the matched MOA and a ranked list of similar MOAs (or an error).
    """
    df = _get_df()

    mask = df["mechanism_of_action"].fillna("").str.lower().str.contains(
        moa.lower(), regex=False
    )
    subset = df[mask]
    if len(subset) == 0:
        return {"error": f"MOA '{moa}' not found"}

    matched_moa = subset["mechanism_of_action"].mode().iloc[0]
    grouped = df[df["mechanism_of_action"].notna()].groupby("mechanism_of_action")
    centroids = {name: group[PCA_COLUMNS].values.mean(axis=0) for name, group in grouped}

    query_centroid = centroids.get(matched_moa)
    if query_centroid is None:
        return {"error": f"No centroid available for MOA '{matched_moa}'"}

    others = {k: v for k, v in centroids.items() if k != matched_moa}
    if not others:
        return {"moa": matched_moa, "neighbors": []}

    query_vec = query_centroid.reshape(1, -1)
    other_names = list(others.keys())
    other_matrix = np.vstack([others[name] for name in other_names])
    sims = cosine_similarity(query_vec, other_matrix)[0]

    neighbors = []
    for name, sim in zip(other_names, sims):
        neighbors.append({"moa": name, "similarity_score": round(float(sim), 4)})

    neighbors.sort(key=lambda x: -x["similarity_score"])
    return {"moa": matched_moa, "neighbors": neighbors[:top_n]}


@mcp.tool()
def disease_summary(
    disease_area: str,
    include_compounds: bool = False,
    limit: int = 100,
) -> dict:
    """
    Summarize a disease area with gene/MOA/phase distributions and stats.

    Args:
        disease_area: Partial or full disease area string (case-insensitive contains match).
        include_compounds: Whether to include per-compound rows.
        limit: Max number of compounds to include when requested.

    Returns:
        Dict with counts, viability/grit stats, top genes, MOA/phase distribution,
        and optional compound list (or an error).
    """
    df = _get_df()

    mask = df["disease_area"].fillna("").str.lower().str.contains(
        disease_area.lower(), regex=False
    )
    subset = df[mask]
    if len(subset) == 0:
        return {"error": f"Disease area '{disease_area}' not found"}

    matched_disease = subset["disease_area"].mode().iloc[0]
    gene_counter = _collect_gene_counts(subset)
    moa_counter = Counter(subset["mechanism_of_action"].dropna())
    phase_counter = Counter(subset["clinical_phase"].dropna())

    result = {
        "disease_area": matched_disease,
        "compound_count": int(len(subset)),
        "cell_viability_stats": _viability_stats(subset["cell_viability_pct"]),
        "grit_score_stats": _grit_stats(subset["grit_score"]),
        "top_genes": _counter_to_list(gene_counter),
        "moa_distribution": _counter_to_list(moa_counter),
        "clinical_phase_distribution": _counter_to_list(phase_counter),
    }

    if include_compounds:
        result["compounds"] = _top_compound_rows(subset, limit)

    return result


@mcp.tool()
def disease_relations_by_phase(
    phase: str,
    direction: str = "high",
    top_n: int = 20,
) -> dict:
    """
    Rank compounds within a clinical phase by cell viability and summarize disease mix.

    Args:
        phase: Partial or full clinical phase string (case-insensitive contains match).
        direction: 'high' for highest viability or 'low' for lowest.
        top_n: Number of top compounds to return.

    Returns:
        Dict with the matched phase, direction, top compounds, and disease distribution
        (or an error).
    """
    df = _get_df()

    direction = direction.lower()
    if direction not in {"high", "low"}:
        return {"error": "direction must be 'high' or 'low'"}

    mask = df["clinical_phase"].fillna("").str.lower().str.contains(
        phase.lower(), regex=False
    )
    subset = df[mask].copy()
    subset = subset[pd.notna(subset["cell_viability_pct"])]
    if len(subset) == 0:
        return {"error": f"No compounds found for phase '{phase}'"}

    subset = subset.sort_values("cell_viability_pct", ascending=(direction == "low"))
    top_compounds = _top_compound_rows(subset, top_n)
    disease_counter = Counter(subset["disease_area"].dropna())

    return {
        "clinical_phase": subset["clinical_phase"].mode().iloc[0],
        "direction": direction,
        "top_compounds": top_compounds,
        "disease_distribution": _counter_to_list(disease_counter),
    }


@mcp.tool()
def phase_summary_for_moa(
    moa: str,
) -> dict:
    """
    Show clinical phase distribution for compounds in a given MOA.

    Args:
        moa: Partial or full MOA string (case-insensitive contains match).

    Returns:
        Dict with the matched MOA and per-phase counts plus sample compounds (or an error).
    """
    df = _get_df()

    mask = df["mechanism_of_action"].fillna("").str.lower().str.contains(
        moa.lower(), regex=False
    )
    subset = df[mask]
    if len(subset) == 0:
        return {"error": f"MOA '{moa}' not found"}

    phase_counter = Counter(subset["clinical_phase"].dropna())
    examples = {}
    for _, row in subset.iterrows():
        phase_val = row.get("clinical_phase")
        if pd.isna(phase_val):
            continue
        examples.setdefault(phase_val, []).append(row.get("compound_name"))

    phases = []
    for phase_val, count in phase_counter.most_common():
        phases.append(
            {
                "clinical_phase": phase_val,
                "count": int(count),
                "sample_compounds": examples.get(phase_val, [])[:3],
            }
        )

    return {"moa": subset["mechanism_of_action"].mode().iloc[0], "phases": phases}


@mcp.tool()
def moa_summary_for_phase(
    phase: str,
) -> dict:
    """
    Show MOA distribution for compounds within a clinical phase.

    Args:
        phase: Partial or full clinical phase string (case-insensitive contains match).

    Returns:
        Dict with the matched phase and per-MOA counts plus sample compounds (or an error).
    """
    df = _get_df()

    mask = df["clinical_phase"].fillna("").str.lower().str.contains(
        phase.lower(), regex=False
    )
    subset = df[mask]
    if len(subset) == 0:
        return {"error": f"No compounds found for phase '{phase}'"}

    moa_counter = Counter(subset["mechanism_of_action"].dropna())
    examples = {}
    for _, row in subset.iterrows():
        moa_val = row.get("mechanism_of_action")
        if pd.isna(moa_val):
            continue
        examples.setdefault(moa_val, []).append(row.get("compound_name"))

    moas = []
    for moa_val, count in moa_counter.most_common():
        moas.append(
            {
                "moa": moa_val,
                "count": int(count),
                "sample_compounds": examples.get(moa_val, [])[:3],
            }
        )

    return {"clinical_phase": subset["clinical_phase"].mode().iloc[0], "moas": moas}


@mcp.tool()
def disease_summary_for_moa(
    moa: str,
) -> dict:
    """
    Summarize disease areas represented within a given MOA.

    Args:
        moa: Partial or full MOA string (case-insensitive contains match).

    Returns:
        Dict with the matched MOA and per-disease counts plus sample compounds (or an error).
    """
    df = _get_df()

    mask = df["mechanism_of_action"].fillna("").str.lower().str.contains(
        moa.lower(), regex=False
    )
    subset = df[mask]
    if len(subset) == 0:
        return {"error": f"MOA '{moa}' not found"}

    disease_counter = Counter(subset["disease_area"].dropna())
    examples = {}
    for _, row in subset.iterrows():
        dis_val = row.get("disease_area")
        if pd.isna(dis_val):
            continue
        examples.setdefault(dis_val, []).append(row.get("compound_name"))

    diseases = []
    for dis_val, count in disease_counter.most_common():
        diseases.append(
            {
                "disease_area": dis_val,
                "count": int(count),
                "sample_compounds": examples.get(dis_val, [])[:3],
            }
        )

    return {"moa": subset["mechanism_of_action"].mode().iloc[0], "diseases": diseases}


@mcp.tool()
def moa_summary_for_disease(
    disease_area: str,
) -> dict:
    """
    Summarize MOAs represented within a given disease area.

    Args:
        disease_area: Partial or full disease area string (case-insensitive contains match).

    Returns:
        Dict with the matched disease area and per-MOA counts plus sample compounds (or an error).
    """
    df = _get_df()

    mask = df["disease_area"].fillna("").str.lower().str.contains(
        disease_area.lower(), regex=False
    )
    subset = df[mask]
    if len(subset) == 0:
        return {"error": f"Disease area '{disease_area}' not found"}

    moa_counter = Counter(subset["mechanism_of_action"].dropna())
    examples = {}
    for _, row in subset.iterrows():
        moa_val = row.get("mechanism_of_action")
        if pd.isna(moa_val):
            continue
        examples.setdefault(moa_val, []).append(row.get("compound_name"))

    moas = []
    for moa_val, count in moa_counter.most_common():
        moas.append(
            {
                "moa": moa_val,
                "count": int(count),
                "sample_compounds": examples.get(moa_val, [])[:3],
            }
        )

    return {"disease_area": subset["disease_area"].mode().iloc[0], "moas": moas}


@mcp.tool()
def genes_for_group(
    group_type: str,
    value: str,
    limit: int = 50,
) -> dict:
    """
    List the most common target genes within an MOA or disease group.

    Args:
        group_type: 'moa' or 'disease' to choose the grouping column.
        value: Partial or full string to match in the chosen group.
        limit: Max number of genes to return.

    Returns:
        Dict with the group info and ranked gene counts (or an error).
    """
    df = _get_df()

    group_type = group_type.lower()
    if group_type not in {"moa", "disease"}:
        return {"error": "group_type must be 'moa' or 'disease'"}

    column = "mechanism_of_action" if group_type == "moa" else "disease_area"
    mask = df[column].fillna("").str.lower().str.contains(value.lower(), regex=False)
    subset = df[mask]
    if len(subset) == 0:
        return {"error": f"No compounds found for {group_type} '{value}'"}

    gene_counter = _collect_gene_counts(subset)
    return {
        "group_type": group_type,
        "value": subset[column].mode().iloc[0],
        "genes": _counter_to_list(gene_counter, limit),
    }


@mcp.tool()
def top_viability_in_group(
    group_type: str,
    value: str,
    direction: str = "high",
    limit: int = 20,
) -> dict:
    """
    Rank compounds by cell viability within a given MOA or clinical phase.

    Args:
        group_type: 'moa' or 'phase' to choose the grouping column.
        value: Partial or full string to match in the chosen group.
        direction: 'high' for highest viability or 'low' for lowest.
        limit: Number of compounds to return.

    Returns:
        Dict with the group info, direction, and ranked compound rows (or an error).
    """
    df = _get_df()

    group_type = group_type.lower()
    if group_type not in {"moa", "phase"}:
        return {"error": "group_type must be 'moa' or 'phase'"}

    direction = direction.lower()
    if direction not in {"high", "low"}:
        return {"error": "direction must be 'high' or 'low'"}

    column = "mechanism_of_action" if group_type == "moa" else "clinical_phase"
    mask = df[column].fillna("").str.lower().str.contains(value.lower(), regex=False)
    subset = df[mask].copy()
    subset = subset[pd.notna(subset["cell_viability_pct"])]
    if len(subset) == 0:
        return {"error": f"No compounds found for {group_type} '{value}'"}

    subset = subset.sort_values("cell_viability_pct", ascending=(direction == "low"))
    return {
        "group_type": group_type,
        "value": subset[column].mode().iloc[0],
        "direction": direction,
        "compounds": _top_compound_rows(subset, limit),
    }


# =============================================================================
# MCP SERVER UTILITIES
# =============================================================================

@mcp.tool()
def reload_dataset() -> dict:
    """
    Reload the compound CSV from disk and refresh the in-memory cache.

    Use this after updating `compound_aggregate_with_annotations.csv` to ensure
    all downstream tools read the latest data. Returns a brief status plus the
    current row and unique compound counts so you can confirm the reload worked.
    """
    df = _reload_df()
    return {
        "status": "reloaded",
        "total_compounds": len(df),
        "unique_compound_names": df["compound_name"].nunique(),
    }


if __name__ == "__main__":
    import os

    transport = os.getenv("MCP_TRANSPORT", "streamable-http")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    run_kwargs = {"transport": transport}
    if transport in {"http", "streamable-http"}:
        run_kwargs.update({"host": host, "port": port})

    mcp.run(**run_kwargs)

from __future__ import annotations

"""
Step 05 - Transcriptomics and spatial omics integration.

Run this after SpatialNI features are available. It links neighbourhood indices
to bulk transcriptomic signatures, spatial interfaces, cell-state weights, and
ligand-receptor scores.
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

"""Python helpers for the spatial-omics side of SpatialNI.
Bulk transcriptomic signature scoring and multi-method immune deconvolution are
kept in the companion R scripts in this folder so the public workflow matches
the manuscript software structure.
"""

NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

def zscore_series(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    return pd.Series(np.zeros(len(series)), index=series.index, dtype=float) if pd.isna(std) or std == 0 else (series - series.mean()) / std

def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(float); out = np.full(len(values), np.nan, float); keep = np.isfinite(values)
    if not keep.any(): return pd.Series(out, index=p_values.index, dtype=float)
    order = np.argsort(values[keep]); ranked = values[keep][order]; adj = ranked * len(ranked) / np.arange(1, len(ranked) + 1); adj = np.minimum.accumulate(adj[::-1])[::-1]
    restored = np.empty_like(adj); restored[order] = np.clip(adj, 0.0, 1.0); out[np.flatnonzero(keep)] = restored
    return pd.Series(out, index=p_values.index, dtype=float)

def parse_formula_column(formula_text: str, frame: pd.DataFrame) -> pd.Series:
    local_dict = {col: pd.to_numeric(frame[col], errors="coerce") for col in frame.columns}; local_dict["np"] = np
    return pd.Series(eval(formula_text, {"__builtins__": {}}, local_dict), index=frame.index, dtype=float)

def normalize_ratio_column_names(spatial_df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {col: col.replace("_has_", "_to_").replace("_ratio", "_NI") for col in spatial_df.columns if col.endswith("_ratio") and "_has_" in col}
    return spatial_df.rename(columns=rename_map)

def build_composite_indices(spatial_df: pd.DataFrame, index_definitions: pd.DataFrame, name_col: str = "Index", formula_col: str = "Formula") -> pd.DataFrame:
    work = normalize_ratio_column_names(spatial_df)
    for _, row in index_definitions[[name_col, formula_col]].dropna().iterrows(): work[str(row[name_col])] = parse_formula_column(str(row[formula_col]), work)
    return work

def aggregate_patient_level(table: pd.DataFrame, patient_col: str, value_cols: list[str]) -> pd.DataFrame:
    return table[[patient_col] + value_cols].dropna(subset=[patient_col]).groupby(patient_col, as_index=False)[value_cols].mean(numeric_only=True).replace([np.inf, -np.inf], np.nan)

def correlate_indices_with_features(analysis_df: pd.DataFrame, index_cols: list[str], feature_cols: list[str], min_pair_n: int = 8) -> pd.DataFrame:
    rows = []
    for idx_name in index_cols:
        for feat_name in feature_cols:
            sub = analysis_df[[idx_name, feat_name]].dropna()
            if len(sub) < min_pair_n or sub[idx_name].std(ddof=0) == 0 or sub[feat_name].std(ddof=0) == 0: rho, p = np.nan, np.nan
            else: rho, p = spearmanr(sub[idx_name], sub[feat_name])
            rows.append({"index": idx_name, "feature": feat_name, "spearman_r": rho, "p_value": p, "n": len(sub)})
    out = pd.DataFrame(rows); keep = out["p_value"].notna(); out.loc[keep, "fdr"] = benjamini_hochberg(out.loc[keep, "p_value"])
    return out.sort_values(["index", "p_value", "feature"]).reset_index(drop=True)

def build_spatial_neighbor_edges(bin_df: pd.DataFrame, row_col: str = "array_row", col_col: str = "array_col", id_col: str = "barcode") -> pd.DataFrame:
    need = {row_col, col_col, id_col}; missing = need.difference(bin_df.columns)
    if missing: raise KeyError(f"Missing required columns: {sorted(missing)}")
    left, right, edges = bin_df.copy(), bin_df.copy(), []
    for drow, dcol in NEIGHBOR_OFFSETS:
        tmp = left.copy(); tmp["neighbor_row"] = tmp[row_col] + drow; tmp["neighbor_col"] = tmp[col_col] + dcol
        merged = tmp.merge(right, left_on=["neighbor_row", "neighbor_col"], right_on=[row_col, col_col], how="inner", suffixes=("_source", "_target"))
        if not merged.empty: edges.append(merged)
    if not edges: return pd.DataFrame(columns=["source_barcode", "target_barcode"])
    edge_df = pd.concat(edges, ignore_index=True).rename(columns={f"{id_col}_source": "source_barcode", f"{id_col}_target": "target_barcode"})
    edge_df = edge_df.loc[edge_df["source_barcode"] != edge_df["target_barcode"]].copy(); edge_df["edge_id"] = edge_df["source_barcode"].astype(str) + "->" + edge_df["target_barcode"].astype(str)
    return edge_df.drop_duplicates(subset=["edge_id"]).reset_index(drop=True)

def compare_interface_vs_background(score_df: pd.DataFrame, value_cols: list[str], group_col: str = "group", interface_label: str = "interface", background_label: str = "baseline") -> pd.DataFrame:
    rows = []
    for value_col in value_cols:
        sub = score_df[[group_col, value_col]].dropna(); a = sub.loc[sub[group_col] == interface_label, value_col].to_numpy(float); b = sub.loc[sub[group_col] == background_label, value_col].to_numpy(float)
        if len(a) == 0 or len(b) == 0: continue
        u, p = mannwhitneyu(a, b, alternative="two-sided"); rows.append({"feature": value_col, "mean_interface": float(a.mean()), "mean_background": float(b.mean()), "log2_fc": float(math.log2((a.mean() + 1e-6) / (b.mean() + 1e-6))), "p_value": float(p), "u_stat": float(u), "n_interface": int(len(a)), "n_background": int(len(b))})
    out = pd.DataFrame(rows)
    if not out.empty: out["fdr"] = benjamini_hochberg(out["p_value"])
    return out.sort_values(["fdr", "p_value", "feature"]).reset_index(drop=True)

def compare_genes_between_groups(expr_df: pd.DataFrame, gene_cols: list[str], group_mask: pd.Series, reference_mask: pd.Series, min_group_n: int = 5) -> pd.DataFrame:
    group_mask = pd.Series(group_mask, index=expr_df.index).fillna(False).astype(bool); reference_mask = pd.Series(reference_mask, index=expr_df.index).fillna(False).astype(bool); rows = []
    for gene in gene_cols:
        if gene not in expr_df.columns: continue
        a = pd.to_numeric(expr_df.loc[group_mask, gene], errors="coerce").dropna().to_numpy(float); b = pd.to_numeric(expr_df.loc[reference_mask, gene], errors="coerce").dropna().to_numpy(float)
        if len(a) < min_group_n or len(b) < min_group_n: continue
        u, p = mannwhitneyu(a, b, alternative="two-sided"); rows.append({"gene": gene, "mean_group": float(a.mean()), "mean_reference": float(b.mean()), "log2_fc": float(math.log2((a.mean() + 1e-6) / (b.mean() + 1e-6))), "p_value": float(p), "u_stat": float(u), "n_group": int(len(a)), "n_reference": int(len(b))})
    out = pd.DataFrame(rows)
    if not out.empty: out["fdr"] = benjamini_hochberg(out["p_value"])
    return out.sort_values(["fdr", "p_value", "gene"]).reset_index(drop=True)

def attach_top1_state(weight_df: pd.DataFrame, state_cols: list[str]) -> pd.DataFrame:
    work = weight_df.copy(); mat = work[state_cols].to_numpy(float); top = mat.argmax(axis=1); work["top1_state"] = [state_cols[i] for i in top]; work["top1_weight"] = mat[np.arange(len(work)), top]; return work

def ligand_receptor_edge_scores(edge_df: pd.DataFrame, lr_pairs: pd.DataFrame, sender_suffix: str = "_sender", receiver_suffix: str = "_receiver", receiver_weight_col: str | None = None) -> pd.DataFrame:
    if not {"ligand", "receptor"}.issubset(lr_pairs.columns): raise KeyError("Missing ligand-receptor columns")
    receiver_weight = edge_df[receiver_weight_col].to_numpy(float)[:, None] if receiver_weight_col is not None and receiver_weight_col in edge_df.columns else 1.0; rows = []
    for _, pair in lr_pairs[["ligand", "receptor"]].drop_duplicates().iterrows():
        ligand, receptor = str(pair["ligand"]), str(pair["receptor"]); lcol, rcol = f"{ligand}{sender_suffix}", f"{receptor}{receiver_suffix}"
        if lcol not in edge_df.columns or rcol not in edge_df.columns: continue
        score = edge_df[lcol].to_numpy(float) * edge_df[rcol].to_numpy(float) * receiver_weight
        rows.append(pd.DataFrame({"source_barcode": edge_df["source_barcode"].astype(str).to_numpy(), "target_barcode": edge_df["target_barcode"].astype(str).to_numpy(), "ligand": ligand, "receptor": receptor, "lr_pair": f"{ligand}->{receptor}", "score": np.asarray(score, float).ravel()}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["source_barcode", "target_barcode", "ligand", "receptor", "lr_pair", "score"])

def summarize_ligand_receptor_scores(score_df: pd.DataFrame) -> pd.DataFrame:
    if score_df.empty: return pd.DataFrame(columns=["lr_pair", "ligand", "receptor", "mean_score", "median_score", "n_edges"])
    return score_df.groupby(["lr_pair", "ligand", "receptor"], as_index=False).agg(mean_score=("score", "mean"), median_score=("score", "median"), n_edges=("score", "size")).sort_values(["mean_score", "n_edges"], ascending=[False, False]).reset_index(drop=True)

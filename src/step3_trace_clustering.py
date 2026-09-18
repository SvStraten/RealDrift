from __future__ import annotations

import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#features
def compute_fixed_features(trace_df, extra_case_cols=()) -> dict:
    activities, resources = trace_df["activity"].tolist(), trace_df["resource"].tolist()
    timestamps = trace_df["timestamp"].tolist()
    duration_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600

    features = {
        "trace_length": float(len(activities)),
        "unique_activities": float(len(set(activities))),
        "unique_resources": float(len(set(resources))),
        "log_duration_hours": float(np.log1p(duration_hours)),
    }
    for col in extra_case_cols:
        features[col] = float(trace_df[col].iloc[0])
    return features


def compute_act_features(trace_df) -> dict:
    activities = trace_df["activity"].tolist()
    n = len(activities)
    return {f"act::{act}": c / n for act, c in Counter(activities).items()}


def compute_trace_features(trace_df, extra_case_cols=()) -> dict:
    return {**compute_fixed_features(trace_df, extra_case_cols), **compute_act_features(trace_df)}


def extract_all_features(df, extra_case_cols=(), progress_every=5000):
    df = df.sort_values(["case_id", "timestamp"])
    all_features = {}
    for i, (cid, trace_df) in enumerate(df.groupby("case_id", sort=False)):
        all_features[cid] = compute_trace_features(trace_df, extra_case_cols)
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  {i+1} traces done")
    return all_features


def cluster_pipeline(feature_dicts, name, target_variance=0.80, k_range=range(2, 21),
                      min_cluster_size=10, sample_size=3000, seed=42, verbose=True,
                      feature_weights=None):
    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform(feature_dicts)
    X_dense = X.toarray()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dense)
    _assert_no_nan(X_scaled, vectorizer, name)

    if feature_weights:
        feat_names = list(vectorizer.get_feature_names_out())
        for fname, w in feature_weights.items():
            if fname in feat_names:
                X_scaled[:, feat_names.index(fname)] *= w
            elif verbose:
                print(f"[{name}] feature_weights: '{fname}' not found in this feature set, ignoring")

    n_components_max = min(300, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca_full = PCA(n_components=n_components_max, random_state=seed)
    pca_full.fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cum_var, target_variance) + 1)

    pca = PCA(n_components=n_components, random_state=seed)
    X_pca = pca.fit_transform(X_scaled)

    sil_scores, disqualified = [], []
    for k in k_range:
        km_trial = KMeans(n_clusters=k, random_state=seed, n_init=10)
        trial_labels = km_trial.fit_predict(X_pca)
        min_size = pd.Series(trial_labels).value_counts().min()
        if min_size < min_cluster_size:
            sil_scores.append(-1.0)
            disqualified.append(k)
            continue
        sil_scores.append(silhouette_score(X_pca, trial_labels, sample_size=min(sample_size, len(trial_labels)),
                                            random_state=seed))

    if max(sil_scores) < 0:
        raise RuntimeError(f"[{name}] every k in {list(k_range)} produced a cluster smaller than "
                            f"min_cluster_size={min_cluster_size} -- lower min_cluster_size or widen k_range.")

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    kmeans = KMeans(n_clusters=best_k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    if verbose:
        print(f"[{name}] matrix={X.shape}, n_components={n_components} ({cum_var[n_components-1]:.1%} variance)")
        if disqualified:
            print(f"[{name}] disqualified k (would leave a cluster < {min_cluster_size}): {disqualified}")
        print(f"[{name}] best_k={best_k}  silhouette={max(sil_scores):.4f}")
        print(pd.Series(labels).value_counts().sort_index().rename("n_traces"))

    return {"name": name, "vectorizer": vectorizer, "scaler": scaler, "pca": pca, "kmeans": kmeans,
            "X_pca": X_pca, "labels": labels, "best_k": best_k, "sil_scores": sil_scores,
            "k_range": list(k_range), "cum_var": cum_var, "n_components": n_components}


def _assert_no_nan(X, vectorizer, name):
    if not np.isfinite(X).all():
        feat_names = np.array(vectorizer.get_feature_names_out())
        bad_cols = feat_names[~np.isfinite(X).all(axis=0)]
        n_bad_rows = (~np.isfinite(X).any(axis=1)).sum()
        raise ValueError(
            f"[{name}] found non-finite values (NaN or inf) in {n_bad_rows} row(s) after "
            f"vectorizing, in feature(s): {list(bad_cols)}. This usually means the input "
            f"dataframe wasn't in chronological order per case before feature extraction "
            f"(log_duration_hours can go NaN from a negative duration), or an extra_case_col "
            f"itself contains NaN -- check those columns before re-running."
        )


def cluster_fixed_k(feature_dicts, name, k, target_variance=0.80, seed=42, feature_weights=None, verbose=True):
    vectorizer = DictVectorizer(sparse=True)
    X = vectorizer.fit_transform(feature_dicts)
    X_dense = X.toarray()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_dense)
    _assert_no_nan(X_scaled, vectorizer, name)

    if feature_weights:
        feat_names = list(vectorizer.get_feature_names_out())
        for fname, w in feature_weights.items():
            if fname in feat_names:
                X_scaled[:, feat_names.index(fname)] *= w
            elif verbose:
                print(f"[{name}] feature_weights: '{fname}' not found in this feature set, ignoring")

    n_components_max = min(300, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca_full = PCA(n_components=n_components_max, random_state=seed)
    pca_full.fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cum_var, target_variance) + 1)

    pca = PCA(n_components=n_components, random_state=seed)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X_pca)

    if verbose:
        print(f"[{name}] matrix={X.shape}, n_components={n_components} ({cum_var[n_components-1]:.1%} variance), k={k} (fixed)")
        print(pd.Series(labels).value_counts().sort_index().rename("n_traces"))

    return {"name": name, "vectorizer": vectorizer, "scaler": scaler, "pca": pca, "kmeans": kmeans,
            "X_pca": X_pca, "labels": labels, "best_k": k, "cum_var": cum_var, "n_components": n_components}


def get_final_concepts(feature_dicts, case_ids, name, min_cluster_size=10,
                        imbalance_ratio=0.3, seed=42, feature_weights=None):
    top = cluster_pipeline(feature_dicts, f"{name}: top level", min_cluster_size=min_cluster_size,
                            seed=seed, feature_weights=feature_weights)
    labels_top = np.asarray(top["labels"])
    sizes = pd.Series(labels_top).value_counts()

    if top["best_k"] == 2 and sizes.min() / sizes.max() < imbalance_ratio:
        small_id, large_id = sizes.idxmin(), sizes.idxmax()
        large_mask = labels_top == large_id
        large_case_ids = [cid for cid, keep in zip(case_ids, large_mask) if keep]
        large_feature_dicts = [fd for fd, keep in zip(feature_dicts, large_mask) if keep]
        sub_min = max(10, int(0.01 * len(large_case_ids)))
        sub = cluster_pipeline(large_feature_dicts, f"{name}: sub-clusters of large group",
                                min_cluster_size=sub_min, k_range=range(2, 11), seed=seed)
        sub_label_of_case = dict(zip(large_case_ids, np.asarray(sub["labels"])))
        raw_concept_of_case = {cid: ("small" if tl == small_id else f"large_{sub_label_of_case[cid]}")
                                for cid, tl in zip(case_ids, labels_top)}
        raw_results = {"top": top, "sub": sub}
    else:
        raw_concept_of_case = {cid: f"{lbl}" for cid, lbl in zip(case_ids, labels_top)}
        raw_results = {"top": top, "sub": None}

    raw_series = pd.Series(raw_concept_of_case).reindex(case_ids)
    order = raw_series.value_counts().index.tolist()
    rename_map = {raw: f"C{i+1}" for i, raw in enumerate(order)}
    concept_series = raw_series.map(rename_map)
    return concept_series, raw_results


def export_sublogs(df, concept_series, dataset_name, out_dir="."):
    os.makedirs(out_dir, exist_ok=True)
    case_concepts = concept_series.rename("concept").rename_axis("case_id").reset_index()
    case_concepts.to_csv(f"{out_dir}/{dataset_name}_case_concepts.csv", index=False)

    df_c = df.merge(case_concepts, on="case_id", how="inner")
    rows = []
    for concept, sub in df_c.groupby("concept"):
        fname = f"{out_dir}/{dataset_name}_sublog_{concept}.csv"
        sub.sort_values(["case_id", "timestamp"]).to_csv(fname, index=False)
        rows.append((concept, sub["case_id"].nunique(), len(sub), fname))
    summary = pd.DataFrame(rows, columns=["concept", "n_cases", "n_events", "file"]).sort_values(
        "n_cases", ascending=False)
    return summary

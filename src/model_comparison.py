"""
model_comparison.py - Compare K-Means, DBSCAN, Agglomerative, GMM.
Returns best model based on silhouette score.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

from src.logger import get_logger

logger = get_logger(__name__)

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)


def _metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """Compute clustering metrics."""

    mask = labels != -1

    Xm = X[mask]
    lm = labels[mask]

    n_clusters = len(set(lm))

    if n_clusters < 2 or len(Xm) < 2:
        return {
            "silhouette": -1,
            "davies_bouldin": 99,
            "calinski_harabasz": 0,
            "n_clusters": n_clusters,
        }

    return {
        "silhouette": round(float(silhouette_score(Xm, lm)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(Xm, lm)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(Xm, lm)), 2),
        "n_clusters": n_clusters,
    }


def compare_models(X: np.ndarray, cfg: dict):

    seed = cfg["model"]["kmeans"]["random_state"]
    n_init = cfg["model"]["kmeans"]["n_init"]

    ####################################################
    # Find Optimal K
    ####################################################

    k_values = []
    sil_scores = []

    best_k = 2
    best_sil = -1

    for k in range(2, 8):

        model = KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=n_init,
        )

        labels = model.fit_predict(X)

        sil = silhouette_score(X, labels)

        k_values.append(k)
        sil_scores.append(sil)

        if sil > best_sil:
            best_sil = sil
            best_k = k

    ####################################################
    # Save Optimal K Plot
    ####################################################

    plt.figure(figsize=(8, 5))

    plt.plot(
        k_values,
        sil_scores,
        marker="o",
        linewidth=2,
    )

    plt.scatter(
        best_k,
        best_sil,
        color="red",
        s=120,
        label=f"Best k = {best_k}",
    )

    plt.xlabel("Number of Clusters")
    plt.ylabel("Silhouette Score")
    plt.title("Optimal Number of Clusters")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "outputs/figures/optimal_k.png",
        dpi=150,
    )

    plt.close()

    logger.info(
        f"Optimal k={best_k} (silhouette={best_sil:.3f})"
    )
    logger.info("Optimal K plot saved.")

    ####################################################
    # Models
    ####################################################

    models = {
        "KMeans": KMeans(
            n_clusters=best_k,
            random_state=seed,
            n_init=n_init,
        ),

        "DBSCAN": DBSCAN(
            eps=cfg["model"]["dbscan"]["eps"],
            min_samples=cfg["model"]["dbscan"]["min_samples"],
        ),

        "Agglomerative": AgglomerativeClustering(
            n_clusters=best_k,
            linkage=cfg["model"]["agglomerative"]["linkage"],
        ),

        "GMM": GaussianMixture(
            n_components=best_k,
            covariance_type=cfg["model"]["gmm"]["covariance_type"],
            random_state=seed,
        ),
    }

    ####################################################
    # Compare Models
    ####################################################

    results = []
    labels_map = {}

    for name, model in models.items():

        labels = model.fit_predict(X)

        labels_map[name] = labels

        m = _metrics(X, labels)

        m["model"] = name

        results.append(m)

        logger.info(
            f"{name:15s} | "
            f"sil={m['silhouette']:.3f} | "
            f"db={m['davies_bouldin']:.3f} | "
            f"k={m['n_clusters']}"
        )

    ####################################################
    # Results Table
    ####################################################

    df_cmp = pd.DataFrame(results)

    df_cmp = df_cmp[
        [
            "model",
            "n_clusters",
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
        ]
    ]

    df_cmp = df_cmp.sort_values(
        "silhouette",
        ascending=False,
    ).reset_index(drop=True)

    df_cmp["rank"] = df_cmp.index + 1

    best_name = df_cmp.iloc[0]["model"]

    best_labels = labels_map[best_name]

    if -1 in best_labels:

        majority = pd.Series(
            best_labels[best_labels != -1]
        ).mode()[0]

        best_labels[best_labels == -1] = majority

    ####################################################
    # Save outputs
    ####################################################

    _plot_comparison(df_cmp)

    df_cmp.to_csv(
        "outputs/reports/model_comparison.csv",
        index=False,
    )

    logger.info(
        f"Best model: {best_name} "
        f"(silhouette={df_cmp.iloc[0]['silhouette']})"
    )

    return df_cmp, best_labels, best_name


def _plot_comparison(df):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
    )

    fig.suptitle(
        "Clustering Algorithm Comparison",
        fontsize=14,
        fontweight="bold",
    )

    colors = [
        "#4C72B0",
        "#DD8452",
        "#55A868",
        "#C44E52",
    ]

    metrics = [
        ("silhouette", "Silhouette Score"),
        ("davies_bouldin", "Davies-Bouldin"),
        ("calinski_harabasz", "Calinski-Harabasz"),
    ]

    for ax, (metric, title) in zip(axes, metrics):

        bars = ax.bar(
            df["model"],
            df[metric],
            color=colors[:len(df)],
        )

        ax.set_title(title)

        ax.tick_params(
            axis="x",
            rotation=20,
        )

        for bar, value in zip(bars, df[metric]):

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()

    plt.savefig(
        "outputs/figures/model_comparison.png",
        dpi=150,
    )

    plt.close()

    logger.info("Model comparison chart saved.")
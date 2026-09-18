from dataclasses import dataclass, field
from typing import Optional, Sequence, Mapping

import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go

DEFAULT_COLORS = ['#F6F926', '#A777F1']


@dataclass
class UmapClusteringResult:
    df_clust: pd.DataFrame
    clusters: np.ndarray
    proj_2d: np.ndarray
    proj_3d: np.ndarray
    fig_2d: go.Figure
    fig_3d: go.Figure
    kmeans: KMeans
    umap_2d_model: UMAP
    umap_3d_model: UMAP
    clusters_list: list = field(default_factory=list)


def run_umap_clustering(
    df_red: pd.DataFrame,
    n_clusters: int = 2,
    random_state: int = 42,
    n_init: int = 40,
    hover_prefix: str = "Participant",
    title_suffix: str = "",
    colors: Optional[Sequence[str]] = None,
    umap_init: str = "random",
    show: bool = True,
    group: Optional[pd.Series] = None,
) -> UmapClusteringResult:

    X = df_red.values


    if colors is None:
        colors = list(DEFAULT_COLORS)
    if len(colors) < n_clusters:
        extra = px.colors.qualitative.Plotly
        colors = list(colors) + [c for c in extra if c not in colors]
    colors = colors[:n_clusters]

    # clustering 
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    clusters = kmeans.fit_predict(X)

    #UMAP 2D et 3d
    umap_2d_model = UMAP(n_components=2, init=umap_init, random_state=random_state)
    umap_3d_model = UMAP(n_components=3, init=umap_init, random_state=random_state)

    proj_2d = umap_2d_model.fit_transform(X)
    proj_3d = umap_3d_model.fit_transform(X)

    hover_labels = [f"{hover_prefix} {i}" for i in df_red.index]

    title_2d = f"UMAP 2D{' - ' + title_suffix if title_suffix else ''}"
    title_3d = f"UMAP 3D{' - ' + title_suffix if title_suffix else ''}"

    # gestion du group (0/1) 
    symbol_arg = None
    symbol_map_arg = None
    if group is not None:
        group_aligned = group.reindex(df_red.index)
        if group_aligned.isna().any():
            missing = group_aligned[group_aligned.isna()].index.tolist()
            raise ValueError(f"'group' contient des NaN pour ces index : {missing}")
        symbol_arg = group_aligned.astype(str)
        symbol_map_arg = {"0": "circle", "1": "diamond"}

    fig_2d = px.scatter(
        proj_2d, x=0, y=1,
        color=clusters.astype(str), labels={'color': 'Cluster'},
        symbol=symbol_arg,
        symbol_map=symbol_map_arg,
        hover_name=hover_labels,
        color_discrete_sequence=colors,
    )
    fig_2d.update_layout(title=title_2d)

    fig_3d = px.scatter_3d(
        proj_3d, x=0, y=1, z=2,
        color=clusters.astype(str), labels={'color': 'Cluster'},
        symbol=symbol_arg,
        symbol_map=symbol_map_arg,
        hover_name=hover_labels,
        color_discrete_sequence=colors,
    )
    fig_3d.update_traces(marker_size=5)
    fig_3d.update_layout(title=title_3d)

    if show:
        fig_2d.show(renderer="browser")
        fig_3d.show(renderer="browser")

   
    df_clust = df_red.copy()
    df_clust['cluster'] = clusters
    if group is not None:
        df_clust['group'] = group.reindex(df_red.index)

    clusters_list = [
        df_clust[df_clust['cluster'] == c] for c in range(n_clusters)
    ]

    print(df_clust['cluster'].value_counts().sort_index())
    for i, d in enumerate(clusters_list):
        msg = f"Cluster {i}: {len(d)} participants"
        if 'group' in df_clust.columns:
            counts = d['group'].value_counts().sort_index()
            detail = ", ".join(f"group {g}: {n}" for g, n in counts.items())
            msg += f" ({detail})"
        print(msg)

    return UmapClusteringResult(
        df_clust=df_clust,
        clusters=clusters,
        proj_2d=proj_2d,
        proj_3d=proj_3d,
        fig_2d=fig_2d,
        fig_3d=fig_3d,
        kmeans=kmeans,
        umap_2d_model=umap_2d_model,
        umap_3d_model=umap_3d_model,
        clusters_list=clusters_list,
    )


def split_umap_clustering_train_test(
    df_red_sains: pd.DataFrame,
    n_clusters: int = 2,
    test_size: float = 0.25,
    random_state: int = 42,
    n_init: int = 40,
    umap_init: str = "random",
    group: Optional[pd.Series] = None,
    stratify_on_group: bool = False,
    **kwargs,
) -> dict:
    """
    Split AVANT fit : KMeans + UMAP sont appris uniquement sur le train.
    Le test set est projeté/prédit via transform/predict, jamais refit.

    Ceci évite dataleakage  : dans run_umap_clustering() seul, UMAP et
    KMeans sont fit sur l'échantillon entier, donc un split train/test fait
    APRÈS coup sur proj_2d/clusters est biaisé (les points "test" ont influencé
    l'embedding et les centroïdes vus par "train", et inversement).

    """
    stratify_arg = None
    if stratify_on_group:
        if group is None:
            raise ValueError("stratify_on_group=True nécessite de fournir `group`.")
        stratify_arg = group.reindex(df_red_sains.index)

    idx_train, idx_test = train_test_split(
        df_red_sains.index,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_arg,
    )

    df_train = df_red_sains.loc[idx_train]
    df_test = df_red_sains.loc[idx_test]

    group_train = group.loc[idx_train] if group is not None else None
    group_test = group.loc[idx_test] if group is not None else None

    # Fit clustering + UMAP UNIQUEMENT sur train
    res_train = run_umap_clustering(
        df_train,
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        umap_init=umap_init,
        group=group_train,
        show=False,
        **kwargs,
    )

    # appliquer (transform/predict) sur test, sans refit
    X_test = df_test.values
    clusters_test = res_train.kmeans.predict(X_test)
    proj_2d_test = res_train.umap_2d_model.transform(X_test)
    proj_3d_test = res_train.umap_3d_model.transform(X_test)

    df_test_clust = df_test.copy()
    df_test_clust["cluster"] = clusters_test
    if group_test is not None:
        df_test_clust["group"] = group_test

    print("\n=== Répartition train (fit) ===")
    print(pd.Series(res_train.clusters).value_counts().sort_index())
    print("\n=== Répartition test (via predict, pas refit) ===")
    print(pd.Series(clusters_test).value_counts().sort_index())

    return {
        "res_train": res_train,        
        "df_train": df_train,
        "df_test": df_test_clust,
        "y_train": res_train.clusters,
        "y_test": clusters_test,
        "X_train_2d": res_train.proj_2d,
        "X_test_2d": proj_2d_test,
        "X_train_3d": res_train.proj_3d,
        "X_test_3d": proj_3d_test,
    }


def summarize_clusters(
    df_clust: pd.DataFrame,
    variables: Mapping[str, pd.Series],
    cluster_col: str = "cluster",
) -> pd.DataFrame:

    clusters_list = [
        df_clust[df_clust[cluster_col] == c] for c in sorted(df_clust[cluster_col].unique())
    ]

    resume = pd.DataFrame({
        f"Cluster {c}": pd.Series({
            name: series.reindex(d.index).mean() for name, series in variables.items()
        })
        for c, d in zip(sorted(df_clust[cluster_col].unique()), clusters_list)
    })
    return resume
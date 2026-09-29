from dataclasses import dataclass, field
from typing import Optional, Sequence, Mapping

import numpy as np
import pandas as pd
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go

DEFAULT_COLORS = [ '#FC6955', '#A777F1']

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


 
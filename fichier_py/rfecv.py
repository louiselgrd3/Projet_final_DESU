"""
le RFECV (étape 3) et la comparaison par sous-groupe
(étape 6) utilisent les 40 items BRUTS, pas le df réduit par corrélation.
Seul le clustering UMAP (étapes 1, 2, 4) reste basé sur l'espace réduit,
déjà validé comme donnant un clustering apprenable.

UMAP + KMeans sont fit UNE SEULE FOIS sur l'ensemble des sains (pas de
split avant fit) ; le split train/test n'intervient qu'après coup, sur
les projections/labels déjà calculés, uniquement pour évaluer le
classifieur et le RFECV.
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
import plotly.express as px

from testclust import run_umap_clustering
from pipeline_ML import train_and_compare_classifiers, apply_to_patients


# ---------------------------------------------------------------------
# ETAPE 1 — UMAP global (fit sur TOUS les sains) + split train/test
# APRÈS coup sur les projections/labels, pour le classifieur/RFECV
# ---------------------------------------------------------------------
def build_global_umap_and_clusters(df_sains_red, group, n_clusters=2,
                                    test_size=0.25, random_state=42):
    result = run_umap_clustering(
        df_sains_red, n_clusters=n_clusters, random_state=random_state,
        group=group, show=False,
    )

    X_all = result.proj_2d
    y_all = result.clusters
    idx_all = df_sains_red.index

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_all, y_all, idx_all,
        test_size=test_size, random_state=random_state, stratify=y_all,
    )

    return {
        "result": result,           # UMAP+KMeans fit sur tous les sains
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "idx_train": idx_train, "idx_test": idx_test,
    }


# ---------------------------------------------------------------------
# ETAPE 2 — Sanity check : le clustering est-il "apprenable" depuis
# l'espace UMAP 2D ? (reprend ta logique existante)
# ---------------------------------------------------------------------
def evaluate_umap_classifier(umap_result, scoring="balanced_accuracy"):
    X_train, X_test = umap_result["X_train"], umap_result["X_test"]
    y_train, y_test = umap_result["y_train"], umap_result["y_test"]

    results, summary, best_model_name, best_model = train_and_compare_classifiers(
        X_train, y_train, X_test, y_test, scoring=scoring
    )
    print(f"[UMAP 2D] Meilleur modèle : {best_model_name}")
    print(summary)
    return best_model, best_model_name, summary


# ---------------------------------------------------------------------
# ETAPE 3 — RFECV sur les ITEMS BRUTS (pas les coordonnées UMAP)
# pour savoir lesquels expliquent le mieux le label de cluster
# ---------------------------------------------------------------------
def run_rfecv_on_items(umap_result, df_items_40, min_features_to_select=3, random_state=42):
    """
    df_items_40 : le df des SAINS avec les 40 items TSQ bruts (ex: df_TSQ),
    PAS le df réduit par corrélation. On récupère les mêmes participants
    (mêmes index) que ceux utilisés dans le split train/test du classifieur
    UMAP, mais avec l'ensemble complet des 40 items comme features candidates.
    """
    train_idx = umap_result["idx_train"]
    test_idx = umap_result["idx_test"]

    # On exclut tout ce qui n'est pas un item brut : score_total est une
    # somme d'items -> le garder créerait une fuite de données (il prédirait
    # le cluster quasi parfaitement puisqu'il en dérive directement).
    colonnes_a_exclure = ["score_total", "group", "cluster", "predicted_cluster"]

    X_items_train = df_items_40.loc[train_idx].drop(columns=colonnes_a_exclure, errors="ignore")
    X_items_test = df_items_40.loc[test_idx].drop(columns=colonnes_a_exclure, errors="ignore")
    X_items_test = X_items_test[X_items_train.columns]  # même colonnes, même ordre

    y_train = umap_result["y_train"]
    y_test = umap_result["y_test"]

    estimator = RandomForestClassifier(
        n_estimators=300, random_state=random_state, class_weight="balanced"
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    rfecv = RFECV(
        estimator=estimator,
        step=1,
        cv=cv,
        scoring="balanced_accuracy",
        min_features_to_select=min_features_to_select,
        n_jobs=-1,
    )
    rfecv.fit(X_items_train, y_train)

    selected_items = list(X_items_train.columns[rfecv.support_])
    test_score = balanced_accuracy_score(y_test, rfecv.predict(X_items_test))

    importances = pd.Series(
        rfecv.estimator_.feature_importances_, index=selected_items
    ).sort_values(ascending=False)

    idx_best = rfecv.n_features_ - min_features_to_select
    cv_score_best = rfecv.cv_results_["mean_test_score"][idx_best]

    print(f"Nombre optimal d'items : {rfecv.n_features_}")
    print(f"Score CV (balanced accuracy) : {cv_score_best:.3f}")
    print(f"Score test (items sélectionnés) : {test_score:.3f}")
    print("\nItems sélectionnés, par importance décroissante :")
    print(importances)

    return {
        "rfecv": rfecv,
        "selected_items": selected_items,
        "importances": importances,
        "cv_score": cv_score_best,
        "test_score": test_score,
    }


def plot_rfecv_curve(rfecv_result, min_features_to_select=3):
    rfecv = rfecv_result["rfecv"]
    n_scores = len(rfecv.cv_results_["mean_test_score"])
    x = list(range(min_features_to_select, min_features_to_select + n_scores))

    fig = px.line(
        x=x, y=rfecv.cv_results_["mean_test_score"],
        labels={"x": "Nombre d'items", "y": "Score CV (balanced accuracy)"},
        title="RFECV — score en fonction du nombre d'items retenus",
    )
    fig.add_vline(x=rfecv.n_features_, line_dash="dash", line_color="red")
    fig.show(renderer="browser")


# ---------------------------------------------------------------------
# ETAPE 4 — Assigner un cluster aux patients (comme dans ta pipeline
# actuelle) via le UMAP fit sur les sains + le meilleur classifieur UMAP-2D
# ---------------------------------------------------------------------
def predict_patients_clusters(best_model, umap_result, df_pd_red):
    df_patients_pred = apply_to_patients(
        best_model,
        umap_result["result"].umap_2d_model,   # fit sur TOUS les sains
        df_pd_red,
    )
    return df_patients_pred


# ---------------------------------------------------------------------
# ETAPE 5 — Sous-groupes diagnostiques
# Tu as déjà défini ces groupes toi-même via iloc sur df_PD (voir
# run_full_analysis ci-dessous : ils y sont recréés à partir du df_PD
# que tu lui passes en argument).
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# ETAPE 6 — Comparer, pour un sous-groupe diagnostique donné, les items
# RFECV (data-driven) vs les items a priori Northoff (théoriques)
# ---------------------------------------------------------------------
def compare_subgroup_with_apriori(df_patients_items, df_patients_pred,
                                   patient_ids, rfecv_result, items_apriori,
                                   label="MDD"):
    common_idx = df_patients_items.index.intersection(patient_ids)
    sub_items = df_patients_items.loc[common_idx]
    sub_pred = df_patients_pred.loc[
        df_patients_pred.index.intersection(patient_ids), "predicted_cluster"
    ]

    print(f"\n{'='*20} Sous-groupe {label} (n={len(sub_items)}) {'='*20}")
    print(sub_pred.value_counts().sort_index())

    if sub_pred.nunique() < 2:
        print("Un seul cluster représenté dans ce sous-groupe : "
              "pas de comparaison inter-cluster possible.")
        return None

    # Différence de moyenne par item entre les 2 clusters, dans ce sous-groupe
    mean_by_cluster = sub_items.groupby(sub_pred).mean().T
    mean_by_cluster["diff_abs"] = (mean_by_cluster[0] - mean_by_cluster[1]).abs()
    mean_by_cluster = mean_by_cluster.sort_values("diff_abs", ascending=False)

    selected_items = set(rfecv_result["selected_items"])

    recap = pd.DataFrame({
        "item": mean_by_cluster.index,
        "diff_moyenne_clusters": mean_by_cluster["diff_abs"].values,
        "selectionne_par_RFECV": [i in selected_items for i in mean_by_cluster.index],
    })

    # Pas de liste a priori pour un groupe comorbide / mixte -> on s'arrête
    # à la comparaison RFECV + différences de moyennes, sans confrontation théorique
    if items_apriori is None:
        recap = recap.sort_values("diff_moyenne_clusters", ascending=False)
        print("\nPas de liste a priori pour ce sous-groupe (comorbidités multiples). "
              "Items triés par différence de moyenne entre clusters :")
        print(recap.head(10))
        return recap

    apriori_items = set(items_apriori)
    recap["item_apriori_Northoff"] = [i in apriori_items for i in recap["item"]]

    convergence = recap[recap["selectionne_par_RFECV"] & recap["item_apriori_Northoff"]]
    rfecv_only = recap[recap["selectionne_par_RFECV"] & ~recap["item_apriori_Northoff"]]
    apriori_only = recap[~recap["selectionne_par_RFECV"] & recap["item_apriori_Northoff"]]

    print(f"\nItems qui CONVERGENT (RFECV ET a priori Northoff) : {len(convergence)}")
    print(convergence["item"].tolist())
    print(f"\nItems retenus par RFECV mais absents de la liste a priori : {len(rfecv_only)}")
    print(rfecv_only["item"].tolist())
    print(f"\nItems a priori Northoff mais non retenus par RFECV : {len(apriori_only)}")
    print(apriori_only["item"].tolist())

    return recap



def run_full_analysis(df_PD, df_sains_40, df_red_sains, df_pd_40, df_red_pd, group,
                       items_core_sz, items_core_ma, items_core_an, items_core_md):
    """
    df_sains_40 : df des SAINS avec les 40 items bruts (ex: df_TSQ) -> utilisé
                  pour le RFECV et les comparaisons par sous-groupe.
    df_red_sains : df des SAINS réduit par corrélation -> utilisé UNIQUEMENT
                   pour fitter l'UMAP + le clustering (espace déjà validé).
    df_pd_40 : df des PATIENTS avec les 40 items bruts (ex: df_TSQ_PD) ->
               utilisé pour la comparaison RFECV / a priori par sous-groupe.
    df_red_pd : df des PATIENTS réduit par corrélation -> utilisé UNIQUEMENT
                pour projeter les patients dans l'UMAP (même dimensionnalité
                que celle utilisée au fit, donc ne peut pas être remplacé
                par les 40 items ici).
    """
    # 1. UMAP + clustering, fit sur TOUS les sains (espace réduit, déjà validé)
    umap_result = build_global_umap_and_clusters(df_red_sains, group)

    # 2. Sanity check : le clustering est-il apprenable via l'espace UMAP 2D ?
    best_model, best_model_name, summary = evaluate_umap_classifier(umap_result)

    # 3. RFECV sur les 40 items BRUTS (pas le df réduit) -> quels items
    #    expliquent le mieux le même label de cluster ?
    rfecv_result = run_rfecv_on_items(umap_result, df_sains_40)
    plot_rfecv_curve(rfecv_result)

    # 4. Assigner un cluster prédit à TOUS les patients diag
    df_patients_pred = predict_patients_clusters(best_model, umap_result, df_red_pd)

    # 5. Sous-groupes diagnostiques déjà définis par l'utilisateur (via iloc sur df_PD)
    sz = df_PD.iloc[[10]]
    man = df_PD.iloc[[4, 21, 22, 26, 29, 30]]
    ax = df_PD.iloc[[1, 12, 13, 15, 24]]
    dep = df_PD.iloc[[2, 3, 5, 6, 7, 8, 9, 16, 18, 19, 28]]
    comor = df_PD.iloc[[0, 11, 14, 17, 20, 23, 25, 27]]

    for nom, sous_df in {"SZ": sz, "MANIE": man, "ANX": ax, "MDD": dep, "COMOR": comor}.items():
        print(f"\n{'='*15} {nom} (n={len(sous_df)}) {'='*15}")
        print(sous_df.iloc[:, 18])

    # 6. Comparer RFECV (data-driven) vs a priori Northoff, groupe par groupe
    #    NB : pas de liste a priori pour "comor" (comorbidités multiples) -> items_apriori=None
    groupes = {
        "SZ": (sz.index, items_core_sz),
        "MANIE": (man.index, items_core_ma),
        "ANX": (ax.index, items_core_an),
        "MDD": (dep.index, items_core_md),
        "COMOR": (comor.index, None),
    }

    recaps = {}
    for label, (patient_ids, items_apriori) in groupes.items():
        recaps[label] = compare_subgroup_with_apriori(
            df_patients_items=df_pd_40,
            df_patients_pred=df_patients_pred,
            patient_ids=patient_ids,
            rfecv_result=rfecv_result,
            items_apriori=items_apriori,
            label=label,
        )

    return {
        "umap_result": umap_result,
        "best_model": best_model,
        "rfecv_result": rfecv_result,
        "df_patients_pred": df_patients_pred,
        "recaps": recaps,
    }

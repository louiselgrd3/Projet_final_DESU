"""
Pipeline exploratoire (data-driven) :
UMAP global (items réduits, sans présupposer de sous-échelle) + clustering
chez les sains, puis RFECV sur les items bruts pour identifier lesquels
expliquent le mieux la séparation des 2 clusters, et enfin comparaison,
sous-groupe diagnostique par sous-groupe diagnostique (ex: dépressifs),
avec les listes a priori "core" de Northoff (items_core_md, items_core_sz, ...).

Logique :
  1. UMAP + KMeans (2 clusters) fit UNIQUEMENT sur les sains, sur l'espace
     des items réduits (post-corrélation), donc SANS présupposer quels
     items appartiennent à quel trouble.
  2. Sanity check : un classifieur sur les 2 coordonnées UMAP retrouve-t-il
     bien les 2 clusters ? (déjà fait dans ta pipeline actuelle)
  3. RFECV sur les ITEMS BRUTS (pas les coords UMAP) -> y = label de cluster
     des sains. Ça donne la liste des items qui expliquent le mieux le
     clustering, indépendamment de toute théorie.
  4. Application du modèle UMAP-2D existant aux patients diag pour leur
     assigner un cluster prédit (comme tu le fais déjà).
  5. Récupération des sous-groupes diagnostiques déjà constitués par
     l'utilisateur (sz, man, ax, dep, comor), via leur index participant.
  6. Comparaison, pour chaque sous-groupe, entre :
       - les items sélectionnés par RFECV (data-driven, global)
       - les items a priori Northoff pour ce trouble (théorique)
     -> table de convergence / divergence.

Pré-requis censés déjà exister dans ton notebook :
  df_PD, df_red_sains, df_red_pd, group,
  items_core_md / items_core_sz / items_core_ma / items_core_an / items_core_pt,
  et les fonctions testclust.split_umap_clustering_train_test,
  pipeline_ML.train_and_compare_classifiers, pipeline_ML.apply_to_patients
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
import plotly.express as px

from testclust import split_umap_clustering_train_test
from pipeline_ML import train_and_compare_classifiers, apply_to_patients


# ---------------------------------------------------------------------
# ETAPE 1 — UMAP global + clustering chez les sains
# (identique à ta cellule 9, isolée ici en fonction pour être réutilisable)
# ---------------------------------------------------------------------
def build_global_umap_and_clusters(df_sains_red, group, n_clusters=2,
                                    test_size=0.25, random_state=42):
    split_result = split_umap_clustering_train_test(
        df_sains_red, n_clusters=n_clusters, test_size=test_size,
        random_state=random_state, group=group,
    )
    return split_result


# ---------------------------------------------------------------------
# ETAPE 2 — Sanity check : le clustering est-il "apprenable" depuis
# l'espace UMAP 2D ? (reprend ta logique existante)
# ---------------------------------------------------------------------
def evaluate_umap_classifier(split_result, scoring="balanced_accuracy"):
    X_train, X_test = split_result["X_train_2d"], split_result["X_test_2d"]
    y_train, y_test = split_result["y_train"], split_result["y_test"]

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
def run_rfecv_on_items(split_result, min_features_to_select=3, random_state=42):
    X_items_train = split_result["df_train"].drop(columns=["group"], errors="ignore")
    X_items_test = split_result["df_test"].drop(columns=["group"], errors="ignore")
    y_train = split_result["y_train"]
    y_test = split_result["y_test"]

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
def predict_patients_clusters(best_model, split_result, df_pd_red):
    df_patients_pred = apply_to_patients(
        best_model,
        split_result["res_train"].umap_2d_model,
        df_pd_red,
    )
    return df_patients_pred


# ---------------------------------------------------------------------
# ETAPE 5 — Sous-groupes diagnostiques
# Tu as déjà défini ces groupes toi-même via iloc sur df_PD. On les
# reprend tels quels ici : chaque variable est un sous-df de df_PD,
# et .index donne les identifiants participants à réutiliser plus loin.
# ---------------------------------------------------------------------
sz = df_PD.iloc[[10]]
man = df_PD.iloc[[4, 21, 22, 26, 29, 30]]
ax = df_PD.iloc[[1, 12, 13, 15, 24]]
dep = df_PD.iloc[[2, 3, 5, 6, 7, 8, 9, 16, 18, 19, 28]]
comor = df_PD.iloc[[0, 11, 14, 17, 20, 23, 25, 27]]

# Petit récap pour vérifier visuellement que chaque groupe correspond
# bien au diagnostic attendu (colonne 18 = diagnostic déclaré)
for nom, sous_df in {"SZ": sz, "MANIE": man, "ANX": ax, "MDD": dep, "COMOR": comor}.items():
    print(f"\n{'='*15} {nom} (n={len(sous_df)}) {'='*15}")
    print(sous_df.iloc[:, 18])


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


def run_full_pipeline(df_red_sains, group, df_red_pd, sz, man, ax, dep, comor,
                      items_core_sz, items_core_ma, items_core_an, items_core_md):
    split_result = build_global_umap_and_clusters(df_red_sains, group)
    best_model, _, _ = evaluate_umap_classifier(split_result)
    rfecv_result = run_rfecv_on_items(split_result)
    df_patients_pred = predict_patients_clusters(best_model, split_result, df_red_pd)

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
            df_red_pd, df_patients_pred, patient_ids,
            rfecv_result, items_apriori, label
        )
    return recaps

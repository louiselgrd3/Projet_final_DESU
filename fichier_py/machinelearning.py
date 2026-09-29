from dataclasses import dataclass
from typing import Optional, Mapping, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    learning_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------------------------
#  Modèles + grilles d'hyperparamètres
# ---------------------------------------------------------------------------

def get_models_and_grids(random_state: int = 42) -> Mapping[str, Mapping[str, Any]]:
    
    return {
        "LogisticRegression": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000,class_weight=None,  random_state=random_state)),
            ]),
            "param_grid": {
                "clf__C": [0.01, 0.1, 1, 10, 100],
                "clf__penalty": ["l2"],
            },
        },
        "SVM_RBF": {
            "pipeline": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="rbf",class_weight=None, probability=True, random_state=random_state)),
            ]),
            "param_grid": {
                "clf__C": [0.1, 1, 10, 100],
                "clf__gamma": ["scale", "auto", 0.01, 0.1, 1],
            },
        },
        "RandomForest": {
            "pipeline": Pipeline([
                ("clf", RandomForestClassifier(random_state=random_state, class_weight=None)),
            ]),
            "param_grid": {
                "clf__n_estimators": [100, 300, 500],
                "clf__max_depth": [None, 3, 5],
                "clf__min_samples_leaf": [5, 10],
            },
        },
        "HistGradientBoosting": {
            "pipeline": Pipeline([
                ("clf", HistGradientBoostingClassifier(random_state=random_state, class_weight=None)),
            ]),
            "param_grid": {
                "clf__max_iter": [100, 200, 400],
                "clf__max_depth": [None, 3, 5],
                "clf__learning_rate": [0.01, 0.05, 0.1],
            },
        },
    }



def plot_learning_curve(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    title: str,
    scoring: str = "accuracy",
    random_state: int = 42,
) -> None:
    """Affiche la learning curve (train vs validation CV) pour un estimateur donné."""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y,
        cv=cv,
        scoring=scoring,
        train_sizes=np.linspace(0.2, 1.0, 5),
        random_state=random_state,
        shuffle=True,
    )
    train_mean, train_std = train_scores.mean(axis=1), train_scores.std(axis=1)
    val_mean, val_std = val_scores.mean(axis=1), val_scores.std(axis=1)

    plt.figure(figsize=(5, 4))
    plt.plot(train_sizes, train_mean, 'o-', label="Train")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.plot(train_sizes, val_mean, 'o-', label="Validation (CV)")
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)
    plt.title(f"Learning curve — {title}")
    plt.xlabel("Taille du train set")
    plt.ylabel(scoring)
    plt.legend()
    plt.tight_layout()
    plt.show()


@dataclass
class ClassifierResult:
    best_estimator: Any
    cv_score: float
    cv_std: float          
    test_score: float
    best_params: dict

def train_and_compare_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    models_and_grids: Optional[Mapping[str, Mapping[str, Any]]] = None,
    scoring: str = "accuracy",
    n_splits: int = 5,#10 trop pour si peu de données risque de faire du bruit d'echantillonage
    random_state: int = 42,
    plot_curves: bool = True,
) -> Tuple[Mapping[str, ClassifierResult], pd.DataFrame, str, Any]:
    """
    Entraîne chaque modèle avec GridSearchCV (CV sur train), affiche sa learning
    curve, puis compare tous les modèles sur le score CV et le score test.

    scoring: "accuracy" par défaut. Utiliser "balanced_accuracy", "f1" ou "roc_auc"
    si les classes sont déséquilibrées.

    Retourne :
        results        : dict {nom_modele: ClassifierResult}
        summary        : DataFrame trié par score CV décroissant
        best_model_name: nom du meilleur modèle
        best_model     : meilleur estimateur (déjà fit)
    """
    if models_and_grids is None:
        models_and_grids = get_models_and_grids(random_state=random_state)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {}

    for name, cfg in models_and_grids.items():
        print(f"\n=== {name} ===")
        grid = GridSearchCV(
            cfg["pipeline"],
            cfg["param_grid"],
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        best_index = grid.best_index_
        cv_std = grid.cv_results_["std_test_score"][best_index]

        print("Meilleurs params :", grid.best_params_)
        print("Meilleur score CV :", round(grid.best_score_, 3), "± ", round(cv_std, 3))



        test_score = grid.score(X_test, y_test)
        print("Score sur test set :", round(test_score, 3))

        results[name] = ClassifierResult(
            best_estimator=grid.best_estimator_,
            cv_score=grid.best_score_,
            cv_std=cv_std,
            test_score=test_score,
            best_params=grid.best_params_,
        )

        if plot_curves:
            plot_learning_curve(
                grid.best_estimator_, X_train, y_train, cv, name,
                scoring=scoring, random_state=random_state,
            )

    summary = pd.DataFrame({
        name: {"CV score": r.cv_score, "CV std": r.cv_std, "Test score": r.test_score}
        for name, r in results.items()
    }).T.sort_values("CV score", ascending=False)

    print("\n=== Comparaison des modèles ===")
    print(summary)

    best_model_name = summary.index[0]
    best_model = results[best_model_name].best_estimator
    print(f"\nMeilleur modèle : {best_model_name}")

    y_pred_test = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_test))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test)
    plt.title(f"Matrice de confusion — {best_model_name} (test sains)")
    plt.show()

    return results, summary, best_model_name, best_model


# ---------------------------------------------------------------------------
#  Application aux patients (transform, jamais refit)
# ---------------------------------------------------------------------------

def apply_to_patients(
    best_model,
    umap_model,
    df_red_pd: pd.DataFrame,
) -> pd.DataFrame:
    """
    Projette les patients dans l'espace UMAP déjà appris sur les sains
    (umap_model.transform, PAS de refit), puis applique le classifieur.

    umap_model : le modèle UMAP fit sur les sains (ex: res_healthy.umap_2d_model
                 ou res_healthy.umap_3d_model, selon l'espace utilisé pour entraîner
                 le classifieur).
    """
    X_patients_umap = umap_model.transform(df_red_pd.values)

    y_pred_patients = best_model.predict(X_patients_umap)

    df_patients_pred = df_red_pd.copy()
    df_patients_pred["predicted_cluster"] = y_pred_patients

    if hasattr(best_model, "predict_proba"):
        y_proba_patients = best_model.predict_proba(X_patients_umap)[:, 1]
        df_patients_pred["proba_cluster1"] = y_proba_patients

    return df_patients_pred


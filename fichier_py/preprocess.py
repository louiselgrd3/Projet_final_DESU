import pandas as pd
from sklearn.impute import SimpleImputer

DATA_PATH = r"C:\TSQ\data\Projet_final_DESU\data2.csv"

COL_DIAGNOSTIC = "Avez-vous déjà reçu un diagnostic psychiatrique dans le passé ou actuellement ?"

# Lignes (index du CSV brut) exclues de l'analyse :
#   84  : diagnostic déclaré qui porte à confusion (pas une pathologie)
#   232 : HPI, ne fait pas partie des diagnostics retenus
#   338 : âge saisi aberrant (192006)
#   43, 91, 196, 299 : ont déclaré avoir déjà rempli le questionnaire
LIGNES_A_EXCLURE = [84, 232, 338, 43, 91, 196, 299]

# Colonnes techniques inutiles à supprimer dès le chargement
COLS_TECHNIQUES = ["Date de soumission", "Date de lancement", "Date de la dernière action", "Quota exit"]

DONNEES_SOC_DEMO = [
    'ID de la réponse', 'Dernière page', 'Langue de départ', 'Tête de série',
    'Être bien âgé entre 18 et 65 ans',
    'Avoir un niveau de maitrise suffisant de la langue française\xa0',
    'Être dans un moment calme et de disponibilité pour compléter le questionnaire',
    'Quel est votre âge ?', 'Comment vous identifiez-vous ?\xa0',
    'Est-ce la première fois que vous remplissez ce questionnaire?',
    'Êtes-vous étudiant (hors Université du Temps Libre) ?',
    'Quelle est votre situation ?', "Quelle est votre filière d'étude ?",
    "Quelle est votre filière d'étude ? [Autre]", 'Quelle est votre profession ?\xa0',
    'En quelle année êtes-vous ?', 'Quel est votre plus haut niveau de diplôme obtenu ?',
    COL_DIAGNOSTIC, 'Si oui, lequel (lesquels) ?\xa0',
    'Prenez-vous un traitement médicamenteux en lien avec ce diagnostic ?',
    'Avez-vous un proche (parent, enfant, fratrie) qui a reçu un diagnostic avéré de trouble psychique sévère ?',
    'Précisez le traitement médicamenteux actuel et son dosage.',
    'Avez-vous consommé des substances ce dernier mois (hors tabac, alcool et/ou traitements médicamenteux) ?',
    'À quelle fréquence ?', 'Quelle(s) substance(s) avez-vous consommé ?'
]

BORNES_BAI = [0, 7, 15, 25, 63]
LABELS_BAI = ["Minimale (0-7)", "Légère (8-15)", "Modérée (16-25)", "Sévère (26-63)"]

COL_TSQ_A_SUPPRIMER = "Nous vous remercions pour votre engagement et vous invitons à passer à la suite de ce questionnaire.\xa0"

# ZTPI : positions (dans le bloc de 56 colonnes) des 2 items exclus -> 54 items restants (car dans version francaise 54 items)
ZTPI_POS_A_EXCLURE = (15, 36)

# Nombre de colonnes attendu par échelle (vérification de sécurité)
N_ATTENDU = {"BAI": 21, "TSQ": 40, "EP": 7, "16items": 32, "sub": 5, "ZTPI": 54}

def _extract_num(df, cols):
    """Extrait le premier nombre de chaque réponse (ex: '3 - Souvent' -> 3.0)."""
    return df[cols].apply(lambda col: col.astype(str).str.extract(r"(\d+)")[0].astype(float))


def _check_n(cols, nom):
    if len(cols) != N_ATTENDU[nom]:
        raise ValueError(
            f"{nom} : {len(cols)} colonnes trouvées au lieu de {N_ATTENDU[nom]}. "
            "Les positions de colonnes ont probablement changé dans le nouveau CSV."
        )


def load_and_split(filepath=DATA_PATH, sep=",", encoding="utf-8", lignes_a_exclure=LIGNES_A_EXCLURE):
    """
    Charge le CSV, exclut les participants écartés, supprime les lignes sans date de
    soumission et les colonnes techniques, puis sépare les participants sans
    diagnostic (PS) et avec diagnostic (PD).

    Retourne : df, df_PS, df_PD
    """
    df = pd.read_csv(filepath, sep=sep, encoding=encoding)
    df = df.drop(df.index[lignes_a_exclure])
    df = df.dropna(subset=["Date de soumission"])
    df = df.drop(columns=COLS_TECHNIQUES, errors="ignore")

    df_PS = df[df[COL_DIAGNOSTIC] == "Non"].copy()
    df_PD = df[df[COL_DIAGNOSTIC] == "Oui"].copy()

    print(f"Shape : {df.shape} | PS : {len(df_PS)} | PD : {len(df_PD)} "
          f"({len(df_PD) / len(df) * 100:.1f}% avec diagnostic)")

    return df, df_PS, df_PD


def extract_socio_demo(df, df_PS, df_PD):
    
    df_SD = df[DONNEES_SOC_DEMO].copy()
    df_SD_PD = df_PD[DONNEES_SOC_DEMO].copy()
    df_SD_PS = df_PS[DONNEES_SOC_DEMO].copy()
    return df_SD, df_SD_PS, df_SD_PD


def clean_impute_socio_demo(df_SD, seuil=0.3, supprimer_colonnes_vides=False):
    
    df_SD = df_SD.copy()

    lignes_vides = df_SD.isnull().all(axis=1).sum()
    print(f"Nombre de lignes complètement vides : {lignes_vides}")

    if supprimer_colonnes_vides:
        df_SD = df_SD.dropna(axis=1, thresh=len(df_SD) * seuil)

    cols_num = df_SD.select_dtypes(include="number").columns
    cols_cat = df_SD.select_dtypes(exclude="number").columns

    if len(cols_num) > 0:
        imputer_num = SimpleImputer(strategy="mean")
        df_SD[cols_num] = imputer_num.fit_transform(df_SD[cols_num])

    imputer_cat = SimpleImputer(strategy="constant", fill_value="Inconnu", keep_empty_features=True)
    df_SD[cols_cat] = pd.DataFrame(
        imputer_cat.fit_transform(df_SD[cols_cat]),
        columns=cols_cat,
        index=df_SD.index
    )

    return df_SD


def process_bai(df_PS, col_start=25, col_end=46):
    cols_bai = df_PS.columns[col_start:col_end]
    _check_n(cols_bai, "BAI")
    df_BAI = df_PS[cols_bai].copy()

    df_BAI["score_total"] = df_BAI.sum(axis=1)

    df_BAI["categorie_anxiete"] = pd.cut(
        df_BAI["score_total"],
        bins=BORNES_BAI,
        labels=LABELS_BAI,
        include_lowest=True
    )

    return df_BAI


def process_bai_pd(df_PD, col_start=25, col_end=46):

    cols_bai = df_PD.columns[col_start:col_end]
    df_BAI_PD = df_PD[cols_bai].copy()
    df_BAI_PD["score_total"] = df_BAI_PD.sum(axis=1)
    return df_BAI_PD


def repartition_bai(df_BAI):
    return df_BAI["categorie_anxiete"].value_counts().reindex(LABELS_BAI)


def process_tsq(df_PS, df_PD, col_start=46, col_end=87):
    cols_TSQ = df_PS.columns[col_start:col_end]
    cols_TSQ_PD = df_PD.columns[col_start:col_end]

    df_TSQ = df_PS[cols_TSQ].copy()
    df_TSQ_PD = df_PD[cols_TSQ_PD].copy()

    df_TSQ = df_TSQ.drop(columns=[COL_TSQ_A_SUPPRIMER], errors="ignore")
    df_TSQ_PD = df_TSQ_PD.drop(columns=[COL_TSQ_A_SUPPRIMER], errors="ignore")
    _check_n(df_TSQ.columns, "TSQ")

    cols_items = df_TSQ.columns
    cols_items_pd = df_TSQ_PD.columns

    df_TSQ[cols_items] = _extract_num(df_TSQ, cols_items)
    df_TSQ_PD[cols_items_pd] = _extract_num(df_TSQ_PD, cols_items_pd)

    df_TSQ["score_total"] = df_TSQ[cols_items].sum(axis=1)
    df_TSQ_PD["score_total"] = df_TSQ_PD[cols_items_pd].sum(axis=1)

    return df_TSQ, df_TSQ_PD


def process_ep(df_PS, df_PD, col_start=87, col_end=94):
    cols_EP = df_PS.columns[col_start:col_end]
    cols_EP_PD = df_PD.columns[col_start:col_end]
    _check_n(cols_EP, "EP")

    df_EP = df_PS[cols_EP].copy()
    df_EP_PD = df_PD[cols_EP_PD].copy()

   
    df_EP[cols_EP] = _extract_num(df_EP, cols_EP)
    df_EP_PD[cols_EP_PD] = _extract_num(df_EP_PD, cols_EP_PD)

    df_EP["score_total"] = df_EP[cols_EP].sum(axis=1)
    df_EP_PD["score_total"] = df_EP_PD[cols_EP_PD].sum(axis=1)

    return df_EP, df_EP_PD



def process_16items(df_PS, df_PD, col_start=94, col_end=126):
  
    cols_16 = df_PS.columns[col_start:col_end]
    cols_16_PD = df_PD.columns[col_start:col_end]
    _check_n(cols_16, "16items")

    df_16 = df_PS[cols_16].copy()
    df_16_PD = df_PD[cols_16_PD].copy()

    echelle1 = cols_16[0::2]   # colonnes "vrai/faux"
    echelle2 = cols_16[1::2]   # colonnes "intensité"

    df_16["nb_vrai"] = (df_16[echelle1] == "VRAI").sum(axis=1)
    df_16_num = df_16[echelle2].apply(pd.to_numeric, errors="coerce")
    df_16["score_total"] = df_16_num.sum(axis=1, skipna=True)

    echelle1_pd = cols_16_PD[0::2]
    echelle2_pd = cols_16_PD[1::2]
    df_16_PD["nb_vrai"] = (df_16_PD[echelle1_pd] == "VRAI").sum(axis=1)
    df_16_PD_num = df_16_PD[echelle2_pd].apply(pd.to_numeric, errors="coerce")
    df_16_PD["score_total"] = df_16_PD_num.sum(axis=1, skipna=True)

    return df_16, df_16_PD, echelle1, echelle2



def process_sub(df_PS, col_start=126, col_end=131):
    cols_sub = df_PS.columns[col_start:col_end]
    _check_n(cols_sub, "sub")
    return df_PS[cols_sub].copy()


def process_ztpi(df_PS, df_PD, col_start=131, col_end=187, pos_a_exclure=ZTPI_POS_A_EXCLURE):
    cols_ZTPI = [c for i, c in enumerate(df_PS.columns[col_start:col_end]) if i not in pos_a_exclure]
    cols_ZTPI_PD = [c for i, c in enumerate(df_PD.columns[col_start:col_end]) if i not in pos_a_exclure]
    _check_n(cols_ZTPI, "ZTPI")

    df_ZTPI = df_PS[cols_ZTPI].copy()
    df_ZTPI_PD = df_PD[cols_ZTPI_PD].copy()

    df_ZTPI[cols_ZTPI] = _extract_num(df_ZTPI, cols_ZTPI)
    df_ZTPI_PD[cols_ZTPI_PD] = _extract_num(df_ZTPI_PD, cols_ZTPI_PD)

    df_ZTPI["score_total"] = df_ZTPI[cols_ZTPI].sum(axis=1)
    df_ZTPI_PD["score_total"] = df_ZTPI_PD[cols_ZTPI_PD].sum(axis=1)

    return df_ZTPI, df_ZTPI_PD



def run_full_pipeline(filepath=DATA_PATH, sep=",", encoding="utf-8"):
    df, df_PS, df_PD = load_and_split(filepath, sep=sep, encoding=encoding)

    df_SD, df_SD_PS, df_SD_PD = extract_socio_demo(df, df_PS, df_PD)
    df_SD_PS = clean_impute_socio_demo(df_SD_PS)

    df_BAI = process_bai(df_PS)
    df_BAI_PD = process_bai_pd(df_PD)
    df_TSQ, df_TSQ_PD = process_tsq(df_PS, df_PD)
    df_EP, df_EP_PD = process_ep(df_PS, df_PD)
    df_16, df_16_PD, echelle1, echelle2 = process_16items(df_PS, df_PD)
    df_sub = process_sub(df_PS)
    df_ZTPI, df_ZTPI_PD = process_ztpi(df_PS, df_PD)

    return {
        "df": df,
        "df_PS": df_PS,
        "df_PD": df_PD,
        "df_SD": df_SD,
        "df_SD_PS": df_SD_PS,
        "df_SD_PD": df_SD_PD,
        "df_BAI": df_BAI,
        "df_BAI_PD": df_BAI_PD,
        "df_TSQ": df_TSQ,
        "df_TSQ_PD": df_TSQ_PD,
        "df_EP": df_EP,
        "df_EP_PD": df_EP_PD,
        "df_16": df_16,
        "df_16_PD": df_16_PD,
        "echelle1": echelle1,
        "echelle2": echelle2,
        "df_sub": df_sub,
        "df_ZTPI": df_ZTPI,
        "df_ZTPI_PD": df_ZTPI_PD,
    }


if __name__ == "__main__":
    resultats = run_full_pipeline()
    print(repartition_bai(resultats["df_BAI"]))

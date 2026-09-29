import pandas as pd


def calculer_scores_tsq(df_TSQ, save_csv=False, output_path=None):
    """
    Calcule les scores normalisés du TSQ (core, périphérie, total, par dimension)
    à partir du DataFrame df_TSQ déjà issu de la pipeline de preprocessing
    (40 colonnes d'items + une colonne 'score_total').

    Retourne le DataFrame enrichi des scores par dimension.
    """
    df = df_TSQ.copy()

    # 1. Récupérer les 40 colonnes d'items (tout sauf score_total)
    cols_tsq_brutes = [c for c in df.columns if c != "score_total"]
    assert len(cols_tsq_brutes) == 40, f"Attendu 40 colonnes d'items, trouvé {len(cols_tsq_brutes)}"

    # 2. Les renommer en TSQ1 à TSQ40
    df = df.rename(columns=dict(zip(cols_tsq_brutes, [f"TSQ{i}" for i in range(1, 41)])))

    # 3. Convertir le texte ("5 - Un peu"...) en numérique
    items = [f"TSQ{i}" for i in range(1, 41)]
    df[items] = df[items].apply(lambda col: col.astype(str).str.extract(r"(\d+)")[0].astype(float))

    # Core items
    items_core_sz = ['TSQ2','TSQ4','TSQ7','TSQ8','TSQ10','TSQ11','TSQ12','TSQ15','TSQ19','TSQ20','TSQ21','TSQ24','TSQ30','TSQ31','TSQ34','TSQ40']
    items_core_md = ['TSQ9','TSQ14','TSQ17','TSQ25','TSQ28','TSQ29','TSQ36']
    items_core_ma = ['TSQ22','TSQ32','TSQ35','TSQ39']
    items_core_an = ['TSQ1','TSQ5','TSQ6','TSQ9','TSQ13','TSQ16','TSQ17','TSQ20','TSQ23','TSQ33','TSQ35','TSQ37','TSQ38','TSQ39']
    items_core_pt = ['TSQ2','TSQ3','TSQ5','TSQ6','TSQ17','TSQ18','TSQ26','TSQ30','TSQ37']

    # Periphery items
    items_per_sz = ['TSQ1','TSQ3','TSQ5','TSQ6','TSQ13','TSQ14','TSQ23','TSQ27','TSQ35','TSQ38','TSQ39']
    items_per_md = ['TSQ1','TSQ2','TSQ8','TSQ12','TSQ16','TSQ18','TSQ19','TSQ20','TSQ23','TSQ27','TSQ31']
    items_per_ma = ['TSQ3','TSQ7','TSQ9','TSQ19','TSQ21','TSQ30','TSQ34','TSQ37','TSQ38']
    items_per_an = ['TSQ2','TSQ3','TSQ12','TSQ18','TSQ19','TSQ22','TSQ25','TSQ27','TSQ32']
    items_per_pt = ['TSQ1','TSQ9','TSQ12','TSQ13','TSQ14','TSQ16','TSQ19','TSQ20','TSQ23','TSQ24','TSQ25','TSQ27','TSQ28','TSQ31','TSQ33','TSQ35','TSQ36','TSQ38','TSQ39','TSQ40']

    # Combined Core+Periphery items
    items_tot_sz = items_core_sz + items_per_sz
    items_tot_md = items_core_md + items_per_md
    items_tot_ma = items_core_ma + items_per_ma
    items_tot_an = items_core_an + items_per_an
    items_tot_pt = items_core_pt + items_per_pt

    df['grand_score'] = df[items].sum(axis=1) / (10 * 40)

    df['core_sz'] = df[items_core_sz].sum(axis=1) / (10 * len(items_core_sz))
    df['core_md'] = df[items_core_md].sum(axis=1) / (10 * len(items_core_md))
    df['core_ma'] = df[items_core_ma].sum(axis=1) / (10 * len(items_core_ma))
    df['core_an'] = df[items_core_an].sum(axis=1) / (10 * len(items_core_an))
    df['core_pt'] = df[items_core_pt].sum(axis=1) / (10 * len(items_core_pt))

    df['per_sz'] = df[items_per_sz].sum(axis=1) / (10 * len(items_per_sz))
    df['per_md'] = df[items_per_md].sum(axis=1) / (10 * len(items_per_md))
    df['per_ma'] = df[items_per_ma].sum(axis=1) / (10 * len(items_per_ma))
    df['per_an'] = df[items_per_an].sum(axis=1) / (10 * len(items_per_an))
    df['per_pt'] = df[items_per_pt].sum(axis=1) / (10 * len(items_per_pt))

    df['tot_sz'] = df[items_tot_sz].sum(axis=1) / (10 * len(items_tot_sz))
    df['tot_md'] = df[items_tot_md].sum(axis=1) / (10 * len(items_tot_md))
    df['tot_ma'] = df[items_tot_ma].sum(axis=1) / (10 * len(items_tot_ma))
    df['tot_an'] = df[items_tot_an].sum(axis=1) / (10 * len(items_tot_an))
    df['tot_pt'] = df[items_tot_pt].sum(axis=1) / (10 * len(items_tot_pt))

    if save_csv and output_path:
        df.to_csv(output_path, index=False)

    return df
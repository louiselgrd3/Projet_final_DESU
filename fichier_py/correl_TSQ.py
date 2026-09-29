import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def compute_tsq_corr(df, items, title="TSQ Correlation Matrix"):
   
    #  Matrice de corrélation Spearman 
    corr_matrix = df[items].corr(method='spearman')

    # p-values 
    n = len(items)
    pval_matrix = pd.DataFrame(np.ones((n, n)), index=items, columns=items)

    for i in range(n):
        for j in range(i + 1, n):
            item_i, item_j = items[i], items[j]
            paired = df[[item_i, item_j]].dropna()

            if len(paired) > 1:
                r, p = spearmanr(paired[item_i], paired[item_j])
            else:
                p = 1.0

            pval_matrix.loc[item_i, item_j] = p
            pval_matrix.loc[item_j, item_i] = p

    #  p-values → étoiles 
    def p_to_stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        return ""

    stars_matrix = pval_matrix.map(p_to_stars)

    # Combiner corr + étoiles
    annot_labels = corr_matrix.round(2).astype(str) + stars_matrix

    # Heatmap 
    plt.figure(figsize=(16, 14))
    sns.heatmap(
        corr_matrix,
        annot=annot_labels,
        fmt="",
        annot_kws={"size": 6},
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.3,
        cbar_kws={"shrink": 0.8}
    )
    plt.title(title + "\n* p<.05  ** p<.01  *** p<.001")
    plt.tight_layout()
    plt.show()

    # Extraire les corrélations fortes
    corr_pairs = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    strong_corr = corr_pairs.stack().reset_index()
    strong_corr.columns = ['item_1', 'item_2', 'correlation']
    strong_corr = strong_corr[strong_corr['correlation'] > 0.6].sort_values('correlation', ascending=False)

    return strong_corr

# greedy gloutton 
import numpy as np
import pandas as pd


def reduce_correlated_features(df, cols, threshold=0.7):

    cols = list(cols)  

    while True:
        
        corr_df = df[cols].corr().abs()

       
        corr_np = corr_df.to_numpy(copy=True)

        # On met la diagonale à zéro pour ignorer les auto-corrélations
        np.fill_diagonal(corr_np, 0)

        # Corrélation maximale dans la matrice
        max_corr = corr_np.max()

        # Si plus aucune corrélation ne dépasse le seuil → on s'arrête
        if max_corr <= threshold:
            break

        # Moyenne des corrélations par variable
        mean_corr = corr_np.mean(axis=0)

        # Indices de la paire la plus corrélée
        i, j = np.unravel_index(np.argmax(corr_np), corr_np.shape)
        var1, var2 = cols[i], cols[j]

        # On supprime la variable la plus "corrélée en moyenne"
        to_drop = var1 if mean_corr[i] > mean_corr[j] else var2
        cols.remove(to_drop)

    return cols

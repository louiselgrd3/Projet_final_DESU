import os
import numpy as np
import matplotlib.pyplot as plt

CATEGORIES = ['SZ', 'MDD', 'MAN', 'ANX', 'PTSD']


def _draw_polar_axis(ax, values, title):
    """Dessine un radar (polaire) sur un axe donné pour une série de valeurs normalisées."""
    values = list(values) + [values[0]]

    angles = np.linspace(0, 2 * np.pi, len(CATEGORIES), endpoint=False).tolist()
    angles.append(angles[0])

    ring_ticks = np.arange(0.1, 1.1, 0.1)
    for tick in ring_ticks:
        ax.plot(np.linspace(0, 2 * np.pi, 500), [tick] * 500,
                 color='gray', linestyle='dotted', linewidth=1.5, alpha=0.8)

    for i in range(len(CATEGORIES)):
        ax.plot([angles[i], angles[i]], [0, 1], color='gray', linewidth=1.5)

        normalized_ticks = np.linspace(0, 1, 6)
        for norm_tick in normalized_ticks:
            if norm_tick == 0:
                continue
            offset = 0.05
            ax.text(angles[i] + offset, norm_tick, f"{norm_tick:.2f}",
                     ha='center', va='center', fontsize=6,
                     color='gray', alpha=0.8, fontweight='bold')

    ax.plot(angles, values, linewidth=2.5, linestyle='solid', label='Normalized Data')
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CATEGORIES, fontsize=8)

    ax.spines['polar'].set_visible(False)
    ax.set_yticks([])

    ax.set_title(title, size=14, pad=10)


def polar_plots(df, output_dir="plots", show=False):
    """
    Génère et sauvegarde un graphique polaire (Core / Periphery / Total) pour
    chaque participant, à partir d'un DataFrame DÉJÀ enrichi des colonnes de
    scores (core_sz, per_sz, tot_sz, etc. — sortie de calculer_scores_tsq).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_subs = len(df)

    for subject in range(n_subs):
        row = df.iloc[subject, :]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5), subplot_kw=dict(polar=True))
        fig.suptitle(f'Subject {subject}', fontsize=16, y=1.05)

        valeurs_par_type = {
            'Core': [row['core_sz'], row['core_md'], row['core_ma'], row['core_an'], row['core_pt']],
            'Periphery': [row['per_sz'], row['per_md'], row['per_ma'], row['per_an'], row['per_pt']],
            'Total': [row['tot_sz'], row['tot_md'], row['tot_ma'], row['tot_an'], row['tot_pt']],
        }

        for ax, cor_per in zip(axs, ['Core', 'Periphery', 'Total']):
            _draw_polar_axis(ax, valeurs_par_type[cor_per], cor_per)

        plt.tight_layout()
        fig.suptitle(f'Subject {subject}', fontsize=16)
        plt.savefig(f'{output_dir}/Subject_{subject}_polar_plots.png', dpi=500)

        if show:
            plt.show()
        else:
            plt.close(fig)

    print(f"{n_subs} graphiques générés dans le dossier '{output_dir}'.")
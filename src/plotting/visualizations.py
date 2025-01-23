# src/plotting/visualizations.py

import matplotlib.pyplot as plt

def plot_medley_scores(feature_names, importances, title="MEDLEY Scores", outfile="MEDLEY_Score.png"):
    """
    Simple bar chart for feature importances.
    """
    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, importances, color='skyblue')
    plt.gca().invert_yaxis()
    plt.xlabel("MEDLEY Score")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.show()

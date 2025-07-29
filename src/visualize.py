# src/visualize.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_visualizations(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['average_score'], bins=20, kde=True)
    plt.title("Distribution of Average Scores")
    plt.xlabel("Average Score")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Data
df_boundary = pd.DataFrame({
    "knn": ["$50$k", "$125$k", "$250$k", "$500$k", "$ 750$k", "$1$M"],
    "speed-up": [ 12.1, 12.3,  10.6, 11.8,  13.9, 13.7]
    })

df_non_manifold = pd.DataFrame({
    "knn": ["$50$k", "$125$k", "$250$k", "$500$k", "$ 750$k", "$1$M"],
    "speed-up": [ 14.3, 15.1,  13.1, 12.9,  17.2, 20.5]
    })

# Styling
custom_palette = ["#6d597a", "#b56576", "#849DAB"]
cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", custom_palette)

# Normalize the KNN values for colormap mapping

df = df_non_manifold

n = len(df)
norm = mcolors.Normalize(vmin=0, vmax=1)
colors = [cmap(norm(i / (n - 1))) for i in range(n)]


# Seaborn theme
sns.set_theme(style="whitegrid", rc={
    "axes.facecolor": "#F3F3F3",
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{lmodern, bm}",
})

# Bar plot
plt.figure(figsize=(8, 4))
sns.barplot(data=df, x="knn", y="speed-up", palette=colors, edgecolor="black")

# Labels and annotations
plt.title("Generating Non-Manifold Edges: Speed-up of \\texttt{trueform} over VTK", fontsize=17)
plt.xlabel("Number of Triangles", fontsize=15)
plt.ylabel("Speed-up", fontsize=15)
plt.ylim(0, 23)
plt.yticks([0, 4, 8, 12])  # <- Set specific y-axis ticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid(False)

# Annotate bar tops with exact values
for i, val in enumerate(df["speed-up"]):
    plt.text(i, val + 0.05, f"{val:.2f}", ha='center', va='bottom', fontsize=15)

plt.tight_layout()
plt.savefig("../../docs/img/vtk_non_manifold_speed_up.png", format="png", dpi=300)



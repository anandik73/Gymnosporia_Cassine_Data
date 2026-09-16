"""
Step 3: Principal component analysis of the 19 bioclimatic variables, niche
centroid/breadth per species, and PERMANOVA tests of whether climate-space
position differs by genus and/or species.

Input : Results/occurrence_climate_CHELSA.csv (from Step 2)
Output: Results/occurrence_climate_CHELSA_withPCs.csv  (input data + PC scores)
        Results/pca_loadings.csv
        Results/niche_breadth_per_species.csv
        Results/permanova_results.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "Results")

df = pd.read_csv(f"{OUT}/occurrence_climate_CHELSA.csv")
biocols = [f"bio{i}" for i in range(1, 20)]
df["genus"] = df["species"].map(lambda s: s.split("_")[0])
ENDEMIC = {"Cassine_glauca": False, "Cassine_paniculata": True,
           "Gymnosporia_emarginata": False, "Gymnosporia_rothiana": True,
           "Gymnosporia_senegalensis": False}
df["endemic"] = df["species"].map(ENDEMIC)

# ---------------- PCA ----------------
Xs = StandardScaler().fit_transform(df[biocols].values)
pca = PCA(n_components=5)
pcs = pca.fit_transform(Xs)
for i in range(5):
    df[f"PC{i+1}"] = pcs[:, i]

print("Explained variance ratio:", np.round(pca.explained_variance_ratio_, 3))
print("Cumulative:", np.round(np.cumsum(pca.explained_variance_ratio_), 3))

loadings = pd.DataFrame(pca.components_[:3].T, index=biocols, columns=["PC1", "PC2", "PC3"])
loadings.to_csv(f"{OUT}/pca_loadings.csv")
df.to_csv(f"{OUT}/occurrence_climate_CHELSA_withPCs.csv", index=False)

# ---------------- niche centroid & breadth ----------------
species_list = sorted(df["species"].unique().tolist())
niche_rows = []
for sp in species_list:
    sub = df[df.species == sp][["PC1", "PC2", "PC3"]].values
    centroid = sub.mean(axis=0)
    dists = np.linalg.norm(sub - centroid, axis=1)
    niche_rows.append({"species": sp, "n": len(sub), "niche_breadth": dists.mean(),
                        "PC1_centroid": centroid[0], "PC2_centroid": centroid[1], "PC3_centroid": centroid[2]})
    print(f"{sp}: n={len(sub)}, niche breadth={dists.mean():.3f}")
pd.DataFrame(niche_rows).to_csv(f"{OUT}/niche_breadth_per_species.csv", index=False)

# ---------------- PERMANOVA ----------------
def permanova(coords, groups, n_perm=999, seed=42):
    groups = np.array(groups)
    n = len(groups)
    diff = coords[:, None, :] - coords[None, :, :]
    D2 = np.sum(diff ** 2, axis=-1)

    def compute_F(g):
        uniq = np.unique(g)
        SS_total = D2[np.triu_indices(n, 1)].sum() / n
        SS_within = 0
        for grp in uniq:
            idx = np.where(g == grp)[0]
            ng = len(idx)
            if ng > 1:
                sub = D2[np.ix_(idx, idx)]
                SS_within += sub[np.triu_indices(ng, 1)].sum() / ng
        SS_among = SS_total - SS_within
        F = (SS_among / (len(uniq) - 1)) / (SS_within / (n - len(uniq)))
        return F

    F_obs = compute_F(groups)
    rng = np.random.default_rng(seed)
    F_perm = np.array([compute_F(rng.permutation(groups)) for _ in range(n_perm)])
    pval = (np.sum(F_perm >= F_obs) + 1) / (n_perm + 1)
    return F_obs, pval

coords5 = df[["PC1", "PC2", "PC3", "PC4", "PC5"]].values
permanova_rows = []

F, p = permanova(coords5, df["genus"].values)
permanova_rows.append({"grouping": "Genus (all species)", "n": len(df), "F": F, "p": p, "groups": 2})
print(f"Genus: F={F:.2f}, p={p:.4f}")

F, p = permanova(coords5, df["species"].values)
permanova_rows.append({"grouping": "Species (all species)", "n": len(df), "F": F, "p": p, "groups": 5})
print(f"Species (all): F={F:.2f}, p={p:.4f}")

for g in ["Gymnosporia", "Cassine"]:
    sub = df[df.genus == g]
    F, p = permanova(sub[["PC1", "PC2", "PC3", "PC4", "PC5"]].values, sub["species"].values)
    n_groups = sub["species"].nunique()
    permanova_rows.append({"grouping": f"Species within {g}", "n": len(sub), "F": F, "p": p, "groups": n_groups})
    print(f"Species within {g}: F={F:.2f}, p={p:.4f}, n={len(sub)}")

pd.DataFrame(permanova_rows).to_csv(f"{OUT}/permanova_results.csv", index=False)
print(f"\nSaved pca_loadings.csv, niche_breadth_per_species.csv, permanova_results.csv to {OUT}")

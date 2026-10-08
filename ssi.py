import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("module_scores_3modules.csv")

module_cols = ["R_score", "D_score", "A_score"]
X = df[module_cols].values

sc = StandardScaler()
Z = sc.fit_transform(X)

pca = PCA(n_components=1, random_state=0)
sbi = pca.fit_transform(Z).ravel()

group = np.array(["CK"] * 6 + ["Cd1"] * 6 + ["Cd2"] * 6 + ["Cd3"] * 6)
if sbi[group == "Cd3"].mean() < sbi[group == "CK"].mean():
    sbi = -sbi

df_out = df[["sample_id"]].copy()
df_out["group"] = group
df_out["SBI"] = sbi

loadings = pca.components_.ravel()
df_load = pd.DataFrame({"module": module_cols, "loading_PC1": loadings})

df_out.to_csv("sbi_table.csv", index=False)
df_load.to_csv("sbi_module_loadings.csv", index=False)

print("Saved: sbi_table.csv, sbi_module_loadings.csv")

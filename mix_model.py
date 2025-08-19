import os
import networkx as nx
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kruskal
import matplotlib.pyplot as plt
from pylab import mpl
from scipy import stats
import statsmodels.formula.api as smf
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA

# ==================================
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']
mpl.rcParams['axes.unicode_minus'] = False

version = "PEP"
output_dir = "Result/"
os.makedirs(output_dir, exist_ok=True)

matrix_path = "data/PEP-matrix.xlsx"
info_path = "data/Knowledge Point information.xlsx"
bloom_path = "data/Bloom’s taxonomy information.xlsx"

theme_map = {1: "Atomic physics", 2: "Optics", 3: "Thermodynamics", 4: "Electromagnetism", 5: "Mechanics"}
level_map = {1: "Basic knowledge points", 2: "Composite knowledge points"}
weights = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]  # Semi-linear weighting (1--3.5)

# ================= function =================
def zscore_standardize(df, columns):
    """z-scores tandardization"""
    for col in columns:
        if df[col].std() == 0:
            df[col] = 0
        else:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df


def calculate_network_properties(G):
    """Centrality"""
    betweenness = nx.betweenness_centrality(G, weight='distance')
    closeness = nx.closeness_centrality(G, distance='distance')
    eigenvector = nx.eigenvector_centrality(G, weight='weight')
    clustering = nx.clustering(G, weight='weight')

    weighted_degrees = {n: sum(G[n][nb]['weight'] for nb in G.neighbors(n)) for n in G.nodes()}
    max_wd = max(weighted_degrees.values())
    W_degree = {n: wd / max_wd for n, wd in weighted_degrees.items()}

    return betweenness, closeness, eigenvector, clustering, W_degree

def kruskal_wallis_with_eta(df, group_col, value_col):
    """Kruskal-Wallis"""
    groups = [df[df[group_col] == g][value_col] for g in df[group_col].unique()]
    kw_result = kruskal(*groups)
    H, k, N = kw_result.statistic, len(groups), len(df)
    eta_squared = (H - (k - 1)) / (N - k) if N > k else np.nan
    return {'Metric': group_col, 'Type': 'Kruskal-Wallis', 'Value': H,
            'p-value': kw_result.pvalue, 'Eta_squared': eta_squared}


def calculate_Cs(cognitive_dict):
    """Cs"""
    scores = [cognitive_dict[lvl] for lvl in ['记忆', '理解', '应用', '分析', '评价', '创造']]
    return np.dot(weights, scores)


def calculate_mixed_r2(model_result):
    """marginal R² and conditional R²"""
    fixed_pred = model_result.predict()
    var_fixed = np.var(fixed_pred)
    var_random = model_result.cov_re.iloc[0, 0]
    var_resid = model_result.scale
    marginal_r2 = var_fixed / (var_fixed + var_random + var_resid)
    condition_r2 = (var_fixed + var_random) / (var_fixed + var_random + var_resid)
    return marginal_r2, condition_r2

# ================= Data reading and network construction=================
df_matrix = pd.read_excel(matrix_path, index_col=0)
nodes = df_matrix.index.tolist()
max_weight = max(df_matrix.max())

G = nx.Graph()
G.add_nodes_from(nodes)
for i in nodes:
    for j in nodes:
        weight = df_matrix.loc[i, j]
        if weight > 0 and i != j:
            G.add_edge(i, j, weight=weight, distance=max_weight / weight)

df_info = pd.read_excel(info_path)
modules = {row['Node']: theme_map[row['Theme']] for _, row in df_info.iterrows()}
levels = {row['Node']: level_map[row['Level']] for _, row in df_info.iterrows()}

# Bloom data
bloom_df = pd.read_excel(bloom_path).rename(columns={'Unnamed: 0': '知识点'})
bloom_data = bloom_df.set_index('知识点')
bloom_order = bloom_df['知识点'].tolist()

for node in G.nodes():
    if node in bloom_data.index:
        G.nodes[node]['cognitive'] = bloom_data.loc[node].to_dict()

# Building G1
G1 = G.copy()
for node in list(G1.nodes()):
    G1.nodes[node]['Cs'] = calculate_Cs(G1.nodes[node]['cognitive'])
nodes_to_remove = [n for n in G1.nodes() if G1.nodes[n]['Cs'] == 0]
G1.remove_nodes_from(nodes_to_remove)
g1_nodes_ordered = [n for n in bloom_order if n in G1.nodes()]
G1 = G1.subgraph(g1_nodes_ordered)

# ================= Indicator calculation =================
betweenness, closeness, eigenvector, clustering, W_degree = calculate_network_properties(G1)
df = pd.DataFrame({
    'node': list(G1.nodes()),
    'Cs': [G1.nodes[n]['Cs'] for n in G1.nodes()],
    'betweenness': list(betweenness.values()),
    'closeness': list(closeness.values()),
    'eigenvector': list(eigenvector.values()),
    'W_degree': list(W_degree.values()),
    'clustering': list(clustering.values()),
    'module': [modules[n] for n in G1.nodes()],
    'level': [levels[n] for n in G1.nodes()]
})

df = zscore_standardize(df, ['betweenness', 'closeness', 'eigenvector', 'W_degree', 'clustering', 'Cs'])

# ================= Statistical analysis =================
kw_level = kruskal_wallis_with_eta(df, 'level', 'Cs')
kw_module = kruskal_wallis_with_eta(df, 'module', 'Cs')
print("KW-level:", kw_level)
print("KW-module:", kw_module)

# VIF
X1 = df[['W_degree', 'eigenvector', 'closeness', 'betweenness']]
vif1 = pd.DataFrame([variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])],
                    index=X1.columns, columns=['VIF'])
print(vif1)

# PCA
pca = PCA(n_components=1)
df['Com_centrality'] = pca.fit_transform(df[['W_degree', 'eigenvector', 'betweenness']]).flatten()
df = zscore_standardize(df, ['Com_centrality'])

# Correlation analysis
corr_features = ['Com_centrality', 'closeness', 'clustering']
corr_results = [{'Metric': feat, 'Type': 'Correlation',
                 'Value': spearmanr(df['Cs'], df[feat])[0],
                 'p-value': spearmanr(df['Cs'], df[feat])[1]} for feat in corr_features]
corr_df = pd.DataFrame(corr_results)
print("CI-Correlation:", corr_df)

# ================= Visualization =================
plt.figure(figsize=(10, 8))
sns.boxplot(x='module', y='Cs', data=df, palette='Set2', hue='module', legend=False)
plt.xlabel("Theme"); plt.ylabel("Cognitive Strength")
plt.savefig(f"{output_dir}{version}-Box_Module.png", dpi=600); plt.close()

plt.figure(figsize=(10, 8))
sns.boxplot(x='level', y='Cs', data=df, palette='Set2', hue='level', legend=False)
plt.xlabel("Level"); plt.ylabel("Cognitive Strength")
plt.savefig(f"{output_dir}{version}-Box_Level.png", dpi=600); plt.close()

# ================= Mixed-effects model analysis =================
df['level'] = df['level'].astype('category')
df['module'] = df['module'].astype('category')

formula = "Cs ~ Com_centrality + closeness + C(module, Treatment(reference='Mechanics'))"
model = smf.mixedlm(formula, data=df, groups=df["level"])
result = model.fit(method='nm', maxiter=30000)

marginal_r2, condition_r2 = calculate_mixed_r2(result)
print(result.summary())
print(f"Marginal R²: {marginal_r2:.3f}, Conditional R²: {condition_r2:.3f}")

print(result.cov_re.iloc[0, 0])
print(result.cov_re.iloc[0, 0] - 1.96 * 0.414)
print(result.cov_re.iloc[0, 0] + 1.96 * 0.414)

# 残差QQ图
residuals = result.resid
plt.figure(figsize=(10, 8))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title("Q-Q Plot of Residuals")
plt.savefig(f"{output_dir}{version}-Q-Q Plot.png", dpi=600); plt.close()

# ================= Save the results =================
df_sorted = df.set_index('node').loc[g1_nodes_ordered].reset_index()
df_sorted.to_excel(f"{output_dir}{version}-metric result.xlsx", index=True)

result_file = f"{output_dir}{version}-mixedlm_result.txt"
with open(result_file, 'w', encoding='utf-8') as f:
    f.write(result.summary().as_text())

print("end")
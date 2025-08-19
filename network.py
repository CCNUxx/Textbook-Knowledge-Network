import os
import networkx as nx
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from pylab import mpl
from sklearn.decomposition import PCA

mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # Chinese fonts
mpl.rcParams['axes.unicode_minus'] = False

# ========== Function ==========
def calculate_interdisciplinary_ratio(G, modules):
    """Cross-theme ratio"""
    cross_edges = sum(1 for u, v in G.edges() if modules[u] != modules[v])
    total_edges = G.number_of_edges()
    return cross_edges / total_edges if total_edges > 0 else 0

def M_cal(a):
    """Monotonicity"""
    N = len(a)
    b = np.array(a)
    c = b[:, 1]
    unique_vals = sorted(set(c))
    counts = [list(c).count(val) for val in unique_vals]
    tmp = sum(cnt * (cnt - 1) for cnt in counts)
    tmp = tmp / (N * (N - 1))
    M = np.power(1 - tmp, 2)
    return M

def power_law(x, a, b):
    """Degree distribution"""
    return a * np.power(x, b)

# ========== data ==========
matrix_path = "data/PEP-matrix.xlsx"
info_path = "data/Knowledge Point information.xlsx"
version = "PEP"
output_dir = "Result/"
os.makedirs(output_dir, exist_ok=True)

df_matrix = pd.read_excel(matrix_path, index_col=0)
nodes = df_matrix.index.tolist()
max_weight = df_matrix.max().max()

G = nx.Graph()
G.add_nodes_from(nodes)
for i in nodes:
    for j in nodes:
        weight = df_matrix.loc[i, j]
        if weight > 0 and i != j:
            norm_inv_weight = max_weight / weight
            G.add_edge(i, j, weight=weight, distance=norm_inv_weight)

df_info = pd.read_excel(info_path)
theme_map = {1: "Atomic physics", 2: "Optics", 3: "Thermodynamics", 4: "Electromagnetism", 5: "Mechanics"}
modules = {row['Node']: theme_map[row['Theme']] for _, row in df_info.iterrows()}

# ========== Network indicators ==========
number_of_nodes = G.number_of_nodes()
number_of_edges = G.number_of_edges()
density = nx.density(G)
avg_degree = 2 * number_of_edges / number_of_nodes

components = nx.connected_components(G)
max_component = max(components, key=len)
mc = list(max_component)

mat = df_matrix.values
nodes_list = df_matrix.index.tolist()
indices = [nodes_list.index(node) for node in mc]
i, j = np.ix_(indices, indices)
d_res = mat[i, j]
G0 = nx.from_numpy_array(d_res)

Diameter = nx.diameter(G0)
avg_shortest_path_length = nx.average_shortest_path_length(G0)

average_clustering = nx.average_clustering(G, weight="weight", count_zeros=True)
transitivity = nx.transitivity(G)
assort = nx.degree_assortativity_coefficient(G, weight="weight")

interdisciplinary_ratio = calculate_interdisciplinary_ratio(G, modules)

gmc = nx.community.greedy_modularity_communities(G, weight="weight")
commu_G = nx.community.modularity(G, gmc, weight="weight")
community_sizes = [len(c) for c in gmc]
num_communities = len(gmc)

# ========== Centrality and monotonicity ==========
max_weight_degree = max(sum(G[node][nbr]['weight'] for nbr in G.neighbors(node)) for node in G.nodes)
weigh_dc = {node: sum(G[node][nbr]['weight'] for nbr in G.neighbors(node)) / max_weight_degree for node in G.nodes}

eigenvector = nx.eigenvector_centrality(G, max_iter=2000, weight="weight")
betweenness = nx.betweenness_centrality(G, weight="distance")

df_centrality = pd.DataFrame({
    "Node": list(G.nodes()),
    "W_degree": [weigh_dc[node] for node in G.nodes()],
    "eigenvector": [eigenvector[node] for node in G.nodes()],
    "betweenness": [betweenness[node] for node in G.nodes()],
})

# PCA
X_pca = df_centrality[['W_degree', 'eigenvector', 'betweenness']]
pca = PCA(n_components=1)
Com_centrality = pca.fit_transform(X_pca)
df_centrality['Com_centrality'] = Com_centrality.flatten()

# monotonicity
W_DC = M_cal(sorted(weigh_dc.items(), key=lambda x: x[1], reverse=True))
EC = M_cal(sorted(eigenvector.items(), key=lambda x: x[1], reverse=True))
W_CC = M_cal(sorted(nx.closeness_centrality(G, distance='distance').items(), key=lambda x: x[1], reverse=True))
BC = M_cal(sorted(betweenness.items(), key=lambda x: x[1], reverse=True))
M_Com_centrality = M_cal(df_centrality[['Node', 'Com_centrality']].values)

# ========== degree distribution ==========
d = dict(nx.degree(G))
dd = list(d.values())
n = len(G.nodes)

x = np.array(sorted(set(dd)))
y = np.array([dd.count(i) for i in x]) / n

mask = x > 1
x, y = x[mask], y[mask]

max_y_index = np.argmax(y)
x_max = x[max_y_index]

fit_mask = x > (x_max - 1)
params, _ = curve_fit(power_law, x[fit_mask], y[fit_mask])
a, b = params

topological_quantities = {
    'Topological quantities': [
        'Number of nodes',
        'Number of edges',
        'Density',
        'Average degree',
        'Diameter',
        'Average shortest path length',
        'Average clustering coefficient',
        'Network transitivity',
        'Assortativity',
        'Modularity',
        'Number of communities (Greedy modularity)',
        'Community structure',
        'Power-law exponent',
        'Interdisciplinary ratio',
        'DC-monotonicity',
        'EC-monotonicity',
        'CC-monotonicity',
        'BC-monotonicity',
        'Com_centrality-monotonicity',
    ],
    'feature': [
        number_of_nodes,
        number_of_edges,
        density,
        avg_degree,
        Diameter,
        avg_shortest_path_length,
        average_clustering,
        transitivity,
        assort,
        commu_G,
        num_communities,
        community_sizes,
        b,
        interdisciplinary_ratio,
        W_DC,
        EC,
        W_CC,
        BC,
        M_Com_centrality,
    ]
}
df_results = pd.DataFrame(topological_quantities)
df_results.to_csv(os.path.join(output_dir, f"{version}-topological_quantities_results.csv"), index=False, encoding='utf-8-sig')

print("end")


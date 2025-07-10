import netket as nk
import numpy as np

# --- 1(a) 建立 decorated-honeycomb graph -------------------

L = 3                         # 3×3 super-cell
basis = np.array([[0,1], [np.sqrt(3)/2,-0.5]])

site_offsets_uc = np.array([
    [0, 0],                  # 0
    [1/3, 0],                # 1
    [1/6, 1/2],              # 2
    [0, 1/2],                # 3
    [1/3, 1/2],              # 4
    [1/6, 1  ],              # 5
])

coords = []
for n1 in range(L):
    for n2 in range(L):
        shift = n1*basis[0] + n2*basis[1]
        coords.extend(site_offsets_uc + shift)

coords = np.asarray(coords)          # shape (N,2)
N = len(coords)

# -------- 2. 建立所有邊 ----------
edges_uc = [
    # triangle A
    (0,1), (1,2), (2,0),
    # triangle B
    (3,4), (4,5), (5,3),
    # inter-triangle 3 bonds
    (2,3),   #  z
    (1,5),   #  x
    (0,4),   #  y
]

edges = []

for n1 in range(L):
    for n2 in range(L):
        base = 6*(n1*L + n2)
        for a, b in edges_uc:
            i = base + a
            j = base + b
            edges.append((i, j))        
        i = base + 2                    
        j = 6*(((n1+1) % L)*L + n2) + 3 
        edges.append((i, j))

# 反向也要加；Graph 在無向圖時不會自動 mirror
edges = edges + [(j, i) for (i, j) in edges]


# -------- 3. 用 CustomGraph 建圖 ----------
graph_star = nk.graph.Graph(
    edges       = edges,        # list[(i,j)]
    n_nodes     = N,            # 一共 N = 6 L² 個點

)

print("sites =", graph_star.n_nodes)
print("edges =", len(graph_star.edges()))



# Coupling constants
Jx = Jy = Jz = 1.0


#triangles = [plaq for plaq in graph_star.plaquettes() if len(plaq)==3]
def find_triangles(graph):
    """
    Return list of 3-site cycles (i,j,k) for any NetKet graph.
    不用 neighbours() / plaquettes()，純靠 edges() 建鄰接表。
    """
    # 1) 先把鄰接表做成 list[set]
    adj = [set() for _ in range(graph.n_nodes)]
    for (u, v) in graph.edges():
        adj[u].add(v)
        adj[v].add(u)

    # 2) 掃描三角形
    tri = set()
    for i in range(graph.n_nodes):
        for j in adj[i]:
            if j <= i:
                continue
            for k in adj[j]:
                if k <= j:
                    continue
                if k in adj[i]:          # (i,k) 也是邊 ⇒ 成三角
                    tri.add(tuple(sorted((i, j, k))))
    return list(tri)

triangles = find_triangles(graph_star)
n_tri     = len(triangles)

print("triangle number =", n_tri)

# 1(a) site → triangle 索引對應表 ----------------------
site_to_triangle_index = {}           # { lattice_site : triangle_id }
for t_id, tri in enumerate(triangles):
    for s in tri:
        site_to_triangle_index[s] = t_id

hi = nk.hilbert.Spin(s=0.5, N=2*n_tri)   # 兩量子比特 / triangle

from netket.operator import PauliStrings, LocalOperator

X,Y,Z = 'X','Y','Z'


H = LocalOperator(hi)        # 先給一個全零起始算符

e_vecs = {
    'x': np.array([ 0.5,  np.sqrt(3)/2]),
    'y': np.array([ 0.5, -np.sqrt(3)/2]),
    'z': np.array([-1.0,  0.0 ]),
}

# 做單位化以便比較
for k in e_vecs:
    e_vecs[k] /= np.linalg.norm(e_vecs[k])

def identify_lambda(u, v):
    """回傳 'x' / 'y' / 'z'，根據邊向量 (v - u)。"""
    d   = coords[v] - coords[u]
    d  /= np.linalg.norm(d)              # 單位向量
    best, best_err = None, 1e9
    for lam, e in e_vecs.items():
        err = min(np.linalg.norm(d-e), np.linalg.norm(d+e))  # 正反皆可
        if err < best_err:
            best, best_err = lam, err
    return best

# -- 4(a) 方向向量--
# 三種鍵: bonds['x'] = [(i,j),...], 同理 'y','z'
bonds = {'x': [], 'y': [], 'z': []}
for (u,v) in graph_star.edges():            # lattice 邊 (u,v)
    # 判斷 (u,v) 是哪一種 λ = x,y,z，並找屬於哪兩個 triangle
    lam = identify_lambda(u,v)               
    ti = site_to_triangle_index[u]         
    tj = site_to_triangle_index[v]
    if ti!=tj:                              # 確保是 inter-triangle 鍵
        bonds[lam].append((ti,tj))

all_sites   = []      # 存放每條字串的 sites
all_strings = []      # 存放每條完整 Pauli 字串
all_weights = []      # 對應係數

# -- 4(b) 疊加各鍵上的四體 Pauli --
def collect_link(lam, ti, tj, Jlam):
    σ_ops  = [(X, X), (Y, Y), (Z, Z)]
    τ_axis = {'x': X, 'y': Y, 'z': Z}[lam]

    for pa_i, pa_j in σ_ops:
        full = ['I'] * hi.size
        full[2*ti]   = pa_i
        full[2*tj]   = pa_j
        full[2*ti+1] = τ_axis
        full[2*tj+1] = τ_axis

        all_sites.append(None)                 # 用 None → NetKet 自動從字串推 sites
        all_strings.append(''.join(full))      # 長度 = hi.size
        all_weights.append(Jlam/4.0)

for ti, tj in bonds['x']:
    collect_link('x', ti, tj, Jx)
for ti, tj in bonds['y']:
    collect_link('y', ti, tj, Jy)
for ti, tj in bonds['z']:
    collect_link('z', ti, tj, Jz)

# --- 一次性建立 Hamiltonian ---
H = PauliStrings(hi, operators=all_strings, weights=all_weights)


gs = nk.exact.lanczos_ed(H, k=1)
print("E0_ED =", gs[0])


# (b) Variational Monte Carlo
import netket.experimental as nkx

ma = nk.models.RBM(alpha=4)
sa = nk.sampler.MetropolisLocal(hi)
vmc = nk.vqs.MCState(sa, ma, n_samples=4096)
opt = nk.optimizer.Sgd(learning_rate=0.01)
sr = nk.optimizer.SR(diag_shift=0.01)
gs = nk.driver.VMC(hamiltonian = H, optimizer = opt, variational_state = vmc, preconditioner= sr)
gs.run(n_iter=300, out='outK')


#或者從 driver 的最後一步讀取 ---
print("E0_VMC (log) ≈", gs.energy.mean)  



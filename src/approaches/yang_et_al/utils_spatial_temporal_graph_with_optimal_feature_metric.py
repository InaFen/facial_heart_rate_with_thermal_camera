import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import cdist
import math

# ---------- helpers ----------
def compute_centroid(coords):
    if coords is None or len(coords) == 0:
        return (np.nan, np.nan)
    arr = np.array(coords, dtype=float)
    return (float(arr[:,0].mean()), float(arr[:,1].mean()))

def build_node_list_and_features(roi_pixel_data):
    """
    Build consistent node ordering and per-node features for all frames.
    Returns:
      node_features: (NT, D) array
      Y_vec: (NT,) observed mean signal vector
      roi_names: list of ROI names in order
      T, N
    """
    T = len(roi_pixel_data)
    if T == 0:
        raise ValueError("Empty roi_pixel_data")
    roi_names = list(roi_pixel_data[0].keys())
    N = len(roi_names)

    # feature dims: mean_value, centroid x, centroid y
    node_features = []
    Y = []

    for t in range(T):
        frame = roi_pixel_data[t]
        for name in roi_names:
            info = frame.get(name, None)
            if info is None:
                meanv = np.nan
                cx, cy = (np.nan, np.nan)
            else:
                meanv = float(info.get("mean_value", np.nan))
                cx, cy = compute_centroid(info.get("coords", []))
            node_features.append([meanv, cx, cy])
            Y.append(meanv)

    node_features = np.array(node_features)   # shape (N*T, 3)
    Y_vec = np.array(Y, dtype=float)         # shape (N*T,)
    return node_features, Y_vec, roi_names, T, N

# ---------- adjacency and laplacian ----------
def compute_weight_matrix(features, M, neighbor_mask=None, clip_diag=True):
    """
    features: (NT, D) array
    M: (D, D) PSD matrix
    neighbor_mask: boolean (NT,NT) mask allowed pairs (True to compute), or None -> all pairs
    returns W dense (NT,NT)
    """
    NT = features.shape[0]
    # Compute pairwise Mahalanobis distances efficiently:
    # d_ij = (f_i - f_j)^T M (f_i - f_j) = fi^T M fi + fj^T M fj - 2 fi^T M fj
    MF = features.dot(M)         # (NT, D)
    diag = np.einsum('ij,ij->i', features, MF)  # fi^T M fi
    # Compute matrix of -2 fi^T M fj
    cross = -2.0 * (MF.dot(features.T))  # (NT,NT)
    dmat = cross + diag[:,None] + diag[None,:]
    # numerical: clip small negative to 0
    dmat = np.maximum(dmat, 0.0)
    # weights
    W = np.exp(-dmat)
    if clip_diag:
        np.fill_diagonal(W, 0.0)
    if neighbor_mask is not None:
        W = W * neighbor_mask.astype(float)
    return W

def build_train_neighbor_mask(T, N, spatial_radius=None, temporal_adjacent_only=True):
    """
    Simple neighbor mask: allow intra-frame full connections and temporal links same-block adjacent frames.
    For efficiency we allow:
      - For nodes i,j in same frame: True (full)
      - For same block across t and t+1: True
      - Optionally restrict intra-frame to k-nearest by centroid distance (not implemented here).
    Returns mask of size (NT,NT).
    """
    NT = N * T
    mask = np.zeros((NT,NT), dtype=bool)
    for t in range(T):
        start = t * N
        mask[start:start+N, start:start+N] = True
    if temporal_adjacent_only:
        for t in range(T-1):
            for b in range(N):
                i = t*N + b
                j = (t+1)*N + b
                mask[i,j] = True
                mask[j,i] = True
    return mask

def laplacian_from_W(W):
    deg = np.sum(W, axis=1)
    D = np.diag(deg)
    L = D - W
    return L

# ---------- V update: closed form ----------
def update_V(Y_vec, L, mu):
    """
    Solve (I + mu * L) V = Y_vec for V.
    Use a sparse solver; if small use numpy.
    """
    NT = L.shape[0]
    A = csr_matrix(identity(NT)) + mu * csr_matrix(L)
    # spsolve expects CSR; for vector Y:
    v = spsolve(A, Y_vec)
    return v

# ---------- M update step ----------
def update_M(M, features, V_vec, alpha, beta, gamma, trace_limit_C, neighbor_mask=None):
    """
    Gradient step on M then PD projection and nuclear-norm shrinkage and trace-scaling.
    M <- Proj_{PSD, trace<=C}[ M - gamma * grad ]
    
    grad = -0.5 * alpha * sum_{i,j} w_ij * (V_i - V_j)^2 * (fi - fj)(fi - fj)^T
    where w_ij = exp(-d_ij), d_ij = (fi-fj)^T M (fi-fj).
    """
    NT, D = features.shape
    # compute current W and dmat
    W = compute_weight_matrix(features, M, neighbor_mask=neighbor_mask)
    # compute delta matrix of V differences squared
    V = V_vec.reshape(-1)
    # compute Δ^2 matrix:
    # Δ2_ij = (V_i - V_j)^2
    Vi = V[:,None]
    Vj = V[None,:]
    delta2 = (Vi - Vj)**2  # NT x NT

    # compute gradient: sum over i<j (but we can sum full matrix and half)
    # Form S = 0.5 * (W * delta2) elementwise
    S = 0.5 * (W * delta2)

    # compute gradM = -alpha * sum_{i,j} S_ij * (fi - fj)(fi - fj)^T
    gradM = np.zeros_like(M)
    # Efficient accumulation:
    # gradM = -alpha * ( sum_i fi * ( sum_j S_ij ) * fi^T  + sum_j fj * ( sum_i S_ij ) * fj^T - 2 sum_{i,j} S_ij fi fj^T )
    # But simpler (and clear): loop over i (but mindful of NT size)
    # We'll implement an O(NT^2 * D^2) safe version for moderate sizes. If NT large, you should use sparse/neighborhood.
    for i in range(NT):
        fi = features[i:i+1].T  # D x 1
        S_row = S[i, :]         # 1 x NT
        # sum_j S_ij * (fi - fj)(fi - fj)^T = ( sum_j S_ij ) fi fi^T - fi (sum_j S_ij fj^T) - (sum_j S_ij fj) fi^T + sum_j S_ij fj fj^T
        sumS = S_row.sum()
        sumSF = (S_row @ features)   # 1 x D
        sumSFfT = sumSF.reshape(D,1)
        term = sumS * (fi @ fi.T) - (fi @ sumSFfT.T) - (sumSFfT @ fi.T) + ( (features.T * S_row) @ features )
        # accumulate
        gradM += term

    gradM *= (-alpha)
    # take a gradient step
    M_new = M - gamma * gradM

    # PD projection and nuclear-norm shrinkage: eigen decompositon
    # Make symmetric
    M_new = 0.5 * (M_new + M_new.T)
    eigvals, eigvecs = np.linalg.eigh(M_new)

    # soft-threshold eigenvalues by gamma*beta (prox of nuclear norm)
    thresh = gamma * beta
    eigvals_shrunk = np.maximum(eigvals - thresh, 0.0)  # sets negatives to 0
    # Reconstruct
    M_proj = (eigvecs * eigvals_shrunk) @ eigvecs.T

    # trace constraint: scale if trace > C
    tr = np.trace(M_proj)
    if tr > trace_limit_C:
        M_proj = M_proj * (trace_limit_C / tr)

    # numerical symmetrize
    M_proj = 0.5 * (M_proj + M_proj.T)
    return M_proj, W

# ---------- full iterative algorithm ----------
def iterative_metric_learning(roi_pixel_data,
                              mu=0.1,      # smoothness weight for V
                              alpha=1.0,   # weight for graph term in Q(M)
                              beta=0.1,    # nuclear-norm weight
                              gamma=1e-3,  # step size for M update
                              trace_limit_C=1.0,
                              max_iters=10,
                              neighbor_mask=None):
    """
    Returns:
      V_vec: denoised signal vector (NT,)
      M: learned PSD metric (D,D)
      W: final adjacency matrix
      history: list of dict steps
    """
    features, Y_vec, roi_names, T, N = build_node_list_and_features(roi_pixel_data)
    NT, D = features.shape

    # Preprocess features: replace NaNs by zeros or means
    # Simple imputation:
    col_mean = np.nanmean(features, axis=0)
    inds = np.where(np.isnan(features))
    features[inds] = np.take(col_mean, inds[1])

    # Normalize features (important)
    mu_feat = features.mean(axis=0)
    std_feat = features.std(axis=0)
    std_feat[std_feat == 0] = 1.0
    features_norm = (features - mu_feat) / std_feat

    features = features_norm

    # initialize M
    M = np.eye(D)
    if neighbor_mask is None:
        neighbor_mask = np.ones((NT,NT), dtype=bool)  # full graph (may be heavy)

    history = []
    V = Y_vec.copy()  # initial V
    for it in range(max_iters):
        # 1) compute W and L from current M
        W = compute_weight_matrix(features, M, neighbor_mask=neighbor_mask)
        L = laplacian_from_W(W)

        # 2) update V (denoise)
        V = update_V(Y_vec, L, mu)

        # 3) update M using current V
        M, W_updated = update_M(M, features, V, alpha, beta, gamma, trace_limit_C, neighbor_mask=neighbor_mask)

        # record
        loss_data = {
            "iter": it,
            "V_norm_diff": float(np.linalg.norm(V - Y_vec)),
            "trace_M": float(np.trace(M)),
            "M_eig_max": float(np.max(np.linalg.eigvalsh(M))),
        }
        history.append(loss_data)
        print(f"Iter {it}: V||Y diff={loss_data['V_norm_diff']:.3f}, tr(M)={loss_data['trace_M']:.3f}, max_eig(M)={loss_data['M_eig_max']:.3f}")

    # final W and L
    W_final = compute_weight_matrix(features, M, neighbor_mask=neighbor_mask)
    L_final = laplacian_from_W(W_final)
    return V, M, W_final, L_final, history

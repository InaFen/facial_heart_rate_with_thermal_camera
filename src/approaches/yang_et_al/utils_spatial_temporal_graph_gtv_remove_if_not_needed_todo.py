import numpy as np
from scipy.spatial.distance import cdist
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Toggle advanced metric-learning path. When True, the M-aware graph construction
# and alternating metric learning routines will be used. Default: False (keep
# previous behavior).
optimize_with_M = False

# Toggle scalable metric-learning path. When True, uses paper-correct form with
# proximal nuclear norm and PSD+trace constraints. Default: False (keep previous).
optimize_with_M_not_scalable = False


def compute_centroid(coords: list[tuple[float, float]]) -> tuple[float, float] | None:
    """
    Compute centroid (mean x, mean y) from a list of (x, y) coordinates.

    Parameters:
        coords: list of (x, y) tuples
    Returns: (cx, cy) tuple of centroid coordinates, or None if coords is empty
    """
    if len(coords) == 0:
        return None

    arr = np.array(coords)     # shape (N, 2)
    cx = arr[:, 0].mean()      # mean of x-values
    cy = arr[:, 1].mean()      # mean of y-values
    return (cx, cy)

def add_centroids_to_roi_data(roi_pixel_data: list) -> list:
    """
    Given roi_pixel_data, compute and add centroids for each ROI in each frame.

    Parameters:
    roi_pixel_data : list of dicts
        roi_pixel_data[frame_idx][roi_name] = {
            "coords": list of (x, y) tuples,
            ...
        }

    Returns: modified roi_pixel_data with added "centroid" key for each ROI containing (cx, cy), also changed in place
    """
    for frame_idx, frame_data in enumerate(roi_pixel_data):
        for roi_name, roi_info in frame_data.items():

            coords = roi_info["coords"]
            centroid = compute_centroid(coords)

            roi_info["centroid"] = centroid
    
    return roi_pixel_data

def build_block_features_from_roi_pixel_data(roi_pixel_data):
    """ Build block features from roi_pixel_data.
    Each block feature vector contains:
    - Mean pixel value
    - Centroid x-coordinate
    - Centroid y-coordinate

    Parameters:
        roi_pixel_data: list per frame: dict {roi_name: {'coords':..., 'adc_values':..., 'mean_value':...}}

    Returns:
        block_features: list of length T; each element is an (N, 3) array of features for N blocks at frame t.
        roi_names: list of ROI names in order
    """
    T = len(roi_pixel_data)
    roi_names = list(roi_pixel_data[0].keys())
    N = len(roi_names)
    block_features = []
    for t in range(T):
        feats_t = []
        for name in roi_names:
            info = roi_pixel_data[t][name]
            meanv = info['mean_value'] if info['mean_value'] is not None else np.nan
            centroid = compute_centroid(info['coords'])  # returns (cx, cy) or None
            if centroid is None:
                cx, cy = np.nan, np.nan
            else:
                cx, cy = centroid
            feats_t.append([meanv, cx, cy])
        block_features.append(np.array(feats_t))  # N x 3
    return block_features, roi_names


def build_spatiotemporal_graph(block_features, temporal_weight=0.1, sigma_spatial=None, temporal_mode='similarity'):
    """
    block_features: list of length T; each element is an (N, D) array of features for N blocks at frame t.
    temporal_weight: default weight for same-block across adjacent frames (if temporal_mode == 'const').
    sigma_spatial: if provided, use exp(-dist^2/(2*sigma^2)), otherwise exp(-dist).
    temporal_mode: 'const' or 'similarity' (if 'similarity', weight by similarity of features across frames).
    Returns: W (adjacency), L (laplacian) - which has format: [t=0 ROI0, t=0 ROI1, ..., t=0 ROI3,
    t=1 ROI0, t=1 ROI1, ..., t=1 ROI3,]
    """
    # If optimize_with_M is
    # enabled or an explicit Mahalanobis metric `M` is provided, the M-aware
    # implementation below will be used instead of GLR denoising only
    if optimize_with_M:
        # Defer to M-aware constructor with identity metric as starting point
        M = np.eye(block_features[0].shape[1])
        return build_spatiotemporal_graph_with_M(block_features, M, temporal_weight=temporal_weight, sigma_spatial=sigma_spatial, temporal_mode=temporal_mode)

    # Default GLR denoising only path (Euclidean distances, no metric learning)
    # Get dimensions
    num_frames = len(block_features) # Number of frames (T)
    num_blocks = block_features[0].shape[0] # Number of blocks (= subregions) (N)
    D = block_features[0].shape[1] # Feature dimension
    total = num_blocks * num_frames # Total number of nodes
    adjacency_matrix = np.zeros((total, total), dtype=float) # Adjacency matrix (W)

    # Flatten all features to compute normalization
    block_values = [f[:, 0:1] for f in block_features]  # value
    block_coords = [f[:, 1:3] for f in block_features]  # x,y

    # Normalize values
    all_vals = np.vstack(block_values) # format: [t=0 ROI0, t=0 ROI1, ..., t=0 ROI3,t=1 ROI0, t=1 ROI1, ..., t=1 ROI3,]
    mean_val = np.nanmean(all_vals, axis=0) # all_vals might have some NaNs if ROIs not detected in every frame
    std_val = np.nanstd(all_vals, axis=0) # all_vals might have some NaNs if ROIs not detected in every frame
    std_val[std_val==0] = 1.0 # to avoid division by zero if all values are the same
    # Temporarily replace NaNs with mean when normalizing
    norm_values = [(np.where(np.isnan(v), mean_val, v) - mean_val)/std_val for v in block_values]

    # Normalize coordinates (similarly to values)
    all_coords = np.vstack(block_coords)
    mean_coord = np.nanmean(all_coords, axis=0)
    std_coord = np.nanstd(all_coords, axis=0)
    std_coord[std_coord==0] = 1.0
    norm_coords = [(np.where(np.isnan(c), mean_coord, c) - mean_coord)/std_coord for c in block_coords]
    
    # Recombine
    norm_feats = [np.hstack([v, c]) for v,c in zip(norm_values, norm_coords)]

    # Build spatial edges (-> nodes within same frame are connected, high weight if features similar)
    for t in range(num_frames):
        feature_matrix = norm_feats[t]  # N x D (F)
        # pairwise Euclidean distances
        pdist = cdist(feature_matrix, feature_matrix, metric='euclidean')  # N x N
        if sigma_spatial is None:
            spatial_weights = np.exp(-pdist)         # exp(-d) 
        else:
            spatial_weights = np.exp(-(pdist**2) / (2 * sigma_spatial**2)) # exp(-d^2/(2*sigma^2))
        
        # Set edges to 0 if original feature was NaN
        mask_nan = np.isnan(block_features[t]).any(axis=1)
        spatial_weights[mask_nan, :] = 0
        spatial_weights[:, mask_nan] = 0

        start = t * num_blocks # starting index in W
        adjacency_matrix[start:start+num_blocks, start:start+num_blocks] = spatial_weights # Fill the diagonal block for frame t with spatial weights

    # Temporal edges
    for t in range(num_frames-1):
        Fcur = norm_feats[t]
        Fnext = norm_feats[t+1] # Loop over consecutive frames
        start_cur = t * num_blocks
        start_next = (t+1) * num_blocks
        if temporal_mode == 'const':
            # Same block across frames has constant weight
            for b in range(num_blocks):
                 # Skip if either block is NaN
                if np.isnan(block_features[t][b]).any() or np.isnan(block_features[t+1][b]).any():
                    continue
                i = start_cur + b
                j = start_next + b
                adjacency_matrix[i, j] = temporal_weight
                adjacency_matrix[j, i] = temporal_weight
        elif temporal_mode == 'similarity':
            # Same block across frames weighted by feature similarity
            # similarity between same block across times
            pd = np.linalg.norm(Fcur - Fnext, axis=1)  # length N
            sim = np.exp(-pd)
            # zero similarity if either feature was NaN
            mask_nan = np.isnan(block_features[t]).any(axis=1) | np.isnan(block_features[t+1]).any(axis=1)
            sim[mask_nan] = 0
            for b in range(num_blocks):
                i = start_cur + b
                j = start_next + b
                adjacency_matrix[i, j] = sim[b]
                adjacency_matrix[j, i] = sim[b]

    # Degree and laplacian
    degree_matrix = np.diag(adjacency_matrix.sum(axis=1)) # Degree matrix (sum of rows)
    laplacian = degree_matrix - adjacency_matrix # Laplacian (used for graph smoothing)
    return adjacency_matrix, laplacian

import numpy as np

def gtv_denoise(Y, W, mu=0.1, rho=1.0, iters=50):
    """
    Graph Total Variation denoising via simple ADMM.
    
    Y : observed signal (N,)
    W : adjacency matrix
    """

    N = len(Y)

    # initialize
    V = Y.copy()
    Z = np.zeros((N, N))
    U = np.zeros((N, N))  # dual variable

    # precompute edges
    edges = [(i, j) for i in range(N) for j in range(N) if W[i, j] > 0]

    for _ in range(iters):

        # -------------------------
        # 1. V-update (quadratic)
        # -------------------------
        V_old = V.copy()

        grad = np.zeros(N)
        for i, j in edges:
            diff = V[i] - V[j] - Z[i, j] + U[i, j]
            grad[i] += diff
            grad[j] -= diff

        V = (Y + rho * grad) / (1 + rho * np.ones(N))

        # -------------------------
        # 2. Z-update (soft thresholding)
        # -------------------------
        for i, j in edges:
            x = V[i] - V[j] + U[i, j]

            # soft threshold (key GTV step)
            Z[i, j] = np.sign(x) * max(abs(x) - mu / rho, 0)

        # -------------------------
        # 3. Dual update
        # -------------------------
        for i, j in edges:
            U[i, j] += V[i] - V[j] - Z[i, j]

    return V

def denoise_via_laplacian(Y, laplacian, mu=0.1):
    """Denoise signal Y via graph Laplacian L."""
    # Y: shape (N*T,) vector or (N*T, C) if multichannel stacked appropriately
    Ntot = laplacian.shape[0] # total number of nodes
    A = sparse.identity(Ntot, format='csr') + mu * sparse.csr_matrix(laplacian) # (I + mu*L)
    # If Y is vector:
    x = spsolve(A, Y) # solve (I + mu*L)x = Y -> GLR
    #x = gtv_denoise(Y, adjacency_matrix, mu=0.1) # for GTV
    return x # denoised signal


# -------------------------
# M-Aware graph & metric learning
# -------------------------
def mahalanobis_pdist(X, M):
    """Compute pairwise Mahalanobis distances between rows of X using metric M."""
    diff = X[:, None, :] - X[None, :, :]        # N x N x d
    Md = diff @ M                               # N x N x d
    dist = np.sum(Md * diff, axis=-1)          # N x N
    return np.sqrt(np.maximum(dist, 1e-12))

def build_spatiotemporal_graph_with_M(block_features,
                                     M,
                                     temporal_weight=0.1,
                                     sigma_spatial=None,
                                     temporal_mode='similarity'):
    """Build spatio-temporal adjacency and laplacian using Mahalanobis metric M.

    Parameters:
        block_features: list of (N, d) arrays per frame
        M: d x d Mahalanobis metric (positive semidef)
    Returns: W, L
    """
    num_frames = len(block_features)
    num_blocks = block_features[0].shape[0]
    total = num_frames * num_blocks

    W = np.zeros((total, total))

    # ---- SPATIAL EDGES ----
    for t in range(num_frames):
        X = block_features[t]

        # handle NaNs (simple masking)
        mask = ~np.isnan(X).any(axis=1)
        X_valid = np.nan_to_num(X)

        pdist = mahalanobis_pdist(X_valid, M)

        if sigma_spatial is None:
            Wt = np.exp(-pdist)
        else:
            Wt = np.exp(-(pdist ** 2) / (2 * sigma_spatial ** 2))

        Wt[~mask, :] = 0
        Wt[:, ~mask] = 0

        start = t * num_blocks
        W[start:start+num_blocks, start:start+num_blocks] = Wt

    # ---- TEMPORAL EDGES ----
    for t in range(num_frames - 1):
        X1 = block_features[t]
        X2 = block_features[t + 1]

        start1 = t * num_blocks
        start2 = (t + 1) * num_blocks

        if temporal_mode == 'const':
            for b in range(num_blocks):
                if np.any(np.isnan(X1[b])) or np.any(np.isnan(X2[b])):
                    continue
                i = start1 + b
                j = start2 + b
                W[i, j] = W[j, i] = temporal_weight

        else:
            d = np.linalg.norm(X1 - X2, axis=1)
            sim = np.exp(-d)

            for b in range(num_blocks):
                if np.any(np.isnan(X1[b])) or np.any(np.isnan(X2[b])):
                    continue
                i = start1 + b
                j = start2 + b
                W[i, j] = W[j, i] = sim[b]

    D = np.diag(W.sum(axis=1))
    L = D - W
    return W, L


def update_metric_simple(block_features, V, M, lr=1e-3):
    """Simple low-rank proxy update for Mahalanobis metric M.

    This implementation follows the user-provided low-rank gradient surrogate
    and projects back to PSD at the end.
    """
    d = M.shape[0]
    grad = np.zeros((d, d))

    N = block_features[0].shape[0]
    T = len(block_features)

    for t in range(T):
        X = block_features[t]

        X = np.nan_to_num(X)

        for i in range(N):
            for j in range(N):

                diff_v = V[t*N + i] - V[t*N + j]
                diff_x = (X[i] - X[j]).reshape(-1, 1)

                grad += (diff_v ** 2) * (diff_x @ diff_x.T)

    # nuclear norm surrogate (trace regularization)
    grad += np.eye(d)

    M = M - lr * grad

    # ---- PSD projection ----
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.clip(eigvals, 0, None)

    M = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return M


def alternating_glr_metric_learning(block_features,
                                    Y,
                                    num_iters=5,
                                    mu=0.1):

    d = block_features[0].shape[1]
    M = np.eye(d)

    V = Y.copy()

    for it in range(num_iters):

        # ---- build graph with current M ----
        W, L = build_spatiotemporal_graph_with_M(block_features, M)

        # ---- GLR denoising ----
        V = denoise_via_laplacian(V, L, mu)

        # ---- update metric ----
        M = update_metric_simple(block_features, V, M)

        print(f"Iter {it+1}: trace(M) = {np.trace(M):.4f}")

    return V, M


# ========================================
# Scalable M-aware graph & metric learning
# (Paper-correct form with nuclear norm)
# ========================================

def mahalanobis_distances(X, M):
	"""Compute pairwise Mahalanobis distances using metric M."""
	diff = X[:, None, :] - X[None, :, :]
	Md = diff @ M
	dist = np.sum(Md * diff, axis=-1)
	return np.sqrt(np.maximum(dist, 1e-12))


def build_graph(block_features, M):
	"""Build spatio-temporal graph adjacency and Laplacian from metric M.
	
	Simplified version: only spatial edges (no temporal), each frame independent.
	"""
	T = len(block_features)
	N = block_features[0].shape[0]
	total = T * N

	W = np.zeros((total, total))

	for t in range(T):
		X = np.nan_to_num(block_features[t])

		D = mahalanobis_distances(X, M)
		Wt = np.exp(-D)

		start = t * N
		W[start:start+N, start:start+N] = Wt

	Dg = np.diag(W.sum(axis=1))
	L = Dg - W

	return W, L


def update_V(Y, L, alpha):
	"""V-step: exact solve (I + alpha*L) V = Y."""
	n = L.shape[0]
	A = sparse.eye(n, format='csr') + alpha * sparse.csr_matrix(L)
	return spsolve(A, Y)


def svt_proximal(X, tau):
	"""Singular Value Thresholding (proximal operator for nuclear norm).
	
	Solves: argmin ||Z||_* + (1/(2*tau))||Z - X||_F^2
	"""
	U, S, Vt = np.linalg.svd(X, full_matrices=False)

	S_shrink = np.maximum(S - tau, 0)

	return U @ np.diag(S_shrink) @ Vt


def project_psd_trace(M, C):
	"""Project matrix M onto PSD cone with trace <= C."""
	eigvals, eigvecs = np.linalg.eigh(M)

	eigvals = np.clip(eigvals, 0, None)

	if eigvals.sum() > C:
		eigvals *= C / eigvals.sum()

	return eigvecs @ np.diag(eigvals) @ eigvecs.T


def update_M(block_features, V, M, beta, C, lr=1.0):
	"""M-step: gradient descent + proximal nuclear norm + PSD+trace projection.
	
	This follows the paper-correct form with SVT and projection.
	"""
	d = M.shape[0]
	grad = np.zeros((d, d))

	T = len(block_features)
	N = block_features[0].shape[0]

	# smoothness-driven gradient
	for t in range(T):
		X = np.nan_to_num(block_features[t])

		for i in range(N):
			for j in range(N):
				dv = V[t*N + i] - V[t*N + j]
				dx = (X[i] - X[j]).reshape(-1, 1)

				grad += (dv ** 2) * (dx @ dx.T)

	# gradient step
	M_tilde = M - lr * grad

	# nuclear norm proximal operator (key paper step)
	M_hat = svt_proximal(M_tilde, beta * lr)

	# projection onto feasible set
	M_proj = project_psd_trace(M_hat, C)

	return M_proj


def paper_glr_metric_learning(block_features, Y,
                              alpha=1.0,
                              beta=0.1,
                              C=10.0,
                              num_iters=10):
	"""Full alternating algorithm (paper version).
	
	Alternates between:
	  - V-step: exact solve with graph Laplacian
	  - M-step: proximal nuclear norm + PSD+trace projection
	"""
	d = block_features[0].shape[1]
	M = np.eye(d)

	V = Y.copy()

	for it in range(num_iters):

		# ---- build graph ----
		W, L = build_graph(block_features, M)

		# ---- V update (exact solve) ----
		V = update_V(Y, L, alpha)

		# ---- M update (proximal nuclear norm) ----
		M = update_M(block_features, V, M, beta, C)

		print(f"iter {it+1}: trace(M)={np.trace(M):.4f}")

	return V, M


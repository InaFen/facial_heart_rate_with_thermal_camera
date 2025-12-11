import numpy as np
from scipy.spatial.distance import cdist

def compute_centroid(coords):
    """
    coords: list of (x, y) tuples
    returns: (cx, cy)
    """
    if len(coords) == 0:
        return None

    arr = np.array(coords)     # shape (N, 2)
    cx = arr[:, 0].mean()      # mean of x-values
    cy = arr[:, 1].mean()      # mean of y-values
    return (cx, cy)

def build_spatiotemporal_graph(block_features, temporal_weight=0.1):
    """
    Build spatial-temporal adjacency matrix W and Laplacian L.

    Parameters
    ----------
    block_features : list of arrays
        block_features[t] is an NxD matrix (N blocks, D features)
        one feature vector per subregion for each frame.
    
    temporal_weight : float
        Weight for temporal edges (same block across frames)

    Returns
    -------
    W : adjacency matrix (numpy array)
    L : graph Laplacian matrix
    """

    T = len(block_features)      # number of frames
    N, D = block_features[0].shape  # number of blocks, feature length

    total_nodes = N * T
    W = np.zeros((total_nodes, total_nodes))

    # Simple Mahalanobis metric (identity)
    M = np.eye(D)

    # ---- SPATIAL EDGES ----
    for t in range(T):
        features_t = block_features[t]

        # Compute Mahalanobis distance between blocks
        diffs = cdist(features_t @ M, features_t @ M, metric='euclidean')

        # Weight function from the paper (Eq. 3)
        W_spatial = np.exp(-diffs)

        # Insert into the global adjacency matrix
        start = t * N
        W[start:start+N, start:start+N] = W_spatial

    # ---- TEMPORAL EDGES ----
    for t in range(T-1):
        idx_curr = t * N
        idx_next = (t+1) * N

        # same block → next frame
        for b in range(N):
            i = idx_curr + b
            j = idx_next + b

            W[i, j] = temporal_weight
            W[j, i] = temporal_weight

    # Degree matrix
    D_mat = np.diag(W.sum(axis=1))

    # Laplacian
    L = D_mat - W

    return W, L

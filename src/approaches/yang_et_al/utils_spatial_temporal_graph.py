import numpy as np
from scipy.spatial.distance import cdist
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


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

def add_centroids_to_roi_data(roi_pixel_data):
    """
    Given roi_pixel_data, compute and add centroids for each ROI in each frame.

    Parameters
    ----------
    roi_pixel_data : list of dicts
        roi_pixel_data[frame_idx][roi_name] = {
            "coords": list of (x, y) tuples,
            ...
        }

    Returns
    -------
    None (modifies roi_pixel_data in place)
    """
    for frame_idx, frame_data in enumerate(roi_pixel_data):
        for roi_name, roi_info in frame_data.items():

            coords = roi_info["coords"]
            centroid = compute_centroid(coords)

            roi_info["centroid"] = centroid
    
    return roi_pixel_data

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


def build_block_features_from_roi_pixel_data(roi_pixel_data):
    """ Build block features from roi_pixel_data.
    Each block feature vector contains:
    - Mean pixel value
    - Centroid x-coordinate
    - Centroid y-coordinate
    Parameters:
        roi_pixel_data: list per frame: dict {roi_name: {'coords':..., 'adc_values':..., 'mean_adc':...}}
    Returns:
        block_features: list of length T; each element is an (N, 3) array of features for N blocks at frame t.
        roi_names: list of ROI names in order"""
    T = len(roi_pixel_data)
    roi_names = list(roi_pixel_data[0].keys())
    N = len(roi_names)
    block_features = []
    for t in range(T):
        feats_t = []
        for name in roi_names:
            info = roi_pixel_data[t][name]
            meanv = info['mean_adc'] if info['mean_adc'] is not None else np.nan
            centroid = compute_centroid(info['coords'])  # returns (cx, cy) or None
            if centroid is None:
                cx, cy = np.nan, np.nan
            else:
                cx, cy = centroid
            feats_t.append([meanv, cx, cy])
        block_features.append(np.array(feats_t))  # N x 3
    return block_features, roi_names

def build_spatiotemporal_graph_2(block_features, temporal_weight=0.1, sigma_spatial=None, temporal_mode='similarity'):
    """
    block_features: list of length T; each element is an (N, D) array of features for N blocks at frame t.
    temporal_weight: default weight for same-block across adjacent frames (if temporal_mode == 'const').
    sigma_spatial: if provided, use exp(-dist^2/(2*sigma^2)), otherwise exp(-dist).
    temporal_mode: 'const' or 'similarity' (if 'similarity', weight by similarity of features across frames).
    Returns: W (adjacency), L (laplacian) - which has format: [t=0 ROI0, t=0 ROI1, ..., t=0 ROI3,
    t=1 ROI0, t=1 ROI1, ..., t=1 ROI3,]
    """
    T = len(block_features) # Number of frames
    N = block_features[0].shape[0] # Number of blocks (= subregions)
    D = block_features[0].shape[1] # Feature dimension
    total = N * T # Total number of nodes
    W = np.zeros((total, total), dtype=float) # Adjacency matrix

    # Optionally normalize features per-dim to zero mean unit var
    # flatten all features to compute normalization
    block_values = [f[:, 0:1] for f in block_features]  # value
    block_coords = [f[:, 1:3] for f in block_features]  # x,y

    # normalize values
    all_vals = np.vstack(block_values) # format: [t=0 ROI0, t=0 ROI1, ..., t=0 ROI3,t=1 ROI0, t=1 ROI1, ..., t=1 ROI3,]
    mu_val = np.nanmean(all_vals, axis=0) # all_vals might have some NaNs if ROIs not detected in every frame
    s_val = np.nanstd(all_vals, axis=0) # all_vals might have some NaNs if ROIs not detected in every frame
    s_val[s_val==0] = 1.0
    # temporarily replace NaNs with mean when normalizing
    norm_values = [(np.where(np.isnan(v), mu_val, v) - mu_val)/s_val for v in block_values]

    # normalize coordinates
    all_coords = np.vstack(block_coords)
    mu_coord = np.nanmean(all_coords, axis=0) # all_coords might have some NaNs if ROIs not detected in every frame
    s_coord = np.nanstd(all_coords, axis=0)
    s_coord[s_coord==0] = 1.0
    #  temporarily replace NaNs with mean when normalizing
    norm_coords = [(np.where(np.isnan(c), mu_coord, c) - mu_coord)/s_coord for c in block_coords]

    # recombine
    norm_feats = [np.hstack([v, c]) for v,c in zip(norm_values, norm_coords)]

    # Build spatial edges (-> nodes within same frame are connected, high weight if features similar)
    for t in range(T):
        F = norm_feats[t]  # N x D
        # pairwise Euclidean distances
        pdist = cdist(F, F, metric='euclidean')  # N x N
        if sigma_spatial is None:
            Wsp = np.exp(-pdist)         # exp(-d) 
        else:
            Wsp = np.exp(-(pdist**2) / (2 * sigma_spatial**2)) # exp(-d^2/(2*sigma^2))
        
        # Set edges to 0 if original feature was NaN
        mask_nan = np.isnan(block_features[t]).any(axis=1)
        Wsp[mask_nan, :] = 0
        Wsp[:, mask_nan] = 0

        start = t * N # starting index in W
        W[start:start+N, start:start+N] = Wsp # Fill the diagonal block for frame t with spatial weights

    # temporal edges
    for t in range(T-1):
        Fcur = norm_feats[t]
        Fnext = norm_feats[t+1] # Loop over consecutive frames
        start_cur = t * N
        start_next = (t+1) * N
        if temporal_mode == 'const':
            # Same block across frames has constant weight
            for b in range(N):
                 # skip if either block is NaN
                if np.isnan(block_features[t][b]).any() or np.isnan(block_features[t+1][b]).any():
                    continue
                i = start_cur + b
                j = start_next + b
                W[i, j] = temporal_weight
                W[j, i] = temporal_weight
        elif temporal_mode == 'similarity':
            # Same block across frames weighted by feature similarity
            # similarity between same block across times
            pd = np.linalg.norm(Fcur - Fnext, axis=1)  # length N
            sim = np.exp(-pd)
            # zero similarity if either feature was NaN
            mask_nan = np.isnan(block_features[t]).any(axis=1) | np.isnan(block_features[t+1]).any(axis=1)
            sim[mask_nan] = 0
            for b in range(N):
                i = start_cur + b
                j = start_next + b
                W[i, j] = sim[b]
                W[j, i] = sim[b]

    # degree and laplacian
    Dmat = np.diag(W.sum(axis=1)) # Degree matrix (sum of rows)
    L = Dmat - W # Laplacian (used for graph smoothing)
    return W, L

def denoise_via_laplacian(Y, L, mu=0.1):
    """Denoise signal Y via graph Laplacian L."""
    # Y: shape (N*T,) vector or (N*T, C) if multichannel stacked appropriately
    Ntot = L.shape[0] # total number of nodes
    A = sparse.identity(Ntot, format='csr') + mu * sparse.csr_matrix(L) # (I + mu*L)
    # If Y is vector:
    x = spsolve(A, Y) # solve (I + mu*L)x = Y
    return x # denoised signal


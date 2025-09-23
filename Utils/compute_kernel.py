import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sp
from scipy.linalg import eigh
import ot
import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph
from utils_other import construct_graph, init_distances, rbf
from collections import namedtuple

def compute_mmd(K_XX, K_YY, K_XY):
    """ Computes MMD between two distributions given their kernel matrices. """
    m = K_XX.shape[0]  # Number of samples in X
    n = K_YY.shape[0]  # Number of samples in Y
    
    term_X = (1 / (m * (m-1))) * np.sum(K_XX - np.diag(np.diag(K_XX)))
    term_Y = (1 / (n * (n-1))) * np.sum(K_YY - np.diag(np.diag(K_YY)))
    term_XY = (2 / (m * n)) * np.sum(K_XY)

    mmd_squared = term_X + term_Y - term_XY
    #print(mmd_squared)
    return np.sqrt(mmd_squared)

def compute_WassKernel(data,n_pt=10000, metric='Euclidean',normalize=False, return_distance=False):
    n = len(data)
    quantiles_all = [0]*len(data)
    print(n_pt)
    for i in range(n):
        if data[i].shape[0] == 0:
            quantiles_all[i] = np.inf + np.zeros( (n_pt))
        else:
            if metric == 'Euclidean':
                C1 = sp.spatial.distance.cdist(data[i], data[i])
            elif metric == 'Graph':
                Xgraph = construct_graph(data[i], k=5, mode="distance",metric="euclidean").todense()

                C1 = init_distances(Xgraph)
            else:
                C1 = diffusion_distance(data[i])
            if normalize:
                C1 = C1/np.median(C1)
            # quantiles_all[i] = np.quantile(C1[np.triu_indices_from(C1, k=1)].ravel(), np.linspace(0,1,n_pt,endpoint=False))
            quantiles_all[i] = np.quantile(C1.ravel(), np.linspace(0,1,n_pt,endpoint=True))
            
    if return_distance == False:       
        return sp.spatial.distance.squareform(sp.spatial.distance.pdist(quantiles_all))/n_pt
    else:
        return sp.spatial.distance.squareform(sp.spatial.distance.pdist(quantiles_all))/n_pt, quantiles_all


def score_wasserstein(distance_matrix, n_2=5):
    overall_quantiles = np.quantile(distance_matrix.ravel(), np.linspace(0, 1, n_2+1))
    scores = np.zeros(distance_matrix.shape[0])
    for i in range(distance_matrix.shape[0]):
        quantiles_i = np.quantile(distance_matrix[i,:], np.linspace(0, 1, n_2+1))
        scores[i] = np.linalg.norm(overall_quantiles - quantiles_i)
    return scores

def distance_matrix_quantiles(distance_matrix, n_1=4, n_2=5, rank_power=1, eps=0):
    # Step 1: Compute row variances
    row_variances = distance_matrix.mean(axis=1)
    for k in range(1,rank_power):
        row_variances += np.mean( distance_matrix ** (k-1), axis=1 ) * np.power(eps,k)
    # row_variances = score_wasserstein(distance_matrix, n_2=n_2)
    
    # Step 2: Get quantile thresholds for variance
    variance_quantiles = np.quantile(row_variances, np.linspace(0, 1, n_1+1))
    
    # Step 3: Initialize result matrix
    result = np.zeros((n_1, n_2))
    
    # Step 4: Compute quantiles for each block
    for i in range(n_1):
        # Get indices of rows within the variance quantile range
        block_indices = np.where((row_variances >= variance_quantiles[i]) & (row_variances <= variance_quantiles[i + 1]))[0]
        not_block_indices = np.where((row_variances < variance_quantiles[i]) | (row_variances > variance_quantiles[i + 1]))[0]
        # Extract submatrix and compute upper triangular distances
        block_distances = distance_matrix[block_indices,:]#[:, not_block_indices]
        pairwise_dists = block_distances.ravel()#block_distances[np.triu_indices(len(block_indices), k=1)]
        
        # Compute n_2 quantiles of the distances and store them
        result[i] = np.quantile(pairwise_dists, np.linspace(0, 1, n_2))
    
    return result

def distance_matrix_block_quantiles(distance_matrix, n_1=4, n_2=5):
    # Step 1: Compute row variances
    # row_variances = distance_matrix.mean(axis=1)
    row_variances = score_wasserstein(distance_matrix, n_2=n_2)
    
    # Step 2: Get quantile thresholds for variance
    variance_quantiles = np.quantile(row_variances, np.linspace(0, 1, n_1+1))
    
    # Step 3: Initialize result matrix
    result = np.zeros((n_1, n_1*n_2))
    
    # Step 4: Compute quantiles for each block
    for i in range(n_1):
        # Get indices of rows within the variance quantile range
        if i < len(variance_quantiles) - 2:
            block_indices = np.where((row_variances >= variance_quantiles[i]) & 
                                    (row_variances < variance_quantiles[i + 1]))[0]
        else:
            block_indices = np.where((row_variances >= variance_quantiles[i]) & 
                                    (row_variances <= variance_quantiles[i + 1]))[0]
        not_block_indices = np.where((row_variances < variance_quantiles[i]) | (row_variances > variance_quantiles[i + 1]))[0]
        for j in range(n_1):
            if i < len(variance_quantiles) - 2:
                block_indices_j = np.where((row_variances >= variance_quantiles[j]) & 
                                        (row_variances < variance_quantiles[j + 1]))[0]
            else:
                block_indices_j = np.where((row_variances >= variance_quantiles[j]) & 
                                        (row_variances <= variance_quantiles[j + 1]))[0]
            block_distances = distance_matrix[np.ix_(block_indices,block_indices_j)]#[:, not_block_indices]
            pairwise_dists = block_distances.ravel()#block_distances[np.triu_indices(len(block_indices), k=1)]
            # print(pairwise_dists.shape)
        
            # Compute n_2 quantiles of the distances and store them
            # print(n_2)
            # print(np.quantile(pairwise_dists, np.linspace(0, 1, n_2)))

            if pairwise_dists.shape[0] > 0:
                result[i][j*n_2:(j+1)*n_2] = np.quantile(pairwise_dists, np.linspace(0, 1, n_2))
    
    return result



def diffusion_distance(point_cloud, n_components=3, epsilon=None, t=1):
    """
    Compute diffusion distances for a point cloud.

    Parameters:
    - point_cloud: np.ndarray of shape (n_samples, n_features)
    - n_components: number of diffusion map components to use
    - epsilon: bandwidth for the Gaussian kernel; if None, use median heuristic
    - t: diffusion time (higher t means more smoothing)

    Returns:
    - dist_matrix: diffusion distance matrix of shape (n_samples, n_samples)
    """
    import numpy as np
    from scipy.spatial.distance import pdist, squareform
    from scipy.linalg import eigh
    # Step 1: Compute pairwise Euclidean distances
    pairwise_dists = squareform(pdist(point_cloud, 'sqeuclidean'))

    # Step 2: Set epsilon if not provided (median heuristic)
    if epsilon is None:
        upper_tri = pairwise_dists[np.triu_indices_from(pairwise_dists, 1)]
        epsilon = .2*np.median(upper_tri)
    else:
        upper_tri = pairwise_dists[np.triu_indices_from(pairwise_dists, 1)]
        epsilon = epsilon*np.median(upper_tri)

    # Step 3: Construct affinity matrix using Gaussian kernel
    K = np.exp(-pairwise_dists / epsilon)

    # Step 4: Row-normalize to create the Markov transition matrix
    d = K.sum(axis=1)
    P = K / d[:, None]

    # Step 5: Eigen-decomposition
    eigvals, eigvecs = eigh(P)
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]

    # Step 6: Compute diffusion map embedding
    lambdas = eigvals[1:n_components + 1]**t  # skip the first (trivial) eigenvalue = 1
    psi = eigvecs[:, 1:n_components + 1]

    # Step 7: Compute diffusion distance
    embedding = psi * lambdas
    dist_matrix = squareform(pdist(embedding, 'euclidean'))

    return dist_matrix


def compute_WassKernel_stratified(data, n_align = 100, n_quantile = 100, metric='Euclidean',k=5, normalize=False, return_distance=False, return_shape=False, block=False, quantile_only=False, rank_power=1, eps=0):
    n = len(data)
    quantiles_all = [0]*len(data)
    min_size = data[0].shape[0]
    for i in range(n):
        if data[i].shape[0]<min_size:
            min_size = data[i].shape[0]
            
    if n_align > min_size:
        n_align = min_size
    if n_quantile > min_size**2//2:
        n_quantile = min_size**2//2

    for i in range(n):
        print(i)
        if data[i].shape[0] == 0:
            quantiles_all[i] = np.inf + np.zeros( (n_align*n_quantile))
        else:
            if metric == 'Euclidean':
                C1 = sp.spatial.distance.cdist(data[i], data[i])
            elif metric == 'Graph':
                Xgraph = construct_graph(data[i], k=k, mode="distance",metric="euclidean").todense()
                C1 = init_distances(Xgraph)
            else:
                C1 = diffusion_distance(data[i],epsilon=1./k)
            if normalize:
                
                C1 = C1/np.median(C1)#np.max([np.mean(C1),np.median(C1)])
            if block == False:
                quantiles_all[i] = distance_matrix_quantiles(C1, n_1=n_align, n_2=n_quantile, rank_power=rank_power, eps=eps).ravel()
            else:
                if n_quantile > round(min_size/n_align/n_align):
                    n_quantile = 2#round(min_size/n_align/n_align)
                quantiles_all[i] = distance_matrix_block_quantiles(C1, n_1=n_align, n_2=n_quantile, rank_power=rank_power, eps=eps).ravel()
            
                
                
    #print(quantiles_all.shape)  
    if quantile_only == True:
        return quantiles_all
    if return_distance == False:
        return sp.spatial.distance.squareform(sp.spatial.distance.pdist(quantiles_all))/n_align/n_quantile
    elif return_shape == False:
        return sp.spatial.distance.squareform(sp.spatial.distance.pdist(quantiles_all))/n_align/n_quantile, quantiles_all
    else:
        return sp.spatial.distance.squareform(sp.spatial.distance.pdist(quantiles_all))/n_align/n_quantile, quantiles_all, n_align, n_quantile


def compute_WassKernel_stratified_improved(data, n_quantile = 100, metric='Euclidean',normalize=False):
    n = len(data)
    # _, quantiles, n_align, n_quantile = compute_WassKernel_stratified(data, n_align = n_quantile, n_quantile = n_quantile, metric='Euclidean',normalize=False, return_distance=True, return_shape=True)
    # for i in range(n):
    #     quantiles[i] = quantiles[i].reshape((n_align, n_quantile))


    quantiles = [0]*n
    for i in range(n):
        C1 = sp.spatial.distance.cdist(data[i], data[i])
        if normalize:
            C1 = C1/np.median(C1)
        n_align = C1.shape[0]
        if n_quantile > 100:
            n_quantile = 100
        # quantiles[i] = distance_matrix_quantiles(C1, n_1=n_align, n_2=n_quantile)
        quantiles[i] = np.zeros((C1.shape[0],n_quantile+1))
        for j in range(C1.shape[0]):
            quantiles[i][j,:] = np.quantile(C1[j,:], np.linspace(0, 1, n_quantile+1))
            
        
    
    distances = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            print(i)
            print(j)
            print(quantiles[i].shape)
            print(quantiles[j].shape)
            M = sp.spatial.distance.cdist(quantiles[i], quantiles[j])
            
            a = np.ones((quantiles[i].shape[0],))
            a = a/a.sum()
            b = np.ones((quantiles[j].shape[0],))
            b = b/b.sum()
            distances[i,j] = ot.emd2(a, b, M)
            distances[j,i] = distances[i,j]


    return distances





def mmd2_permutation(joint_kernel, n_X, n_perm=500, u_stat=False):

    """
    joint_kernel: should be an array of shape [n_X + n_Y, n_X + n_Y] with kernel values,
                  ie   [ K_XX  K_XY ]
                       [ K_YX  K_YY ]
    n_X: number of entries in the first set
    n_perm: total number of permutations, including the identity

    If biased is True, uses the plug-in estimator (MMD between empirical distributions).
    If False, it uses the U-statistic estimator, which is unbiased but drops k(x_i, y_i) terms.
    (I'm not sure how to implement the "unbiased estimator" (which includes those terms) efficiently.)
    """
    PermutationResult = namedtuple(
    "PermutationResult", ["estimate", "p_value", "permuted_estimates"]
)
    K = joint_kernel = torch.as_tensor(joint_kernel)
    device = K.device
    dtype = K.dtype

    n = K.shape[0]
    if K.shape != (n, n):
        raise ValueError(f"joint_kernel should be square, got {K.shape}")
    n_X = int(n_X)
    n_Y = n - n_X
    if n_X <= 0 or n_Y <= 0:
        raise ValueError("need a positive number of samples from each")

    if u_stat:
        if n_X != n_Y:
            raise ValueError("u-stat estimator only defined for equal sample sizes")
        w_X = 1
        w_Y = -1
    else:
        w_X = 1 / n_X
        w_Y = -1 / n_Y

    # construct permutations
    # there probably should be a faster way to do this but, idk
    perms = torch.stack(
        [torch.arange(n, device=device)]
        + [torch.randperm(n, device=device) for _ in range(n_perm - 1)]
    )
    X_inds = perms[:, :n_X]
    Y_inds = perms[:, n_X:]

    # set weights to w_X for things in X_inds, w_Y for others
    ws = torch.full((n_perm, n), w_Y, device=device, dtype=dtype)
    ws.scatter_(1, X_inds, w_X)

    # the "basic" estimate; either the biased est or a constant times it
    ests = torch.einsum("pi,ij,pj->p", ws, joint_kernel, ws)

    if u_stat:
        # need to subtract \sum_i k(X_i, X_i) + k(Y_i, Y_i) + 2 k(X_i, Y_i)
        # first two are just trace
        # for the last one, we need to see which ones were lined up
        # NOTE: this makes an unnecessary copy if joint_kernel isn't already contiguous,
        #       but this generally shouldn't be a big deal
        cross_terms = joint_kernel.take(X_inds * n + Y_inds).sum(1)
        ests = (ests - joint_kernel.trace() + 2 * cross_terms) / (n_X * (n_X - 1))

    p_val = (ests >= ests[0]).float().mean()
    return PermutationResult(ests[0], p_val, ests)



def hsic2_permutation(K, L, n_perm=500):
    """
    K: the pairwise kernel matrix of X values
    L: the pairwise kernel matrix of corresponding Y values
    n_perm: total number of permutations, including the identity
    """
    PermutationResult = namedtuple(
    "PermutationResult", ["estimate", "p_value", "permuted_estimates"]
)
    K = torch.as_tensor(K)
    device = K.device
    dtype = K.dtype
    L = torch.as_tensor(L).to(device=device, dtype=dtype)

    n = K.shape[0]
    if K.shape != (n, n):
        raise ValueError(f"K should be square, got {K.shape}")
    if L.shape != (n, n):
        raise ValueError(f"L should be same shape as K ({K.shape}), got {L.shape}")

    row_mean = K.mean(dim=0, keepdim=True)
    col_mean = K.mean(dim=1, keepdim=True)
    HKH_flat = (K - row_mean - col_mean + K.mean()).ravel()

    L_flats = torch.empty((n_perm, n * n), device=device, dtype=dtype)
    L_flats[0, :] = L.ravel()
    for i in range(1, n_perm):
        perm = torch.randperm(n, device=device)
        L_flats[i, :] = L[perm.unsqueeze(1), perm.unsqueeze(0)].ravel()

    ests = L_flats @ HKH_flat

    p_val = (ests >= ests[0]).float().mean()
    return PermutationResult(ests[0], p_val, ests)


def hsic_permutation(K, L, n_perm=1000):
    """
    Computes the HSIC statistic with permutation testing using efficient einsum operations.

    Args:
        K (torch.Tensor): Kernel matrix for X (size n x n).
        L (torch.Tensor): Kernel matrix for Y (size n x n).
        n_perm (int): Number of permutations.

    Returns:
        (float, float, torch.Tensor): HSIC statistic, p-value, all permutation values.
    """
    PermutationResult = namedtuple(
    "PermutationResult", ["estimate", "p_value", "permuted_estimates"]
)
    # Ensure tensors
    K = torch.as_tensor(K)
    L = torch.as_tensor(L)
    device = K.device
    dtype = K.dtype

    # Check square matrices
    n = K.shape[0]
    if K.shape != (n, n) or L.shape != (n, n):
        raise ValueError("K and L must be square matrices of the same size")

    # Centering matrix: H = I - (1/n) * 1 * 1^T
    H = torch.eye(n, device=device, dtype=dtype) - (1 / n)

    # Compute centered kernels: Kc = H K H, Lc = H L H
    Kc = torch.einsum("ij,jk,kl->il", H, K, H)
    Lc = torch.einsum("ij,jk,kl->il", H, L, H)

    # Compute HSIC statistic: Tr(Kc * Lc) / (n-1)^2
    hsic_stat = torch.einsum("ij,ij->", Kc, Lc) / (n - 1) ** 2

    # Generate permutations
    perms = torch.stack([torch.arange(n, device=device)] + 
                        [torch.randperm(n, device=device) for _ in range(n_perm - 1)])

    # Permute L rows and columns
    L_perm = L[:, perms]  # Reorder columns
    L_perm = L_perm.permute(2, 0, 1)  # Move permutations to batch dimension

    # Compute centered permuted kernels: L_perm_c = H L_perm H
    L_perm_c = torch.einsum("ij,pjk,kl->pil", H, L_perm, H)

    # Compute permuted HSIC statistics: Tr(Kc * L_perm_c) / (n-1)^2
    perm_stats = torch.einsum("ij,pij->p", Kc, L_perm_c) / (n - 1) ** 2

    # Compute p-value: fraction of permuted HSIC values >= original HSIC
    p_val = (perm_stats >= hsic_stat).float().mean()

    return PermutationResult(hsic_stat.item(), p_val.item(), perm_stats)#hsic_stat.item(), p_val.item(), perm_stats

def compute_kernel_matrix(data, kernel_type='GW', normalize=False, bw='median', n_quantile=100, n_align=100, metric='Euclidean',norm_const=100, rank_power=1, eps=0):
    n = len(data)
    print(n)
    if norm_const is None:
        C1=sp.spatial.distance.cdist(data[0], data[0])
        norm_const = C1.max()
    K = np.zeros( (n,n))
    if kernel_type == 'W_mine':
        K = compute_WassKernel(data, metric=metric,normalize=normalize,n_pt=n_quantile)
        # K = K ** 2
    elif kernel_type == 'W_stratified':
        K = compute_WassKernel_stratified(data, metric=metric,normalize=normalize,n_quantile=n_quantile, n_align=n_align, rank_power=rank_power, eps=eps)
        # K = K ** 2
    elif kernel_type == 'W_block':
        K = compute_WassKernel_stratified(data, metric=metric,normalize=normalize,n_quantile=n_quantile, n_align=n_align, block=True)
    elif kernel_type == 'W_improved':
        K = compute_WassKernel_stratified_improved(data, metric=metric,normalize=normalize,n_quantile=n_align)
    elif kernel_type == 'GW' or kernel_type=='qGW':

        C = [0]*n
        # norm_const = 0
        for i in range(n):
            C[i] = [0]
            C[i] = sp.spatial.distance.cdist(data[i], data[i])
            # if C[i].max() > norm_const:
            #     norm_const = C[i].max()

        for i in range(n):
            print(i)
            if data[i].shape[0] == 0:
                K[i,:] = np.inf
                K[:,i] = np.inf
            for j in range(n):
                # print(i)
                # print(j)
                #print('computing pair '+str(i)+' and '+str(j) )
                if j > i and data[i].shape[0]>0 and data[j].shape[0]>0:
                    if kernel_type == 'GW':
                        if metric == 'Euclidean':
                            # C1 = sp.spatial.distance.cdist(data[i], data[i])
                            # C2 = sp.spatial.distance.cdist(data[j], data[j])
                            C1 = C[i]
                            C2 = C[j]
                        else:
                            Xgraph = construct_graph(data[i], k=3, mode="distance",metric="euclidean").todense()
                            C1 = init_distances(Xgraph)
                            Ygraph = construct_graph(data[j], k=3, mode="distance",metric="euclidean").todense()
                            C2 = init_distances(Ygraph)
                        if normalize == False:
                            # norm_c = np.max( [C1.max(), C2.max()] )
                            
                            C1 /= norm_const#C1.max()
                            C2 /= norm_const#C2.max()
                        else:
                            C1 /= C1.max()
                            C2 /= C2.max()
                        K[i,j] =  ot.gromov.gromov_wasserstein2( C1, C2)
                    else:
                        _,_,_,log =  ot.gromov.quantized_fused_gromov_wasserstein_samples(data[i],data[j],npart1=n_align,npart2=n_align,log=True)
                        K[i,j] = log['qFGW_dist']       
                        
                    
                    K[j,i] = K[i,j]
            
    return K

def permutation_test_mmd(K_XX, K_YY, K_XY,  n_permutations=100000):
    """
    Performs a permutation test based on MMD with precomputed kernel matrices.

    Args:
    - K_XX (np.ndarray): Kernel matrix for samples from distribution X (m x m).
    - K_YY (np.ndarray): Kernel matrix for samples from distribution Y (n x n).
    - K_XY (np.ndarray): Kernel matrix between samples from X and Y (m x n).
    - n_permutations (int): Number of permutations to perform.

    Returns:
    - p-value (float): The p-value from the permutation test.
    - true_mmd (float): The MMD value for the original data.
    - mmd_values (np.ndarray): The MMD values from the permuted samples.
    """
    m = K_XX.shape[0]  # Number of samples in X
    n = K_YY.shape[0]  # Number of samples in Y
    

    # Combine the indices of the kernel matrices
    combined_size = m + n
    indices = np.arange(combined_size)

    # Compute the true MMD value
    true_mmd = compute_mmd(K_XX, K_YY, K_XY)

    # Combine all the kernel values (XX, YY, XY) into a single matrix
    combined_kernel = np.block([
        [K_XX, K_XY],
        [K_XY.T, K_YY]
    ])


    # Perform permutation test
    mmd_values = []
    for _ in range(n_permutations):
        # Permute the indices
        permuted_indices = np.random.permutation(combined_size)

        # Extract permuted blocks for X and Y
        perm_X_indices = permuted_indices[:m]  # First m elements
        perm_Y_indices = permuted_indices[m:]  # Last n elements

        # Get the permuted kernel matrices
        K_perm_XX = combined_kernel[np.ix_(perm_X_indices, perm_X_indices)]
        K_perm_YY = combined_kernel[np.ix_(perm_Y_indices, perm_Y_indices)]
        K_perm_XY = combined_kernel[np.ix_(perm_X_indices, perm_Y_indices)]

        # Compute MMD for the permuted data
        perm_mmd = compute_mmd(K_perm_XX, K_perm_YY, K_perm_XY)
        mmd_values.append(perm_mmd)

    # Convert to array
    mmd_values = np.array(mmd_values)

    # Compute p-value: proportion of permuted MMDs greater than or equal to true MMD
    p_value = np.mean(mmd_values >= true_mmd)

    return p_value, true_mmd, mmd_values


def hsic_statistic(K, L):
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n  # Centering matrix
    HKH = H @ K @ H
    HMH = H @ L @ H
    return np.trace(HKH @ HMH) / (n - 1)**2

def hsic_permutation_test(K, L, num_permutations=100, seed=None):
    """
    Perform a permutation test for HSIC.
    
    Parameters:
        K (np.ndarray): Kernel matrix for the first variable (n x n).
        L (np.ndarray): Kernel matrix for the second variable (n x n).
        num_permutations (int): Number of permutations.
        seed (int): Random seed for reproducibility.
    
    Returns:
        float: p-value from the permutation test.
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = K.shape[0]
    assert K.shape == (n, n) and L.shape == (n, n), "Kernel matrices must be square and have the same dimensions."
    
    # Compute observed HSIC
    observed_hsic = hsic_statistic(K, L)
    
    # Permutation test
    permuted_hsics = np.zeros(num_permutations)
    for i in range(num_permutations):
        perm = np.random.permutation(n)
        permuted_hsics[i] = hsic_statistic(K, L[perm][:, perm])
    
    # Calculate p-value
    p_value = (np.sum(permuted_hsics >= observed_hsic) + 1) / (num_permutations + 1)
    
    return p_value



def compute_witness_function(idx,K_XX, K_XY):
    return np.mean(K_XX[idx,:]) - np.mean(K_XY[idx,:])

def compute_witness_function_hsic(K, L):
    n = K.shape[0]
    

    H = np.eye(n) - np.ones((n, n)) / n
    K = H @ K @ H
    L = H @ L @ H

    # Element-wise product of the centered kernel matrices
    K_hadamard_L = K * L
    
    # Witness function: sum across rows
    witness = np.sum(K_hadamard_L, axis=1) / n
    
    return witness




# HSIC computation between two sets of variables X and Y
def compute_hsic(C_X, Y, sigma_X = 1.0, sigma_Y=1.0):
    # Centering matrix
    n = C_X.shape[0]
    H = torch.eye(n) - torch.ones((n, n)) / n
    C_X = torch.tensor(C_X)
    Y = torch.tensor(Y)
    # Compute RBF kernels
    K_X = torch.exp( - C_X / abs(sigma_X) )
    K_Y = rbf(Y, sigma_Y)
    
    # Center the kernels
    K_X_centered = H @ K_X @ H
    K_Y_centered = H @ K_Y @ H
    
    # HSIC value (normalized by 1/(n-1)^2 for unbiased estimate)
    hsic_value = torch.trace(K_X_centered @ K_Y_centered) / (n - 1) ** 2
    return hsic_value

# Minimize HSIC with respect to Y
def maximize_hsic(C_X, Y, sigma_X=1.0, sigma_Y = 1.0, learning_rate=0.01, num_steps=1000, change_sx=False):
    # Convert numpy arrays to torch tensors
    C_X = torch.tensor(C_X, dtype=torch.float32, requires_grad=False)
    
    
    
    Y = torch.tensor(Y, dtype=torch.float32, requires_grad=True)  # We optimize Y
    if change_sx:
        # Optimizer to minimize Y
        sigma_X = torch.tensor( sigma_Y, dtype = torch.float32, requires_grad = True)
        optimizer = optim.Adam([Y, sigma_X], lr=learning_rate)
    else:
        optimizer = optim.Adam([Y], lr=learning_rate)
   
    # Optimization loop
    for step in range(num_steps):
        optimizer.zero_grad()
        
        # Compute HSIC
        hsic_value = -compute_hsic(C_X, Y, sigma_X, sigma_Y) - torch.abs(torch.std( Y )-1) * 0.001 #0.0000*compute_hsic( rbf(Y, sigma_Y), Y, sigma_X, sigma_Y )   #torch.std( Y ) * 0.00001
        
        # Backpropagate to compute gradients
        hsic_value.backward()
        
        # Take a gradient step
        optimizer.step()
        
        # Print progress
        if step % 100 == 0:
            print(f'Step {step}: HSIC = {hsic_value.item()}')
            print(sigma_X)    
    return Y.detach().numpy()
                    
            
  



import numpy as np
import scipy.stats as stats

def split_gram_matrix(K, m):
    """
    Splits the Gram matrix into the two-sample components.
    Args:
        K (ndarray): (m+n) x (m+n) Gram matrix.
        m (int): Number of samples in population 1.
    Returns:
        K_XX, K_YY, K_XY
    """
    K_XX = K[:m, :m]  # Sample 1 similarities
    K_YY = K[m:, m:]  # Sample 2 similarities
    K_XY = K[:m, m:]  # Cross similarities
    return K_XX, K_YY, K_XY


def gamma_approximation(K_XX, K_YY, K_XY):
    """
    Computes the Gamma approximation p-value for MMD².
    """
    m, n = K_XX.shape[0], K_YY.shape[0]
    K = np.block([[K_XX, K_XY], [K_XY.T, K_YY]])
    H = np.eye(m + n) - np.ones((m + n, m + n)) / (m + n)
    Kc = H @ K @ H  # Centered Gram matrix
    eigvals = np.linalg.eigvalsh(Kc) / (m + n)
    alpha = np.sum(eigvals) ** 2 / (2 * np.sum(eigvals ** 2))
    beta = (2 * np.sum(eigvals ** 2)) / np.sum(eigvals)
    mmd2 = compute_mmd(K_XX, K_YY, K_XY)
    p_value = 1 - stats.gamma.cdf(mmd2, alpha, scale=beta)
    return p_value

def spectral_p_value(K_XX, K_YY, K_XY, num_samples=1000):
    """
    Computes p-value using the spectral method.
    """
    m, n = K_XX.shape[0], K_YY.shape[0]
    K = np.block([[K_XX, K_XY], [K_XY.T, K_YY]])
    H = np.eye(m + n) - np.ones((m + n, m + n)) / (m + n)
    Kc = H @ K @ H
    eigvals = np.linalg.eigvalsh(Kc) / (m + n)
    mmd2 = compute_mmd(K_XX, K_YY, K_XY)
    null_samples = np.sum(eigvals[:, None] * np.random.normal(size=(len(eigvals), num_samples)) ** 2, axis=0)
    p_value = np.mean(null_samples > mmd2)
    return p_value

def wild_bootstrap_p_value(K_XX, K_YY, K_XY, num_bootstrap=1000):
    """
    Computes p-value using wild bootstrap.
    """
    m, n = K_XX.shape[0], K_YY.shape[0]
    mmd2 = compute_mmd(K_XX, K_YY, K_XY)
    null_samples = []
    for _ in range(num_bootstrap):
        signs = np.random.choice([-1, 1], size=m + n, replace=True)
        K_boot = np.outer(signs, signs) * np.block([[K_XX, K_XY], [K_XY.T, K_YY]])
        K_XX_b, K_YY_b, K_XY_b = split_gram_matrix(K_boot, m)
        null_samples.append(compute_mmd(K_XX_b, K_YY_b, K_XY_b))
    p_value = np.mean(np.array(null_samples) > mmd2)
    return p_value

def compute_p_values(K_XX, K_YY, K_XY, method="gamma"):
    """
    Computes p-values using the chosen method.
    """
    # K_XX, K_YY, K_XY = split_gram_matrix(K, m)
    # print(method)

    if method == "gamma":
        # print('gamma')
        return gamma_approximation(K_XX, K_YY, K_XY)
    elif method == "spectral":
        return spectral_p_value(K_XX, K_YY, K_XY)
    elif method == "wild_bootstrap":
        return wild_bootstrap_p_value(K_XX, K_YY, K_XY)
    elif method == 'permutation':
        p_value,_,_ = permutation_test_mmd(K_XX, K_YY, K_XY)
        return p_value
        # raise ValueError("Unknown method")
                
def hsic_p_value(K, L, method="gamma", num_samples=1000):
    """
    Computes HSIC p-value using different methods.
    """
    hsic_value = hsic_statistic(K, L)
    if method == "gamma":
        eigvals_K = np.linalg.eigvalsh(K)
        eigvals_L = np.linalg.eigvalsh(L)
        alpha = np.sum(eigvals_K) * np.sum(eigvals_L) / (2 * np.sum(eigvals_K * eigvals_L))
        beta = (2 * np.sum(eigvals_K * eigvals_L)) / (np.sum(eigvals_K) * np.sum(eigvals_L))
        return 1 - stats.gamma.cdf(hsic_value, alpha, scale=beta)
    elif method == "wild_bootstrap":
        null_samples = []
        for _ in range(num_samples):
            signs = np.random.choice([-1, 1], size=K.shape[0], replace=True)
            K_boot = np.outer(signs, signs) * K
            L_boot = np.outer(signs, signs) * L
            null_samples.append(hsic_statistic(K_boot, L_boot))
        return np.mean(np.array(null_samples) > hsic_value)
    else:
        return hsic_permutation_test(K, L)


def mmd2_u_stat(K_XX, K_XY, K_YY):
    n = K_XX.shape[0]
    m = K_YY.shape[0]

    # Remove diagonals
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)

    term_xx = K_XX.sum() / (n * (n - 1))
    term_yy = K_YY.sum() / (m * (m - 1))
    term_xy = K_XY.sum() / (n * m)

    return term_xx + term_yy - 2 * term_xy

def hoeffding_variance(K_XX, K_XY, K_YY):
    n = K_XX.shape[0]
    m = K_YY.shape[0]

    # Expectation over y for each x
    mu_X = K_XY.mean(axis=1)  # shape (n,)
    mu_XX = (K_XX.sum(axis=1) - np.diag(K_XX)) / (n - 1)

    # Expectation over x for each y
    mu_Y = K_XY.mean(axis=0)  # shape (m,)
    mu_YY = (K_YY.sum(axis=1) - np.diag(K_YY)) / (m - 1)

    h_X = mu_XX - mu_X
    h_Y = mu_YY - mu_Y

    var_X = np.var(h_X, ddof=1)
    var_Y = np.var(h_Y, ddof=1)

    return (4 / n) * var_X + (4 / m) * var_Y

def hsic1_unbiased(K, L):
    """
    Compute the unbiased HSIC_1 estimator (Eq. 5 from the paper).

    Parameters:
        K (ndarray): Kernel matrix for X (m x m)
        L (ndarray): Kernel matrix for Y (m x m)

    Returns:
        float: Unbiased HSIC_1 estimate
    """
    m = K.shape[0]
    assert K.shape == L.shape, "Kernel matrices must have the same shape"

    # Zero the diagonals: K_tilde = K - diag(K)
    K_tilde = K.copy()
    L_tilde = L.copy()
    np.fill_diagonal(K_tilde, 0)
    np.fill_diagonal(L_tilde, 0)

    one = np.ones((m, 1))

    # Terms in HSIC_1 formula
    trace_term = np.trace(K_tilde @ L_tilde)
    sum_K = one.T @ K_tilde @ one
    sum_L = one.T @ L_tilde @ one
    cross_term = one.T @ K_tilde @ L_tilde @ one

    # Final HSIC_1 estimator
    hsic1 = (1 / (m * (m - 3))) * (
        trace_term +
        (sum_K * sum_L)[0, 0] / ((m - 1) * (m - 2)) -
        (2 / (m - 2)) * cross_term[0, 0]
    )
    if hsic1 < 0:
        hsic1 = 0
    return hsic1

def hsic1_variance_estimate(K, L, hsic1_value):
    """
    Estimate variance of HSIC_1 using fast U-statistic approximation.

    Parameters:
        K_tilde (ndarray): Centered kernel matrix for X (m x m)
        L_tilde (ndarray): Centered kernel matrix for Y (m x m)
        hsic1_value (float): Previously computed unbiased HSIC_1 estimate

    Returns:
        float: Estimated variance σ²_HSIC₁
    """
    K_tilde = K.copy()
    L_tilde = L.copy()
    np.fill_diagonal(K_tilde, 0)
    np.fill_diagonal(L_tilde, 0)

    m = K_tilde.shape[0]
    ones = np.ones((m, 1))

    # Elementwise product
    KL = K_tilde * L_tilde  # (m x m)

    # Compute components of h
    term1 = (m - 2)**2 * KL @ ones
    term2 = (m - 2) * (
        np.trace(K_tilde @ L_tilde) * ones -
        K_tilde @ L_tilde @ ones -
        L_tilde @ K_tilde @ ones
    )

    K1 = K_tilde @ ones
    L1 = L_tilde @ ones
    term3 = -m * (K1 * L1)


    term4 = (ones.T @ L_tilde @ ones)[0, 0] * K_tilde @ ones
    term5 = (ones.T @ K_tilde @ ones)[0, 0] * L_tilde @ ones
    term6 = (ones.T @ K_tilde @ L_tilde @ ones)[0, 0] * ones

    # Assemble h
    h = term1 + term2 + term3 + term4 + term5 - term6  # (m x 1)

    # Compute R
    denom = 4 * m * ((m - 1)*(m - 2)*(m - 3))**2
    R = np.sum(h ** 2) / denom

    # Final variance
    variance = (16 / m) * (R - hsic1_value**2)
    if variance < 0:
        variance = 0
    return variance
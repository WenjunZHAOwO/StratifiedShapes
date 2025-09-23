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


# Function to compute the RBF kernel in PyTorch
def rbf(X, sigma):
    pairwise_dists = torch.cdist(X, X, p=2) ** 2
    K = torch.exp(-pairwise_dists / (2 * sigma ** 2))
    return K


def construct_graph(X, k, mode= "connectivity", metric="correlation"):
	assert (mode in ["connectivity", "distance"]), "Norm argument has to be either one of 'connectivity', or 'distance'. "
	if mode=="connectivity":
		include_self=True
	else:
		include_self=False

	Xgraph=kneighbors_graph(X, k, mode=mode, metric=metric, include_self=include_self)
	#self.ygraph=kneighbors_graph(self.y, k, mode=mode, metric=metric, include_self=include_self)

	return Xgraph
def init_distances(Xgraph):
	# Compute shortest distances
	X_shortestPath=dijkstra(csgraph= csr_matrix(Xgraph), directed=False, return_predecessors=False)
	#y_shortestPath=dijkstra(csgraph= csr_matrix(self.ygraph), directed=False, return_predecessors=False)

	# Deal with unconnected stuff (infinities):
	X_max=np.nanmax(X_shortestPath[X_shortestPath != np.inf])
	#y_max=np.nanmax(y_shortestPath[y_shortestPath != np.inf])
	X_shortestPath[X_shortestPath > X_max] = X_max
	#y_shortestPath[y_shortestPath > y_max] = y_max

		# Finnally, normalize the distance matrix:
	Cx=X_shortestPath/X_shortestPath.max()
	#self.Cy=y_shortestPath/y_shortestPath.max()

	return Cx

def compute_diffusion_map(gram_matrix, n_components=2, alpha=0.5):
    """
    Computes the diffusion map from a Gram (similarity) matrix.

    Args:
    - gram_matrix (np.ndarray): The Gram (similarity) matrix.
    - n_components (int): The number of dimensions (eigenvectors) to extract (e.g., 2 for 2D).
    - alpha (float): Normalization parameter. alpha=0.5 corresponds to the normalized Laplacian.

    Returns:
    - embedding (np.ndarray): The diffusion map embedding (n_samples x n_components).
    - eigenvalues (np.ndarray): The corresponding eigenvalues.
    """
    # Step 1: Normalize the Gram matrix
    degrees = np.sum(gram_matrix, axis=1)
    D_alpha = np.diag(degrees**(-alpha))
    diffusion_matrix = D_alpha @ gram_matrix @ D_alpha
    
    # Step 2: Eigen decomposition of the diffusion matrix
    eigenvalues, eigenvectors = eigh(diffusion_matrix)
    
    # Step 3: Sort eigenvectors by eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 4: Take the top n_components eigenvectors as the diffusion map embedding
    embedding = eigenvectors[:, 1:n_components + 1]  # Skip the first eigenvector (corresponding to eigenvalue 1)
    
    return embedding, eigenvalues[1:n_components + 1]

def compute_dimension_reduction_wasskernel(data, n_dim=2, normalize=False):
    n = len(data)
    
    data_reduc = np.zeros( (n,n_dim) )
    weights = np.linspace( 0,1, n_dim + 2 )
    weights = weights[1:n_dim+1]
    print(weights)
    for i in range(n):
        
        
        
        C1 = sp.spatial.distance.cdist(data[i], data[i])
                
        if normalize == True:
            C1 /= C1.max()


        for k in range(n_dim):
            data_reduc[i,k] = np.quantile(C1.ravel(), weights[k])
    return data_reduc

def is_valid_distance_matrix(D):
    n = D.shape[0]
    
    # Check square matrix
    if D.shape[0] != D.shape[1]:
        return False, "Matrix is not square."
    
    # Check non-negativity and zero diagonal
    if not np.all(D >= 0):
        return False, "Negative distances detected."
    
    if not np.all(np.diag(D) == 0):
        return False, "Diagonal entries must be zero."
    
    # Check symmetry
    if not np.allclose(D, D.T):
        return False, "Matrix is not symmetric."
    
    # Check triangle inequality
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if D[i, j] > D[i, k] + D[k, j]:
                    return False, f"Triangle inequality violated at ({i}, {j}, {k})."
    
    return True, "Matrix is a valid distance matrix."   
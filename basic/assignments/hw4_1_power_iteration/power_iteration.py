import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray - symmetric diagonalizable real-valued matrix
    num_steps: int - number of power method steps
    
    Returns:
    eigenvalue: float - dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray - corresponding eigenvector estimation
    """
    assert data is not None
    assert num_steps > 0
    
    eigenvector = np.random.uniform(size=data.shape[1])
    eigenvalue = None
    for i in range(num_steps):
        data_eigenvector = data @ eigenvector
        eigenvector = data_eigenvector / np.linalg.norm(data_eigenvector)
        eigenvalue = eigenvector @ data_eigenvector / (eigenvector @ eigenvector)
    return float(eigenvalue), eigenvector
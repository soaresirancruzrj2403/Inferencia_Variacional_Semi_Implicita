
import numpy as np

class PMM:

    def __init__(self) -> None:
        
        pass

    def U(self, x: np.ndarray) -> np.ndarray:

        return x
    
    def A(self, eta: np.ndarray) -> float:

        return np.exp(eta)
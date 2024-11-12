
import numpy as np

class UGMM_Simples:

    def __init__(self, sigma:float) -> None:
        
        self.sigma = sigma

    def U(self, x: np.ndarray) -> np.ndarray:

        return x/self.sigma
    
    def A(self, eta: np.ndarray) -> float:

        return eta**2/2

import numpy as np

class UMMM:

    def __init__(self, NT: int) -> None:
        
        self.NT = NT

    def U(self, x: np.ndarray) -> np.ndarray:

        return x
    
    def A(self, eta: np.ndarray) -> float:

        return self.NT*np.log(1 + np.exp(eta))
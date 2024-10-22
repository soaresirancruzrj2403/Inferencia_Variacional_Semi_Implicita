
import numpy as np

class MCMC_VMP:

    def __init__(
            
        self, 
        
        Dados: np.ndarray,

        mu_0: np.ndarray, sigma_0: np.ndarray,

        alpha_0: np.ndarray, beta_0: np.ndarray

    ) -> None:

        self.X = Dados

        self.N = self.X.shape[0]

        self.mu_0 = mu_0

        self.sigma_0 = sigma_0

        self.m = 0

        self.sigma = 0

        self.mu = 0

        self.mu_quadrado = 0

        self.alpha_0 = alpha_0

        self.beta_0 = beta_0

        self.alpha = 0

        self.beta = 0

        self.gamma = 0

        self.ln_gamma = 0

    def atualiza_mu(self) -> None:

        self.mu_quadrado = np.random.normal(loc = self.m, scale = 1/self.sigma, size = 1000)

        self.mu = self.mu_quadrado.mean()

    def atualiza_mu_quadrado(self) -> None:

        self.mu_quadrado = np.random.normal(loc = self.m, scale = 1/self.sigma, size = 1000)

        self.mu_quadrado = self.mu_quadrado**2

        self.mu_quadrado = self.mu_quadrado.mean()

    def atualiza_gamma(self) -> None:

        self.gamma = np.random.gamma(shape = self.alpha, scale = 1/self.beta, size = 1000)

        self.gamma = self.gamma.mean()

    def atualiza_ln_gamma(self) -> None:

        self.ln_gamma = np.random.gamma(shape = self.alpha, scale = 1/self.beta, size = 1000)

        self.ln_gamma = np.log(self.ln_gamma)

        self.ln_gamma = self.ln_gamma.mean()

    def atualiza_sigma(self) -> None:

        self.sigma = (self.sigma_0 + self.N*self.gamma)

    def atualiza_m(self) -> None:

        self.m = self.sigma*self.mu_0 + self.gamma*self.X.sum()

        self.m = self.m/self.sigma

    def atualiza_beta(self) -> None:

        self.beta = self.beta_0

        self.beta += (self.X**2).sum()/2

        self.beta -= self.X.sum()*self.mu

        self.beta += self.N*self.mu_quadrado/2

    def atualiza_alpha(self) -> None:

        self.alpha = self.alpha_0 + self.N/2

    def inicializa_modelo(self) -> None:

        self.mu = np.random.uniform(low = self.X.min(), high = self.X.max())

        self.mu_quadrado = self.mu**2

        self.gamma = np.random.uniform(low = 0, high = self.X.max())

        self.ln_gamma = np.log(self.gamma)

    def passa_menssagens(self, max: int = 100) -> None:

        self.inicializa_modelo()

        for i in range(max):

            self.atualiza_sigma()

            self.atualiza_m()

            self.atualiza_mu()

            self.atualiza_mu_quadrado()

            self.atualiza_alpha()

            self.atualiza_beta()

            self.atualiza_gamma()

            self.atualiza_ln_gamma()
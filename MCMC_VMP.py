
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.mixture import GaussianMixture

class MCMC_VMP:

    def __init__(
            
        self,
        
        Dados: np.ndarray, Categorias: int,

        nu_0: float, chi_0: np.ndarray,

        Distribuicao, 
        
        Amostra: int = 10, Descartadas:int = 20

    ) -> None:
        
        # Dados observados
        
        self.X = Dados

        # Número de observações

        self.N = self.X.shape[0]

        # Número de categorias latentes

        self.K = Categorias

        # Parâmetros das categorias latentes

        self.r = np.zeros(shape = (self.N, self.K))

        self.N_barra = np.zeros(shape = (self.K))

        # Parâmetros a priori

        self.nu_0 = nu_0

        self.chi_0 = chi_0

        # Dimensão dos parâmetros

        self.D = self.chi_0.shape[0]

        # Estatística suficiente conjunta

        self.u_barra = np.zeros(shape = (self.K, self.N, self.D))

        # Parâmetros a priori a posteriori

        self.nu = np.zeros(shape = (self.K))

        self.chi = np.zeros(shape = (self.K, self.D))

        # Número de amostras consideradas

        self.M = Amostra

        # Número de amostras descartadas

        self.B = Descartadas

        # Amostras do parâmetro natural

        self.tau = np.zeros(shape = (self.K, self.M, self.D))

        # Estimador de bayes

        self.eta = np.zeros(shape = (self.K, self.D))

        # Classe de distribuições da família exponencial

        self.F = Distribuicao

    def inicializa_modelo(self) -> None:

        self.r = GaussianMixture(

            n_components = self.K, max_iter = 0, 
            
            init_params = 'k-means++'

        ).fit(self.X).predict_proba(self.X)

    def log_q(self, k:int, eta: np.ndarray) -> None:

        log_q = np.dot(eta, self.chi[k])

        log_q -= self.nu[k]*self.F.A(eta = eta)

        return log_q
    
    def atualiza_u_barra(self) -> None:

        for k in range(self.K):

            for n in range(self.N):

                self.u_barra[k][n] = self.F.U(self.X[n])

                self.u_barra[k][n] *= self.r[n][k]

    def atualiza_chi(self) -> None:

        self.chi = self.u_barra.sum(axis = 1)

        self.chi += self.chi_0.reshape((1, self.D))

    def atualiza_N_barra(self) -> None:

        self.N_barra = self.r.sum(axis = 0)

    def atualiza_nu(self) -> None:

        self.nu = self.nu_0 + self.N_barra

    def atualiza_tau(self) -> None:

        for k in range(self.K):

            self.tau[k] = tfp.mcmc.sample_chain(
            
                num_results = self.M, num_burnin_steps = self.B,

                current_state = np.ones(shape = self.D),

                kernel = tfp.mcmc.RandomWalkMetropolis(

                    target_log_prob_fn = lambda eta: self.log_q(k = k, eta = eta)

                ), trace_fn = None

            )

    def atualiza_eta(self) -> None:

        self.eta = self.tau.mean(axis = 1)

    def atualiza_r(self) -> None:

        for k in range(self.K):

            for n in range(self.N):

                self.r[n][k] = self.eta[k] @ self.F.U(x = self.X[n])

                self.r[n][k] -= self.F.A(eta = self.eta[k])

        self.r = np.exp(self.r)

        self.r /= self.r.sum(axis = 1).reshape((self.N, 1))

    def atualiza_modelo(self) -> None:

        self.atualiza_u_barra()

        self.atualiza_chi()

        self.atualiza_N_barra()

        self.atualiza_nu()

        self.atualiza_tau()

        self.atualiza_eta()

        self.atualiza_r()

    def estima_modelo(self, num: int = 30) -> None:

        self.inicializa_modelo()

        self.atualiza_modelo()

        for i in range(num):

            self.atualiza_modelo()

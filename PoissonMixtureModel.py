
import numpy as np

from sklearn.cluster import kmeans_plusplus

from scipy.special import psi

from scipy.special import softmax

class PoissonMixtureModel:

    def __init__(self, X : np.ndarray, M : int) -> None:
        
        self.X = X

        self.N = self.X.shape[0]

        self.M = M

        self.gamma = np.zeros(shape = (self.N, self.M))

        self.N_barra = np.zeros(shape = (self.M))

        self.T_barra = np.zeros(shape = (self.M))

        self.alpha_0 = 1/self.M

        self.rho_0 = 1

        self.tau_0 = 1

        self.alpha = np.zeros(shape = (self.M))

        self.rho = np.zeros(shape = (self.M))

        self.tau = np.zeros(shape = (self.M))

        self.eta = np.zeros(shape = (self.M))

        self.epsilon = 0

    def initialize_parameters(self) -> None:

        self.alpha = np.repeat(self.alpha_0, repeats = self.M)

        self.rho = np.repeat(self.rho_0, repeats = self.M)

        self.tau = np.repeat(self.tau_0, repeats = self.M)

        self.eta = np.squeeze(kmeans_plusplus(np.expand_dims(self.X, axis = 1), n_clusters = self.M)[0])

    def update_gamma(self) -> None:

        for m in range(self.M):

            for n in range(self.N):

                self.gamma[n, m] = psi(self.alpha[m]) - psi(self.alpha.sum())

                self.gamma[n, m] -= self.eta[m]

                self.gamma[n, m] += self.X[n]*(psi(self.tau[m]) - np.log(self.rho[m]))

        self.gamma = softmax(self.gamma, axis = 1)

    def update_N_barra(self) -> None:

        self.N_barra = self.gamma.sum(axis = 0)

    def update_alpha(self) -> None:

        self.alpha = self.alpha_0 + self.N_barra

    def update_rho(self) -> None:

        self.rho = self.rho_0 + self.N_barra

    def update_tau(self) -> None:

        self.tau = self.tau_0 + self.gamma.T @ self.X

    def update_eta(self) -> None:

        self.eta = self.tau/self.rho

    def update_parameters(self) -> None:

        self.update_gamma()

        self.update_N_barra()

        self.update_alpha()

        self.update_rho()

        self.update_tau()

        self.update_eta()

    def update_model(self, MAX : int = 1000, TOL : float = 1e-6) -> None:

        self.initialize_parameters()

        for i in range(MAX):

            self.epsilon = self.eta.astype(float)

            self.update_parameters()

            self.epsilon -= self.eta

            self.epsilon = np.linalg.norm(self.epsilon)

            if self.epsilon < TOL:

                break
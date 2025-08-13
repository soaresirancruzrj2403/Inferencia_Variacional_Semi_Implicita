
import numpy as np

from scipy.special import psi

from sklearn.cluster import kmeans_plusplus

from scipy.special import softmax

from scipy.stats import multivariate_t

class GaussianMixtureModel:

    def __init__(self, X : np.array, M : int):
        
        self.X = X

        self.N, self.D = self.X.shape

        self.M = M

        self.alpha_0 = 1/self.M

        self.tau_0 = 1

        self.mu_0 = np.zeros(shape = (self.D))

        self.nu_0 = self.D

        self.Sigma_0 = np.identity(n = self.D)

        self.Lambda_0 = np.linalg.inv(self.Sigma_0)

        self.E_log_pi = np.zeros(shape = (self.M))

        self.E_log_det_Lambda = np.zeros(shape = (self.M))

        self.gamma = np.zeros(shape = (self.N, self.M))

        self.N_barra = np.zeros(shape = (self.M))

        self.X_barra = np.zeros(shape = (self.M, self.D))

        self.S_barra = np.zeros(shape = (self.M, self.D, self.D))

        self.alpha = np.zeros(shape = (self.M))

        self.tau = np.zeros(shape = (self.M))

        self.mu = np.zeros(shape = (self.M, self.D))

        self.nu = np.zeros(shape = (self.M))

        self.Phi = np.zeros(shape = (self.M, self.D, self.D))

        self.Psi = np.zeros(shape = (self.M, self.D, self.D))

        self.pi = np.zeros(shape = (self.M))

        self.Z = np.zeros(shape = (self.N))

        self.nu = np.zeros(shape = (self.M))

        self.Sigma = np.zeros(shape = (self.M, self.D, self.D))

        self.Lambda = np.zeros(shape = (self.M, self.D, self.D))

        self.epsilon = 0

    def initialize_parameters(self) -> None:

        self.alpha = np.repeat(self.alpha_0, repeats = self.M)

        self.tau = np.repeat(self.tau_0, repeats = self.M)

        self.mu = kmeans_plusplus(self.X, n_clusters = self.M)[0]

        self.nu = np.repeat(self.nu_0, repeats = self.M)

        self.Phi = np.tile(self.Sigma_0, reps = (self.M, 1, 1))

        self.Psi = np.linalg.inv(self.Phi)

    def update_E_log_pi(self) -> None:

        self.E_log_pi = psi(self.alpha)

        self.E_log_pi -= psi(self.alpha.sum())

    def update_E_log_det_Lambda(self) -> None:

        self.E_log_det_Lambda = np.log(np.linalg.det(self.Psi))

        self.E_log_det_Lambda += self.D*np.log(2)

        for d in range(self.D):

            self.E_log_det_Lambda += psi((self.nu - d)/2)

    def update_gamma(self) -> None:

        for m in range(self.M):

            for n in range(self.N):

                self.gamma[n, m] = self.E_log_pi[m]

                self.gamma[n, m] += self.E_log_det_Lambda[m]

                self.gamma[n, m] -= (self.nu[m]*(self.X[n] - self.mu[m]).T @ self.Psi[m] @ (self.X[n] - self.mu[m]) + self.D/self.tau[m])/2

        self.gamma = softmax(self.gamma, axis = 1)

    def update_N_barra(self) -> None:

        self.N_barra = self.gamma.sum(axis = 0)

    def update_X_barra(self) -> None:

        self.X_barra = self.gamma.T @ self.X

        self.X_barra /= np.expand_dims(self.N_barra, axis = 1)

    def update_S_barra(self) -> None:

        self.S_barra.fill(0)

        for m in range(self.M):

            for n in range(self.N):

                self.S_barra[m] += self.gamma[n, m]*np.outer(self.X[n] - self.X_barra[m], self.X[n] - self.X_barra[m])

            self.S_barra[m] /= self.N_barra[m]

    def update_alpha(self) -> None:

        self.alpha = self.alpha_0 + self.N_barra

    def update_tau(self) -> None:

        self.tau = self.tau_0 + self.N_barra

    def update_mu(self) -> None:

        self.mu = np.expand_dims(self.N_barra, axis = 1)*self.X_barra

        self.mu -= self.tau_0*self.mu_0

        self.mu /= np.expand_dims(self.tau, axis = 1)

    def update_nu(self) -> None:

        self.nu = self.nu_0 + self.N_barra

    def update_Phi(self) -> None:

        for m in range(self.M):

            self.Phi[m] = self.Sigma_0

            self.Phi[m] += self.N_barra[m]*self.S_barra[m]

            self.Phi[m] += self.tau_0*self.N_barra[m]/self.tau[m]*np.outer(self.X_barra[m] - self.mu_0, self.X_barra[m] - self.mu_0)

    def update_Psi(self) -> None:

        self.Psi = np.linalg.inv(self.Phi)

    def update_parameters(self) -> None:

        self.update_E_log_pi()

        self.update_E_log_det_Lambda()

        self.update_gamma()

        self.update_N_barra()

        self.update_X_barra()

        self.update_S_barra()

        self.update_alpha()

        self.update_tau()

        self.update_mu()

        self.update_nu()

        self.update_Phi()

        self.update_Psi()

    def estimates_pi(self) -> None:

        self.pi = self.alpha/self.alpha.sum()

    def estimates_Z(self) -> None:

        self.Z = np.argmax(self.gamma, axis = 1)

    def estimates_Sigma(self) -> None:

        self.Sigma = self.Phi/(np.expand_dims(self.nu, axis = (1, 2)) + self.D + 1)

    def estimates_Lambda(self) -> None:

        self.Lambda = np.expand_dims(self.nu, axis = (1, 2))*self.Psi

    def estimates_parameters(self) -> None:

        self.estimates_pi()

        self.estimates_Z()

        self.estimates_Sigma()

        self.estimates_Lambda()

    def update_model(self, MAX : int = 1000, TOL : float = 1e-6) -> None:

        self.initialize_parameters()

        for i in range(MAX):

            self.epsilon = self.mu.copy()

            self.update_parameters()

            self.epsilon -= self.mu.copy()

            self.epsilon = np.linalg.norm(self.epsilon, axis = 1).max()

            if self.epsilon < TOL:

                break

        self.estimates_parameters()

    def predictive_distribution(self, X) -> None:

        density = 0

        for m in range(self.M):

            density += self.pi[m]*multivariate_t.pdf(

                x = X,

                loc = self.mu[m],

                df = self.nu[m] + 1 - self.D,

                shape = (1 + self.tau[m])/(self.tau[m]*(self.nu[m] + 1 - self.D))*self.Phi[m]

            )

        return density

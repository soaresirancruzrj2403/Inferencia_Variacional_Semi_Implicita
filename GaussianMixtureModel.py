
import numpy as np

from scipy.special import psi

from sklearn.cluster import kmeans_plusplus

from scipy.special import softmax

from scipy.special import loggamma

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

        for mu in range(self.M):

            for n in range(self.N):

                self.gamma[n, mu] = self.E_log_pi[mu]

                self.gamma[n, mu] += self.E_log_det_Lambda[mu]

                self.gamma[n, mu] -= (self.nu[mu]*(self.X[n] - self.mu[mu]).T @ self.Psi[mu] @ (self.X[n] - self.mu[mu]) + self.D/self.tau[mu])/2

        self.gamma = softmax(self.gamma, axis = 1)

    def update_N_barra(self) -> None:

        self.N_barra = self.gamma.sum(axis = 0)

    def update_X_barra(self) -> None:

        self.X_barra = self.gamma.T @ self.X

        self.X_barra /= np.expand_dims(self.N_barra, axis = 1)

    def update_S_barra(self) -> None:

        self.S_barra.fill(0)

        for mu in range(self.M):

            for n in range(self.N):

                self.S_barra[mu] += self.gamma[n, mu]*np.outer(self.X[n] - self.X_barra[mu], self.X[n] - self.X_barra[mu])

            self.S_barra[mu] /= self.N_barra[mu]

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

        for mu in range(self.M):

            self.Phi[mu] = self.Sigma_0

            self.Phi[mu] += self.N_barra[mu]*self.S_barra[mu]

            self.Phi[mu] += self.tau_0*self.N_barra[mu]/self.tau[mu]*np.outer(self.X_barra[mu] - self.mu_0, self.X_barra[mu] - self.mu_0)

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

    def ln_C_alpha_0(self) -> float:

        ln_C_alpha_0 = loggamma(self.alpha_0*self.M)

        ln_C_alpha_0 -= self.M*loggamma(self.alpha_0)

        return ln_C_alpha_0
    
    def ln_C_alpha(self) -> float:

        ln_C_alpha = loggamma(self.alpha.sum())

        ln_C_alpha -= loggamma(self.alpha).sum()

        return ln_C_alpha
    
    def ln_B_V_0_nu_0(self) -> float:

        ln_B_V_0_nu_0 = (self.nu_0 - np.arange(stop = self.D))/2

        ln_B_V_0_nu_0 = -loggamma(ln_B_V_0_nu_0).sum()

        ln_B_V_0_nu_0 -= self.D*(self.D - 1)/4*np.log(np.pi)

        ln_B_V_0_nu_0 -= self.nu_0*self.D/2*np.log(2)

        ln_B_V_0_nu_0 -= self.nu_0/2*np.log(np.linalg.det(self.Lambda_0))

        return ln_B_V_0_nu_0
    
    def ln_B_V_nu(self) -> np.ndarray:

        ln_B_V_nu = np.add.outer(self.nu, -np.arange(stop = self.D))/2

        ln_B_V_nu = -loggamma(ln_B_V_nu).sum(axis = 1)

        ln_B_V_nu -= self.D*(self.D - 1)/4*np.log(np.pi)

        ln_B_V_nu -= self.nu*self.D/2*np.log(2)

        ln_B_V_nu -= self.nu/2*np.log(np.linalg.det(self.Psi))

        return ln_B_V_nu

    def H_Lambda(self) -> np.ndarray:

        H_Lambda = -self.ln_B_V_nu()

        H_Lambda -= (self.nu - self.D - 1)/2*self.E_log_det_Lambda

        H_Lambda += self.nu*self.D/2

        return H_Lambda
    
    def atualiza_E_ln_p_pi(self) -> float:

        E_ln_p_pi = self.ln_C_alpha_0()

        E_ln_p_pi += (self.alpha_0 - 1)*self.E_log_pi.sum()

        return E_ln_p_pi
    
    def atualiza_E_ln_p_Z(self) -> float:

        E_ln_p_Z = self.gamma @ self.E_log_pi

        E_ln_p_Z = E_ln_p_Z.sum()

        return E_ln_p_Z
    
    def atualiza_E_ln_p_mu_Lambda(self) -> float:

        E_ln_p_mu_Lambda = np.einsum('kd, kdd, kd -> k', self.mu - self.mu_0, self.Psi, self.mu - self.mu_0)

        E_ln_p_mu_Lambda *= -self.tau_0*self.nu

        E_ln_p_mu_Lambda = self.D*np.log(self.tau_0/(2*np.pi))

        E_ln_p_mu_Lambda += (self.nu_0 - self.D)*self.E_log_det_Lambda

        E_ln_p_mu_Lambda -= self.D*self.tau_0/self.tau

        E_ln_p_mu_Lambda += self.ln_B_V_0_nu_0()

        E_ln_p_mu_Lambda -= self.nu*np.trace(self.Sigma_0 @ self.Psi, axis1 = 1, axis2 = 2)

        E_ln_p_mu_Lambda = E_ln_p_mu_Lambda.sum()/2

        return E_ln_p_mu_Lambda

    def atualiza_E_ln_p_X(self) -> float:

        E_ln_p_X = self.E_log_det_Lambda

        E_ln_p_X -= self.D/self.tau

        E_ln_p_X -= self.nu*np.trace(self.S_barra @ self.Psi, axis1 = 1, axis2 = 2)

        E_ln_p_X -= self.nu*np.einsum('kd, kdd, kd -> k', self.X_barra - self.mu, self.Psi, self.X_barra - self.mu)

        E_ln_p_X -= self.D*np.log(2*np.pi)

        E_ln_p_X *= self.N_barra

        E_ln_p_X = E_ln_p_X.sum()/2

        return E_ln_p_X
    
    def atualiza_E_ln_q_pi(self) -> float:

        E_ln_q_pi = (self.alpha - 1)*self.E_log_pi

        E_ln_q_pi += self.ln_C_alpha()

        E_ln_q_pi = E_ln_q_pi.sum()

        return E_ln_q_pi
    
    def atualiza_E_ln_q_Z(self) -> float:

        E_ln_q_Z = self.gamma*np.log(self.gamma)

        E_ln_q_Z = E_ln_q_Z.sum()

        return E_ln_q_Z
    
    def atualiza_E_ln_q_mu_Lambda(self) -> float:

        E_ln_q_mu_Lambda = self.E_log_det_Lambda/2

        E_ln_q_mu_Lambda += self.D/2*np.log(self.tau/(2*np.pi))

        E_ln_q_mu_Lambda -= self.D/2

        E_ln_q_mu_Lambda -= self.H_Lambda()

        E_ln_q_mu_Lambda = E_ln_q_mu_Lambda.sum()

        return E_ln_q_mu_Lambda
    
    def atualiza_ELBO(self) -> float:

        ELBO = self.atualiza_E_ln_p_pi()

        ELBO += self.atualiza_E_ln_p_Z()

        ELBO += self.atualiza_E_ln_p_mu_Lambda()

        ELBO += self.atualiza_E_ln_p_X()

        ELBO += self.atualiza_E_ln_q_pi()

        ELBO += self.atualiza_E_ln_q_Z()

        ELBO += self.atualiza_E_ln_q_mu_Lambda()

        return ELBO

    def predictive_distribution(self, X) -> None:

        density = 0

        for mu in range(self.M):

            density += self.pi[mu]*multivariate_t.pdf(

                x = X,

                loc = self.mu[mu],

                df = self.nu[mu] + 1 - self.D,

                shape = (1 + self.tau[mu])/(self.tau[mu]*(self.nu[mu] + 1 - self.D))*self.Phi[mu]

            )

        return density

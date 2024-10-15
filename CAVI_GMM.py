
import numpy as np
import scipy as spy
from sklearn.mixture import GaussianMixture

class CAVI_GMM:

    def __init__(self, Dados: np.ndarray, categorias: int) -> None:
        
        # Dados observados
        
        self.X = Dados

        #

        self.N, self.D = self.X.shape

        # Número de categorias latentes

        self.K = categorias

        #

        self.r = np.zeros(shape = (self.N, self.K))

        # Estatísticas suficientes

        self.N_barra = np.zeros(shape = (self.K))

        self.X_barra = np.zeros(shape = (self.K, self.D))

        self.S_barra = np.zeros(shape = (self.K, self.D, self.D))

        # Parâmetros a priori

        self.alpha_0 = 1/self.K

        self.kappa_0 = 1e-6

        self.mu_0 = np.zeros(shape = (self.D))

        self.nu_0 = self.D

        self.W_0 = np.cov(self.X.T)

        self.V_0 = np.linalg.inv(self.W_0)

        # Parâmetros a posteriori

        self.alpha = np.zeros(shape = (self.K))

        self.kappa = np.zeros(shape = (self.K))

        self.m = np.zeros(shape = (self.K, self.D))

        self.nu = np.zeros(shape = (self.K))

        self.W = np.zeros(shape = (self.K, self.D, self.D))

        self.V = np.zeros(shape = (self.K, self.D, self.D))

        # Estimadores de Bayes

        self.pi = np.zeros(shape = (self.K))

        self.Z = np.zeros(shape = (self.N))

        self.mu = np.zeros(shape = (self.K, self.D))

        self.Sigma = np.zeros(shape = (self.K, self.D, self.D))

        self.Lambda = np.zeros(shape = (self.K, self.D, self.D))

        #

        self.ELBO = []

    def inicializa_modelo(self) -> None:

        GMM_Inicial = GaussianMixture(n_components = self.K, max_iter = 0)

        GMM_Inicial.fit(X = self.X)

        self.r = GMM_Inicial.predict_proba(X = self.X)

    def E_ln_det_Lambda(self) -> np.ndarray:
        
        E_ln_det_Lambda = np.add.outer(self.nu, -np.arange(stop = self.D))/2

        E_ln_det_Lambda = spy.special.psi(E_ln_det_Lambda).sum(axis = 1)

        E_ln_det_Lambda += self.D*np.log(2)

        E_ln_det_Lambda += np.log(np.linalg.det(self.V))

        return E_ln_det_Lambda
    
    def E_ln_pi(self) -> np.ndarray:

        E_ln_pi = spy.special.psi(self.alpha)

        E_ln_pi -= spy.special.psi(self.alpha.sum())

        return E_ln_pi
    
    def atualiza_N_barra(self) -> None:
        
        self.N_barra = self.r.sum(axis = 0)

    def atualiza_X_barra(self) -> None:

        self.X_barra = self.r.T @ self.X

        self.X_barra /= self.N_barra.reshape((self.K, 1))

    def atualiza_S_barra(self) -> None:

        for k in range(self.K):

            S_k = 0

            for n in range(self.N):

                S_nk = self.X[n] - self.X_barra[k]

                S_nk = np.outer(S_nk, S_nk)
                
                S_k += self.r[n][k]*S_nk

            self.S_barra[k] = S_k/self.N_barra[k]

    def atualiza_alpha(self) -> None:

        self.alpha = self.alpha_0 + self.N_barra

    def atualiza_kappa(self) -> None:

        self.kappa = self.kappa_0 + self.N_barra

    def atualiza_m(self) -> None:

        self.m = self.N_barra.reshape((self.K, 1))*self.X_barra

        self.m += self.kappa_0*self.mu_0

        self.m /= self.kappa.reshape((self.K, 1))

    def atualiza_nu(self) -> None:

        self.nu = self.nu_0 + self.N_barra

    def atualiza_W(self) -> None:

        self.W = np.einsum('kD, kd -> kDd', self.X_barra - self.mu_0, self.X_barra - self.mu_0)
            
        self.W *= (self.kappa_0*self.N_barra/self.kappa).reshape((self.K, 1, 1))

        self.W += self.N_barra.reshape((self.K, 1, 1))*self.S_barra

        self.W += self.W_0.reshape((1, self.D, self.D))

    def atualiza_V(self) -> None:

        self.V = np.linalg.inv(self.W)

    def atualiza_r(self) -> None:

        self.r = -np.einsum('nd, kdd, nd -> nk', self.X, self.V, self.X)

        self.r += 2*np.einsum('kd, kdd, nd -> nk', self.m, self.V, self.X)

        self.r -= np.einsum('kd, kdd, kd -> k', self.m, self.V, self.m)

        self.r *= self.nu/2

        self.r -= self.D/(2*self.kappa)

        self.r += self.E_ln_det_Lambda()/2

        self.r += self.E_ln_pi()

        self.r = np.exp(self.r)

        self.r = self.r/self.r.sum(axis = 1).reshape((self.N, 1))
    
    def atualiza_modelo(self) -> None:

        self.atualiza_N_barra()

        self.atualiza_X_barra()

        self.atualiza_S_barra()

        self.atualiza_alpha()

        self.atualiza_kappa()

        self.atualiza_m()

        self.atualiza_nu()

        self.atualiza_W()

        self.atualiza_V()

        self.atualiza_r()

    def ln_C_alpha_0(self) -> float:

        ln_C_alpha_0 = spy.special.loggamma(self.alpha_0*self.K)

        ln_C_alpha_0 -= self.K*spy.special.loggamma(self.alpha_0)

        return ln_C_alpha_0
    
    def ln_C_alpha(self) -> float:

        ln_C_alpha = spy.special.loggamma(self.alpha.sum())

        ln_C_alpha -= spy.special.loggamma(self.alpha).sum()

        return ln_C_alpha
    
    def ln_B_V_0_nu_0(self) -> float:

        ln_B_V_0_nu_0 = (self.nu_0 - np.arange(stop = self.D))/2

        ln_B_V_0_nu_0 = -spy.special.loggamma(ln_B_V_0_nu_0).sum()

        ln_B_V_0_nu_0 -= self.D*(self.D - 1)/4*np.log(np.pi)

        ln_B_V_0_nu_0 -= self.nu_0*self.D/2*np.log(2)

        ln_B_V_0_nu_0 -= self.nu_0/2*np.log(np.linalg.det(self.V_0))

        return ln_B_V_0_nu_0
    
    def ln_B_V_nu(self) -> np.ndarray:

        ln_B_V_nu = np.add.outer(self.nu, -np.arange(stop = self.D))/2

        ln_B_V_nu = -spy.special.loggamma(ln_B_V_nu).sum(axis = 1)

        ln_B_V_nu -= self.D*(self.D - 1)/4*np.log(np.pi)

        ln_B_V_nu -= self.nu*self.D/2*np.log(2)

        ln_B_V_nu -= self.nu/2*np.log(np.linalg.det(self.V))

        return ln_B_V_nu

    def H_Lambda(self) -> np.ndarray:

        H_Lambda = -self.ln_B_V_nu()

        H_Lambda -= (self.nu - self.D - 1)/2*self.E_ln_det_Lambda()

        H_Lambda += self.nu*self.D/2

        return H_Lambda
    
    def atualiza_E_ln_p_pi(self) -> float:

        E_ln_p_pi = self.ln_C_alpha_0()

        E_ln_p_pi += (self.alpha_0 - 1)*self.E_ln_pi().sum()

        return E_ln_p_pi
    
    def atualiza_E_ln_p_Z(self) -> float:

        E_ln_p_Z = self.r @ self.E_ln_pi()

        E_ln_p_Z = E_ln_p_Z.sum()

        return E_ln_p_Z
    
    def atualiza_E_ln_p_mu_Lambda(self) -> float:

        E_ln_p_mu_Lambda = np.einsum('kd, kdd, kd -> k', self.m - self.mu_0, self.V, self.m - self.mu_0)

        E_ln_p_mu_Lambda *= -self.kappa_0*self.nu

        E_ln_p_mu_Lambda = self.D*np.log(self.kappa_0/(2*np.pi))

        E_ln_p_mu_Lambda += (self.nu_0 - self.D)*self.E_ln_det_Lambda()

        E_ln_p_mu_Lambda -= self.D*self.kappa_0/self.kappa

        E_ln_p_mu_Lambda += self.ln_B_V_0_nu_0()

        E_ln_p_mu_Lambda -= self.nu*np.trace(self.W_0 @ self.V, axis1 = 1, axis2 = 2)

        E_ln_p_mu_Lambda = E_ln_p_mu_Lambda.sum()/2

        return E_ln_p_mu_Lambda

    def atualiza_E_ln_p_X(self) -> float:

        E_ln_p_X = self.E_ln_det_Lambda()

        E_ln_p_X -= self.D/self.kappa

        E_ln_p_X -= self.nu*np.trace(self.S_barra @ self.V, axis1 = 1, axis2 = 2)

        E_ln_p_X -= self.nu*np.einsum('kd, kdd, kd -> k', self.X_barra - self.m, self.V, self.X_barra - self.m)

        E_ln_p_X -= self.D*np.log(2*np.pi)

        E_ln_p_X *= self.N_barra

        E_ln_p_X = E_ln_p_X.sum()/2

        return E_ln_p_X
    
    def atualiza_E_ln_q_pi(self) -> float:

        E_ln_q_pi = (self.alpha - 1)*self.E_ln_pi()

        E_ln_q_pi += self.ln_C_alpha()

        E_ln_q_pi = E_ln_q_pi.sum()

        return E_ln_q_pi
    
    def atualiza_E_ln_q_Z(self) -> float:

        E_ln_q_Z = self.r*np.log(self.r)

        E_ln_q_Z = E_ln_q_Z.sum()

        return E_ln_q_Z
    
    def atualiza_E_ln_q_mu_Lambda(self) -> float:

        E_ln_q_mu_Lambda = self.E_ln_det_Lambda()/2

        E_ln_q_mu_Lambda += self.D/2*np.log(self.kappa/(2*np.pi))

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
    
    def estima_pi(self) -> None:

        self.pi = self.alpha/self.alpha.sum()

    def estima_Z(self) -> None:

        self.Z = np.argmax(self.r, axis = 1)

    def estima_mu(self) -> None:

        self.mu = self.m

    def estima_Sigma(self) -> None:

        self.Sigma = self.W
        
        self.Sigma /= (self.nu - self.D - 1).reshape((self.K, 1, 1))

    def estima_Lambda(self) -> None:

        self.Lambda = self.V

        self.Lambda *= self.nu.reshape((self.K, 1, 1))

    def estima_modelo(self) -> None:

        self.estima_pi()
        
        self.estima_Z()

        self.estima_mu()

        self.estima_Sigma()

        self.estima_Lambda()

    def ajusta_modelo(self, max: int = 1000, tol: float = 1e-6) -> None:

        self.inicializa_modelo()

        self.atualiza_modelo()

        self.ELBO.append(self.atualiza_ELBO())

        for i in range(max):

            self.atualiza_modelo()

            self.ELBO.append(self.atualiza_ELBO())

            if np.abs(self.ELBO[-1] - self.ELBO[-2]) < tol:

                break

        self.estima_modelo()
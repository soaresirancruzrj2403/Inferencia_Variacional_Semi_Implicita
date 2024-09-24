
import numpy as np
import scipy as spy
import pandas as pd

Parametros = np.load('Modelo\\Parametros.npy', allow_pickle = True).item()

class CAVI_GMM:

    def __init__(self, Dados: np.ndarray, K: int) -> None:

        # Dados observados
        
        self.X = Dados

        # Número de observações

        self.N = self.X.shape[0]

        # Dimensão das observações

        self.D = self.X.shape[1]

        # Quantidade de classes latentes

        self.K = K

        # Responsabilidade da classe k na observação n

        self.r_nk = np.array([])

        # Estatísticas suficientes da classe k

        self.N_k = np.array([])

        self.x_barra_k = np.array([])

        self.S_k = np.array([])

        # Parâmetros da priori

        self.alpha_0 = 1e-3

        self.mu_0 = Parametros['mu']

        self.kappa_0 = 1e-3

        self.nu_0 = self.D

        self.V_0 = np.diag([1e-3]*self.D)

        self.V_inv_0 = np.diag([1e3]*self.D)

        # Parâmetros da posteriori da classe k

        self.alpha_k = np.array([])

        self.E_pi_k = np.array([])

        self.nu_k = np.array([])

        self.kappa_k = np.array([])

        self.V_inv_k = np.array([])

        self.V_k = np.array([])

        self.mu_k = np.array([])

        # Valor do ELBO em cada iteração

        self.historico_ELBO = []

        # Cota inferior do ln da evidência

        self.ELBO = 0

        # Quantidades importantes para o cálculo de outros valores

        self.E_ln_det_Lambda_k = np.array([])

        self.E_ln_pi_k = np.array([])

    def inicializa_params_posteriori(self) -> None:

        self.alpha_k = np.random.uniform(low = 0, high = self.N, size = self.K)

        self.nu_k = np.random.randint(low = self.D, high = self.N, size = self.K)

        self.kappa_k = np.random.uniform(low = 0, high = self.N, size = self.K)

        self.V_k = Parametros['Lambda']

        self.V_inv_k = np.linalg.inv(self.V_k)

        self.mu_k = Parametros['mu']

    def atualiz_E_ln_det_Lambda_k(self) -> None:

        self.E_ln_det_Lambda_k = np.add.outer(self.nu_k, -np.arange(self.D))/2

        self.E_ln_det_Lambda_k = spy.special.psi(self.E_ln_det_Lambda_k).sum(1)

        self.E_ln_det_Lambda_k += np.log(np.linalg.det(self.V_k))

        self.E_ln_det_Lambda_k += self.D*np.log(2)

    def atualiza_E_ln_pi_k(self) -> None:

        self.E_ln_pi_k = spy.special.psi(self.alpha_k)
        
        self.E_ln_pi_k -= spy.special.psi(self.alpha_k.sum())

    def atualiza_r_nk(self) -> None:

        self.r_nk = np.zeros((self.N, self.K))

        for n in range(self.N):

            for k in range(self.K):
                
                self.r_nk[n][k] = (self.X[n] - self.mu_k[k]).T @ self.V_k[k] @ (self.X[n] - self.mu_k[k])

                self.r_nk[n][k] *= -self.nu_k[k]/2

                self.r_nk[n][k] -= self.D/(2*self.kappa_k[k])

        self.r_nk = self.r_nk/np.abs(self.r_nk).max()

        self.r_nk = self.E_ln_pi_k*np.sqrt(self.E_ln_det_Lambda_k)*np.exp(self.r_nk)

        self.r_nk = self.r_nk/self.r_nk.sum(axis = 1)[:, np.newaxis]

    def atualiza_N_k(self) -> None:

        self.N_k = self.r_nk.sum(0)

    def atualiza_x_barra_k(self) -> None:

        self.x_barra_k = self.r_nk.T @ self.X/self.N_k[:, np.newaxis]
    
    def atualiza_S_k(self) -> None:

        self.S_k = np.zeros((self.K, self.D, self.D))

        for k in range(self.K):
                
            for n in range(self.N):

                self.S_k[k] += self.r_nk[n][k]*(self.X[n] - self.x_barra_k[k])[:, np.newaxis] @ (self.X[n] - self.x_barra_k[k])[:, np.newaxis].T
                    
                self.S_k[k] /= self.N_k[k]

    def atualiza_alpha_k(self) -> None:

        self.alpha_k = self.alpha_0 + self.N_k

    def atualiza_nu_k(self) -> None:

        self.nu_k = self.nu_0 + self.N_k

    def atualiza_kappa_k(self) -> None:

        self.kappa_k = self.kappa_0 + self.N_k

    def atualiza_mu_k(self) -> None:

        self.mu_k = self.kappa_0*self.mu_0

        self.mu_k += self.N_k[:, np.newaxis]*self.x_barra_k

        self.mu_k /= self.kappa_k[:, np.newaxis]

    def atualiza_V_k(self) -> None:

        for k in range(self.K):

            self.V_inv_k[k] = np.outer(self.x_barra_k[k] - self.mu_0[k], self.x_barra_k[k] - self.mu_0[k])

            self.V_inv_k[k] *= self.kappa_0*self.N_k[k]/(self.N_k[k] + self.kappa_0)

            self.V_inv_k[k] += self.V_inv_0 + self.N_k[k]*self.S_k[k]

            self.V_k[k] = np.linalg.inv(self.V_inv_k[k])

    def atualiza_params(self) -> None:

        # Atualiza quantidades importantes para o cálculo de outros valores

        self.atualiz_E_ln_det_Lambda_k()

        self.atualiza_E_ln_pi_k()

        # Atualiza responsabilidade da classe k na observação n

        self.atualiza_r_nk()

        # Atualiza estatísticas suficientes da classe k

        self.atualiza_N_k()

        self.atualiza_x_barra_k()

        self.atualiza_S_k()

        # Atualiza parâmetros da posteriori da classe k

        self.atualiza_alpha_k()

        self.atualiza_nu_k()

        self.atualiza_kappa_k()

        self.atualiza_mu_k()

        self.atualiza_V_k()

    def ajusta_modelo(self):

        for i in range(5):

            self.atualiza_params()

Dados = pd.read_csv('Modelo\\Amostra.csv', usecols = ['x_1', 'x_2'])

Objetos_Teste = CAVI_GMM(Dados = np.array(Dados), K = 5)

Objetos_Teste.inicializa_params_posteriori()

Objetos_Teste.ajusta_modelo()

print(Objetos_Teste.mu_k)
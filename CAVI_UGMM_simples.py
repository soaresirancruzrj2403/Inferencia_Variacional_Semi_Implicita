import numpy as np

class CAVI_UGMM_simples:

    def __init__(self, dados:np.ndarray, num_categorias:int, var_priori: float):

        self.dados = dados

        self.num_categorias = num_categorias

        self.prob = []

        self.var_priori = var_priori

        self.media = []

        self.var = []

        self.historico_ELBO = []

        self.historico_media = []

        self.historico_var = []

    def inicializa_params(self):

        self.prob = np.array([[1/self.num_categorias]*self.num_categorias]*self.dados.shape[0])

        self.media = np.random.uniform(low = self.dados.min(), high = self.dados.max(), size = self.num_categorias)

        self.var = np.random.uniform(low = 0, high = self.dados.max(), size = self.num_categorias)

    def calcula_ELBO(self):

        ELBO = -np.add.outer(self.dados**2, self.var + self.media**2)/2 + np.outer(self.dados, self.media) - np.log(self.prob)

        ELBO = (self.prob*ELBO).sum()

        ELBO += (np.log(self.var) - (self.media**2 + self.var)/self.var_priori).sum()/2

        return ELBO
    
    def atualiza_prob(self):

        prob_kernel = np.exp(np.outer(self.dados, self.media) - (self.media**2 + self.var)[np.newaxis, :]/2)

        self.prob = prob_kernel/prob_kernel.sum(1)[:, np.newaxis]

    def atualiza_media(self):

        self.media = (self.prob*self.dados[:, np.newaxis]).sum(0)/(1/self.var_priori + self.prob.sum(0))

    def atualiza_var(self):

        self.var = (1/self.var_priori + self.prob.sum(0))**(-1)

    def ajusta_modelo(self, max_iteracoes: int = int(10e3), tol:float = 10e-6):

        self.inicializa_params()

        self.historico_ELBO = [self.calcula_ELBO()]

        self.historico_media = [self.media]

        self.historico_var = [self.var]

        for iteracao in range(max_iteracoes):

            self.atualiza_prob()

            self.atualiza_media()

            self.atualiza_var()

            self.historico_ELBO.append(self.calcula_ELBO())

            self.historico_media.append(self.media)

            self.historico_var.append(self.var)

            if np.abs(self.historico_ELBO[-2] - self.historico_ELBO[-1]) <= tol:

                break
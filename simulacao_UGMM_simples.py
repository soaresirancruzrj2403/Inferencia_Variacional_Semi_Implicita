import numpy as np
import pandas as pd

num_categorias = 5

num_dados = int(10e5)

prob_categorias = [1/num_categorias]*num_categorias

media_categorias = np.linspace(start = -num_categorias, stop = num_categorias, num = num_categorias)

var_latentes = np.random.choice(a = media_categorias, p = prob_categorias, size = num_dados)

var_observadas = np.array([np.random.normal(loc = var_latentes[i]) for i in range(num_dados)])

dados_simulados = pd.DataFrame({'var_observada': var_observadas, 'var_latente': var_latentes})

dados_simulados.to_csv('UGMM_simples_simulado.csv', index =  False)
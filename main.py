import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.treinamento import activation_function, treinar, predicao
from utils.pre_processamento import divisao_teste_treino, min_max_scaler
from utils.metrics import classification_report, erros_epocas, mean_squared_error

df = pd.read_csv(r"data\data_banknote_authentication.txt",
                 sep= ',', header= None)

dic = { 
       0 : 'variance',
       1 : 'skewness',
       2 : 'curtosis',
       3 : 'entropy',
       4 : 'class',
       }

df.rename( columns= dic, inplace= True )

num_neuronios_saida = len(df['class'].unique())
divisao_treino = 0.7
X_treino, y_treino, X_teste, y_teste = divisao_teste_treino( df, divisao_treino, num_neuronios_saida, seed = True)
X_treino_normalizado = min_max_scaler( X_treino )

learning_rate = 0.05
epocas = 100
pesos_t, erros_historico = treinar(X_treino, y_treino, learning_rate, epocas, num_neuronios_saida )

# Erros por época
erros_epocas(erros_historico, show=False, location='images')

# Rede Neural finalizada
y_predito = predicao( X_teste, pesos_t )

y_teste = np.argmax( y_teste, axis= 1)

# Relatório de Classificação
relatorio = classification_report(y_teste, y_predito, show=False, location='images')
print("Relatório de Classificação:")
for metric, valor in relatorio.items():
    print(f"  {metric}: {valor:.2%}")

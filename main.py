import pandas as pd
import numpy as np
from treinamento import activation_function, treinar, predicao
from pre_processamento import divisao_teste_treino, min_max_scaler

df = pd.read_csv(r"C:\Users\user\Desktop\Kayky\Projeto Rede Neural\Neural_Network_NP\data_banknote_authentication.txt",
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

learning_rate = 0.1
epocas = 1000
pesos_t = treinar(X_treino, y_treino, learning_rate, epocas, num_neuronios_saida )
y_predito = predicao( X_teste, pesos_t )

y_teste = np.argmax( y_teste, axis= 1)

# Avaliação da Acurácia
acertos = np.sum(y_predito == y_teste)
print(f"Acurácia no Teste: {acertos / len(y_teste):.2%}\n")
        
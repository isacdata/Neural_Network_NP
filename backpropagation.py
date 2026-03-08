import numpy as np

def activation_function(s):
    # Para tarefas de classificação, será utilizado a função sigmoide
    return 1 / (1 + np.exp(-s))

def backpropagation( X_treino: np.array , y_treino: np.array , learning_rate: float, epocas: int, num_neuronios: int ) -> np.array :
    
    # Incializar a matriz de pesos aleatórios e de y_predição
    num_entradas = X_treino.shape[1]
    rng = np.random.default_rng()
    pesos = rng.random( ( num_neuronios, num_entradas ) )
    pesos_t = np.transpose( pesos )
    y_predicao = np.zeros( (X_treino.shape[0], num_neuronios, ) )
    
    # Treinar em todas as épocas
    for epoca in range( epocas ):
        for line in range( len(X_treino) ):
            for neuronio in range( num_neuronios ):
                # Feedforward
                s = np.matmul( X_treino[ line ] , pesos_t[: , neuronio]  )
                y_predicao[line, neuronio] = activation_function(s)
                
                # Erro de predição
                erro = y_treino[line, neuronio] - y_predicao[line, neuronio]
                
                # Backpropagation
                derivada_s = activation_function(s) * (1- activation_function(s))
                delta_w =- (-2) * (learning_rate) * erro * derivada_s * X_treino[ line ]
                
                # Gradient Descent
                pesos_t[:, neuronio] = pesos_t[:, neuronio] + delta_w
                
    return pesos_t 
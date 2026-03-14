import numpy as np
import pandas as pd

def divisao_teste_treino(df: pd.DataFrame, divisao_treino: float, num_neuronios_saida: int, seed: bool = False) -> tuple:
    semente = None
    if seed:
        semente = 42
    
    df_copia = df.copy()
    df_copia = df_copia.sample(frac=1, random_state=semente).reset_index(drop=True)
    
    num_linhas = df_copia.shape[0]
    
    # Separa os labels e aplica One-Hot Encoding
    y = pd.get_dummies(df_copia['class']).values
    df_copia = df_copia.drop(columns=['class'])
    
    # Adiciona a coluna de bias (1s) ao final das features
    X = np.append(df_copia, np.ones((num_linhas, 1)), axis=1)
    
    # Calcula o ponto de corte para o treino
    qtd_treino = int(num_linhas * divisao_treino)
    
    # Divide os dados
    X_treino = X[:qtd_treino, :]
    y_treino = y[:qtd_treino, :]
    X_teste = X[qtd_treino:, :]
    y_teste = y[qtd_treino:, :]
    
    return X_treino, y_treino, X_teste, y_teste


def min_max_scaler(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.min()) / (df.max() - df.min())
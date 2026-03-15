import itertools
import numpy as np
from utils.treinamento import treinar, predicao
from utils.metrics import precision, accuracy, recall, f1_score

import itertools
import numpy as np
from utils.treinamento import treinar, predicao
from utils.metrics import precision, accuracy, recall, f1_score

# Adicionado X_val e y_val para avaliar os parâmetros corretamente sem enviesar o modelo
def grid_search(taxas_aprendizado, lista_epocas, X_treino, y_treino, X_val, y_val, metric='accuracy'):

    combinacoes = list(itertools.product(taxas_aprendizado, lista_epocas))

    # Variáveis para guardar os melhores resultados
    melhor_valor_metrica = float('inf') if metric == 'mse' else -float('inf')
    melhores_parametros = {}
    melhores_pesos = None
    melhor_historico = []

    print("="*60)
    print(f"Iniciando Grid Search com {len(combinacoes)} combinações...")

    for lr, epocas in combinacoes:
        print(f"  Treinando -> Learning Rate: {lr} | Épocas: {epocas}")
        
        # 3. Treina a rede com a combinação atual (usando dados de TREINO)
        pesos, historico_mse = treinar(X_treino, y_treino, learning_rate=lr, epocas=epocas, num_neuronios_saida=2)
        
        # 4. Avalia o resultado (usando dados de VALIDAÇÃO)
        y_predito_val = predicao(X_val, pesos)

        if y_val.ndim == 2:
            y_val_labels = np.argmax(y_val, axis=1)
        else:
            y_val_labels = y_val

        # Determina o valor da métrica atual
        if metric == 'accuracy':
            valor_atual = accuracy(y_val_labels, y_predito_val)
            is_best = valor_atual > melhor_valor_metrica
            
        elif metric == 'precision':
            valor_atual = precision(y_val_labels, y_predito_val)
            is_best = valor_atual > melhor_valor_metrica
            
        elif metric == 'recall':
            valor_atual = recall(y_val_labels, y_predito_val)
            is_best = valor_atual > melhor_valor_metrica
            
        elif metric == 'f1_score':
            valor_atual = f1_score(y_val_labels, y_predito_val)
            is_best = valor_atual > melhor_valor_metrica
            
        else: # Default para MSE
            valor_atual = historico_mse[-1] 
            print(f"Erro (MSE) de Treino Final: {valor_atual:.6f}")
            is_best = valor_atual < melhor_valor_metrica # Para erro, menor é melhor

        # 5. Se for o melhor resultado encontrado até agora, salva TUDO!
        if is_best:
            melhor_valor_metrica = valor_atual
            melhores_parametros = {'learning_rate': lr, 'epocas': epocas}
            melhores_pesos = pesos
            melhor_historico = historico_mse

    # 6. Exibe o grande vencedor
    print("="*60)
    print(f"🎯 MELHOR COMBINAÇÃO ENCONTRADA:")
    print(f"Parâmetros: {melhores_parametros}")
    print("="*60)

    # Retornar os pesos também costuma ser muito útil!
    return melhores_parametros
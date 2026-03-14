import itertools
import numpy as np
from utils.treinamento import treinar

def grid_search(taxas_aprendizado, lista_epocas, X_treino, y_treino):

    # 2. Cria todas as combinações possíveis
    # Exemplo: (0.01, 100), (0.01, 500), (0.05, 100)...
    combinacoes = list(itertools.product(taxas_aprendizado, lista_epocas))

    # Variáveis para guardar os melhores resultados
    melhor_mse = float('inf') # Começa com infinito para que qualquer erro seja menor
    melhores_parametros = {}
    melhores_pesos = None
    melhor_historico = []

    print("="*60)

    print(f"Iniciando Grid Search com {len(combinacoes)} combinações...\n")

    for lr, epocas in combinacoes:
        print(f"Treinando -> Learning Rate: {lr} | Épocas: {epocas}")
        
        # 3. Treina a rede com a combinação atual
        # (Assumindo que X_treino e y_treino já estão definidos e num_neuronios_saida=2)
        pesos, historico_mse = treinar(X_treino, y_treino, learning_rate=lr, epocas=epocas, num_neuronios_saida=2)
        
        # 4. Avalia o resultado (usamos o MSE da última época)
        erro_final = historico_mse[-1]
        
        # 5. Se for o menor erro encontrado até agora, salva tudo!
        if erro_final < melhor_mse:
            melhor_mse = erro_final
            melhores_parametros = {'learning_rate': lr, 'epocas': epocas}
            melhores_pesos = pesos
            melhor_historico = historico_mse

    # 6. Exibe o grande vencedor
    print("\n" + "="*60)
    print(f"🎯 MELHOR COMBINAÇÃO ENCONTRADA:")
    print(f"Parâmetros: {melhores_parametros}")
    print(f"Erro (MSE) Final: {melhor_mse:.6f}")
    print("="*60)

    return melhores_parametros
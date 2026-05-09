import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE

SEED       = 7
K_MIN      = 1    # k_neighbors mínimo para classes com poucos exemplos
RATIO_ALVO = 0.5  # proporção desejada: positivos / negativos (ex.: 1:2)

# Carregar dados binários
X = np.load('X_bin.npy')
Y = np.load('Y_bin.npy')

contagem_antes = Counter(Y)
print(f'Dados carregados        : {X.shape}')
print(f'Distribuição original   : {dict(contagem_antes)}')
print(f'  Classe 0 (negativo)   : {contagem_antes[0]:,}')
print(f'  Classe 1 (positivo)   : {contagem_antes[1]:,}')

# k_neighbors adaptativo — evita erro quando a classe positiva é muito pequena
n_pos = contagem_antes[1]
k_neighbors = min(5, n_pos - 1) if n_pos > 1 else K_MIN
print(f'\nk_neighbors SMOTE       : {k_neighbors}  (adaptativo ao tamanho da classe)')

# SMOTE
smote = SMOTE(
    sampling_strategy=RATIO_ALVO,
    k_neighbors=k_neighbors,
    random_state=SEED
)
X_smote, Y_smote = smote.fit_resample(X, Y)

contagem_depois = Counter(Y_smote)
print(f'\nDistribuição após SMOTE : {dict(contagem_depois)}')
print(f'  Classe 0 (negativo)   : {contagem_depois[0]:,}')
print(f'  Classe 1 (positivo)   : {contagem_depois[1]:,}')
print(f'  Samples sintéticos    : {contagem_depois[1] - contagem_antes[1]:,}')
print(f'  Nova razão pos/neg    : {contagem_depois[1]/contagem_depois[0]:.2f}')

# Salvar
np.save('X_smote.npy', X_smote)
np.save('Y_smote.npy', Y_smote)
print('\nArquivos salvos: X_smote.npy, Y_smote.npy')
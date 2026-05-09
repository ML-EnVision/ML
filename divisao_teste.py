import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

SEED       = 7
TEST_SIZE  = 0.2   # 20% para teste, 80% para treino

# Carregar dados balanceados
X = np.load('X_smote.npy')
Y = np.load('Y_smote.npy')

print(f'Dados carregados      : {X.shape}')
print(f'Distribuição total    : {dict(Counter(Y))}')

# Divisão estratificada — preserva proporção de classes em treino e teste
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=TEST_SIZE,
    stratify=Y,
    random_state=SEED
)

print(f'\nConjunto de treino')
print(f'  Shape X_train       : {X_train.shape}')
print(f'  Distribuição        : {dict(Counter(Y_train))}')

print(f'\nConjunto de teste')
print(f'  Shape X_test        : {X_test.shape}')
print(f'  Distribuição        : {dict(Counter(Y_test))}')

# Verificação de proporcionalidade
for classe in np.unique(Y):
    pct_treino = (Y_train == classe).mean() * 100
    pct_teste  = (Y_test  == classe).mean() * 100
    print(f'  Classe {classe} — treino: {pct_treino:.1f}%  teste: {pct_teste:.1f}%  ✓')

# Salvar
np.save('X_train.npy', X_train)
np.save('X_test.npy',  X_test)
np.save('Y_train.npy', Y_train)
np.save('Y_test.npy',  Y_test)
print('\nArquivos salvos: X_train.npy, X_test.npy, Y_train.npy, Y_test.npy')
print(f'\nPasso 4 concluído — dados prontos para modelagem (Passo 5).')
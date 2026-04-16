# -*- coding: utf-8 -*-
# PASSO 2 — PCA

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

SEED = 7

# Carregar dados normalizados
X_norm = np.load('X_norm.npy')
Y      = np.load('Y.npy')

print(f'Dados carregados: {X_norm.shape}')

# PCA (99% variância)
pca   = PCA(n_components=0.99, random_state=SEED)
X_pca = pca.fit_transform(X_norm)

print(f'Bandas originais  : {X_norm.shape[1]}')
print(f'Componentes PCA   : {X_pca.shape[1]}')
print(f'Variância mantida : {pca.explained_variance_ratio_.cumsum()[-1]*100:.2f}%')

# Salvar resultado
np.save('X_pca.npy', X_pca)

# Gráfico
var_acum = pca.explained_variance_ratio_.cumsum() * 100

plt.figure(figsize=(10, 4))
plt.plot(range(1, len(var_acum)+1), var_acum, 'o-', ms=4)
plt.axhline(99, linestyle='--', label='99% variância')
plt.axhline(95, linestyle='--', label='95% variância')

plt.xlabel('Número de Componentes Principais')
plt.ylabel('Variância Explicada Acumulada (%)')
plt.title(f'{X_pca.shape[1]} componentes para 99% da variância')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('pca_variancia.png', dpi=150)
plt.show()

# Variância por componente
print('\nVariância por componente:')
acum = 0
for i, v in enumerate(pca.explained_variance_ratio_, 1):
    acum += v * 100
    print(f'  PC{i:>2}: {v*100:>5.2f}%  acumulada: {acum:.1f}%')
    if acum >= 99:
        break
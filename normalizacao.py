# -*- coding: utf-8 -*-
# PASSO 1 — Normalização

import os, numpy as np, scipy.io
from sklearn.preprocessing import StandardScaler

ARQUIVO  = os.path.join(os.path.dirname(__file__), 'CRISM_labeled_pixels_ratioed.mat')
IMAGEM_ID = 10

# Carregar dados
data    = scipy.io.loadmat(ARQUIVO)
pixims  = data['pixims'].flatten()
mask    = pixims == IMAGEM_ID

X = data['pixspec'][mask]
Y = data['pixlabs'].flatten()[mask]

print(f'Pixels carregados: {X.shape[0]:,}  |  Bandas: {X.shape[1]}')

# Normalização
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

print(f'Após normalização : média: {X_norm.mean():.4f}  desvio: {X_norm.std():.4f}')

# Salvar dados normalizados
np.save('X_norm.npy', X_norm)
np.save('Y.npy', Y)
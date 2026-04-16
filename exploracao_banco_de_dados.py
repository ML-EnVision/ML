# -*- coding: utf-8 -*-
# PASSO 0 — Exploração do Banco de Dados CRISM

import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from collections import Counter

ARQUIVO = os.path.join(os.path.dirname(__file__), 'CRISM_labeled_pixels_ratioed.mat')

# Carregar arquivo completo
data     = scipy.io.loadmat(ARQUIVO)
spectra  = data['pixspec']
labels   = data['pixlabs'].flatten()
pixims   = data['pixims'].flatten()
im_names = data['im_names'].flatten()

# 1. Estrutura geral
print('variaveis do arquivo: ')
for k, v in data.items():
    if not k.startswith('__'):
        print(f'  {k:<12} shape={getattr(v,"shape","N/A")}  dtype={getattr(v,"dtype","N/A")}')

# 2. Distribuição de classes
contagem = Counter(labels.astype(int))
total    = len(labels)
print(f'\nTotal de pixels : {total:,}')
print(f'Classes únicas  : {len(contagem)}')
print(f'\n{"Classe":<10} {"Pixels":>10} {"Porcentagem":>12}')
print('-' * 35)
for cl, qt in sorted(contagem.items()):
    print(f'  {cl:<10} {qt:>10,} {qt/total*100:>11.2f}%')

# 3. Minerais por imagem
print('\n minerais por imagem:')
for img_id in sorted(set(pixims.astype(int))):
    mask = pixims == img_id
    classes = sorted(set(labels[mask].astype(int)))
    nome    = str(im_names[img_id - 1])
    npix    = mask.sum()
    print(f'  Img {img_id:>2} | {npix:>6,} pixels | classes: {classes} | {nome}')

# 4. Gráfico de distribuição
classes_sorted = sorted(contagem.keys())
qtds = [contagem[c] for c in classes_sorted]
plt.figure(figsize=(14, 5))
plt.bar([str(c) for c in classes_sorted], qtds, color='steelblue', edgecolor='white')
plt.yscale('log')
plt.title('Distribuição de pixels por classe mineral (escala log)')
plt.xlabel('Classe mineral')
plt.ylabel('Número de pixels')
plt.tight_layout()
plt.savefig('distribuicao_classes.png', dpi=150)
plt.show()
print('\nGráfico salvo: distribuicao_classes.png')

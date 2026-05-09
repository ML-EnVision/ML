import os, numpy as np, scipy.io
import matplotlib.pyplot as plt

ARQUIVO  = os.path.join(os.path.dirname(__file__), 'CRISM_labeled_pixels_ratioed.mat')
IMAGEM_ID = 10   # mesma imagem usada na normalização

# Carregar arquivo .mat
print('Carregando arquivo .mat...')
data     = scipy.io.loadmat(ARQUIVO)
pixspec  = data['pixspec']
pixlabs  = data['pixlabs'].flatten()
pixims   = data['pixims'].flatten()
pixcrds  = data['pixcrds']
im_names = [str(n).strip() for n in data['im_names'].flatten()]

print(f'Total de pixels no banco : {pixspec.shape[0]:,}')
print(f'Total de imagens          : {len(im_names)}')
print(f'Imagem selecionada        : ID={IMAGEM_ID} — {im_names[IMAGEM_ID - 1]}')

# Filtrar pela imagem escolhida
mask   = pixims == IMAGEM_ID
X_filt = pixspec[mask]
Y_filt = pixlabs[mask]
coords = pixcrds[mask]   

print(f'\nPixels da imagem {IMAGEM_ID}  : {X_filt.shape[0]:,}')
print(f'Bandas espectrais          : {X_filt.shape[1]}')
print(f'Dimensão das coordenadas   : {coords.shape}')

# Distribuição de classes na imagem
classes, contagens = np.unique(Y_filt, return_counts=True)
print(f'\nClasses presentes na imagem {IMAGEM_ID}: {len(classes)}')
print(f"{'Classe':>8}  {'Pixels':>8}  {'%':>6}")
print('-' * 28)
for c, n in zip(classes, contagens):
    print(f'{c:>8}  {n:>8,}  {n/len(Y_filt)*100:>5.1f}%')

# Salvar coordenadas (X e Y já salvos pela normalizacao.py + pca.py)
np.save('pixcrds_filtrado.npy', coords)
np.save('Y_filtrado.npy', Y_filt)
print('\nArquivos salvos: pixcrds_filtrado.npy, Y_filtrado.npy')

# Gráfico de distribuição de classes
plt.figure(figsize=(10, 4))
plt.bar(classes, contagens, color='steelblue', edgecolor='white', linewidth=0.5)
plt.xlabel('Classe Mineral')
plt.ylabel('Número de Pixels')
plt.title(f'Distribuição de Classes — Imagem {IMAGEM_ID}')
plt.yscale('log')
plt.xticks(classes)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('distribuicao_imagem_filtrada.png', dpi=150)
plt.show()
print('Gráfico salvo: distribuicao_imagem_filtrada.png')
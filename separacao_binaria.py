import numpy as np
import matplotlib.pyplot as plt

MINERAL_ALVO = 7   # classe a ser detectada (1 = presente, 0 = ausente)

# Carregar dados reduzidos pelo PCA
X_pca = np.load('X_pca.npy')
Y     = np.load('Y_filtrado.npy')

print(f'Dados carregados   : {X_pca.shape}')
print(f'Total de pixels    : {len(Y):,}')
print(f'Classes únicas     : {np.unique(Y)}')

# Conversão para rótulo binário
Y_bin = (Y == MINERAL_ALVO).astype(int)

n_pos = Y_bin.sum()
n_neg = len(Y_bin) - n_pos
ratio = n_pos / len(Y_bin) * 100

print(f'\nMineral alvo       : Classe {MINERAL_ALVO}')
print(f'Pixels positivos   : {n_pos:,}  ({ratio:.1f}%)')
print(f'Pixels negativos   : {n_neg:,}  ({100 - ratio:.1f}%)')
print(f'Razão desbalamento : 1 : {n_neg // max(n_pos, 1)}')

if n_pos == 0:
    raise ValueError(
        f'Classe {MINERAL_ALVO} não encontrada na imagem filtrada. '
        f'Escolha outra classe em MINERAL_ALVO.'
    )

# Salvar arrays binários
np.save('X_bin.npy', X_pca)
np.save('Y_bin.npy', Y_bin)
print('\nArquivos salvos: X_bin.npy, Y_bin.npy')

# Gráfico de proporção
plt.figure(figsize=(5, 4))
labels  = [f'Mineral {MINERAL_ALVO}\n(positivo)', 'Outros\n(negativo)']
valores = [n_pos, n_neg]
cores   = ['#2196F3', '#E0E0E0']
bars = plt.bar(labels, valores, color=cores, edgecolor='white')
for bar, v in zip(bars, valores):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + max(valores) * 0.01,
            f'{v:,}', ha='center', va='bottom', fontsize=10)
plt.ylabel('Número de Pixels')
plt.title(f'Separação Binária — Mineral {MINERAL_ALVO}')
plt.tight_layout()
plt.savefig('separacao_binaria.png', dpi=150)
plt.show()
print('Gráfico salvo: separacao_binaria.png')
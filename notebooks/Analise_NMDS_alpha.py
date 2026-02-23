# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] id="o8v8w4nGdP3n" jp-MarkdownHeadingCollapsed=true
# # Imports
#

# %% colab={"base_uri": "https://localhost:8080/"} id="nh5Pzk9yhNS7" outputId="e61902ef-d7d1-40c6-b965-a6a9c94b892e"
pip install scikit-bio


# %% colab={"base_uri": "https://localhost:8080/", "height": 944} id="VsznwQJOwC3c" outputId="70786185-ce99-4919-dfc2-a6a25979f9cf"
# Baixar a fonte Times New Roman e instalar no diretório de fontes do sistema
# !wget -q -O /usr/share/fonts/truetype/times.ttf https://github.com/google/fonts/raw/main/apache/roboto/Roboto-Regular.ttf

# Atualizar o cache de fontes
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Verificar se a fonte foi instalada
fm.findSystemFonts(fontpaths=["/usr/share/fonts/truetype/"])

# Definir a fonte "Times New Roman" como padrão
plt.rcParams["font.family"] = "Times New Roman"

# Testar um gráfico com a nova fonte
plt.text(0.5, 0.5, 'Teste com Times New Roman', fontsize=12, ha='center')
plt.show()


# %% id="ScTCyQIudSa6"
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from skbio.diversity import alpha_diversity
from scipy.spatial.distance import pdist, squareform
from skbio.diversity import beta_diversity


# %% [markdown] id="OQBG01N0dL_I" jp-MarkdownHeadingCollapsed=true
# # Loading
#

# %% id="RENy0EjPdLMj"
def load_asv_file(file_path):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep='\t', index_col=0)
        if df.empty:
            raise ValueError(f"Erro: O arquivo ASV foi carregado, mas está vazio: {file_path}")

        # Transpor a tabela (inversão de linhas e colunas)
        df_transposed = df.T

        return df_transposed
    else:
        raise FileNotFoundError(f"Erro: Arquivo ASV não encontrado: {file_path}")



# %% id="xFM_FeKdf8Mn"
meta_all_file = "sample_data/01_metadata_all2.txt"
meta_all_csv = pd.read_csv(meta_all_file, sep="\t", index_col=0)
HRT_metadata_ML_all  = meta_all_csv.copy()

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="1C_UYBvMFCtJ" outputId="c9aec8b2-136c-43d6-af41-635dd336cdaf"
HRT_metadata_ML_all

# %% colab={"base_uri": "https://localhost:8080/"} id="yZg-R7NydueV" outputId="373e528c-26ee-4f89-b315-5bd0fc4389be"
asv_all_file = "sample_data/map_complete_relative_abundance_table.tsv"
asv_all_csv = load_asv_file(asv_all_file)
# Salvar o original
asv_all_csv_original = asv_all_csv.copy()

# Converter todas as colunas para numérico (se apropriado)
asv_all_csv = asv_all_csv.apply(pd.to_numeric, errors='coerce')

# Substituir NA por um pseudo-count pequeno (1e-6)
pseudo_count = 1e-6
asv_all_csv.fillna(pseudo_count, inplace=True)
print(asv_all_csv.isna().sum().sum())

# %% colab={"base_uri": "https://localhost:8080/", "height": 270} id="1oiVFkSTIWIJ" outputId="1d8aa932-7ff2-4c6b-f225-0ccb26ab98fa"
asv_all_csv.head()

# %% id="ebNbxoeVet9B"
# Salvando em arquivos CSV
asv_all_csv.to_csv("sample_data/05_asv_bin_ML_ra.csv")
HRT_asv_ML_all = asv_all_csv.copy()


# %% [markdown] id="z0kZOEC8KyPk" jp-MarkdownHeadingCollapsed=true
# # Adição de colunas ao mapping

# %% id="O4WU3XNdfzu-"
# Adicionar SampleID como coluna temporária
HRT_asv_ML_all['SampleID'] = HRT_asv_ML_all.index
HRT_asv_ML_all['SampleID'] = HRT_asv_ML_all['SampleID'].astype(str).str.strip()

# %% id="YS0da7TWJRmX"
# Adicionar SampleID como coluna temporária
HRT_metadata_ML_all['SampleID'] = HRT_metadata_ML_all.index
HRT_metadata_ML_all['SampleID'] = HRT_metadata_ML_all['SampleID'].astype(str).str.strip()

# %% colab={"base_uri": "https://localhost:8080/", "height": 238} id="GQCpv1tTI6ZJ" outputId="b434fe24-212a-48bc-a988-d848101f870a"
HRT_metadata_ML_all.head()

# %% id="f5yIn2ZLf0Pm"
HRT_ML_reactor = pd.merge(HRT_asv_ML_all, HRT_metadata_ML_all[['SampleID', 'Gut compartiment', 'Experiment', 'Category']], on='SampleID', how='inner')

# %% id="eT_zbfSdgP0u"
# Transformar SampleID de volta em índice e remover a coluna SampleID
HRT_ML_reactor.set_index('SampleID', inplace=True)
HRT_ML_reactor.index.name = None

# %% id="xLqbWEaVJg3Z"
# Transformar SampleID de volta em índice e remover a coluna SampleID
HRT_metadata_ML_all.set_index('SampleID', inplace=True)
HRT_metadata_ML_all.index.name = None

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="sMLRnXjCJnH8" outputId="a04445b6-fc43-4a10-d791-f7cedc6ba789"
HRT_ML_reactor

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} id="Bkrq1SOjJ1r2" outputId="7342df81-7adc-40fd-a5a6-66b387beac32"
HRT_metadata_ML_all.head()

# %% id="YCqxgt2rg73q"
# 1) Filtro de prevalência (remover ASVs que são zero em todas as amostras)
zero_asvs = HRT_ML_reactor.columns[(HRT_ML_reactor == 0).all()]
HRT_ML_reactor.drop(columns=zero_asvs, inplace=True)

# %% colab={"base_uri": "https://localhost:8080/", "height": 458} id="Vte0BzUmhJxu" outputId="0ad96406-3c2e-409a-97b1-123b5fddf283"
HRT_ML_reactor.dtypes


# %% id="eeGmcUoHLVvY"
# Função para alterar os valores da coluna 'categoria'
def update_categoria(row):
    if row['Category'] == '1T':
      return f"1T-{row['Category2']}"
    elif row['Category'] == 'Inoculum':
        return f"Inoculum-{row['Category2']}"
    return row['Category']

# Aplicando a função na coluna 'categoria'
HRT_metadata_ML_all['Category'] = HRT_metadata_ML_all.apply(update_categoria, axis=1)



# %% [markdown] id="lP6TUPHddsGP" jp-MarkdownHeadingCollapsed=true
# # Anlyses
#

# %% colab={"base_uri": "https://localhost:8080/"} id="3FxoYqoFKKV8" outputId="5a504da8-5abc-4334-9889-0f27a328dd15"
# Cada linha representa uma amostra e as colunas são OTUs
data = HRT_asv_ML_all.copy()
data.shape

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="A03DOAyy3mwe" outputId="75cef4fb-5df8-41b6-bae3-c80d135d9984"
metadata = HRT_metadata_ML_all.loc[:, ['Gut compartiment', 'Experiment', 'Category']].copy()
metadata

# %% colab={"base_uri": "https://localhost:8080/"} id="SqItK-VxMpEU" outputId="b03cb85a-10ea-4d41-b82c-ff1dc74cbab0"
metadata.shape

# %% colab={"base_uri": "https://localhost:8080/"} id="HTtwU8RT5xCE" outputId="aceba065-3483-41cf-c485-4879c31b88f0"
print(data.dtypes)


# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="tws7m-SMx2s2" outputId="68eb56b0-72a8-44cf-d665-f71c2ad4896a"
metadata['Category']

# %% colab={"base_uri": "https://localhost:8080/"} id="8HvZnEMkxrhs" outputId="fe9c7ff5-422e-44ad-c344-8c233420d777"
metadata.columns

# %% colab={"base_uri": "https://localhost:8080/"} id="Hs_DH1kQEgHP" outputId="33077d3f-22a1-478a-de82-2c3220de4f3a"
data_numeric.shape

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="fpsSxkjoc5V3" outputId="bdc0d1f3-bad2-461d-b9ae-b04eb0cde5d3"
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from skbio.diversity import alpha_diversity
from skbio.stats.distance import permanova
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_ind
from skbio import DistanceMatrix
from scipy.stats import kruskal



# Remover a coluna 'SampleID' antes de calcular as distâncias
data_numeric = data.drop(columns=['SampleID'])



# Configuração dos gráficos
markers = {
    '1T': 'o',           # círculo
    '2T': 's',           # quadrado
    '3T': 'D',           # losango
    'R1': '^',           # triângulo
    'R2': 'P',           # pentágono
    'R3': 'X',           # cruz
    'Foam': '*',         # estrela
    'Inoculum': 'v',     # triângulo invertido

    # Novos símbolos preenchidos para as categorias específicas
    'Inoculum-1': 'p',   # pentágono
    'Inoculum-2': 'H',   # hexágono
    'Inoculum-R1': 'd',  # pequeno losango
    'Inoculum-R2': 'h',  # hexágono
    'Inoculum-R3': 'X',  # cruz
    '1T-1': 'o',         # círculo
    '1T-2': 's'          # quadrado
}

markers_exp = {
    'Enrichement': 'o',  # círculo
    'Reactor': 's',     # quadrado
}
# Cálculo de distâncias (usando distância Bray-Curtis, comum em microbiomas)
dist_matrix = pdist(data_numeric, metric='braycurtis')
dist_matrix_square = squareform(dist_matrix)

######## ANÁLISE GUT COMPARTMENTS
# Calcular a diversidade alfa usando o índice de Shannon e número de ASVs observados
diversity_shannon = alpha_diversity('shannon', data_numeric.values, ids=metadata.index)
diversity_observed_asv = np.sum(data_numeric.values > 0, axis=1)  # Contagem de ASVs observados
metadata['Shannon'] = diversity_shannon
metadata['Observed_OTU'] = diversity_observed_asv

# Teste t pareado entre HRT de 8 dias e 2 dias (Shannon e Observed ASV)
group_8d = metadata[metadata['Gut compartiment'] == 'Midgut']
group_2d = metadata[metadata['Gut compartiment'] == 'Hindgut']

# Teste t para Shannon
t_shannon, p_shannon = ttest_ind(group_8d['Shannon'], group_2d['Shannon'])

# Teste t para ASV observados
t_asv, p_asv = ttest_ind(group_8d['Observed_OTU'], group_2d['Observed_OTU'])



# Plotando o boxplot da diversidade de Shannon por segmento "Gut"
plt.figure(figsize=(8, 6))
sns.boxplot(data=metadata, x='Gut compartiment', y='Shannon', palette='Set2')
sns.stripplot(data=metadata, x='Gut compartiment', y='Shannon', color='black', jitter=True)
plt.title('Alpha Diversity (Shannon) by Gut Segment')
plt.ylabel('Shannon Diversity Index')
plt.xlabel('Gut Segment')
plt.grid(True)
plt.show()

# Plotando o boxplot da contagem de ASVs observados
plt.figure(figsize=(8, 6))
sns.boxplot(data=metadata, x='Gut compartiment', y='Observed_OTU', palette='Set2')
sns.stripplot(data=metadata, x='Gut compartiment', y='Observed_OTU', color='black', jitter=True)
plt.title('Observed ASVs by Gut Segment')
plt.ylabel('Observed ASVs Count')
plt.xlabel('Gut Segment')
plt.grid(True)
plt.show()

# Criar um objeto DistanceMatrix com os IDs das amostras
dist_matrix_obj = DistanceMatrix(dist_matrix_square, ids=data.index)
print(f"Teste t para Shannon: t={t_shannon:.4f}, p={p_shannon:.4f}")
print(f"Teste t para ASVs observados: t={t_asv:.4f}, p={p_asv:.4f}")
permanova_results = permanova(dist_matrix_obj, metadata['Gut compartiment'], permutations=999)
print(f"PERMANOVA results: pseudo-F={permanova_results['test statistic']:.4f}, p-value={permanova_results['p-value']:.4f}")

#------------------------------------------------------------------------------------------------
# Calcular a diversidade alfa usando o índice de Shannon e número de ASVs observados
diversity_shannon = alpha_diversity('shannon', data_numeric.values, ids=metadata.index)
diversity_observed_asv = np.sum(data_numeric.values > 0, axis=1)  # Contagem de ASVs observados
metadata['Shannon'] = diversity_shannon
metadata['Observed_ASV'] = diversity_observed_asv

# Teste t pareado entre HRT de 8 dias e 2 dias (Shannon e Observed ASV)
group_8d = metadata[metadata['Experiment'] == 'Enrichement']
group_2d = metadata[metadata['Experiment'] == 'Reactor']

# Teste t para Shannon
t_shannon, p_shannon = ttest_ind(group_8d['Shannon'], group_2d['Shannon'])

# Teste t para ASV observados
t_asv, p_asv = ttest_ind(group_8d['Observed_ASV'], group_2d['Observed_ASV'])



# Plotando o boxplot da diversidade de Shannon por segmento "Gut"

fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))
# Plotando o boxplot da diversidade de Shannon por segmento "Gut"
sns.boxplot(data=metadata, x='Experiment', y='Shannon', palette='Set2', ax=ax3)
sns.stripplot(data=metadata, x='Experiment', y='Shannon', color='black', jitter=True, ax=ax3)
ax3.set_title('Diversidade alfa (Shannon) por Experimento')
ax3.set_ylabel('Índice de diversidade Shannon')
ax3.set_xlabel('Experimento')
ax3.grid(True)

# Plotando o boxplot da contagem de ASVs observados
sns.boxplot(data=metadata, x='Experiment', y='Observed_OTU', palette='Set2', ax=ax4)
sns.stripplot(data=metadata, x='Experiment', y='Observed_OTU', color='black', jitter=True, ax=ax4)
ax4.set_title('OTU´s observados por Experimento')
ax4.set_ylabel('Conta de OTU´s observada')
ax4.set_xlabel('Experimento')
ax4.grid(True)

# Ajustar o espaçamento entre os gráficos
plt.tight_layout()
plt.show()

print(f"Teste t para Shannon: t={t_shannon:.4f}, p={p_shannon:.4f}")
print(f"Teste t para ASVs observados: t={t_asv:.4f}, p={p_asv:.4f}")


# Realizando NMDS (utilizando MDS do sklearn como alternativa)
mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
nmds_results = mds.fit_transform(dist_matrix_square)

# Adicionar NMDS1 e NMDS2 no DataFrame de metadata
metadata['NMDS1'] = nmds_results[:, 0]
metadata['NMDS2'] = nmds_results[:, 1]

# Plotando o gráfico de NMDS com separação por cor (Gut) e forma (Experiment)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=metadata, x='NMDS1', y='NMDS2', hue='Gut compartiment', style='Experiment',
                palette='Set1', s=100, markers=markers_exp)
plt.title('NMDS da composição microbiana')
plt.xlabel('NMDS1')
plt.ylabel('NMDS2')
plt.legend(title='Intestino e Experimento')
plt.grid(True)
plt.show()

# Realizando PERMANOVA para diversidade beta

# Criar um objeto DistanceMatrix com os IDs das amostras
dist_matrix_obj = DistanceMatrix(dist_matrix_square, ids=data.index)
permanova_results = permanova(dist_matrix_obj, metadata['Experiment'], permutations=999)
print(f"PERMANOVA results: pseudo-F={permanova_results['test statistic']:.4f}, p-value={permanova_results['p-value']:.4f}")



################## MAIOR GRANULARIDADE DOS SEGMENTOS
#------------------------------------------------------------------------------------------------
# Calcular a diversidade alfa usando o índice de Shannon e número de ASVs observados
diversity_shannon = alpha_diversity('shannon', data_numeric.values, ids=metadata.index)
diversity_observed_asv = np.sum(data_numeric.values > 0, axis=1)  # Contagem de ASVs observados
metadata['Shannon'] = diversity_shannon
metadata['Observed_ASV'] = diversity_observed_asv

# Agrupar dados por diferentes classes no campo "Experiment"
# Lista para armazenar os valores de Shannon e Observed_ASV para cada classe
shannon_groups = []
asv_groups = []

# Adiciona os dados de cada grupo em listas para o teste Kruskal-Wallis
for group in metadata['Experiment'].unique():
    shannon_groups.append(metadata[metadata['Experiment'] == group]['Shannon'])
    asv_groups.append(metadata[metadata['Experiment'] == group]['Observed_ASV'])

# Teste de Kruskal-Wallis para Shannon
h_shannon, p_shannon = kruskal(*shannon_groups)

# Teste de Kruskal-Wallis para ASV observados
h_asv, p_asv = kruskal(*asv_groups)

print(f"Teste Kruskal-Wallis para Shannon: H={h_shannon:.4f}, p={p_shannon:.4f}")
print(f"Teste Kruskal-Wallis para ASVs observados: H={h_asv:.4f}, p={p_asv:.4f}")

# Configurar a figura com subplots 1x2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Boxplot da contagem de ASVs observados (Observed_OTU) por categoria
sns.boxplot(data=metadata, x='Category', y='Observed_OTU', palette='Set2', ax=ax1)
sns.stripplot(data=metadata, x='Category', y='Observed_OTU', color='black', jitter=True, ax=ax1)
ax1.set_title("OTU's observados por Experimento")
ax1.set_ylabel("Conta de OTU's observada")
ax1.set_xlabel("Experimento")
# Ajustando a rotação dos rótulos do eixo x para evitar sobreposição
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax1.grid(True)

# Boxplot da diversidade de Shannon por categoria
sns.boxplot(data=metadata, x='Category', y='Shannon', palette='Set2', ax=ax2)
sns.stripplot(data=metadata, x='Category', y='Shannon', color='black', jitter=True, ax=ax2)
ax2.set_title("Diversidade alfa (Shannon) por Experimento")
ax2.set_ylabel("Índice de diversidade Shannon")
ax2.set_xlabel("Experimento")
# Ajustando a rotação dos rótulos do eixo x para evitar sobreposição
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
ax2.grid(True)

# Ajustar espaçamento entre os gráficos
plt.tight_layout()
plt.show()


# Plotando o gráfico de NMDS com separação por cor (Gut) e forma (Category)
plt.figure(figsize=(8, 6))
sns.scatterplot(data=metadata, x='NMDS1', y='NMDS2', hue='Gut compartiment', style='Category',
                palette='Set1', s=100, markers=markers)
plt.title('NMDS of Microbial Composition')
plt.xlabel('NMDS1')
plt.ylabel('NMDS2')
plt.legend(title='Gut and Category')


# Ajustando a rotação dos rótulos do eixo x para evitar sobreposição
plt.xticks(rotation=45)

# Ajustar o layout para que os rótulos fiquem menos sobrepostos
plt.tight_layout()

# Exibir a grade e o gráfico
plt.grid(True)

#mostrar grafico
plt.show()

# Criar um objeto DistanceMatrix com os IDs das amostras
dist_matrix_obj = DistanceMatrix(dist_matrix_square, ids=data.index)
permanova_results = permanova(dist_matrix_obj, metadata['Gut compartiment'], permutations=999)
print(f"PERMANOVA results: pseudo-F={permanova_results['test statistic']:.4f}, p-value={permanova_results['p-value']:.4f}")





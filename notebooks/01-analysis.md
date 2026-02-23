---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: md,ipynb
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python (bioreactor)
    language: python
    name: bioreactor
---

# Diversity Analysis Notebook


## Imports and configs


### Fonts setup

```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fm.findSystemFonts(fontpaths=["/usr/share/fonts/truetype/"])

plt.rcParams["font.family"] = "Times New Roman"

plt.text(0.5, 0.5, 'Teste com Times New Roman', fontsize=12, ha='center')
plt.show()

```

### Imports

```python
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from skbio.diversity import alpha_diversity
from scipy.spatial.distance import pdist, squareform
from skbio.diversity import beta_diversity
```

### Auxiliar functions

```python
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
```

## Data Preparation and Wrangling


### Metadata Loading

```python
meta_all = pd.read_csv('../drive/Bettle_experiments/01_metadata_productivity.txt', sep='\t', index_col = 0)
```

```python
meta_all.info()
```

### ASV loading

```python
asv_all = load_asv_file("../drive/Bettle_experiments/01_map_complete_relative_abundance_table.tsv")
asv_all.info()
```

### Merging on 'SampleID'

```python
merged = pd.merge(meta_all[['Gut compartiment', 'Experiment', 'Category']], asv_all,
                  how='inner', left_index=True, right_index=True)
merged.info()
```

#### Checking ASVs with count zero in all samples

```python
merged.columns[(merged==0).all()]
```

### Updating Category in Metadata Dataset

This part of the original notebook is kind of nebulous. 
The author doesn't explain why change only the '1T' and
'Inoculum' values of the 'Category' column. 

```python
def update_categoria(row):
    if row['Category'] == '1T':
      return f"1T-{row['Category2']}"
    elif row['Category'] == 'Inoculum':
        return f"Inoculum-{row['Category2']}"
    return row['Category']
```

```python
meta_all['Category'] = meta_all.apply(update_categoria, axis=1)
meta_all['Category'].unique()
```

## Visual Analysis


### Relevant imports

```python
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
```

```python
metadata = meta_all.loc[:, ['Gut compartiment', 'Experiment', 'Category']].copy()
metadata.shape
```

### Markers for graphs:

```python
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
```

### Bray-Curtis distance

```python
dist_matrix = pdist(asv_all, metric='braycurtis')
dist_matrix_square = squareform(dist_matrix)
print(dist_matrix.shape)
print(dist_matrix_square.shape)
```

### Alpha Diversity with Shannon index

#### A problem with the author's notebook?

Note that the author calculates alpha diversity using asv count values,
but metadata index. This produces different results than calculating
with asv values AND index, as shown below. I continue by calculating
with asv index, passing to a Series and then merging.

```python
alpha_diversity('shannon', asv_all.values, ids=asv_all.index).head()
```

```python
diversity = alpha_diversity('shannon', asv_all.values, ids=asv_all.index)
diversity.name = 'Shannon'
metadata = pd.merge(metadata, diversity, left_index=True, right_index=True, how='inner')
metadata['Shannon']
```

### Observed OTUs

```python
metadata['Observed_OTUs'] = np.sum(asv_all.values > 0, axis=1)
```

### Plots and T-Test by Gut Segment


#### Plotting Shannon Diversity Boxplot by Gut Segment

```python
plt.figure(figsize=(8, 6))

sns.boxplot(data=metadata, x='Gut compartiment', y='Shannon', 
            hue='Gut compartiment', palette='Set2')

sns.stripplot(data=metadata, x='Gut compartiment', y='Shannon',
              color='black', jitter=True)

plt.title('Alpha Diversity (Shannon) by Gut Segment')
plt.ylabel('Shannon Diversity Index')
plt.xlabel('Gut Segment')
plt.grid(True)
plt.show()
```

#### Plotting OTU Count Boxplot by Gut Segment

```python
plt.figure(figsize=(8, 6))
sns.boxplot(data=metadata, x='Gut compartiment', y='Observed_OTUs',
            hue='Gut compartiment', palette='Set2')
sns.stripplot(data=metadata, x='Gut compartiment', y='Observed_OTUs', color='black', jitter=True)
plt.title('Observed ASVs by Gut Segment')
plt.ylabel('Observed ASVs Count')
plt.xlabel('Gut Segment')
plt.grid(True)
plt.show()
```

#### T-Test by Gut Compartiment

```python
mid_group = metadata[metadata['Gut compartiment'] == 'Midgut']
hind_group = metadata[metadata['Gut compartiment'] == 'Hindgut']

t_shannon, p_shannon = ttest_ind(mid_group['Shannon'], hind_group['Shannon'])

t_asv, p_asv = ttest_ind(mid_group['Observed_OTUs'], hind_group['Observed_OTUs'])

dist_matrix_obj = DistanceMatrix(dist_matrix_square, ids=asv_all.index)
print(f"Teste t para Shannon: t={t_shannon:.4f}, p={p_shannon:.4f}")
print(f"Teste t para ASVs observados: t={t_asv:.4f}, p={p_asv:.4f}")
```

#### PERMANOVA

```python
permanova_results = permanova(dist_matrix_obj, metadata['Gut compartiment'], permutations=999)
print(f"PERMANOVA results: pseudo-F={permanova_results['test statistic']:.4f}, p-value={permanova_results['p-value']:.4f}")
```
### T-Test and Plots by Experiment

I suspect this could be problematic because, as the code comments
suggest, the author intended to do a T-Test of Shannon index and
Observed ASVs between HRT of 8 days and 2 days. The thing is, she
assigns to `group_8d` the metadata columns with 'Experiment' == 'Enrichment'
and `group_2d` with 'Experiment' == 'Reactor'. The enrichment experiment
metadata table has no info on HRT both in it's metadata file and in
the experiment's article. This is, then, a T-Test between experiments,
not HRTs.


#### T-Testing

```python
enrich_group = metadata[metadata['Experiment'] == 'Enrichement']
reactor_group = metadata[metadata['Experiment'] == 'Reactor']

t_shannon, p_shannon = ttest_ind(enrich_group['Shannon'], reactor_group['Shannon'])

t_asv, p_asv = ttest_ind(enrich_group['Observed_OTUs'], reactor_group['Observed_OTUs'])


print(f"Teste t para Shannon: t={t_shannon:.4f}, p={p_shannon:.4f}")
print(f"Teste t para ASVs observados: t={t_asv:.4f}, p={p_asv:.4f}")
```

#### Plotting Shannon and OTU Count By Experiment

```python
fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))

# Plotando o boxplot da diversidade de Shannon por Experiment
sns.boxplot(data=metadata, x='Experiment', y='Shannon', 
            hue='Experiment', palette='Set2', ax=ax3)

sns.stripplot(data=metadata, x='Experiment', y='Shannon', 
              color='black', jitter=True, ax=ax3)

ax3.set_title('Diversidade alfa (Shannon) por Experimento')
ax3.set_ylabel('Índice de diversidade Shannon')
ax3.set_xlabel('Experimento')
ax3.grid(True)

# Plotando o boxplot da contagem de ASVs observados
sns.boxplot(data=metadata, x='Experiment', y='Observed_OTUs', 
            hue='Experiment', palette='Set2', ax=ax4)

sns.stripplot(data=metadata, x='Experiment', y='Observed_OTUs', 
              color='black', jitter=True, ax=ax4)

ax4.set_title('OTU´s observados por Experimento')
ax4.set_ylabel('Conta de OTU´s observada')
ax4.set_xlabel('Experimento')
ax4.grid(True)

# Ajustar o espaçamento entre os gráficos
plt.tight_layout()
plt.show()
```

### NDMS Analysis


```python
mds = MDS(n_components=2, metric='precomputed', random_state=42,
          n_init=4, init='random')
nmds_results = mds.fit_transform(dist_matrix_square)

metadata['NMDS1'] = nmds_results[:, 0]
metadata['NMDS2'] = nmds_results[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(data=metadata, x='NMDS1', y='NMDS2', hue='Gut compartiment', style='Experiment',
                palette='Set1', s=100, markers=markers_exp)
plt.title('NMDS da composição microbiana')
plt.xlabel('NMDS1')
plt.ylabel('NMDS2')
plt.legend(title='Intestino e Experimento')
plt.grid(True)
plt.show()
```

### PERMANOVA for Beta Diversity

```python
dist_matrix_obj = DistanceMatrix(dist_matrix_square, ids=asv_all.index)
permanova_results = permanova(dist_matrix_obj, metadata['Experiment'], permutations=999)
print(f"PERMANOVA results: pseudo-F={permanova_results['test statistic']:.4f}, p-value={permanova_results['p-value']:.4f}")
```

### Kruskal-Wallis Test

```python
# Agrupar dados por diferentes classes no campo "Experiment"
# Lista para armazenar os valores de Shannon e Observed_ASV para cada classe
shannon_groups = []
asv_groups = []

# Adiciona os dados de cada grupo em listas para o teste Kruskal-Wallis
for group in metadata['Experiment'].unique():
    shannon_groups.append(metadata[metadata['Experiment'] == group]['Shannon'])
    asv_groups.append(metadata[metadata['Experiment'] == group]['Observed_OTUs'])

# Teste de Kruskal-Wallis para Shannon
h_shannon, p_shannon = kruskal(*shannon_groups)

# Teste de Kruskal-Wallis para ASV observados
h_asv, p_asv = kruskal(*asv_groups)

print(f"Teste Kruskal-Wallis para Shannon: H={h_shannon:.4f}, p={p_shannon:.4f}")
print(f"Teste Kruskal-Wallis para ASVs observados: H={h_asv:.4f}, p={p_asv:.4f}")
```

### Plotting Observed OTUs and Shannon Diversity per Experiment Category

First, set plot to two side-by-side plots:

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Observed OTUs by Category

sns.boxplot(data=metadata, x='Category', y='Observed_OTUs', 
            hue='Category', palette='Set2', ax=ax1)
sns.stripplot(data=metadata, x='Category', y='Observed_OTUs', 
              color='black', jitter=True, ax=ax1)
ax1.set_title("OTU's observados por Experimento")
ax1.set_ylabel("Contagem de OTU's observada")
ax1.set_xlabel("Experimento")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax1.grid(True)

# Shannon Diversity by Category

sns.boxplot(data=metadata, x='Category', y='Shannon', 
            hue='Category', palette='Set2', ax=ax2)
sns.stripplot(data=metadata, x='Category', y='Shannon', 
              color='black', jitter=True, ax=ax2)
ax2.set_title("Diversidade alfa (Shannon) por Experimento")
ax2.set_ylabel("Índice de diversidade Shannon")
ax2.set_xlabel("Experimento")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
ax2.grid(True)

plt.tight_layout()
plt.show()
```

### Plotting NDMS by Gut Compartiment and Category

```python
plt.figure(figsize=(8, 6))
sns.scatterplot(data=metadata, x='NMDS1', y='NMDS2', hue='Gut compartiment', style='Category',
                palette='Set1', s=100, markers=markers)
plt.title('NMDS of Microbial Composition')
plt.xlabel('NMDS1')
plt.ylabel('NMDS2')
plt.legend(title='Gut and Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)

#mostrar grafico
plt.show()

# Criar um objeto DistanceMatrix com os IDs das amostras
dist_matrix_obj = DistanceMatrix(dist_matrix_square, ids=asv_all.index)
permanova_results = permanova(dist_matrix_obj, metadata['Gut compartiment'], permutations=999)
print(f"PERMANOVA results: pseudo-F={permanova_results['test statistic']:.4f}, p-value={permanova_results['p-value']:.4f}")
```

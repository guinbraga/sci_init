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


#### Matplotlib markers

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

#### Plotly markers

```python
px_markers = {
    '1T': 0,           # círculo
    '2T': 1,           # quadrado
    '3T': 2,           # diamante
    'R1': 3,           # cruz
    'R2': 17,           # estrela
    'R3': 5,           # triangulo-cima
    'Foam': 6,         # triangulo-baixo
    'Inoculum': 18,     # hexagrama

    # Novos símbolos preenchidos para as categorias específicas
    'Inoculum-1': 13,   # pentágono
    'Inoculum-2': 313,   # pentagono aberto
    'Inoculum-R1': 22,  # diamante-estrela
    'Inoculum-R2': 122,  # diamante-estrela aberto
    'Inoculum-R3': 322,  # diamante-estrela aberto com ponto
    '1T-1': 0,         # círculo
    '1T-2': 300          # circulo aberto com ponto
}

px_markers_exp = {
    'Enrichement': 0,  # círculo
    'Reactor': 1,     # quadrado
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
### Plots and T-Test by Experiment

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

sns.boxplot(data=metadata, x='Experiment', y='Shannon', 
            hue='Experiment', palette='Set2', ax=ax3)

sns.stripplot(data=metadata, x='Experiment', y='Shannon', 
              color='black', jitter=True, ax=ax3)

ax3.set_title('Diversidade alfa (Shannon) por Experimento')
ax3.set_ylabel('Índice de diversidade Shannon')
ax3.set_xlabel('Experimento')
ax3.grid(True)

sns.boxplot(data=metadata, x='Experiment', y='Observed_OTUs', 
            hue='Experiment', palette='Set2', ax=ax4)

sns.stripplot(data=metadata, x='Experiment', y='Observed_OTUs', 
              color='black', jitter=True, ax=ax4)

ax4.set_title('OTU´s observados por Experimento')
ax4.set_ylabel('Conta de OTU´s observada')
ax4.set_xlabel('Experimento')
ax4.grid(True)

plt.tight_layout()
plt.show()
```

#### NDMS Analysis


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

#### PERMANOVA for Beta Diversity

```python
dist_matrix_obj = DistanceMatrix(dist_matrix_square, ids=asv_all.index)
permanova_results = permanova(dist_matrix_obj, metadata['Experiment'], permutations=999)
print(f"PERMANOVA results: pseudo-F={permanova_results['test statistic']:.4f}, p-value={permanova_results['p-value']:.4f}")
```

#### Kruskal-Wallis Test

```python
shannon_groups = []
asv_groups = []

for group in metadata['Experiment'].unique():
    shannon_groups.append(metadata[metadata['Experiment'] == group]['Shannon'])
    asv_groups.append(metadata[metadata['Experiment'] == group]['Observed_OTUs'])

h_shannon, p_shannon = kruskal(*shannon_groups)

h_asv, p_asv = kruskal(*asv_groups)

print(f"Teste Kruskal-Wallis para Shannon: H={h_shannon:.4f}, p={p_shannon:.4f}")
print(f"Teste Kruskal-Wallis para ASVs observados: H={h_asv:.4f}, p={p_asv:.4f}")
```

### Plots and staticstical tests per Experiment Category


#### Order set and category count

We'll set an order to be used in all boxplots:

```python
order = ['Inoculum-1', 'Inoculum-2', '1T-1', '1T-2', '2T', '3T',
         'Inoculum-R1', 'Inoculum-R2', 'Inoculum-R3', 'R1', 'R2',
         'R3', 'Foam']
cat_order = pd.api.types.CategoricalDtype(categories=order, ordered=True)
metadata['Category'] = metadata['Category'].astype(cat_order)
```

We'll also value_count by category:

```python
metadata.groupby('Category')['Category'].value_counts()
```


```python
metadata.groupby('Gut compartiment')['Category'].value_counts().sort_index()
```

#### No gut compartiment discretion

##### Observed_OTUs by category

```python
fig, ax1 = plt.subplots(figsize=(10, 6))

sns.boxplot(data=metadata, x='Category', y='Observed_OTUs', 
            hue='Category', palette='Set2', ax=ax1, order=order,
           dodge=False)

sns.stripplot(data=metadata, x='Category', y='Observed_OTUs', 
              color='black', jitter=True, ax=ax1, order=order,
             dodge=False)

ax1.set_title("OTU's observados por Experimento")
ax1.set_ylabel("Contagem de OTU's observada")
ax1.set_xlabel("Experimento")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax1.grid(True)
```
##### Shannon by category

```python
fig, ax2 = plt.subplots(figsize=(10, 6))

sns.boxplot(data=metadata, x='Category', y='Shannon', 
            hue='Category', palette='Set2', ax=ax2, order=order)

sns.stripplot(data=metadata, x='Category', y='Shannon', 
              color='black', jitter=True, ax=ax2, order=order)

ax2.set_title("Diversidade alfa (Shannon) por Experimento")
ax2.set_ylabel("Índice de diversidade Shannon")
ax2.set_xlabel("Experimento")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
ax2.grid(True)

```

#### Midgut compartiment discretion

First we'll select the midgut data:

```python
mid_data = metadata[metadata['Gut compartiment'] == 'Midgut']
```

##### Observed_OTUs by experiment category

```python
fig, ax1 = plt.subplots(figsize=(10, 6))

sns.boxplot(data=mid_data, x='Category', y='Observed_OTUs', 
            hue='Category', palette='Set2', ax=ax1, order=order,
           dodge=False)

sns.stripplot(data=mid_data, x='Category', y='Observed_OTUs', 
              color='black', jitter=True, ax=ax1, order=order,
           dodge=False)

ax1.set_title("Observed Midgut OTUs by Experiment Category")
ax1.set_ylabel("Observed OTUs count")
ax1.set_xlabel("Experiment category")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax1.grid(True)
```

##### Shannon index by experiment category


```python
fig, ax2 = plt.subplots(figsize=(10, 6))

sns.boxplot(data=mid_data, x='Category', y='Shannon', 
            hue='Category', palette='Set2', ax=ax2, order=order,
           dodge=False)

sns.stripplot(data=mid_data, x='Category', y='Shannon', 
              color='black', jitter=True, ax=ax2, order=order,
             dodge=False)

ax2.set_title("Midgut Shannon Alpha Diversity by experiment category")
ax2.set_ylabel("Shannon Alpha Diversity Index")
ax2.set_xlabel("Experiment Category")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
ax2.grid(True)
```

#### Hindgut compartiment discretion

Select hindgut data:

```python
hind_data = metadata[metadata['Gut compartiment'] == 'Hindgut']
```

##### Observed_OTUs by experiment category

```python
fig, ax1 = plt.subplots(figsize=(10, 6))

sns.boxplot(data=hind_data, x='Category', y='Observed_OTUs', 
            hue='Category', palette='Set2', ax=ax1, order=order,
           dodge=False)

sns.stripplot(data=hind_data, x='Category', y='Observed_OTUs', 
              color='black', jitter=True, ax=ax1, order=order,
              dodge=False)
ax1.set_title("Observed Hindgut OTUs by Experiment Category")
ax1.set_ylabel("Observed OTUs count")
ax1.set_xlabel("Experiment category")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax1.grid(True)
```

##### Shannon index by experiment category


```python
fig, ax2 = plt.subplots(figsize=(10, 6))

sns.boxplot(data=hind_data, x='Category', y='Shannon', 
            hue='Category', palette='Set2', ax=ax2, order=order,
           dodge=False)

sns.stripplot(data=hind_data, x='Category', y='Shannon', 
              color='black', jitter=True, ax=ax2, order=order,
             dodge=False)

ax2.set_title("Hindgut Shannon Alpha Diversity by experiment category")
ax2.set_ylabel("Shannon Alpha Diversity Index")
ax2.set_xlabel("Experiment Category")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
ax2.grid(True)
```

#### Side-by-Side gut compartiment comparison

We'll do Shannon, then Observed OTUs side by side for
better visualization

##### Shannon Diversity

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sns.boxplot(data=mid_data, x='Category', y='Shannon', 
            hue='Category', palette='Set2', ax=ax1, order=order,
           dodge=False)

sns.stripplot(data=mid_data, x='Category', y='Shannon', 
              color='black', jitter=True, ax=ax1, order=order,
           dodge=False)

ax1.set_title("Midgut Shannon Alpha Diversity by experiment category")
ax1.set_ylabel("Shannon Alpha Diversity Index")
ax1.set_xlabel("Experiment Category")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax1.grid(True)

sns.boxplot(data=hind_data, x='Category', y='Shannon', 
            hue='Category', palette='Set2', ax=ax2, order=order,
           dodge=False)

sns.stripplot(data=hind_data, x='Category', y='Shannon', 
              color='black', jitter=True, ax=ax2, order=order,
           dodge=False)

ax2.set_title("Hindgut Shannon Alpha Diversity by experiment category")
ax2.set_ylabel("Shannon Alpha Diversity Index")
ax2.set_xlabel("Experiment Category")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
ax2.grid(True)

```

##### Observed OTUs

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sns.boxplot(data=mid_data, x='Category', y='Observed_OTUs', 
            hue='Category', palette='Set2', ax=ax1, order=order,
           dodge=False)

sns.stripplot(data=mid_data, x='Category', y='Observed_OTUs', 
              color='black', jitter=True, ax=ax1, order=order,
           dodge=False)

ax1.set_title("Midgut Observed OTUs by experiment category")
ax1.set_ylabel("Observed OTUs Count")
ax1.set_xlabel("Experiment Category")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
ax1.grid(True)

sns.boxplot(data=hind_data, x='Category', y='Observed_OTUs', 
            hue='Category', palette='Set2', ax=ax2, order=order,
           dodge=False)

sns.stripplot(data=hind_data, x='Category', y='Observed_OTUs', 
              color='black', jitter=True, ax=ax2, order=order,
           dodge=False)

ax2.set_title("Hindgut Observed OTUs by experiment category")
ax2.set_ylabel("Observed OTUs Count")
ax2.set_xlabel("Experiment Category")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
ax2.grid(True)

```
### Plotting NDMS by Gut Compartiment and Category

#### Plotly interactive plot

```python
import plotly.express as px
fig = px.scatter(metadata, x='NMDS1', y='NMDS2', color='Gut compartiment',
                symbol='Category', category_orders={'Gut compartiment': ['Midgut', 'Hindgut'],
                                                    'Category': order},
                symbol_map=px_markers, title='nMDS of Microbial Distribution')
fig.update_traces(marker={'size':14})
fig.update_layout(width=1200, height=700)
fig.update_yaxes(range=[-0.7, 0.8])
fig.update_xaxes(range=[-0.8, 0.8])
for point in fig.data:
    group = point.legendgroup.split(',')[0]
    point.legendgroup = group
    point.legendgrouptitle.text = group
    count = len(point.x)
    point.name = point.name.split(',')[1].strip() + f' (n = {count})'
fig.update_layout(legend_groupclick="toggleitem")

show_all = [True] * len(fig.data)
hide_all = ['legendonly'] * len(fig.data)
midgut_only = [True if trace.legendgroup == 'Midgut' else 'legendonly' for trace in fig.data]
hindgut_only = [True if trace.legendgroup == 'Hindgut' else 'legendonly' for trace in fig.data]

fig.update_layout(
    title_pad_l=300,
    updatemenus=[
        dict(
            type='buttons',
            direction='left',
            xanchor='left',
            x=1.0, # Adjust X and Y to position the buttons where you like
            y=1.16,
            showactive=False,
            buttons=[
                dict(label='Show All', method='restyle', args=['visible', show_all]),
                dict(label='Hide All', method='restyle', args=['visible', hide_all]),
            ]
        ),
        dict(
            type='buttons',
            direction='left',
            xanchor='left',
            x=1.0, # Adjust X and Y to position the buttons where you like
            y=1.1,
            showactive=False,
            buttons=[
                dict(label='Midgut Only', method='restyle', args=['visible', midgut_only]),
                dict(label='Hindgut Only', method='restyle', args=['visible', hindgut_only]),
            ]
        )
    ]
)
fig.show()
fig.write_html('nmds_gut_cat.html')
```


#### Matplotlib regular plot

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=metadata, x='NMDS1', y='NMDS2', hue='Gut compartiment',
                style='Category', palette='Set1', s=100, markers=markers)
plt.title('NMDS of Microbial Composition')
plt.xlabel('NMDS1')
plt.ylabel('NMDS2')
plt.legend(title='Gut and Category', bbox_to_anchor=(1.04, 1), loc='upper left')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)

#mostrar grafico
plt.show()
```

```python

# Criar um objeto DistanceMatrix com os IDs das amostras
dist_matrix_obj = DistanceMatrix(dist_matrix_square, ids=asv_all.index)
permanova_results = permanova(dist_matrix_obj, metadata['Gut compartiment'], permutations=999)
print(f"PERMANOVA results: pseudo-F={permanova_results['test statistic']:.4f}, p-value={permanova_results['p-value']:.4f}")
```

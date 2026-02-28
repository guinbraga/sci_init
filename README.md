# Metagenomic Bioindicator Analysis for Biogas Production Assessment in Anaerobic Microbiomes
## Overview
This repository contains analyses of two datasets composed of metagenomes from microbiomes. The project focuses on selecting relevant bioindicators for predicting quantitative metrics involved in biogas production assessment. Exploratory analyses will be included to understand the microbial diversity in the samples along with the interpretation of trained classification models.

## Repository Structure
- notebooks/: Jupyter notebooks containing the analyses.
- scripts/: Scripts used in the analyses.
- outputs/: Results from the analyses.
-docs/: Additional documentation.

## Setup
1. Clone this repository:
    ```bash
    https://github.com/Anajulia-gon/Bettle-data-analysis.git
    ```
2. Install the dependencies:
    ```bash
    pip install -r docs/requirements.txt
    ```
    or
    ```bash
    poetry install
    ```
    and activate virtual environment created
    ```bash
    poetry shell
    ```
   
## Execution
- **Data Collection**: The datasets are stored on a private server, consisting of separate FASTA files averaging 20.8475GB and 16.4627GB, sourced from Data Sources 1 and 2, respectively. The total size is 2,125.63GB.
- **Data Preprocessing**: Using the MuDoGeR tool, sequences were cleaned and organized, while biologically significant information was extracted and summarized into tables.
- **Data Loading and Transformation**: Relative abundance tables of OTUs were loaded into DataFrames for further manipulation. Typing and grouping were performed, and metadata tables were merged to create categorical variable columns.
- **Exploratory Data Analysis (EDA)**: Alpha and Beta diversity of the communities were explored, along with the visualization of categorical data.
- **Data Modeling**: The data was prepared for machine learning pipelines.
- **Model Evaluation**: Model performance was assessed using G-mean and GridSearch.
- Model Interpretation: SHAP plots were generated for model interpretation.

## Main Results
### Microbial Diversity
The composition and diversity of the microbiota varied between the enrichment and reactor experiments. Alpha diversity metrics showed significantly lower observed OTU counts (paired t-test, P < 0.05) and slightly higher, though not statistically significant, Shannon index values (paired t-test, P > 0.05) in the enrichment experiment compared to the reactor (Fig.3). This suggests that, while total species richness (observed OTUs) was higher in the enrichment, diversity measured by the Shannon index did not differ significantly.

Beta diversity analysis revealed a significant difference between the communities in the two experiments (PERMANOVA; Pseudo-F = 14.0135, P < 0.01), indicating that community structures between the enrichment and reactor experiments are clearly distinct, despite some general similarities in alpha diversity metrics. This is further confirmed by NMDS analysis (Fig.2), which shows a considerable separation between microbial communities based on composition and diversity across experiments (represented by squares and circles).

In summary, these results indicate that each experiment’s microbiota exhibits significant differences in structure and composition, even though alpha diversity metrics do not display substantial variation. Consequently, the models will address a straightforward classification task to differentiate samples from the enrichment experiment and the bioreactor. This pattern is also observed in the first datase
### Microbial Functions
Initial analysis of the role played by the main attributes of the microbial communities.(avaliable in oficial reports).

### Contact information
Autor: Ana Julia  
Email: ana.tendulini@gmail.com

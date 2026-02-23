#!/usr/bin/env python
# coding: utf-8

# # 05-hrt_processing
#
# ## Imports and configs

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ## Preparing and Merging datasets

# In[ ]:


metadata = pd.read_csv(
    "../drive/C6-C8_productivity_experiment/01_metadata_productivity.txt"
)
# metadata.info()


# In[ ]:


# metadata['HRT'].value_counts()


# In[ ]:


abs_hits = pd.read_csv(
    "../drive/C6-C8_productivity_experiment/01_map_complete_relative_abundance_table.csv",
    index_col=0,
)
# abs_hits.info()


# In[ ]:


abs_hits.columns.name = "sample"
abs_hits.index.name = "ASV"
# len(abs_hits.columns)


# In[ ]:


# print(abs_hits.isna().sum().sum())
# print(metadata.isna().sum().sum())


# In[ ]:


merge = pd.merge(
    abs_hits.T,
    metadata[["Sample", "HRT"]],
    left_index=True,
    right_on="Sample",
    how="inner",
).set_index("Sample")
# merge.info()


# In[ ]:


merge = merge[(merge["HRT"] == 2) | (merge["HRT"] == 8)]
# merge.info()


# ## Train Test split

# In[ ]:


y = merge["HRT"]
X = merge.drop("HRT", axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)


# In[ ]:


class_mapping = dict(
    zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_)
)
print(f"Mapeamento de classes: {class_mapping}")


# In[ ]:


print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


#

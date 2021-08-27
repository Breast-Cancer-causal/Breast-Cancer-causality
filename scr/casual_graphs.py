# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 00:29:48 2021

@author: Smegn
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from causalnex.structure.notears import from_pandas
from IPython.display import Image
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
# silence warnings
import warnings
warnings.filterwarnings("ignore")

from causalnex.structure import StructureModel
sm = StructureModel()

plt.rcParams.update({'figure.figsize': (12.0, 8.0)})
plt.rcParams.update({'font.size': 14})


# Load the dataset
data = pd.read_csv(r'C:/Users/Smegn/Documents/GitHub/Breast-Cancer/data/data.csv')
#drop unnecessary column
#print(data.isnull())
data.drop('Unnamed: 32', axis=1, inplace=True)
# add these relationships into our structure model:
non_numeric_columns = list(data.select_dtypes(exclude=[np.number]).columns)
le = LabelEncoder()
for col in non_numeric_columns:
    data[col] = le.fit_transform(data[col])
#apply the NOTEARS algorithm to learn the structure.

sm = from_pandas(data)

#visualise the learned StructureModel using the plot function.
viz = plot_structure(sm, graph_attributes={"scale": "0.5"},all_node_attributes=NODE_STYLE.WEAK, all_edge_attributes=EDGE_STYLE.WEAK)
Image(viz.draw(format='png'))
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np; np.random.seed(42)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

 
df = pd.read_csv('donnees/base_prospect.csv')

feature = ['ca_total_FL','ca_export_FK']


data_plot = df[feature]

sns.boxplot(x="variable", y="value", data=pd.melt(data_plot),showfliers=True,whis=1.5)

plt.show()
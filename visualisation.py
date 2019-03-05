#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from pandas.plotting import radviz

cmap = cm.get_cmap('gnuplot')

data = pd.read_csv("base_prospect.csv", encoding='latin-1')

labels =[ 'dept', 'effectif', 'ca_total_FL', 'ca_export_FK', 'endettement',
       'evo_benefice', 'ratio_benef', 'evo_effectif', 'evo_risque', 'age',
       'chgt_dir', 'rdv']

for label in data.columns:
	nan  = data.loc[lambda df: df[label].isnull()]
	nombre = len(nan[label])
	percent = (nombre*100)/float(np.size(data,0))
	if(percent >0):
                print("=========="+label+"==========")
                print("nombre 	:"+str(nombre))
                print("Pourcentage 	:"+str(percent))
                plt.pie([percent,100-percent], labels=["NA","other"], autopct='%.0f%%')
                plt.title(label)
                plt.savefig(label)
                plt.close()
#plt.plot(data.ca_total_FL,data.ratio_benef)
#plt.scatter(data.ca_total_FL,data.ratio_benef)
#plt.savefig("corr")

#Encodage et nature de données.
"""
  ==========ca_export_FK==========
  nombre  :6477
  Pourcentage   :5.96540671972
  iL ya de valeurs manquantes dans cette colonne correspondant à 5.97 % des donnees de la base.
  => Correspond à 6477, un chiffre non negligeable.
  Solutions:
    - remplacement des valeurs par la moyenne de la plage de valeurs grace à la classe SimpleImputer de la librairie sklearn.

"""

"""
   ==========risque==========
  nombre  :936
  Pourcentage   :0.862068965517

  Pourcentage trés faible correspondant à  936 lignes.
  => suppression des lignes
"""
risque_undefined_rows = data.loc[lambda df: df['risque'].isnull()]
data.drop(risque_undefined_rows.index)


"""
  ==========evo_risque==========
  nombre  :1538
  Pourcentage   :1.41651930445

  Pourcentage faible correspondant à 1538 ligne.

  On remarque aussi que 87% de ces instances(où evo_risque == NA) on pour valeur de rdv =0.
  vu qu'à la base 89.7 % des instances de la base on un rdv =0, donc la meilleure solution est d'effacer
  toutes les instances dont la colonne evo_risque == NA pour que cet attribut soit un peu plus discrminant pour l'attribut rdv.
  
"""
evo_risque_undefined_rows = data.loc[lambda df: df['evo_risque'].isnull()]
data.drop(evo_risque_undefined_rows.index)



"""
  ==========type_com==========
  nombre  :1079
  Pourcentage   :0.99377394636
  pourcentage trés faible correspondant à 1079 lignes.
"""
type_com_undefined_rows = data.loc[lambda df: df['type_com'].isnull()]
data.drop(evo_risque_undefined_rows.index)


"""
  ==========chgt_dir==========
  nombre  :35766
  Pourcentage   :32.9409814324
  attribut de nature catégoricale encodé en numérique [0,1].
  pourcentage trés elevés.
  La meilleure solution est de créer une nouvelle categorie reprensenté par 2.
"""
exit()
for label in labels:
	data[label]=data[label].apply(lambda x: x if pd.notnull(x) else 1.0) 
print(min(data["age"]))
exit()
cmap = cm.get_cmap('gnuplot')
df = data[labels]
df.astype('int64')
#pd.plotting.scatter_matrix(df,c= data['dept'], s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap = cmap)
#data.plot.hist()
#pd.plotting.scatter_matrix(data,c= data['effectif'], s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap = cmap, diagonal="kde")
for attr in labels:
	df[attr].plot.hist()
	plt.savefig ('distribution_'+attr)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from pandas.plotting import radviz
from sklearn.impute import SimpleImputer
import math
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from R_square_clustering import r_square
from sklearn. cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from scipy.cluster import hierarchy


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
   ==========risque==========
  nombre  :936
  Pourcentage   :0.862068965517

  Pourcentage trés faible correspondant à  936 lignes.
  => suppression des lignes
  les notes sont supposées etre toutes superieur ou égal à 7.
  donc on supprime aussi les instances don le risaue est "1-6"
  Encodage:

"""
risque_undefined_rows = data.loc[lambda df: df['risque'].isna()]

data = data.drop(risque_undefined_rows.index)


"""
  ==========evo_risque==========
  nombre  :1538
  Pourcentage   :1.41651930445

  Pourcentage faible correspondant à 1538 ligne.

  On remarque aussi que 87% de ces instances(où evo_risque == NA) on pour valeur de rdv =0.
  vu qu'à la base 89.7 % des instances de la base on un rdv =0, donc la meilleure solution est d'effacer
  toutes les instances dont la colonne evo_risque == NA pour que cet attribut soit un peu plus discrminant pour l'attribut rdv.
  
"""
evo_risque_undefined_rows = data.loc[lambda df: df['evo_risque'].isna()]
data = data.drop(evo_risque_undefined_rows.index)



"""
  ==========type_com==========
  nombre  :1079
  Pourcentage   :0.99377394636
  pourcentage trés faible correspondant à 1079 lignes.
"""
type_com_undefined_rows = data.loc[lambda df: df['type_com'].isna()]
data =data.drop(type_com_undefined_rows.index)

"""
  ==========chgt_dir==========
  nombre  :35766
  Pourcentage   :32.9409814324
  attribut de nature catégoricale encodé en numérique [0,1].
  pourcentage trés elevés.
  La meilleure solution est de créer une nouvelle categorie reprensenté par 2.
"""
data['chgt_dir']=data['chgt_dir'].apply(lambda x: 2 if math.isnan(x) else x)

"""
  ==========ca_export_FK==========
  nombre  :6477
  Pourcentage   :5.96540671972
  iL ya de valeurs manquantes dans cette colonne correspondant à 5.97 % des donnees de la base.
  => Correspond à 6477, un chiffre non negligeable.
  Solutions:
    - remplacement des valeurs par la moyenne de la plage de valeurs grace à la classe SimpleImputer de la librairie sklearn.

"""
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
data[['ca_export_FK','evo_risque']]=imp_mean.fit_transform(data[['ca_export_FK','evo_risque']])

standardscaler = preprocessing.StandardScaler()
'''
profil : geographique
'''
'''
  CLASSIFICATION NON SUPERVISE( k-MEAN)
'''
labels.remove('rdv')
labels.remove('dept')

X_norm = standardscaler.fit_transform(data[labels])

lst_k=range(1,20)
lst_rsq = []
'''for k in lst_k:
    est=KMeans(n_clusters=k)
    est.fit (X_norm)
    lst_rsq.append(r_square(X_norm, est.cluster_centers_,est.labels_,k))
fig = plt. figure ()
plt.plot(lst_k, lst_rsq, 'bx-')
plt.xlabel('k')
plt.ylabel('RSQ')
plt.title ('The Elbow Method showing the optimal k')
plt.savefig('r_square')
plt.close(fig)'''


cmap = cm.get_cmap('gnuplot')
est=KMeans(n_clusters=11)


data['clusters'] = est.fit_predict(data[labels])
labels.extend(['clusters','rdv','dept','code_cr','x','y'])
pca = PCA(n_components=2)
data['x'] = pca.fit_transform(X_norm)[:,0]
data['y'] = pca.fit_transform(X_norm)[:,1]

centroids = est.cluster_centers_
'''plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='d',
            cmap=cmap)
plt.scatter(data.loc[lambda df:df['rdv']==0]['x'], data.loc[lambda df:df['rdv']==0]['y'],marker='o', cmap=cmap)
'''

# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labs = ['cluster 1', 'cluster 2', 'cluster 3', 'cluster 4', 'cluster 5', 'cluster 6', 'cluster 7', 'cluster 8','cluster 9','cluster 10','cluster 11']
m=[]
gt = data.loc[lambda df:df['rdv']==1.0].groupby('clusters').count()['rdv']
for lab in labs:
	m.append(gt[labs.index(lab)])
explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
print(m)
exit()
fig1, ax1 = plt.subplots()
ax1.pie(m, labels=labs, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()
plt.title("11 clusters")
plt.savefig("kmeans0")
exit()
plt.scatter(data.loc[:,"ca_total_FL"], data.loc[:,"ratio_benef"],marker='o', c=data['rdv'])
centroids = est.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='o',
            cmap=cmap)

plt.title("11 clusters")
plt.savefig("kmeans")
exit()
clf = svm.SVC(gamma='scale')

X_norm = standardscaler.fit_transform(data[labels])
X_train, X_test, y_train, y_test = train_test_split(data[labels], data['rdv'], test_size = 0.20) 
clf.fit(X_train, y_train)
exit()

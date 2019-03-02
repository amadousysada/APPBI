import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

data = pd.read_csv("base_prospect.csv", encoding='latin-1')

labels =[ 'dept', 'effectif', 'ca_total_FL', 'ca_export_FK', 'endettement',
       'evo_benefice', 'ratio_benef', 'evo_effectif', 'evo_risque', 'age',
       'chgt_dir', 'rdv']
for label in data.columns:
	nan  = data.loc[lambda df: df[label].isnull()]
	nombre = len(nan[label])
	percent = (nombre*100)/np.size(data,0)
	print("=========="+label+"==========")
	print("nombre 	:"+str(nombre))
	print("Pourcentage 	:"+str(percent))
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
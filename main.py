import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, BicScore, K2Score
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import KBinsDiscretizer

data = pd.read_csv("data/data_auto_mpg.csv")
# data = pd.DataFrame(np.random.randn(500, 5), columns=list('ABCDE'))
# data['F'] = data['A'] * data['B']

for col in data.columns:
	if(data[col].dtype == np.float64 or data[col].dtype == np.float32):
		# bin_size = np.unique(data[col].values).shape[0]
		# kbins = KBinsDiscretizer(n_bins=bin_size, encode='ordinal', strategy='uniform').fit(data[col].values.reshape(-1,1))
		# data[col] = kbins.transform(data[col].values.reshape(-1,1)).astype(np.int64)
		data[col] = data[col].astype(np.int64)

data = data.iloc[:, :10]

print(data.dtypes)
print(data)

print("aq")
est = HillClimbSearch(data, scoring_method=K2Score(data))
print("aq")
model = est.estimate(max_indegree=5)	
print("aq")

print(model.edges)

plt.figure()
nx.draw_networkx(model)
plt.show()
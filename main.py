import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, BicScore, K2Score
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import KBinsDiscretizer

data = pd.read_csv("data/housing.csv")
# data = pd.DataFrame(np.random.randn(500, 5), columns=list('ABCDE'))
# data['F'] = data['A'] * data['B']

for col in data.columns:
	if(data[col].dtype == np.float64 or data[col].dtype == np.float32):
		data[col] = np.round(data[col], 2)

data = data.iloc[:, :5]

print(data.dtypes)

print("aq")
est = HillClimbSearch(data, scoring_method=BicScore(data))
print("aq")
model = est.estimate()
print("aq")

print(model.edges)

plt.figure()
nx.draw_networkx(model)
plt.show()
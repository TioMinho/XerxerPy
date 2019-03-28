import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, BicScore, K2Score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import networkx as nx
import pickle
import sys

def BuildCaRT(T, X, y):
	m = 0
	
	if(T[y].dtype == np.float64 or T[y].dtype == np.float32):
		m = DecisionTreeRegressor()
	else:
		m = DecisionTreeClassifier()

	m.fit(T[list(X)], T[y])
	modelSize = sys.getsizeof(pickle.dumps(m))

	return (m, modelSize)


def MaxIndependentSet(T, e, G, neighborhood):
	# Algorithm
	X_mat = set(T.columns);	X_pred = set()
	PRED = { Xi : set() for Xi in T.columns }
	improve = True

	while improve:
		mater_neighbors = {}
		NEW_PRED = {}
		cost_change = np.zeros(len(X_mat))

		for i,Xi in enumerate(X_mat):
			# Line 5
			neigh_Xi = neighborhood(G, Xi)
			mater_neighbors[Xi] = X_mat.intersection(neigh_Xi)
			for x in neigh_Xi:
				mater_neighbors[Xi].union(PRED[x])
			mater_neighbors[Xi].difference(Xi)

			# print(mater_neighbors[Xi])
			#

			# M1, size_M1 = BuildCaRT(T, mater_neighbors[Xi], Xi)
			# Let PRED(Xi) C mater_neighbors(Xi) be the set of predictor attributes used in M
			PRED[Xi] = mater_neighbors[Xi]	

			cost_change[i] = 0
			for Xj in [xj for xj in X_pred if Xi in PRED[xj]]:
				print("A")
				NEW_PRED[(i, Xj)] = PRED[Xj].difference(Xi).union(PRED[Xi])
				
				# M2, size_M2 = BuildCaRT(T, NEWPRED[i, Xj], Xj)
				# Set NEW_PRED_i(Xj) to the (sub)set of predictors attributes used in M
				(_, predCost_j) 	= BuildCaRT(T, PRED[Xj], Xj)
				(_, newPredCost_j) 	= BuildCaRT(T, NEW_PRED[i, Xj], Xj)

				cost_change[i] += predCost_j - newPredCost_j

				# print(cost_change[i])

		# Build an undirected, node-weighted graph Gtemp = (Xmat, Etemp) such that:
		#	Etemp := {(X,Y) : for all pair(X,Y) in PRED(Xj) for some Xj in X_pred} U {(Xi,Y) : for all Y in PRED(Xi), Xi in X_mat}
		#	weight(Xi) = MaterCost(Xi) - PredCost(PRED(Xi) -> Xi) + cost_change[i]    for each Xi in X_mat
		#
		# Solves the Weighted Maximum Independent Set (WMIS) or Maximum Weight Clique (MWC) problem
		#	S = FindWMIS(Gtemp)
		#
		# See: [1] JIANG, Hua; CHU-MIN, Li; MANYÀ, Felip. An Exact Algorithm for the Maximum Weight Clique Problem in Large Graphs. 
		# 			Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence (AAAI-17)

		if(sum([weight(x) for x in S]) <= 0): improve = false
		else:
			for Xj in X_pred:
				if(PRED[Xj].intersection(S) == set(Xi)):
					PRED[X_j] = NEW_PRED[i,Xj]

				X_mat 	= X_mat.difference(S)
				X_pred 	= X_pred.intersection(S)


		break

def pi(G, Xi):
	return set([p for p,f in G.edges if f == Xi])

def beta(G, xi):
	pass

data = pd.read_csv("../data/asia.csv")
newData = data.copy()

for col in newData.columns:
	if(newData[col].dtype == np.float64 or newData[col].dtype == np.float32):
		newData[col] = newData[col].astype(np.int64)

newData = newData.iloc[:, :7]
e_t = [1, 1, 1, 1, 1, 1]

G = HillClimbSearch(newData, scoring_method=K2Score(newData)).estimate(max_indegree=5)

MaxIndependentSet(data, e_t, G, pi)


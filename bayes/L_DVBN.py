import numpy as np 
import pandas as pd 
from scipy.special import gammaln as lfact

def combine_spouses_data(spouse_value_matrix):
	n_ins = spouse_value_matrix.shape[0]
	combined_data = np.array([spouse_value_matrix[i,:] for i in range(0, n_ins)])
	return combined_data

def parent_class_combine(data_matrix, parent_set):
	n_parent = len(parent_set)
	N = data_matrix.shape[0]

	parent_data_set = np.array([data_matrix[:, parent_set[i]] for i in range(1,n_parent)])
	parent_data_set = combine_spouses_data(parent_data_set)

	class_combine = len(np.unique(parent_data_set))

	return class_combine

def log_prob_single_edge_last_term(class_):
	n = len(class_)
	uniq_cl = np.unique(class_)
	val_cl = len(uniq_cl)

	distr_table = np.zeros([n, val_cl])
	for index_data in range(0, n):
		class_cl = np.where(uniq_cl == class_[index_data])[0]
		z = np.zeros(val_cl)
		z_modify = np.zeros(val_cl)
		z_modify[class_cl] = 1
		distr_table[index_data] = z_modify

	distr_intval_table = np.zeros([n, n, n])
	for ind_init in range(0,n):
		for ind_end in range(ind_init, n):
			if ind_init == ind_end:
				current_distr = distr_table[ind_init]
			else:
				current_distr = distr_intval_table[ind_init, ind_end-1] + distr_table[ind_end]

			distr_intval_table[ind_init, ind_end] = current_distr

	inv_p = np.zeros([n, n])
	for ind_init in range(0,n):
		for ind_end in range(0,n):
			if ind_end < ind_init:
				inv_p[ind_init, ind_end] = np.Inf
			else:
				distr_in_intval = distr_intval_table[ind_init, ind_end]
				current = lfact(ind_end-ind_init+1 + 1)
				for ind_cl in range(0, val_cl):
					current -= lfact(distr_in_intval[ind_cl] + 1)

				inv_p[ind_init, ind_end] = current

	return inv_p

def log_prob_spouse_child_data(child, spouse):
	n = len(child)
	uniq_sp = np.unique(spouse)
	uniq_ch = np.unique(child)
	val_sp = len(uniq_sp)
	val_ch = len(uniq_ch)

	distr_table = np.zeros([n, val_sp, val_ch])

	for index_data in range(0, n):
		class_ch = np.where(uniq_ch == child[index_data])[0]
		class_sp = np.where(uniq_sp == spouse[index_data])[0]
		z = np.zeros(val_ch)
		z_modify = np.zeros(val_ch)
		z_modify[class_ch] = 1

		for class_sp_index in range(0, val_sp):
			if(class_sp_index == class_sp):
				distr_table[index_data, class_sp_index] = z_modify
			else:
				distr_table[index_data, class_sp_index] = z

	distr_intval_table = np.zeros([n, n, val_sp, val_ch])

	for ind_val_sp in range(0, val_sp):
		for ind_init in range(0, n):
			for ind_end in range(0, n):
				if ind_init == ind_end:
					current_distr = distr_table[ind_end, ind_val_sp]
				else:
					current_distr = distr_intval_table[ind_init, ind_end-1, ind_val_sp] + distr_table[ind_end, ind_val_sp]

				distr_intval_table[ind_init, ind_end, ind_val_sp] = current_distr

	inv_p = np.zeros([n, n])

	for ind_init in range(0, n):
		for ind_end in range(0, n):
			if ind_end < ind_init:
				inv_p[ind_init, ind_end] = np.Inf
			else:
				current_val = 0.0
				for ind_val_sp in range(0, val_sp):
					distr_in_intval = distr_intval_table[ind_init, ind_end, ind_val_sp]
					total_num = sum(distr_in_intval)

					current = lfact(total_num+1)
					for ind_val_ch in range(0, val_ch):
						current -= lfact(distr_in_intval[ind_val_ch]+1)

					current += lfact(total_num + val_ch -1+1) - lfact(val_ch-1+1) - lfact(total_num+1)

					current_val += current 

				inv_p[ind_init, ind_end] = current_val

	return inv_p

tst = np.random.randn(500,4)

tst2 = log_prob_spouse_child_data(tst[:,2], tst[:,3])
print(tst2)




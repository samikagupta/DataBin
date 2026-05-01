import pandas as pd
import random
from sklearn.model_selection import train_test_split
from scipy.interpolate import make_interp_spline
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
from matplotlib import pyplot as plt
import mdlp as md
import copy
from scipy import interpolate


def gen_freq_bin(column_data, num_bins, increment):
    # print(column_data)
    val_ct_all = column_data.value_counts().to_dict()
    sorted_vals = dict(sorted(val_ct_all.items()))
    bin_freq = len(column_data)/num_bins
    count = 0
    bounds = [column_data.min()-increment]
    labels = []
    for v in sorted_vals:
        if sorted_vals[v] + count > bin_freq:
            bounds.append(v)
            count = 0
            labels.append((bounds[-1]+bounds[-2])/2)
        else:
            count += sorted_vals[v] 
    bounds.append(column_data.max()+increment)
    labels.append((bounds[-1]+bounds[-2])/2)
    column_data = pd.cut(column_data, bounds, labels=labels).astype(float)
    return column_data, bounds, labels


##########################################################################################

def gen_equal_bin(column_data, num_bins, increment):
    range_val = column_data.max()-column_data.min()
    inc = range_val/num_bins
    bin_b = [column_data.min()-increment]
    labels = [column_data.min() + inc/2]
    for x in range(1,num_bins):
        bin_b.append(bin_b[-1]+inc)
        labels.append(bin_b[-1]+inc/2)
    bin_b.append(column_data.max()+increment)
    column_data = pd.cut(column_data, bin_b, labels=labels).astype(float)

    return column_data, bin_b, labels


##########################################################################################

def gen_random_bin(column_data, num_bins, increment):
    bin_options = list(column_data.unique())
    list_vals = random.sample(bin_options, num_bins-1)
    list_vals = sorted(list_vals)
    list_vals.insert(0, column_data.min()-increment)
    list_vals.append(column_data.max()+increment)
    labels = []
    for x in range(len(list_vals)-1):
        labels.append((list_vals[x]+list_vals[x+1])/2)
    column_data = pd.cut(column_data, list_vals, labels=labels).astype(float)
    return column_data, list_vals, labels


##########################################################################################

def gen_kmeans_bin(column_data, num_bins, increment):
    est = KBinsDiscretizer( n_bins=num_bins, encode='ordinal', strategy='kmeans', subsample=None)
    xx = column_data.values.reshape(len(column_data),1)
    Xt = est.fit_transform(xx)
    bins = list(est.bin_edges_)[0]
    bins[0] = bins[0] - increment
    bins[-1] = bins[-1] + increment
    column_data = pd.cut(column_data, bins ,labels=bins[0:-1])
    return column_data, list(bins), list(bins[0:-1])


##########################################################################################

def gen_entropy_bin_stop(column_data, y_data, num_bins, increment):
    min_val = column_data.min()
    max_val = column_data.max()
    bins = list(md.cut_points(column_data.to_numpy(), y_data.to_numpy(), num_bins))     
    bins.insert(0, min_val-increment)
    bins.append(max_val+increment)
    column_data = pd.cut(column_data, bins=bins, labels=bins[0:-1])
    return column_data, bins


##########################################################################################

def train_test(data, columns, target, rs=42):
    cols = list(columns)
    try:
        cols.remove(target)
    except:
        print("Target already removed from columns")
    y_label = data[target]
    data2 = pd.DataFrame(data, columns=cols)
    train_x, test_x, train_label, test_label = train_test_split(data2, y_label, train_size=0.7, random_state=rs)
    return train_x, train_label, test_x, test_label


########################################################################################## 

def gen_derivative_bin(column_data, num_bins,count,decrement, increment, show_image=False):
    data_dict = column_data.value_counts().to_dict()

    lists = sorted(data_dict.items())
    x, y = zip(*lists) # unpack a list of pairs into two tuples
    X_Y_Spline = interpolate.PchipInterpolator(x, y)
    x = np.array(x)
    not_found = True


    X_ = np.linspace(x.min(), x.max(), int(len(x)*(1 - (count*decrement))))

    Y_ = X_Y_Spline(X_)

    dydx = np.gradient(Y_, X_)
    dydx2 = np.gradient(dydx, X_)
    inflc = []
    new_bins = []
    for d in range(0, len(dydx2)-1):
        if dydx2[d] < 0 < dydx2[d+1]:
            # print(X_[d],X_[d+1])
            inflc.append([d,d+1])
            new_bins.append((X_[d]+X_[d+1])/2)
    if len(new_bins) > num_bins-1:
        while len(new_bins) > num_bins-1:
            new_bins = sorted(new_bins)

            diff = list(np.diff(new_bins))
            idx = diff.index(min(diff))
            diff.remove(min(diff))
            new_bins.pop(idx)

        
    elif len(new_bins) < num_bins-1:
        temp_df = pd.DataFrame({"X": X_[1:], "dydx":dydx[1:], "dydx2":dydx2[1:]})
        values_keep = temp_df.sort_values(by="dydx2",ascending=False, key=abs).head(num_bins-len(new_bins))
        new_bins.extend(values_keep["X"].to_list())

    bins = new_bins
    if show_image:
        plt.plot(X_, Y_, label="smooth")
    # plt.plot(x, y, label="class1")
        plt.plot(X_, dydx, label="der1")
        plt.plot(X_, dydx2, label="der2")
        plt.legend( loc='upper center')
        for b in bins:
            # print((X_[b[0]]+X_[b[1]])/2)
            plt.axvline(x = b, linestyle="dotted")
        plt.show()
    bins = sorted(bins)
    bins_noedge = copy.deepcopy(bins)
    bins.insert(0, column_data.min()-increment)
    bins.append(column_data.max()+increment)
    # print(bins)
    column_data = pd.cut(column_data, bins, labels=bins[0:-1]).astype(float)
    return column_data, bins, bins_noedge

##########################################################################################

##########################################################################################
# DP Binning Algorithm
# Inputs:
# arr = list of values containing all possible bin boundaries (After reduction)
# data = Current Bin choice that is considered
# err = Current total error for the bin choice (data)
# start = current index (starts at 0)
# end =  total number of values (length of original arr)
# index = current number of bins
# r = number of bins (pre-determined), should be as number of bins-1 in this case
# dve = summarized values (from threshold functions )
# val_ct = contains mapping of bin boundaries and counts 
# final = output of function
# storage = contains dictionary of computed values 

#Ouput:
# final = contains the smallest error and best bin choice in the last row of the list (contains all possible best errors)

##########################################################################################
def DP_bin(arr, data, err, start, 
					end, index, r, dve, val_ct, final, storage):
	if (index == r):
		err[index] = dve[data[index-1]][1]
		if len(final) == 0 or final[-1][0] > sum(err):
			bin = copy.deepcopy(data)
			final.append([sum(err), bin])
		return final;

	i = start; 
	while(i <= end and end - i + 1 >= r - index):
		data[index] = arr[i]
		temp_val_ct = copy.deepcopy(val_ct)
		if index == 0:

			err[index] = dve[data[index]][0]
			err[index+1] = dve[data[index]][1]
		else:
			err[index] = 0
			err[index+1] = dve[data[index]][1]
			try:
				err[index] = storage[data[index-1]][data[index]]
			
			except:
				for vc in val_ct:
					if data[index-1] < vc < data[index]:
						sum1 = (abs(data[index] - vc) * val_ct[vc])
						sum2 = (abs(data[index-1] - vc) * val_ct[vc])
						if sum2 <= sum1:
							err[index] += sum2
						else:
							err[index] += sum1
						del temp_val_ct[vc] 
						
						if final == [] or err[index] < final[-1][0]:
							# print(data)
							storage[data[index-1]][data[index]] = err[index]    
		
		if index+1 == r:
			err[index+1] = dve[data[index]][1]
		# print(data, err)
		if len(final) == 0 or final[-1][0] > sum(err[:index]):
			# print(data)
			final = DP_bin(arr, data, err, i + 1, end, index + 1, r, dve, temp_val_ct, final,storage)
			# print(data)
		# else:
		# 	print("here")
		i += 1
	return final

#
def gen_DP_bin(data, final, col):
    dp_bin = copy.deepcopy(final[-1][1])
    dp_bin.append(data[col].max()+0.1)
    dp_bin.insert(0, data[col].min()-0.1)
    return dp_bin

# DP Binning algorithm Threshold/Value Reducing Functions

def reduce_large_values(column_data, threshold):
    val_ct_sorted = dict(sorted(column_data.value_counts().to_dict().items()))
    freq_list = list(val_ct_sorted.values())
    values_list = list(val_ct_sorted.keys())
    
    sum_freq = 0 
    new_freq_list = []
    new_values_list = []
    idx = 0
    for s in range(len(freq_list)):
        sum_freq += freq_list[s]
        if sum_freq > threshold/2:
            idx = s
        if sum_freq > threshold:
            new_freq_list.append(sum_freq)
            values_list
            new_values_list.append(values_list[idx])
            idx = s
            sum_freq = 0
    return new_values_list, new_freq_list

def reduce_val_ct(val_ct_all):   
    val_ct = val_ct_all
    val_ct, sum(val_ct.values())/ len(val_ct) 
    dict_reduce = {k:v for (k,v) in val_ct.items() if v > (sum(val_ct.values())/ len(val_ct))+1}
    if len(dict_reduce) == 0:
        # print("here")
        # reduce_large_values
        return None

    dict_reduce_keys = sorted(dict_reduce)

    diff = []
    for drk in range(len(dict_reduce_keys)-1):
        diff.append(dict_reduce_keys[drk+1]- dict_reduce_keys[drk])
        
    diff_val = sum(diff)/len(diff)
    new_dict = {}
    dr = 0 
    count = 1
    list_temp = []
    while dr < len(dict_reduce_keys)-1:
        list_temp.append(dict_reduce_keys[dr])
        if count == len(dict_reduce_keys)-1:
            if dict_reduce_keys[dr] + diff_val > dict_reduce_keys[count]:
                list_temp.append(dict_reduce_keys[count])
            sum_v = 0
            for l in list_temp:
                sum_v += dict_reduce[l]
            new_dict[list_temp[-1]] = sum_v
            list_temp = []
            break
        while dict_reduce_keys[dr] + diff_val > dict_reduce_keys[count]:
            list_temp.append(dict_reduce_keys[count])
            count += 1
            
        else:
            dr += len(list_temp)
            count = dr + 1
            sum_v = 0
            for l in list_temp:
                sum_v += dict_reduce[l]
            new_dict[list_temp[-1]] = sum_v
            list_temp = []
            continue
    return new_dict


# DP Algorithm Helper Functions (Creating summary dictionary for reducing repeating calculations)
def dict_val_error(val_ct_all):
    dict_val_error = {}
    storage = {}
    for vc in sorted(list(val_ct_all.keys())):
        error1 = 0
        error2 = 0
        for v in val_ct_all:
            if vc < v:
                error1 += (abs(v - vc) * val_ct_all[v])
            if vc > v:
                error2 += (abs(v-vc) * val_ct_all[v])
            dict_val_error[vc] = [error2,error1]
            storage[vc] = {}
    return dict_val_error, storage

def dict_val_error_reduce_mod(val_ct_all, rvc):
    dict_val_error = {}
    storage = {}
    for vc in sorted(list(rvc.keys())):
        error1 = 0
        error2 = 0
        for v in val_ct_all:
            if vc < v:
                error1 += (abs(v - vc) * val_ct_all[v])
            if vc > v:
                error2 += (abs(v-vc) * val_ct_all[v])
            dict_val_error[vc] = [error2,error1]
            storage[vc] = {}
    return dict_val_error, storage

def dict_val_error_reduce_mod2(val_ct_all, rvc):
    dict_val_error = {}
    storage = {}
    for vc in rvc:
        error1 = 0
        error2 = 0
        for v in val_ct_all:
            if vc < v:
                error1 += (abs(v - vc) * val_ct_all[v])
            if vc > v:
                error2 += (abs(v-vc) * val_ct_all[v])
            dict_val_error[vc] = [error2,error1]
            storage[vc] = {}
    return dict_val_error, storage



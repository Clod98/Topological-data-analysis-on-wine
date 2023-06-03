import gudhi as gd 
import pandas as pd
from sklearn import preprocessing
import numpy as np
from time import time 

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def assoc_fun_vert (v,S_labeled):
    if v in S_labeled.index:
        if S_labeled.loc[v,"quality"] > 5:
            return np.asarray([0,1])
        else:
            return np.asarray([1,0])
    else:
        return np.asarray([0,0])

def assoc_fun (mu,S_labeled):
    s = np.asarray([0,0])
    for v in mu:
        k = assoc_fun_vert(v,S_labeled)
        s = s+k
    return s

def extension (x,K,S_labeled):
    list_simpl = K.get_star([x])
    res = np.asarray([0,0])
    for i in list_simpl:
        filtr_lev = i[1]
        if filtr_lev != 0:
            simplex = i[0]
            phi = assoc_fun(simplex,S_labeled)
            res = res + phi/filtr_lev
    res_class = np.argmax(res)
    return res_class



df_origin = pd.read_csv(r"C:\Users\39345\Downloads\winequality-white.csv",sep=";")
"""
There are about 4900 wines in the dataset 
1080 of these have a rating strictly higher than 6 or stricty lower than 4
We want to classify them as good ones (above 6), or bad ones (below 4).
"""
fraction_dataset = 0.05
df_origin = df_origin.sample(frac = fraction_dataset)
df_origin.index = range(len(df_origin))
"""
We took about 10% of the dataset as testing set, the remaining part as training set
"""
test = df_origin
# test = df_origin[(df_origin["quality"]>5) | (df_origin["quality"]<5) ]
# test = df_origin[(df_origin["quality"]>6) | (df_origin["quality"]<4) ]
# test = df_origin[(df_origin["quality"]>7) | (df_origin["quality"]<3) ]
max_test = len(test)
frac_test = 0.1
ssample_size = min(max_test,int(len(df_origin)*(frac_test)))
T_labeled = df_origin[(df_origin["quality"]>6) | (df_origin["quality"]<4) ].sample(n = ssample_size)
S_labeled = df_origin.drop(T_labeled.index)

names = df_origin.columns
min_max_scaler = preprocessing.MinMaxScaler()
df = min_max_scaler.fit_transform(df_origin)
df = pd.DataFrame(df,columns=names,index=df_origin.index)
df = df.drop("quality",axis=1)

print("°°°°°°°°°°°°°°")

print(f"dimension of data set: {len(df)}")
print(f"dimension of test set: {len(T_labeled)}")
print(f"dimension of training set: {len(S_labeled)}")

print("°°°°°°°°°°°°°°")

rips = gd.RipsComplex(points=df.values.tolist())
st = rips.create_simplex_tree(max_dimension=1)
q = 6
c = q
st.collapse_edges(c)
st.expansion(q)

start = time()
st.compute_persistence()
elapsed = time() - start
print("°°°°°°°°°°°°°°")
print(f"It took {elapsed} seconds to compute the (co)homology groups")
print(f"Number of topological features of dim 0:  {len(st.persistence_intervals_in_dimension(0))}")
print(f"Number of topological features of dim 1:  {len(st.persistence_intervals_in_dimension(1))}")
print(f"Number of topological features of dim 2:  {len(st.persistence_intervals_in_dimension(2))}")
print(f"Number of topological features of dim 3:  {len(st.persistence_intervals_in_dimension(3))}")
print(f"Number of topological features of dim 4:  {len(st.persistence_intervals_in_dimension(4))}")
print(f"Number of topological features of dim 5:  {len(st.persistence_intervals_in_dimension(5))}")
print(f"Number of topological features of dim 6:  {len(st.persistence_intervals_in_dimension(6))}")
print(f"Number of topological features of dim 7:  {len(st.persistence_intervals_in_dimension(7))}")
# print(f"Number of topological features of dim 8:  {len(st.persistence_intervals_in_dimension(0))}")
# print(f"Number of topological features of dim 9:  {len(st.persistence_intervals_in_dimension(0))}")
print("°°°°°°°°°°°°°°")

print(f"dimension of the simplicial complex = {st.dimension()}")
print(f"number of simplices = {st.num_simplices()}")
print(f"number of vertices = {st.num_vertices()}")
print("°°°°°°°°°°°°°°")

l = []
i = q
while len(l)==0 and i>=0:
    l = st.persistence_intervals_in_dimension(i)
    i = i-1
print(f"Maximal topological dimension found: {i+1}")


lifes = l[:,1]-l[:,0]
print(f"The lifes lengths for the features of highest homological dimension are : {lifes}")

print("°°°°°°°°°°°°°°")

mean = np.mean(lifes)
indx = find_nearest(lifes,mean)
davg = l[indx,0]

dmax = l[np.argmax(lifes),0]

if np.shape(l)[0]>1:
    drand = l[np.random.randint(0,np.shape(l)[0]),0]
else:
    drand = l[0,0]

print(f"d_average is : {davg}")
print(f"d_maximal is : {dmax}")
print(f"d_random is : {drand}")

print("°°°°°°°°°°°°°°")

K_rand = gd.RipsComplex(points=df.values.tolist(),max_edge_length = drand)
K_rand = K_rand.create_simplex_tree()
K_avg = gd.RipsComplex(points=df.values.tolist(),max_edge_length = davg)
K_avg = K_avg.create_simplex_tree()
K_max = gd.RipsComplex(points=df.values.tolist(),max_edge_length = dmax)
K_max = K_max.create_simplex_tree()

true_class = T_labeled.to_dict("index")
predicted_class_avg = {}
predicted_class_max = {}
predicted_class_rnd = {}

errors_avg = 0
errors_max = 0
errors_rnd = 0

for i in T_labeled.index:
    predicted_class_avg[i] = extension(i,K_avg,S_labeled)
    predicted_class_max[i] = extension(i,K_max,S_labeled)
    predicted_class_rnd[i] = extension(i,K_rand,S_labeled)

    true_quality = T_labeled.loc[i,"quality"]
    quality = 0
    if true_quality > 5:
        quality = 1
    errors_avg = errors_avg + abs(quality-predicted_class_avg[i])
    errors_max = errors_max + abs(quality-predicted_class_max[i])
    errors_rnd = errors_rnd + abs(quality-predicted_class_rnd[i])


rel_error_avg = errors_avg/len(T_labeled)
rel_error_max = errors_max/len(T_labeled)
rel_error_rnd = errors_rnd/len(T_labeled)

print(f"The relative error for average case is : {rel_error_avg}")
print(f"The relative error for maximum case is : {rel_error_max}")
print(f"The relative error for random case is : {rel_error_rnd}")
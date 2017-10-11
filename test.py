import numpy as np
from algorithms.rpca import R_pca 
import matplotlib.pyplot as plt
import pandas as pd

N = 100
num_groups = 1
num_values_per_group = 2
p_missing = 0.2


infile = "./testdata/data.csv"
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

df = pd.read_csv(infile, parse_dates=['timestamp'], date_parser=dateparse)
y = df
y = y.fillna(y.bfill())

v = y["value"]
a=[]
for i in range(0,len(v)):
   a.append([i, v[i]])


# y.plot(figsize=(15, 6))
# plt.show()
print(a)

a = np.array(a)

# use R_pca to estimate the degraded data as L + S, where L is low rank, and S is sparse
rpca = R_pca(a)
L, S = rpca.fit(max_iter=1000, iter_print=100)

# visually inspect results (requires matplotlib)
rpca.plot_fit()
plt.show()
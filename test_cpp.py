import numpy as np
import sys
sys.path.insert(1,'./build')
import biest_call
import time

# print(biest_call.test(True, 1, 1))

# time1 = time.time()
# # biest_call.testb()
# time2 = time.time()
# print('1000 eval time:', time2 - time1)


a = 0.66 # np.random.random() * 0.2 + 0.8
b_list = [0.66, 0.77, 0.88] # np.random.random() * 0.2 + 0.8

# Generating a test surface and a few test functions
nfp = 1
Nt = 70
Np = 20
DIM = 3
gamma = np.zeros((Nt, Np, DIM))
func_in = np.zeros((Nt, Np, len(b_list)))
for k in range(len(b_list)):
    for i in np.arange(Nt):
        for j in np.arange(Np):
            b = b_list[k]
            phi = 2 * np.pi * i / Nt;
            theta = 2 * np.pi * j / Np;
            R = 1 + 0.25 * np.cos(theta);
            x = R * np.cos(phi);
            y = R * a * np.sin(phi);
            z = 0.25 * np.sin(theta);
            gamma[i, j, 0] = x;
            gamma[i, j, 1] = y;
            gamma[i, j, 2] = z;
            func_in[i, j, k] = x + y + b * z;
            
time1 = time.time()
results = np.zeros_like(func_in, dtype=np.float64)
biest_call.integrate_multi(
    gamma, # xt::pyarray<double> &gamma,
    func_in, # xt::pyarray<double> &func_in,
    results, # xt::pyarray<double> &result,
    True,
    10, # int digits,
    1, # int nfp
)
time2 = time.time()
print('Eval time:', time2 - time1)

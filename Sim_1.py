import numpy as np
import matplotlib.pyplot as plt
import time
from encoder import Encoder, Algorithm, SymbolSize

'''
class that simulates the experiment
n = log2(k) - 1, with k as the number of elements of the array
z = log2(q), with q as the q-ary basis of the symbol
elements as the number of elements of the simulation

output is a dict, with z, n indices
dict = {'prob_rs':prob_rs_3d, 'prob_sha1':prob_sha1_3d, 'prob_sha2':prob_sha2_3d, 'prob_nc':prob_nc_3d, 'time_rs':time_rs_3d, 'time_sha1':time_sha1_3d, 'time_sha2':time_sha2_3d, 'time_nc':time_nc_3d, 'tp_rs':tp_rs, 'tp_sha1':tp_sha1, 'tp_sha2':tp_sha2, 'tp_nc':tp_nc}
'''

class sim_FP:
    def __init__(self, n, z, elements):
        self.k = n
        self.z = z
        self.elements = elements

    def prob_various(self, z_max, k_max):
        if (z_max > 3):
            z_max = 0
        z = 0
        elements = 1
        rs = "rs"
        sha1 = "sha1"
        sha2 = "sha2"
        nc = "nc"

        prob_rs_3d = np.zeros([z_max, k_max])
        prob_sha1_3d = np.zeros([z_max, k_max])
        prob_sha2_3d = np.zeros([z_max, k_max])
        prob_nc_3d = np.zeros([z_max, k_max])

        time_rs_3d = np.zeros([z_max, k_max])
        time_sha1_3d = np.zeros([z_max, k_max])
        time_sha2_3d = np.zeros([z_max, k_max])
        time_nc_3d = np.zeros([z_max, k_max])

        while(z < z_max):

            if z==0:
                Symbol = SymbolSize.G2x8
            elif z==1:
                Symbol = SymbolSize.G2x16
            elif z==2:
                Symbol = SymbolSize.G2x32

            k = 0

            while(k < k_max):
                length = np.power(2, k + 1)
                encoder_rs = Encoder(length, Symbol, Algorithm.ReedSalomon)
                encoder_sha2 = Encoder(length, Symbol, Algorithm.Sha2)
                encoder_sha1 = Encoder(length, Symbol, Algorithm.Sha1)
                encoder_nc = Encoder(length, Symbol, Algorithm.NoCode)

                ts_rs = time.time()
                encoder_rs.generate()
                collions_rs = encoder_rs.encode(elements)
                prob_rs = collions_rs/elements
                te_rs = time.time() - ts_rs

                ts_sha1 = time.time()
                encoder_sha1.generate()
                collions_sha1 = encoder_sha1.encode(elements)
                prob_sha1 = collions_sha1/elements
                te_sha1 = time.time() - ts_sha1

                ts_sha2 = time.time()
                encoder_sha2.generate()
                collions_sha2 = encoder_sha2.encode(elements)
                prob_sha2 = collions_sha2/elements
                te_sha2 = time.time() - ts_sha2

            
                ts_nc = time.time()
                encoder_nc.generate()
                collions_nc = encoder_nc.encode(elements)
                prob_nc = collions_nc/elements
                te_nc = time.time() - ts_nc
            

                prob_rs_3d[z, k] = prob_rs
                prob_sha1_3d[z, k] = prob_sha1
                prob_sha2_3d[z, k] = prob_sha2
                prob_nc_3d[z, k] = prob_nc

                time_rs_3d[z, k] = te_rs
                time_sha1_3d[z, k] = te_sha1
                time_sha2_3d[z, k] = te_sha2
                time_nc_3d[z, k] = te_nc
                k += 1

            z += 1

        prob_dict = {'prob_rs':prob_rs_3d, 'prob_sha1':prob_sha1_3d, 'prob_sha2':prob_sha2_3d, 'prob_nc':prob_nc_3d, 'time_rs':time_rs_3d, 'time_sha1':time_sha1_3d, 'time_sha2':time_sha2_3d, 'time_nc':time_nc_3d}
        return prob_dict

    def add_matrix(self, a, b):
        return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

    def elmul_matrix(self, a, b):
        return [[(a[i][j]) * (b[i][j]) for j in range(len(a[0]))] for i in range(len(a))]        

    def min_nested(self, a):
        it = 3
        min_list = np.empty(it)
        i = 0
        while(i < it):
            np.append(min_list, self.non_zero_min(a[i][:]))
            i+=1
    
        return self.non_zero_min(min_list)

    def non_zero_min(self, a):
        arr = np.array(a)
        indices = np.nonzero(arr)[0]
        b = np.array([arr[i] for i in indices])
        if b.size==0:
            b = [0]
        return min(b)

    def sim_avg(self):
        k = self.k
        z = self.z
        num_sim = self.elements
        i = 0
        prob_rs_s = np.zeros([z, k])
        prob_sha1_s = np.zeros([z, k])
        prob_sha2_s = np.zeros([z, k])
        prob_nc_s = np.zeros([z, k])

        time_rs_s = np.zeros([z, k])
        time_sha1_s = np.zeros([z, k])
        time_sha2_s = np.zeros([z, k])
        time_nc_s = np.zeros([z, k])

        prob_rs_b = prob_rs_s
        prob_sha1_b = prob_sha1_s
        prob_sha2_b = prob_sha2_s
        prob_nc_b = prob_nc_s

        time_rs_b = time_rs_s
        time_sha1_b = time_sha1_s
        time_sha2_b = time_sha2_s
        time_nc_b = time_nc_s

        while(i<num_sim):
            prob = self.prob_various(z, k)

            prob_rs_s = prob_rs_b + prob['prob_rs']
            prob_sha1_s = prob_sha1_b + prob['prob_sha1']
            prob_sha2_s = prob_sha2_b + prob['prob_sha1']
            prob_nc_s = prob_nc_b + prob['prob_nc']

            time_rs_s = time_rs_b + prob['time_rs']
            time_sha1_s = time_sha1_b + prob['time_sha1']
            time_sha2_s = time_sha2_b + prob['time_sha1']
            time_nc_s = time_nc_b + prob['time_sha1']

            prob_rs_b = prob_rs_s
            prob_sha1_b = prob_sha1_s
            prob_sha2_b = prob_sha2_s
            prob_nc_b = prob_nc_s

            time_rs_b = time_rs_s
            time_sha1_b = time_sha1_s
            time_sha2_b = time_sha2_s
            time_nc_b = time_nc_s

            #print(i)
            i+=1

        prob_rs_3d = np.divide(prob_rs_s, num_sim)
        prob_sha1_3d = np.divide(prob_sha1_s, num_sim)
        prob_sha2_3d = np.divide(prob_sha2_s, num_sim)
        prob_nc_3d = np.divide(prob_nc_s, num_sim)
        time_rs_3d = np.divide(time_rs_s, num_sim)
        time_sha1_3d = np.divide(time_sha1_s, num_sim)
        time_sha2_3d = np.divide(time_sha2_s, num_sim)
        time_nc_3d = np.divide(time_nc_s, num_sim)

        tp_rs = self.elmul_matrix(prob_rs_3d, time_rs_3d)
        tp_sha1 = self.elmul_matrix(prob_sha1_3d, time_sha1_3d)
        tp_sha2 = self.elmul_matrix(prob_sha2_3d, time_sha2_3d)
        tp_nc = self.elmul_matrix(prob_nc_3d, time_nc_3d)

        prob_dict = {'prob_rs':prob_rs_3d, 'prob_sha1':prob_sha1_3d, 'prob_sha2':prob_sha2_3d, 'prob_nc':prob_nc_3d, 'time_rs':time_rs_3d, 'time_sha1':time_sha1_3d, 'time_sha2':time_sha2_3d, 'time_nc':time_nc_3d, 'tp_rs':tp_rs, 'tp_sha1':tp_sha1, 'tp_sha2':tp_sha2, 'tp_nc':tp_nc}
        return prob_dict

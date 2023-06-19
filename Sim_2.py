import numpy as np
import matplotlib.pyplot as plt
import time
from encoder import Encoder, Algorithm, SymbolSize
import pandas as pd


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

    def prob_various(self, z_max, k_max, interval):
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
        
        thrp_rs_3d = np.zeros([z_max, k_max])
        thrp_sha1_3d = np.zeros([z_max, k_max])
        thrp_sha2_3d = np.zeros([z_max, k_max])
        thrp_nc_3d = np.zeros([z_max, k_max])

        while(z < z_max):

            if z==0:
                Symbol = SymbolSize.G2x8
                z_n = 8
            elif z==1:
                Symbol = SymbolSize.G2x16
                z_n = 16
            elif z==2:
                Symbol = SymbolSize.G2x32
                z_n = 32

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
                te_rs = time.time() - ts_rs
                prob_rs = collions_rs/elements

                ts_sha1 = time.time()
                encoder_sha1.generate()
                collions_sha1 = encoder_sha1.encode(elements)
                te_sha1 = time.time() - ts_sha1
                prob_sha1 = collions_sha1/elements

                ts_sha2 = time.time()
                encoder_sha2.generate()
                collions_sha2 = encoder_sha2.encode(elements)
                te_sha2 = time.time() - ts_sha2
                prob_sha2 = collions_sha2/elements
            
                ts_nc = time.time()
                encoder_nc.generate()
                collions_nc = encoder_nc.encode(elements)
                te_nc = time.time() - ts_nc
                prob_nc = collions_nc/elements

                thrp_nc = self.thrp(length, z_n, te_nc)
                thrp_sha1 = self.thrp(length, z_n, te_sha1)
                thrp_sha2 = self.thrp(length, z_n, te_sha2)
                thrp_rs = self.thrp(length, z_n, te_rs)
            

                prob_rs_3d[z, k] = prob_rs
                prob_sha1_3d[z, k] = prob_sha1
                prob_sha2_3d[z, k] = prob_sha2
                prob_nc_3d[z, k] = prob_nc

                time_rs_3d[z, k] = te_rs
                time_sha1_3d[z, k] = te_sha1
                time_sha2_3d[z, k] = te_sha2
                time_nc_3d[z, k] = te_nc

                thrp_rs_3d[z, k] = thrp_rs
                thrp_sha1_3d[z, k] = thrp_sha1
                thrp_sha2_3d[z, k] = thrp_sha2
                thrp_nc_3d[z, k] = thrp_nc

                k += interval

            z += 1

        prob_dict = {'prob_rs':prob_rs_3d, 'prob_sha1':prob_sha1_3d, 'prob_sha2':prob_sha2_3d, 'prob_nc':prob_nc_3d, 'time_rs':time_rs_3d, 'time_sha1':time_sha1_3d, 'time_sha2':time_sha2_3d, 'time_nc':time_nc_3d, 'thrp_nc':thrp_nc_3d, 'thrp_rs':thrp_rs_3d, 'thrp_sha1':thrp_sha1_3d, 'thrp_sha2':thrp_sha2_3d}
        return prob_dict
    
    def prob_z(self, z_max, k, interval):
        if (z_max > 3):
            z_max = 0
        z = 0
        elements = 1
        rs = "rs"
        sha1 = "sha1"
        sha2 = "sha2"
        nc = "nc"

        prob_rs_3d = np.zeros([z_max])
        prob_sha1_3d = np.zeros([z_max])
        prob_sha2_3d = np.zeros([z_max])
        prob_nc_3d = np.zeros([z_max])

        time_rs_3d = np.zeros([z_max])
        time_sha1_3d = np.zeros([z_max])
        time_sha2_3d = np.zeros([z_max])
        time_nc_3d = np.zeros([z_max])
        
        thrp_rs_3d = np.zeros([z_max])
        thrp_sha1_3d = np.zeros([z_max])
        thrp_sha2_3d = np.zeros([z_max])
        thrp_nc_3d = np.zeros([z_max])

        while(z < z_max):

            if z==0:
                Symbol = SymbolSize.G2x8
                z_n = 8
            elif z==1:
                Symbol = SymbolSize.G2x16
                z_n = 16
            elif z==2:
                Symbol = SymbolSize.G2x32
                z_n = 32
            
            length = np.power(2, k + 1)
            encoder_rs = Encoder(length, Symbol, Algorithm.ReedSalomon)
            encoder_sha2 = Encoder(length, Symbol, Algorithm.Sha2)
            encoder_sha1 = Encoder(length, Symbol, Algorithm.Sha1)
            encoder_nc = Encoder(length, Symbol, Algorithm.NoCode)

            ts_rs = time.time()
            encoder_rs.generate()
            collions_rs = encoder_rs.encode(elements)
            te_rs = time.time() - ts_rs
            prob_rs = collions_rs/elements

            ts_sha1 = time.time()
            encoder_sha1.generate()
            collions_sha1 = encoder_sha1.encode(elements)
            te_sha1 = time.time() - ts_sha1
            prob_sha1 = collions_sha1/elements

            ts_sha2 = time.time()
            encoder_sha2.generate()
            collions_sha2 = encoder_sha2.encode(elements)
            te_sha2 = time.time() - ts_sha2
            prob_sha2 = collions_sha2/elements
            
            ts_nc = time.time()
            encoder_nc.generate()
            collions_nc = encoder_nc.encode(elements)
            te_nc = time.time() - ts_nc
            prob_nc = collions_nc/elements

            thrp_nc = self.thrp(length, z_n, te_nc)
            thrp_sha1 = self.thrp(length, z_n, te_sha1)
            thrp_sha2 = self.thrp(length, z_n, te_sha2)
            thrp_rs = self.thrp(length, z_n, te_rs)
            

            prob_rs_3d[z] = prob_rs
            prob_sha1_3d[z] = prob_sha1
            prob_sha2_3d[z] = prob_sha2
            prob_nc_3d[z] = prob_nc

            time_rs_3d[z] = te_rs
            time_sha1_3d[z] = te_sha1
            time_sha2_3d[z] = te_sha2
            time_nc_3d[z] = te_nc

            thrp_rs_3d[z] = thrp_rs
            thrp_sha1_3d[z] = thrp_sha1
            thrp_sha2_3d[z] = thrp_sha2
            thrp_nc_3d[z] = thrp_nc

            z += 1

        prob_dict = {'prob_rs':prob_rs_3d, 'prob_sha1':prob_sha1_3d, 'prob_sha2':prob_sha2_3d, 'prob_nc':prob_nc_3d, 'time_rs':time_rs_3d, 'time_sha1':time_sha1_3d, 'time_sha2':time_sha2_3d, 'time_nc':time_nc_3d, 'thrp_nc':thrp_nc_3d, 'thrp_rs':thrp_rs_3d, 'thrp_sha1':thrp_sha1_3d, 'thrp_sha2':thrp_sha2_3d}
        return prob_dict
    
    def prob_k(self, k_max, z, interval):
        k = 0
        elements = 1
        rs = "rs"
        sha1 = "sha1"
        sha2 = "sha2"
        nc = "nc"

        prob_rs_3d = np.zeros([k_max])
        prob_sha1_3d = np.zeros([k_max])
        prob_sha2_3d = np.zeros([k_max])
        prob_nc_3d = np.zeros([k_max])

        time_rs_3d = np.zeros([k_max])
        time_sha1_3d = np.zeros([k_max])
        time_sha2_3d = np.zeros([k_max])
        time_nc_3d = np.zeros([k_max])
        
        thrp_rs_3d = np.zeros([k_max])
        thrp_sha1_3d = np.zeros([k_max])
        thrp_sha2_3d = np.zeros([k_max])
        thrp_nc_3d = np.zeros([k_max])

        while(k < k_max):

            if z==0:
                Symbol = SymbolSize.G2x8
                z_n = 8
            elif z==1:
                Symbol = SymbolSize.G2x16
                z_n = 16
            elif z==2:
                Symbol = SymbolSize.G2x32
                z_n = 32
            
            length = np.power(2, k + 1)
            encoder_rs = Encoder(length, Symbol, Algorithm.ReedSalomon)
            encoder_sha2 = Encoder(length, Symbol, Algorithm.Sha2)
            encoder_sha1 = Encoder(length, Symbol, Algorithm.Sha1)
            encoder_nc = Encoder(length, Symbol, Algorithm.NoCode)

            ts_rs = time.time()
            encoder_rs.generate()
            collions_rs = encoder_rs.encode(elements)
            te_rs = time.time() - ts_rs
            prob_rs = collions_rs/elements

            ts_sha1 = time.time()
            encoder_sha1.generate()
            collions_sha1 = encoder_sha1.encode(elements)
            te_sha1 = time.time() - ts_sha1
            prob_sha1 = collions_sha1/elements

            ts_sha2 = time.time()
            encoder_sha2.generate()
            collions_sha2 = encoder_sha2.encode(elements)
            te_sha2 = time.time() - ts_sha2
            prob_sha2 = collions_sha2/elements
            
            ts_nc = time.time()
            encoder_nc.generate()
            collions_nc = encoder_nc.encode(elements)
            te_nc = time.time() - ts_nc
            prob_nc = collions_nc/elements

            thrp_nc = self.thrp(length, z_n, te_nc)
            thrp_sha1 = self.thrp(length, z_n, te_sha1)
            thrp_sha2 = self.thrp(length, z_n, te_sha2)
            thrp_rs = self.thrp(length, z_n, te_rs)
            

            prob_rs_3d[k] = prob_rs
            prob_sha1_3d[k] = prob_sha1
            prob_sha2_3d[k] = prob_sha2
            prob_nc_3d[k] = prob_nc

            time_rs_3d[k] = te_rs
            time_sha1_3d[k] = te_sha1
            time_sha2_3d[k] = te_sha2
            time_nc_3d[k] = te_nc

            thrp_rs_3d[k] = thrp_rs
            thrp_sha1_3d[k] = thrp_sha1
            thrp_sha2_3d[k] = thrp_sha2
            thrp_nc_3d[k] = thrp_nc

            k += interval

        prob_dict = {'prob_rs':prob_rs_3d, 'prob_sha1':prob_sha1_3d, 'prob_sha2':prob_sha2_3d, 'prob_nc':prob_nc_3d, 'time_rs':time_rs_3d, 'time_sha1':time_sha1_3d, 'time_sha2':time_sha2_3d, 'time_nc':time_nc_3d, 'thrp_nc':thrp_nc_3d, 'thrp_rs':thrp_rs_3d, 'thrp_sha1':thrp_sha1_3d, 'thrp_sha2':thrp_sha2_3d}
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

    def sim_avg(self, interval = 1):
        k = self.k
        z = self.z
        num_sim = self.elements
        i = 0

        prob_rs = np.zeros([z, k, num_sim])
        prob_sha1 = np.zeros([z, k, num_sim])
        prob_sha2 = np.zeros([z, k, num_sim])
        prob_nc = np.zeros([z, k, num_sim])

        time_rs = np.zeros([z, k, num_sim])
        time_sha1 = np.zeros([z, k, num_sim])
        time_sha2 = np.zeros([z, k, num_sim])
        time_nc = np.zeros([z, k, num_sim])

        thrp_rs = np.zeros([z, k, num_sim])
        thrp_sha1 = np.zeros([z, k, num_sim])
        thrp_sha2 = np.zeros([z, k, num_sim])
        thrp_nc = np.zeros([z, k, num_sim])


        while(i<num_sim):
            prob = self.prob_various(z, k, interval)

            prob_rs[:, :, i] = prob['prob_rs']
            prob_sha1[:, :, i] = prob['prob_sha1']
            prob_sha2[:, :, i] = prob['prob_sha1']
            prob_nc[:, :, i] = prob['prob_nc']

            time_rs[:, :, i] = prob['time_rs']
            time_sha1[:, :, i] = prob['time_sha1']
            time_sha2[:, :, i] = prob['time_sha1']
            time_nc[:, :, i] = prob['time_sha1']

            thrp_rs[:, :, i] = prob['thrp_rs']
            thrp_sha1[:, :, i] = prob['thrp_sha1']
            thrp_sha2[:, :, i] = prob['thrp_sha1']
            thrp_nc[:, :, i] = prob['thrp_sha1']

            print(i)
            i+=1

        prob_rs_3d = np.average(prob_rs, axis = 2)
        prob_sha1_3d = np.average(prob_sha1, axis = 2)
        prob_sha2_3d = np.average(prob_sha2, axis = 2)
        prob_nc_3d = np.average(prob_nc, axis = 2)

        time_rs_3d = np.average(time_rs, axis = 2)
        time_sha1_3d = np.average(time_sha1, axis = 2)
        time_sha2_3d = np.average(time_sha2, axis = 2)
        time_nc_3d = np.average(time_nc, axis = 2)

        thrp_rs_3d = np.average(thrp_rs, axis = 2)
        thrp_sha1_3d = np.average(thrp_sha1, axis = 2)
        thrp_sha2_3d = np.average(thrp_sha2, axis = 2)
        thrp_nc_3d = np.average(thrp_nc, axis = 2)

        prob_rs_var = np.var(prob_rs, axis = 2)
        prob_sha1_var = np.var(prob_sha1, axis = 2)
        prob_sha2_var = np.var(prob_sha2, axis = 2)
        prob_nc_var = np.var(prob_nc, axis = 2)

        time_rs_var = np.var(time_rs, axis = 2)
        time_sha1_var = np.var(time_sha1, axis = 2)
        time_sha2_var = np.var(time_sha2, axis = 2)
        time_nc_var = np.var(time_nc, axis = 2)

        thrp_rs_var = np.var(thrp_rs, axis = 2)
        thrp_sha1_var = np.var(thrp_sha1, axis = 2)
        thrp_sha2_var = np.var(thrp_sha2, axis = 2)
        thrp_nc_var = np.var(thrp_nc, axis = 2)


        tp_rs = np.array(self.elmul_matrix(prob_rs_3d, time_rs_3d))
        tp_sha1 = np.array(self.elmul_matrix(prob_sha1_3d, time_sha1_3d))
        tp_sha2 = np.array(self.elmul_matrix(prob_sha2_3d, time_sha2_3d))
        tp_nc = np.array(self.elmul_matrix(prob_nc_3d, time_nc_3d))

        prob_dict = {'prob_rs_avg':prob_rs_3d, 'prob_sha1_avg':prob_sha1_3d, 'prob_sha2_avg':prob_sha2_3d, 'prob_nc_avg':prob_nc_3d, 'time_rs_avg':time_rs_3d, 'time_sha1_avg':time_sha1_3d, 'time_sha2_avg':time_sha2_3d, 'time_nc_avg':time_nc_3d, 'tp_rs':tp_rs, 'tp_sha1':tp_sha1, 'tp_sha2':tp_sha2, 'tp_nc':tp_nc, 'thrp_nc_avg':thrp_nc_3d, 'thrp_rs_avg':thrp_rs_3d, 'thrp_sha1_avg':thrp_sha1_3d, 'thrp_sha2_avg':thrp_sha2_3d, 'prob_rs':prob_rs, 'prob_sha1':prob_sha1, 'prob_sha2':prob_sha2, 'prob_nc':prob_nc, 'time_rs':time_rs, 'time_sha1':time_sha1, 'time_sha2':time_sha2, 'time_nc':time_nc, 'thrp_nc':thrp_nc, 'thrp_rs':thrp_rs, 'thrp_sha1':thrp_sha1, 'thrp_sha2':thrp_sha2, 'prob_rs_var':prob_rs_var, 'prob_sha1_var':prob_sha1_var, 'prob_sha2_var':prob_sha2_var, 'prob_nc_var':prob_nc_var, 'time_rs_var':time_rs_var, 'time_sha1_var':time_sha1_var, 'time_sha2_var':time_sha2_var, 'time_nc_var':time_nc_var, 'thrp_nc_var':thrp_nc_var, 'thrp_rs_var':thrp_rs_var, 'thrp_sha1_var':thrp_sha1_var, 'thrp_sha2_var':thrp_sha2_var}
        return prob_dict
    
    def sim_z(self, k, interval = 1):
        z = self.z
        num_sim = self.elements
        i = 0

        prob_rs = np.zeros([z, num_sim])
        prob_sha1 = np.zeros([z, num_sim])
        prob_sha2 = np.zeros([z, num_sim])
        prob_nc = np.zeros([z, num_sim])

        time_rs = np.zeros([z, num_sim])
        time_sha1 = np.zeros([z, num_sim])
        time_sha2 = np.zeros([z, num_sim])
        time_nc = np.zeros([z, num_sim])

        thrp_rs = np.zeros([z, num_sim])
        thrp_sha1 = np.zeros([z, num_sim])
        thrp_sha2 = np.zeros([z, num_sim])
        thrp_nc = np.zeros([z, num_sim])


        while(i<num_sim):
            prob = self.prob_z(z, k, interval)

            prob_rs[:, i] = prob['prob_rs']
            prob_sha1[:, i] = prob['prob_sha1']
            prob_sha2[:, i] = prob['prob_sha1']
            prob_nc[:, i] = prob['prob_nc']

            time_rs[:, i] = prob['time_rs']
            time_sha1[:, i] = prob['time_sha1']
            time_sha2[:, i] = prob['time_sha1']
            time_nc[:, i] = prob['time_sha1']

            thrp_rs[:, i] = prob['thrp_rs']
            thrp_sha1[:, i] = prob['thrp_sha1']
            thrp_sha2[:, i] = prob['thrp_sha1']
            thrp_nc[:, i] = prob['thrp_sha1']

            print(i)
            i+=1

        prob_rs_3d = np.average(prob_rs, axis = 1)
        prob_sha1_3d = np.average(prob_sha1, axis = 1)
        prob_sha2_3d = np.average(prob_sha2, axis = 1)
        prob_nc_3d = np.average(prob_nc, axis = 1)

        time_rs_3d = np.average(time_rs, axis = 1)
        time_sha1_3d = np.average(time_sha1, axis = 1)
        time_sha2_3d = np.average(time_sha2, axis = 1)
        time_nc_3d = np.average(time_nc, axis = 1)

        thrp_rs_3d = np.average(thrp_rs, axis = 1)
        thrp_sha1_3d = np.average(thrp_sha1, axis = 1)
        thrp_sha2_3d = np.average(thrp_sha2, axis = 1)
        thrp_nc_3d = np.average(thrp_nc, axis = 1)

        prob_rs_var = np.var(prob_rs, axis = 1)
        prob_sha1_var = np.var(prob_sha1, axis = 1)
        prob_sha2_var = np.var(prob_sha2, axis = 1)
        prob_nc_var = np.var(prob_nc, axis = 1)

        time_rs_var = np.var(time_rs, axis = 1)
        time_sha1_var = np.var(time_sha1, axis = 1)
        time_sha2_var = np.var(time_sha2, axis = 1)
        time_nc_var = np.var(time_nc, axis = 1)

        thrp_rs_var = np.var(thrp_rs, axis = 1)
        thrp_sha1_var = np.var(thrp_sha1, axis = 1)
        thrp_sha2_var = np.var(thrp_sha2, axis = 1)
        thrp_nc_var = np.var(thrp_nc, axis = 1)


        tp_rs = np.zeros([z, num_sim])
        tp_sha1 = np.zeros([z, num_sim])
        tp_sha2 = np.zeros([z, num_sim])
        tp_nc = np.zeros([z, num_sim])

        prob_dict = {'prob_rs_avg':prob_rs_3d, 'prob_sha1_avg':prob_sha1_3d, 'prob_sha2_avg':prob_sha2_3d, 'prob_nc_avg':prob_nc_3d, 'time_rs_avg':time_rs_3d, 'time_sha1_avg':time_sha1_3d, 'time_sha2_avg':time_sha2_3d, 'time_nc_avg':time_nc_3d, 'tp_rs':tp_rs, 'tp_sha1':tp_sha1, 'tp_sha2':tp_sha2, 'tp_nc':tp_nc, 'thrp_nc_avg':thrp_nc_3d, 'thrp_rs_avg':thrp_rs_3d, 'thrp_sha1_avg':thrp_sha1_3d, 'thrp_sha2_avg':thrp_sha2_3d, 'prob_rs':prob_rs, 'prob_sha1':prob_sha1, 'prob_sha2':prob_sha2, 'prob_nc':prob_nc, 'time_rs':time_rs, 'time_sha1':time_sha1, 'time_sha2':time_sha2, 'time_nc':time_nc, 'thrp_nc':thrp_nc, 'thrp_rs':thrp_rs, 'thrp_sha1':thrp_sha1, 'thrp_sha2':thrp_sha2, 'prob_rs_var':prob_rs_var, 'prob_sha1_var':prob_sha1_var, 'prob_sha2_var':prob_sha2_var, 'prob_nc_var':prob_nc_var, 'time_rs_var':time_rs_var, 'time_sha1_var':time_sha1_var, 'time_sha2_var':time_sha2_var, 'time_nc_var':time_nc_var, 'thrp_nc_var':thrp_nc_var, 'thrp_rs_var':thrp_rs_var, 'thrp_sha1_var':thrp_sha1_var, 'thrp_sha2_var':thrp_sha2_var}
        return prob_dict
    
    def sim_k(self, z, interval = 1):
        k = self.k

        num_sim = self.elements
        i = 0

        prob_rs = np.zeros([k, num_sim])
        prob_sha1 = np.zeros([k, num_sim])
        prob_sha2 = np.zeros([k, num_sim])
        prob_nc = np.zeros([k, num_sim])

        time_rs = np.zeros([k, num_sim])
        time_sha1 = np.zeros([k, num_sim])
        time_sha2 = np.zeros([k, num_sim])
        time_nc = np.zeros([k, num_sim])

        thrp_rs = np.zeros([k, num_sim])
        thrp_sha1 = np.zeros([k, num_sim])
        thrp_sha2 = np.zeros([k, num_sim])
        thrp_nc = np.zeros([k, num_sim])


        while(i<num_sim):
            prob = self.prob_k(k, z, interval)

            prob_rs[:, i] = prob['prob_rs']
            prob_sha1[:, i] = prob['prob_sha1']
            prob_sha2[:, i] = prob['prob_sha1']
            prob_nc[:, i] = prob['prob_nc']

            time_rs[:, i] = prob['time_rs']
            time_sha1[:, i] = prob['time_sha1']
            time_sha2[:, i] = prob['time_sha1']
            time_nc[:, i] = prob['time_sha1']

            thrp_rs[:, i] = prob['thrp_rs']
            thrp_sha1[:, i] = prob['thrp_sha1']
            thrp_sha2[:, i] = prob['thrp_sha1']
            thrp_nc[:, i] = prob['thrp_sha1']

            print(i)
            i+=1

        prob_rs_3d = np.average(prob_rs, axis = 1)
        prob_sha1_3d = np.average(prob_sha1, axis = 1)
        prob_sha2_3d = np.average(prob_sha2, axis = 1)
        prob_nc_3d = np.average(prob_nc, axis = 1)

        time_rs_3d = np.average(time_rs, axis = 1)
        time_sha1_3d = np.average(time_sha1, axis = 1)
        time_sha2_3d = np.average(time_sha2, axis = 1)
        time_nc_3d = np.average(time_nc, axis = 1)

        thrp_rs_3d = np.average(thrp_rs, axis = 1)
        thrp_sha1_3d = np.average(thrp_sha1, axis = 1)
        thrp_sha2_3d = np.average(thrp_sha2, axis = 1)
        thrp_nc_3d = np.average(thrp_nc, axis = 1)

        prob_rs_var = np.var(prob_rs, axis = 1)
        prob_sha1_var = np.var(prob_sha1, axis = 1)
        prob_sha2_var = np.var(prob_sha2, axis = 1)
        prob_nc_var = np.var(prob_nc, axis = 1)

        time_rs_var = np.var(time_rs, axis = 1)
        time_sha1_var = np.var(time_sha1, axis = 1)
        time_sha2_var = np.var(time_sha2, axis = 1)
        time_nc_var = np.var(time_nc, axis = 1)

        thrp_rs_var = np.var(thrp_rs, axis = 1)
        thrp_sha1_var = np.var(thrp_sha1, axis = 1)
        thrp_sha2_var = np.var(thrp_sha2, axis = 1)
        thrp_nc_var = np.var(thrp_nc, axis = 1)


        tp_rs = np.zeros([k, num_sim])
        tp_sha1 = np.zeros([k, num_sim])
        tp_sha2 = np.zeros([k, num_sim])
        tp_nc = np.zeros([k, num_sim])

        prob_dict = {'prob_rs_avg':prob_rs_3d, 'prob_sha1_avg':prob_sha1_3d, 'prob_sha2_avg':prob_sha2_3d, 'prob_nc_avg':prob_nc_3d, 'time_rs_avg':time_rs_3d, 'time_sha1_avg':time_sha1_3d, 'time_sha2_avg':time_sha2_3d, 'time_nc_avg':time_nc_3d, 'tp_rs':tp_rs, 'tp_sha1':tp_sha1, 'tp_sha2':tp_sha2, 'tp_nc':tp_nc, 'thrp_nc_avg':thrp_nc_3d, 'thrp_rs_avg':thrp_rs_3d, 'thrp_sha1_avg':thrp_sha1_3d, 'thrp_sha2_avg':thrp_sha2_3d, 'prob_rs':prob_rs, 'prob_sha1':prob_sha1, 'prob_sha2':prob_sha2, 'prob_nc':prob_nc, 'time_rs':time_rs, 'time_sha1':time_sha1, 'time_sha2':time_sha2, 'time_nc':time_nc, 'thrp_nc':thrp_nc, 'thrp_rs':thrp_rs, 'thrp_sha1':thrp_sha1, 'thrp_sha2':thrp_sha2, 'prob_rs_var':prob_rs_var, 'prob_sha1_var':prob_sha1_var, 'prob_sha2_var':prob_sha2_var, 'prob_nc_var':prob_nc_var, 'time_rs_var':time_rs_var, 'time_sha1_var':time_sha1_var, 'time_sha2_var':time_sha2_var, 'time_nc_var':time_nc_var, 'thrp_nc_var':thrp_nc_var, 'thrp_rs_var':thrp_rs_var, 'thrp_sha1_var':thrp_sha1_var, 'thrp_sha2_var':thrp_sha2_var}
        return prob_dict
    

    def save_to_csv(self, d, filename ="simluation_data.csv" ):
        #df = pd.DataFrame(d)
        df = pd.concat([pd.DataFrame(v) for k, v in d.items()], axis = 1, keys = list(d.keys()))
        df.to_csv(filename, index=False)

    def thrp(self, k, z, t):
        cons = np.power(1024, 3)
        thrp = k*z/(8*t*cons)
        return thrp
    

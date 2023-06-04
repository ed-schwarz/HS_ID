import hashlib
import numpy as np
import math
import random
from reedsolo import RSCodec, ReedSolomonError
import collections
import matplotlib.pyplot as plt
import time
from collections import Iterable


class dummy_encoder:
    def __init__(self, k, z, alg, elements):
        self.k = k
        self.z = z
        self.alg = alg
        self.elements = elements

#generates a random number between 0 and range
    def generate_random_ID(self, ran):
        return random.randrange(0, ran)

#encodes to hash, withoutput as hex
    def encode_hash(self, id):
        m = hashlib.sha256()
        m.update(repr(id).encode())
        sha = m.hexdigest()
        return sha

#encodes to reed solomon, withoutput as hex
    def encode_rs(self, id, size):
        rsc = RSCodec(size)
        rs = rsc.encode(id)
        out = ''.join(format(x, '02x') for x in rs)
        return out

#convert to binary from int
    def convert_to_bin(self, id, size):
        bin_list = list(bin(id)[2:].zfill(size))
        bin_list.reverse()
        return bin_list[0:size-1]

#convert to q-ary from binary
    def convert_to_symbol(self, binv, exp):
        q = int(math.log(exp, 2))
        if q == 1:
            return [ int(x) for x in binv ]
        vec = binv
        if(len(binv)%q != 0):
            extra_l = q - len(binv)%q
            extra_a = list(''.zfill(extra_l))
            vec = binv + extra_a  
    
        q_size = int(len(binv)/q)
        q_list = []
        q_index = 0
        while(q_index < q_size):
            in_vec = vec[q_index*q:q_index*q+(q-1)]
            in_vec.reverse()
            sym = int(''.join(in_vec), 2)
            q_list.append(sym)
            q_index += 1
        return q_list

#convert to q-ary from hex
    def convert_from_hex(self, id, size, q):
        num_int = int(id, 16)
        num_bin = self.convert_to_bin(num_int, size)
        num_sym = self.convert_to_symbol(num_bin, q)
        return num_sym

#encode to hash and return q-ary
    def encode_and_convert_hash(self, id, size, q):
        sha = self.encode_hash(id)
        out = self.convert_from_hex(sha, size, q)
        return out

#encode to reed solomon and return q-ary
    def encode_and_convert_rs(self, id, size, q):
        rs_size = int(size/2)
        rs = self.encode_rs(id, rs_size)
        out = self.convert_from_hex(rs, size, q)
        return out

#generate random id and convert to q-ary
    def generate_encoded_id(self, ran, size, q):
        #id = self.generate_random_ID(ran)
        #id_bin = self.convert_to_bin(id, size)
        #id_vec = self.convert_to_symbol(id_bin, q)
        r = random.randint(0, q)
        id_vec = []
        id_vec.append(r)
        return id_vec

#compare two vectors in specific position
    def compare_vector(self, id1, id2, index):
        if(id1[index] == id2[index]):
            return True
        else:
            return False
    
#compare two ids and return true if false positive
    def detect_fp(self, id1, id2):
        ran = len(id1)
        ran2 = len(id2)
        if(ran!=ran2):
            print("Error")
        index = random.randrange(0, ran)
        ran_ind = self.compare_vector(id1, id2, index)
        if(collections.Counter(id1) != collections.Counter(id2)):
            if(ran_ind):
                return True
            else:
                return False
        else:
            return False

#create a simulation, and return a dictionary of arrays with the values
#q is the q-ary basis of simbol
#elements is the number of iterations of the simulation
#size is the size of the binary numbers
#ran is the maximal random number for id
    def simulate(self):
        q = math.pow(2, self.z) #2^z
        size = self.k
        ran = pow(2, size) #2^size
        elements = self.elements
        i = 0
        id1_array = []
        id2_array = []
        en1_array = []
        en2_array = []
        fp_array = []
        enc1_time_array = []
        enc2_time_array = []

        if self.alg == "rs":
            while(i < elements):
                id1 = self.generate_encoded_id(ran, size, q)
                id2 = self.generate_encoded_id(ran, size, q)

                t_start = time.perf_counter_ns()
                rs1 = self.encode_and_convert_rs(id1, size, q)
                t_end = time.perf_counter_ns()
                rs1_t = (t_end - t_start)
                print(rs1_t)
                t_start2 = time.perf_counter_ns()
                rs2 = self.encode_and_convert_rs(id2, size, q)
                t_end2 = time.perf_counter_ns()
                rs2_t = t_end2 - t_start2
                print(rs2_t)

                fp = self.detect_fp(rs1, rs2)

                id1_array.append(id1)
                id2_array.append(id2)

                en1_array.append(rs1)
                en2_array.append(rs2)

                fp_array.append(fp)

                enc1_time_array.append(rs1_t)
                enc2_time_array.append(rs2_t)

                i += 1
                #print(i)

        elif self.alg == "sha":
            while(i < elements):
                id1 = self.generate_encoded_id(ran, size, q)
                id2 = self.generate_encoded_id(ran, size, q)

                t_start = time.perf_counter_ns()
                sha1 = self.encode_and_convert_hash(id1, size, q)
                t_end = time.perf_counter_ns()
                sha1_t = t_end - t_start
                t_start2 = time.perf_counter_ns()
                sha2 = self.encode_and_convert_hash(id2, size, q)
                t_end2 = time.perf_counter_ns()
                sha2_t = t_end2 - t_start2

                fp = self.detect_fp(sha1, sha2)

                id1_array.append(id1)
                id2_array.append(id2)

                en1_array.append(sha1)
                en2_array.append(sha2)

                fp_array.append(fp)

                enc1_time_array.append(sha1_t)
                enc2_time_array.append(sha2_t)

                i += 1
                #print(i)

        elif self.alg == "nc":
            while(i < elements):
                t_start = time.perf_counter_ns()
                id1 = self.generate_encoded_id(ran, size, q)
                t_end = time.perf_counter_ns()
                nc1_t = t_end - t_start
                t_start2 = time.perf_counter_ns()
                id2 = self.generate_encoded_id(ran, size, q)
                t_end2 = time.perf_counter_ns()
                nc2_t = t_end2 - t_start2


                fp = self.detect_fp(id1, id2)

                id1_array.append(id1)
                id2_array.append(id2)

                fp_array.append(fp)

                enc1_time_array.append(nc1_t)
                enc2_time_array.append(nc2_t)

                i += 1
                #print(i) #nur damit Fortschritt zu sehen ist

        sim_arays = {'id1':id1_array, 'id2':id2_array, 'en1':en1_array, 'en2':en2_array, \
                     'fp':fp_array, 'en1_time':enc1_time_array, 'en2_time':enc2_time_array}
        return sim_arays
    
    def calculate_prob(self):
        sim = self.simulate()
        fp = sim['fp'] #FP aus Dict in Array
        enc_time = sim['en1_time']
        count = 0 #zaehlt Anzahl FP in fp-Array

        for i in range(0, self.elements):
            if fp[i]: #wenn fp[i] == true
                count +=1

        prob = count/self.elements #FP/(FP+TN)

        return prob, enc_time


#output a dict containing 2x2 probability matrixes, with (z, k) dimensions
def prob_various(z_max, k_max):
    z = 1
    elements = 10
    rs = "rs"
    sha = "sha"
    nc = "nc"

    prob_rs_3d = np.empty([z_max, k_max])
    prob_sha_3d = np.empty([z_max, k_max])
    prob_nc_3d = np.empty([z_max, k_max])
    # rs_time = np.empty([z_max, k_max])
    rs_time = []
    sha_time = []
    nc_time = []

    while(z < z_max):

        k = 10  #??? nicht k=1 ???

        while(k < k_max):
            enc_rs = dummy_encoder(k, z, rs, elements)
            enc_sha = dummy_encoder(k, z, sha, elements)
            enc_nc = dummy_encoder(k, z, nc, elements)

            prob_rs_temp = enc_rs.calculate_prob()
            prob_rs = prob_rs_temp[0]
            prob_sha_temp = enc_sha.calculate_prob()
            prob_sha = prob_sha_temp[0]
            prob_nc_temp = enc_nc.calculate_prob()
            prob_nc = prob_nc_temp[0]

            #print(prob_rs_temp[0])
            #print(prob_rs_temp[1])

            prob_rs_3d[z, k] = prob_rs
            prob_sha_3d[z, k] = prob_sha
            prob_nc_3d[z, k] = prob_nc

            rs_time.append(prob_rs_temp[1])
            sha_time.append(prob_sha_temp[1])
            nc_time.append(prob_nc_temp[1])

            k += 1

        #rs_time.append(rs_time)
        #sha_time.append(sha_time)
        #nc_time.append(nc_time)
        z += 1

    prob_dict = {'rs': prob_rs_3d, 'sha': prob_sha_3d, 'nc': prob_nc_3d, \
                 'rs_t': rs_time, 'sha_t': sha_time, 'nc_t': nc_time}
    return prob_dict


def flatten(arr): #https://stackoverflow.com/questions/17485747/how-to-convert-a-nested-list-into-a-one-dimensional-list-in-python
    for item in arr:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for f in flatten(item):
                yield f
        else:
            yield item
    return arr


def plot_multiple(rs_arr, sha_arr, nc_arr):
    title = ''
    title_ax5 = ''
    y_label = ''
    save = ''
    if np.array_equal(rs_arr, prob['rs']) and np.array_equal(sha_arr, prob['sha']) and np.array_equal(nc_arr, prob['nc']):
        title = 'FP-Probability of Reed Solomon, Sha and NoCode'
        title_ax5 = 'FP-Probability of RS, SHA and NC'
        y_label = 'Probability'
        save = "PropPlot_All.png"
    elif np.array_equal(rs_arr, time_rs) and np.array_equal(sha_arr, time_sha) and np.array_equal(nc_arr, time_nc):
        title = 'Encryption time of RS, SHA and NC'
        title_ax5 = 'Encryption time of RS, SHA and NC'
        y_label = 'Time in ns'
        save = 'EncTime_All.png'

    rs = list(flatten(rs_arr))
    sha = list(flatten(sha_arr))
    nc = list(flatten(nc_arr))

    data = [rs, sha, nc]
    names = ['Reed Solomon', 'Sha', 'NC']
    color = ['g', 'm', 'b']
    x_label = 'Number of Encryptions'
    fontdict = {'fontsize': 10}  # font size of titles


    #rs_t_sort = np.sort(t_rs_fl) #bring values in order
    #sha_t_sort = np.sort(time_sha) #bring values in order
    #nc_t_sort = np.sort(time_nc) #bring values in order
    
    length = len(rs)
    x = np.linspace(0, length, length, endpoint=True) #x-Achse der Plots

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(8.27, 11.69), dpi=300)
    fig.suptitle(title, fontsize=16)

    ax1.plot(x, rs, color='g', ls=':', marker='o', markersize=1)
    ax1.set_title('Reed Solomon', fontdict=fontdict)
    ax1.set_ylabel(y_label, fontsize=8)
    ax1.set_xlabel(x_label, fontsize=8)

    ax2.plot(x, sha, color='m', ls=':', marker='o', markersize=1)
    ax2.set_title('Sha', fontdict=fontdict)
    ax2.set_ylabel(y_label, fontsize=8)
    ax2.set_xlabel(x_label, fontsize=8)

    ax3.plot(x, nc, color='b', ls=':', marker='o', markersize=1)
    ax3.set_title('No Code', fontdict=fontdict)
    ax3.set_ylabel(y_label, fontsize=8)
    ax3.set_xlabel(x_label, fontsize=8)

    # boxplot
    ax4.boxplot(data, patch_artist=True,
                showmeans=True, showfliers=True,
                medianprops={"color": "white", "linewidth": 0.5},
                boxprops={"facecolor": "C0", "edgecolor": "white",
                          "linewidth": 0.5},
                whiskerprops={"color": "C0", "linewidth": 1.5},
                capprops={"color": "C0", "linewidth": 1.5})
    ax4.set_title('Boxplot', fontdict=fontdict)
    ax4.set_xticks(range(1, len(names) + 1), names)
    ax4.set_ylabel('Time in ns', fontdict=fontdict)

    for i in range(len(names)):
        ax5.plot(x, data[i], color=color[i], label=names[i], ls=':', marker='o', markersize=0.5)
    ax5.legend(loc='upper right', ncols=3)
    for j in range(len(names)):
        plt.gca().get_legend().legend_handles[j].set_color(color[j])
    ax5.set_title(title_ax5, fontdict=fontdict)
    ax5.set_ylabel(y_label, fontsize=8)
    ax5.set_xlabel(x_label, fontsize=8)

    plt.subplots_adjust(bottom=0.1, top=0.9, hspace=0.5, wspace=0.3)
    plt.savefig(save)
    np.save(save, rs, sha, nc)
    plt.show()

    #return rs, sha, nc

def plot_prob(prob_data_2D):
    title = ''
    title_ax1 = ''
    title_ax2 = ''
    title_ax3 = ''
    title_ax4 = ''
    save = '' #file name
    prop2D = prob_data_2D
    fontdict = {'fontsize': 9} #font size of titles
    y_label = 'propability'
    x_label = 'Number of Encryptions'

    if np.array_equal(prop2D, prob['rs']):
        title = 'FP-Probability of Reed Solomon'
        title_ax1 = 'FP-Probability of Reed Solomon (linear)'
        title_ax2 = 'FP-Probability of Reed Solomon (log)'
        title_ax3 = 'FP-Probability of Reed Solomon (normalized)'
        title_ax4 = 'FP-Probability of Reed Solomon (Boxplot)'
        save = "PropPlot_RS_subplots.png"
    elif np.array_equal(prop2D, prob['sha']):
        title = 'FP-Probability of Sha'
        title_ax1 = 'FP-Probability of Sha (linear)'
        title_ax2 = 'FP-Probability of Sha (log)'
        title_ax3 = 'FP-Probability of Sha (normalized)'
        title_ax4 = 'FP-Probability of Sha (Boxplot)'
        save = "PropPlot_Sha_subplots.png"
    elif np.array_equal(prop2D, prob['nc']):
        title = 'FP-Probability of No Code'
        title_ax1 = 'FP-Probability of NoCode (linear)'
        title_ax2 = 'FP-Probability of NoCode (log)'
        title_ax3 = 'FP-Probability of NoCode (normalized)'
        title_ax4 = 'FP-Probability of NoCode (Boxplot)'
        save = "PropPlot_NC_subplots.png"

    # prep data array for plotting
    prop = prob_data_2D.flatten() # 2D -> 1D-Array

    #option 1: sorted values --> use prop_sort
    #prop_sort = np.sort(prop) #bring values in order
    #prop_data_1D_scaled = [val * scal_fact for val in prop_data_1D_sort] #bc of super small values
    #prop_sort[prop_sort < 1e-5] = 0 #ignore values that are too small

    #option 2: non sorted --> use prop
    prop[prop < 1e-5] = 0  # ignore values that are too small

    length = len(prop)
    x = np.linspace(0, length, length, endpoint=True) #x-Achse der Plots

    prop_norm = prop / np.max(prop) #normalize values
    is_infinity = np.isinf(prop) #identify if value is infinity
    prop_noInf = prop[~is_infinity]

    #plotten auf verschiedene Arten, um übersichtlichste Lösung zu finden
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle(title, fontsize=16)

    #regular histogram
    ax1.plot(x, prop, color='m', ls=':', marker='o', markersize=1)
    ax1.set_title(title_ax1, fontdict=fontdict)
    ax1.set_ylabel(y_label, fontsize=8)
    ax1.set_xlabel(x_label, fontsize=8)

    #histogram with log-axis
    ax2.plot(x, prop_noInf, color='b', ls=':', marker='o', markersize=1)
    ax2.set_title(title_ax2, fontdict=fontdict)
    ax2.set_yscale('log')
    ax2.set_ylabel(y_label, fontsize=8)
    ax2.set_xlabel(x_label, fontsize=8)

    #normalized histogram
    ax3.plot(x, prop_norm, color='g', ls=':', marker='.', markersize=1)
    ax3.set_title(title_ax3, fontdict=fontdict)
    ax3.set_ylabel(y_label, fontsize=8)
    ax3.set_xlabel(x_label, fontsize=8)

    #boxplot
    ax4.boxplot(prop, patch_artist=True,
                showmeans=True, showfliers=True,
                medianprops={"color": "white", "linewidth": 0.5},
                boxprops={"facecolor": "C0", "edgecolor": "white",
                          "linewidth": 0.5},
                whiskerprops={"color": "C0", "linewidth": 1.5},
                capprops={"color": "C0", "linewidth": 1.5})
    ax4.set_title(title_ax4, fontdict=fontdict)
    ax4.set_ylabel(y_label, fontsize=8)
    #ax4.set_ybound(np.min(prop_data_1D_scaled) - 1, np.max(prop_data_1D_scaled))
    #ax4.legend([f'Scale: {scal_fact}'])

    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.savefig(save)
    np.save(save, prop)

    plt.show()

    return prop


prob = prob_various(8, 20)

#plotten der FP-Wahrscheinlichkeiten
prob_rs = prob['rs']
prob_sha = prob['sha']
prob_nc = prob['nc']

# plot_prob(prob_nc) #enthält sehr große negative Werte?? NOCHMAL PRÜFEN!!
# plot_prob(prob_sha)
# plot_prob(prob_rs)


#plotten der Encryption-Zeit
time_rs = prob['rs_t']
time_sha = prob['sha_t']
time_nc = prob['nc_t']
plot_multiple(time_rs, time_sha, time_nc)

# plot_multiple(prob_rs, prob_sha, prob_nc)

print('\n finished')

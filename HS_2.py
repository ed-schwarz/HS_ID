import hashlib
import numpy as np
import math
import random
from reedsolo import RSCodec, ReedSolomonError
import collections


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
        q = math.pow(2, self.z)
        size = self.k
        ran = pow(2, size)
        elements = self.elements
        i = 0
        id1_array = []
        id2_array = []
        en1_array = []
        en2_array = []
        fp_array = []

        if self.alg == "rs":
            while(i < elements):
                id1 = self.generate_encoded_id(ran, size, q)
                id2 = self.generate_encoded_id(ran, size, q)

                rs1 = self.encode_and_convert_rs(id1, size, q)
                rs2 = self.encode_and_convert_rs(id2, size, q)

                fp = self.detect_fp(rs1, rs2)

                id1_array.append(id1)
                id2_array.append(id2)

                en1_array.append(rs1)
                en2_array.append(rs2)

                fp_array.append(fp)


                i += 1

        elif self.alg == "sha":
            while(i < elements):
                id1 = self.generate_encoded_id(ran, size, q)
                id2 = self.generate_encoded_id(ran, size, q)

                sha1 = self.encode_and_convert_hash(id1, size, q)
                sha2 = self.encode_and_convert_hash(id2, size, q)

                fp = self.detect_fp(sha1, sha2)

                id1_array.append(id1)
                id2_array.append(id2)

                en1_array.append(sha1)
                en2_array.append(sha2)

                fp_array.append(fp)

                i += 1

        elif self.alg == "nc":
            while(i < elements):
                id1 = self.generate_encoded_id(ran, size, q)
                id2 = self.generate_encoded_id(ran, size, q)

                fp = self.detect_fp(id1, id2)

                id1_array.append(id1)
                id2_array.append(id2)

                fp_array.append(fp)

                i += 1
                
        sim_arays = {'id1':id1_array, 'id2':id2_array, 'en1':en1_array, 'en2':en2_array, 'fp':fp_array}

        return sim_arays
    
    def calculate_prob(self):
        sim = self.simulate()
        fp = sim['fp']
        count = 0

        for i in range(0, self.elements):
            if fp[i]:
                count +=1
        
        prob = count/self.elements

        return prob


def prob_various(z_max, k_max):
    z = 1
    elements = 10000
    rs = "rs"
    sha = "sha"
    nc = "nc"

    prob_rs_3d = np.empty([z_max, k_max])
    prob_sha_3d = np.empty([z_max, k_max])
    prob_nc_3d = np.empty([z_max, k_max])

    while(z < z_max):

        k=10

        while(k < k_max):
            enc_rs = dummy_encoder(k, z, rs, elements)
            enc_sha = dummy_encoder(k, z, sha, elements)
            enc_nc = dummy_encoder(k, z, nc, elements)

            prob_rs = enc_rs.calculate_prob()
            prob_sha = enc_sha.calculate_prob()
            prob_nc = enc_nc.calculate_prob()

            prob_rs_3d[z, k] = prob_rs
            prob_sha_3d[z, k] = prob_sha
            prob_nc_3d[z, k] = prob_nc

            k += 1

        z += 1

    prob_dict = {'rs':prob_rs_3d, 'sha':prob_sha_3d, 'nc':prob_nc_3d}
    return prob_dict


prob = prob_various(8, 20)
prob_rs = prob['rs']
prob_sha = prob['sha']
prob_nc = prob['nc']
print(prob_sha)
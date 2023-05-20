import hashlib
import numpy as np
import math
import random
from reedsolo import RSCodec, ReedSolomonError
import collections

#generates a random number between 0 and range
def generate_random_ID(ran):
    return random.randrange(0, ran)

#encodes to hash, withoutput as hex
def encode_hash(id):
    m = hashlib.sha256()
    m.update(repr(id).encode())
    sha = m.hexdigest()
    return sha

#encodes to reed solomon, withoutput as hex
def encode_rs(id, size):
    rsc = RSCodec(size)
    rs = rsc.encode(id)
    out = ''.join(format(x, '02x') for x in rs)
    return out

#convert to binary from int
def convert_to_bin(id, size):
    bin_list = list(bin(id)[2:].zfill(size))
    bin_list.reverse()
    return bin_list[0:size-1]

#convert to q-ary from binary
def convert_to_symbol(binv, exp):
    q = int(math.log(exp, 2))
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
def convert_from_hex(id, size, q):
    num_int = int(id, 16)
    num_bin = convert_to_bin(num_int, size)
    num_sym = convert_to_symbol(num_bin, q)
    return num_sym

#encode to hash and return q-ary
def encode_and_convert_hash(id, size, q):
    sha = encode_hash(id)
    out = convert_from_hex(sha, size, q)
    return out

#encode to reed solomon and return q-ary
def encode_and_convert_rs(id, size, q):
    rs_size = int(size/2)
    rs = encode_rs(id, rs_size)
    out = convert_from_hex(rs, size, q) 
    return out

#generate random id and convert to q-ary
def generate_encoded_id(ran, size, q):
    id = generate_random_ID(ran)
    id_bin = convert_to_bin(id, size)
    id_vec = convert_to_symbol(id_bin, q)
    return id_vec

#compare two vectors in specific position
def compare_vector(id1, id2, index):
    ran = len(id1)
    if(id1[index] == id2[index]):
        return True
    else:
        return False
    
#compare two ids and return true if false positive
def detect_fp(id1, id2):
    ran = len(id1)
    ran2 = len(id2)
    if(ran!=ran2):
        print("Error")
    index = random.randrange(0, ran-1)
    ran_ind = compare_vector(id1, id2, index)
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
def simulate(q, elements, size, ran):
    i = 0
    id1_array = []
    id2_array = []
    sha1_array = []
    sha2_array = []
    rs1_array = []
    rs2_array = []
    fp_NC_array = []
    fp_SHA_array = []
    fp_RS_array = []

    while(i < elements):
        id1 = generate_encoded_id(ran, size, q)
        id2 = generate_encoded_id(ran, size, q)

        sha1 = encode_and_convert_hash(id1, size, q)
        sha2 = encode_and_convert_hash(id2, size, q)

        rs1 = encode_and_convert_rs(id1, size, q)
        rs2 = encode_and_convert_rs(id2, size, q)

        fp_NC = detect_fp(id1, id2)
        fp_SHA = detect_fp(sha1, sha2)
        fp_RS = detect_fp(rs1, rs2)

        id1_array.append(id1)
        id2_array.append(id2)
        sha1_array.append(sha1)
        sha2_array.append(sha2)
        rs1_array.append(rs1)
        rs2_array.append(rs2)
        fp_NC_array.append(fp_NC)
        fp_SHA_array.append(fp_SHA)
        fp_RS_array.append(fp_RS)
        i += 1

    sim_arays = {'id1':id1_array, 'id2':id2_array, 'sha1':sha1_array, 'sha2':sha2_array, 'rs1':rs1_array, 'rs2':rs2_array, 'fp_NC':fp_NC_array, 'fp_SHA':fp_SHA_array, 'fp_RS':fp_RS_array}

    return sim_arays

q= 8
elements = 1
size = q*q
range_id = pow(2, size)
sim_array = simulate(q, elements, size, range_id)
print("Done")





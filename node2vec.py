import numpy as np

def generate_node_vector(input_file):
    line = []
    with open(input_file, 'r') as f:
            lines = f.read().split("\n")
            line.append(lines)
    line = line[0]        
    info = line[0].split(" ")
    dim = int(info[-1])
    aa_vectors = np.zeros((20,dim)) 
    for aa in line[1:-1]:
        vect = aa.split(" ")
        aa_id = int(vect[0])
        aa_vectors[aa_id] = [float(i)  for i in vect[1:]]    
    return aa_vectors




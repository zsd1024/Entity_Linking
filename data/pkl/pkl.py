import pickle
import numpy as np
np.set_printoptions(threshold=1000000000000000)

path = 'pos_embed_128.pkl'
file = open(path,'rb')
inf = pickle.load(file)
print(len(inf))
for i in range(len(inf)):
    print('i'+str(i)+':'+str(inf[i].size()))

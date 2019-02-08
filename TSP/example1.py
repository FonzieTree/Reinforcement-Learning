# this code was primarily produced bu Shuming Fang
# email: fangshuming519@gmail.com
import numpy as np
import copy
epsilon = 0.5
gamma = 0.98
lr = 0.1
distance =np.array([[0,7,6,1,3],[7,0,3,7,8],[6,3,0,12,11],[1,7,12,0,2],[3,8,11,2,0]])
R_table = 11-distance
space = [0,1,2,3,4]
Q_table = np.zeros((5,5))
for i in range(10):
    path = [0]
    for j in range(4):
        #print(path)
        s = list(path)[j]
        s_row = Q_table[s]
        remain = set(space)-set(path)
        max_value = -1000
        for rm in remain:
            Q_ = Q_table[s, rm]
            if Q_>max_value:
                max_value = Q_
                a = rm
        if np.random.uniform()<epsilon:
            a = np.random.choice(np.array(list(set(space)-set(path))))
        if j!=3:
            Q_table[s,a] =  (1-lr)*Q_table[s,a]+lr*(R_table[s,a]+gamma*max_value)
        else:
            Q_table[s,a] =  (1-lr)*Q_table[s,a]+lr*R_table[s,a]
        path.append(a)
    Q_table[a,0] =  (1-lr)*Q_table[a,0]+lr*R_table[a,0]
    path.append(0)
print(Q_table)
#[[ 0.          0.79082557  2.07108478  5.97007988  0.        ]
# [ 0.76        0.          3.53067459  0.4         0.60528   ]
# [ 0.95        3.03408288  0.         -0.1         0.        ]
# [ 1.9         0.7992      0.241388    0.          4.3765839 ]
# [ 2.7512      0.8982012   0.0587412   1.71        0.        ]]
# the final solution would be [1,4,5,2,3,1], and the mimimum distance is 20.

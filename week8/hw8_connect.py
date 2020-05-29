import time
import numpy as np
t_start = time.time()
matrix=np.array([[0,0,0,0,0,0,0,0,0,0],
                 [0,1,1,1,1,0,0,0,1,1],
                 [0,0,1,1,1,0,1,1,1,1],
                 [0,0,1,1,0,0,1,1,1,1],
                 [0,1,1,1,0,0,1,1,1,0],
                 [0,0,0,0,0,0,0,0,1,0],
                 [0,0,0,1,1,1,0,0,1,0],
                 [0,0,0,1,1,1,1,0,0,0],
                 [0,0,0,1,1,1,1,0,0,0],
                 [0,0,0,0,0,0,1,0,0,0]])
labels=np.zeros((matrix.shape), dtype=int)
tag=0
connect=[]
def mark():
    global labels,i,j,tag,connect
    if labels[j,i-1]==0 and labels[j-1,i]==0:
        tag+=1
        labels[j,i]=tag
    elif labels[j,i-1]!=0 and labels[j-1,i]!=0:
        labels[j,i]=min(labels[j-1,i],labels[j,i-1])
        if labels[j,i-1]!=labels[j-1,i]:
            connect.append([min(labels[j-1,i],labels[j,i-1]),max(labels[j-1,i],labels[j,i-1])])
    else:
        labels[j,i]=max(labels[j-1,i],labels[j,i-1])

    if matrix[j,i]==1:
        return i,j
    mark()
for j in range(matrix.shape[0]):
    for i in range(matrix.shape[1]):
        if matrix[j,i]==1:
            mark()
garbage = set(tuple(l) for l in connect)
connect = [list(l) for l in garbage]
garbage = list(range(1,tag+1))
for i in connect:   
    labels[labels==i[1]]=i[0]
    garbage.remove(i[1])
bulk=[]
for i in garbage:
    bulk.append(np.argwhere(labels==i))
print('there are',len(bulk),'object')
for i,j in enumerate(bulk):
    print('the area of',i,'object is',len(j))
t_end = time.time()
print('time elapsed: ' + str(round(t_end-t_start,6)) + ' seconds')

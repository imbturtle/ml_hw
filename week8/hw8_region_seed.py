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
def floodfill(x,y):
    global area
    if x>0:
        if matrix[y,x-1]==1:
            if check([x-1,y])==0:
                area.append([x-1,y])
                floodfill(x-1,y)
    if x<matrix.shape[1]-1:
       if matrix[y,x+1]==1:
           if check([x+1,y])==0:
               area.append([x+1,y])
               floodfill(x+1,y)
    if y>0:
       if matrix[y-1,x]==1:
           if check([x,y-1])==0:
               area.append([x,y-1])
               floodfill(x,y-1)
    if y<matrix.shape[0]-1:
       if matrix[y+1,x]==1:
           if check([x,y+1])==0:
               area.append([x,y+1])
               floodfill(x,y+1)
    return area
def check(x):
    global area
    for address in bulk:
        if x in address:  
            return 1
    if x in area:  
        return 1
    return 0
bulk=[]
area=[]
for j in range(matrix.shape[0]):
    for i in range(matrix.shape[1]):
        if matrix[j,i]==1:
            if check([i,j]):
                continue
            area=[[i,j]]
            bulk.append(floodfill(i,j))
print('there are',len(bulk),'object')
for i,j in enumerate(bulk):
    print('the area of',i,'object is',len(j))
t_end = time.time()
print('time elapsed: ' + str(round(t_end-t_start,6)) + ' seconds')
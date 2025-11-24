# Version 1.0

from itertools import combinations

PERMTYPE1=['()','(01)','(20)','(12)','(012)','(210)']
PERMTYPE2=['012','102','210','021','120','201']
MAXPERM=6
def rowperm(permtype, lol):
    if permtype==0:
        return lol[:]
    elif permtype==1: 
        return [lol[1],lol[0],lol[2]]
    elif permtype==2: 
        return [lol[2],lol[1],lol[0]]
    elif permtype==3:  
        return [lol[0],lol[2],lol[1]]
    elif permtype==4: 
        return [lol[2],lol[0],lol[1]]
    elif permtype==5: 
        return [lol[1],lol[2],lol[0]]
    else:
        assert(False)

def colperm(permtype, lol):
    return list(zip(*rowperm(permtype,list(zip(*lol)))))

N=4

cnt=0
maxmatrices=[]
for tupel in combinations([0,1,2,3,4,5,6,7,8],N):
    cnt+=1
    # generate 3x3 matrix
    grid=[[0,0,0],[0,0,0],[0,0,0]]
    for index in tupel:
        rowindex=index//3
        colindex=index%3
        grid[rowindex][colindex]=1
    grid=list(map(tuple,grid))
    # optimize
    maxweight=(0,)*9
    maxmatrix=None
    for i in range(MAXPERM):
        for j in range(MAXPERM):
            matrix=grid[:]
            matrix=colperm(j,rowperm(i,matrix))
            weight=tuple(item for row in matrix for item in row)
            if weight>maxweight:
                maxweight=weight
                maxmatrix=matrix[:]
    if maxmatrix not in maxmatrices:
        maxmatrices.append(maxmatrix)

cnt=0
for matrix in maxmatrices:
    cnt+=1
    print(cnt)
    print (matrix[0])
    print (matrix[1])
    print (matrix[2])
    print()


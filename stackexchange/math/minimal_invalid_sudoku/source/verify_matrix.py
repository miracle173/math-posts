# verify Matrix
from  verify_matrix_config import Testmatrix
from itertools import permutations
from copy import deepcopy
from time import time
from random import shuffle


class ImplementationError(Exception):
    '''feature works not as expected'''
matrix990=[
    [ 0, 1, 2, 3, 4, 5, 6, 7, 8,],
    [10,11,12,13,14,15,16,17,18,],
    [20,21,22,23,24,25,26,27,28,],
    [30,31,32,33,34,35,36,37,38,],
    [40,41,42,43,44,45,46,47,48,],
    [50,51,52,53,54,55,56,57,58,],
    [60,61,62,63,64,65,66,67,68,],
    [70,71,72,73,74,75,76,77,78,],
    [80,81,82,83,84,85,86,87,88,],
    ]

    
matrix991=[    
    [11,12,13,14,15,16,17,18,19,],
    [21,22,23,24,25,26,27,28,29,],
    [31,32,33,34,35,36,37,38,39,],
    [41,42,43,44,45,46,47,48,49,],
    [51,52,53,54,55,56,57,58,59,],
    [61,62,63,64,65,66,67,68,69,],
    [71,72,73,74,75,76,77,78,79,],
    [81,82,83,84,85,86,87,88,89,],
    [91,92,93,94,95,96,97,98,99,],
    ]

permutation9=[4,7,3,8,1,0,5,2,6]
group3=1
permutation3=[2,0,1]


'''
group3permutation3=[0,1,2,5,3,4,6,7,8]


permuted9Matrix991=[    
    [51,52,53,54,55,56,57,58,59,],
    [81,82,83,84,85,86,87,88,89,],
    [41,42,43,44,45,46,47,48,49,],
    [91,92,93,94,95,96,97,98,99,],
    [21,22,23,24,25,26,27,28,29,],
    [11,12,13,14,15,16,17,18,19,],
    [61,62,63,64,65,66,67,68,69,],
    [31,32,33,34,35,36,37,38,39,],
    [71,72,73,74,75,76,77,78,79,],
    ]
'''
    
matrix330=[
    [ 0, 1, 2,],
    [10,11,12,],
    [20,21,22,],
    ]


matrix331=[
    [11,12,13,],
    [21,22,23,],
    [31,32,33,],
    ]
    
    
matrixsparse991=[
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0],
    ]
    
sparse991=[
    [6,3,1],
    [2,2,1],
    [7,7,1],
    [3,5,1],
    ]
    
matrixsparse99v=[
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0],
    [0,0,0,0,0,2,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,3,0,0,0,0,0],
    [0,0,0,0,0,0,0,4,0],
    [0,0,0,0,0,0,0,0,0],
    ]
sparse99v=[
    [6,3,3],
    [2,2,1],
    [7,7,4],
    [3,5,2],
]

        
        
def compareMatrices(matrix1, matrix2):
    if type(matrix1)!=type(matrix2):
        raise ImplementationError("matrices have different type")
    for r in range(len(matrix1)):
        if type(matrix1[r])!=type(matrix2[r]):
            raise ImplementationError("matrices have different type of row %d"%r)
    if len(matrix1)!=len(matrix2):
        raise ImplementationError("")
    if len(matrix1[0])!=len(matrix2[0]):
        raise ImplementationError("matrices have different col dimensions")
    for r in range(len(matrix1)):
        for c in range(len(matrix1[0])):
            if matrix1[r][c]!=matrix2[r][c]:
                raise ImplementationError("elements at (%d,%d) differ in matrices"%(r,c))
    



sparsematrix1=Testmatrix.fromRectangular(matrixsparse99v)
sparsematrix2=Testmatrix(9,9,sparse99v)

try:
    compareMatrices(sparsematrix1.toRectangular(),sparsematrix2.toRectangular())
except:
    print("Error")
    printMatrix(sparsematrix1.toRectangular(),"sparsematrix1")
    printMatrix( sparsematrix2.toRectangular(),"sparsematrix2")
    raise
    
    
# test "copy"    
sparsematrix1=Testmatrix.fromRectangular(matrix991)
sparsematrix2=Testmatrix.copy(sparsematrix1)
try:
    compareMatrices(sparsematrix1.toRectangular(),sparsematrix2.toRectangular())
except:
    print("Error")
    printMatrix(sparsematrix1.toRectangular(),"sparsematrix1")
    printMatrix( sparsematrix2.toRectangular(),"sparsematrix2")
    raise


# test "permuteRows"
sparsematrix1=Testmatrix.fromRectangular(matrix991)
sparsematrix1.permuteRows(permutation9)

matrix2=deepcopy(matrix991)
matrix2=[matrix2[permutation9[r]] for r in range(len(matrix2))]

try:
    compareMatrices(sparsematrix1.toRectangular(),matrix2)
except:
    print("Error")
    print("permutation", permutation9)
    printMatrix(sparsematrix1.toRectangular(),"sparsematrix1")
    printMatrix( matrix2,"matrix2")
    raise



# test "permuteCols"
sparsematrix1=Testmatrix.fromRectangular(matrix991)
sparsematrix1.permuteCols(permutation9)

matrix2=deepcopy(matrix991)
matrix2=list(zip(*matrix2))
matrix2=[matrix2[permutation9[r]] for r in range(len(matrix2))]
matrix2=list(map(list,list(zip(*matrix2))))

try:
    compareMatrices(sparsematrix1.toRectangular(),matrix2)
except:
    print("Error")
    print("permutation", permutation9)
    printMatrix(sparsematrix1.toRectangular(),"sparsematrix1")
    printMatrix( matrix2,"matrix2")
    print(sparsematrix1.toRectangular())
    print(matrix2)
    raise

# permute rows in band

sparsematrix1=Testmatrix.fromRectangular(matrix991)
sparsematrix1.permuteRowsInBand(group3, permutation3)
matrix2=deepcopy(matrix991)
matrix2=[matrix2[3*group3+permutation3[r%3] if r//3==group3 else r] for r in range(len(matrix2))]

try:
    compareMatrices(sparsematrix1.toRectangular(),matrix2)
except:
    print("Error")
    print("Band", group3)
    print("permutation", permutation3)
    printMatrix(sparsematrix1.toRectangular(),"sparsematrix1")
    printMatrix( matrix2,"matrix2")
    raise


# permute cols in stack

sparsematrix1=Testmatrix.fromRectangular(matrix991)
sparsematrix1.permuteColsInStack(group3, permutation3)
matrix2=deepcopy(matrix991)
matrix2=list(zip(*matrix2))
matrix2=[matrix2[3*group3+permutation3[r%3] if r//3==group3 else r] for r in range(len(matrix2))]
matrix2=list(map(list,list(zip(*matrix2))))



try:
    compareMatrices(sparsematrix1.toRectangular(),matrix2)
except:
    print("Error")
    print("Stack", group3)
    print("permutation", permutation3)
    printMatrix(sparsematrix1.toRectangular(),"sparsematrix1")
    printMatrix( matrix2,"matrix2")
    raise

# transpose

sparsematrix1=Testmatrix.fromRectangular(matrix991)
sparsematrix1.transpose()
matrix2=deepcopy(matrix991)
matrix2=[list(row) for row in list(zip(*matrix2))]

try:
    compareMatrices(sparsematrix1.toRectangular(),matrix2)
except:
    print("Error")
    print("transpose")
    printMatrix(sparsematrix1.toRectangular(),"sparsematrix1")
    printMatrix( matrix2,"matrix2")
    raise


TestName=["transpose","permuteRows","permuteCols","permuteBands","permuteStacks","permuteRowsInBand","permuteColsInStack","transpose"]
DensematrixFunction=[transpose, permuteRows, permuteCols, permuteBands, permuteStacks, permuteRowsInBand, permuteColsInStack]
TestmatrixMethods=[sparsematrix.transpose, sparsematrix.permuteRows, sparsematrix.permuteCols, sparsematrix.permuteBands, sparsematrix.permuteStacks, sparsematrix.permuteRowsInBand, sparsematrix.permuteColsInStack]


sparsematrix1=Testmatrix.fromRectangular(matrix331)
#print(1,(sparsematrix1[1]))


sparsematrix=Testmatrix(9,9,sparse99v)
#densematrix=sparsematrix.toRectangular()
densematrix=matrix991

# performance test

diffTimeDense={}
diffTimeSparse={}
N=1000000
startTest=time()
for n in range(len(TestmatrixMethods)):
    if n==0:
        sparsematrixArgs=[]
    elif n in [1,2]:
        sparsematrixArgs=[permutation9]
    elif n in [3,4]:
        sparsematrixArgs=[permutation3]
    elif n in [5,6]:
        sparsematrixArgs=[group3, permutation3]
    densematrixArgs=[densematrix]
    densematrixArgs.extend(sparsematrixArgs)
    startTime=time()
    for _ in range(N):
        DensematrixFunction[n](*densematrixArgs)
    diffTimeDense[TestName[n]]=time()-startTime
    startTime=time()
    for _ in range(N):
        TestmatrixMethods[n](*sparsematrixArgs)
    diffTimeSparse[TestName[n]]=time()-startTime
diffTest=time()-startTest
print("test duration = %.1f s"%diffTest)
print("%-20s %10s %10s %10s %10s %10s"%("N = %d"%N, "sprs/dns", "sparse", "dense","tot.sparse","tot.dense"))
for n in range(len(TestmatrixMethods)):
    print("%-20s %10.1f %10.1e %10.1e %10.1e %10.1e"%(TestName[n],
        diffTimeSparse[TestName[n]]/diffTimeDense[TestName[n]],
        diffTimeSparse[TestName[n]]/N, diffTimeDense[TestName[n]]/N,
        diffTimeSparse[TestName[n]], diffTimeDense[TestName[n]]))
     
        






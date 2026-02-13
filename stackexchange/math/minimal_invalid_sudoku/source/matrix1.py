from copy import deepcopy

def transpose(matrix):
    return list(zip(*matrix))

def permuteRows(matrix, permutation): 
    return [matrix[permutation[r]] for r in range(len(matrix))]

def permuteCols(matrix, permutation):
    # myMatrix=list(zip(*matrix))
    # myMatrix=[myMatrix[permutation[r]] for r in range(len(myMatrix))]
    # return=list(map(list,list(zip(*myMatrix))))
    return (transpose(permuteRows(transpose(matrix),permutation)))

def permuteBands(matrix, permutation):
    return [matrix[3*permutation[r//3]+r%3] for r in range(len(matrix))]

def permuteStacks(matrix, permutation):
    return(transpose(permuteBands(transpose(matrix), permutation)))
    
def permuteRowsInBand(matrix, index, permutation):
    return [matrix[3*index+permutation[r%3] if r//3==index else r] for r in range(len(matrix))]

def permuteColsInStack(matrix, index, permutation):
        # myMatrix=list(zip(*matrix))
        # myMatrix=[myMatrix[3*index+permutation[r%3] if r//3==index else r] for r in range(len(myMatrix))]
        # return list(map(list,list(zip(*myMatrix))))
        return(transpose(permuteRowsInBand(transpose(matrix), index, permutation)))

def toRectangularMatrix(aMatrix):
    return deepcopy(aMatrix)
    
def printMatrix(first, second=None):
    if second is None:
        aMatrix=first
        hasMessage=False
    else:
        aMatrix=second
        message=first
        hasMessage=True
    if hasMessage:
        print(message)    
    myMatrix=toRectangularMatrix(aMatrix)
    myRowDim=len(myMatrix)
    myColDim=len(myMatrix[0])
    width=[0]*myColDim
    for r in range(myRowDim):
        for c in range(myColDim):
            width[c]=max(width[c],len(str(myMatrix[r][c])))
    for r in range(myRowDim):
        for c in range(myColDim):
            if c==0:
                line=''
            else:
                line+=' '
            line+='%0'+str(width[c])+'d'
        print(line%tuple(myMatrix[r]))


# def printMatrix(matrix, message=None):
    # if message is not None:
        # print(message)
    # for row in matrix:
        # print(row)

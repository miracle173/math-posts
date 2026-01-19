'''
python file: simple_sudoku
'''
from copy import deepcopy
from itertools import combinations, permutations
from tools_sudoku import *
from extensions_sudoku import *

# trace flags
TRACE_GLOBAL_OPTIMIZATION=True

# trace functions
if TRACE_GLOBAL_OPTIMIZATION:
    countTraceGlobalOptimization=0
    def traceGlobalOptimization(aMessage, aObject):
        global countTraceGlobalOptimization
        countTraceGlobalOptimization+=1
        print(aMessage+" "+str(countTraceGlobalOptimization))
        print(aObject)
else:
    def traceGlobalOptimization(aMessage, aObject):
        pass
    

# from typing import List, Tuple

# Reflection = bool
# Permutation = Tuple[int, int, int]
# RowPermutation = Permutation
# ColPermuation = Permutation

# LocalTransform = Tuple[int, Permutation]
# GlobalTransform = Tuple[Reflection, RowPermutation, ColPermuation]

# Row = Tuple[int, int, int]
# Block = Tuple[ Row, Row, Row]
# Band = Tuple [ Block, Block, Block]
# Sudoku = Tuple [ Band, Band, Band]



################################
### constants
################################

IdPermutation=(0,1,2)
AllPermutations=[
    [],
    [IdPermutation],
    [IdPermutation, (1,0,2)],
    [IdPermutation, (0,2,1), (1,0,2), (1,2,0), (2,0,1),(2,1,0)]]


################################
### operations on objects
################################

def transposeMatrix(matrix):
    return tuple(zip(*matrix))

        
def permuteBands(block, permutation):
    return tuple(block[permutation[r]] for r in range(3))

def permuteStacks(block, permutation):
     return transposeMatrix(permuteBands(transposeMatrix(block),permutation))

################################    
### Sudoku operations
################################

def transposeSudoku(sudoku):
    return tuple(
        tuple(transposeMatrix(block) for block in band) for band in transposeMatrix(sudoku))

    
def permuteRowsInBand(sudoku, band, permutation):
    return tuple(
        tuple(sudoku[b][s] if b!=band else 
            (permuteBands(sudoku[b][s], permutation)) for s in range(3))  for b in range(3))


def permuteColsInStack(sudoku, stack, permutation):
     return tuple(
        tuple(sudoku[b][s] if s!=stack else 
             tuple(zip(*(permuteBands(tuple(zip(*sudoku[b][s])), permutation))))
            for s in range(3))  for b in range(3))

def noDuplicates(values):
    seen = set()
    for v in values:
        if v == 0:
            continue
        if v in seen:
            return False
        seen.add(v)
    return True


def isValidSudoku(sudoku):
    # --- check blocks ---
    for b in range(3):
        for s in range(3):
            block = [
                sudoku[b][s][r][c]
                for r in range(3)
                for c in range(3)
            ]
            if not noDuplicates(block):
                return False

    # --- check rows ---
    for global_row in range(9):
        b = global_row // 3
        r = global_row % 3
        row = [
            sudoku[b][s][r][c]
            for s in range(3)
            for c in range(3)
        ]
        if not noDuplicates(row):
            return False

    # --- check columns ---
    for global_col in range(9):
        s = global_col // 3
        c = global_col % 3
        col = [
            sudoku[b][s][r][c]
            for b in range(3)
            for r in range(3)
        ]
        if not noDuplicates(col):
            return False

    return True



def analyzeBlock(m, rowStart=0, colStart=0):
    '''
    # by chatGPT
    argument:
        block: 3*3 Matrix
    result:
        ((rowPerm,colPerm),(rowCount,colCount),contigous):
        (rowPerm,colPerm): row-permuation/column-permuation to make the block contigous
        (rowCount,colCount):the number of rows and the number of columns of the contigous block
        contigous: True if the block is already contigous, false else.
        a block is contigous, if all allzero column are at the right 
        and all allzero rows at the bottom of the matrix
    '''    # --- Zeilen ---
    fixed_rows = list(range(rowStart))
    rest_rows = list(range(rowStart, 3))

    nonzero_rows = []
    zero_rows = []

    contigous=True
    zeroAlreadyFound=False
    for i in rest_rows:
        if any(m[i][j] != 0 for j in range(3)):
            nonzero_rows.append(i)
            if zeroAlreadyFound:
                contigous=False
        else:
            zeroAlreadyFound=True
            zero_rows.append(i)

    row_perm = fixed_rows + nonzero_rows + zero_rows
    rowCount = len(nonzero_rows)
    # --- Spalten ---
    fixed_cols = list(range(colStart))
    rest_cols = list(range(colStart, 3))

    nonzero_cols = []
    zero_cols = []
    zeroAlreadyFound=False
    for j in rest_cols:
        if any(m[i][j] != 0 for i in range(3)):
            nonzero_cols.append(j)
            if zeroAlreadyFound:
                contigous=False
        else:
            zeroAlreadyFound=True
            zero_cols.append(j)

    col_perm = fixed_cols + nonzero_cols + zero_cols
    colCount = len(nonzero_cols)
    return ((row_perm, col_perm),
        (rowCount, colCount),
        contigous)


    
    
def compose(left, right):
    '''
    purpose: compose two permutations
    Arguments:
        left, right: permutation of the numbers 0....n for a number n
            represented by lists
            the meaning is: left(i)=left[i]
    Return:
        the permutation left o right, with 
        (left o right)(x) = left(right(x))
    '''
    assert(len(left)==len(right))
    assert(sorted(left)==list(range(len(left))))
    assert(sorted(right)==list(range(len(right))))
    return([left[right[i]] for i in range(len(right))])
      

##################################
### optimization
##################################      
def optimizeBlock(matrix, breakIfNotMaximal=False):
    '''
    purpose:
        retrieves the maximal matrix ftom the equivalence class of the 
        input matrix. Two matrices are equivalemnt if they can be transformed 
        one to the other by reflectio, row or column permutation.
        The subroutine can also be used to decide if a matrix is the 
        maximal element of its equivalence class.
    arguments:
        matrix: a 3*3 0-1 matrix that should be optimized
        breakIfNotMaximal: default False. If set to True, the function  returns None as soon as a larger matrix is derived from matrix because then matrix is not a maximum.
    return:
        None | (optMatrix, [stabilizer  ...])
        stabilizer = (reflection (row permutation, column permutation))
        reflection is a boolean value. True means matrix transposition, False means idendity
        the permutations are 3 element lists of permutations of the numbers 0,1,2
        optMatrix is the maximal element
    '''
    assert(type(matrix)==type(tuple()))
    (rowperm,colperm), (rowmax, colmax), contigous = analyzeBlock(matrix)
    if not contigous:
        if breakIfNotMaximal:
            traceGlobalOptimization("global optimization: skip, not contigous", matrix)
            return None
        matrix=permuteBands(matrix, rowperm)
        matrix=permuteStacks(matrix, colperm)
    assert(type(matrix)==type(tuple()))
    optPermutationList=[]
    for reflection in [False, True]:
        if reflection:
            baseMatrix=transposeMatrix(matrix)
            baseRowPerm,baseColPerm=colperm,rowperm
        else:
            baseMatrix=matrix
            baseRowPerm,baseColPerm=rowperm,colperm
        for rp in AllPermutations[rowmax]:
            for cp in AllPermutations[colmax]:
                copyMatrix=deepcopy(baseMatrix)
                copyMatrix=permuteBands(copyMatrix,rp)
                copyMatrix=permuteStacks(copyMatrix,cp)
                if optPermutationList==[] or copyMatrix>=optMatrix:
                    optPermutation=(reflection, (compose(rp,baseRowPerm),compose(cp,baseColPerm)))
                    if optPermutationList==[] or copyMatrix>optMatrix:
                        assert(not (optPermutationList==[]) or(reflection==False 
                            and rp==IdPermutation
                            and cp==IdPermutation))
                        if optPermutationList and breakIfNotMaximal:
                            traceGlobalOptimization("global optimization: skip, not optimal", matrix)
                            return None
                        optMatrix=copyMatrix
                        optPermutationList=[optPermutation]
                    else: 
                        assert(copyMatrix==optMatrix)
                        optPermutationList.append(optPermutation)
    traceGlobalOptimization("global optimization: keep", matrix)
    return(optMatrix, optPermutationList)


def all01BlockRepresentatives(aBlockCount):
    '''
    purpose:
        create all optimal 3*3 matrices with aBlockCount fille with 1
        and the remaining with 0. OPtimal with regards to reflection,
        row and column permutation and the row by row ordering
    arguments:
        aBlockCount: 
            the number of blocks that should contain 1
    return:
        a list of pairs. the first element is a matrix, 
        tjhe second element is a list of stabilizers
    '''

    result=[]
    for matrix in generate01Matrices(3,3,aBlockCount):
        temp=optimizeBlock(matrix, True)
        if temp is not None:
            result.append(temp)
    return result

def fillBlocksWithWeights(matrix, weights):
    '''
    purpose:
        the ones in the matrix are replaced by the elements of the list weight.
        the number of elements of weights is equal to the number of ones of matrix
   arguments:
        matrix: a 0-1 matrix
        weights: a list of numbers
    return:
        a the matrix were the oens are replacers by the elements of weight
    '''
    assert(sum([sum(rows) for rows in matrix])==len(weights))
    w=list(reversed(weights)) 
    #[w.pop() if x==1 else 0 for x in ll]
    return tuple(tuple(w.pop() if item==1 else 0 for item in row) for row in matrix)

def fillSudokuWithSymbols(sudoku, symbols):
    mySymbols=list(reversed(symbols))
    return tuple(tuple(tuple(tuple(mySymbols.pop() if item==1 else 0 for item in row) for row in stack) for stack in band) for band in sudoku)

def countClues(aSudoku):
    return sum([sum([sum([sum(row) for row in stack]) for stack in band])  for band in aSudoku])


def generate01Matrices(rows, cols, count):
    '''
    Purpose:
        generates the list of all possible 0-1-matrices of dimension rows*cols  with exactly count ones
    Arguments:
        rows, cols: reows*cols is the dimension of the generated matricew
        count: the number of ones in the generated matrices
    Return Value:
        a list of all rows*cols-matrices with exactly "count" ones. 
        the remaining elements are zeros.
        the matrices are represented as tuple of tuples
    '''
    matrices=[]
    for ones in combinations(range(rows*cols), count):
        matrices.append(tuple(tuple(
            1 if (cols*row+col  in ones and row<rows and col<cols) else 0 for col in range(3)) 
                for row in range(3)))
    return matrices
    
def createEmptySudoku():
    return tuple([tuple([tuple([tuple([0]*3) 
        for r in range(3)])
            for s in range(3)])
                for b in range(3)])

def copyOfReplaced(sudoku,subblock ,band,stack):
    #newSudoku=deepcopy(sudoku)
    newSudoku=[[s for s in b] for b in sudoku]
    rows=len(subblock)
    cols=len(subblock[0])
    newSudoku[band][stack]=tuple([tuple([subblock[r][c] if r<rows and c<cols else 0 
        for c in range(3)]) 
            for r in range(3)])
    newSudoku=tuple([tuple([s for s in b]) for b in newSudoku])
    return newSudoku


def generate01SudokusFromWeights(weightMatrix):    
    sudoku=createEmptySudoku()
    stackWidth=[0]*3
    bandHeight=[0]*3
    currSolutions=[(sudoku, (bandHeight, stackWidth))]
    for b in range(3):
        for s in range(3):
            nextSolutions=[]
            
            for (sudoku, (bandHeight, stackWidth)) in currSolutions:
                if s==0:
                    prevBandHeight=0
                else:
                    prevBandHeight=bandHeight[b]
                if b==0:
                    prevStackWidth=0
                else:
                    prevStackWidth=stackWidth[s]

                if weightMatrix[b][s]==0:
                    nextBandHeight=bandHeight[:]
                    nextStackWidth=stackWidth[:]
                    nextBandHeight[b]=prevBandHeight
                    nextStackWidth[s]=prevStackWidth                   
                    nextSolutions.append((sudoku, (nextBandHeight, nextStackWidth,)))
                    continue
                     
                possibleBandHeight=min(prevBandHeight+weightMatrix[b][s],3)
                possibleStackWidth=min(prevStackWidth+weightMatrix[b][s],3)
                for subBlock in generate01Matrices(
                    possibleBandHeight,
                    possibleStackWidth,
                    weightMatrix[b][s]):
                    (rowperm,colperm), (rowmax, colmax), contigous = analyzeBlock(subBlock, prevBandHeight, prevStackWidth)
                    if contigous:
                        nextBandHeight=bandHeight[:]
                        nextStackWidth=stackWidth[:]
                        nextBandHeight[b]=max(bandHeight[b],rowmax)
                        nextStackWidth[s]=max(nextStackWidth[s],colmax)

                        nextSolutions.append((copyOfReplaced(sudoku,subBlock ,b,s), (nextBandHeight, nextStackWidth,)))
                       
            currSolutions=nextSolutions[:] 
    nextSolutions=[]
    for (cell01Matrix, (nextBandHeight, nextStackWidth,)) in currSolutions:
        result=localOptimize(cell01Matrix, breakIfNotMaximal=True)
        if result is not None:
            nextSolutions.append(result)
    return nextSolutions

def compactifyLocal(sudoku, breakIfNotMaximal=False):
    # construct permutations
    bandHeight=[0]*3
    stackWidth=[0]*3
    rowInBandPerm=[IdPermutation]*3
    colInStackPerm=[IdPermutation]*3
    for b in range(3):
        for s in range(3):
            if s==0:
                prevBandHeight=0
                prevRowInBandPerm=IdPermutation
            else:
                prevBandHeight=bandHeight[b]
                prevRowInBandPerm=rowInBandPerm[b]
            if b==0:
                prevStackWidth=0
                prevColInStackPerm=IdPermutation
            else:
                prevStackWidth=stackWidth[s]
                prevColInStackPerm=colInStackPerm[s]
            #preprocess for efficiency
            (rowperm,colperm), (rowcount, colcount), contigous = analyzeBlock(sudoku[b][s], prevBandHeight, prevStackWidth)
            if not contigous:
                if breakIfNotMaximal:
                    return None
                sudoku=permuteRowsInBand(sudoku, b, rowperm)
                sudoku=permuteColsInStack(sudoku, s, colperm)
            bandHeight[b]=min(prevBandHeight+rowcount,3)
            stackWidth[s]=min(prevStackWidth+colcount,3)
            rowInBandPerm[b]=compose(rowperm, prevRowInBandPerm)
            colInStackPerm[s]=compose(colperm, prevColInStackPerm)
    return (sudoku, (bandHeight, stackWidth), (rowInBandPerm, colInStackPerm))

def localOptimize(sudoku, breakIfNotMaximal=False):
    result= compactifyLocal(sudoku, breakIfNotMaximal)
    if result is None:
        return None
    (sudoku, (bandHeight, stackWidth), (rowInBandPerm, colInStackPerm)) = result
    currPermutations=None
    for r0 in permutations(range(bandHeight[0])):
        for r1 in permutations(range(bandHeight[1])):
            for r2 in permutations(range(bandHeight[2])):
                for c0 in permutations(range(stackWidth[0])):
                    for c1 in permutations(range(stackWidth[1])):
                        for c2 in permutations(range(stackWidth[2])):
                            tr0=list(r0)+list(range(bandHeight[0],3))
                            tr1=list(r1)+list(range(bandHeight[1],3))
                            tr2=list(r2)+list(range(bandHeight[2],3))
                            tc0=list(c0)+list(range(stackWidth[0],3))
                            tc1=list(c1)+list(range(stackWidth[1],3))
                            tc2=list(c2)+list(range(stackWidth[2],3))
                            perSod=sudoku
                            perSod=permuteRowsInBand(perSod,0,tr0)
                            perSod=permuteRowsInBand(perSod,1,tr1)
                            perSod=permuteRowsInBand(perSod,2,tr2)
                            perSod=permuteColsInStack(perSod,0,tc0)
                            perSod=permuteColsInStack(perSod,1,tc1)
                            perSod=permuteColsInStack(perSod,2,tc2)
                            if perSod>sudoku and breakIfNotMaximal:
                                return None
                            if currPermutations is not None and perSod<currOptimum:
                                continue
                            localPermutation=(((0,tr0),(1,tr1),(2,tr2)),
                                ((0,tc0),(1,tc1),(2,tc2)))
                            if currPermutations is None:
                                currPermutations=[localPermutation]
                                currOptimum=perSod
                            elif perSod==currOptimum:
                                currPermutations.append(localPermutation)
                            elif perSod>currOptimum:
                                currPermutations=[localPermutation]
                                currOptimum=perSod
    return (currOptimum, currPermutations)                           
    
def allWeightBlockRepresentatives(aClueCount):
    mySolutions=[]
    for myBlockCount in range(2,aClueCount+1):
        myPartitions=allPartitions(aClueCount, myBlockCount)
        for (myMatrix, myGlobalStabilizers) in all01BlockRepresentatives(myBlockCount):
            for myWeights in myPartitions:
                myWeightMatrix=fillBlocksWithWeights(myMatrix, myWeights)
                # check if minimal
                myWeightIsOptimal=True
                myWeightMatrixStabilizers =[]
                for myStabilizer in myGlobalStabilizers:
                    myEquivalentMatrix=applyBlockFunction(myStabilizer, myWeightMatrix)
                    if myEquivalentMatrix <myWeightMatrix:
                        #  myWeightMatrix not optimal
                        myWeightIsOptimal=False
                        break
                    elif myEquivalentMatrix==myWeightMatrix:
                        # this is a stabilizer
                        myWeightMatrixStabilizers.append(myStabilizer)
                if myWeightIsOptimal:
                    mySolutions.append((myWeightMatrix,myWeightMatrixStabilizers))
                # otherwise skip
    return mySolutions

def all01CellRepresentatives(aClueCount):
    mySolutions=[]
    for (myWeightMatrix,myWeightMatrixStabilizers) in allWeightBlockRepresentatives(aClueCount):
        for (myCell01Matrix, myLocalStabilizers) in generate01SudokusFromWeights(myWeightMatrix):
            myStabilizers=[]
            for myGlobalStabilizer in myWeightMatrixStabilizers:
                myCell01MatrixNotMaximal=False
                myMatrixVariant=applyGlobalTransform(myGlobalStabilizer, myCell01Matrix)
                result=localOptimize(myMatrixVariant)
                if result[0] >myCell01Matrix:
                    myCell01MatrixNotMaximal=True
                    break
                if result[0]==myCell01Matrix:
                    for myLocalStabilizer in result[1]:
                        myStabilizers.append((myGlobalStabilizer,myLocalStabilizer))
            if myCell01MatrixNotMaximal:
                break
            mySolutions.append((myCell01Matrix,myStabilizers))
    return mySolutions
            
def allSymbolCellRepresentatives(aClueCount):
    RgsList=restrictedGrowthSequence(aClueCount)
    mySolutions=[]
    for (myCell01Matrix,myPrevStabilizers) in all01CellRepresentatives(aClueCount):
        for symbols in RgsList:
            mySudoku=fillSudokuWithSymbols(myCell01Matrix,symbols)
            if isValidSudoku(mySudoku):
                result=filterOptimum(mySudoku, myPrevStabilizers, applyFullTransform, isBetterBySymbols)
                if result is not None:
                    mySolutions.append(result[0])
    return mySolutions
    

def applyGlobalTransform(aNineFunction, aNineMatrix):
    (reflection, (rowPermute, colPermute))=aNineFunction
    if reflection:
        aNineMatrix=transposeMatrix(aNineMatrix)
    aNineMatrix=permuteBands(aNineMatrix,rowPermute)
    aNineMatrix=permuteStacks(aNineMatrix, colPermute)
    return aNineMatrix
    
def applyLocalTransform(aFunction, aSudoku):
    (myRowInBandTransformations, myColInStackTransformations)=aFunction
    for myBand, myPerm in myRowInBandTransformations:
        aSudoku=permuteRowsInBand(aSudoku, myBand, myPerm)
    for myStack, myPerm in myColInStackTransformations:
        aSudoku=permuteColsInStack(aSudoku, myBand, myPerm)
    return aSudoku
    pass
        
        
def applyFullTransform(aFunction, aSudoku):
    myGlobalTransformedSudoku=applyGlobalTransform(aFunction[0], aSudoku)
    myFullTransformedSudoku=applyLocalTransform(aFunction[1], myGlobalTransformedSudoku)
    return myFullTransformedSudoku
    
   
def filterOptimum(aArgument, aFunctions, aApplyFunction, aIsBetter):
    myItemStabilisators=[]
    for myFunction in aFunctions:
        myImage=aApplyFunction(myFunction,aArgument)
        myCompareResult = aIsBetter(myImage,aArgument)
        if myCompareResult==0:
            myItemStabilisators.append(myFunction)
        elif myCompareResult==1:
            # aArgument is not an optimum in its equivalency class
            return None
        # else: skip this function, it is not a stabilisator and its image is not an optimum
    return (aArgument,myItemStabilisators)

def applyBlockFunction(aBlockFunction, aBlock):
    (reflection, (rowPermute, colPermute))=aBlockFunction
    if reflection:
        aBlock=transposeMatrix(aBlock)
    aBlock=permuteBands(aBlock,rowPermute)
    aBlock=permuteStacks(aBlock, colPermute)
    return aBlock
    
def isBetterBlockwise(aBlock1, aBlock2):
    if aBlock1>aBlock2:
        return 1
    elif  aBlock1<aBlock2:
        return -1
    else:
        return 0

def retrieveSymbols(aSudoku):
    return [[[[item  for item in row if item>0] for row in stack] for stack in band] for band in aSudoku]

def isBetterBySymbols(aSudoku1, aSudoku2):
    mySymbols1=retrieveSymbols(aSudoku1)
    mySymbols2=retrieveSymbols(aSudoku2)
    if mySymbols1< mySymbols2:
        return 1
    elif mySymbols1>mySymbols2:
        return -1
    else:
        return 0
    
    
    
    
    
def allPartitions(aSum: int, aCount: int):
    """
    purpose:
        Return all ordered partitions (compositions) of aSum into aCount
        positive integers
    arguments:
        aSum: the number the summand should sum to 
        aCount: the number of summands
    return:
        a list of tuples of summands
    
    Example:
        allPartitions(4, 2) → [(1,3), (2,2), (3,1)]
    """
    results = []

    def backtrack(remaining_sum, remaining_count, prefix):
        # If no summands left to place:
        if remaining_count == 0:
            if remaining_sum == 0:
                results.append(tuple(prefix))
            return

        # Each summand must be at least 1
        # Max value allowed is remaining_sum - (remaining_count-1)
        # so that the remaining summands can each be at least 1
        min_val = 1
        max_val = remaining_sum - (remaining_count - 1)

        if max_val < min_val:
            return

        for v in range(min_val, max_val + 1):
            backtrack(remaining_sum - v, remaining_count - 1, prefix + [v])

    backtrack(aSum, aCount, [])
    return results


def restrictedGrowthSequence(n: int):
    """
    Generate all restricted growth sequences (RGS) of length n.

    A restricted growth sequence is a sequence a[0..n-1] such that:
      - a[0] == 1
      - a[i] <= 1 + max(a[0..i-1]) for all i >= 1
    """
    if n <= 0:
        return []

    result = []

    def backtrack(seq, current_max):
        if len(seq) == n:
            result.append(seq[:])
            return

        # Next value can range from 1 to current_max + 1
        for v in range(1, current_max + 2):
            seq.append(v)
            backtrack(seq, max(current_max, v))
            seq.pop()

    backtrack([1], 1)
    return result
    
 

def convertMatrixToSudoku(matrix):
    return tuple([tuple([tuple([tuple([matrix[3*b+r][3*s+c] 
        for c in range(3)]) 
            for r in range(3)]) 
                for s in range(3)]) 
                    for b in range(3)])

        
if __name__=="__main__":
    import test_simple_sudoku     
    

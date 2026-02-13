'''
python file: simple_sudoku_07, open
12.2.2026

'''
from copy import deepcopy
from header_sudoku import *
from itertools import combinations, permutations, product, chain
from tools_sudoku import *
from extensions_sudoku import *


try:
    from trace_sudoku import *
except ModuleNotFoundError:
    def trace(trcMessage, trcObject, note=None, step=None):
        pass


globalTraceId=1
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

IdLocalTransformation=(((0,1,2),(0,1,2),(0,1,2),),((0,1,2),(0,1,2),(0,1,2),))
IdGlobalTransformation=(False,((0,1,2),(0,1,2)))
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


def optimizeBlock1(matrix, breakIfNotMaximal=False):
    '''
    purpose:
        retrieves the maximal matrix from the equivalence class of the 
        input matrix. Two matrices are equivalemnt if they can be transformed 
        one to the other by reflection, row or column permutation.
        The subroutine can also be used to decide if a matrix is the 
        maximal element of its equivalence class.
    arguments:
        matrix: a 3*3 0-1 matrix that should be optimized
        breakIfNotMaximal: default False. If set to True, the function  returns None as soon as a larger matrix is derived from matrix because then matrix is not a maximum.
    return:
        None | (optMatrix, [stabilizer  ...])
        stabilizer = (reflection, (row permutation, column permutation))
        reflection is a boolean value. True means matrix transposition, False means idendity
        the permutations are 3 element lists of permutations of the numbers 0,1,2
        optMatrix is the maximal element
    '''
    assert(type(matrix)==type(tuple()))
    (rowperm,colperm), (rowmax, colmax), contigous = analyzeBlock(matrix)
    if not contigous:
        if breakIfNotMaximal:
            trace(trcMsgReject, matrix)
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
                            trace(trcMsgReject, matrix)
                            return None
                        optMatrix=copyMatrix
                        optPermutationList=[optPermutation]
                    else: 
                        assert(copyMatrix==optMatrix)
                        optPermutationList.append(optPermutation)
    trace(trcMsgAccept, matrix)
    return(optMatrix, optPermutationList)


def optimizeBlock2(aMatrix, breakIfNotMaximal=False):
    '''
    purpose:
        retrieves the maximal matrix from the equivalence class of the 
        input matrix. Two matrices are equivalemnt if they can be transformed 
        one to the other by reflectio, row or column permutation.
        The subroutine can also be used to decide if a matrix is the 
        maximal element of its equivalence class.
    arguments:
        aMatrix: a 3*3 0-1 matrix that should be optimized
        breakIfNotMaximal: default False. If set to True, the function  returns None as soon as a larger matrix is derived from aMatrix because then aMatrix is not a maximum.
    return:
        None | (optMatrix, [stabilizer  ...])
        stabilizer = (reflection (row permutation, column permutation))
        reflection is a boolean value. True means matrix transposition, False means idendity
        the permutations are 3 element lists of permutations of the numbers 0,1,2
        optMatrix is the maximal element
    '''
    myGlobalFunctions=[]
    assert(type(aMatrix)==type(tuple()))
    (rowperm, colperm), (rowmax, colmax), contigous = analyzeBlock(aMatrix)
    
    # preprocessing
    if not contigous:
        if breakIfNotMaximal:
            trace(trcMsgReject, aMatrix)
            return None
        myMatrixCont=permuteBands(aMatrix, rowperm)
        myMatrixCont=permuteStacks(aMatrix, colperm)
    else:
        myMatrixCont=aMatrix

    assert(type(aMatrix)==type(tuple()))
    myGlobalFunctions=[]
    for reflection in [False, True]:
        for myRowPermutation in AllPermutations[rowmax]:
            for myColPermutation in AllPermutations[colmax]:
                myGlobalFunctions.append((reflection,(myRowPermutation, myColPermutation)))
                        
    result=filterGlobalOptimum(myMatrixCont, myGlobalFunctions, applyGlobalTransform, isGreater01Block)
    if result is not None:
        # post processing
        optimizingTransformations=[]
        for (reflection,(myRowPermutation, myColPermutation)) in result[1]:
            if reflection:
                baseRowPerm, baseColPerm = colperm, rowperm
            else:
                baseRowPerm, baseColPerm = rowperm, colperm
            optimizingTransformations.append((reflection, 
                (compose(myRowPermutation,baseRowPerm),
                compose(myColPermutation,baseColPerm))))
        trace(trcMsgAccept, aMatrix)
        return(aMatrix, optimizingTransformations)
    else:
        trace(trcMsgReject, aMatrix)
    
def optimizeBlock3(aMatrix, breakIfNotMaximal=False):
    '''
    purpose:
        retrieves the maximal matrix from the equivalence class of the 
        input matrix. Two matrices are equivalemnt if they can be transformed 
        one to the other by reflectio, row or column permutation.
        The subroutine can also be used to decide if a matrix is the 
        maximal element of its equivalence class.
    arguments:
        aMatrix: a 3*3 0-1 matrix that should be optimized
        breakIfNotMaximal: default False. If set to True, the function  returns None as soon as a larger matrix is derived from aMatrix because then aMatrix is not a maximum.
    return:
        None | (optMatrix, [stabilizer  ...])
        stabilizer = (reflection (row permutation, column permutation))
        reflection is a boolean value. True means matrix transposition, False means idendity
        the permutations are 3 element lists of permutations of the numbers 0,1,2
        optMatrix is the maximal element
    '''
    assert(type(aMatrix)==type(tuple()))
    (rowperm,colperm), (rowmax, colmax), contigous = analyzeBlock(aMatrix)
    if not contigous:
        if breakIfNotMaximal:
            trace(trcMsgReject, aMatrix)
            return None
        myMatrixCont=permuteBands(aMatrix, rowperm)
        myMatrixCont=permuteStacks(aMatrix, colperm)
    else:
        myMatrixCont=aMatrix

    assert(type(aMatrix)==type(tuple()))
    optPermutationList=[]
    for reflection in [False, True]:
        if reflection:
            myMatrixRefl=transposeMatrix(myMatrixCont)
            baseRowPerm,baseColPerm=colperm,rowperm
        else:
            myMatrixRefl=myMatrixCont
            baseRowPerm,baseColPerm=rowperm,colperm
        for rp in AllPermutations[rowmax]:
            myMatrixRow=permuteBands(myMatrixRefl,rp)
            for cp in AllPermutations[colmax]:
                myMatrixCol=permuteStacks(myMatrixRow,cp)
                if optPermutationList==[] or myMatrixCol>=optMatrix:
                    optPermutation=(((reflection, (compose(rp,baseRowPerm),compose(cp,baseColPerm))),IdLocalTransformation))
                    if optPermutationList==[] or myMatrixCol>optMatrix:
                        assert(not (optPermutationList==[]) or(reflection==False 
                            and rp==IdPermutation
                            and cp==IdPermutation))
                        if optPermutationList and breakIfNotMaximal:
                            trace(trcMsgReject, aMatrix)
                            return None
                        optMatrix=myMatrixCol
                        optPermutationList=[optPermutation]
                    else: 
                        assert(myMatrixCol==optMatrix)
                        optPermutationList.append(optPermutation)
                        
    trace(trcMsgAccept, aMatrix)
    return(optMatrix, optPermutationList)
    
    
def optimizeBlock4(aMatrix, breakIfNotMaximal=False):
    # initialize
    myStartGrid=deepcopy(aMatrix)
    #rowCount=3
    #colCount=3
    
    myTranposedGrid=transposeMatrix(aMatrix)
    if breakIfNotMaximal:
        if myTranposedGrid>myStartGrid:
            return None
    myNextSolutions=[
        (myStartGrid,(False, (IdPermutation, IdPermutation))), 
        (myTranposedGrid, (True, (IdPermutation, IdPermutation)))]
    myNextMinFreeCol=0
    myGridAlreadyStored=False
    for r in list(range(3)):
        myNextRowIsUsed=False
        for c in list(range(3)):
            # all coefficient of the matrices in currSolutions  
            # from position (0,0) until the predecessor of (r,c) match
            # 
            # all coefficient of the matrices in myNextSolutions  
            # from position (0,0) until (r,c) match
            # 
            myCurrOpt=0
            myCurrSolutions=myNextSolutions
            myNextSolutions=[]
            
            # all coefficients in positions (i, j), i<r, j>=myMinFreeCol are = 0
            # (all coefficients in positions (r,j), j<c are 0 ) <==> myNextRowIsUsed
            myMinFreeCol=myNextMinFreeCol
            myRowIsUsed=myNextRowIsUsed
            for (myGrid, myGlobalTransformation) in myCurrSolutions:
                # try to increase the coefficient at position (r,c) by permuting rows  and columns 
                # without touching the columns < myMinFreeCol and without touching the rows < r. 
                # The row r can be touched, if now element left of column c of tow r = 1.
                
                myGridAlreadyStored=False
                for i in list(range(r,3)):
                    for j in list(range(3)):
                        myCanSwap=(((r==i) 
                            or ((r<i) and not myRowIsUsed)) 
                            and ((c==j) or ((myMinFreeCol==c) and (c<j))))
                        myCanSwap=(((r==i) and (c==j)) 
                            or (( c>=myMinFreeCol) and  (j>= myMinFreeCol and i == r) ) 
                            or ((not myRowIsUsed and c<myMinFreeCol)  and (i>r and c==j)) 
                            or ((not myRowIsUsed and c>=myMinFreeCol) and ( (j>=myMinFreeCol and i>r) )) )
                        if myCanSwap and myGrid[i][j]>=myCurrOpt:
                            if myGrid[i][j]>0:
                                myGridAlreadyStored=False
                                myGridCopy=deepcopy(myGrid)
                                myNextRowIsUsed=True
                                assert(c<=myMinFreeCol)
                                rowPerm=list(range(3))
                                rowPerm[i],rowPerm[r]=rowPerm[r],rowPerm[i]
                                colPerm=list(range(3))
                                colPerm[j],colPerm[c]=colPerm[c],colPerm[j]
                                myGridCopy=permuteBands(myGrid,rowPerm)
                                myGridCopy=permuteStacks(myGridCopy, colPerm)
                                if breakIfNotMaximal:
                                    if myGridCopy>myStartGrid:
                                        return None
                                myNewGlobalTransformation=composeBlockFunctions(
                                    (False, ( rowPerm, colPerm)),
                                    myGlobalTransformation)
                                myNextMinFreeCol=max(myMinFreeCol,c+1)
                            if myGrid[i][j]>myCurrOpt:
                                myCurrOpt=myGrid[i][j]
                                myNextSolutions=[(myGridCopy,myNewGlobalTransformation)]

                            elif myGrid[i][j]> 0:
                                myNextSolutions.append((myGridCopy,myNewGlobalTransformation))
                            elif not myGridAlreadyStored:
                            #else:
                                myGridCopy=deepcopy(myGrid)
                                myNextSolutions.append((myGridCopy,myGlobalTransformation))
                                myGridAlreadyStored=True
    myTransformList=[]
    for (myMatrix, myTranform) in myNextSolutions:
        assert(myMatrix==myNextSolutions[0][0])
        myTransformList.append(myTranform)
    return (myNextSolutions[0][0], myTransformList)

    
optimizeBlock=optimizeBlock3

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
            trace(trcMsgAccept, matrix)
            result.append(temp)
        else: 
            trace(trcMsgReject, matrix)
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
    trace(trcMsgInfo, weightMatrix, 'entry')
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

                trace(trcMsgInfo, sudoku)
                trace(trcMsgInfo,"(b, s), (height, width): (%d, %d), (%d, %d)"%
                        (b,s,possibleBandHeight, possibleStackWidth))
                for subBlock in generate01Matrices(
                    possibleBandHeight,
                    possibleStackWidth,
                    weightMatrix[b][s]):
                    (rowperm,colperm), (rowmax, colmax), contigous = analyzeBlock(subBlock, prevBandHeight, prevStackWidth)
                    myAugmentedSudoku=copyOfReplaced(sudoku,subBlock ,b,s)
                    if contigous:
                        nextBandHeight=bandHeight[:]
                        nextStackWidth=stackWidth[:]
                        nextBandHeight[b]=max(nextBandHeight[b],prevBandHeight+rowmax)
                        nextStackWidth[s]=max(nextStackWidth[s],prevStackWidth+colmax)
                        
                        nextSolutions.append((myAugmentedSudoku, (nextBandHeight, nextStackWidth,)))
                        trace(trcMsgAccept, myAugmentedSudoku)
                    else:
                        trace(trcMsgReject, myAugmentedSudoku)
            currSolutions=nextSolutions[:] 
    trace(trcMsgInfo,"filter out non optimal")
    nextSolutions=[]
    for (cell01Matrix, (nextBandHeight, nextStackWidth,)) in currSolutions:
        result=localOptimize(cell01Matrix, breakIfNotMaximal=True)
        if result is not None:
            trace(trcMsgAccept, cell01Matrix)
            trace(trcMsgInfo, result[1])
            nextSolutions.append(result)
        else:
            trace(trcMsgReject, cell01Matrix)
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
                            localPermutation=((tr0,tr1,tr2),(tc0,tc1,tc2))
                            myGlobalTransformation=(IdGlobalTransformation,localPermutation)
                            if currPermutations is None:
                                currPermutations=[myGlobalTransformation]
                                currOptimum=perSod
                            elif perSod==currOptimum:
                                currPermutations.append(myGlobalTransformation)
                            elif perSod>currOptimum:
                                currPermutations=[myGlobalTransformation]
                                currOptimum=perSod
    return (currOptimum, currPermutations)                           
    
def allWeightBlockRepresentatives(aClueCount):
    mySolutions=[]
    for myBlockCount in range(2,aClueCount+1):
        myPartitions=allPartitions(aClueCount, myBlockCount)
        for (myMatrix, myTransform) in all01BlockRepresentatives(myBlockCount):
            for myWeights in myPartitions:
                myWeightMatrix=fillBlocksWithWeights(myMatrix, myWeights)
                filterResult=filterGlobalOptimum(myWeightMatrix, myTransform, applyBlockFunction, isLessWeightBlock)
                if filterResult is not None:
                    trace(trcMsgAccept, myWeightMatrix)
                    mySolutions.append(filterResult)
                # otherwise skip
                else:
                    trace(trcMsgReject, myWeightMatrix)
    return mySolutions
    
def all01CellRepresentatives_old(aClueCount):
    mySolutions=[]
    for (myWeightMatrix,myWeightMatrixStabilizers) in allWeightBlockRepresentatives(aClueCount):
        trace(trcMsgInfo, myWeightMatrixStabilizers,step=1)
        for (myCell01Matrix, _) in generate01SudokusFromWeights(myWeightMatrix):
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
                        myStabilizers.append((myGlobalStabilizer[0],myLocalStabilizer[1]))
            if myCell01MatrixNotMaximal:
                continue
            #assert(myStabilizers!=[])
            mySolutions.append((myCell01Matrix,myStabilizers))
    return mySolutions
            
def all01CellRepresentatives(aClueCount):
    mySolutions=[]
    for (myWeightMatrix,myWeightMatrixStabilizers) in allWeightBlockRepresentatives(aClueCount):
        for (myCell01Matrix, _) in generate01SudokusFromWeights(myWeightMatrix):
            result=filterGlobalOptimum(myCell01Matrix, myWeightMatrixStabilizers, 
            applyGlobalTransform, isGreater01Cell, optimizeLocal=True)
            if result is not None:
                mySolutions.append(result)
    return mySolutions            
            
def allSymbolCellRepresentatives(aClueCount):
    RgsList=restrictedGrowthSequence(aClueCount)
    trace(trcMsgInfo, "symbols %s"%RgsList)
    mySolutions=[]
    for (myCell01Matrix,myPrevStabilizers) in all01CellRepresentatives(aClueCount):
        trace(trcMsgInfo,myCell01Matrix)
        #print("debug", myCell01Matrix,myPrevStabilizers) 
        trace(trcMsgInfo, myPrevStabilizers) # here the error is raised
        for symbols in RgsList:
            mySudoku=fillSudokuWithSymbols(myCell01Matrix,symbols)
            if isValidSudoku(mySudoku):
                result=filterGlobalOptimum(mySudoku, myPrevStabilizers, applyFullTransform, isLessSymbolCell)
                if result is not None:
                    trace(trcMsgAccept, mySudoku)
                    mySolutions.append(result[0])
                else:
                    trace(trcMsgReject, mySudoku)
                    pass
            else:
                trace(trcMsgReject, mySudoku, "invalid sudoku")
                pass
    return mySolutions
    

def testGet01CellRepresentatives(aWeightBlock):
    mySolutions=[]
    (myWeightMatrix,myWeightMatrixStabilizers)=optimizeBlock1(aWeightBlock, breakIfNotMaximal=False)
    my01SudokuList=generate01SudokusFromWeights(myWeightMatrix)
    for (myCell01Matrix, myLocalStabilizers) in my01SudokuList:
        myStabilizers=[]
        trace(trcMsgInfo, myCell01Matrix, "0-1-Sudoku")
        for myGlobalStabilizer in myWeightMatrixStabilizers:
            myCell01MatrixNotMaximal=False
            myMatrixVariant=applyGlobalTransform(myGlobalStabilizer, myCell01Matrix)
            trace(trcMsgInfo, [myGlobalStabilizer], "stabilizer ")
            trace(trcMsgInfo, myMatrixVariant)
            result=localOptimize(myMatrixVariant)
            if result[0] >myCell01Matrix:
                myCell01MatrixNotMaximal=True
                # skip this config one, because it is not an optiomum
                break
            if result[0]==myCell01Matrix:
                for myLocalStabilizer in result[1]:
                    myStabilizers.append((myGlobalStabilizer[0],myLocalStabilizer[1]))
            else:
                trace(trcMsgReject, result[0])

        if myCell01MatrixNotMaximal:
            trace(trcMsgReject, result[0])
            continue
        mySolutions.append((myCell01Matrix,myStabilizers))
        trace(trcMsgAccept, result[0])
    return mySolutions    

def applyGlobalTransform(aTransform, aMatrix):
    (reflection, (bandPermute, stackPermute))=aTransform[0]
    if reflection:
        aMatrix=transposeMatrix(aMatrix)
    aMatrix=permuteBands(aMatrix,bandPermute)
    aMatrix=permuteStacks(aMatrix, stackPermute)
    return aMatrix
    
def applyLocalTransform(aFunction, aSudoku):
    '''
    (myRowInBandTransformations, myColInStackTransformations)=aFunction
    for myBand, myPerm in myRowInBandTransformations:
        aSudoku=permuteRowsInBand(aSudoku, myBand, myPerm)
    for myStack, myPerm in myColInStackTransformations:
        aSudoku=permuteColsInStack(aSudoku, myBand, myPerm)
    return aSudoku
    pass
    '''
    (_, (myRowInBandTransformations, myColInStackTransformations))=aFunction
    for myBand, myPerm in enumerate(myRowInBandTransformations):
        aSudoku=permuteRowsInBand(aSudoku, myBand, myPerm)
    for myStack, myPerm in enumerate(myColInStackTransformations):
        aSudoku=permuteColsInStack(aSudoku, myBand, myPerm)
    return aSudoku

        

def restrictedGrowthSequencePermutation(aSequence):
    '''
    purpose: 
        replace the values of the symbols 
        so that the resulting sequence is a 
        restricted groth sequence         
    '''
    mySymbolPermutation=[0]*10
    # myPermutation[0]==0 for technical reasons
    # this value is not used
    # the myNewSequence is not used, we keep it to document its creation

    nextValue=1
    myNewSequence=[]
    for value in aSequence:
        if mySymbolPermutation[value]==0:
            mySymbolPermutation[value]=nextValue
            nextValue+=1
        myNewSequence.append(mySymbolPermutation[value])
    for i in range(10):
        if i==0:
            mySymbolPermutation[i]=i
    return mySymbolPermutation

def retrieveSymbolSequence(aSudoku):
    # retrive the sysmbosl of the sudoku in the 
    # order from top left to bottom right, 
    # band by band and for each block row by row
    return [col
        for band in aSudoku
            for stack in band
                for row in stack
                    for col in row
                        if col!=0]

def replaceSymbols(aSudoku, aSymbolPermutation):
    return tuple([tuple([tuple([tuple([aSymbolPermutation[col]
        for col in row])
            for row in stack])
                for stack in band])
                    for band in aSudoku])
    

def applyFullTransform(aFunction, aSudoku):
    myGlobalTransformedSudoku=applyGlobalTransform(aFunction, aSudoku)
    myFullTransformedSudoku=applyLocalTransform(aFunction, myGlobalTransformedSudoku)
    mySequence=retrieveSymbolSequence(myFullTransformedSudoku)
    mySymbolPermutation=restrictedGrowthSequencePermutation(mySequence)
    mySymbolTransformedSudoku=replaceSymbols(myFullTransformedSudoku,mySymbolPermutation)
    return mySymbolTransformedSudoku
    


def filterOptimum(aArgument, aFunctions, aApplyFunction, aIsBetter, breakIfNotOptimum=True):
    '''
    purpose: 
        to applies transformations from a lidt to a grid 
        and to find out which transformations transforms the grid to the best one
    input:
        aArgument:
            the grid
        aFunctions:
            the list of transformations
        aApplyFunctions:
            a function that applies a transformation to the object
            aApplyFunction(myFunction,aGrid} -> set of grids 
            aFunction: set of grids -> set of grids
            aGrids is from set of grids
        aIsBetter:
            a function that decides which of the two grids is the better one
            aIsBetter(grid1,grid2) -> {-1, 0, +1}
                +1 if grid1 is better
                -1 if grid2 is better
                0 if both are equally good, but then they are identical (total order)
        breakIfNotOptimum: bool
            if True, stops, as soon as a transformation is found that mps the given grid to a better one. In this case 'None' is returned because "grid" is not the best.
    result:
        a pair is returned: the first component ist the best grid, the second is a list o the transformations that transform the argument grid to this best gfrid. If the argument grid ist already a maximum, the returned list of transformations are the  stabilisators from "aFunctions"
    '''
    myItemStabilisators=[]
    myCurrentOptimum=aArgument
    for myFunction in aFunctions:
        myImage=aApplyFunction(myFunction,aArgument)
        myCompareResult = aIsBetter(myImage,myCurrentOptimum)
        if myCompareResult==0:
            myItemStabilisators.append(myFunction)
        elif myCompareResult==1: 
            # myImage is better ,so aArgument is not 
            # an optimum in its equivalency class
            if breakIfNotOptimum:
                return None
            else:
                myCurrentOptimum=myImage
                myItemStabilisators=[myFunction]
        # else: skip this function, it is not a stabilisator and its image is not an optimum

    return (myCurrentOptimum,myItemStabilisators)

def filterGlobalOptimum(aArgument, aBlockTransforms, aApplyFunction, 
    aIsBetter, optimizeLocal=False, breakIfNotOptimum=True): # debug: breakIfNotOptimum= ist richtig
    # input:
    #   a 01 Sudoku and a list of block tansforms that map the 
    
    myCurrTransforms=[]
    myCurrentOptimum=aArgument
    trace(trcMsgInfo, aArgument,'enter')
    trace(trcMsgInfo, aBlockTransforms)
    for myBlockTransform in aBlockTransforms:
        myGlobalVariant=aApplyFunction(myBlockTransform, aArgument)
        trace(trcMsgInfo, myGlobalVariant)
        if optimizeLocal:
            (myImage, myLocalTansforms)=localOptimize(myGlobalVariant)
        else:
            myImage=myGlobalVariant
            myLocalTansforms=[(IdGlobalTransformation, IdLocalTransformation)]
        trace(trcMsgInfo, myImage)        
        trace(trcMsgInfo,myLocalTansforms,step=3)
        myCompareResult = aIsBetter(myImage,myCurrentOptimum)
        if myCompareResult == 0:
            myCurrTransforms.extend([(myBlockTransform[0], t[1]) for t in myLocalTansforms])
        elif myCompareResult == 1:
            if breakIfNotOptimum:
                return None
            myCurrentOptimum=myImage
            myCurrTransforms=[(myBlockTransform[0], t[1]) for t in myLocalTansforms]
    trace(trcMsgInfo, myCurrentOptimum,'return')        
    trace(trcMsgInfo,myCurrTransforms)   
    return(myCurrentOptimum, myCurrTransforms)


def applyBlockFunction(aBlockFunction, aBlock):
    (reflection, (rowPermute, colPermute))=aBlockFunction[0]
    if reflection:
        aBlock=transposeMatrix(aBlock)
    aBlock=permuteBands(aBlock,rowPermute)
    aBlock=permuteStacks(aBlock, colPermute)
    return aBlock

def composeBlockFunctions(aLeft, aRight):
    (leftReflex,(leftRowPerm, leftColPerm))=aLeft
    (rightReflex,(rightRowPerm,rightColPerm))=aRight
    # return (leftReflex!=rightReflex, (
        # compose(leftRowPerm, rightColPerm if leftReflex else rightRowPerm),
        # compose(leftColPerm, rightRowPerm if leftReflex else rightColPerm)))
        
    # return (rightReflex!=leftReflex, (
        # compose(rightRowPerm, leftColPerm if rightReflex else leftRowPerm),
        # compose(rightColPerm, leftRowPerm if rightReflex else leftColPerm)))
        
    return (leftReflex!=rightReflex, (
        compose(rightColPerm if leftReflex else rightRowPerm, leftRowPerm),
        compose(rightRowPerm if leftReflex else rightColPerm,leftColPerm)))
        
       
def isGreater01Block(aBlock1, aBlock2):
    # we only compare 01-matrices where the numnber of 1s is the same
    assert (sum([col for row in aBlock1 for col in row]) == sum([col for row in aBlock2 for col in row]))
    if aBlock1>aBlock2:
        return 1
    elif  aBlock1<aBlock2:
        return -1
    else:
        return 0   

def isGreater01Cell(aSudoku1, aSudoku2):
    # we only compare 01-matrices where the numnber of 1s is the same
    assert (sum([col for band in aSudoku1 for stack in band for row in stack for col in row]) 
    == sum([col for band in aSudoku2 for stack in band for row in stack for col in row]))
    if aSudoku1>aSudoku2:
        return 1
    elif  aSudoku1<aSudoku2:
        return -1
    else:
        return 0   
def isLessWeightBlock(aBlock1, aBlock2):
    # we only compare Weight matrices where the positions of the weight !=0 is the same
    assert ([1 if col else 0 for row in aBlock1 for col in row] == [1 if col else 0 for row in aBlock2 for col in row])
    # we will change the comparison meaning of block1 and block2 later
    if aBlock1>aBlock2:
        return 1
    elif  aBlock1<aBlock2:
        return -1
    else:
        return 0

def sudokuToList(aSudoku):
    return [col for band in aSudoku for stack in band for row in stack for col in row]
# def isBetterBySymbols(aSudoku1, aSudoku2):
    # myList1=[col for band in aSudoku1 for stack in band for row in stack for col in row]
    # if aSudoku1<aSudoku2:
        # return 1
    # elif aSudoku1>aSudoku2:
        # return -1
    # else:
        # return 0
    
 
def isLessSymbolCell(aSudoku1, aSudoku2):
    if sudokuToList(aSudoku1)<sudokuToList(aSudoku2):
        return 1
    elif sudokuToList(aSudoku1)>sudokuToList(aSudoku2):
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

####################################################################################

def putRowInconsistent(aSymbol, aRowBounds, aInconsistencyMatrix):
    for r in range(len(aRowBounds)):
        if aSymbol in aRowBounds[r]:
            for c in range(3):
                aInconsistencyMatrix[r][c]|=aSymbol
    return None
    
def putColInconsistent(aSymbol, aColBounds, aInconsistencyMatrix):
    for r in range(len(aColBounds)):
        if aSymbol in aColBounds[r]:
            for c in range(3):
                aInconsistencyMatrix[r][c]|=aSymbol
    return None
    
def putInconsistent(aSymbol, aRowBounds, aColBounds, aInconsistencyMatrix):
    putRowInconsistent(aSymbol, aRowBounds, aInconsistencyMatrix)
    putColInconsistent(aSymbol, aColBounds, aInconsistencyMatrix)
    return None

def creatInconsistencyMatrix(aSymbol, aRowBounds, aColBounds):
    return [[set()]*3 for _ in range (3)]
    
def isColBoundariesConsistent(aBlock, aColBounds):
    for c in range(len(aColBounds)):
        for r in range(3):
            if aBlock[r][c] in aColBounds[c]:
                return False
    return True   

def isRowBoundaryConsistent(aBlock, aRowBounds):
    raise NotImplementedError("'isRowBoundaryConsistent' habe ich nicht implementiert")

def isBoundaryConsistent(aBlock, aRowBounds,  aColBounds):
    return  (isRowBoundaryConsistent(aBlock, aRowBounds)
        and isColBoundariesConsistent(aBlock, aColBounds))
    
def verifyConfigWithBoundaries(rowBounds, colBounds, otherBounds):
    assert(len(rowBounds)<=3)
    assert(len(colBounds)<=3)
    
    myAllBounds=set([])
    #print([s for t in [rowBounds, colBounds, otherBounds] for s in t ])
    myAllBounds=otherBounds|set([e for t in [rowBounds, colBounds] for s in t for e in s])
    myAllSymbols=set(myAllBounds)
        
    maxBoundarySymbol=max(myAllSymbols)
    assert(len(myAllSymbols)<=4)
    assert(myAllSymbols==set(range(1,maxBoundarySymbol+1)))
    #freeRows=list(range(len(rowBounds),3))
    #freeCols=list(range(len(colBounds),3))
    freeRows=len(rowBounds)
    freeCols=len(colBounds)
    for myMatrix in generate01Matrices(3,3,maxBoundarySymbol):
        for myAssignement in permutations(list(range(1,maxBoundarySymbol+1))):
            myBlock=fillBlocksWithWeights(myMatrix, myAssignement)
            if not isBoundaryConsistent(myBlock, rowBounds, colBounds):
                continue
                
            myReachablePosition=[False]*9
            for rowPermParts in product(
                [tuple(range(freeRows))], permutations(range(freeRows,3))):
                rowPerm=list(chain.from_iterable(rowPermParts))
                myRowPermutedBlock=permuteBands(myBlock, rowPerm)
                for colPermParts in product(
                    [tuple(range(freeCols))], permutations(range(freeCols,3))):
                    colPerm=list(chain.from_iterable(colPermParts))
                    myPermutedBlock=permuteStacks(myRowPermutedBlock,colPerm)
                    for i in range(3):
                        for j in range(3):
                            if myPermutedBlock[i][j]==0:
                                myReachablePosition[3*i+j]=True
            if not all(myReachablePosition):
                return( myBlock, myReachablePosition)
    return None


def verifyConfigWithBoundaries(aSymbol, rowBounds, colBounds, otherBounds):
    assert(len(rowBounds)<=3)
    assert(len(colBounds)<=3)
    
    myAllBounds=set([])
    myAllBounds=otherBounds|set([e for t in [rowBounds, colBounds] for s in t for e in s])
    myAllSymbols=set(myAllBounds)
        
    maxBoundarySymbol=max(myAllSymbols)
    assert(len(myAllSymbols)<=4)
    assert(myAllSymbols==set(range(1,maxBoundarySymbol+1)))
    #freeRows=list(range(len(rowBounds),3))
    #freeCols=list(range(len(colBounds),3))
    freeRows=len(rowBounds)
    freeCols=len(colBounds)
    for myMatrix in generate01Matrices(3,3,maxBoundarySymbol):
        for myAssignement in permutations(list(range(1,maxBoundarySymbol+1))):
            myBlock=fillBlocksWithWeights(myMatrix, myAssignement)
            if not isBoundaryConsistent(myBlock, rowBounds, colBounds):
                continue
                
            myReachablePosition=[False]*9
            for rowPermParts in product(
                [tuple(range(freeRows))], permutations(range(freeRows,3))):
                rowPerm=list(chain.from_iterable(rowPermParts))
                myRowPermutedBlock=permuteBands(myBlock, rowPerm)
                for colPermParts in product(
                    [tuple(range(freeCols))], permutations(range(freeCols,3))):
                    colPerm=list(chain.from_iterable(colPermParts))
                    myPermutedBlock=permuteStacks(myRowPermutedBlock,colPerm)
                    for i in range(3):
                        for j in range(3):
                            if myPermutedBlock[i][j]==0:
                                myReachablePosition[3*i+j]=True
            if not all(myReachablePosition):
                return( myBlock, myReachablePosition)
    return None                    
            
def get01BlockMatrix(aSudoku):
    return tuple(tuple(1 if any([aSudoku[b][s][r][c] 
        for c in range(3) for r in range(3)]) else 0
            for s in range(3)) for b in range(3))

def getWeightBlockMatrix(aSudoku):
    return tuple(tuple(sum([1 if aSudoku[b][s][r][c]!=0 else 0  
        for c in range(3) for r in range(3)])
            for s in range(3)) for b in range(3))

def get01CellMatrix(aSudoku):
    return tuple(tuple(tuple(tuple(1 if aSudoku[b][s][r][c]!=0 else 0
        for c in range(3) ) for r in range(3))
            for s in range(3) ) for b in range(3))

def normalizeMatrix(aSudoku):

    trace(trcMsgInfo, aSudoku)
    
    my01BlockMatrix=get01BlockMatrix(aSudoku)
    trace(trcMsgInfo,my01BlockMatrix)
    
    # normalize01BlockMatrix
    #
    # find optimal matix
    (myOptimal01BlockMatrix, my01BlockTransforms)=optimizeBlock(my01BlockMatrix, breakIfNotMaximal=False )
    trace(trcMsgInfo, my01BlockTransforms, "Transforms")
    (_, my01BlockStabilizers) = optimizeBlock(
        myOptimal01BlockMatrix, breakIfNotMaximal=False )
    trace(trcMsgInfo,myOptimal01BlockMatrix)
    trace(trcMsgInfo, my01BlockStabilizers,"Stabilizers")
    # apply one of transforms to Sudoku
    mySudoku2=applyGlobalTransform(my01BlockTransforms[0], aSudoku)
    
    trace(trcMsgInfo,mySudoku2)
    myWeightBlockMatrix=getWeightBlockMatrix(mySudoku2)
    
    trace(trcMsgInfo,myWeightBlockMatrix)
    (myOptimalWeightBlockMatrix, myWeightBlockTransforms)=filterGlobalOptimum(
        myWeightBlockMatrix, my01BlockStabilizers, applyBlockFunction,
        isLessWeightBlock,breakIfNotOptimum=False)
    trace(trcMsgInfo, myWeightBlockTransforms, "Transforms")
    trace(trcMsgInfo,myOptimalWeightBlockMatrix)
    (_, myWeightBlockStabilizers)=filterGlobalOptimum(
        myOptimalWeightBlockMatrix, my01BlockStabilizers, 
        applyBlockFunction, isLessWeightBlock)
    trace(trcMsgInfo, myWeightBlockStabilizers, "Stabilizers")
    mySudoku3=applyGlobalTransform(myWeightBlockTransforms[0], mySudoku2)

    trace(trcMsgInfo,mySudoku3)
    my01CellMatrix=get01CellMatrix(mySudoku3)

    trace(trcMsgInfo,my01CellMatrix)
    
    (myOptimal01CellMatrix, my01CellTransforms)=filterGlobalOptimum(my01CellMatrix, myWeightBlockStabilizers, 
            applyGlobalTransform, isGreater01Cell, 
            optimizeLocal=True,breakIfNotOptimum=False)
    (_ , myCell01Stabilizers)=filterGlobalOptimum(myOptimal01CellMatrix, myWeightBlockStabilizers, 
            applyGlobalTransform, isGreater01Cell, 
            optimizeLocal=True)
    mySudoku4=applyGlobalTransform(my01CellTransforms[0], mySudoku3)

    mySudoku5=applyFullTransform((IdGlobalTransformation, IdLocalTransformation), mySudoku4)
        
    #(mySudoku4, myTransforms)=localOptimize(my01CellMatrix)
    return(mySudoku5, mySudoku4,mySudoku3,myOptimalWeightBlockMatrix, myWeightBlockMatrix,  mySudoku2,myOptimal01BlockMatrix, my01BlockMatrix, aSudoku)
    # return(mySudoku5, mySudoku4,my01CellMatrix, mySudoku3,myOptimalWeightBlockMatrix, myWeightBlockMatrix,  mySudoku2,myOptimal01CellMatrix, myOptimal01BlockMatrix, my01BlockMatrix, aSudoku)
    return mySudoku5



if __name__=="__main__":
    #import test_simple_sudoku     
    pass    


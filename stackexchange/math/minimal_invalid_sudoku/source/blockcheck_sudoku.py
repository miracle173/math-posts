''' blockcheck_sudoku_02.py open
(previous filename python_blockcheckpy.py)
'''
from itertools import combinations, permutations, product, chain
from random import shuffle
from copy import deepcopy
from matrix1 import permuteRows, permuteCols, printMatrix

SquareCellPositions=[(x,y) for x in [0,1,2] for y in [0,1,2]]

        
# def toRectangularMatrix(aSparseMatrix):
    # return deepcopy(aSparseMatrix)

def nullMatrix(aRowDim, aColDim):
    return ([[0]*aColDim for _ in range(aRowDim)])
    
def createMatrix(aRowDim, aColDim, aElementList):
    mySparseMatrix=nullMatrix(aRowDim, aColDim)
    for (myRow, myCol, myValue) in aElementList:
        setElem(mySparseMatrix, myRow, myCol, myValue)
    return mySparseMatrix

def testmatrix(aRowDim, aColDim):
    myMatrix=nullMatrix(aRowDim, aColDim)
    for r in range(aRowDim):
            for c in range(aColDim):
                setElem(myMatrix, r, c, 10*r+c)
    return myMatrix

def assignStartPosistions(aSymbols, aForbidden):
    mySymbolList=list(aSymbols)
    pairs = [(x, y) for x in range(len(aForbidden)) for y in range(len(aForbidden[0]))]
    def backtrack(i, used_pairs, current):
        # alle s erfolgreich belegt
        if i == len(mySymbolList):
            return current
        s = mySymbolList[i]
        for x, y in pairs:
            if (x, y) not in used_pairs and s not in aForbidden[x][y]: 
                result = backtrack(
                    i + 1,
                    used_pairs | {(x, y)},
                    current + [(s, x, y)]
                )
                if result is not None:
                    return result
        return None  # keine Moeglichkeit fuer dieses s
    solution = backtrack(0, set(), [])
    if solution is None:
        raise ValueError("Keine gueltige Belegung existiert")
    return solution

def createForbiddenMatrix(aSymbols,aBlock, aRowClues, aColClues):
    myMatrix=[[aSymbols.copy() if aBlock[r][c]!=0 else set() for c in range(3)] for r in range(3)]
    for r in range(len(aRowClues)):
        for c in range(3):
            myMatrix[r][c]|=aRowClues[r]
    for r in range(3):
        for c in range(len(aColClues)):
            myMatrix[r][c]|=aColClues[c]
    return myMatrix
            
    

def possibleMatrices(aRowDim, aColDim, aValueset ):
    # creates all possible sparse matrices to a valueset
    for myPositions in combinations(range(aRowDim*aColDim),len(aValueset)):
        for myValues in permutations(aValueset):
            myMatrix=nullMatrix(aRowDim, aColDim)
            for mySparse in zip(myPositions, myValues):
                setElem(myMatrix,mySparse[0]//aColDim, mySparse[0]%aColDim, mySparse[1])
            yield myMatrix


# def printMatrix(aSparseMatrix):
    # myMatrix=toRectangularMatrix(aSparseMatrix)
    # myRowDim=len(myMatrix)
    # myColDim=len(myMatrix[0])
    # width=[0]*myColDim
    # for r in range(myRowDim):
        # for c in range(myColDim):
            # width[c]=max(width[c],len(str(myMatrix[r][c])))
    # for r in range(myRowDim):
        # for c in range(myColDim):
            # if c==0:
                # line=''
            # else:
                # line+=' '
            # line+='%0'+str(width[c])+'d'
        # print(line%tuple(myMatrix[r]))
        
    
def getElem(aSparseMatrix,aRow, aCol):
    return aSparseMatrix[aRow][aCol]
    
def setElem(aSparseMatrix,aRow, aCol, aValue):
    aSparseMatrix[aRow][aCol]=aValue

def delElem(aSparseMatrix,aRow, aCol, aValue):
    aSparseMatrix[aRow][aCol]=aValue


def colDimension(aSparseMatrix):
    return len(aSparseMatrix[0])
    
def rowDimension(aSparseMatrix):
    return len(aSparseMatrix)
    

# def permuteCols(aSparseMatrix, aPermutation):
    # myMatrix=list(zip(*aSparseMatrix))
    # myMatrix= permuteRows(myMatrix, aPermutation)
    # myMatrix=list(zip(*aSparseMatrix))
    # myMatrix=list(map(list,myMatrix))
    # return myMatrix

# def permuteRows(aSparseMatrix, aPermutation):
    myMatrix=deepcopy(aSparseMatrix)
    return [myMatrix[aPermutation[r]] for r in range(len(myMatrix))]
    # return [aSparseMatrix[aPermutation[r]] for r in range(len(aSparseMatrix))]


      
def colValues(aSparseMatrix, aCol):
    return [aSparseMatrix[r][aCol] for r in range(len(aSparseMatrix))]

def rowValues(aSparseMatrix, aRow):
    return [aSparseMatrix[aRow][c] for c in range(len(aSparseMatrix[0]))]
    

def compose(left, right):
    return [left[right[i]] for i in range(len(left))]

def invertPermutation(aPermutation):
        return  [v for (u,v) in sorted([(y,x) for x,y in enumerate(aPermutation)])]
               # [v for (u,v) in sorted([(y,x) for x,y in enumerate(aPermutation)])]
        
def isColBoundsConsistent(aSparseMatrix, aColConstraints):
    for col, bound in enumerate(aColConstraints):
        for value in colValues(aSparseMatrix, col):
            #print('value, bound', value, bound)
            if value in bound:
                return False
    return True
    

def isRowBoundsConsistent(aSparseMatrix, aRowConstraints):
    for row, bound in enumerate(aRowConstraints):
        for value in rowValues(aSparseMatrix, row):
            #print('value, bound', value, bound)
            if value in bound:
                return False
    return True
    
def isBoundsConsistent(aSparseMatrix, aRowConstraints, aColConstraints):
    return ( isRowBoundsConsistent(aSparseMatrix, aRowConstraints)
    and isColBoundsConsistent(aSparseMatrix, aColConstraints))
        

def isExtendibleBy(aSymbols, aConstraints):
    '''
    checks if the empty block constrainted by the aRowConstraints, colCLues and the remaining aOtherGivens can be arbitrary extended by the symbols of the symbol set
    aConstraints = '[' RowConstraints ',' ColConstraints ',' OtherConstraints ']'
    RowConstraints = Tuple Of Sets of Symbols or Empty Tuple
    ColConstraints = Tuple Of Sets of Symbols  or empty Tuple
    OtherConstraints = List of Symbols or empty List
    '''
    (aRowConstraints, aColConstraints, aOtherGivens) = aConstraints
    myLineGivens=[e for t in [aRowConstraints, aColConstraints] for s in t for e in s]
    myLineGivens.extend(aOtherGivens)
    myLineGivens.sort()
    myGivensSymbols=set(myLineGivens)
    myGivensCount=len(myLineGivens)
    

    myPresetSymbols=myGivensSymbols - aSymbols
    myBoundedSymbols=sorted(list(myGivensSymbols & aSymbols))
    myNewSymbols=aSymbols - myGivensSymbols
    myFreeRowsStart=len(aRowConstraints)
    myFreeColsStart=len(aColConstraints)
    
    '''
    there are three cases:
    1.) the symbol of the clue in that block (aSymbol)  is different to the other clue symbols:
        we have to reach every position in the block form every position of the block, we cann change the symbol of the unused symbols to our symbol
    2.) the symbol is equal to a symbol of the otherBounds but not a colBound or row bound:
        we still have to reach every position from every position but we are not allowed to change the symbol
    3.) the symbol is a row bound or col bound: we cannot cjhange the symbol but we only have to reach the positions
    in the block that are not forbidden for this symbol
    '''


    #ForSymbolConsistencyTest
    myTestMatrix=nullMatrix(3,3)
    # we try to positions the other clues' symbols in the block
    # if we have a contradiction we skip the block
    myProblems=[]
    for myBlock in possibleMatrices(3, 3, myGivensSymbols ):
        # test only consistent configurations
        if not isBoundsConsistent(myBlock, aRowConstraints, aColConstraints):
            continue
        myUnreachables=findUnreachable(myBlock, len(myNewSymbols), myBoundedSymbols, aConstraints)
        if myUnreachables is not None:
            myProblems.append(myBlock)
    return(myProblems)

'''
full extending configurations
=============================
'''

def extractSymbolsAndConstraints(aConfig, aBandIdx, aStackIdx):
    '''  
    takes a block from the sudoku and retrieves the constraints and the symbols
    '''
    return(extractSymbols(aConfig, aBandIdx, aStackIdx), (
    extractRowConstraints(aConfig, aBandIdx, aStackIdx),
    extractColConstraints(aConfig, aBandIdx, aStackIdx),
    extractOtherConstraints(aConfig, aBandIdx, aStackIdx))
    )
                
def extractSymbols(aConfig, aBandIdx, aStackIdx):
    return set([aConfig[3*aBandIdx+r][3*aStackIdx+c] 
        for r in range(3) for c in range(3) 
            if aConfig[3*aBandIdx+r][3*aStackIdx+c]!=0])
                
def extractColConstraints(aConfig, aBandIdx, aStackIdx):
    myTransposed=list(zip(*aConfig))
    return extractRowConstraints(myTransposed, aStackIdx, aBandIdx)

def extractRowConstraints(aConfig, aBandIdx, aStackIdx):
    return sorted([set(m) for r in range (3) 
        if (m:=[aConfig[3*aBandIdx+r][c] 
            for c in range(9) 
                if aConfig[3*aBandIdx+r][c]!=0 and c//3!=aStackIdx])
                !=[]],key=lambda t:len(t))

def extractOtherConstraints(aConfig, aBandIdx, aStackIdx):
    return sorted([aConfig[r][c] 
        for r in range(9)  for c in range (9) 
            if r//3!= aBandIdx and c//3!=aStackIdx and aConfig[r][c]!=0])
                    
def extractUsedBlocks(aConfig):
    return [(b,s) for b in range(3) for s in range(3) if any([aConfig[3*b+r][3*s+c] for r in range(3) for c in range(3)])]
    
def isFullExtensible(aConfig):
    '''
    checks if a configuraion could be created by adding a block to a smaller configuration
    Arguments:
        aConfig: a Sudokukonfiguration
    returns:
        a list [(b,s),...] 
        The configuration aConfig could be created by full-extending the configuraion at block (b,s) (define "full-extending" anyhwere, extending the configuration, were the block b,s is empty with the symbols that should be finally contained in this block)
    '''
    resultList=[]
    for (myBandIdx, myStackIdx) in extractUsedBlocks(aConfig):
        mySymbols,myConstraints=extractSymbolsAndConstraints(aConfig, myBandIdx, myStackIdx)
        if isExtendibleBy(mySymbols,myConstraints)==[]:
            resultList.append((myBandIdx, myStackIdx))
    if resultList==[]:
        return None
    else:
        return resultList
        
def fromBlockwiseToMatrix(aSudoku):
    return [[aSudoku[b][s][r][c] 
        for s in range(3) for c in range(3) ] 
            for b in range(3) for r in range(3)]

def gatherPermutedPositions(aBlock, aFreeSymbolsCount, aBoundedSymbolList):
    '''
    returns the positions where the bounded Symbols are located combined with all position combinations for the free symbosl. these are the cells containing 0
    '''
    myReachedPositions=set()
    myBoundedPositions=[(i,j) for (s,i,j) in sorted([(aBlock[i][j], i, j)
        for i in range(3) 
            for j in range(3) if aBlock[i][j] in aBoundedSymbolList
                ])]
    myFreePositions=[(i,j) for i in range(3) for  j in range(3) if aBlock[i][j]==0]            
    for myFreePosComb in combinations(myFreePositions, aFreeSymbolsCount):
        for myConfigPositions in permutations(myFreePosComb):
            myAllPositions=list(myConfigPositions)
            myAllPositions.extend(myBoundedPositions)
            myReachedPositions.add(tuple(myAllPositions)) 
    return myReachedPositions

def gatherPresetBlockPositions(aBlock, aFreeSymbolsCount, aBoundedSymbols, aConstraints):
    aFreeRowsStart=len(aConstraints[0])
    aFreeColsStart=len(aConstraints[1])
    myReachedPositions=set()
    
    debugPrint=True
    for myRowPermParts in product(
        [tuple(range(aFreeRowsStart))], permutations(range(aFreeRowsStart,3))):
        
        myRowPerm=list(chain.from_iterable(myRowPermParts))
        myRowPermutedBlock=permuteRows(aBlock, myRowPerm) 
        # print("myRowPerm", myRowPerm)
        
        for myColPermParts in product(
            [tuple(range(aFreeColsStart))], permutations(range(aFreeColsStart,3))):
            myColPerm=list(chain.from_iterable(myColPermParts))
            myPermutedBlock=permuteCols(myRowPermutedBlock,myColPerm)
            # print("block", ''.join([str(myPermutedBlock[i][j]) for i in range(3) for j in range(3)]))
            # printMatrix(myPermutedBlock)
            # if debugPrint:
                # print("myColPerm", myColPerm)
            
            myReachedPositions|=gatherPermutedPositions(myPermutedBlock, aFreeSymbolsCount, aBoundedSymbols)
        debugPrint=False
    return myReachedPositions

def createNeededPositions(aFreeSymbolsCount, aBoundedSymbols, aConstraints):
    myRowConstraints=aConstraints[0]
    myColConstraints=aConstraints[1]
    
    myAllValidPositions=set()
    for myPositions in product(SquareCellPositions,repeat=aFreeSymbolsCount+len(aBoundedSymbols)):
        # two symbols cannot occupy the same cell
        if len(myPositions)!=len(set(myPositions)):
            continue  
        
        # remove all tuples were at least on position cannot 
        # be reached because of the constraints
        foundUnreachable=False
        for symIdx, sym in enumerate(aBoundedSymbols):
            posIdx=aFreeSymbolsCount+symIdx
            row,col=myPositions[posIdx]
            if row<len(myRowConstraints) and (sym in myRowConstraints[row]):
                foundUnreachable=True
                break
            if col<len(myColConstraints) and (sym in myColConstraints[col]):
                foundUnreachable=True
                break
        if not foundUnreachable:
            myAllValidPositions.add(myPositions)
    return myAllValidPositions
        
def findUnreachable(aBlock, aFreeSymbolsCount, aBoundedSymbols, aConstraints):
    myNeededPositions=createNeededPositions(
        aFreeSymbolsCount, aBoundedSymbols, aConstraints)
    myReachedPositions=gatherPresetBlockPositions(aBlock, aFreeSymbolsCount, aBoundedSymbols, aConstraints)
    # print("myNeededPositions ",list(myNeededPositions)[0])
    # print("myReachedPositions",list(myReachedPositions)[0])
    # print()
    # print("debug")
    # print(myNeededPositions-myReachedPositions)
    myDifference=myNeededPositions-myReachedPositions
    if myDifference!=set():
        return(myDifference)
    else:
        return None
    
    
def printExtendQuery(aSymbols, aConstraints):
    myRowConstraints=[sorted([c for c in row]) for row in aConstraints[0]]
    myColConstraints=[sorted([c for c in row]) for row in aConstraints[1]]
    myOtherGivens=sorted(aConstraints[2])
    
    myLineGivens=[e for t in [myRowConstraints, myColConstraints] for s in t for e in s]
    myLineGivens.extend(myOtherGivens)
    myLineGivens.sort()
    myGivensSymbols=set(myLineGivens)
    myGivensCount=len(myLineGivens)

    myPresetSymbols=myGivensSymbols - aSymbols
    myBoundedSymbols=sorted(list(myGivensSymbols & aSymbols))
    myNewSymbols=sorted(list(aSymbols - myGivensSymbols))


    line=''
    for s in myBoundedSymbols:
        line+=str(s)
    if myBoundedSymbols:
        line+=' '
    for s in myNewSymbols:
        line+=str(s)
    print(line)

    for r in range(3):
        line='...'
        if r<len(myRowConstraints):
            for s in myRowConstraints[r]:
                line+=str(s)
        print(line)
    for r in range(max(1,0 if len(myColConstraints)==0 else           len(myColConstraints[0]))):
        line=''
        for c in range(3):
            if c<len(myColConstraints) and r<len(myColConstraints[c]):
                line+=str(myColConstraints[c][r])
            else:
                line+=' '
        if r==0:
            for s in myOtherGivens:
                line+=str(s)
        print(line)
        
def printIsExtendibleBy(aMatrixList):
    for matrix in aMatrixList:
        for row in matrix:
            line=''
            for s in row:
                line+=str(s) if s!=0 else '.'
            print(line)
        print()
        
def printReachedPositions(aPositionList):     
    matrix=[['.']*3 for _ in range(3)]
    for myPosition in aPositionList:
        matrix[myPosition[0][0]][myPosition[0][1]]='x'
    for row in matrix:
        line=''
        for s in row:
            line+=s
        print(line)
    
        
        
#########################################

if __name__=='main':

    from time import time
    from contextlib import redirect_stdout
    with open('C:\\Users\\guent\\OneDrive\\work\\sudoku\\out.txt', 'w') as f:
        pass

    # with open('C:\\Users\\guent\\OneDrive\\work\\sudoku\\out.txt', 'a') as f:
        # with redirect_stdout(f):
            # pass
            # sss=0
            # N=1000
            # row,col=3,3
            # myBounds=(({1,2},),({3},),set())
            # diffTime=[]
            # startTime=time()
            # for _ in range(N):
                # for matrix in possibleMatrices(row,col,[1,2,3]):
                    # if isBoundsConsistent(matrix,myBounds[0],myBounds[1]):
                        # print()
                        # print("hash =","".join([str(getElem(matrix,r,c)) for r in range(3) for c in range(3)]))
                        # printMatrix(matrix)
                        # sss+=1
            # endTime=time()
            # diffTime.append(endTime-startTime)

    # for nr, diff in enumerate(diffTime):
        # print(nr, "   %8.2e"%diff)
        # print(sss)


    ##########################################################

    # mySymbols=set([4,5])
    # myBounds=(({1,2},),({3},),set())
    # myBounds=((set(),{5,},{5,}),(set(),{5,},{5,}),set())
    # myBlock=createMatrix(3,3,[(1,0,1),(2,2,2),(0,2,3)])
    # print("mySymbols", mySymbols)
    # print("row constraints")
    # for nr, cons in enumerate(myBounds[0]):
        # print(nr, cons)
    # print("Block")
    # printMatrix(myBlock)
    # print("col constraints")
    # for nr, cons in enumerate(myBounds[1]):
        # print(nr, cons)
    # myForbidden=createForbiddenMatrix(mySymbols, myBlock, myBounds[0], myBounds[1])
    # for row in myForbidden:
        # print(row)
        
    # print(assignStartPosistions(mySymbols, myForbidden))

    ###############################################################

    # myBlock=[[0,1,0],[2,0,0],[0,0,0]]
    # myFreeSymbolsCount=2
    # myBoundedSymbolList=[2]

    # result=gatherPermutedPositions(myBlock, myFreeSymbolsCount, myBoundedSymbolList)
    # for myCoordinatesresult in result:
        # print(myCoordinatesresult)
     
    ################################################################ 

    # with open('C:\\Users\\guent\\OneDrive\\work\\sudoku\\out.txt', 'a') as f:
        # with redirect_stdout(f):

            # for myCoords in createNeededPositions( 2, [ 
                # (
                # (False,False,False),
                # (True,True,True),
                # (True,True,True),
                # ),
                # (
                # (True,True,True),
                # (False,False,False),
                # (True,True,True),
                # ),
                # ]):
                # print(myCoords)

    myConstraints=(({1},),(),[])
    mySymbols={1,2}

    myBlock=[[0,0,0],[1,0,0],[2,0,0]]


    myRowConstraints=[sorted([c for c in row]) for row in myConstraints[0]]
    myColConstraints=[sorted([c for c in row]) for row in myConstraints[1]]
    myOtherGivens=sorted(myConstraints[2])

    myLineGivens=[e for t in [myRowConstraints, myColConstraints] for s in t for e in s]
    myLineGivens.extend(myOtherGivens)
    myLineGivens.sort()
    myGivensSymbols=set(myLineGivens)
    myGivensCount=len(myLineGivens)

    myPresetSymbols=myGivensSymbols - mySymbols
    myBoundedSymbols=sorted(list(myGivensSymbols & mySymbols))
    myNewSymbols=sorted(list(mySymbols - myGivensSymbols))

    myFreeSymbolsCount=len(myNewSymbols)

    print("myBoundedSymbols", myBoundedSymbols)
    print("myNewSymbols", myNewSymbols)

    printExtendQuery(mySymbols,myConstraints)

    N=10000
    startTime=time()
    for _ in range(N):
        result=isExtendibleBy(mySymbols,myConstraints)
    print("time", (time()-startTime)/N)
    if result == []:
        print("can extend")
    else:
        print("cannot extend")
        printIsExtendibleBy(result)
        
    print("#########")

    # '''gatherPresetBlockPositions'''
    # myGatheredPos=gatherPresetBlockPositions(myBlock, myFreeSymbolsCount, myBoundedSymbols,  myConstraints) 
    # print("gatherPresetBlockPositions")
    # for pos in  (myGatheredPos):
        # print(pos)
        
    #print(myGatheredPos)
    # printReachedPositions(myGatheredPos)

    '''createNeededPositions(aFreeSymbolsCount, aBoundedSymbols, aConstraints)'''
    # print("createNeededPositions")
    # myNeededPos=createNeededPositions(myFreeSymbolsCount, myBoundedSymbols,  myConstraints) 
    # for pos in myNeededPos:
        # print(pos)

    # printReachedPositions(myNeededPos)


    # '''gatherPermutedPositions(aBlock, aFreeSymbolsCount, aBoundedSymbolList):'''
    # printMatrix(myBlock)
    # for pos in  (gatherPermutedPositions(myBlock, myFreeSymbolsCount, myBoundedSymbols)):
        # print(pos)
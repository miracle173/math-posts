from itertools import accumulate, pairwise, chain, compress, combinations, permutations, zip_longest
from copy import deepcopy

VERSION='250108_0540_open'

class SegmentedMatrix:
    def __init__(self, aBandWidths, aStackWidths, aMatrix, addShadowMatrix=False, recordTransformations=False):
        self.bandWidths=aBandWidths[:]
        self.stackWidths=aStackWidths[:]
        self.rowDim=sum(self.bandWidths)
        self.colDim=sum(self.stackWidths)
        self.readData(aMatrix)
        if addShadowMatrix:
            self._shadowMatrix=SegmentedMatrix(aBandWidths, aStackWidths,
                [[1+row*sum(aStackWidths)+col for col in range(sum(aStackWidths))] for row in range(sum(aBandWidths))])
        else:
            self._shadowMatrix=None
        if recordTransformations:
            self._transformations=[]
        else:
            self._transformations=None

    def __eq__(self, other):
        if isinstance(other,SegmentedMatrix):
            return(self._data==other._data
                   and self.bandWidths==other.bandWidths
                   and self.stackWidths==other.stackWidths)
        return(False)
        
# I/O
    def readData(self, aMatrix):
        self._data=[myRow[:] for myRow in aMatrix]
        if (self.rowDim != len(self._data)):
            raise ValueError("row dimension does not macht data")
        for myCol in self._data:
            if (self.colDim!=len(myCol)):
                raise ValueError("col dim does not macht data")

    def print(self):
        """
        Print segmentation in ASCII. Use '.' for zero cells.
        Horizontal dash count per stack: dash_count = 2 * w + 1
        Example for three stacks with w=3: +-------+-------+-------+
        """
        if not self._data:
            print("+ +")
            return

        # build top line segments
        segments = []
        for w in self.stackWidths:
            dash_count = 2 * w + 1
            segments.append("+" + "-" * dash_count)
        top_line = "".join(segments) + "+"

        row_ptr = 0
        for b, bw in enumerate(self.bandWidths):
            print(top_line)
            for r in range(bw):
                row = self._data[row_ptr]
                out = "|"
                col_ptr = 0
                for s, sw in enumerate(self.stackWidths):
                    # print sw cells with leading space
                    for c in range(sw):
                        val = row[col_ptr + c]
                        out += " " + (str(val) if val != 0 else ".")
                    out += " |"
                    col_ptr += sw
                print(out)
                row_ptr += 1
        print(top_line)
   

    def _permuteRows(self, aPermutation):
        self._data=[self._data[idx] for idx in aPermutation]
        if self._shadowMatrix:
            self._shadowMatrix._permuteRows(aPermutation)

    def _permuteCols(self, aPermutation):
        self._transpose()
        self._permuteRows(aPermutation)
        self._transpose()
    
    def switchRows(self, first, second):
        if first==second:
            return
        p=list(range(self.rowDim))
        p[first], p[second]=second, first
        self._permuteRows(p)
        if not self._transformations is None:
            self._transformations.append(('sR',(first, second)))
            
            
        
    def switchCols(self, first, second):
        if first==second:
            return
        p=list(range(self.colDim))
        p[first], p[second]=second, first
        self._permuteCols(p)
        if not self._transformations is None:
            self._transformations.append(('sC',(first, second)))
     
    def debugPrint(self, message):
        print()
        print(message)
        self.print()

    def optimizeMatrix(self):
        # initialize
        myStartGrid=deepcopy(self)
        #rowCount=3
        #colCount=3
        myTranposedGrid=deepcopy(myStartGrid)
        myTranposedGrid.transpose()
        myNextSolutions=[myStartGrid, myTranposedGrid]
        myNextMinFreeCol=0
        #assert(self.rowDim==self.colDim)
        #assert(self.colDim==3)
        myGridAlreadyStored=False
        for r in range(self.rowDim):
            myNextRowIsUsed=False
            for c in range(self.colDim):
                myCurrOpt=0
                myCurrSolutions=myNextSolutions
                myNextSolutions=[]
                myMinFreeCol=myNextMinFreeCol
                myRowIsUsed=myNextRowIsUsed
                for myGrid in myCurrSolutions:
                    myGridAlreadyStored=False
                    for i in range(self.rowDim):
                        for j in range(self.colDim):
                            myCanSwap=(((r==i) 
                                or ((r<i) and not myRowIsUsed)) 
                                and ((c==j) or ((myMinFreeCol==c) and (c<j))))
                            if myCanSwap and myGrid._data[i][j]>=myCurrOpt:
                                if myGrid._data[i][j]>0:
                                    myGridAlreadyStored=False
                                    myGridCopy=deepcopy(myGrid)
                                    myNextRowIsUsed=True
                                    assert(c<=myMinFreeCol)
                                    myGridCopy.switchRows(i,r)
                                    myGridCopy.switchCols(j,c)
                                    myNextMinFreeCol=max(myMinFreeCol,c+1)
                                if myGrid._data[i][j]>myCurrOpt:
                                    myCurrOpt=myGrid._data[i][j]
                                    myNextSolutions=[myGridCopy]
                                elif myGrid._data[i][j]> 0:
                                    myNextSolutions.append(myGridCopy)
                                elif not myGridAlreadyStored:
                                #else:
                                    myGridCopy=deepcopy(myGrid)
                                    myNextSolutions.append(myGridCopy)
                                    myGridAlreadyStored=True
        return myNextSolutions
            
               
        

# row/col manipulation

    def permutationRowsOfBand(self, aBandIdx, aPermutation):
        assert(aBandIdx<len(self.bandWidths))
        assert(len(aPermutation)<=self.bandWidths[aBandIdx])
        assert(sorted(aPermutation)==list(range(len(aPermutation))))
        myStartCellIdx=sum(self.bandWidths[:aBandIdx])
        myEndCellIdx=myStartCellIdx+self.bandWidths[aBandIdx]
        myPermutation=list(range(sum(self.bandWidths)))
        myPermutation[myStartCellIdx:myEndCellIdx]=[p+myStartCellIdx for p in aPermutation]
        return(myPermutation)
    
    def permutationColsOfStack(self, aStackIdx, aPermutation):
        assert(aStackIdx<len(self.stackWidths))
        assert(len(aPermutation)<=self.stackWidths[aStackIdx])
        assert(sorted(aPermutation)==list(range(len(aPermutation))))
        myStartCellIdx=sum(self.stackWidths[:aStackIdx])
        myEndCellIdx=myStartCellIdx+self.stackWidths[aStackIdx]
        myPermutation=list(range(sum(self.stackWidths)))
        myPermutation[myStartCellIdx:myEndCellIdx]=[p+myStartCellIdx for p in aPermutation]
        return(myPermutation)

    def permutationBands(self, aPermutation):
        assert(len(aPermutation)==len(self.bandWidths))
        assert(sorted(aPermutation)==list(range(len(self.bandWidths))))
        return([sum(self.bandWidths[:k]) +p for k in aPermutation for p in list(range(self.bandWidths[k]))])
        
    def permutationStacks(self, aPermutation):
        assert(len(aPermutation)==len(self.stackWidths))
        assert(sorted(aPermutation)==list(range(len(self.stackWidths))))
        return([sum(self.stackWidths[:k]) +p for k in aPermutation for p in list(range(self.stackWidths[k]))])
        
    def _transpose(self):
        self._data = [list(i) for i in zip(*self._data)]
        self.bandWidths, self.stackWidths=self.stackWidths, self.bandWidths
        if self._shadowMatrix:
            self._shadowMatrix.transpose()
 
    def transpose(self):
        self._transpose()
        if not self._transformations is None:
            self._transformations.append(('t',))        

    def rotate(self):
        self._data=[list(row) for row in zip(*self._data[::-1])]
        self.bandWidths, self.stackWidths=self.stackWidths, self.bandWidths[::-1]
        if self._shadowMatrix:
            self._shadowMatrix.rotate()
        if not self._transformations is None:
            self._transformations.append(('r',))        

    @staticmethod
    def compose(left, right):
        assert(len(left)==len(right))
        assert(sorted(left)==list(range(len(left))))
        assert(sorted(right)==list(range(len(right))))
        return([left[right[i]] for i in range(len(right))])

    def permuteRowsOfBand(self, aBandIdx, aPermutation):
        self._permuteRows(self.permutationColsOfStack(aBandIdx, aPermutation))
        if not self._transformations is None:
            self._transformations.append(('pRB',(aBandIdx, aPermutation)))        

    def permuteColsOfStack(self, aStackIdx, aPermutation):
        self._permuteCols(self.permutationColsOfStack(aStackIdx, aPermutation))
        if not self._transformations is None:
            self._transformations.append(('pCS',(aBandIdx, aPermutation)))
        
    def permuteBands(self, aPermutation):
        self._permuteRows(self.permutationBands(aPermutation))
        if not self._transformations is None:
            self._transformations.append(('pB',(aBandIdx, aPermutation)))        

    def permuteStacks(self, aPermutation):
        self._permuteCols(self.PermutationStacks(aPermutation))
        if not self._transformations is None:
            self._transformations.append(('pS',(aBandIdx, aPermutation)))        


    def _nullRows(self):
        return [any(r) for r in self._data]

    # def _expandRows(self, aToWidths):
        # if self._shadowMatrix:
            # self._shadowMatrix._expandRows(aToWidths)
            
        # myListPair=[self.bandWidths, aToWidths]
        # myRowLength=len(self._data[0])
        # myNewData=[]
        # myHoldPrevLength=0
        # myHoldPostLength=0
        # myPrevStart=0
        # myPostStart=0
        # for  (myPrevLength, myPostLength) in list(zip(*myListPair)):
            # myPrevStart+=myHoldPrevLength
            # myPostStart+=myHoldPostLength
            # myHoldPrevLength=myPrevLength
            # myHoldPostLength=myPostLength
            # myNewData.extend([self._data[myPrevStart+k] if k < myPrevLength 
                # else [0]*myRowLength for k in range(myPostLength)])
        # myNewData.extend([[0]*myRowLength]*(sum(aToWidths)-(myHoldPostLength+myPostLength)))
        # self._data=myNewData
        # self.bandWidths=aToWidths[:]
      
    def expand(self, aNewBandWidths=[3,3,3], aNewStackWidths=[3,3,3]):
        self._expandRows(aNewBandWidths)
        self._transpose()
        self._expandRows(aNewStackWidths)
        self._transpose()
        
 
    def _reduceRows(self):
        mySelector=self._nullRows()
        self._data=list(compress(self._data,mySelector))
        if self._shadowMatrix:
            self._shadowMatrix._reduceRows()
        myAccSelector=list(accumulate(mySelector))
        myWidthsBoundaries=list(pairwise(chain([0],accumulate(self.bandWidths))))
        self.bandWidths=[item for item in list(map(lambda x: myAccSelector[x[1][1]-1]-(0 if x[0]==0 else myAccSelector[x[1][0]-1]), enumerate(myWidthsBoundaries))) if item>0]
        self.bandWidths
        
    def reduce(self):
        self._reduceRows()
        self._transpose()
        self._reduceRows()
        self._transpose()
        
        
       
    
    def _blockLine(self, aBlock):
        myBandWidthsBoundaries=list(pairwise(chain([0],accumulate(self.bandWidths))))
        myLoRow=myBandWidthsBoundaries[aBlock[0]][0]
        myHiRow=myBandWidthsBoundaries[aBlock[0]][1]
        myStackWidthsBoundaries=list(pairwise(chain([0],accumulate(self.stackWidths))))
        myLoCol=myStackWidthsBoundaries[aBlock[1]][0]
        myHiCol=myStackWidthsBoundaries[aBlock[1]][1]
        myBlock=[self._data[row][col] for row in range(myLoRow,myHiRow)  for col in range(myLoCol ,myHiCol)]
        return (myBlock)
    
    @staticmethod
    def generateMaximal01Blocks (aBlockCount):
        resultList=[]
        for comb in combinations(range(1,9),aBlockCount-1):
            sm=SegmentedMatrix([3],[3],[[ (1 if 3*row+col  in set(chain([0],comb)) else 0 ) for col in range(3) ] for row in range(3)], addShadowMatrix=True)

            # filter 
            smCopy=deepcopy(sm)
            myBandWidths=sm.bandWidths[:]
            myStackWidths=sm.stackWidths[:]
            smCopy.reduce()
            smCopy.expand(myBandWidths,myStackWidths)
            if smCopy!=sm:
                continue
            
            # filter
            if max([sum(row) for row in sm._data])!=sum(sm._data[0]):
                continue
            
            exitLoops=False
            for myDoReflection in [True, False]:                    
                for myInBandPerm in permutations([0,1,2]):
                    for myInStackPerm in permutations([0,1,2]):
                        smCopy=deepcopy(sm)
                        if myDoReflection:
                            smCopy.transpose()
                        smCopy.permuteRowsOfBand(0,myInBandPerm)
                        smCopy.permuteColsOfStack(0,myInStackPerm)
                        if smCopy._data>sm._data:
                            exitLoops=True
                            break
                    if exitLoops:
                        break
                if exitLoops:
                    break
            if not exitLoops:
                resultList.append(sm)
        # find stabilizers
        stabilizers01Block={}
        for sm in resultList:
            myStabilizers=[]
            for myDoReflection in [True, False]:                    
                for myInBandPerm in permutations([0,1,2]):
                    for myInStackPerm in permutations([0,1,2]):
                        smCopy=deepcopy(sm)
                        if myDoReflection:
                            smCopy.transpose()
                        smCopy.permuteRowsOfBand(0,myInBandPerm)
                        smCopy.permuteColsOfStack(0,myInStackPerm)
                        if smCopy._data==sm._data:
                            myStabilizers.append((myDoReflection, myInBandPerm, myInStackPerm))
            sm.stabilizers01Block=myStabilizers
        return(resultList)

    @staticmethod
    def generateMaximalWeightedBlocks (aClueCount):
        resultList=[]
        for myBlockCount in range(1,aClueCount+1):
            myAllPartions=SegmentedMatrix.allPartitions(aClueCount, myBlockCount)
            for sm in SegmentedMatrix.generateMaximal01Blocks(myBlockCount):
                for myPartition in myAllPartions:
                    # generate all weighted block matrices 
                    # of this 01 block matrix
                    smWeighted=deepcopy(sm)
                    idx=0
                    for row in range(3):
                        for col in range (3):
                            if smWeighted._data[row][col]:
                                smWeighted._data[row][col]=myPartition[idx]
                                idx+=1
                    
                    # find all stabilizers
                    # filter out non maximas
                    smFixpoint=deepcopy(smWeighted)
                    smWeighted.stabilizersWeightedBlock=[]
                    exitLoop=False
                    for my01Stabilizer in smWeighted.stabilizers01Block:
                        smFixpoint=deepcopy(smWeighted)
                        if my01Stabilizer[0]:
                            smFixpoint.transpose()
                        smFixpoint.permuteRowsOfBand(0,my01Stabilizer[1])
                        smFixpoint.permuteColsOfStack(0,my01Stabilizer[2])
                        if smFixpoint==smWeighted:
                            smWeighted.stabilizersWeightedBlock.append(my01Stabilizer[:])
                        if smFixpoint._data>smWeighted._data:
                            exitLoop=True
                            break
                    if not exitLoop:
                        resultList.append(smWeighted)
        return(resultList)
                    
                            
                    
            
    """
    @staticmethood
    def transposesUnusedBlocksOnly(aMatrix, aStabilizer):
        myTranspose, myRowPermutation,myColPermutation=aStabilizer
        if not myTranspose:
            if  (
                (any([for (i, row) in enumerate(myRowPermutation) if i!=row]) 
                and not any([aMatrix[row][col] for (i, row) in enumerate(myRowPermutation) if i!=row for col in range(3)]))
                or 
                (any([for (i, col) in enumerate(myColPermutation) if i!=col]) 
                and not any([aMatrix[row][col] for (i, col) in enumerate(myColPermutation) if i!=col for row in range(3)]))
                ):
                return True
        else:
            return False
    """
                
    @staticmethod
    def allPartitions(aSum: int, aCount: int):
        """
        Note: THis function was generated by ChatGPT
        Return all ordered partitions (compositions) of aSum into aCount
        positive integers.
        
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


    @staticmethod
    def restrictedGrowthSequence(n: int):
        """
        Note: THis function was generated by ChatGPT
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


                    
                    
            
    def _expandRows(self, aToWidths):
        if self._shadowMatrix:
            self._shadowMatrix._expandRows(aToWidths)
        self._data=[self._data[row] if row - u[1][0]+u[0][0] < u[0][1] else [0]*len(self._data[0])   for u in zip_longest(pairwise(chain([0],accumulate(self.bandWidths))), pairwise(chain([0],accumulate(aToWidths))), fillvalue=(0,0)) for row in range(u[1][0],u[1][1]) ]
        self.bandWidths=aToWidths[:] 
        
  

        
def removeItems(aItems, aComponentsToRemove):
    myRowComponentsToRemove, myColComponentsToRemove= aComponentsToRemove
    myOldRowComponents=set()
    myOldColComponents=set()
    myNewRowComponents=set()
    myNewColComponents=set()
    myKeptItems=set()
    for myItem in aItems:
        myRowComponent, myColComponent = myItem
        myOldRowComponents.add(myRowComponent)
        myOldColComponents.add(myColComponent)
        if (myRowComponent in myRowComponentsToRemove) or (myColComponent in myColComponentsToRemove):
            continue
        myNewRowComponents.add(myRowComponent)
        myNewColComponents.add(myColComponent)
        myKeptItems.add(myItem)
    """
    # debug
    print("myOldRowComponents", myOldRowComponents)
    print("myOldColComponents", myOldColComponents)
    print("myNewRowComponents", myNewRowComponents)
    print("myNewColComponents", myNewColComponents)
    """
    myNewRowComponentsToRemove=(myOldRowComponents-myNewRowComponents) 
    myNewColComponentsToRemove=(myOldColComponents-myNewColComponents) 
    myNewComponentsToRemove=(myNewRowComponentsToRemove, myNewColComponentsToRemove)
    return(myKeptItems, myNewComponentsToRemove)   

def cleanupGrid(aGrid, aPosition, aRemoveList):
    # modifies aGrid !!!
    myRowDim=len(aGrid)
    myColDim=len(aGrid[0])
    myRow,myCol=aPosition
    myControlGrid=[[[set(),set()]  for j in range(myColDim)] for i in range(myRowDim)]
    myControlGrid[myRow][myCol]=list(aRemoveList)
    myGridIsClean=False
    while not myGridIsClean:
        myGridIsClean=True
        for i in range(myRowDim):
            for j in range(myColDim):
                if myControlGrid[i][j]!=[set(),set()]:
                    assert(len(myControlGrid[i][j])==2)
                    print("debug")
                    print_grid("grid",aGrid)
                    print_grid("control",myControlGrid)
                    print("i,j",i,j)
                    print("from", aGrid[i][j])
                    print("remove", myControlGrid[i][j])

                    (myKeptItems, (myRowComponentes, myColComponents))=removeItems(aGrid[i][j], myControlGrid[i][j])
                    

                    print("remains", myKeptItems)
                    print("remove Rows", myRowComponentes)
                    print("remove Cols", myColComponents)


                    aGrid[i][j]=myKeptItems
                    if myRowComponentes:
                        myGridIsClean=False
                        for k in range(myRowDim):
                            if k!=i:
                                myControlGrid[k][j][0]|=myRowComponentes
                    if myColComponents:
                        myGridIsClean=False
                        for k in range(myColDim):
                            if k!=j:
                                myControlGrid[i][k][1]|=myColComponents
                    '''
                    print("debug","i,j", i,j)
                    print("debug", myControlGrid)
                    '''
    return aGrid
    
def print_grid(message, aGrid):
    print(message)
    myRowDim=len(aGrid)
    myColDim=len(aGrid[0])
    for i in range(myRowDim):
        for j in range(myColDim):
            print(i,",",j,":",aGrid[i][j])

        

        
        

            
        

        
        


        


if __name__=="__main__":
    import unit_test2
        

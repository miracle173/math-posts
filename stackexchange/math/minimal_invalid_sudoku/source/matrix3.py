from copy import deepcopy

class SquareMatrix:

    def __init__(self, aDim):
        if aDim == 9 or aDim == 3:
            self._data=[(0,)*aDim for _ in range(aDim)]
            self.dim=aDim
            self.bmDim=aDim//3
        elif aDim==0:
            self._data=None
            self.dim=0
            self.bmDim=0
        else:
            raise ValueError("invalid dimension in SquareMatrix creation")
            
    def copy(self):
        if self._data is None:
            raise ValueError("cannot copy None-SquareMatrix")
        myCopy=self.__class__(0)
        myCopy._data=[self._data[r] for r in range(self.dim)]
        myCopy.dim=self.dim
        myCopy.bmDim=self.bmDIm
    
    def transpose(self):
        self._data=list(zip(*self._data))

    def permuteRows(self, aPermutation): 
        assert(len(aPermutation)==self.dim)
        self._data=[self._data[aPermutation[r]] for r in range(len(aPermutation))]

    def permuteCols(self, aPermutation):
        assert(len(aPermutation)==self.dim)
        self.transpose()
        self.permuteRows(aPermutation)
        self.transpose()

    def permuteBands(self, aPermutation):
        assert(len(aPermutation)==3 and self.dim==9)
        #self._data=[self._data[3*aPermutation[r//3]+r%3] for r in range(len(self._data))]
        self.permuteRows([3*aPermutation[r//3]+r%3 for r in range(self.dim)])

    def permuteStacks(self, aPermutation):
        assert(len(aPermutation)==3 and self.dim==9)
        self.transpose()
        self.permuteBands(aPermutation)
        self.transpose()
        
    def permuteRowsInBand(self, aIndex, aPermutation):
        assert(len(aPermutation)==3 and self.dim==9)
        #self._data=[self._data[3*aIndex+aPermutation[r%3] if r//3==aIndex else r] for r in range(len(self._data))]
        self.permuteRows([
            3*aIndex+aPermutation[r%3] 
                if r//3==aIndex else r 
                for r in range(self.dim)])

    def permuteColsInStack(self, aIndex, aPermutation):
        assert(len(aPermutation)==3 and self.dim==9)
        self.transpose()
        self.permuteRowsInBand(aIndex, aPermutation)
        self.transpose()

    def byRows(self):
        return self._data
        
    def byCols(self):
        return list(zip(*self._data))
        
    def row(self, aIndex):
        return self._data[aIndex]

    def col(self, aIndex):
        return tuple(self._data[r][aIndex] for r in range(self.dim))

    def toRectangularMatrix(self):
        return [[col for col in row] for row in self._data]
     
    def __repr__(self):
        return self._data.__repr__()
        
    def __setitem__(self, aPosition, aValue):
        (myRow, myCol)=aPosition
        self._data[myRow]=[(self._data[
            myRow][c] if c!=myCol else aValue) 
                for c in range(self.dim)]

    # HB63AB521/45
    
    def __getitem__(self, aPosition):
        (myRow, myCol) = aPosition
        return self._data[myRow][myCol]
        
    def __le__(self, other):
        if self.as01BlockList()==other.as01BlockList():
            if self.asWeightBlockList()==other.asWeightBlockList():
                if self.BmDim==1:
                    return True
                if self.as01CellList()==other.as01CellList():
                    if self.asSymbolCellList()==other.asSymbolCellList():
                        return True
                    else:
                        return self.asSymbolCellList()<other.asSymbolCellList()       
                else:
                    return self.as01CellList()<other.as01CellList()
            else:
                return self.asWeightBlockList()<other.asWeightBlockList()
        else:
            return self.as01BlockList()<other.as01BlockList()

    def __lt__(self, other):
        return (self.__le__(other)
            and self._data!=other._data)

    def __ge__(self, other):
        return not self.__lt__(other)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ne__(self, other):
        return  not self.__eq__(other)

    def __eq__(self, other):
        return self._data==other._data

    def as01BlockList(self):
        return [any([self._data[3*b+r][3*s+c] 
            for c in range(3) 
                for r in range(3) ])
                    for s in range(self.bmDim) 
                        for b in range(self.bmDim)] 
                        
    def asWeightBlockList(self):
         return [sum([self._data[3*b+r][3*s+c]!=0
            for c in range(3) 
                for r in range(3) ])
                    for s in range(self.bmDim) 
                        for b in range(self.bmDim)] 
                    
    def as01CellList(self):
        return [self._data[3*b+r][3*s+c]!=0 
            for c in range(3) 
                for r in range(3) 
                    for s in range(3) 
                            for b in range(3)]
                            
    def asSymbolCellList(self):
        return [self._data[3*b+r][3*s+c] 
            for c in range(3) 
                for r in range(3) 
                    for s in range(3) 
                        for b in range(3)]
      
    def fillWithValues(self, aSequence):
        '''
        purpose: replace the non-0 values of the matrix 
            by the values of the sequence. The matrix is traversed 
            block by block, from left to right, from top to bottom. 
            Each block is then processed row by row, 
            from left to right, from top to bottom. This we call the sudoku order to traverse a matrix' elements
        '''
        myIter=iter(aSequence)
        for row,col in [(3*b+r,3*s+c)
            for b in range(3) 
                for s in range(3) 
                    for r in range(3) 
                            for c in range(3)
                                if self[3*b+r,3*s+c]!=0]:
                                    self[row,col]=next(myIter)
        
        
    def replaceSymbols(self, aSymbolPermutation):
        '''
        purpose: 
            replace the symbols of the matrix
            by the permuted values. 
        '''
        assert(sorted(aSymbolPermutation)==list(range(10)))
        for row,col in [(3*b+r,3*s+c)
            for c in range(3) 
                for r in range(3) 
                    for s in range(3) 
                            for b in range(3)
                                if self[3*b+r,3*s+c]!=0]:
                                    self[row,col]=aSymbolPermutation[self[row,col]]
    
    def retrieveSymbols(self):
        '''
        purpose: 
            retrieve the symbols of a matrix in 
            the order that is to used to sort 
            the sudokus
        returns: 
            the sequence of tymbols in the order 
            they are found in the sudoku
        '''
        return [self[3*b+r,3*s+c]
        for b in range(3) 
            for s in range(3) 
                for r in range(3) 
                        for c in range(3)
                            if self[3*b+r,3*s+c]!=0]
                            
    def normalizeSymbols(self):
        mySequence=self.retrieveSymbols()
        (myNormedSequence, myPermutation)=self.normedSymbolSequence(mySequence)
        self.fillWithValues(myNormedSequence)
        
    @staticmethod                        
    def normedSymbolSequence(aSequence):
        '''
        purpose: 
            replace the values of the symbols 
            so that the resulting sequence is a 
            restricted groth sequence         
        '''
        myPermutation=[0]*10
        # myPermutation[0]==0 for technical reasons
        # this value is not used
        nextValue=1
        myNewSequence=[]
        for value in aSequence:
            if myPermutation[value]==0:
                myPermutation[value]=nextValue
                nextValue+=1
            myNewSequence.append(myPermutation[value])
                
        return (myNewSequence, myPermutation)
            
    @classmethod
    def fromList(cls, aList):
        '''
        purpose:
            create a 3*3 or 9*9 matrix from aList
        '''
        myMatrix=cls(0)
        myMatrix.myDim=int(len(aList)**0.5)
        assert(myMatrix.myDim**2==len(aList))
        assert(myMatrix.dim in {3,9})
        myMatrix.BmDim=myMatrix.dim//3
        myMatrix._data=[[aList[myMatrix.myDim*r+c] for c in range(myMatrix.myDim)] for r in range(myMatrix.myDim)]
        return myMatrix
                
            
    @classmethod        
    def testmatrix0(cls, aDim):
        me=cls(aDim)
        me._data=[tuple(10*r+c for c in range(aDim)) for r in range (aDim)]
        return me
        
    @classmethod        
    def testmatrix1(cls, aDim):
        me=cls(aDim)
        me._data=[tuple(10*(r+1)+(c+1) for c in range(aDim)) for r in range (aDim)]
        return me
    
    def print(self):
        myMatrix=self._data
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


    def printf(self, title=None):
        '''
        purpose:
        print formatted
        '''
        if self.dim==9:
            if title is not None:
                print(title)
            for r in range(9):
                line='|'
                for c in range(9):
                    if c%3==0 and c!=0:
                        line+='|'
                    #s=self._data[r//3][c//3][r%3][c%3]
                    s=self._data[r][c]
                    line+=('.' if s==0 else str(s))
                line+='|'
                if r%3==0:
                    print('+'+('-'*3+'+')*2+'-'*3+'+')
                print(line)
            print('+'+('-'*3+'+')*2+'-'*3+'+')
        elif self.dim==3:
            for row in self._data:
                line=''
                for col in row:
                    line+=str(col) if col!=0 else '.'
                print(line)
        else:
            assert(False)

    def printr(self, title=None):
        '''
        purpose:
            print in reduced format
        '''
        if not title is None:
            print(title)
        if self.dim==9:
            lastBand=-1
            lastRow=[-1]*3
            for b in range(2,-1,-1):
                for r in range(2,-1,-1):
                    index=3*b+r
                    if any(self.byRows()[index]) and lastRow[b]==-1:
                        lastRow[b]=r
                        if lastBand==-1:
                            lastBand=b
            lastStack=-1
            lastCol=[-1]*3
            for b in range(2,-1,-1):
                for r in range(2,-1,-1):
                    index=3*b+r
                    if any(self.byCols()[index]) and lastCol[b]==-1:
                        lastCol[b]=r
                        if lastStack==-1:
                            lastStack=b
            bandSeparator='+'
            for s in range(lastStack+1):
                for c in range(lastCol[s]+1):
                    bandSeparator+='-'
                bandSeparator+='+'
            for b in range(lastBand+1):
                print(bandSeparator)
                for r in range(lastRow[b]+1):
                    line=''
                    for s in range(lastStack+1):
                        line+='|'
                        for c in range(lastCol[s]+1):
                            digit=self.byRows()[3*b+r][3*s+c]
                            line+=str(digit) if digit!=0 else '.'
                    line+='|'
                    print(line)
            print(bandSeparator)
        elif self.dim==3:
            lastBand=-1
            for b in range(2,-1,-1):
                if any(self._data[b]) and lastBand==-1:
                    lastBand=b
            lastStack=-1
            for s in range(2,-1,-1):
                if any(list(zip(*self._data))[s]) and lastStack==-1:
                    lastStack=s
            bandSeparator='+-'
            for s in range(lastStack+1):
                bandSeparator+='--'
            bandSeparator+='+'
            print(bandSeparator)
            for b in range(lastBand+1):
                line='| '
                for s in range(lastStack+1):
                    digit=self._data[b][s]
                    line+=str(digit) if digit!=0 else '.'
                    line+=' '
                line+='|'
                print(line)
            print(bandSeparator)
        else:
            assert(False)
            



a=SquareMatrix.testmatrix0(9)
b=SquareMatrix.testmatrix1(9)
p=[1,2,3,4,5,6,7,8,0]
q=[1,2,0]

from random import shuffle, randint

p=list(range(9))
q=list(range(3))
i=0


a.print()
if a!=SquareMatrix.testmatrix0(9):
    raise ValueError("Fehler")

c=SquareMatrix(9)

c[3,4]=1
c[4,1]=1
c[5,8]=1
c[2,2]=1

# c.printr()
# c.printf()

sq=[1,3,2,3]
c.fillWithValues(sq)
print("fill with values",sq)
# c.printr()
# c.printf()
# sq2=[3,7,4,2,4,8,5,3,5,7,8,8,5,3]
# normed=c.normedSymbolSequence(sq2)
# print("sequence  ", sq2)
# print("normalizes",normed[0])
# print()
# print("permutation", list(range(10)))
# print("           ", normed[1])

c.printr("source")
s=c.retrieveSymbols()
print("symbols", s)

print("normedSymbolSequence",c.normedSymbolSequence(s))
c.normalizeSymbols()

c.printr()
c.printf()
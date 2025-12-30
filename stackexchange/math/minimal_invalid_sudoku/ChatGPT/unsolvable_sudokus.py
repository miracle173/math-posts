from itertools import accumulate, pairwise, chain, compress

class SegmentedMatrix:
    def __init__(self, aBandWidths, aStackWidths, aMatrix):
        self.bandWidths=aBandWidths
        self.stackWidths=aStackWidths
        self.rowDim=sum(self.bandWidths)
        self.colDim=sum(self.stackWidths)
        self.readData(aMatrix)

    def __eq__(self, other):
        if isinstance(other,SegmentedMatrix):
            return(self._data==other._data
                   and self.bandWidths==other.bandWidths
                   and self.stackWidths==other.stackWidths)
        return(False)
        
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

    def _permuteCols(self, aPermutation):
        self.transpose()
        self._permuteRows(aPermutation)
        self.transpose()

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
        
    def transpose(self):
        self._data = [list(i) for i in zip(*self._data)]
        self.bandWidths, self.stackWidths=self.stackWidths, self.bandWidths

    def rotate(self):
        self._data=[list(row) for row in zip(*self._data[::-1])]
        self.bandWidths, self.stackWidths=self.stackWidths, self.bandWidths[::-1]

    @staticmethod
    def compose(left, right):
        assert(len(left)==len(right))
        assert(sorted(left)==list(range(len(left))))
        assert(sorted(right)==list(range(len(right))))
        return([left[right[i]] for i in range(len(right))])

    def permuteRowsOfBand(self, aBandIdx, aPermutation):
        self._permuteRows(self.permutationColsOfStack(aBandIdx, aPermutation))

    def permuteColsOfStack(self, aStackIdx, aPermutation):
        self._permuteCols(self.permutationColsOfStack(aStackIdx, aPermutation))
        
    def permuteBands(self, aPermutation):
        self._permuteRows(self.permutationBands(aPermutation))

    def permuteStacks(self, aPermutation):
        self._permuteCols(self.PermutationStacks(aPermutation))

    def _nullRows(self):
        return [any(r) for r in self._data]

    def _expandRows(self, aFromWidths, aToWidths):
        myListPair=[aFromWidths, aToWidths]
        myRowLength=len(self._data[0])
        myNewData=[]
        myHoldPrevLength=0
        myHoldPostLength=0
        myPrevStart=0
        myPostStart=0
        for  (myPrevLength, myPostLength) in list(zip(*myListPair)):
            myPrevStart+=myHoldPrevLength
            myPostStart+=myHoldPostLength
            myHoldPrevLength=myPrevLength
            myHoldPostLength=myPostLength
            myNewData.extend([self._data[myPrevStart+k] if k < myPrevLength 
                else [0]*myRowLength for k in range(myPostLength)])
        myNewData.extend([[0]*myRowLength]*(sum(aToWidths)-(myHoldPostLength+myPostLength)))
        self._data=myNewData
      
    def expand(self, aNewBandWidths=[3,3,3], aNewStackWidths=[3,3,3]):
        self._expandRows(self.bandWidths, aNewBandWidths)
        self.transpose()
        self._expandRows(self.bandWidths, aNewStackWidths)
        self.transpose()
        self.bandWidths=aNewBandWidths[:]
        self.stackWidths=aNewStackWidths[:]
 
    def _reduceRows(self):
        mySelector=self._nullRows()
        print("selector", mySelector)
        self._data=list(compress(self._data,mySelector))
        myAccSelector=list(accumulate(mySelector))
        myWidthsBoundaries=list(pairwise(chain([0],accumulate(self.bandWidths))))
        self.bandWidths=[item for item in list(map(lambda x: myAccSelector[x[1][1]-1]-(0 if x[0]==0 else myAccSelector[x[1][0]-1]), enumerate(myWidthsBoundaries))) if item>0]
        self.bandWidths
        
    def reduce(self):
        self._reduceRows()
        self.transpose()
        self._reduceRows()
        self.transpose()
        
        

    


if __name__=="__main__":
    import unit_test1
        

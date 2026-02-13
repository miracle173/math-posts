# matrix

class Matrix2:
    def __init__(self, rows, cols, data=None ):
        #if issubclass(rows, Matrix2): # make a copy
        self.rowDim=rows
        self.colDim=cols
        if data is None:
            self.row=[]
            self.col=[]
            self.val=[]
        else:  # sequence of elements
            (self.row, self.col, self.val)=zip(*data)
            
    def __getitem__(self, r: int):
        return _RowAccessor(self, r)

    def getElement(self,row, col):
        if not (row<len(self.val) and col<len(self.val)):
            raise IndexError
        for i in range(len(self.val)):
            if self.row[i]==row and self.col[i]==col:
                return(self.val[i])
        return 0
                
    def setElemelemt(self, row, col,val):
        self.row.append(row)
        self.col.append(col)
        self.val.append(val)
        
    def transpose(self):
        for i in range(len(self.val)):
            self.row,self.col=self.col,self.row
    
    def permuteRows(self, permutation):
        self.row=self._permuteLine(self.row, permutation)
        
    def permuteCols(self, permutation):
        self.col=self._permuteLine(self.col, permutation)
    
    def permuteRowsInBand(self, index, permutation):
        self.row=self._permuteLineInGroup(self.row, index, permutation)
    
    def permuteColsInStack(self, index, permutation):
        self.col=self._permuteLineInGroup(self.col, index, permutation)
        
    def permuteBands(self, permutation):
        self.row=self._permuteGroup(self.row, permutation)

    def permuteStacks(self, permutation):
        self.col=self._permuteGroup(self.col, permutation)
        
    def replace(self, values): #???
        # replace 1s by the elements of values in a 0-1 matrix
        assert(len(values)==len(self.val))
        self.val=values
    
    @classmethod
    def copy(cls, matrix):
        matrixcopy=cls(matrix.rowDim, matrix.colDim )
        matrixcopy.row, matrixcopy.col, matrixcopy.val = matrix.row[:], matrix.col[:], matrix.val[:]
        return matrixcopy

        

    def toRectangular(self):
        matrix=[[0] * self.colDim for _ in range(self.rowDim)]
        for i in range(len(self.val)):
            matrix[self.row[i]][self.col[i]]=self.val[i]
        return matrix
    
    @classmethod
    def fromRectangular(cls, matrix):
        return cls(len(matrix), len(matrix[0]), [(r, c, matrix[r][c]) 
            for r in range(len(matrix)) 
                for c in range(len(matrix[0])) if matrix[r][c] !=0 ])
                
                
    
    def print(self):
        for row in self.toRectangular():
            print(row)
            
    @staticmethod
    def _permuteLine(line, permutation):
        #return [permutation[i] for i in line]
        return [i for n in line for i in range(len(permutation)) if permutation[i]==n ]

        
    @staticmethod
    def _permuteLineInGroup(line, index, permutation):
        #return [3*index+permutation[line[i]%3] if i//3==index else line[i] for i in line]
        return [3*index+i if n//3==index else n for n in line for i in range(len(permutation)) if ((permutation[i]==n%3) and n//3==index) or (n//3!=index and n%3==i)]
        
    @staticmethod
    def _permuteGroup(line, permutation):
        return [3*permutation[(i//3)]+i%3 for i in line]
            
class _RowAccessor:
    def __init__(self, matrix: Matrix2, r: int):
        self._m = matrix
        self._r = r
        
    def __getitem__(self, c: int):
        return self._m.getElement(self._r,c)
        
    def __setitem__(self, c: int, value: int):
        self._m.setElem(self._r, c, value)

        
        
    
        
# if __name__=='main':
# matrix=Matrix2(3,3,[(r,c,3*r+c+1) for c in range(3) for r in range(3) ])
# print("matrix")
# matrix.print()
# print("transpose")
# matrix.transpose()
# matrix.print()
# perm=[0,2,1]
# print("row", perm)
# print("before", matrix.row)
# matrix.permuteRow(perm)
# print("after", matrix.row)
# matrix.print()
# print("col", perm)
# print("before", matrix.col)
# matrix.permuteCol(perm)
# print("after", matrix.col)
# matrix.print()



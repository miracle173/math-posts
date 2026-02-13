#https://stackoverflow.com/questions/1685389/possible-to-use-more-than-one-argument-on-getitem
#https://docs.python.org/3/reference/datamodel.html#object.__getitem__
class Matrix:
    def __init__(self, rowDim, colDim ):
        self.rowDim=rowDim
        self.colDim=colDim
        self._data=[[0]*colDim for _ in range(rowDim)]

    def __getitem__(self, r: int):
        return _RowAccessor(self, r)
                       
class _RowAccessor:
    def __init__(self, matrix: Matrix, r: int):
        self._m = matrix
        if not (r<self._m.rowDim):
            raise IndexError
        self._r = r
        
    def __getitem__(self, c: int):
        if not ( c<self._m.colDim):
            raise IndexError
        else:
            return self._m._data[self._r][c]
        
        
    def __setitem__(self, c: int, value: int):
        if not ( c<self._m.colDim):
            raise IndexError
        self._m._data[self._r][ c] = value
        print(self._m._data)
        
        
matrix=Matrix(3,3)

matrix[0][0]=3

print("matrix[0][0]", matrix[0][0])
print("matrix[1][2]", matrix[1][2])

print("matrix[1]", matrix[1])

print("matrix", matrix)

for row in matrix:
    print(row)
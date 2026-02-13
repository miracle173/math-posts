from itertools import groupby 

def byRows(sudoku):
    '''
    purpose:
        Returns a ListOfLists of rows of a sudoku
    arguments:
        a Sudoku in LoLoLoL format
    return:
        a LoL, the lists of the List are the rows 
        of the SSudoku
    '''
    return [[sudoku[b][s][r][c] 
    for s in range(3) 
        for c in range(3)] 
            for b in range(3) 
                for r in range(3)]

def byCols(sudoku):
    '''
    purpose:
        Returns a ListOfLists of columns of a sudoku
    arguments:
        a Sudoku in LoLoLoL format
    return:
        a LoL, the lists of the list are the columns  
        of the Sudoku
    '''
    return [[sudoku[b][s][r][c] 
    for b in range(3) 
        for r in range(3)] 
            for s in range(3) 
                for c in range(3)]


def stringSudoku(sudoku):
    return ''.join([str(c) for b in sudoku for s in b for r in s for c in r])
    
def hash01NineMatrix(sudoku):
    #return hex(int(''.join([str(c) for b in sudoku for s in b for r in s for c in r]),2))
    return hex(int(stringSudoku(sudoku),2))

def stringMatrix(aMatrix):
    return ''.join([str(c) for r in aMatrix for c in r])

def hashThreeMatrix(aMatrix):
    return stringMatrix(aMatrix)

def hashMatrix(aMatrix):
    return stringMatrix(aMatrix)

def hashSudoku(aSudoku):
    return ''.join([(str(len(list(y))) if x=='0' else chr(96+int(x))*len(list(y))) for x,y in groupby(stringSudoku(aSudoku))])


def transpose(matrix):
    return tuple(zip(*matrix))
    
def transposeSudoku(sudoku):
    return tuple(
        tuple(transpose(sudoku[b][s]) for b in range(3)) for s in range(3))

def transposeSudoku2(sudoku):
    return tuple(
        tuple(transpose(block) for block in band) for band in transpose(sudoku))

def permuteBands(block, permutation):
    return tuple(block[permutation[r]] for r in range(3))

def permuteStacks(block, permutation):
    #return tuple(zip(*[tuple(zip(*block))[permutation[r]] for r in range(3)]))
    #return tuple(zip(*permuteBands(tuple(zip(*block)),permutation)))
    return transpose(permuteBands(transpose(block),permutation))
    
    
def permuteRowsInBand(sudoku, band, permutation):
    return tuple(
        tuple(sudoku[b][s] if b!=band else 
            (permuteBands(sudoku[b][s], permutation)) for s in range(3))  for b in range(3))

def permuteColsInStack(sudoku, stack, permutation):
     return tuple(
        tuple(sudoku[b][s] if s!=stack else 
             tuple(zip(*(permuteBands(tuple(zip(*sudoku[b][s])), permutation))))
            for s in range(3))  for b in range(3))
                
"""
# unklar:
def convertToBlocks(matrix):
    '''
    Input: matrix, a 9*9 matrix
    Return: a 9*9 matrix where the (3b+s)-th tow are the elements of block (b,s) row by row from top to bottom
    This can be used to sort the matrices
    '''
    # using itertool funtions: list(batched(list(chain.from_iterable(list(chain.from_iterable(list(zip(*[list(batched(t,n=3)) for t in m])))))),n=9))
    return [[matrix[3*b+r][3*s+c] for c in range(3) for r in range(3)] for s in range(3) for b in range(3)]

def convertToOneBand (matrix):
    return [[matrix[3*b+r][3*s+c] for b in range(3) for s in range(3) for c in range(3)] for r in range(3)]
"""

def extended_permutation_demo():
    from operator import concat
    from itertools import repeat, permutations, starmap,product,chain
    
    allperms=[permutations(range(3)),[tuple(range(3,6))], permutations(range(6,8))]
    #for prod in starmap(concat,zip(*fixperms)):
    #for prod in zip(*allperms):
    # for prod in product(*allperms):
        # print(list(chain.from_iterable(prod)))
    for prod in (product(*allperms)):
        print(list(chain.from_iterable(prod)))
    print("###")
    for prod in (product(permutations(range(3)),[tuple(range(3,6))], permutations(range(6,8)))):
        print(list(chain.from_iterable(prod)))
            
  

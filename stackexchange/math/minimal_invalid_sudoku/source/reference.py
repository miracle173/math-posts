
'''
test_simple_sudoku_01
'''

print("file =",__file__,": __name__ =",__name__)
from simple_sudoku import *
#from simple_sudoku.py.2026-01-18_220621.bak import *
from extensions_sudoku import *
from tools_sudoku import *
import trace_sudoku
from itertools import groupby
from time import time
from contextlib import redirect_stdout
from blockcheck_sudoku import isExtendibleBy, printMatrix, printExtendQuery,fromBlockwiseToMatrix, extractSymbolsAndConstraints,isFullExtensible


if trace_sudoku.ENABLE_TRACE:
    print("trace is enabled")
    filepart='trace'
else:
    print("trace is NOT enabled")
    filepart='notrace'


refdir='C:\\Users\\guent\\work\\sudoku\\testdir\\reffile'+filepart
testdir='C:\\Users\\guent\\work\\sudoku\\testdir\\testfile'+filepart
    
#filedir=refdir
filedir=testdir
N=4

with open(filedir+'_01.txt', 'w') as f:
    with redirect_stdout(f):
        print("all01BlockRepresentatives, N=%d"%N)
        for nr, (sudoku,_) in enumerate(all01BlockRepresentatives(N)):
            print("01Block matrix %3d"%nr)
            printMatrix(sudoku)
            print(stringMatrix(sudoku))
with open(filedir+'_02.txt', 'w') as f:
    with redirect_stdout(f):
        print("allWeightBlockRepresentatives, N=%d"%N)
        for nr, (sudoku,_) in enumerate(allWeightBlockRepresentatives(N)):
            print("WeightBlock matrix %3d"%nr)
            printMatrix(sudoku)
            print(stringMatrix(sudoku))
with open(filedir+'_03.txt', 'w') as f:
    with redirect_stdout(f):
        print("all01CellRepresentatives, N=%d"%N)
        for nr, (sudoku,_) in enumerate(all01CellRepresentatives(N)):
            printSudokuReduced("01Cell matrix %3d"%nr, sudoku)
            print(stringSudoku(sudoku))

with open(filedir+'_04.txt', 'w') as f:
    with redirect_stdout(f):
        print("allSymbolCellRepresentatives, N=%d"%N)
        for nr, sudoku in enumerate(allSymbolCellRepresentatives(N)):
            printSudokuReduced("SymbolCell matrix %3d"%nr, sudoku)
            print(stringSudoku(sudoku))
with open(filedir+'_05.txt', 'w') as f:
    with redirect_stdout(f):
        N=3
        ShowExtendible=True
        ShowUnextendible=True
        mySolvableCounter=SudokuConfigCounter("Solvable Sudokus")
        myUnclearCounter=SudokuConfigCounter("Unclear Sudokus")
        myCounter=SudokuConfigCounter("All Sudokus")
        for nr,sudoku in enumerate(allSymbolCellRepresentatives(N)):
            matrix=fromBlockwiseToMatrix(sudoku)
            myBlockIndices=isFullExtensible(matrix)
            if (ShowUnextendible and (myBlockIndices is None)):
                printSudokuReduced("matrix %3d"%nr, sudoku)
            elif (ShowExtendible) and myBlockIndices is not None:
                printSudokuReduced("matrix %3d"%nr, sudoku)
                print("Block Indices",myBlockIndices)
            if myBlockIndices is not None:
                mySolvableCounter.count(sudoku)     
            else:
                myUnclearCounter.count(sudoku)
            myCounter.count(sudoku)
        myCounter.print()
        mySolvableCounter.print()
        myUnclearCounter.print()
        
with open(filedir+'_06.txt', 'w') as f:
    with redirect_stdout(f):
        myConfig=((((0,0,0),(0,0,0),(0,0,0)),((0,0,0),(0,0,0),(0,0,0)),((3,0,0),(0,0,0),(0,0,0))),
            (((0,0,0),(0,0,0),(0,1,0)),((0,0,0),(0,0,0),(0,0,0)),((0,0,0),(0,0,0),(0,0,0))),
            (((0,0,0),(0,0,0),(0,0,0)),((0,0,0),(0,2,0),(0,0,4)),((0,2,0),(0,0,0),(0,0,0))))
        printSudokuReduced(normalizeMatrix(myConfig))

            
        # for N in [10, 100]:
            # startTime=time()    
            # s=0            
            # for _ in range(N):
                # s+=len(allWeightBlockRepresentatives(6))
            # endTime=time()
            # diffTime=endTime-startTime
            # print("modul %s"%str(optimizeBlock)[23:24],"per iteration %9.2e"%(diffTime/N), "iterations%6d"%N, "run time %7.2f"%(diffTime), "items %5d"%(s/N))
        # print()

        # for (matrix, transformation) in all01BlockRepresentatives(5):
            # printMatrix(matrix)
            # print(transformation)
        # printMatrix("start", [[0,1,0],[1,1,1],[1,0,0]])
        
        # for matrix in [((0,0,0),(0,0,1),(0,0,0)), ((0,1,0),(1,1,1),(1,0,0)), ((0,1,1),(1,0,1),(1,0,0))]:
            # print()
            # print()
            # print()
            # printMatrix("matrix", matrix)
            # for opt in [optimizeBlock3, optimizeBlock4]:
                # print()
                # solution=opt(matrix)
                # printMatrix(print(opt.__name__),solution[0])
                # for trans in sorted(solution[1]):
                    # print(trans)
                    # printMatrix("transformed", applyBlockFunction(trans,matrix))


import filecmp

allEqual=True
for suffix in ['01','02','03','04','05','06']:
    filesEqual=filecmp.cmp(
        refdir+'_'+suffix+'.txt',
        testdir+'_'+suffix+'.txt',
        shallow=False)
    if not filesEqual:
        print('file "%s" differs'%(testdir+'_'+suffix+'.txt'))
    allEqual=allEqual and filesEqual
        
if allEqual:
    print("All files match!!!")
else:
    raise ValueError("Files differ")
    

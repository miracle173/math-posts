'''
test_simple_sudoku_01
'''
print("file =",__file__,": __name__ =",__name__)
from simple_sudoku import *
#from simple_sudoku.py.2026-01-18_220621.bak import *
from extensions_sudoku import *
from tools_sudoku import *
#from simple_sudoku_bak import *
from itertools import groupby
from time import time
from contextlib import redirect_stdout
from blockcheck_sudoku import isExtendibleBy, printMatrix, printExtendQuery,fromBlockwiseToMatrix, extractSymbolsAndConstraints,isFullExtensible


            
        






with open('C:\\Users\\guent\\work\\sudoku\\out.txt', 'w') as f:
    # pass

#with open('C:\\Users\\guent\\OneDrive\\work\\sudoku\\out.txt', 'a') as f:
    with redirect_stdout(f):
        
        # aWeightBlock=((2,1,1),(0,0,0),(0,0,0))
        # for nr,(myCell01Matrix,myStabilizers) in enumerate(testGet01CellRepresentatives(aWeightBlock)):
            # printSudokuReduced("matrix %3d"%nr, myCell01Matrix)
            # for stab in myStabilizers:
                    # print("    ",stab)


        myConfig=((((0,0,0),(0,0,0),(0,0,0)),((0,0,0),(0,0,0),(0,0,0)),((3,0,0),(0,0,0),(0,0,0))),
            (((0,0,0),(0,0,0),(0,1,0)),((0,0,0),(0,0,0),(0,0,0)),((0,0,0),(0,0,0),(0,0,0))),
            (((0,0,0),(0,0,0),(0,0,0)),((0,0,0),(0,2,0),(0,0,4)),((0,2,0),(0,0,0),(0,0,0))))

        normalizeMatrix(myConfig)
        N=4
        
        # print()
        # print("all01BlockRepresentatives")
        # for nr, (sudoku,_) in enumerate(all01BlockRepresentatives(N)):
            # printMatrix("matrix %3d"%nr, sudoku)
        # print()
        # print("allWeightBlockRepresentative")
        # for nr, (sudoku,_) in enumerate(allWeightBlockRepresentatives(N)):
            # printMatrix("matrix %3d"%nr, sudoku)
            # print(stringMatrix(sudoku))

        # print()
        # print("all01CellRepresentatives")
        # for nr, (sudoku,_) in enumerate(all01CellRepresentatives(N)):
            # printSudokuReduced("matrix %3d"%nr, sudoku)
            # print(stringSudoku(sudoku))

        # myCounter=SudokuConfigCounter()
        # myCounter.start()
        # allResults=allSymbolCellRepresentatives(N)
        # myCounter.stop()
        # for nr, sudoku in enumerate(allResults):
            # myCounter.count(sudoku)
        # myCounter.print()
            
        # printSudokuReduced("matrix %3d"%nr, sudoku)

        #'''
        space=''
        ShowExtendible=False
        ShowUnextendible=True
        mySolvableCounter=SudokuConfigCounter("Solvable Sudokus")
        myUnclearCounter=SudokuConfigCounter("Unclear Sudokus")
        myCounter=SudokuConfigCounter("All Sudokus")
        myCounter.start()
        lnr=0
        for nr,sudoku in enumerate(allSymbolCellRepresentatives(N)):
            matrix=fromBlockwiseToMatrix(sudoku)
            myBlockIndices=isFullExtensible(matrix)
            if (ShowUnextendible and (myBlockIndices is None)):
                printSudokuReduced(space+"matrix %3d/%d"%(nr,lnr), sudoku)
                #printSudoku(space+"matrix %3d"%nr, sudoku)
                #print(stringSudoku(sudoku))
                #print()
                lnr+=1

            elif (ShowExtendible) and myBlockIndices is not None:
                printSudokuReduced(space+"matrix %3d/%d"%(nr,lnr), sudoku)
                #printSudoku(space+"matrix %3d"%nr, sudoku)
                #print(stringSudoku(sudoku))
                #print()
                lnr+=1
            if myBlockIndices is not None:
                mySolvableCounter.count(sudoku)     
            else:
                myUnclearCounter.count(sudoku)
            myCounter.count(sudoku)
        myCounter.stop()
        myCounter.print()
        mySolvableCounter.print()
        myUnclearCounter.print()
        
        #"""
            
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

from tools_sudoku import SudokuMatrixString, printSudoku,convertMatrixToSudoku, printMatrix
from blockcheck_sudoku import isExtendibleBy,printExtendQuery,extractSymbolsAndConstraints
from extensions_sudoku import stringSudoku
inputLine='''
+--+-+
|12|3|
|..|2|
+--+-+
'''

aBandIdx=1
aStackIdx=2

myMatrix=SudokuMatrixString.parseSudokuDiagram(inputLine)
aSymbols,aConstraints=extractSymbolsAndConstraints(myMatrix, aBandIdx, aStackIdx)
printSudoku(convertMatrixToSudoku(myMatrix))
print(stringSudoku(convertMatrixToSudoku(myMatrix)))
print(convertMatrixToSudoku(myMatrix))

print("Block",aBandIdx, aStackIdx)
print("Symbols",aSymbols)
print("Constraint",aConstraints)

printExtendQuery(aSymbols, aConstraints)

myFound=False
for myProblem in isExtendibleBy(aSymbols, aConstraints):
    myFound=True
    printMatrix(myProblem)
    break
if not myFound:
    print("Extendible by Block (%d,%d)"%(aBandIdx, aStackIdx))
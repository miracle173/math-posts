'''
test_blockcheck_sudok
'''

from blockcheck_sudoku import isExtendibleBy, printMatrix, printExtendQuery,fromBlockwiseToMatrix, extractSymbolsAndConstraints,isFullExtensible
from time import time
from contextlib import redirect_stdout                  

#with open('C:\\Users\\guent\\OneDrive\\work\\sudoku\\out.txt', 'w') as f:
with open('out.txt', 'w') as f:
    # pass

# with open('C:\\Users\\guent\\OneDrive\\work\\sudoku\\out.txt', 'a') as f:
    with redirect_stdout(f):
        
        
        print(
        '''
        (1)   (2)   (3)   (4)     (5)  (6)    (7)   (8)   (9)   (10)

        ...123 ...12 ...1  ...12  ...1  ...12  ...1  ...1  ...1  ...
        ...    ...3  ...2  ...    ...2  ...    ...2  ...   ...   ...
        ...    ...   ...3  ...    ...   ...    ...   ...   ...   ...
                           3      3        3      3  2  3     23    123
                           
        ''')
        
        boundaries4Clues=[
            [({1,2,3},),(),[]],
            [({1,2},{3},),(),[]],
            [({1},{2},{3},),(),[]],
            [({1,2},),({3},),[]],
            [({1},{2},),({3},),[]],
            [({1,2},),(),{3}],
            [({1},{2},),(),{3},],
            [({1},),({2},),{3}],
            [({1},),(),{2,3}],
            [(),(),{1,2,3}]]     
            
        boundaries3Clues=[
        [({1,2},),(),[]],
        [({1},{2}),(),[]],
        [({1},),({2},),[]],
        [({1},),(),{2}],
        [(),(),{1,2}],
        ]
        
        

            
        # N=0
        # symbols={2}
        # for nr, boundaries in enumerate(boundaries3Clues, start=1):
            # print()
            # print()
            # print("nr =",nr)
            ##print("result=isExtendibleBy(%s,%s)"%((symbols),str(boundaries)))
            # print("=======")
            # printExtendQuery(symbols, boundaries)
            # result=isExtendibleBy(symbols,boundaries)
            # if result == []:
                # print("    success")
            # else:
                # print("    failed")
                # for lnr, matrix in enumerate(result):
                    # printMatrix(matrix)
                    # print()
                    # if lnr>=N-1:
                        # break
                # print('and',len(result)-N,"other")
                    
                
aWikipediaSudoku=[
    # wikipedia
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9]]
    

aSudoku1=((((1, 0, 0), (0, 0, 0), (0, 0, 0)), ((2, 0, 0), (0, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0), (0, 0, 0))), (((2, 0, 0), (0, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 1, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0), (0, 0, 0))), (((0, 0, 0), (0, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0), (0, 0, 0))))

aSudoku2=((((1, 0, 0), (0, 0, 0), (0, 0, 0)), ((0, 0, 0), (1, 2, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0), (0, 0, 0))), (((0, 3, 0), (0, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0), (0, 0, 0))), (((0, 0, 0), (0, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0), (0, 0, 0))))
'''
matrix   6
+-+-+
|1|.|
|2|.|
|.|1|
+-+-+
Block Indices [(0, 1), (0, 2)]
((((1, 0, 0), (2, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0), (1, 0, 0)), ((0, 0, 0), (0, 0, 0), (0, 0, 0))), (((0, 0, 0), (0, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0), (0, 0, 0))), (((0, 0, 0), (0, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0), (0, 0, 0))))
'''

aSudoku=((((1, 0, 0), (2, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0), (1, 0, 0)), ((0, 0, 0), (0, 0, 0), (0, 0, 0))), (((0, 0, 0), (0, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0), (0, 0, 0))), (((0, 0, 0), (0, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0), (0, 0, 0))))
aMatrix=fromBlockwiseToMatrix(aSudoku)

myTransposed=list(zip(*aMatrix))
aBandIdx=1
aStackIdx=0

mySymbols, myConstraint = extractSymbolsAndConstraints(aMatrix, aBandIdx,aStackIdx)

#print(aSudoku)
print("=Matrix")

matrix991=[    
    [11,12,13,14,15,16,17,18,19,],
    [21,22,23,24,25,26,27,28,29,],
    [31,32,33,34,35,36,37,38,39,],
    [41,42,43,44,45,46,47,48,49,],
    [51,52,53,54,55,56,57,58,59,],
    [61,62,63,64,65,66,67,68,69,],
    [71,72,73,74,75,76,77,78,79,],
    [81,82,83,84,85,86,87,88,89,],
    [91,92,93,94,95,96,97,98,99,],
    ]

sudoku991=((((11,12,13),(21,22,23),(31,32,33)),((14,15,16),(24,25,26),(34,35,36)),((17,18,19),(27,28,29),(37,38,39))),
(((41,42,43),(51,52,53),(61,62,63)),((44,45,46),(54,55,56),(64,65,66)),((47,48,49),(57,58,59),(67,68,69))),
(((71,72,73),(81,82,83),(91,92,93)),((74,75,76),(84,85,86),(94,95,96)),((77,78,79),(87,88,89),(97,98,99))))
#printMatrix(matrix991)
print("converted")

printMatrix([[sudoku991[b][s][r][c] 
    for s in range(3) 
        for c in range(3) ] 
            for b in range(3) 
                for r in range(3)])

print()
#printMatrix(fromBlockwiseToMatrix(sudoku991))
print("==Matrix")
printMatrix(aMatrix)
print()
#printMatrix(myTransposed)

printExtendQuery(mySymbols, myConstraint)

#for matrix in isExtendibleBy(mySymbols, myConstraint):
    #print(matrix)
    #print()
    

print("isFullExtensible",isFullExtensible(aMatrix))



isExtendibleBy({3},(({2},),({1},),[1]))
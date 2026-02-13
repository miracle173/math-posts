from random import randint
  from copy import deepcopy
def removeItems(aItems, aComponentsToRemove):
    myRowComponentsToRemove, myColComponentsToRemove= aComponentsToRemove
    myOldRowComponents=set()
    myOldColComponents=set()
    myNewRowComponents=set()
    myNewColComponents=set()
    myKeptItems=set()
    for myItem in aItems:
        myRowComponent, myColComponent = myItem
        myOldRowComponents.add(myRowComponent)
        myOldColComponents.add(myColComponent)
        if (myRowComponent in myRowComponentsToRemove) or (myColComponent in myColComponentsToRemove):
            continue
        myNewRowComponents.add(myRowComponent)
        myNewColComponents.add(myColComponent)
        myKeptItems.add(myItem)
    """
    # debug
    print("myOldRowComponents", myOldRowComponents)
    print("myOldColComponents", myOldColComponents)
    print("myNewRowComponents", myNewRowComponents)
    print("myNewColComponents", myNewColComponents)
    """
    myNewRowComponentsToRemove=(myOldRowComponents-myNewRowComponents) 
    myNewColComponentsToRemove=(myOldColComponents-myNewColComponents) 
    myNewComponentsToRemove=(myNewRowComponentsToRemove, myNewColComponentsToRemove)
    return(myKeptItems, myNewComponentsToRemove)   

def cleanupGrid(aGrid, aPosition, aRemoveList):
    # modifies aGrid !!!
    myRowDim=len(aGrid)
    myColDim=len(aGrid[0])
    myRow,myCol=aPosition
    myControlGrid=[[[set(),set()]  for j in range(myColDim)] for i in range(myRowDim)]
    myControlGrid[myRow][myCol]=list(aRemoveList)
    myGridIsClean=False
    while not myGridIsClean:
        myGridIsClean=True
        for i in range(myRowDim):
            for j in range(myColDim):
                if myControlGrid[i][j]!=[set(),set()]:
                    assert(len(myControlGrid[i][j])==2)
                    print("debug")
                    print_grid("grid",aGrid)
                    print_grid("control",myControlGrid)
                    print("i,j",i,j)
                    print("from", aGrid[i][j])
                    print("remove", myControlGrid[i][j])

                    (myKeptItems, (myRowComponentes, myColComponents))=removeItems(aGrid[i][j], myControlGrid[i][j])
                    

                    print("remains", myKeptItems)
                    print("remove Rows", myRowComponentes)
                    print("remove Cols", myColComponents)


                    aGrid[i][j]=myKeptItems
                    if myRowComponentes:
                        myGridIsClean=False
                        for k in range(myRowDim):
                            if k!=i:
                                myControlGrid[k][j][0]|=myRowComponentes
                    if myColComponents:
                        myGridIsClean=False
                        for k in range(myColDim):
                            if k!=j:
                                myControlGrid[i][k][1]|=myColComponents
                    '''
                    print("debug","i,j", i,j)
                    print("debug", myControlGrid)
                    '''
    return aGrid
    
def print_grid(message, aGrid):
    print(message)
    myRowDim=len(aGrid)
    myColDim=len(aGrid[0])
    for i in range(myRowDim):
        for j in range(myColDim):
            print(i,",",j,":",aGrid[i][j])

"""
print({(3, 1), (1, 2), (3, 3)})
print(({1,2},{3,4}))
result=removeItems({(3, 1), (1, 2), (3, 3)},({1,2},{3,4}))
print(result)
"""

"""
# test
M=15
N=17
R=3


myPosition=(4,4)
myRemoveComponents=[{1,2},{3,4}]

myMatrix=[[{(randint(1,R), randint(1,R)) for _ in range(N)} for i in range(M)] for j in range(M)]
print_grid("matrix", myMatrix)
result=cleanupGrid(myMatrix,myPosition,myRemoveComponents)

"""


myMatrix=[[{(1,1),(1,2),(1,3),(2,1),(2,2)} ,{(1,2),(1,3),(2,1),(3,1)}], 
          [{(1,1),(3,2),(1,3),(3,1),(2,2)} ,{(1,2),(3,3),(3,1),(3,1)}]]
       
myPosition=(1,1)
myRemoveComponents=[{1,2},{3,4}]
print_grid("matrix", myMatrix)
myCopyMatrix=deepcopy(myMatrix)
result=cleanupGrid(myCopyMatrix,myPosition,myRemoveComponents)          



print("position", myPosition)
print("remove", myRemoveComponents)

print_grid("result", result)



"""
Matrix [[{(1, 2), (2, 1), (2, 2), (1, 1), (1, 3)}, {(3, 1), (1, 2), (1, 3), (2, 1)}], [{(2, 2), (3, 1), (1, 1), (1, 3), (3, 2)}, {(3, 1), (1, 2), (3, 3)}]]
position (1, 1)
remove ({1, 2}, {3, 4})
[[{(1, 1), (1, 2), (1, 3)}, {(3, 1), (1, 2), (1, 3), (2, 1)}], [{(3, 1), (1, 1), (1, 3)}, {(3, 1)}]]


(1, 2)   (3, 1)
(2, 1)   (1, 2)
(2, 2)   (1, 3)
(1, 1)   (2, 1)
(1, 3)

(2, 2)  |(3, 1)|
(2, 2)  |(1, 2)|
(2, 2)  |(3, 3)|
(3, 1) 
(1, 1) 
(1, 3) 
(3, 2)

----

 1   3 |  1   3
 2   4 |  2   4
 
Result
-------


(1, 1)   (3, 1)
(1, 2)   (1, 2)
(1, 3)   (1, 3)
         (2, 1)

(3, 1)   (3, 1)
(1, 1)  
(1, 3) 




"""

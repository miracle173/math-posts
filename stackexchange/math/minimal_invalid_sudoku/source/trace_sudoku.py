# trace_sudoku_02 open

# ENABLE_TRACE must set to True 
# False
ENABLE_TRACE=False


# trace flags
TRACE_GLOBALOPTIMIZATION=True
TRACE_NORMALIZEMATRIX=True
TRACE_ALL01BLOCKS=True
TRACE_ALLWEIGHTBLOCKS=True
TRACE_ALL01CELLS=True
TRACE_ALLSYMBOLCELLS=True

TRACE_TEST='' #'normalizeMatrix'

TRACE_SHORTFORMAT=True


TRACE_REJECT=True
TRACE_ACCEPT=True

TRACE_COUNTALL=True

TRACE_LOWEST=-1
TRACE_HIGHEST=-1

TRACE_PRINTHASH=True

TRACE_FUNTIONLIST=True


####################################################

"""
TRACE_GLOBALOPTIMIZATION = [ False | True ]
# trace the global optimiztation procedure

TRACE_ALL01BLOCKS= [ False | True ]
# trace the creation of thelist of the 01 block matrices

TRACE_REJECTOBJECT = [ False | True ]
# trace the objects that are rejected

TRACE_ACCEPTOBJECT = [ False | True ]
# trace the objects that are accepted

TRACE_COUNTALL = [ False | True ]
# True: the numbering counts all objects that are creted, even if they are not displayed
# False: only the objects that are displayed are counted


TRACE_LOWEST=-1
TRACE_HIGHEST=-1
# only the objects with a count between TRACE_LOWEST and TRACE_HIGHEST are printed, 
# A value of TRACE_HIGHEST <0  means there is no upper limit.
TRACE_PRINTHASH=True
# print a hash value to each entry of an object in the trace output

"""

#####################################################

from header_sudoku import *
from tools_sudoku import printMatrix,printSudoku, printSudokuReduced
from extensions_sudoku import hashMatrix, hashSudoku


# trace functions

def traceDummy(trcMessage, trcObject, note=None, step=None):
    pass
    


import inspect  # to retrieve name of calling procedure

myTraceObjectCount=0


def typeOfTraceObject(aObject):
    indexedType=set([type(tuple()),type(list())])
    if type(aObject   )==type(''):
        return("Text")
    elif (type(aObject) in indexedType):
        if type(aObject[0][0]) == type(1):
            return ('Matrix')
        elif (type(aObject[0][0][0])==type(True) 
            and (type(aObject[0][0][1]) in indexedType)):
            return ('Functionlist')
        elif type(aObject[0][0][0]) in indexedType:
            return 'Sudoku'
        else:
            assert(False)       
    else:
        assert(False)



def trace(trcMessage, trcObject, note=None, step=None):
    
    # get name of calling procedure
    # https://stackoverflow.com/a/2654130/754550
    myCurFrame = inspect.currentframe()
    myCalFrame = inspect.getouterframes(myCurFrame, 2)
    myProcedureName = myCalFrame[1][3]
    
    # get type of trace object
    myTraceObjectType=typeOfTraceObject(trcObject)
    
    myDoTraceOfProcedure=False
    if (myProcedureName == TRACE_TEST):  # just for developement of this trace implementation
        myDoTraceOfProcedure=True

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
        
    if (myProcedureName == "optimizeBlock1") and TRACE_GLOBALOPTIMIZATION:
        myDoTraceOfProcedure=True
    if (myProcedureName == "optimizeBlock2") and TRACE_GLOBALOPTIMIZATION:
        myDoTraceOfProcedure=True
    if (myProcedureName == "optimizeBlock3") and TRACE_GLOBALOPTIMIZATION:
        myDoTraceOfProcedure=True
    if (myProcedureName == "optimizeBlock4") and TRACE_GLOBALOPTIMIZATION:
        myDoTraceOfProcedure=True
    if (myProcedureName == "all01BlockRepresentatives") and TRACE_ALL01BLOCKS:
        myDoTraceOfProcedure=True
    if (myProcedureName == "filterGlobalOptimum") and TRACE_ALLWEIGHTBLOCKS:
        myDoTraceOfProcedure=True
    if (myProcedureName == "testGet01CellRepresentatives") and TRACE_ALL01CELLS:
        myDoTraceOfProcedure=True
    if (myProcedureName == "generate01SudokusFromWeights") and TRACE_ALL01CELLS:
        myDoTraceOfProcedure=True
    if (myProcedureName == "allSymbolCellRepresentatives") and TRACE_ALLSYMBOLCELLS:
        myDoTraceOfProcedure=True
    if (myProcedureName == "normalizeMatrix") and TRACE_NORMALIZEMATRIX:
        myDoTraceOfProcedure=True
        
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<        


    global myTraceObjectCount


    #is the object number in the range that should be printed?
    myCountInRange = ((TRACE_LOWEST<=myTraceObjectCount) and ((myTraceObjectCount<=
        TRACE_HIGHEST or TRACE_HIGHEST<0))) 

    myDoTraceOfAction=((trcMessage == trcMsgReject and TRACE_REJECT) 
        or (trcMessage == trcMsgAccept and TRACE_ACCEPT)
        or (trcMessage==trcMsgInfo))

    # should the trace statement be printed
    myPrintTrace =myCountInRange and myDoTraceOfProcedure and  myDoTraceOfAction  
    
    
    # should the object number be incremented
    if (TRACE_COUNTALL or myPrintTrace):
        if myTraceObjectType in {'Sudoku', 'Matrix'}:
            myTraceObjectCount+=1        
    
    
    if myPrintTrace:
        
        if trcMessage==trcMsgReject:
            myTraceMessage='reject'
        elif trcMessage==trcMsgAccept:
            myTraceMessage='accept'
        else:
            assert(trcMessage==trcMsgInfo)
            if myTraceObjectType=='Functionlist':
                myTraceMessage='Functionlist'
            elif myTraceObjectType == 'Text':
                myTraceMessage='Info'
            elif myTraceObjectType == 'Matrix':
                myTraceMessage='Matrix'
            elif myTraceObjectType == 'Sudoku':
                myTraceMessage='Sudoku'
            

        if step is None:
            myProcNameStep=myProcedureName
        else:
            myProcNameStep=myProcedureName+'@'+str(step)
            
        myAnnotation='trace %s, grid %d,'%(myProcNameStep, myTraceObjectCount)
        if myTraceObjectType=='Sudoku':
            if trcMessage in {trcMsgReject, trcMsgAccept}: # note will be ignored
                myAnnotation+=(' '+myTraceMessage)
            elif note is not None:
                myAnnotation+=(' '+note)
            if TRACE_SHORTFORMAT:
                printSudokuReduced(myAnnotation, trcObject)
            else:
                printSudoku(myAnnotation, trcObject)
            if TRACE_PRINTHASH:
                print("hash = ", hashSudoku(trcObject))
        elif myTraceObjectType=='Matrix':
            if trcMessage in {trcMsgReject, trcMsgAccept}: # note will be ignored
                myAnnotation+=(' '+myTraceMessage)
            elif note is not None:
                myAnnotation+=(' '+note)
            printMatrix(myAnnotation, trcObject)
            if TRACE_PRINTHASH:
                print("hash = ", hashMatrix(trcObject))
        elif myTraceObjectType=='Text':
            myAnnotation+=(' '+'info:')
            if note is not None:
                myAnnotation+=(' '+note)
            print(myAnnotation, trcObject)
        elif myTraceObjectType=='Functionlist' and TRACE_FUNTIONLIST:
            if note is not None:
                myAnnotation+=(' '+note)
            print(myAnnotation)
            for myFunction in trcObject:
                ((myReflection,(myBandPerm, myStackPerm)),
                 ((myPermR1,myPermR2,myPermR3),
                  (myPermC1,myPermC2,myPermC3)))=myFunction
                if TRACE_SHORTFORMAT:
                    print(": %s %d%d%d %d%d%d | %d%d%d-%d%d%d-%d%d%d %d%d%d-%d%d%d-%d%d%d"
                        %tuple(['-' if myReflection else '+']
                        +list(myBandPerm)+list(myStackPerm)+list(myPermR1)+list(myPermR2)
                        +list(myPermR3)+list(myPermC1)+list(myPermC2)+list(myPermC3)))
                else:
                    myFeed='  : '
                    if myReflection:
                        print(myFeed,'transpose')
                    else:
                        print(myFeed,'idendity')
                    print("Band Permutation %s   Stack Permutation %s"%(myBandPerm,myStackPerm))
                    print ("band  row :  0 %s   1 %s   2 %s"%(myPermR1,myPermR2,myPermR3))
                    print ("stack col :  0 %s   1 %s   2 %s"%(myPermC1,myPermC2,myPermC3))
        else:
            assert(False)

    
try:
    if ENABLE_TRACE:
        pass
    else:
        trace=traceDummy
except:
    trace=traceDummy

    
    
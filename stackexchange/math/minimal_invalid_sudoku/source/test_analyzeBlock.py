from itertools import product
from simple_sudoku import analyzeBlock

from contextlib import redirect_stdout

with open('C:\\Users\\guent\\OneDrive\\work\\python\\out.txt', 'w') as f:
    with redirect_stdout(f):
        N=3

        def compactify(line, start):
            #permutation=list(range(len(line)))
            permutation=[0,1,2]
            count=0
            for i in range(start,3):
                if line[i]>0:
                    count+=1
            compact=False
            resultLine=line[:]
            while not compact:
                compact=True
                for i in range(start+1,3):
                    if (resultLine[i-1]==0) and (resultLine[i]!=0):
                        resultLine[i],resultLine[i-1]=resultLine[i-1],resultLine[i]
                        permutation[i],permutation[i-1]=permutation[i-1],permutation[i]
                        compact=False
                        break
            return(resultLine,count, permutation)
         
        def simply_orig(block, baseRow=0):
            contigous=True
            rowPerm=list(range(3))
            zeroCount=0
            rowCount=0
            for r in range(baseRow,3):
                if not block[r]:
                    zeroCount+=1
                    rowPerm[3-zeroCount]=r
                else:
                    rowCount=r-zeroCount
                    rowPerm[rowCount]=r
                    rowCount+=1
                    if zeroCount>0:
                        contigous=False
            for r in range((rowCount+4)//2,3):
                rowPerm[r],rowPerm[3-r]=rowPerm[3-r],rowPerm[r]
            return (rowPerm,rowCount)


        def simply_neu(block, baseRow,permutation):
            contigous=True
            rowPerm=list(range(3))
            zeroCount=0
            rowCount=0
            for r in range(baseRow,3):
                if block[r]:
                    rowPerm[rowCount]=r
                    rowCount+=1
                else:
                    rowPerm[2-zeroCount]=r
                    zeroCount+=1
            hold=rowPerm[:]
            lowerBound=baseRow+rowCount
            upperBound=(baseRow+rowCount+2)//2
            for r in range(lowerBound,upperBound):
                rowPerm[r],rowPerm[2-r]=rowPerm[2-r],rowPerm[r]
    
            if permutation!=rowPerm:
                print("block",block)
                print("rowPerm old",hold)
                print("rowPerm new",rowPerm)
                print("correct",permutation)
                print("first 0",baseRow + rowCount)
                print("baseRow, rowCount, zeroCount", baseRow, rowCount, zeroCount)
                print("lowerBound,upperBound",lowerBound,upperBound)
                for r in (lowerBound,upperBound):
                    print("r,2-r", r,2-r)
                    hold[r],hold[2-r]=hold[2-r],hold[r]
                    print("changed",hold)
                print('##################################')
                print()
               
            return (rowPerm,rowCount)

        def chatGPT(m, rowStart, colStart):
            # --- Zeilen ---
            fixed_rows = list(range(rowStart))
            rest_rows = list(range(rowStart, 3))

            nonzero_rows = []
            zero_rows = []

            for i in rest_rows:
                if m[i] != 0 :
                    nonzero_rows.append(i)
                else:
                    zero_rows.append(i)

            row_perm = fixed_rows + nonzero_rows + zero_rows

            # 1-basierte Permutationen zurückgeben
            # return (
                # [i + 1 for i in row_perm], len(nonzero_rows)
            # )
            return row_perm, len(nonzero_rows)

        simply=chatGPT
        for base in range(N):
            for prod in product([0,1],repeat=N):
                (resultLine,count, permutation)=compactify(list(prod),base)
                rowPerm,rowCount=simply(list(prod),base,permutation)
                
                if count!=rowCount and permutation!=rowPerm:
                    print("==>",prod,base, resultLine)
                    print('ok  ', count, permutation)
                    print('    ', rowCount,rowPerm)
                elif count!=rowCount:
                    print("==>",prod,base, resultLine)
                    print('ok  ', count)
                    print('    ', rowCount)
                if permutation!=rowPerm:
                    print("==>",prod,base, resultLine)
                    print('ok  ', permutation)
                    print('    ', rowPerm)
             

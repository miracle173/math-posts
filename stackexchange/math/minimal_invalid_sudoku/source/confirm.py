from itertools import product
import simple_sudoku, tools_sudoku


# test "simple_sudoku.compose" function
if False:
    for perm1, perm2 in product(simple_sudoku.AllPermutations[3], repeat=2):
        perm3=simple_sudoku.compose(perm2,perm1)
        for data in [['a','b','c'],['d','r','g']]:
            # apply perm1 to data, then perm2
            image4=[data[perm2[perm1[i]]] for i in range(3)]
            image3=[data[perm3[i]] for i in range(3)]
            try:
                assert(image3==image4)
            except:
                print("data", data)
                print("perm1",perm1)
                print("perm2",perm2)
                print("image4",image4)
                print("perm3",perm3)
                print("image3",image3)
                raise
           



if True:
    #for matrix in simple_sudoku.generate01Matrices(3,3,5):
    for matrix in [[['a','b','c'],['d','e','f'],['g','h','i']]]:
        for (rowperm1,colperm1, rowperm2,colperm2), (reflect1, reflect2) in product(
            product(simple_sudoku.AllPermutations[3], repeat=4), 
            product([False, True], repeat=2)):
            gtrans1=(reflect1,(rowperm1,colperm1))
            gtrans2=(reflect2,(rowperm2,colperm2))
            # gtrans3=(reflect1 != reflect2,(
                # simple_sudoku.compose(rowperm2,rowperm1), 
                # simple_sudoku.compose(colperm2, colperm1)))
            gtrans3=simple_sudoku.composeBlockFunctions(gtrans2, gtrans1)
            image3=simple_sudoku.applyBlockFunction(gtrans3,matrix)
            image4=simple_sudoku.applyBlockFunction(gtrans2, simple_sudoku.applyBlockFunction(gtrans1, matrix))
            try:
                assert(image3==image4)
            except:
                tools_sudoku.printMatrix("matrix",matrix)
                print("gtrans3", gtrans3)
                print("compose row",simple_sudoku.compose(gtrans2[1][0],gtrans1[1][0]))
                print("compose col",simple_sudoku.compose(gtrans2[1][1],gtrans1[1][1]))
                tools_sudoku.printMatrix("composed",image3)
                print("gtrans1",gtrans1)
                print("gtrans2",gtrans2)
                tools_sudoku.printMatrix("applied",image4)
                raise
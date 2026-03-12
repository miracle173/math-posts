[This link](https://www.youtube.com/watch?v=4TnJhQPvZB0) points to a youtube video that illustrates the [proof of @quasi](https://math.stackexchange.com/a/4201256/11206). And this link points to a [cut-the-now article](https://www.cut-the-knot.org/proofs/2ColorsOnLine.shtml) about this topic with references to related problems.

The following simple Python program proves the more specific statement:
# Statement
> If there are 9 equidistant points on a line colored with two colors for at least three of these points holds:
>
> all three points have the same color and one point is the midpoint of the two other points.

Therefore it is a proof for the original statement, too.

# Program
    import itertools
    numberOfPoints=2
    while True:
        numberOfPoints+=1
        for coloring in itertools.product([0,1], repeat=numberOfPoints):
            foundGoodTripleForThatColoring=False
            for (point1,point2) in itertools.combinations(range(numberOfPoints),2):
                if (point2+point1)%2==0:
                    midpoint=(point1+point2)//2
                    if ((coloring[point1]==coloring[point2]) 
                        and (coloring[point1]==coloring[midpoint])):
                        foundGoodTripleForThatColoring=True
                        break
            if not foundGoodTripleForThatColoring:
                print(numberOfPoints, coloring)
                break
        if foundGoodTripleForThatColoring:
            print("triples always found for number of points =",numberOfPoints)
            break  
            
[itertools.product](https://docs.python.org/3/library/itertools.html#itertools.product) and [itertools.combinations](https://docs.python.org/3/library/itertools.html#itertools.combinations) are described in the [Python documentations](https://docs.python.org/3/library/itertools.html).

## Combinatoric iterators:

|Iterator | Arguments | Results|
|---------|-----------|--------|
|product()|p, q, … [repeat=1] |cartesian product, equivalent to a nested for-loop|
|combinations()|p, r|r-length tuples, in sorted order, no repeated element|

|Examples| Results|
|--------|--------|
|product('ABCD', repeat=2)|AA AB AC AD BA BB BC BD CA CB CC CD DA DB DC DD|
|combinations('ABCD', 2)|AB AC AD BC BD CD|

# The Output of the Program
For each number of points it shows one coloring with the colors 0 and 1 were there are no three points with the requested property. For 9 points no such coloring exists.

    3 (0, 0, 1)
    4 (0, 0, 1, 0)
    5 (0, 0, 1, 0, 0)
    6 (0, 0, 1, 0, 0, 1)
    7 (0, 0, 1, 0, 0, 1, 1)
    8 (0, 0, 1, 1, 0, 0, 1, 1)
    triples always found for number of points = 9

If the number of points is n the program has to check $2^n \frac 1 2 \binom n  2 \approx 2^{n-2} n^2$ triples.
By changing the line 

    for coloring in itertools.product([0,1], repeat=numberOfPoints):
    
to

    for coloring in itertools.product([0,1,2], repeat=numberOfPoints):
    
we could check if there exists such a triple if we use three different colors. Now the runtime for a given n is $\approx 3^{n} n^2 / 4$. I ran this modified program but stopped it after a while after it outputs the following line

    20 (0, 0, 1, 0, 0, 1, 1, 2, 2, 0, 0, 1, 0, 0, 1, 1, 2, 2, 1, 2)

The above statement for two colors can easily be proved without using a program:
# Proof
We can assume that a sequence always starts with `0`. Otherwise we will swap the colors. There are two possible beginnings: `001` and `011` and then the following colors are uniquely determined: after `00` and `01`can only follow `1`, after `10` and `11` can only follow `0`. So there are the following sequences starting with `0`:
    
    123456789
    ---------
    00110011
    01100110
    
On position $9$ the rules require a `0` but in this case the colors at position $1$,$5$ and $9$ are all `0`, so the sequences cannot be extended further.



The following Python program proves the following more specific statement:
> If there are 9 equidistant points on a line colored with two colors for at least three of these points holds:
> all points have the same color and one point is the midpoint of the two other points.

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
                foundNoGoodTripleForThatNumberOfPoints=True
                print(numberOfPoints, coloring)
                break
            foundNoGoodTripleForThatNumberOfPoints=False
        if not foundNoGoodTripleForThatNumberOfPoints:
            print("no good triple found for numberOfPoints =",numberOfPoints)
            break
The output of the program
For each number of points it shows one coloring with the colors 0 and 1 were there are no three points with the requested property. For 9 points no such coloring exists.

    3 (0, 0, 1)
    4 (0, 0, 1, 0)
    5 (0, 0, 1, 0, 0)
    7 (0, 0, 1, 0, 0, 1, 1)
    8 (0, 0, 1, 1, 0, 0, 1, 1)
    no good triple found for numberOfPoints = 9



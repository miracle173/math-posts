[This link](https://www.youtube.com/watch?v=4TnJhQPvZB0) points to a youtube video that illustrates the [proof of @quasi](https://math.stackexchange.com/a/4201256/11206). And this link points to a [cut-the-now article](https://www.cut-the-knot.org/proofs/2ColorsOnLine.shtml) about this topic with references to related problems.

But here is another way to prove the statement. We prove the more specifific statement:
**Statement 1:**
> If there are 9 equidistant points on a line colored with two colors for at least three of these points holds:
>
> all three points have the same color and one point is the midpoint of the two other points.

or equivalently:

**Statement 1:**
If there is a sequence $(c_0, c_1, c_2, \dots)$, where $c_i\in\{0,1\}, \forall  i \in \mathbb{N_0}$, then there exist three indices $i,m,j \in \mathbb{N_0}$ such that  $i<j$, $m=(i+j)/2$ and $c_i=c_m=c_j$.

From Statement 1 the original statement follows. If we have a coloring of the real line, we get a coloring of the points $0,1,2,\dots$ of the real line and from Statement 1 follows that three points have the requested property.

We can check all sequences of a given length of they have the property

$\textbf{for } n=3,4,5,\dots:$  
$\hspace{10pt}\text{check if for all colorings of the points }0,\dots,n-1 $  
$\hspace{10pt}\text{there is at least one triple }i,j,m \text{ in } \{0,...,n-1\}$  
$\hspace{10pt} \text{such that } m=(i+j)/2$  
$\hspace{10pt} \text{and } color[i]=color[j]=color[m].$  
$\hspace{20pt}\text{if so then we are done. Output }n\text{  and exit program}$  
$\hspace{20pt}\text{if not, output the color sequence that}$  
$\hspace{30pt}\text{does not have such a triple and try next }n$  

A more detailed description of the algorithm:

$n\leftarrow 2$  
$\textbf{repeat}:$  
$\hspace{10pt}n\leftarrow n+1$  
$\hspace{10pt}all\_sequences\leftarrow \textbf{true}$   
$\hspace{10pt}\textbf{for each } \text{0-1-sequence}\; color \; \text{of length}\; n$:  
$\hspace{20pt}found\_triple?\textbf{false}$  
$\hspace{20pt}\textbf{for each}\;\text{subset}\; \{i,j\} \; \text{of} \; \{0,\dots,n-1\}$:  
$\hspace{30pt}\textbf{if}\; (i+j) \bmod 2 = 0$:  
$\hspace{40pt}m\leftarrow \frac{i+j}2$  
$\hspace{40pt}\textbf{if} \;(color_i=color_j)  \;\textbf {and }\; (color_j=color_m)$:  
$\hspace{50pt}/\!*\textit{a triple was found}*\!/$  
$\hspace{50pt}found\_triple\leftarrow \textbf{true} $  
$\hspace{40pt}\textbf{endif}$  
$\hspace{30pt}\textbf{endif}$  
$\hspace{20pt}\textbf{endfor} $  
$\hspace{20pt}all\_sequences\leftarrow (all\_sequences\textbf{ and }found\_triple)$  
$\hspace{10pt}\textbf{endfor} $  
$\hspace{0pt}\textbf{until }all\_sequences$  
$\textbf{write }\text{"for each sequence of length "},n,\text{"a triple exists"}$    

The following simple Python program implements the algorithm. So we can start it an lean back and wait if if will finish.

**Program:**  

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



**The Output of the Program**  

For each number of points it shows one coloring with the colors 0 and 1 were there are no three points with the requested property. For 9 points no such coloring exists.

    3 (0, 0, 1)
    4 (0, 0, 1, 0)
    5 (0, 0, 1, 0, 0)
    6 (0, 0, 1, 0, 0, 1)
    7 (0, 0, 1, 0, 0, 1, 1)
    8 (0, 0, 1, 1, 0, 0, 1, 1)
    triples always found for number of points = 9

This result tells us that each 0-1-sequence of length $9$ has the requested property and therefore each sequence of a lenth larger than 9.

so we have proven that on the real line coloured with two colours always three points with the requested property exist.

Statement 1  can easily be proved without using a program:

**Proof of Statement 1**  

We can assume that a sequence always starts with `0`. Otherwise we will swap the colors. There are two possible beginnings: `001` and `011` and then the following colors are uniquely determined: after `00` and `01`can only follow `1`, after `10` and `11` can only follow `0`. So there are the following sequences starting with `0`:
    
    123456789
    ---------
    00110011
    01100110
    
On position $9$ the rules require a `0` but in this case the colors at position $1$,$5$ and $9$ are all `0`, so the sequences cannot be extended further.


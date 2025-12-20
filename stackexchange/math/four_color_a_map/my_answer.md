# https://math.stackexchange.com/questions/3247841/how-can-this-planar-map-be-coloured-using-4-colours

Asumme that all countries of a map are already colored except one. And this one country has three neigbourghs. Then we can easily complete
the coloring because there is at least one color that is not assigned ot one of the three neigburghs and such a color can be assigned to this 
remaining country. If the remaining country ha four neigbourghs that have four different colors then it is possible to change the color of one 
neighbourgh to the color of its opposite neighbor. then only three different colors are used by the four countries and the fourth colour can be 
used to color the still uncolored country. This mehod was already dicovered by Kemke. 
Kemke also published a method to recolor a graph when the uncolored country has five neighbourghs that use all four colors. But unfortunately the 
proof of this medthod contained an error and there are situations where the method won't work. But one can try by repeatingly applying this method 
to succeed but it is unclear if this will work. 

The following theorem is well knonw and follows from Eulers Theorem:

Theorem:
Each map has at least one country, that has not more than five neighbourghs. 

Assume that already all but one node of a the graph are clolored by four colors. I this last node has 
only 3 or less neigbors we cann assigne immediately a color that is not used by its neighbors to this node and 
the whole graph is colored by using only 4 colours.

If the region has exactly four neighbors with pairwise different colors. 


If there are four neighbors 
 
For each node we list its neighbors (adjacency list)
    01: 02,04,09,12,14
    02: 01,03,04,06,07,08,10
    03: 02,07,08,12,13,15
    04: 01,02,05,06,09
    05: 04,06,09
    06: 02,04,05,07,09,10,11,14,15
    07: 02,03,06,10,15
    08: 02,03
    09: 01,04,05,06,11,14
    10: 02,06,07
    11: 06,09,14
    12: 01,03,13,14,15
    13: 03,12
    14: 01,06,09,11,12,15
    15: 03,06,07,12,14

Now we remove all nodes with three or less neighbors

    remove 05,08,10,11,13

The remainig grapth has only nodes with four or more neighbors. We remove a nodee with four neighbors

    remove 12 (4 neighbors)

Now we again have  nodes with three neighbors and we remove tthese node:
    
    remove 03

an we repeat this process

    remove 07,15
    remove 02,14

the remaining graph ist

    01: 04,09
    04: 01,06,09
    06: 04,09,14
    09: 01,04,06

and we can start with the coloering


    set color for node 01,04,06,09
    set color for node 02,14
    set color for node 07,15:
    set color for node 03
    set color for node 12 and swap

    r 01: 02,04,09,12,14
    y 02: 01,03,04,06,07
    b 03: 02,07,12,15
    g 04: 01,02,06,09
    b 06: 02,04,07,09,14,15
    g 07: 02,03,06,15
    y 09: 01,04,06,14
    ? 12: 01,03,14,15
    g 14: 01,06,09,12,15
    y 15: 03,06,07,12,14

12; 1-14-15-3-

1-15: ry  r:1, y:2,9; r:none

swap r y  01,02,09
set color 12 r

    y 01: 02,04,09,12,14
    r 02: 01,03,04,06,07
    b 03: 02,07,12,15
    g 04: 01,02,06,09
    b 06: 02,04,07,09,14,15
    g 07: 02,03,06,15
    r 09: 01,04,06,14
    r 12: 01,03,14,15
    g 14: 01,06,09,12,15
    y 15: 03,06,07,12,14


    set color for node 05,08,10,11,13
    y 01: 02,04,09,12,14
    r 02: 01,03,04,06,07,08,10
    b 03: 02,07,08,12,13,15
    g 04: 01,02,05,06,09
    y 05: 04,06,09
    b 06: 02,04,05,07,09,10,11,14,15
    g 07: 02,03,06,10,15
    y 08: 02,03
    r 09: 01,04,05,06,11,14
    y 10: 02,06,07
    y 11: 06,09,14
    r 12: 01,03,13,14,15
    g 13: 03,12
    g 14: 01,06,09,11,12,15
    y 15: 03,06,07,12,14


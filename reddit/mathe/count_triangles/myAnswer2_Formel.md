In einem der Kommentare hier wird gefragt, ob es eine Formel zur Berechnung dieser Dreiecke gibt. Ein anderer Kommentar meint, dass möglicherweise die Graphentheorie Methoden dafür liefert. Fragen wir die Google-KI, was sie zur Anzahl der Dreiecke in Graphen zu sagen gibt, dann erhalten wir die folgende Antwort:

>"Die Anzahl der Dreiecke in einem Graphen kann mit der Methode der Adjazenzmatrix berechnet werden: Wenn A die Adjazenzmatrix des ungerichtetenGraphen ist, dann ist die Anzahl der Dreiecke gleich trace(A^3)/6. Für einen vollständigen Graphen mit n Knoten K_n ist die Anzahl der Dreiecke gegeben durch die Formel (n über 3), die (n(n-1)(n-2))/6 entspricht."

Die Graphentheoretiker unter uns werden uns die Korrektheit dieser Aussagen  bestätigen, und diese reichen um eine Formel für unser Problem zu finden.

Wir konstruieren uns aus der Zeichnung einen Graphen auf folgende Art:
1. Die Knoten sind die Schnittpunkte der Geraden
2. Zwei Knoten haben eine gemeinsame Kante, wenn sie auf einer Geraden liegen.

Wenn wir in unserem Bild die Schnittpunkte von links oben nach rechts unten durchnummerieren, wie ich im Post
(https://www.reddit.com/r/mathe/comments/1nqw129/comment/ngaa9g6) beschrieben habe, bedeutet das, das nicht nur (1,2) und (2,3) Kanten sind, sondern auch (1,3).
Ich habe die Berechnung mit dem CAS  Maxima durchgeführt:



    (%i1) A: matrix(
     [0,1,1,0,0,0,1,0,0], 
     [1,0,1,1,1,1,1,1,1], 
     [1,1,0,0,0,0,0,0,1], 
     [0,1,0,0,1,1,1,1,0], 
     [0,1,0,1,0,1,0,1,0], 
     [0,1,0,1,1,0,0,1,1], 
     [1,1,0,1,0,0,0,1,1], 
     [0,1,0,1,1,1,1,0,1], 
     [0,1,1,0,0,1,1,1,0]
    )$

    (%i2) A^^3
    (%o2) matrix(
      [ 4, 14,  8,  7,  7, 10, 12, 10,  7],
      [14, 28, 14, 24, 21, 24, 23, 27, 23],
      [ 8, 14,  4, 10,  7,  7,  7, 10, 12],
      [ 7, 24, 10, 16, 16, 20, 19, 21, 14],
      [ 7, 21,  7, 16, 12, 16, 13, 19, 13],
      [10, 24,  7, 20, 16, 16, 14, 21, 19],
      [12, 23,  7, 19, 13, 14, 12, 20, 19],
      [10, 27, 10, 21, 19, 21, 20, 22, 20],
      [ 7, 23, 12, 14, 13, 19, 19, 20, 12]
     )
     
    (%i3) mat_trace(A^^3)/6;
    (%o3) 21

Die Trace-Funktion `mat_trace`liefert  126, also 21 und nicht 15 Dreiecke, weil auch die degenerierten Dreiecke, deren Eckpunkte jeweils auf einer Geraden liegen, mitgezählt werden. Man muss also die Anzahl der degenerierten Dreiecke abziehen. In unserem Fall haben wir sechs Gerade, auf denen jeweils ein degeneriertes Dreieck liegt. Eines ist zum Beispiel 1-2-3. Liegen auf einer Gerade genau  k Punkte, dann können daraus C(k,3) verschiedene degenerierte Dreiecke gebildet werden. Diese Anzahl der degenerierten Dreiecke muss dann von der Anzahl aller Dreiecke abgezogen werden, damit wir die Anzahl der nicht degenerierten Dreiecke bekommen. Diese Anzahl ist also

    trace(A^3)/6-Summe(g(k)*C(k,3)k=3,...,n)

Dabei ist n die Anzahl der Knoten des Graphs und g(k) die Anzahl der Geraden, auf denen genau k Punkte liegen.

Erklärung:
Die Adjazenzenmatrix eines Graphen mit den Knoten 1,...,n ist eine nxn-Matrixm A=(a_ij), der Koeffizient a_ij 1 ist, wenn die Knoten i und j durch eine Kante verbunden sind, 0 wenn nicht. A kann auch so interpretiert werden: a_ij ist die Anzahl der Pfade vomKNoten i zum Knoten j. Man Kann zeigen: der i-j-Koeffizient der Matrix A^m ist die Anzahl der Pfade der Länge m vom Knoten i zum Knoten j. Die i-i-Komponente von A^3 ist dann ie Anzahl der geschlossenen Pfade der Länge 3 zum Knoten i. Der Trace ist die Summe der Elemente der Hauptdiagonale einer Matrix, also der i-i-Komponenten. Jedes Dreieck kann durch 6 Pfade dargestellt werden, da es drei verschieden Startpunkte für einen Pfad gibt, und zwei Richtungen. Also muss der Trace durch 6 geteilt werden. Danach muss wie oben beschrieben noch die Anzahl der degenerierten Dreiecke abgezogen werden.


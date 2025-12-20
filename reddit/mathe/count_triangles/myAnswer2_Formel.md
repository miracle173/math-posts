https://www.reddit.com/r/mathe/comments/1nqw129/comment/nheaz4k/

In einem der Kommentare hier wird gefragt, ob es eine Formel zur Berechung dieser Dreiecke gibt. Ein anderer Kommentr meint, dass möglicherweise die Graphentheorie Methoden dafür liefert. Fragen wir die Google-KI, was sie zur Anzahl der Dreiecke in Graphen zu sagen gibt, dann erhalten wir die folgende Antwort:
>"Die Anzahl der Dreiecke in einem Graphen kann mit der Methode der Adjazenzmatrix berechnet werden: Wenn A die Adjazenzmatrix des ungerichtetenGraphen ist, dann ist die Anzahl der Dreiecke gleich trace(A^3)/6. Für einen vollständigen Graphen mit n Knoten K_n ist die Anzahl der Dreiecke gegeben durch die Formel (n über 3), die (n(n-1)(n-2))/6 entspricht."
Die Graphentheoretiker unter uns werden uns die Korrektheit dieser Aussagen  bestätigen, und diese reichen um eine Formel für unser Problem zu finden.
Wir konstruieren uns aus der Zeichnung einen Graphen auf folgende Art:
Die Knoten sind die Schnittpunkte der Geraden
Zwei Knoten haben eine gemeinsame Kante, wenn sie auf einer Geraden liegen.
Die Formel ist dann damit wir die Anzahl der nicht degenerierten Dreiecke bekommen. Diese Anzahl ist also
trace(A^3)/6-Summe(g(k)*C(k,3)k=3,...,n)
Dabei ist n die Anzahl der Knoten des Graphs und g(k) die Anzahl der Geraden, auf denen genau k Punkte liegen.
Erklärung:
Die Adjazenzenmatrix eines Graphen mit den Knoten 1,...,n inst eine nxn-Matrix A=(a_ij), dere Koeffizient a_ij 1 ist, wenn die Knoten i und j durch eine Kante verbunden sind, 0 wenn nicht. A kann auch so interpretiert werden: a_ij ist die Anzahl der Pfade vom Knoten i zum Knoten j. Man Kann zeigen: der i-j-Koeffizient der Matrix A^m ist die Anzahl der Pfade der Länge m vom Knoten i zum Knoten j. Die i-i-Komponente von A^3 ist dann ie Anzahl der geschlossenen Pfade der Länge 3 zum Knoten i. Der Trace ist die Summe der Elemente der Hauptdiagonale einer Matrix, also der i-i-Kpmüoneneten. Jedes Dreiek kann durch 6 Pfade dargestellt werden, da es drei verschiden Startpunkte für einen Pfad gibt, und zwei Richtungen. Also muss der Trace durch 6 geteilt werden. Danach muss noch die Anzahl der degenerierten Dreiecke abgezogen werden, das sind jeden, deren drei Eckpunkte auf einer Geraden. liegen,

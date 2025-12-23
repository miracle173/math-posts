#Sortierreihenfolge für n Clues:

## Komponenten des Sortierschlüssels
1. Absteigend nach der Anzahl der insgesamt belegten Blöcke sortiert

2. Absteigend, die belegten Blöcke der 3*3 Matrix von links nach rechts und von oben nach unten als Binärstring interpretiert

3. Absteigend nach der Anzahl der Clues pro Block, Blöcke wie bei (2.)

4. Absteigend, die belegten Zellen,pro Block von links nach rechts und von oben nach unten, die Blöcke in der Reihenfolge von 2

5. Aufsteigend, die Zellen von links nach rechts und von oben nach unten, die Blöcke von links nach rechts und von oben nach unten, werden bei den Zellen die Werten durch 1,2,3,... ausgetauscht

6. Aufsteigend, die Zellen von links nach rechts und von oben nach unten, die Blöcke von links nach rechts und von oben nach unten, die Werte der Zellen


Anmerkungen:
Von links nach rechts und von oben nach unten bedeutet, es werden zuerst die Elemente der ersten Zeile angegeben, dann die der zweiten usw.

1. Die Anzahl der Blöcke ist immer zwischen 1 und der Anzahl der Clues

2. Die Reihenfolge ist also B_{1,1}, B_{1,2}, B_{1,3}, B_{2,1}, B_{2,2}, B_{2,3},  B_{3,1}, B_{3,2}, B_{3,3}.
Wenn wir reduzierte Matrizen betrachten, müssen für den Vergleich auf dieser Stufe die expandierte Matrizen betrachtet werden, wie das folgende Beispiel zeigt:

<pre><code>
Blockmatrices Example:

Reduced:
110
101

ReducedId:
110101

ReducedRank:
2, 110101 < 111001


Expanded:
110
101
000

ExpandedId:
110101000

ExpandedRank:
1, 110101000 > 110100010


reflected Matrix:

Reduced:
11
10
01

ReducedId:
111001 

ReducedRank:
1, 111001 > 110101

Expanded:
110
100
010

ExpandedId:
110100010

ExpandedRank
2, 110100010 < 110101000
</code></pre>

3. Hier reicht es, nach die Folge der Belegungszahlen, in der in 2 gegebenen Reihenfolge, also von oben 
links nach unten rechts, unter Weglassung der 0, zu sortieren. Bei den  Matrizen, die auf diesem Level verglichen werden,   stimmen die Folgen auf den weggelassenen Blöcken überein.


4. Hier reicht es, die nichtleeren Blöcke zu vergleichen. Diese müssen aber auf vollständige 3*3-Blöcke expandiert werden. Auf den weggelassenen Blöcken stimmen die Folgen deshalb überein.

5. Hier reicht es, die nichtleeren Elemente zu anzuführen, in der Reihenfolge Block für Block, Zeile für Zeile, . Auf den weggelassenen Blöcken stimmen die Folgen deshalb überein.

6. Hier reicht es, die nichtleeren Elemente zu anzuführen, in der Reihenfolge Block für Block, Zeile für Zeile, . Auf den weggelassenen Blöcken stimmen die Folgen deshalb überein. 

##Beispiel:

<pre><code>
.9.|86
4..|..
---+--
..6|..
</pre></code>


1. es sind 4 Blöcke belegt, das ist der erste Schlüssel, nachdem sortiert wird, die Zahl 4.

2. Die Blockmatrix ist

<pre><code>
11
10
</pre></code>
die expandierte Matrix ist
<pre><code>
110
100
000
</pre></code>
und der Schlüssel, nach dem sortiert wird, ist
<pre><code>
110 100 000
</pre></code>

Die Abstände in dieser Darstellung des  Sortierschlüssel dienen nur zur Verdeutlichung und müssen bei der Implementation des Sortierschlüssels weggelassen werden.

3. Die Anzahl der belegten Zellen pro Block ist
<pre><code>
220
100
000
</pre></code>

beim Sortierschlüssel können die leeren Blöcke weggelassen werden, ergibtist also
<pre><code>
22 1
</pre></code>

Die Abstände in dieser Darstellung des  Sortierschlüssel dienen nur zur Verdeutlichung und müssen bei der Implementation des Sortierschlüssels weggelassen werden.

4. Die binäre Matrix ist
<pre><code>
010|11
100|00
---+--
001|00
</pre></code>
der Schlüsselwert wird von

<pre><code>
010|110
100|000
000|000
---+--
001|
000|
000|
</pre></code>
gebildet, also

<pre><code>
010100000 110000000 001000000
</pre></code>

Die Abstände in dieser Darstellung des  Sortierschlüssel dienen nur zur Verdeutlichung und müssen bei der Implementation des Sortierschlüssels weggelassen werden.

5. Wir substituieren nun die Zahlen der Folge
<pre><code>
 94 86 6
</code></pre
   von links nach rechts durch 1,2,3,... und erhalten

<pre><code>
12 34 4
</pre></code>
die wir von der Anzahl der Clues substrahiern. Da die Anzahl der Clues 5 ist, haben wir

<pre><code>
43 21 1
</pre></code>
Die Abstände in dieser Darstellung des  Sortierschlüssel dienen wiederum nur zur Verdeutlichung und müssen bei der Implementation des Sortierschlüssels weggelassen werden.

Anmerkung:
Wenn man die beiden letzten Spalten der Matrix vertauscht, erhält man

<pre><code>
.9.|68
4..|..
---+--
..6|..
</pre></code>
Substituiert man hier durch 1,2,3,... erhält man
<pre><code>
.1.|34
2..|..
---+--
..3|..
</pre></code>
also den Teilschküssel
<pre><code>
12 34 3
</pre></code>
bzw,
<pre><code>
43 21 2</pre></code>
also einen lexikographisch größeren Wert
Dies ist auch der größte Wert, der in dieser Äquivalenzklasse erreicht werden kann.

Die SortId ist also
<pre><code>
4 110100000 221 010100000110000000001000000 43211
</pre></code>
Die Abstände hier dienen zur Trennung der Sortierschlüsselkomponenten und sind deshalb Teil des Schlüssels.

from copy import deepcopy
from typing import List, Sequence, Tuple, Any
from itertools import combinations, product, permutations
import itertools




class SegmentedMatrix:
    """
    SegmentedMatrix with internal flat row/column storage:
      self.data[row][col]

    The segmentation is defined by:
      self.bandWidths  # list of ints, sum = number of rows
      self.stackWidths # list of ints, sum = number of cols

    Accessors still allow box-style addressing:
      m[bandIndex][stackIndex][rowIndex][colIndex]
    """

    # class variables
    cntRotate=0
    cntBandStackPermute=0
    cntRowColPermute=0
    cntSymbolPermute=0

    # -------------------------
    # Constructor / basic API
    # -------------------------
    def __init__(self, aBandWidths: Sequence[int] = None, aStackWidths: Sequence[int] = None):
        if aBandWidths is None:
            aBandWidths = []
        if aStackWidths is None:
            aStackWidths = []

        # validation: at most 3 bands/stacks and each width 1..3
        if len(aBandWidths) > 3 or len(aStackWidths) > 3:
            raise ValueError("At most 3 bands and 3 stacks are allowed.")
        if any(w < 1 or w > 3 for w in (list(aBandWidths) + list(aStackWidths))):
            raise ValueError("Each band/stack width must be in range 1..3.")

        self.bandWidths: List[int] = list(aBandWidths)
        self.stackWidths: List[int] = list(aStackWidths)

        # initialize data (rows × cols) filled with zeros
        rows = sum(self.bandWidths)
        cols = sum(self.stackWidths)
        self.data: List[List[int]] = [[0] * cols for _ in range(rows)]
        self._sortPrefix = []  # writable list, initially empty
        self.info = None   # free-form metadata

    @classmethod
    def resetPermutationCounters(cls):
        cls.cntRotate=0
        cls.cntBandStackPermute=0
        cls.cntRowColPermute=0
        cls.cntSymbolPermute=0

    @classmethod
    def printPermutationCounters(cls):
        print("cntRotate =", cls.cntRotate)
        print("cntBandStackPermute =", cls.cntBandStackPermute)
        print("cntRowColPermute =", cls.cntRowColPermute)
        print("cntSymbolPermute =", cls.cntSymbolPermute)


    # -------------------------
    # Properties
    # -------------------------
    @property
    def rowDim(self) -> int:
        return len(self.data)

    @property
    def height(self) -> int:
        """Total number of rows."""
        return sum(self.bandWidths)

    @property
    def width(self) -> int:
        """Total number of columns."""
        return sum(self.stackWidths)
    

    @property
    def colDim(self) -> int:
        return len(self.data[0]) if self.data else 0

    @property
    def sortPrefix(self):
        return self._sortPrefix

    @sortPrefix.setter
    def sortPrefix(self, value):
        # Must be a list
        if not isinstance(value, list):
            raise TypeError("sortPrefix must be a list")
        self._sortPrefix = value

    # 2025-12-16 22:18 begin
    @property
    def sortId(self):
        """
        sortId = [*sortPrefix, 81 Zellen]

        Interpretation als 9x9 SegmentedMatrix mit
          bandWidths  = [3,3,3]
          stackWidths = [3,3,3]

        Ausgabereihenfolge:
          Band →
            Stack →
              Zeilen des Blocks →
                Spalten des Blocks

        Auffüllen:
          - fehlende Zeilen: unten IM SELBEN Band
          - fehlende Spalten: rechts IM SELBEN Stack
          - fehlende Bands: 3 Nullzeilen
          - fehlende Stacks: 3 Nullspalten
        """

        cells = []

        B = len(self.bandWidths)
        S = len(self.stackWidths)

        band_row_offset = 0

        # --- Bands ---
        for bi in range(3):
            bw = self.bandWidths[bi] if bi < B else 0

            stack_col_offset = 0

            # --- Stacks ---
            for sj in range(3):
                sw = self.stackWidths[sj] if sj < S else 0

                # --- Zeilen im Block ---
                for r in range(3):
                    real_row = (r < bw)

                    if real_row and bi < B:
                        row = self.data[band_row_offset + r]
                    else:
                        row = None  # Nullzeile im Block

                    # --- Spalten im Block ---
                    for c in range(3):
                        if row is None:
                            cells.append(0)
                        else:
                            if c < sw and sj < S:
                                cells.append(row[stack_col_offset + c])
                            else:
                                cells.append(0)

                if sj < S:
                    stack_col_offset += sw

            if bi < B:
                band_row_offset += bw

        return (tuple(self.sortPrefix), tuple(cells))
    # 2025-12-16 22:18 end


    # -------------------------
    # Cloning
    # -------------------------

    def clone(self) -> "SegmentedMatrix":
        """Return a deep copy of the SegmentedMatrix, including sortPrefix and info."""
        c = SegmentedMatrix(self.bandWidths[:], self.stackWidths[:])
        c.data = deepcopy(self.data)
        # copy sortPrefix as well (use deepcopy to be safe if user stores nested lists)
        c.sortPrefix = deepcopy(getattr(self, "sortPrefix", []))
        # copy info if present; default to None
        c.info = deepcopy(getattr(self, "info", None))
        return c

    # -------------------------
    # Box-style access helpers:
    # m[band][stack][row][col]  should work for get/set
    # -------------------------
    def __getitem__(self, band_index: int):
        return _BandAccessor(self, band_index)

    # -------------------------
    # Helper: convert box indices -> global indices
    # -------------------------
    def _global_row_index(self, band_index: int, row_index: int) -> int:
        if band_index < 0 or band_index >= len(self.bandWidths):
            raise IndexError("band_index out of range")
        if row_index < 0 or row_index >= self.bandWidths[band_index]:
            raise IndexError("row_index out of range for band")
        return sum(self.bandWidths[:band_index]) + row_index

    def _global_col_index(self, stack_index: int, col_index: int) -> int:
        if stack_index < 0 or stack_index >= len(self.stackWidths):
            raise IndexError("stack_index out of range")
        if col_index < 0 or col_index >= self.stackWidths[stack_index]:
            raise IndexError("col_index out of range for stack")
        return sum(self.stackWidths[:stack_index]) + col_index

    # -------------------------
    # Permutations
    # -------------------------
    def permuteBands(self, permutation: Sequence[int]):
        """
        Permute bands. permutation is a permutation of range(len(bandWidths)).
        This reorders the bandWidths and reorders the corresponding row blocks in self.data.
        """
        n = len(self.bandWidths)
        if sorted(permutation) != list(range(n)):
            raise ValueError("Invalid band permutation")
        # build new bandWidths
        new_bw = [self.bandWidths[i] for i in permutation]
        # reorder row blocks
        row_blocks = []
        start = 0
        for w in self.bandWidths:
            block = [r[:] for r in self.data[start:start + w]]
            row_blocks.append(block)
            start += w
        new_rows = []
        for i in permutation:
            new_rows.extend(deepcopy(row_blocks[i]))
        # assign
        self.bandWidths = new_bw
        self.data = new_rows

    def permuteStacks(self, permutation: Sequence[int]):
        """
        Permute stacks. permutation is a permutation of range(len(stackWidths)).
        This reorders stackWidths and reorders column blocks in every row.
        """
        m = len(self.stackWidths)
        if sorted(permutation) != list(range(m)):
            raise ValueError("Invalid stack permutation")
        old_sw = self.stackWidths[:]
        # compute column blocks per old stacks
        starts = []
        s = 0
        for w in old_sw:
            starts.append(s)
            s += w
        new_sw = [old_sw[i] for i in permutation]
        # for every row, build new row by concatenating chosen blocks
        new_data = []
        for row in self.data:
            new_row = []
            for idx in permutation:
                start = starts[idx]
                w = old_sw[idx]
                new_row.extend(row[start:start + w])
            new_data.append(new_row)
        self.stackWidths = new_sw
        self.data = new_data

    def permuteRows(self, bandIndex: int, permutation: Sequence[int]):
        """
        Permute rows within the given band.
        permutation must be a permutation of range(bandWidths[bandIndex]).
        """
        bw = self.bandWidths[bandIndex]
        if sorted(permutation) != list(range(bw)):
            raise ValueError("Invalid row permutation for band")
        start = sum(self.bandWidths[:bandIndex])
        block = [self.data[start + i] for i in range(bw)]
        new_block = [deepcopy(block[i]) for i in permutation]
        for i in range(bw):
            self.data[start + i] = new_block[i]

    def permuteCols(self, stackIndex: int, permutation: Sequence[int]):
        """
        Permute columns within the given stack.
        permutation must be a permutation of range(stackWidths[stackIndex]).
        """
        sw = self.stackWidths[stackIndex]
        if sorted(permutation) != list(range(sw)):
            raise ValueError("Invalid column permutation for stack")
        start = sum(self.stackWidths[:stackIndex])
        for r in range(len(self.data)):
            segment = self.data[r][start:start + sw]
            new_seg = [segment[i] for i in permutation]
            self.data[r][start:start + sw] = new_seg

    # -------------------------
    # Rotation 90° counterclockwise
    # -------------------------
    def rotate(self):
        """
        Rotate the full matrix 90 degrees counterclockwise.
        After rotation, bandWidths and stackWidths are swapped.
        """
        R = self.rowDim
        C = self.colDim
        # build rotated array using explicit mapping to avoid zip confusion
        rotated = [[0] * R for _ in range(C)]
        for r in range(R):
            for c in range(C):
                new_r = C - 1 - c
                new_c = r
                rotated[new_r][new_c] = self.data[r][c]
        # swap widths
        new_bandWidths = self.stackWidths[::-1]
        new_stackWidths = self.bandWidths[:]
        self.bandWidths = new_bandWidths
        self.stackWidths = new_stackWidths
        self.data = rotated

    # -------------------------
    # Deletions
    # -------------------------
    def deleteBand(self, bandIndex: int):
        "Delete an entire band and its rows."
        if bandIndex < 0 or bandIndex >= len(self.bandWidths):
            raise IndexError("bandIndex out of range")
        start = sum(self.bandWidths[:bandIndex])
        w = self.bandWidths[bandIndex]
        del self.data[start:start + w]
        del self.bandWidths[bandIndex]

    def deleteStack(self, stackIndex: int):
        "Delete an entire stack and its columns."
        if stackIndex < 0 or stackIndex >= len(self.stackWidths):
            raise IndexError("stackIndex out of range")
        start = sum(self.stackWidths[:stackIndex])
        w = self.stackWidths[stackIndex]
        for r in range(len(self.data)):
            del self.data[r][start:start + w]
        del self.stackWidths[stackIndex]

    def deleteRow(self, bandIndex: int, rowIndex: int):
        "Delete a single row within a band."
        gi = self._global_row_index(bandIndex, rowIndex)
        del self.data[gi]
        self.bandWidths[bandIndex] -= 1
        if self.bandWidths[bandIndex] == 0:
            del self.bandWidths[bandIndex]

    def deleteCol(self, stackIndex: int, colIndex: int):
        "Delete a single column within a stack."
        gi = self._global_col_index(stackIndex, colIndex)
        for r in range(len(self.data)):
            del self.data[r][gi]
        self.stackWidths[stackIndex] -= 1
        if self.stackWidths[stackIndex] == 0:
            del self.stackWidths[stackIndex]

    # -------------------------
    # reduce(): remove all-zero rows and columns
    # -------------------------
    def reduce(self):
        """
        Remove rows and columns composed entirely of zeros.
        Update bandWidths and stackWidths accordingly.
        """
        if not self.data:
            self.bandWidths = []
            self.stackWidths = []
            return

        R = self.rowDim
        C = self.colDim

        # keep_rows: indices of rows that contain a non-zero
        keep_rows = [r for r in range(R) if any(self.data[r][c] != 0 for c in range(C))]
        # If no rows remain => clear everything
        if not keep_rows:
            self.data = []
            self.bandWidths = []
            self.stackWidths = []
            return

        # rebuild data with kept rows
        new_data = [deepcopy(self.data[r]) for r in keep_rows]

        # keep_cols: indices of columns that contain a non-zero (in new_data)
        new_C = C
        keep_cols = [c for c in range(new_C) if any(new_data[r][c] != 0 for r in range(len(new_data)))]

        if not keep_cols:
            # no columns remain
            self.data = []
            self.bandWidths = []
            self.stackWidths = []
            return

        # rebuild rows with kept columns only
        new_data2 = [[row[c] for c in keep_cols] for row in new_data]

        # compute new bandWidths by counting how many kept rows fall into each original band
        new_bw = []
        row_ptr = 0
        orig_row_start = 0
        for w in self.bandWidths:
            # original band rows indices are [orig_row_start .. orig_row_start + w - 1]
            count = sum(1 for r in keep_rows if orig_row_start <= r < orig_row_start + w)
            if count > 0:
                new_bw.append(count)
            orig_row_start += w

        # compute new stackWidths by counting kept columns per original stack
        new_sw = []
        orig_col_start = 0
        for w in self.stackWidths:
            count = sum(1 for c in keep_cols if orig_col_start <= c < orig_col_start + w)
            if count > 0:
                new_sw.append(count)
            orig_col_start += w

        self.data = new_data2
        self.bandWidths = new_bw
        self.stackWidths = new_sw

    # -------------------------
    # expand(): extend to 3×3 bands with width 3 (9×9)
    # -------------------------
    def expand(self):
        """
        Expand matrix to 3 bands × 3 stacks with widths 3, i.e. 9×9.
        Missing rows/cols/bands/stacks are appended (bottom/right) and filled with zeros.
        If the matrix is already larger than 9×9, raise an exception.
        """
        if self.rowDim > 9 or self.colDim > 9:
            raise ValueError("Matrix too large to expand into 9x9 (would require shrinking).")

        # target widths
        target_bw = [3, 3, 3]
        target_sw = [3, 3, 3]

        # Validate we don't have widths > 3 (shouldn't happen per class invariant)
        if any(w > 3 for w in self.bandWidths + self.stackWidths):
            raise ValueError("Existing band/stack width > 3 cannot be expanded to canonical 3x3 blocks.")

        # Ensure band count is 3 (append bands of zeros at bottom)
        while len(self.bandWidths) < 3:
            self.bandWidths.append(3)
            # add 3 new zero-rows of current column length
            self.data.extend([[0] * self.colDim for _ in range(3)])

        # Ensure each band has width 3 (possibly add rows to each existing band)
        row_cursor = 0
        for i in range(3):
            bw = self.bandWidths[i]
            missing = 3 - bw
            if missing > 0:
                # insert `missing` zero rows after existing rows of this band
                insert_pos = row_cursor + bw
                for _ in range(missing):
                    self.data.insert(insert_pos, [0] * self.colDim)
                    insert_pos += 1
                self.bandWidths[i] = 3
            row_cursor += 3  # after expansion each band has 3 rows

        # Ensure stack count is 3 (append stacks to the right)
        while len(self.stackWidths) < 3:
            self.stackWidths.append(3)
            # append 3 zero-columns for every row
            for r in range(len(self.data)):
                self.data[r].extend([0] * 3)

        # Ensure each stack has width 3 by inserting columns where needed
        col_cursor = 0
        for i in range(3):
            sw = self.stackWidths[i]
            missing = 3 - sw
            if missing > 0:
                insert_pos = col_cursor + sw
                for r in range(len(self.data)):
                    # insert missing zeros at correct position in each row
                    self.data[r][insert_pos:insert_pos] = [0] * missing
                self.stackWidths[i] = 3
            col_cursor += 3

        # final check
        if self.rowDim != 9 or self.colDim != 9 or self.bandWidths != [3, 3, 3] or self.stackWidths != [3, 3, 3]:
            raise RuntimeError("expand() failed to produce canonical 9x9 segmentation.")

    # -------------------------
    # Group generators
    # -------------------------

    # 2025-12-20 16:02 begin
    def generate_neighbors(self, permuteSymbols=False, withTransform=False):
        """
        Liefert alle Nachbarn der Matrix bzgl. des Erzeugendensystems:
        - Vertauschen zweier Reihen innerhalb eines Bands
        - Vertauschen zweier Spalten innerhalb eines Stacks
        - Vertauschen zweier Bands
        - Vertauschen zweier Stacks
        - Rotation um 90° im Uhrzeigersinn
        - (optional) Vertauschen zweier Symbole

        Falls withTransform=True, wird jeweils (Matrix, Transform)
        zurückgegeben, sonst nur die Matrix.
        """

        H = len(self.data)
        W = len(self.data[0])

        # -------------------------
        # Reihen innerhalb eines Bands
        # -------------------------
        row0 = 0
        for bw in self.bandWidths:
            for i in range(bw):
                for j in range(i + 1, bw):
                    r1 = row0 + i
                    r2 = row0 + j

                    m = self.clone()
                    m.data[r1], m.data[r2] = m.data[r2], m.data[r1]

                    row_map = list(range(H))
                    row_map[r1], row_map[r2] = row_map[r2], row_map[r1]
                    col_map = list(range(W))
                    transform = CoordinateTransform(row_map, col_map)

                    yield (m, transform) if withTransform else m
            row0 += bw

        # -------------------------
        # Spalten innerhalb eines Stacks
        # -------------------------
        col0 = 0
        for sw in self.stackWidths:
            for i in range(sw):
                for j in range(i + 1, sw):
                    c1 = col0 + i
                    c2 = col0 + j

                    m = self.clone()
                    for r in range(H):
                        m.data[r][c1], m.data[r][c2] = m.data[r][c2], m.data[r][c1]

                    row_map = list(range(H))
                    col_map = list(range(W))
                    col_map[c1], col_map[c2] = col_map[c2], col_map[c1]
                    transform = CoordinateTransform(row_map, col_map)

                    yield (m, transform) if withTransform else m
            col0 += sw

        # -------------------------
        # Bands vertauschen
        # -------------------------
        band_starts = []
        acc = 0
        for bw in self.bandWidths:
            band_starts.append((acc, bw))
            acc += bw

        for i in range(len(band_starts)):
            for j in range(i + 1, len(band_starts)):
                (r1, h1) = band_starts[i]
                (r2, h2) = band_starts[j]
                if h1 != h2:
                    continue

                m = self.clone()
                for k in range(h1):
                    m.data[r1 + k], m.data[r2 + k] = m.data[r2 + k], m.data[r1 + k]

                row_map = list(range(H))
                for k in range(h1):
                    row_map[r1 + k], row_map[r2 + k] = row_map[r2 + k], row_map[r1 + k]
                col_map = list(range(W))
                transform = CoordinateTransform(row_map, col_map)

                yield (m, transform) if withTransform else m

        # -------------------------
        # Stacks vertauschen
        # -------------------------
        stack_starts = []
        acc = 0
        for sw in self.stackWidths:
            stack_starts.append((acc, sw))
            acc += sw

        for i in range(len(stack_starts)):
            for j in range(i + 1, len(stack_starts)):
                (c1, w1) = stack_starts[i]
                (c2, w2) = stack_starts[j]
                if w1 != w2:
                    continue

                m = self.clone()
                for r in range(H):
                    for k in range(w1):
                        m.data[r][c1 + k], m.data[r][c2 + k] = (
                            m.data[r][c2 + k],
                            m.data[r][c1 + k],
                        )

                row_map = list(range(H))
                col_map = list(range(W))
                for k in range(w1):
                    col_map[c1 + k], col_map[c2 + k] = col_map[c2 + k], col_map[c1 + k]
                transform = CoordinateTransform(row_map, col_map)

                yield (m, transform) if withTransform else m

        # -------------------------
        # Rotation 90° im Uhrzeigersinn
        # -------------------------
        if H == W:
            m = self.clone()
            n = H
            new_data = [[0] * n for _ in range(n)]
            for r in range(n):
                for c in range(n):
                    new_data[c][n - 1 - r] = m.data[r][c]
            m.data = new_data

            transform = RotationTransform(n)
            yield (m, transform) if withTransform else m

        # -------------------------
        # Symbolvertauschung (optional)
        # -------------------------
        if permuteSymbols:
            symbols = sorted({v for row in self.data for v in row if v != 0})
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    a, b = symbols[i], symbols[j]
                    m = self.clone()
                    for r in range(H):
                        for c in range(W):
                            if m.data[r][c] == a:
                                m.data[r][c] = b
                            elif m.data[r][c] == b:
                                m.data[r][c] = a

                    # Keine Koordinatenänderung
                    yield (m, None) if withTransform else m

    # 2025-12-20 16:02 end






    def isConsistent(self) -> bool:
        """
        Check whether the SegmentedMatrix is consistent:
        - no non-zero number appears twice in any row
        - no non-zero number appears twice in any column
        - no non-zero number appears twice in any block
        """

        data = self.data
        nrows = len(data)
        ncols = len(data[0]) if nrows > 0 else 0

        # --- rows ---
        for r in range(nrows):
            seen = set()
            for v in data[r]:
                if v != 0:
                    if v in seen:
                        return False
                    seen.add(v)

        # --- columns ---
        for c in range(ncols):
            seen = set()
            for r in range(nrows):
                v = data[r][c]
                if v != 0:
                    if v in seen:
                        return False
                    seen.add(v)

        # --- blocks ---
        row_start = 0
        for bw in self.bandWidths:
            col_start = 0
            for sw in self.stackWidths:
                seen = set()
                for r in range(row_start, row_start + bw):
                    for c in range(col_start, col_start + sw):
                        v = data[r][c]
                        if v != 0:
                            if v in seen:
                                return False
                            seen.add(v)
                col_start += sw
            row_start += bw

        return True


    # -------------------------
    # Maximal elements under group action
    # -------------------------

    # 2025-12-14 19:06 begin
    # 2025-12-14 20:05 begin
    @classmethod
    def filterMaximalSegmentedMatrices(cls, matrices, permuteSymbols=False):
        """
        Entfernt alle SegmentedMatrices, aus denen sich mit den zulässigen
        Gruppenoperationen noch Matrizen mit größerer sortId erzeugen lassen.
        """

        # unknown: sortId -> SegmentedMatrix
        unknown = {m.sortId: m.clone() for m in matrices}
        allMaxima = {}

        while unknown:
            open_set = {}
            visited = set()
            currentMaxima = {}

            # Start mit beliebigem Element
            sid, start = unknown.popitem()
            open_set[sid] = start

            maxSortId = None

            while open_set:
                sid, sm = open_set.popitem()

                if sid in visited:
                    continue
                visited.add(sid)

                # Maximum aktualisieren
                if maxSortId is None or sid > maxSortId:
                    maxSortId = sid
                    currentMaxima = {sid: sm}
                elif sid == maxSortId:
                    currentMaxima[sid] = sm

                # Nachbarn über Erzeugendensystem
                for nb in sm.generate_neighbors(permuteSymbols=permuteSymbols):
                    nb_sid = nb.sortId

                    # >>> EINZIGE KORREKTUR BEGINN <<<
                    if nb_sid not in visited and nb_sid not in open_set:
                        if nb_sid in unknown:
                            open_set[nb_sid] = unknown.pop(nb_sid)
                        else:
                            open_set[nb_sid] = nb
                    # >>> EINZIGE KORREKTUR ENDE <<<

            # Aktuelle Maxima sichern
            allMaxima.update(currentMaxima)

        return list(allMaxima.values())
    # 2025-12-14 20:05 end


    # -------------------------
    # Filters
    # -------------------------

    """
    def hasIsolatedSingletons(self):
        # search singleton:
        myBandEnd=0
        for myBandWidth in self.bandWidths:
            myBandStart=myBandEnd
            myBandEnd=myBandStart+myBandWidth
            myStackeEnd=0
            for myStackWidth in self.stackWidth:
                myStackStart=myStackeEnd
                myStackeEnd=myStackStart+myStackWidth
                myBlockElements=((i,j) for i in range(myBandStart ,myBandEnd)  for self.data[i][j] for j in range(myStackStart, myStackEnd) if self.data[i][j] !=0 )
                if len(myBlockElements)==1:
     """               
                    
    # 2025-12-16 22:37 begin
    def extractIsolatedSingletons(self):
        """
        Returns a list of (row, col) index pairs of all isolated singletons.

        A cell (r,c) with value != 0 is an isolated singleton if it is the only
        nonzero element in its row, its column, and its block.
        """

        isolated = []

        nrows = len(self.data)
        ncols = len(self.data[0]) if nrows > 0 else 0

        # --- Precompute band boundaries ---
        band_starts = []
        acc = 0
        for bw in self.bandWidths:
            band_starts.append(acc)
            acc += bw

        stack_starts = []
        acc = 0
        for sw in self.stackWidths:
            stack_starts.append(acc)
            acc += sw

        def find_band(r):
            for i, start in enumerate(band_starts):
                if start <= r < start + self.bandWidths[i]:
                    return i
            return None

        def find_stack(c):
            for i, start in enumerate(stack_starts):
                if start <= c < start + self.stackWidths[i]:
                    return i
            return None

        for r in range(nrows):
            for c in range(ncols):
                val = self.data[r][c]
                if val == 0:
                    continue

                # --- check row ---
                row_count = sum(
                    1 for cc in range(ncols)
                    if self.data[r][cc] != 0
                )
                if row_count != 1:
                    continue

                # --- check column ---
                col_count = sum(
                    1 for rr in range(nrows)
                    if self.data[rr][c] != 0
                )
                if col_count != 1:
                    continue

                # --- check block ---
                b = find_band(r)
                s = find_stack(c)

                block_row_start = band_starts[b]
                block_row_end = block_row_start + self.bandWidths[b]

                block_col_start = stack_starts[s]
                block_col_end = block_col_start + self.stackWidths[s]

                block_count = 0
                for rr in range(block_row_start, block_row_end):
                    for cc in range(block_col_start, block_col_end):
                        if self.data[rr][cc] != 0:
                            block_count += 1

                if block_count != 1:
                    continue

                isolated.append((r, c))

        return isolated
    # 2025-12-16 22:37 end
             
            
   

    # -------------------------
    # all representatives of all types
    # -------------------------

    # 2025-12-14 19:27 begin
    @classmethod
    def all01BlockRepresentatives(cls, clueCount: int):
        """
        Generate canonical representatives of 3×3 0/1 SegmentedMatrices
        for all k = 1..clueCount, based on toptimize().
        Each representative gets sortPrefix = [k].
        """

        result = {}  # dict: sortId → SegmentedMatrix
        positions = list(range(9))  # 3x3 grid positions

        for k in range(1, clueCount + 1):

            for ones in combinations(positions, k):

                # Create empty 3x3 SegmentedMatrix
                m = cls([1,1,1], [1,1,1])
                m.data = [[0]*3 for _ in range(3)]

                # Insert the 1s
                for p in ones:
                    r = p // 3
                    c = p % 3
                    m.data[r][c] = 1

                # MUST set prefix BEFORE getting sortId
                m.sortPrefix = [k]

                # Now compute canonical ID including prefix
                sid = m.sortId

                # If this representative has not yet been seen:
                if sid not in result:
                    result[sid] = m.clone()

        # result: dict mapping sortId -> SegmentedMatrix
        # We build a new dict `filtered_result` with only the survivors.

        rotated_result = {}

        for sid, sm in result.items():

            # Compute rotated + optimized representative
            rotated = sm.clone()
            rotated.rotate()

            # Otherwise keep the original entry
            # (use the stored matrix; ensure we store a clone to avoid aliasing)
            rotated_result[rotated.sortId] = rotated

        # Replace result dict with filtered_result
        result = {**result, **rotated_result}

        filtered=cls.filterMaximalSegmentedMatrices(list(result.values()))

        filtered.sort(key=lambda sm: sm.sortId, reverse=True)

        # Jede Matrix zuerst reduzieren
        for sm in filtered:
            sm.reduce()

        # Neue sortPrefix-Nummerierung
        for i, sm in enumerate(filtered, start=1):
            sm.sortPrefix = [i]

        return filtered
    # 2025-12-14 19:27 end


    # 2025-12-14 19:27 begin
    @classmethod
    def allWeightedBlockRepresentatives(cls, clueCount: int):
        """
        Erweiterte Version:
        - erzeugt gewichtete Matrizen basierend auf all01BlockRepresentatives
        - reduziert alle Matrizen
        - filtert:
            A: toptimize(M) > M  -> entferne M
            B: toptimize(rotate(M)) > M -> entferne M
        - sortiert das Ergebnis absteigend nach sortId
        """

        bases = cls.all01BlockRepresentatives(clueCount)
        results = []

        for base in bases:
            # Anzahl der Einsen in der Basis-Matrix
            one_positions = [(r, c) for r, row in enumerate(base.data)
                                      for c, v in enumerate(row) if v == 1]
            k = len(one_positions)
            if k == 0:
                continue

            # Partitionen von clueCount in k Summanden
            parts = SegmentedMatrix.allPartitions(clueCount, k)

            # SortId der Basismatrix → wird sortPrefix
            base_sortKey = base.sortId

            for P in parts:
                sm = base.clone()
                sm.info = {"partition": P}

                # Alles auf 0 setzen
                for r in range(len(sm.data)):
                    for c in range(len(sm.data[0])):
                        sm.data[r][c] = 0

                # Einsen durch Gewichte ersetzen
                for (r, c), weight in zip(one_positions, P):
                    sm.data[r][c] = weight

                # Reduzieren (wichtig!)
                sm.reduce()

                results.append(sm)

        # ------------------------------------------
        # Filterphase
        # ------------------------------------------

        unique={}
        for sm in results:
            sid = sm.sortId
            # If this representative has not yet been seen:
            if sid not in unique:
                unique[sid] = sm
        rotated={}
        for sm in unique.values():
            rot=sm.clone()
            rot.rotate()
            sid=rot.sortId
            if sid not in rotated:
                rotated[sid]=rot
        unique={**unique, ** rotated}
                

        filtered=cls.filterMaximalSegmentedMatrices(unique.values())

        filtered.sort(key=lambda sm: sm.sortId)

        # Neue sortPrefix-Nummerierung für gewichtete Matrizen
        for i, sm in enumerate(filtered, start=1):
            sm.sortPrefix = [i]

        return filtered
    # 2025-12-14 19:27 end


    @classmethod
    def allWeightedCellRepresentatives(cls, aClueCount: int):
        wblist = cls.allWeightedBlockRepresentatives(aClueCount)
        result = []

        for wb in wblist:
            # Anzahl der Bänder und Stacks des wb
            B = len(wb.bandWidths)
            S = len(wb.stackWidths)

            # --- 1) neue bandWidths bestimmen ---
            new_bw = []
            row_start = 0
            for b in range(B):
                bw = wb.bandWidths[b]
                rows = range(row_start, row_start + bw)

                # Summe aller Werte in diesem Band
                total = 0
                for r in rows:
                    total += sum(wb.data[r])

                new_bw.append(min(total, 3))
                row_start += bw

            # --- 2) neue stackWidths bestimmen ---
            new_sw = []
            col_start = 0
            for s in range(S):
                sw = wb.stackWidths[s]
                cols = range(col_start, col_start + sw)

                # Summe aller Werte in diesem Stack
                total = 0
                for r in range(len(wb.data)):
                    for c in cols:
                        total += wb.data[r][c]

                new_sw.append(min(total, 3))
                col_start += sw

            # Falls alles 0 => überspringen
            if sum(new_bw) == 0 or sum(new_sw) == 0:
                continue

            # --- 3) neue Matrix anlegen ---
            sm = SegmentedMatrix(new_bw, new_sw)

            # --- 4) Blockwerte setzen ---
            row_start = 0
            for b in range(B):
                bw = wb.bandWidths[b]
                band_rows = range(row_start, row_start + bw)

                col_start = 0
                for s in range(S):
                    sw = wb.stackWidths[s]
                    stack_cols = range(col_start, col_start + sw)

                    # Summiere Blockwert = Summe des Blocks, gekappt auf 1 Zelle
                    val = 0
                    for r in band_rows:
                        for c in stack_cols:
                            val += wb.data[r][c]

                    # In Zielmatrix Block setzen (nur oben links)
                    if new_bw[b] > 0 and new_sw[s] > 0:
                        sm[b][s][0][0] = val

                    col_start += sw
                row_start += bw

            # SortPrefix & info kopieren
            sm.sortPrefix = wb.sortPrefix[:]
            sm.info = wb.info

            result.append(sm)

            result.sort(key=lambda sm: sm.sortId)

        return result
    


    # 2025-12-14 19:06 begin
    @classmethod
    def all01CellRepresentatives(cls, aClueCount: int):
        weighted = cls.allWeightedCellRepresentatives(aClueCount)

        all_binary = []

        for wc in weighted:
            bm_list = wc.expand_weighted_matrix_to_binary()
            all_binary.extend(cls.filterMaximalSegmentedMatrices(bm_list))

        no_isolated_singletons=[]
        for bm in all_binary:
            if len(bm.extractIsolatedSingletons())==0:
                no_isolated_singletons.append(bm)
        # Maxima bestimmen
        maxima = cls.filterMaximalSegmentedMatrices(no_isolated_singletons)

        for bm in maxima:
            bm.reduce()

        for sm in maxima:
            sm.sortPrefix = [-sm.sortPrefix[0]]


        # sortieren (sortId ist Property!)
        maxima.sort(key=lambda m: m.sortId, reverse=True)
        #maxima.sort(key=lambda m: m.sortId)

        # neu nummerieren
        for i, sm in enumerate(maxima, start=1):
            sm.sortPrefix = [i]

        return maxima
    # 2025-12-14 19:06 end


    # 2025-12-14 20:36 begin
    @classmethod
    def allSymbolRepresentatives(cls, nClues: int):
        """
        Erzeugt symbolische SegmentedMatrices aus 0-1-Zellrepräsentanten
        unter Verwendung aller Restricted-Growth-Sequences der Länge nClues.
        """

        # 1) Alle Restricted Growth Sequences der Länge nClues
        rgs_list = SegmentedMatrix.restrictedGrowthSequence(nClues)
        rgs_list=[[nClues+1-s for s in rgs] for rgs in rgs_list]

        # 2) Alle 0-1 Zellrepräsentanten
        base_matrices = cls.all01CellRepresentatives(nClues)
        for sm in base_matrices:
            sm.sortPrefix=[-sm.sortPrefix[0]]

        collected = []

        # 3) Für jede Basis-Matrix
        for base in base_matrices:

            # Positionen aller Einsen in Block-→Zeilen-→Spalten-Reihenfolge
            one_positions = []

            row_start = 0
            for b, bw in enumerate(base.bandWidths):
                col_start = 0
                for s, sw in enumerate(base.stackWidths):

                    for i in range(bw):
                        for j in range(sw):
                            if base[b][s][i][j] == 1:
                                r = row_start + i
                                c = col_start + j
                                one_positions.append((r, c))

                    col_start += sw
                row_start += bw

            # Anzahl der Einsen muss exakt passen
            if len(one_positions) != nClues:
                continue

            # 4) Für jede Restricted Growth Sequence
            for rgs in rgs_list:
                sm = base.clone()

                for (r, c), value in zip(one_positions, rgs):
                    sm.data[r][c] = value

                if not sm.isConsistent():
                    continue

                collected.append(sm)

        # 5) Maxima bestimmen
        maxima = cls.filterMaximalSegmentedMatrices(collected, permuteSymbols=True)
        maxima.sort(key=lambda sm: sm.sortId,reverse=True)
        for nr,sm in enumerate(maxima, start=1):
            sm.sortPrefix=[nr]

        for m in maxima:
            m.data=[[0 if s==0 else nClues+1-s for s in row] for row in m.data]
        return maxima

    # 2025-12-14 20:36 end

    # -------------------------
    # read(): fill rows (list of lists)
    # -------------------------
    def read(self, rows: Sequence[Sequence[int]]):
        """
        Fill the matrix by rows. Input is a list (or tuple) of row-lists.
        If fewer row-lists are provided than required, the remaining rows are zero-filled.
        If a row-list is shorter than the number of columns, it is padded with zeros.
        If it is longer, it is truncated.
        The provided rows overwrite existing data in row-major order.
        """
        # Normalize input (accept tuple or list)
        provided_rows = list(rows)

        total_rows = sum(self.bandWidths)
        total_cols = sum(self.stackWidths)

        # If underlying data is smaller/larger than total_rows/cols, ensure shape
        # If data currently empty, initialize appropriate size
        if len(self.data) != total_rows:
            # either expand or truncate data to total_rows
            if len(self.data) < total_rows:
                # append zero rows
                for _ in range(total_rows - len(self.data)):
                    self.data.append([0] * total_cols)
            else:
                # truncate
                self.data = self.data[:total_rows]

        if self.colDim != total_cols:
            # adjust columns in every row
            for r in range(len(self.data)):
                if len(self.data[r]) < total_cols:
                    self.data[r].extend([0] * (total_cols - len(self.data[r])))
                elif len(self.data[r]) > total_cols:
                    self.data[r] = self.data[r][:total_cols]

        # Now overwrite row by row
        for r in range(total_rows):
            if r < len(provided_rows) and isinstance(provided_rows[r], (list, tuple)):
                rowvals = list(provided_rows[r][:total_cols])
                if len(rowvals) < total_cols:
                    rowvals.extend([0] * (total_cols - len(rowvals)))
                self.data[r] = rowvals
            else:
                # missing: zero row
                self.data[r] = [0] * total_cols

    # -------------------------
    # ASCII print
    # -------------------------
    def print(self):
        """
        Print segmentation in ASCII. Use '.' for zero cells.
        Horizontal dash count per stack: dash_count = 2 * w + 1
        Example for three stacks with w=3: +-------+-------+-------+
        """
        if not self.data:
            print("+ +")
            return

        # build top line segments
        segments = []
        for w in self.stackWidths:
            dash_count = 2 * w + 1
            segments.append("+" + "-" * dash_count)
        top_line = "".join(segments) + "+"

        row_ptr = 0
        for b, bw in enumerate(self.bandWidths):
            print(top_line)
            for r in range(bw):
                row = self.data[row_ptr]
                out = "|"
                col_ptr = 0
                for s, sw in enumerate(self.stackWidths):
                    # print sw cells with leading space
                    for c in range(sw):
                        val = row[col_ptr + c]
                        out += " " + (str(val) if val != 0 else ".")
                    out += " |"
                    col_ptr += sw
                print(out)
                row_ptr += 1
        print(top_line)

    def expand_weighted_matrix_to_binary(self):
        """
        Expand a WeightedBlockMatrix into all binary SegmentedMatrices
        such that each block contains exactly as many 1s as the value
        in the top-left cell of the block.
        """

        B = len(self.bandWidths)
        S = len(self.stackWidths)

        # Precompute row ranges for bands
        band_rows = []
        r = 0
        for bw in self.bandWidths:
            band_rows.append(list(range(r, r + bw)))
            r += bw

        # Precompute column ranges for stacks
        stack_cols = []
        c = 0
        for sw in self.stackWidths:
            stack_cols.append(list(range(c, c + sw)))
            c += sw

        # For each block, compute all possible placements
        block_choices = []

        for b in range(B):
            for s in range(S):
                rows = band_rows[b]
                cols = stack_cols[s]

                block_cells = [(r, c) for r in rows for c in cols]

                k = self.data[rows[0]][cols[0]]  # block value from top-left

                if k < 0 or k > len(block_cells):
                    return []  # impossible block → no matrices

                choices = list(combinations(block_cells, k))
                block_choices.append(choices)

        # Cartesian product of all block choices
        results = []

        def backtrack(i, current):
            if i == len(block_choices):
                sm = SegmentedMatrix(self.bandWidths[:], self.stackWidths[:])
                sm.data = [[0] * sm.width for _ in range(sm.height)]

                for (r, c) in current:
                    sm.data[r][c] = 1

                sm.sortPrefix = self.sortPrefix[:]
                sm.info = deepcopy(self.info)
                results.append(sm)
                return

            for choice in block_choices[i]:
                backtrack(i + 1, current + list(choice))

        backtrack(0, [])
        return results

    @staticmethod
    def allPartitions(aSum: int, aCount: int):
        """
        Return all ordered partitions (compositions) of aSum into aCount
        positive integers.
        
        Example:
            allPartitions(4, 2) → [(1,3), (2,2), (3,1)]
        """
        results = []

        def backtrack(remaining_sum, remaining_count, prefix):
            # If no summands left to place:
            if remaining_count == 0:
                if remaining_sum == 0:
                    results.append(tuple(prefix))
                return

            # Each summand must be at least 1
            # Max value allowed is remaining_sum - (remaining_count-1)
            # so that the remaining summands can each be at least 1
            min_val = 1
            max_val = remaining_sum - (remaining_count - 1)

            if max_val < min_val:
                return

            for v in range(min_val, max_val + 1):
                backtrack(remaining_sum - v, remaining_count - 1, prefix + [v])

        backtrack(aSum, aCount, [])
        return results


    @staticmethod
    def restrictedGrowthSequence(n: int):
        """
        Generate all restricted growth sequences (RGS) of length n.

        A restricted growth sequence is a sequence a[0..n-1] such that:
          - a[0] == 1
          - a[i] <= 1 + max(a[0..i-1]) for all i >= 1
        """
        if n <= 0:
            return []

        result = []

        def backtrack(seq, current_max):
            if len(seq) == n:
                result.append(seq[:])
                return

            # Next value can range from 1 to current_max + 1
            for v in range(1, current_max + 2):
                seq.append(v)
                backtrack(seq, max(current_max, v))
                seq.pop()

        backtrack([1], 1)
        return result



    # 2025-12-20 11:08 begin
    def matchesClueMatrixOrRaise(self, clueMatrix):
        """
        Prüft, ob diese Sudoku-Matrix durch zulässige Transformationen
        (Band/Stack/Row/Col, Rotation, Symbolumbenennung)
        auf die Clue-Matrix abgebildet werden kann.

        Rückgabe:
            Liste von (orig_row, orig_col, pattern_symbol)

        Wirft:
            ValueError, wenn keine Abbildung existiert.
        """

        # --- Validierung ---
        if self.bandWidths != [3, 3, 3] or self.stackWidths != [3, 3, 3]:
            raise ValueError("Matrix ist kein Sudoku (falsche Band-/Stack-Struktur).")
        if self.rowDim != 9 or self.colDim != 9:
            raise ValueError("Matrix ist kein 9x9 Sudoku.")

        # --- Clues sammeln ---
        clues = []
        for r in range(9):
            for c in range(9):
                v = clueMatrix.data[r][c]
                if v != 0:
                    clues.append((r, c, v))

        if not clues:
            return []

        from collections import deque

        start = self.clone()
        coord_map = [[(r, c) for c in range(9)] for r in range(9)]

        queue = deque([(start, coord_map)])
        visited = {start.sortId}

        while queue:
            sm, cmap = queue.popleft()

            # --- BIJEKTIVE Symbolprüfung ---
            p2s = {}
            s2p = {}
            ok = True

            for r, c, pval in clues:
                sval = sm.data[r][c]
                if sval == 0:
                    ok = False
                    break

                if pval in p2s and p2s[pval] != sval:
                    ok = False
                    break
                if sval in s2p and s2p[sval] != pval:
                    ok = False
                    break

                p2s[pval] = sval
                s2p[sval] = pval

            if ok:
                result = []
                for r, c, pval in clues:
                    orig_r, orig_c = cmap[r][c]
                    result.append((orig_r, orig_c, pval))
                return result

            # --- Nachbarn ---
            # 2025-12-20 11:32 begin
            for nb, transform in sm.generate_neighbors(withTransform=True):
                sid = nb.sortId
                if sid in visited:
                    continue
                visited.add(sid)

                # coord_map korrekt transformieren
                new_cmap = [[None]*9 for _ in range(9)]

                for r in range(9):
                    for c in range(9):
                        rr, cc = transform.apply(r, c)
                        new_cmap[rr][cc] = cmap[r][c]

                queue.append((nb, new_cmap))
            # 2025-12-20 11:32 end

        raise ValueError("Matrix kann nicht auf die Clue-Matrix abgebildet werden.")
    # 2025-12-20 11:08 end



# -------------------------
# Accessor helper classes to allow m[b][s][r][c] and m[b][s][r][c] = v
# -------------------------
class _BandAccessor:
    def __init__(self, matrix: SegmentedMatrix, b: int):
        self._m = matrix
        self._b = b

    def __getitem__(self, s: int):
        return _StackAccessor(self._m, self._b, s)


class _StackAccessor:
    def __init__(self, matrix: SegmentedMatrix, b: int, s: int):
        self._m = matrix
        self._b = b
        self._s = s

    def __getitem__(self, r: int):
        return _RowAccessor(self._m, self._b, self._s, r)


class _RowAccessor:
    def __init__(self, matrix: SegmentedMatrix, b: int, s: int, r: int):
        self._m = matrix
        self._b = b
        self._s = s
        self._r = r

    def __getitem__(self, c: int):
        global_r = self._m._global_row_index(self._b, self._r)
        global_c = self._m._global_col_index(self._s, c)
        return self._m.data[global_r][global_c]

    def __setitem__(self, c: int, value: int):
        global_r = self._m._global_row_index(self._b, self._r)
        global_c = self._m._global_col_index(self._s, c)
        self._m.data[global_r][global_c] = value



# 2025-12-20 12:23 begin

class SwapRows:
    def __init__(self, r1, r2):
        self.r1 = r1
        self.r2 = r2

    def apply(self, r, c):
        if r == self.r1:
            return self.r2, c
        if r == self.r2:
            return self.r1, c
        return r, c


class SwapCols:
    def __init__(self, c1, c2):
        self.c1 = c1
        self.c2 = c2

    def apply(self, r, c):
        if c == self.c1:
            return r, self.c2
        if c == self.c2:
            return r, self.c1
        return r, c


class Rotate90:
    def apply(self, r, c):
        # für 9×9 Sudoku
        return c, 8 - r

# 2025-12-20 12:23 end


# 2025-12-20 12:36 begin

class SwapBands:
    def __init__(self, band1, band2, bandWidths):
        self.band1 = band1
        self.band2 = band2
        self.bandWidths = bandWidths

        self.b1_start = sum(bandWidths[:band1])
        self.b2_start = sum(bandWidths[:band2])
        self.bw = bandWidths[band1]

    def apply(self, r, c):
        if self.b1_start <= r < self.b1_start + self.bw:
            return r + (self.b2_start - self.b1_start), c
        if self.b2_start <= r < self.b2_start + self.bw:
            return r - (self.b2_start - self.b1_start), c
        return r, c


class SwapStacks:
    def __init__(self, stack1, stack2, stackWidths):
        self.stack1 = stack1
        self.stack2 = stack2
        self.stackWidths = stackWidths

        self.s1_start = sum(stackWidths[:stack1])
        self.s2_start = sum(stackWidths[:stack2])
        self.sw = stackWidths[stack1]

    def apply(self, r, c):
        if self.s1_start <= c < self.s1_start + self.sw:
            return r, c + (self.s2_start - self.s1_start)
        if self.s2_start <= c < self.s2_start + self.sw:
            return r, c - (self.s2_start - self.s1_start)
        return r, c

# 2025-12-20 12:36 end

# 2025-12-20 16:02 begin

class CoordinateTransform:
    """
    Allgemeine Koordinatentransformation durch explizite
    Row- und Column-Abbildungen.
    """
    def __init__(self, row_map, col_map):
        self.row_map = row_map
        self.col_map = col_map

    def apply(self, r, c):
        return self.row_map[r], self.col_map[c]


class RotationTransform:
    """
    90°-Rotation im Uhrzeigersinn für quadratische Matrizen.
    """
    def __init__(self, n):
        self.n = n

    def apply(self, r, c):
        return c, self.n - 1 - r



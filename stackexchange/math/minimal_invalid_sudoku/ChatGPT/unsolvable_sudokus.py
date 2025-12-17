from copy import deepcopy
from itertools import permutations
from typing import List, Sequence, Tuple, Any
from itertools import combinations



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
    def generate_neighbors(self, permuteSymbols=False):
        """
        Erzeuge alle Nachbarn dieser Matrix mittels eines Erzeugendensystems
        der zulässigen Operationen.
        """
        neighbors = []

        # 1) Rotation
        r = self.clone()
        r.rotate()
        neighbors.append(r)
        SegmentedMatrix.cntRotate+=1

        # 2) Band 0 <-> k
        for k in range(1, len(self.bandWidths)):
            p = list(range(len(self.bandWidths)))
            p[0], p[k] = p[k], p[0]
            x = self.clone()
            x.permuteBands(p)
            neighbors.append(x)
            SegmentedMatrix.cntBandStackPermute+=1

        # 3) Stack 0 <-> k
        for k in range(1, len(self.stackWidths)):
            p = list(range(len(self.stackWidths)))
            p[0], p[k] = p[k], p[0]
            x = self.clone()
            x.permuteStacks(p)
            neighbors.append(x)
            SegmentedMatrix.cntBandStackPermute+=1

        # 4) Zeile 0 <-> k in Band j
        for j, bw in enumerate(self.bandWidths):
            for k in range(1, bw):
                p = list(range(bw))
                p[0], p[k] = p[k], p[0]
                x = self.clone()
                x.permuteRows(j, p)
                neighbors.append(x)
                SegmentedMatrix.cntRowColPermute+=1

        # 5) Spalte 0 <-> k in Stack j
        for j, sw in enumerate(self.stackWidths):
            for k in range(1, sw):
                p = list(range(sw))
                p[0], p[k] = p[k], p[0]
                x = self.clone()
                x.permuteCols(j, p)
                neighbors.append(x)
                SegmentedMatrix.cntRowColPermute+=1

         # 6) symbole permutation
        if permuteSymbols:
            myList=sorted(list(set([a for row in self.data for a in row if a>0])), reverse=True)
            u = myList.pop()
            for v in myList:
                x = self.clone()
                x.data=[[u if a==v else v if a==u else a for a in row ] for row in self.data]
                neighbors.append(x)
                SegmentedMatrix.cntSymbolPermute+=1
                
            


             

        return neighbors

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
        from itertools import product

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
            parts = allPartitions(clueCount, k)

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
            bm_list = expand_weighted_matrix_to_binary(wc)
            all_binary.extend(cls.filterMaximalSegmentedMatrices(bm_list))

        # Maxima bestimmen
        maxima = cls.filterMaximalSegmentedMatrices(all_binary)

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
        rgs_list = restrictedGrowthSequence(nClues)
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

    # -------------------------
    # toptimize(): canonical minimal sortId under allowed permutations
    # -------------------------
    def toptimize(self) -> "SegmentedMatrix":
        """
        Return a deep-copy of a SegmentedMatrix that has the lexicographically smallest
        sortId reachable via:
          - any permutation of bands (band order)
          - any permutation of stacks (stack order)
          - any permutation of rows inside each band
          - any permutation of columns inside each stack

        The algorithm does an exhaustive search over the small permutation spaces
        but uses independent optimization for rows/cols inside bands/stacks to prune.
        This is exact (not heuristic) for the <=3 sizes.
        """
        # quick clones for manipulation
        base = self.clone()

        # 1) optimize rows within each band independently:
        for b in range(len(base.bandWidths)):
            bw = base.bandWidths[b]
            best = None
            best_id = None
            for perm in permutations(range(bw)):
                tmp = base.clone()
                tmp.permuteRows(b, perm)
                sid = tmp.sortId
                if best_id is None or sid > best_id:
                    best_id = sid
                    best = tmp
            base = best

        # 2) optimize columns within each stack independently:
        for s in range(len(base.stackWidths)):
            sw = base.stackWidths[s]
            best = None
            best_id = None
            for perm in permutations(range(sw)):
                tmp = base.clone()
                tmp.permuteCols(s, perm)
                sid = tmp.sortId
                if best_id is None or sid > best_id:
                    best_id = sid
                    best = tmp
            base = best

        # 3) try band permutations (full)
        best = None
        best_id = None
        for perm_b in permutations(range(len(base.bandWidths))):
            tmp_b = base.clone()
            tmp_b.permuteBands(perm_b)
            # for each such, we will consider stack permutations next
            # 4) try stack permutations (full)
            for perm_s in permutations(range(len(base.stackWidths))):
                tmp = tmp_b.clone()
                tmp.permuteStacks(perm_s)
                sid = tmp.sortId
                if best_id is None or sid > best_id:
                    best_id = sid
                    best = tmp
        # best contains the minimal one
        return best.clone() if best is not None else self.clone()


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


# -------------------------
# BinaryBoxMatrix (derived)
# -------------------------
class BinaryBoxMatrix(SegmentedMatrix):
    """
    BinaryBoxMatrix is a 3×3 block matrix where each box is 1×1 by default.
    Defaults: bandWidths=[1,1,1], stackWidths=[1,1,1]
    """

    def __init__(self):
        super().__init__([1, 1, 1], [1, 1, 1])

    @property
    def sortId(self) -> Tuple:
        """
        sortId: [count_nonzero, flattened_list_up_to_last_nonzero]
        Represented as (count_nonzero, flattened_tuple...) for lexicographic comparisons.
        """
        flat = [v for row in self.data for v in row]
        nonzero_count = sum(1 for v in flat if v != 0)
        last_idx = -1
        for i, v in enumerate(flat):
            if v != 0:
                last_idx = i
        if last_idx == -1:
            trimmed = ()
        else:
            trimmed = tuple(flat[: last_idx + 1])
        return (nonzero_count,) + trimmed


def test_clone():
    """
    Regression test for SegmentedMatrix.clone()
    Tests multiple object states including:
    - default empty matrix
    - non-empty data
    - sortPrefix set
    - info set
    Ensures clone is a deep copy and changes to the clone do not affect the original.
    """
    print("Running clone() regression tests...")

    from copy import deepcopy

    # 1) Default empty matrix
    sm1 = SegmentedMatrix([1,1,1], [1,1,1])
    c1 = sm1.clone()
    assert c1.data == sm1.data
    assert c1.sortPrefix == sm1.sortPrefix
    assert c1.info == sm1.info
    assert c1 is not sm1
    print("Test 1 passed: empty matrix clone")

    # 2) Non-empty data
    sm2 = SegmentedMatrix([1,1,1], [1,1,1])
    sm2.data[0][0] = 1
    c2 = sm2.clone()
    assert c2.data[0][0] == 1
    c2.data[0][0] = 9
    assert sm2.data[0][0] == 1  # ensure deep copy
    print("Test 2 passed: data deep copy")

    # 3) sortPrefix set
    sm3 = SegmentedMatrix([1,1,1], [1,1,1])
    sm3.sortPrefix = [3]
    c3 = sm3.clone()
    assert c3.sortPrefix == [3]
    c3.sortPrefix[0] = 9
    assert sm3.sortPrefix == [3]  # ensure deep copy
    print("Test 3 passed: sortPrefix deep copy")

    # 4) info set
    sm4 = SegmentedMatrix([1,1,1], [1,1,1])
    sm4.info = {"note": "test", "list": [1,2,3]}
    c4 = sm4.clone()
    assert c4.info == sm4.info
    c4.info["list"][0] = 99
    assert sm4.info["list"][0] == 1  # ensure deep copy
    print("Test 4 passed: info deep copy")

    # 5) Combination
    sm5 = SegmentedMatrix([1,1,1], [1,1,1])
    sm5.data[0][1] = 2
    sm5.sortPrefix = [5]
    sm5.info = {"note": "combo", "values": [4,5]}
    c5 = sm5.clone()
    assert c5.data[0][1] == 2
    assert c5.sortPrefix == [5]
    assert c5.info == sm5.info
    c5.data[0][1] = 0
    c5.sortPrefix[0] = 0
    c5.info["values"][0] = 0
    # original unchanged
    assert sm5.data[0][1] == 2
    assert sm5.sortPrefix[0] == 5
    assert sm5.info["values"][0] == 4
    print("Test 5 passed: combination deep copy")

    print("All clone() regression tests passed successfully!")





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


from typing import List





##############################################################


import itertools

def binary_blocks(h, w, k):
    """
    Erzeugt alle h×w-0-1-Matrizen mit genau k Einsen
    """
    n = h * w
    for ones in itertools.combinations(range(n), k):
        block = [[0]*w for _ in range(h)]
        for idx in ones:
            r = idx // w
            c = idx % w
            block[r][c] = 1
        yield block




from itertools import combinations

def all_binary_blocks(height, width, ones):
    """
    Liefert alle 0-1-Matrizen (flach als Liste),
    der Größe height×width mit genau `ones` Einsen.
    """
    size = height * width
    if ones > size:
        return []

    blocks = []
    for pos in combinations(range(size), ones):
        block = [0] * size
        for p in pos:
            block[p] = 1
        blocks.append(block)

    return blocks


from itertools import product
from copy import deepcopy

def expand_weighted_matrix_to_binary(wb: SegmentedMatrix):
    """
    Expand a WeightedBlockMatrix into all binary SegmentedMatrices
    such that each block contains exactly as many 1s as the value
    in the top-left cell of the block.
    """

    B = len(wb.bandWidths)
    S = len(wb.stackWidths)

    # Precompute row ranges for bands
    band_rows = []
    r = 0
    for bw in wb.bandWidths:
        band_rows.append(list(range(r, r + bw)))
        r += bw

    # Precompute column ranges for stacks
    stack_cols = []
    c = 0
    for sw in wb.stackWidths:
        stack_cols.append(list(range(c, c + sw)))
        c += sw

    # For each block, compute all possible placements
    block_choices = []

    for b in range(B):
        for s in range(S):
            rows = band_rows[b]
            cols = stack_cols[s]

            block_cells = [(r, c) for r in rows for c in cols]

            k = wb.data[rows[0]][cols[0]]  # block value from top-left

            if k < 0 or k > len(block_cells):
                return []  # impossible block → no matrices

            choices = list(combinations(block_cells, k))
            block_choices.append(choices)

    # Cartesian product of all block choices
    results = []

    def backtrack(i, current):
        if i == len(block_choices):
            sm = SegmentedMatrix(wb.bandWidths[:], wb.stackWidths[:])
            sm.data = [[0] * sm.width for _ in range(sm.height)]

            for (r, c) in current:
                sm.data[r][c] = 1

            sm.sortPrefix = wb.sortPrefix[:]
            sm.info = deepcopy(wb.info)
            results.append(sm)
            return

        for choice in block_choices[i]:
            backtrack(i + 1, current + list(choice))

    backtrack(0, [])
    return results



from copy import deepcopy





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





# ----------------------
# Example usage (small demonstration)
# ----------------------
if __name__ == "__main__":

    pass

    nclues=2
    print()
    print("all01CellRepresentatives(nclues")
    #for nr, m in enumerate(allWeightedBlockRepresentatives(nclues)):
    #for nr, m in enumerate(all01BlockRepresentatives(nclues)):
    for nr, m in enumerate(SegmentedMatrix.all01CellRepresentatives(nclues)):
        print()
        print(nr)
        print(m.sortId)
        print(m.sortPrefix)
        m.print()


    import time

    """
    # test case
    maxNClues=3
    from contextlib import redirect_stdout
    with open('C:\\Users\\guent\\OneDrive\\work\\python\\out.txt', 'w') as f:
        startTime = time.time()
        with redirect_stdout(f):
            for nclues in list(range(2,maxNClues+1)):
                print('nClues =', nclues)
                print()
                print('all01BlockRepresentatives')
                print()
                SegmentedMatrix.resetPermutationCounters()
                for nr, m in enumerate(SegmentedMatrix.all01BlockRepresentatives(nclues)):
                    print(nr)
                    print(m.data,m.bandWidths,m.stackWidths)
                    m.print()
                    print()
                SegmentedMatrix.printPermutationCounters()
                print('allWeightedBlockRepresentatives')
                print()
                SegmentedMatrix.resetPermutationCounters()
                for nr, m in enumerate(SegmentedMatrix.allWeightedBlockRepresentatives(nclues)):
                    print(nr)
                    print(m.data,m.bandWidths,m.stackWidths)
                    m.print()
                    print()
                SegmentedMatrix.printPermutationCounters()
                print('allWeightedCellRepresentatives')
                print()
                SegmentedMatrix.resetPermutationCounters()
                for nr, m in enumerate(SegmentedMatrix.allWeightedCellRepresentatives(nclues)):
                    print(nr)
                    print(m.data,m.bandWidths,m.stackWidths)
                    m.print()
                    print()
                SegmentedMatrix.printPermutationCounters()
                print('all01CellRepresentatives')
                print()
                SegmentedMatrix.resetPermutationCounters()
                for nr, m in enumerate(SegmentedMatrix.all01CellRepresentatives(nclues)):
                    print(nr)
                    print(m.data,m.bandWidths,m.stackWidths)
                    m.print()
                    print()
                SegmentedMatrix.printPermutationCounters()
                print('allSymbolRepresentatives')
                print()
                SegmentedMatrix.resetPermutationCounters()
                for nr, m in enumerate(SegmentedMatrix.allSymbolRepresentatives(nclues)):
                    print(nr)
                    print(m.data,m.bandWidths,m.stackWidths)
                    m.print()
                    print()
                SegmentedMatrix.printPermutationCounters()
        endTime = time.time()
        print("time elapse:",endTime - startTime)

    """

    """
    # test case
    '''
        10|00
        01|00
        --+--
        00|11
        01|00
    '''
    sm=SegmentedMatrix([2,2],[2,2])
    sm.read([[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,1,0,0]])
    print("Matrix")
    sm.print()
    print("Optimum")
    sm.toptimize().print()
    print("Matrix")
    sm.print()
    print("Optimum")
    for myMatrix in SegmentedMatrix.filterMaximalSegmentedMatrices([sm]):
        myMatrix.print()
    
    """

    """
    # test case
    sm=SegmentedMatrix([1, 1],[2, 1])
    sm.read([[1, 0, 1], [0, 1, 0]])
    print("Matrix")
    sm.print()
    print("Optimum")
    for myMatrix in SegmentedMatrix.filterMaximalSegmentedMatrices([sm]):
        myMatrix.print()
    """

    maxNClues=4
    
    minNClues=2
    from contextlib import redirect_stdout
    with open('C:\\Users\\guent\\OneDrive\\work\\python\\out.txt', 'w') as f:
        startTime_all = time.time()
        with redirect_stdout(f):
            for nclues in list(range(minNClues,maxNClues+1)):
                print('nClues =', nclues)
                print()

                for method in [SegmentedMatrix.all01BlockRepresentatives, SegmentedMatrix.allWeightedBlockRepresentatives,
                               SegmentedMatrix.allWeightedCellRepresentatives,SegmentedMatrix.all01CellRepresentatives,
                               SegmentedMatrix.allSymbolRepresentatives]:
                    print('function:', method.__name__)
                    print()
                    SegmentedMatrix.resetPermutationCounters()
                    startTime=time.time()
                    for nr, m in enumerate(method(nclues)):
                        print(nr)
                        print(m.sortPrefix)
                        print(m.data,m.bandWidths,m.stackWidths)
                        m.print()
                        print()
                    endTime=time.time()
                    print("time elapse for",method.__name__+", Clues =",str(nclues)+":",endTime - startTime)
                    SegmentedMatrix.printPermutationCounters()
                    print()
        endTime_all = time.time()
        print("time elapse:",endTime_all - startTime_all)
        


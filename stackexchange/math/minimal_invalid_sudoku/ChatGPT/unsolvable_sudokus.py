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

    # -------------------------
    # Properties
    # -------------------------
    @property
    def rowDim(self) -> int:
        return len(self.data)

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

    @property
    def sortId(self) -> Tuple:
        """
        sortId = tuple(sortPrefix) + flattened data tuple.
        This makes sortPrefix the leading part of the lexicographic key.
        """
        prefix = tuple(self.sortPrefix)   # ensure immutable
        flat = tuple(v for row in self.data for v in row)
        #return prefix + flat
        #return (prefix,flat)
        return prefix+(flat,)

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
        new_bandWidths = self.stackWidths[:]
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





def all01Representatives(clueCount: int):
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
            m = SegmentedMatrix([1,1,1], [1,1,1])
            m.data = [[0]*3 for _ in range(3)]

            # Insert the 1s
            for p in ones:
                r = p // 3
                c = p % 3
                m.data[r][c] = 1

            # Optimize under symmetries
            opt = m.toptimize()

            # MUST set prefix BEFORE getting sortId
            opt.sortPrefix = [k]

            # Now compute canonical ID including prefix
            sid = opt.sortId

            # If this representative has not yet been seen:
            if sid not in result:
                result[sid] = opt.clone()

    # result: dict mapping sortId -> SegmentedMatrix
    # We build a new dict `filtered_result` with only the survivors.

    filtered_result = {}

    for sid, sm in result.items():
        # Keep original sortId for comparison (tuple)
        original_id = sid

        # Compute rotated + optimized representative
        rotated = sm.clone()
        rotated.rotate()
        opt_rot = rotated.toptimize()

        # If opt_rot has a strictly larger sortId, discard the original entry
        if opt_rot.sortId > original_id:
            # discard: do not add to filtered_result
            continue

        # Otherwise keep the original entry
        # (use the stored matrix; ensure we store a clone to avoid aliasing)
        filtered_result[original_id] = sm.clone()

    # Replace result dict with filtered_result
    result = filtered_result
                

    # produce sorted output (by full sortId including prefix)
    return [result[key] for key in sorted(result.keys())]
    #return result


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


def allWeightedRepresentatives(clueCount: int):
    """
    Erweiterte Version:
    - erzeugt gewichtete Matrizen basierend auf all01Representatives
    - reduziert alle Matrizen
    - filtert:
        A: toptimize(M) > M  -> entferne M
        B: toptimize(rotate(M)) > M -> entferne M
    - sortiert das Ergebnis absteigend nach sortId
    """
    from itertools import product

    bases = all01Representatives(clueCount)
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
            sm.sortPrefix = list(base_sortKey)

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

    filtered = []
    for sm in results:
        orig_id = sm.sortId

        # Bedingung A: toptimize(M) größer als M?
        topo = sm.toptimize()
        if topo.sortId > orig_id:
            continue

        # Bedingung B: rotate(M).toptimize() größer als M?
        rot = sm.clone()
        rot.rotate()
        rot_opt = rot.toptimize()

        if rot_opt.sortId > orig_id:
            continue

        filtered.append(sm)

    # Sortieren nach sortId absteigend
    filtered.sort(key=lambda sm: sm.sortId, reverse=True)

    return filtered



# -------------------------
# Example usage / quick test
# -------------------------
if __name__ == "__main__":
    # create a classic segmented matrix with band/stack widths 3,3,3
    sm = SegmentedMatrix([3, 3, 3], [3, 3, 3])
    # fill a few values using box notation
    sm[0][0][0][0] = 1
    sm[0][0][0][1] = 2
    sm[0][0][1][0] = 3
    sm[0][0][1][2] = 4
    print("Original:")
    sm.print()

    # read with row lists (shorter rows are padded)
    rows = [
        [1, 2, 0, 0, 0, 0, 0, 0, 0],
        [3, 0, 4],  # will be padded to 9 cols
        [],  # becomes zeros
    ] + [[0] * 9 for _ in range(6)]  # rest rows
    sm.read(rows)
    print("\nAfter read(rows):")
    sm.print()

    # reduce (should remove nothing here)
    sm.reduce()
    print("\nAfter reduce():")
    sm.print()

    # expand small example: a 1x1 block matrix
    bm = BinaryBoxMatrix()
    bm[0][0][0][0] = 1
    bm[2][2][0][0] = 1
    print("\nBinaryBoxMatrix before expand:")
    bm.print()
    bm.expand()
    print("\nBinaryBoxMatrix after expand():")
    bm.print()

    # toptimize example (will produce canonical representative)
    sm2 = SegmentedMatrix([1, 1, 1], [1, 1, 1])
    sm2.read([[9], [1], [2]])
    print("\nBefore toptimize (sm2):")
    sm2.print()
    best = sm2.toptimize()
    print("\nAfter toptimize (best):")
    best.print()



# ----------------------
# Example usage (small demonstration)
# ----------------------
if __name__ == "__main__":
    nclues=6
    #print(allWeightedRepresentatives(4))
    print()
    print("All Representativesfor", nclues, " clues")
    for nr, m in enumerate(allWeightedRepresentatives(nclues)):
        print()
        print(nr)
        print(m.sortId)
        print(m.sortPrefix)
        m.print()




# Run the regression test
# test_clone()



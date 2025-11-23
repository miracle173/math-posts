def generate_sequences(n):
    result = []
    seq = [0] * n
    seq[0] = 1

    def backtrack(i, current_max):
        if i == n:
            result.append(tuple(seq))   # tuple macht spätere Sortierung einfacher
            return

        # Option 1: benutze eine schon verwendete Zahl (1 .. current_max)
        for x in range(1, current_max + 1):
            seq[i] = x
            backtrack(i + 1, current_max)

        # Option 2: benutze current_max + 1 (neue Zahl)
        seq[i] = current_max + 1
        backtrack(i + 1, current_max + 1)

    backtrack(1, 1)

    # Lexikographisch sortieren
    result.sort()

    return result


# Beispiel
if __name__ == "__main__":
    while True:
        n = int(input("n = "))
        if (n==0):
            break
        sequences = gene rate_sequences(n)
        for s in sequences:
            print(s)
        print(f"Anzahl: {len(sequences)}")

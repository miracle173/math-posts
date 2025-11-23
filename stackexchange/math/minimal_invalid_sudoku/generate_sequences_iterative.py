def generate_sequences_iterative(n):
    if n == 0:
        return []

    # wir bauen von kürzeren zu längeren Folgen auf
    sequences = [(1,)]  # Start: nur eine Folge der Länge 1

    for length in range(2, n + 1):
        new_sequences = []
        for seq in sequences:
            current_max = max(seq)

            # Option 1: Wiederverwendung eines alten Wertes
            for x in range(1, current_max + 1):
                new_sequences.append(seq + (x,))

            # Option 2: neues Element current_max + 1
            new_sequences.append(seq + (current_max + 1,))

        sequences = new_sequences

    # lexikographisch sortieren
    sequences.sort()
    return sequences


# Beispiel
if __name__ == "__main__":
    while True:
        n = int(input("n = "))
        if (n==0):
            break
        sequences = generate_sequences_iterative(n)
        for s in sequences:
            print(s)
        print(f"Anzahl: {len(sequences)}")

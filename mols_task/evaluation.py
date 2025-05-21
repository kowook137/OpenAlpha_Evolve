# mols_task/evaluation.py

def evaluate(squares) -> dict[str, float]:
    def latin_score(square):
        rows = sum(len(set(row)) == 10 for row in square)
        cols = sum(len(set(col)) == 10 for col in zip(*square))
        return (rows + cols) / 20  # 0.0 ~ 1.0

    def orth_score(a, b):
        seen = set()
        dup = 0
        for i in range(10):
            for j in range(10):
                pair = (a[i][j], b[i][j])
                if pair in seen:
                    dup += 1
                seen.add(pair)
        return 1 - dup / 100

    latin_total = sum(latin_score(sq) for sq in squares) / len(squares)
    orth_total = (
        orth_score(squares[0], squares[1]) +
        orth_score(squares[1], squares[2]) +
        orth_score(squares[2], squares[0])
    ) / 3

    return {
        "score": 0.6 * latin_total + 0.4 * orth_total,
        "latin_score": latin_total,
        "orthogonality_score": orth_total
    }
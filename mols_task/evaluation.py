# mols_task/evaluation.py

def evaluate(squares) -> dict[str, float]:
    def latin_score(square):
        rows = sum(len(set(row)) == 10 for row in square)
        cols = sum(len(set(col)) == 10 for col in zip(*square))
        return (rows + cols) / 20

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

    # Soft penalty: 라틴 조건이 깨질수록 orthogonal의 비중을 줄임
    orth_weight = min(1.0, max(0.0, (latin_total - 0.6) / 0.4))  # 0.6~1.0 구간에서 선형 변화
    latin_weight = 1.0 - orth_weight

    score = latin_weight * latin_total + orth_weight * orth_total

    return {
        "score": score,
        "latin_score": latin_total,
        "orthogonality_score": orth_total
    }
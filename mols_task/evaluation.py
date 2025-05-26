from mols_task.reference_output import EXAMPLE_OUTPUT

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

    latin_total = sum(latin_score(sq) for sq in squares) / 3
    orth_total = (
        orth_score(squares[0], squares[1]) +
        orth_score(squares[1], squares[2]) +
        orth_score(squares[2], squares[0])
    ) / 3

    # === 복사 방지: 예시 output과 완전 일치하면 무조건 탈락
    if squares == EXAMPLE_OUTPUT:
        return {
            "score": 0.0,
            "latin_score": 0.0,
            "orthogonality_score": 0.0
        }

    # === 구조 유사성 감점 ===
    def is_structure_similar(sq1, sq2):
        for i in range(10):
            if sq1[i] == sq2[i]:
                return True  # row-wise identical
        for j in range(10):
            col1 = [sq1[i][j] for i in range(10)]
            col2 = [sq2[i][j] for i in range(10)]
            if col1 == col2:
                return True  # col-wise identical
        return False

    similarity_penalty = 0
    for ref_sq in EXAMPLE_OUTPUT:
        for test_sq in squares:
            if is_structure_similar(test_sq, ref_sq):
                similarity_penalty += 0.15  # per matched structure

    # Clamped between 0 and 1
    similarity_penalty = min(0.6, similarity_penalty)

    # 기존 soft weighting 유지
    orth_weight = min(1.0, max(0.0, (latin_total - 0.6) / 0.4))
    latin_weight = 1.0 - orth_weight

    raw_score = latin_weight * latin_total + orth_weight * orth_total
    penalized_score = max(0.0, raw_score * (1 - similarity_penalty))

    return {
        "score": penalized_score,
        "latin_score": latin_total,
        "orthogonality_score": orth_total
    }
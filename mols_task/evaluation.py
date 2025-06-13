def evaluate(squares) -> dict[str, float]:
    def is_perfect_latin(square):
        """완벽한 Latin square인지 검사 (4x4)"""
        for row in square:
            if len(set(row)) != 4 or not all(0 <= x <= 3 for x in row):
                return False
        for j in range(4):
            col = [square[i][j] for i in range(4)]
            if len(set(col)) != 4 or not all(0 <= x <= 3 for x in col):
                return False
        return True

    def latin_score(square):
        """
        Latin square 품질 평가 (4x4)
        - 완벽한 Latin square는 1.0 반환
        - 그 외에는 각 행/열의 품질에 따른 점수 반환
        """
        if is_perfect_latin(square):
            return 1.0

        row_scores = []
        col_scores = []
        range_penalty = 0
        
        for row in square:
            if not all(0 <= x <= 3 for x in row):
                range_penalty += 0.1
            unique_ratio = len(set(row)) / 4
            row_scores.append(unique_ratio)
        
        for j in range(4):
            col = [square[i][j] for i in range(4)]
            if not all(0 <= x <= 3 for x in col):
                range_penalty += 0.1
            unique_ratio = len(set(col)) / 4
            col_scores.append(unique_ratio)
        
        base_score = (sum(row_scores) + sum(col_scores)) / 8  # 4 rows + 4 cols = 8
        if base_score > 0.95:
            base_score = base_score * 1.1
        final_score = max(0, base_score - range_penalty)
        return min(0.99, final_score)

    def orth_score(a, b):
        """
        직교성 평가 (4x4)
        - 중복 쌍의 수와 분포를 고려
        - 완벽한 직교성에 보너스 부여
        """
        seen = {}
        total_pairs = 16  # 4x4
        for i in range(4):
            for j in range(4):
                pair = (a[i][j], b[i][j])
                seen[pair] = seen.get(pair, 0) + 1
        duplicates = sum(count - 1 for count in seen.values())
        base_score = 1 - duplicates / total_pairs
        max_duplicates = max(seen.values())
        if max_duplicates <= 1:
            distribution_bonus = 0.05
        else:
            distribution_bonus = 0
        return min(1.0, base_score + distribution_bonus)

    latin_scores = [latin_score(sq) for sq in squares]
    latin_total = sum(latin_scores) / 2
    orth_scores = [orth_score(squares[0], squares[1])]
    orth_total = orth_scores[0]
    all_perfect_latin = all(is_perfect_latin(sq) for sq in squares)
    
    if all_perfect_latin:
        orth_weight = 0.8
        latin_weight = 0.2
    else:
        if latin_total > 0.95:
            orth_weight = 0.4
            latin_weight = 0.6
        else:
            orth_weight = 0.2
            latin_weight = 0.8
    final_score = latin_weight * latin_total + orth_weight * orth_total
    if not all_perfect_latin:
        final_score *= 0.8
    return {
        "score": final_score,
        "latin_score": latin_total,
        "orthogonality_score": orth_total,
        "latin_scores": latin_scores,
        "orth_scores": orth_scores,
        "is_perfect_latin": all_perfect_latin
    }
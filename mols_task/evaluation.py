def evaluate(squares) -> dict[str, float]:
    def is_perfect_latin(square):
        """완벽한 Latin square인지 검사"""
        for row in square:
            if len(set(row)) != 3 or not all(0 <= x <= 2 for x in row):
                return False
        for j in range(3):
            col = [square[i][j] for i in range(3)]
            if len(set(col)) != 3 or not all(0 <= x <= 2 for x in col):
                return False
        return True

    def latin_score(square):
        """
        Latin square 품질 평가
        - 완벽한 Latin square는 1.0 반환
        - 그 외에는 각 행/열의 품질에 따른 점수 반환
        """
        # 완벽한 Latin square인 경우 즉시 1.0 반환
        if is_perfect_latin(square):
            return 1.0

        row_scores = []
        col_scores = []
        range_penalty = 0
        
        # 행 평가
        for row in square:
            if not all(0 <= x <= 2 for x in row):
                range_penalty += 0.1
            unique_ratio = len(set(row)) / 3
            row_scores.append(unique_ratio)
        
        # 열 평가
        for j in range(3):
            col = [square[i][j] for i in range(3)]
            if not all(0 <= x <= 2 for x in col):
                range_penalty += 0.1
            unique_ratio = len(set(col)) / 3
            col_scores.append(unique_ratio)
        
        # 기본 점수 계산
        base_score = (sum(row_scores) + sum(col_scores)) / 6  # 3 rows + 3 cols = 6
        
        # 완벽도에 따른 보너스 점수
        if base_score > 0.95:  # 95% 이상 정확도
            base_score = base_score * 1.1  # 10% 보너스
        
        # 범위 페널티 적용
        final_score = max(0, base_score - range_penalty)
        
        # 1.0 초과 방지
        return min(0.99, final_score)  # 완벽하지 않은 경우 최대 0.99

    def orth_score(a, b):
        """
        직교성 평가
        - 중복 쌍의 수와 분포를 고려
        - 완벽한 직교성에 보너스 부여
        """
        seen = {}  # 각 쌍의 출현 횟수를 기록
        total_pairs = 9  # 3x3
        
        for i in range(3):
            for j in range(3):
                pair = (a[i][j], b[i][j])
                seen[pair] = seen.get(pair, 0) + 1
        
        # 중복 쌍 수 계산
        duplicates = sum(count - 1 for count in seen.values())
        
        # 기본 점수 계산
        base_score = 1 - duplicates / total_pairs
        
        # 분포도 보너스: 중복이 고르게 분포되어 있으면 약간의 보너스
        max_duplicates = max(seen.values())
        if max_duplicates <= 1:  # 3x3에서는 중복이 없어야 함
            distribution_bonus = 0.05
        else:
            distribution_bonus = 0
            
        # 최종 점수
        return min(1.0, base_score + distribution_bonus)

    # 각 스퀘어의 Latin 점수 계산 (2개만)
    latin_scores = [latin_score(sq) for sq in squares]
    latin_total = sum(latin_scores) / 2  # 2개 square의 평균

    # 직교성 점수 계산 (1개 쌍만)
    orth_scores = [orth_score(squares[0], squares[1])]
    orth_total = orth_scores[0]  # 1개 쌍만 있음

    # 완벽한 Latin square 여부 확인
    all_perfect_latin = all(is_perfect_latin(sq) for sq in squares)
    
    if all_perfect_latin:
        # 모든 square가 완벽한 Latin square인 경우
        # orthogonality에 더 큰 가중치 부여
        orth_weight = 0.8
        latin_weight = 0.2
    else:
        # Latin square가 완벽하지 않은 경우
        # Latin 품질 향상에 더 큰 가중치 부여
        if latin_total > 0.95:  # 거의 완벽한 경우
            orth_weight = 0.4
            latin_weight = 0.6
        else:  # 개선이 많이 필요한 경우
            orth_weight = 0.2
            latin_weight = 0.8

    # 최종 점수 계산 (reference penalty 완전 제거)
    final_score = latin_weight * latin_total + orth_weight * orth_total

    # 완벽한 Latin square가 아닌 경우 점수 감소
    if not all_perfect_latin:
        final_score *= 0.8  # 20% 감점

    return {
        "score": final_score,
        "latin_score": latin_total,
        "orthogonality_score": orth_total,
        "latin_scores": latin_scores,
        "orth_scores": orth_scores,
        "is_perfect_latin": all_perfect_latin
    }
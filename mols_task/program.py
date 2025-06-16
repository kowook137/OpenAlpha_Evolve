# mols_task/program.py - AlphaEvolve Style Multi-Component Algorithm

# EVOLVE-BLOCK-START
import random
import itertools
import numpy as np
from typing import List, Tuple, Optional
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
def base_square_generator(size: int = 4) -> List[List[int]]:
    """Base Latin square generation strategy (4x4 기준)"""
    square = []
    for i in range(size):
        row = [(i + j) % size for j in range(size)]
        square.append(row)
    return square
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
def orthogonality_transformer(base_square: List[List[int]], transformation_id: int = 0) -> List[List[int]]:
    """Transform base square to create orthogonal partner (4x4 기준, 다양한 변이 추가)"""
    size = len(base_square)
    if transformation_id == 0:
        transformed = []
        for i in range(size):
            row = [(base_square[i][j] * 2 + j) % size for j in range(size)]
            transformed.append(row)
        return transformed
    elif transformation_id == 1:
        transformed = []
        for i in range(size):
            row = [(base_square[i][j] + i * 2) % size for j in range(size)]
            transformed.append(row)
        return transformed
    elif transformation_id == 2:
        # transpose
        return [list(row) for row in zip(*base_square)]
    elif transformation_id == 3:
        # reverse rows
        return [row[::-1] for row in base_square]
    elif transformation_id == 4:
        # reverse columns
        return base_square[::-1]
    elif transformation_id == 5:
        # 완전 랜덤 라틴스퀘어
        return random_latin_square(size)
    else:
        # cyclic shift by transformation_id
        transformed = []
        for i in range(size):
            row = [(base_square[i][(j+transformation_id)%size]) for j in range(size)]
            transformed.append(row)
        return transformed
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
def pair_optimizer(square1: List[List[int]], square2: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
    """Optimize the pair of squares for better orthogonality (4x4 기준, 다양한 변이 추가)"""
    import copy
    size = len(square1)
    best_s1, best_s2 = [row[:] for row in square1], [row[:] for row in square2]
    best_score = calculate_orthogonality_score(best_s1, best_s2)
    for _ in range(20):  # 반복 횟수 증가
        s1_copy = copy.deepcopy(best_s1)
        s2_copy = copy.deepcopy(best_s2)
        op = random.choice(['swap_row', 'swap_col', 'randomize', 'block_swap', 'value_mutate'])
        if op == 'swap_row':
            i1, i2 = random.sample(range(size), 2)
            s2_copy[i1], s2_copy[i2] = s2_copy[i2], s2_copy[i1]
        elif op == 'swap_col':
            j1, j2 = random.sample(range(size), 2)
            for row in s2_copy:
                row[j1], row[j2] = row[j2], row[j1]
        elif op == 'randomize':
            # 완전 랜덤 라틴스퀘어 생성
            s2_copy = random_latin_square(size)
        elif op == 'block_swap':
            # 2x2 블록 swap (예시)
            if size >= 2:
                r, c = random.randint(0, size-2), random.randint(0, size-2)
                for dr in range(2):
                    for dc in range(2):
                        s2_copy[r+dr][c+dc], s2_copy[(r+dr+1)%size][(c+dc+1)%size] = s2_copy[(r+dr+1)%size][(c+dc+1)%size], s2_copy[r+dr][c+dc]
        elif op == 'value_mutate':
            i, j = random.randint(0, size-1), random.randint(0, size-1)
            s2_copy[i][j] = random.randint(0, size-1)
        # latin성 체크
        if is_latin_square(s2_copy):
            score = calculate_orthogonality_score(s1_copy, s2_copy)
            if score > best_score:
                best_s2 = [row[:] for row in s2_copy]
                best_score = score
    return best_s1, best_s2
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
def algorithm_config():
    """Configuration parameters for the MOLS algorithm (4x4 기준)"""
    return {
        'base_strategy': 'cyclic',
        'transformation_method': 0,
        'optimization_iterations': 10,
        'mutation_probability': 0.3,
        'search_depth': 3
    }
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
def generate_MOLS_n(n: int = 4) -> List[List[List[int]]]:
    """Main MOLS generation algorithm combining all components for 4x4 squares"""
    config = algorithm_config()
    base_square = base_square_generator(n)
    partner_square = orthogonality_transformer(base_square, config['transformation_method'])
    optimized_s1, optimized_s2 = pair_optimizer(base_square, partner_square)
    return [optimized_s1, optimized_s2]
# EVOLVE-BLOCK-END

# Helper functions (not evolved)
def calculate_orthogonality_score(s1: List[List[int]], s2: List[List[int]]) -> float:
    """Calculate orthogonality score between two squares"""
    pairs = set()
    duplicates = 0
    total_pairs = len(s1) * len(s1[0])
    
    for i in range(len(s1)):
        for j in range(len(s1[0])):
            pair = (s1[i][j], s2[i][j])
            if pair in pairs:
                duplicates += 1
            pairs.add(pair)
    
    return (total_pairs - duplicates) / total_pairs

def is_latin_square(square: List[List[int]]) -> bool:
    """Check if square is a valid Latin square"""
    size = len(square)
    expected_set = set(range(size))
    
    # Check rows
    for row in square:
        if set(row) != expected_set:
            return False
    
    # Check columns
    for j in range(size):
        col = [square[i][j] for i in range(size)]
        if set(col) != expected_set:
            return False
    
    return True

def is_valid_swap(square: List[List[int]], i1: int, j1: int, i2: int, j2: int) -> bool:
    """Check if swapping two elements would maintain Latin square property"""
    # Simple heuristic - can be evolved
    return i1 != i2 and j1 != j2

def random_latin_square(size):
    while True:
        square = []
        for i in range(size):
            row = list(range(size))
            np.random.shuffle(row)
            square.append(row)
        if is_latin_square(square):
            return square
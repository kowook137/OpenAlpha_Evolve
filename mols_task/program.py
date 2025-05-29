# mols_task/program.py - AlphaEvolve Style Multi-Component Algorithm

# EVOLVE-BLOCK-START
import random
import itertools
import numpy as np
from typing import List, Tuple, Optional
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
def base_square_generator(size: int = 3) -> List[List[int]]:
    """Base Latin square generation strategy"""
    # Standard cyclic Latin square
    square = []
    for i in range(size):
        row = [(i + j) % size for j in range(size)]
        square.append(row)
    return square
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
def orthogonality_transformer(base_square: List[List[int]], transformation_id: int = 0) -> List[List[int]]:
    """Transform base square to create orthogonal partner"""
    size = len(base_square)
    
    if transformation_id == 0:
        # Row permutation strategy
        transformed = []
        for i in range(size):
            row = [(base_square[i][j] * 2 + j) % size for j in range(size)]
            transformed.append(row)
        return transformed
    elif transformation_id == 1:
        # Column permutation strategy  
        transformed = []
        for i in range(size):
            row = [(base_square[i][j] + i * 2) % size for j in range(size)]
            transformed.append(row)
        return transformed
    else:
        # Hybrid strategy
        transformed = []
        for i in range(size):
            row = [(base_square[i][j] + i + j) % size for j in range(size)]
            transformed.append(row)
        return transformed
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
def pair_optimizer(square1: List[List[int]], square2: List[List[int]]) -> Tuple[List[List[int]], List[List[int]]]:
    """Optimize the pair of squares for better orthogonality"""
    size = len(square1)
    
    # Local search optimization
    best_s1, best_s2 = [row[:] for row in square1], [row[:] for row in square2]
    best_score = calculate_orthogonality_score(best_s1, best_s2)
    
    # Try small mutations
    for _ in range(10):  # Limited iterations for efficiency
        # Random swap in second square
        s2_copy = [row[:] for row in square2]
        i1, j1 = random.randint(0, size-1), random.randint(0, size-1)
        i2, j2 = random.randint(0, size-1), random.randint(0, size-1)
        
        if is_valid_swap(s2_copy, i1, j1, i2, j2):
            s2_copy[i1][j1], s2_copy[i2][j2] = s2_copy[i2][j2], s2_copy[i1][j1]
            
            if is_latin_square(s2_copy):
                score = calculate_orthogonality_score(square1, s2_copy)
                if score > best_score:
                    best_s2 = s2_copy
                    best_score = score
    
    return best_s1, best_s2
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
def algorithm_config():
    """Configuration parameters for the MOLS algorithm"""
    return {
        'base_strategy': 'cyclic',  # 'cyclic', 'random', 'systematic'
        'transformation_method': 0,  # 0, 1, 2
        'optimization_iterations': 10,
        'mutation_probability': 0.3,
        'search_depth': 3
    }
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
def generate_MOLS_3() -> List[List[List[int]]]:
    """Main MOLS generation algorithm combining all components"""
    config = algorithm_config()
    
    # Step 1: Generate base square
    base_square = base_square_generator(3)
    
    # Step 2: Create orthogonal partner
    partner_square = orthogonality_transformer(base_square, config['transformation_method'])
    
    # Step 3: Optimize the pair
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
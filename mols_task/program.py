# mols_task/program.py

# EVOLVE-BLOCK-START
import random
import itertools
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
def generate_MOLS_10():
    square1 = [[(i + j) % 10 for j in range(10)] for i in range(10)]
    square2 = [[(i + 3*j) % 10 for j in range(10)] for i in range(10)]
    square3 = [[(i + 7*j) % 10 for j in range(10)] for i in range(10)]
    return [square1, square2, square3]
# EVOLVE-BLOCK-END
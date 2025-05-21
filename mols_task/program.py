# mols_task/program.py

# EVOLVE-BLOCK-START
def generate_MOLS_10():
    square = [[(i + j) % 10 for j in range(10)] for i in range(10)]
    return [square, square, square]
# EVOLVE-BLOCK-END
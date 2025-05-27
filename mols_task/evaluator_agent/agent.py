import os
from mols_task.evaluation import evaluate

class EvaluatorAgent:
    def __init__(self):
        self.best_program = None
        self.best_fitness = None
        self.best_squares = None

    def print_matrix(self, matrix, name="Matrix"):
        """행렬을 보기 좋게 출력하는 함수"""
        print(f"\n{name}:")
        print("-" * (4 * len(matrix) + 1))
        for row in matrix:
            print("|", end=" ")
            for val in row:
                print(f"{val:2d}", end=" ")
            print("|")
        print("-" * (4 * len(matrix) + 1))

    def print_best_result(self):
        """최적의 프로그램과 결과를 출력"""
        if self.best_program and self.best_squares:
            print("\n=== Alpha Evolve가 찾은 최적의 프로그램 ===")
            print(self.best_program)
            
            print("\n=== 생성된 MOLS ===")
            for i, square in enumerate(self.best_squares, 1):
                self.print_matrix(square, f"Latin Square {i}")
            
            print("\nOrthogonality Check:")
            print("-" * 40)
            for i in range(len(self.best_squares)):
                for j in range(i + 1, len(self.best_squares)):
                    pairs = set()
                    duplicates = 0
                    for r in range(len(self.best_squares[i])):
                        for c in range(len(self.best_squares[i])):
                            pair = (self.best_squares[i][r][c], self.best_squares[j][r][c])
                            if pair in pairs:
                                duplicates += 1
                            pairs.add(pair)
                    print(f"Squares {i+1} and {j+1}: {duplicates} duplicate pairs")

    def evaluate_program(self, program_code, program_id):
        try:
            # 프로그램 실행
            namespace = {}
            exec(program_code, namespace)
            squares = namespace['generate_MOLS_10']()
            
            # 평가 수행
            fitness = evaluate(squares)
            
            # 최고 성능 갱신 시 저장
            if self.best_fitness is None or fitness['score'] > self.best_fitness['score']:
                self.best_fitness = fitness
                self.best_program = program_code
                self.best_squares = squares
            
            return fitness
            
        except Exception as e:
            print(f"Error evaluating program {program_id}: {str(e)}")
            return {
                'score': 0,
                'latin_score': 0,
                'orthogonality_score': 0,
                'error': str(e)
            } 
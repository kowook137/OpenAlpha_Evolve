import os
import asyncio
import time
import traceback
from mols_task.evaluation import evaluate
from core.interfaces import EvaluatorAgentInterface, Program, TaskDefinition, BaseAgent
from typing import Optional, Dict, Any

class EvaluatorAgent(EvaluatorAgentInterface, BaseAgent):
    def __init__(self, task_definition: Optional[TaskDefinition] = None):
        super().__init__()
        self.task_definition = task_definition
        self.best_program = None
        self.best_fitness = None
        self.best_squares = None
        self._cache = {}  # Simple in-memory cache

    async def execute(self, program: Program, task: TaskDefinition) -> Program:
        return await self.evaluate_program(program, task)

    async def evaluate_program(self, program: Program, task: TaskDefinition) -> Program:
        """Asynchronous program evaluation"""
        # Check cache
        cache_key = f"{program.code}_{task.id}"
        if cache_key in self._cache:
            cached_result = self._cache[cache_key]
            program.status = cached_result["status"]
            program.errors = cached_result["errors"]
            program.fitness_scores = cached_result["fitness_scores"]
            return program

        program.status = "evaluating"
        program.errors = []
        program.fitness_scores = {}

        try:
            # Safe environment code execution
            start_time = time.time()
            squares = await self._execute_safely(program.code)
            execution_time = (time.time() - start_time) * 1000  # ms conversion

            # Evaluation
            fitness = evaluate(squares)
            
            # Update best performance
            if self.best_fitness is None or fitness['score'] > self.best_fitness['score']:
                self.best_fitness = fitness
                self.best_program = program.code
                self.best_squares = squares

            # Record results
            program.fitness_scores = {
                **fitness,  # Existing scores
                "runtime_ms": execution_time,
                "code_length": len(program.code.split('\n'))
            }
            program.status = "evaluated"

            # Cache the result
            self._cache[cache_key] = {
                "status": program.status,
                "errors": program.errors,
                "fitness_scores": program.fitness_scores
            }

        except Exception as e:
            program.status = "failed_evaluation"
            program.errors.append(str(e))
            program.errors.append(traceback.format_exc())
            program.fitness_scores = {
                'score': 0.0,
                'latin_score': 0.0,
                'orthogonality_score': 0.0,
                'error': str(e)
            }

        return program

    async def _execute_safely(self, code: str):
        """Safe environment code execution"""
        try:
            namespace = {}
            exec(code, namespace)
            squares = namespace['generate_MOLS_n'](4)
            return squares
        except Exception as e:
            raise RuntimeError(f"Code execution failed: {str(e)}")

    def print_matrix(self, matrix, name="Matrix"):
        """Function to print matrix in a readable format"""
        print(f"\n{name}:")
        print("-" * (4 * len(matrix) + 1))
        for row in matrix:
            print("|", end=" ")
            for val in row:
                print(f"{val:2d}", end=" ")
            print("|")
        print("-" * (4 * len(matrix) + 1))

    def print_best_result(self):
        """Print the best program and results"""
        if self.best_program and self.best_squares:
            print("\n=== Best Program Found by Alpha Evolve ===")
            print(self.best_program)
            
            print("\n=== Generated MOLS ===")
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
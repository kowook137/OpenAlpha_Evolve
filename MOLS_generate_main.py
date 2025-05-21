"""
Main entry point for the AlphaEvolve Pro application.
Orchestrates the different agents and manages the evolutionary loop.
"""
import asyncio
import logging
import sys
import os

                                               
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from task_manager.agent import TaskManagerAgent
from core.interfaces import TaskDefinition
from config import settings

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.LOG_FILE, mode="a")        # logging 을 위한 설정 필요함.
    ]
)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting OpenAlpha_Evolve autonomous algorithmic evolution")
    logger.info(f"Configuration: Population Size={settings.POPULATION_SIZE}, Generations={settings.GENERATIONS}")
    logger.info(f"LLM Models: Pro={settings.GEMINI_PRO_MODEL_NAME}, Flash={settings.GEMINI_FLASH_MODEL_NAME}, Eval={settings.GEMINI_EVALUATION_MODEL}")


    task = TaskDefinition(
        id="generate 10x10 MOLS problem",
        description=(
            "Implement a function `generate_mols()` that returns a list of three 10x10 Latin squares. "
            "Each Latin square is a 10x10 grid where integers from 0 to 9 appear exactly once in each row and each column. "
            "The goal is to generate three Latin squares that are as close as possible to being mutually orthogonal. "
            "Two Latin squares A and B are orthogonal if the set of ordered pairs (A[i][j], B[i][j]) for all positions (i, j) "
            "are all distinct. Full mutual orthogonality among three squares means that each of the three square pairs "
            "(A, B), (A, C), and (B, C) satisfy this condition."

            "The scoring function rewards both Latin validity and orthogonality. "
            "Each Latin square is evaluated on how closely it satisfies the Latin condition: having no duplicates in any row or column. "
            "Each pair of squares is then evaluated based on the number of duplicate pairs (A[i][j], B[i][j])—fewer duplicates mean higher orthogonality. "
            "Partial credit is given to squares that approximate these conditions even if they are not fully satisfied."

            "The function should return: List[List[List[int]]], where the outer list contains 3 squares, each represented as a 10x10 grid of integers 0–9."
        ),
        function_name_to_evolve="generate_MOLS_10",
        input_output_examples = [
            {
                "input": [],
                "output": [
                    [   # Latin Square 1
                        [0, 8, 9, 7, 5, 6, 4, 2, 3, 1],
                        [9, 1, 4, 6, 2, 7, 3, 8, 0, 5],
                        [7, 4, 2, 5, 1, 3, 8, 6, 9, 0],
                        [8, 6, 5, 3, 9, 2, 1, 0, 4, 7],
                        [6, 2, 1, 8, 4, 0, 9, 5, 7, 3],
                        [4, 9, 3, 2, 7, 5, 0, 1, 6, 8],
                        [5, 3, 7, 1, 0, 8, 6, 9, 2, 4],
                        [3, 5, 0, 9, 8, 4, 2, 7, 1, 6],
                        [1, 7, 6, 0, 3, 9, 5, 4, 8, 2],
                        [2, 0, 8, 4, 6, 1, 7, 3, 5, 9]
                    ],
                    [   # Latin Square 2
                        [0, 7, 8, 9, 1, 2, 3, 4, 5, 6],
                        [9, 0, 6, 1, 8, 3, 2, 5, 4, 7],
                        [7, 2, 0, 4, 3, 9, 1, 8, 6, 5],
                        [8, 5, 3, 0, 2, 1, 7, 6, 9, 4],
                        [6, 9, 5, 3, 0, 7, 4, 2, 1, 8],
                        [4, 1, 7, 6, 5, 0, 8, 9, 3, 2],
                        [5, 4, 2, 8, 9, 6, 0, 3, 7, 1],
                        [3, 6, 1, 7, 4, 8, 5, 0, 2, 9],
                        [1, 8, 4, 2, 6, 5, 9, 7, 0, 3],
                        [2, 3, 9, 5, 7, 4, 6, 1, 8, 0]
                    ],
                    [  # Latin Square 3
                        [0, 7, 8, 9, 1, 2, 3, 4, 5, 6],
                        [6, 4, 2, 8, 9, 5, 1, 3, 7, 0],
                        [4, 9, 5, 3, 2, 7, 6, 0, 1, 8],
                        [5, 1, 7, 6, 4, 3, 8, 9, 0, 2],
                        [3, 2, 9, 0, 7, 1, 5, 6, 8, 4],
                        [1, 0, 3, 7, 6, 8, 2, 5, 4, 9],
                        [2, 8, 0, 1, 3, 4, 9, 7, 6, 5],
                        [9, 5, 4, 2, 8, 6, 0, 1, 3, 7],
                        [7, 3, 6, 5, 0, 9, 4, 8, 2, 1],
                        [8, 6, 1, 4, 5, 0, 7, 2, 9, 3]
                    ]
                ]
            }
        ],
        allowed_imports=["random", "itertools"],

    )

    task_manager = TaskManagerAgent(
        task_definition=task
    )

    best_programs = await task_manager.execute()

    if best_programs:
        logger.info(f"Evolutionary process completed. Best program(s) found: {len(best_programs)}")
        for i, program in enumerate(best_programs):
            logger.info(f"Final Best Program {i+1} ID: {program.id}")
            logger.info(f"Final Best Program {i+1} Fitness: {program.fitness_scores}")
            logger.info(f"Final Best Program {i+1} Code:\n{program.code}")
    else:
        logger.info("Evolutionary process completed, but no suitable programs were found.")

    logger.info("OpenAlpha_Evolve run finished.")

if __name__ == "__main__":
    asyncio.run(main())
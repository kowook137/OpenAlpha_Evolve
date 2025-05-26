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
            "Implement a function `generate_mols()` → List[List[List[int]]]."
            "It should return 3 Latin squares of size 10×10, each as a 2D list of integers 0–9."
            "Each square must have no repeated numbers in any row or column."
            "Two squares A and B are orthogonal if all (A[i][j], B[i][j]) pairs are unique across all positions."
            "Return three Latin squares that are as mutually orthogonal as possible — i.e., all pairs (A,B), (A,C), and (B,C) should satisfy this condition."
            "Partial credit is given for:"
            "- Nearly valid Latin squares (few duplicate values in rows/columns),"
            "- Nearly orthogonal square pairs (few repeated pairs)."
            "Return format: a list of 3 elements, each a 10×10 list of integers from 0 to 9."

        ),
        function_name_to_evolve="generate_MOLS_10",
        input_output_examples = [
            {
                "input": [],
                "output": [
                    [   # Latin Square 1
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                        [2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
                        [3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
                        [4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
                        [5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                        [6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
                        [7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                        [8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
                        [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]
                    ],
                    [   # Latin Square 2
                        [0, 2, 4, 6, 8, 1, 3, 5, 7, 9],
                        [1, 3, 5, 7, 9, 2, 4, 6, 8, 0],
                        [2, 4, 6, 8, 0, 3, 5, 7, 9, 1],
                        [3, 5, 7, 9, 1, 4, 6, 8, 0, 2],
                        [4, 6, 8, 0, 2, 5, 7, 9, 1, 3],
                        [5, 7, 9, 1, 3, 6, 8, 0, 2, 4],
                        [6, 8, 0, 2, 4, 7, 9, 1, 3, 5],
                        [7, 9, 1, 3, 5, 8, 0, 2, 4, 6],
                        [8, 0, 2, 4, 6, 9, 1, 3, 5, 7],
                        [9, 1, 3, 5, 7, 0, 2, 4, 6, 8]
                    ],
                    [  # Latin Square 3
                        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                        [8, 7, 6, 5, 4, 3, 2, 1, 0, 9],
                        [7, 6, 5, 4, 3, 2, 1, 0, 9, 8],
                        [6, 5, 4, 3, 2, 1, 0, 9, 8, 7],
                        [5, 4, 3, 2, 1, 0, 9, 8, 7, 6],
                        [4, 3, 2, 1, 0, 9, 8, 7, 6, 5],
                        [3, 2, 1, 0, 9, 8, 7, 6, 5, 4],
                        [2, 1, 0, 9, 8, 7, 6, 5, 4, 3],
                        [1, 0, 9, 8, 7, 6, 5, 4, 3, 2],
                        [0, 9, 8, 7, 6, 5, 4, 3, 2, 1]
                    ]
                ]
            },
            {
                "input": [],
                "output": [
                      [  # square 1
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        [2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
                        [4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
                        [6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
                        [8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                        [3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
                        [5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                        [7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                        [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]
                    ],
                    [  # square 2
                        [0, 2, 4, 6, 8, 1, 3, 5, 7, 9],
                        [1, 3, 5, 7, 9, 2, 4, 6, 8, 0],
                        [2, 4, 6, 8, 0, 3, 5, 7, 9, 1],
                        [3, 5, 7, 9, 1, 4, 6, 8, 0, 2],
                        [4, 6, 8, 0, 2, 5, 7, 9, 1, 3],
                        [5, 7, 9, 1, 3, 6, 8, 0, 2, 4],
                        [6, 8, 0, 2, 4, 7, 9, 1, 3, 5],
                        [7, 9, 1, 3, 5, 8, 0, 2, 4, 6],
                        [8, 0, 2, 4, 6, 9, 1, 3, 5, 7],
                        [9, 1, 3, 5, 7, 0, 2, 4, 6, 8]
                    ],
                    [  # square 3 (random but Latin)
                        [0, 4, 8, 2, 6, 1, 5, 9, 3, 7],
                        [1, 5, 9, 3, 7, 2, 6, 0, 4, 8],
                        [2, 6, 0, 4, 8, 3, 7, 1, 5, 9],
                        [3, 7, 1, 5, 9, 4, 8 ,2 ,6 ,0],
                        [4, 8, 2, 6, 0, 5, 9, 3, 7, 1],
                        [5, 9, 3, 7, 1, 6, 0, 4, 8, 2],
                        [6, 0, 4, 8, 2, 7, 1, 5, 9, 3],
                        [7, 1, 5, 9, 3, 8, 2, 6, 0, 4],
                        [8, 2, 6, 0, 4, 9, 3, 7, 1, 5],
                        [9, 3, 7, 1, 5, 0, 4, 8, 2, 6]
                    ]
                ]
            },
            {
                "input":[],
                "output": [
                     [  # square 1
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0],
                        [2, 3, 4, 5, 6, 7, 8, 9, 0, 1],
                        [3, 4, 5, 6, 7, 8, 9, 0, 1, 2],
                        [4, 5, 6, 7, 8, 9, 0, 1, 2, 3],
                        [5, 6, 7, 8, 9, 0, 1, 2, 3, 4],
                        [6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
                        [7, 8, 9, 0, 1, 2, 3, 4, 5, 6],
                        [8, 9, 0, 1, 2, 3, 4, 5, 6, 7],
                        [9, 0, 1, 2, 3, 4, 5, 6, 7, 8]
                    ],
                    [  # square 2 (same as square1 rotated → very low orthogonality)
                        [0, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                        [1, 0, 9, 8, 7, 6, 5, 4, 3, 2],
                        [2, 1, 0, 9, 8, 7, 6, 5, 4, 3],
                        [3, 2, 1, 0, 9, 8, 7, 6, 5, 4],
                        [4, 3, 2, 1, 0, 9, 8, 7, 6, 5],
                        [5, 4, 3, 2, 1, 0, 9, 8, 7, 6],
                        [6, 5, 4, 3, 2, 1, 0, 9, 8, 7],
                        [7, 6, 5, 4, 3, 2, 1, 0, 9, 8],
                        [8, 7, 6, 5, 4, 3, 2, 1, 0, 9],
                        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
                    ],
                    [  # square 3 (partial noise but Latin)
                        [0, 1, 3, 2, 4, 5, 7, 6, 9, 8],
                        [1, 2, 4, 3, 5, 6, 8, 7, 0, 9],
                        [2, 3, 5, 4, 6, 7, 9, 8, 1, 0],
                        [3, 4, 6, 5, 7, 8, 0, 9, 2, 1],
                        [4, 5, 7, 6, 8, 9, 1, 0, 3, 2],
                        [5, 6, 8, 7, 9, 0, 2, 1, 4, 3],
                        [6, 7, 9, 8, 0, 1, 3, 2, 5, 4],
                        [7, 8, 0, 9, 1, 2, 4, 3, 6, 5],
                        [8, 9, 1, 0, 2, 3, 5, 4, 7, 6],
                        [9, 0, 2, 1, 3, 4, 6, 5, 8, 7]
                    ]
                ]

            }
        ],
        allowed_imports=["random", "itertools", "numpy"],

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
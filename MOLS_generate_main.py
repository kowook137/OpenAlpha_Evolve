"""
Main entry point for the AlphaEvolve Pro application with Island Model.
Orchestrates the different agents and manages the evolutionary loop using parallel islands.
"""
import asyncio
import logging
import sys
import os

                                               
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from task_manager.gpu_island_task_manager import GPUIslandTaskManager
from task_manager.island_task_manager import IslandTaskManager
from core.interfaces import TaskDefinition
from config import settings
from mols_task.evaluator_agent.agent import EvaluatorAgent

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
    logger.info("Starting OpenAlpha_Evolve with Island Model for MOLS generation")
    logger.info("=" * 80)
    
    # Log Island Model configuration
    island_config = settings.get_island_config()
    map_elites_config = settings.get_map_elites_config()
    parallel_config = settings.get_parallel_config()
    
    logger.info("Island Model Configuration:")
    logger.info(f"  Enabled: {island_config['enable_island_model']}")
    logger.info(f"  Number of Islands: {island_config['num_islands']}")
    logger.info(f"  Population per Island: {island_config['population_per_island']}")
    logger.info(f"  Migration Interval: {island_config['migration_interval']} generations")
    logger.info(f"  Migration Rate: {island_config['migration_rate']}")
    logger.info(f"  Migration Topology: {island_config['topology']}")
    logger.info(f"  Elite Migration: {island_config['elite_migration']}")
    
    logger.info("MAP-Elites Configuration:")
    logger.info(f"  Enabled: {map_elites_config['enable_map_elites']}")
    logger.info(f"  Config: {map_elites_config['config_name']}")
    logger.info(f"  Behavior Dimensions: {map_elites_config['behavior_dimensions']}")
    
    logger.info("Parallel Processing Configuration:")
    logger.info(f"  Max Parallel Islands: {parallel_config['max_parallel_islands']}")
    logger.info(f"  Evolution Timeout: {parallel_config['island_evolution_timeout']}s")
    logger.info(f"  Evaluation Batch Size: {parallel_config['parallel_evaluation_batch_size']}")
    
    logger.info("=" * 80)
    logger.info(f"Total Population Size: {island_config['num_islands'] * island_config['population_per_island']}")
    logger.info(f"Generations: {settings.GENERATIONS}")
    logger.info(f"LLM Models: Pro={settings.GEMINI_PRO_MODEL_NAME}, Flash={settings.GEMINI_FLASH_MODEL_NAME}, Eval={settings.GEMINI_EVALUATION_MODEL}")

    task = TaskDefinition(
        id="generate_10x10_MOLS_island_model",
        description=(
            "Implement a function `generate_MOLS_10()` → List[List[List[int]]]."
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

    # Initialize MOLS-specific evaluator
    evaluator = EvaluatorAgent()
    
    # GPU 가속 사용 여부 확인
    use_gpu = getattr(settings, 'ENABLE_GPU_ACCELERATION', False)
    
    if use_gpu:
        # GPU 설정 로깅
        gpu_config = settings.get_gpu_config()
        logger.info("GPU Acceleration Configuration:")
        for key, value in gpu_config.items():
            logger.info(f"  {key}: {value}")
        
        # GPU MAP-Elites 설정 로깅
        gpu_map_elites_config = settings.get_gpu_map_elites_config()
        logger.info("GPU MAP-Elites Configuration:")
        for key, value in gpu_map_elites_config.items():
            logger.info(f"  {key}: {value}")
        
        # GPU Task Manager 설정
        task_manager_config = {
            "max_generations": settings.GENERATIONS,
            "population_size": gpu_config["num_islands"] * gpu_config["population_per_island"],
            "num_islands": gpu_config["num_islands"],
            "gpu_batch_size": gpu_config["gpu_batch_size"],
            "use_mixed_precision": gpu_config["use_mixed_precision"],
            "enable_gpu_profiling": gpu_config["enable_profiling"],
            "island_manager": {
                "gpu_batch_size": gpu_config["gpu_batch_size"],
                "use_mixed_precision": gpu_config["use_mixed_precision"],
                "num_gpu_workers": gpu_config["num_gpu_workers"]
            },
            "migration_policy": {
                "migration_interval": gpu_config["migration_interval"],
                "base_migration_rate": gpu_config["migration_rate"],
                "topology": gpu_config["topology"],
                "batch_migration": True,
                "use_gpu_sorting": True
            },
            "map_elites": {
                "behavior_space": {
                    dim: {"bounds": (0.0, 1.0), "resolution": gpu_map_elites_config["dimension_bins"][dim]}
                    for dim in gpu_map_elites_config["behavior_dimensions"]
                },
                "gpu_batch_size": gpu_map_elites_config["gpu_batch_size"],
                "use_mixed_precision": gpu_config["use_mixed_precision"],
                "cache_behavior_vectors": gpu_map_elites_config["cache_behavior_vectors"]
            }
        }
        
        # GPU Island Task Manager 초기화
        task_manager = GPUIslandTaskManager(task_manager_config)
        logger.info("Initialized GPU-accelerated Island Task Manager")
        
    else:
        # CPU 기반 Island Task Manager 초기화 (기존 방식)
        task_manager = IslandTaskManager(
            task_definition=task,
            evaluator=evaluator
        )
        logger.info("Initialized CPU-based Island Task Manager")

    logger.info("Starting island-based evolution for MOLS generation...")
    
    if use_gpu:
        # GPU 버전은 다른 실행 방식 사용
        from core.program_generator import ProgramGeneratorAgent
        generator = ProgramGeneratorAgent()
        
        # 초기 프로그램 생성
        initial_programs = []
        for i in range(task_manager_config["population_size"]):
            program = await generator.generate_program(task)
            if program:
                initial_programs.append(program)
        
        logger.info(f"Generated {len(initial_programs)} initial programs for GPU evolution")
        
        # GPU 진화 실행
        evolution_results = await task_manager.run_evolution(task, initial_programs)
        best_programs = evolution_results.get("best_programs", [])
        
        # GPU 성능 통계 로깅
        gpu_performance = evolution_results.get("gpu_performance", {})
        logger.info("GPU Performance Statistics:")
        for key, value in gpu_performance.items():
            if isinstance(value, dict):
                logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    logger.info(f"    {sub_key}: {sub_value}")
            else:
                logger.info(f"  {key}: {value}")
        
    else:
        # CPU 버전 실행 (기존 방식)
        best_programs = await task_manager.execute()

    if best_programs:
        logger.info("=" * 80)
        logger.info(f"Island-based evolutionary process completed successfully!")
        logger.info(f"Best program(s) found: {len(best_programs)}")
        logger.info("=" * 80)
        
        for i, program in enumerate(best_programs):
            logger.info(f"Final Best Program {i+1} ID: {program.id}")
            logger.info(f"Final Best Program {i+1} Fitness: {program.fitness_scores}")
            logger.info(f"Final Best Program {i+1} Generation: {program.generation}")
            if program.parent_id:
                logger.info(f"Final Best Program {i+1} Parent: {program.parent_id}")
            logger.info(f"Final Best Program {i+1} Code:\n{program.code}")
            logger.info("-" * 60)
            
            # Execute and visualize the best program
            try:
                namespace = {}
                exec(program.code, namespace)
                squares = namespace['generate_MOLS_10']()
                
                logger.info(f"Executing Best Program {i+1}:")
                evaluator.print_matrix(squares[0], "Latin Square 1")
                evaluator.print_matrix(squares[1], "Latin Square 2")
                evaluator.print_matrix(squares[2], "Latin Square 3")
                
                print("\nOrthogonality Analysis:")
                print("-" * 40)
                total_duplicates = 0
                for sq1 in range(len(squares)):
                    for sq2 in range(sq1 + 1, len(squares)):
                        pairs = set()
                        duplicates = 0
                        for r in range(len(squares[sq1])):
                            for c in range(len(squares[sq1])):
                                pair = (squares[sq1][r][c], squares[sq2][r][c])
                                if pair in pairs:
                                    duplicates += 1
                                pairs.add(pair)
                        total_duplicates += duplicates
                        orthogonality_score = (100 - duplicates) / 100.0
                        print(f"Squares {sq1+1} and {sq2+1}: {duplicates} duplicate pairs (Orthogonality: {orthogonality_score:.2%})")
                
                overall_orthogonality = max(0, (300 - total_duplicates) / 300.0)
                print(f"Overall Orthogonality Score: {overall_orthogonality:.2%}")
                print("=" * 40)
                
            except Exception as e:
                logger.error(f"Error executing/visualizing program {i+1}: {str(e)}")
        
        # Log Island Model statistics if available
        if hasattr(task_manager, 'island_manager') and task_manager.island_manager:
            island_stats = task_manager.island_manager.get_island_statistics()
            logger.info("Island Model Final Statistics:")
            logger.info(f"  Total Islands: {island_stats['num_islands']}")
            logger.info(f"  Total Population: {island_stats['total_population']}")
            logger.info(f"  Migration Events: {island_stats['migration_events']}")
            
            for island_detail in island_stats['island_details']:
                logger.info(f"  {island_detail['id']}: Strategy={island_detail['strategy']}, "
                           f"Pop={island_detail['population_size']}, "
                           f"Best Fitness={island_detail['best_fitness']:.3f}")
        
        # Log MAP-Elites statistics if available
        if hasattr(task_manager, 'map_elites') and task_manager.map_elites:
            map_stats = task_manager.map_elites.get_archive_statistics()
            logger.info("MAP-Elites Final Statistics:")
            logger.info(f"  Archive Size: {map_stats['archive_size']}")
            logger.info(f"  Behavior Space Coverage: {map_stats['coverage']:.2%}")
            logger.info(f"  Archive Update Rate: {map_stats['update_rate']:.2%}")
            logger.info(f"  Total Evaluations: {map_stats['total_evaluations']}")
            
    else:
        logger.warning("Island-based evolutionary process completed, but no suitable programs were found.")
        logger.info("Consider adjusting island model parameters or increasing generations.")

    logger.info("=" * 80)
    logger.info("OpenAlpha_Evolve Island Model run finished.")
    logger.info("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
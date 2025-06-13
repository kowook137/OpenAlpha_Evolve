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

# CPU 기반 Task Manager는 항상 import
from task_manager.island_task_manager import IslandTaskManager
from core.interfaces import TaskDefinition, Program
from config import settings
from mols_task.evaluator_agent.agent import EvaluatorAgent

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.LOG_FILE, mode="a")        # Added logging configuration
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
        id="generate_4x4_MOLS_discovery",
        function_name_to_evolve="generate_MOLS_n",
        description=(
            "Discover an algorithm to generate two 4x4 mutually orthogonal Latin squares. "
            "A Latin square is a 4x4 grid where each row and column contains each symbol (0,1,2,3) exactly once. "
            "Two Latin squares are orthogonal if, when superimposed, each ordered pair occurs exactly once. "
            "Your function should return a list of two 4x4 nested lists with values [0,1,2,3]. "
            "Explore algorithmic approaches: constraint-based methods, systematic construction, or mathematical transformations."
        ),
        input_output_examples = [
            {
                "input": [],
                "output": [
                    "[[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]",
                    "[[0, 2, 1, 3], [2, 0, 3, 1], [1, 3, 2, 0], [3, 1, 0, 2]]"
                ],
                "explanation": "Two 4x4 Latin squares where each row and column contains {0,1,2,3} exactly once, and when overlaid, all 16 coordinate pairs are unique."
            }
        ],
        allowed_imports=["random", "itertools", "numpy"],
    )

    # Initialize MOLS-specific evaluator
    evaluator = EvaluatorAgent()
    
    # Check if GPU acceleration is enabled
    use_gpu = getattr(settings, 'ENABLE_GPU_ACCELERATION', False)
    
    if use_gpu:
        # Import GPU modules only when GPU is enabled (to avoid torch dependencies)
        try:
            from task_manager.gpu_island_task_manager import GPUIslandTaskManager
        except ImportError as e:
            logger.error(f"GPU acceleration enabled but required packages not installed: {e}")
            logger.info("Falling back to CPU mode...")
            use_gpu = False
    
    if use_gpu:
        # Log GPU configuration
        gpu_config = settings.get_gpu_config()
        logger.info("GPU Acceleration Configuration:")
        for key, value in gpu_config.items():
            logger.info(f"  {key}: {value}")
        
        # Log GPU MAP-Elites configuration
        gpu_map_elites_config = settings.get_gpu_map_elites_config()
        logger.info("GPU MAP-Elites Configuration:")
        for key, value in gpu_map_elites_config.items():
            logger.info(f"  {key}: {value}")
        
        # GPU Task Manager configuration
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
        
        # Initialize GPU Island Task Manager
        task_manager = GPUIslandTaskManager(task_manager_config)
        logger.info("Initialized GPU-accelerated Island Task Manager")
        
    else:
        # Initialize CPU-based Island Task Manager (existing method)
        task_manager = IslandTaskManager(
            task_definition=task,
            evaluator=evaluator
        )
        logger.info("Initialized CPU-based Island Task Manager")

    logger.info("Starting island-based evolution for MOLS generation...")
    
    if use_gpu:
        # GPU version uses different execution method
        from code_generator.agent import CodeGeneratorAgent
        from prompt_designer.agent import PromptDesignerAgent
        
        generator = CodeGeneratorAgent()
        prompt_designer = PromptDesignerAgent(task)
        
        # Generate initial programs
        initial_programs = []
        for i in range(task_manager_config["population_size"]):
            try:
                # Generate initial prompt
                initial_prompt = prompt_designer.design_initial_prompt()
                
                # Generate code
                generated_code = await generator.generate_code(initial_prompt)
                
                if generated_code:
                    program = Program(
                        id=f"initial_gpu_program_{i}",
                        code=generated_code,
                        generation=0
                    )
                    initial_programs.append(program)
            except Exception as e:
                logger.warning(f"Failed to generate initial program {i}: {e}")
        
        logger.info(f"Generated {len(initial_programs)} initial programs for GPU evolution")
        
        # Execute GPU evolution
        evolution_results = await task_manager.run_evolution(task, initial_programs)
        best_programs = evolution_results.get("best_programs", [])
        
        # Log GPU performance statistics
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
        # Execute CPU version (existing method)
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
                squares = namespace['generate_MOLS_n'](4)
                
                logger.info(f"Executing Best Program {i+1}:")
                evaluator.print_matrix(squares[0], "Latin Square 1")
                evaluator.print_matrix(squares[1], "Latin Square 2")
                
                print("\nOrthogonality Analysis:")
                print("-" * 40)
                total_duplicates = 0
                pairs = set()
                duplicates = 0
                for r in range(len(squares[0])):
                    for c in range(len(squares[0])):
                        pair = (squares[0][r][c], squares[1][r][c])
                        if pair in pairs:
                            duplicates += 1
                        pairs.add(pair)
                total_duplicates = duplicates
                orthogonality_score = (16 - duplicates) / 16.0  # 4x4 = 16 pairs total
                print(f"Squares 1 and 2: {duplicates} duplicate pairs (Orthogonality: {orthogonality_score:.2%})")
                overall_orthogonality = orthogonality_score
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
"""
Island-based Task Manager that integrates Island Model, MAP-Elites, and Migration Policy.
This is the enhanced version of the original TaskManager with parallel island evolution.
"""
import asyncio
import logging
import uuid
import time
from typing import List, Dict, Any, Optional

from core.interfaces import (
    TaskManagerInterface, TaskDefinition, Program, Island, EvolutionStrategy
)
from core.island_manager import IslandManager
from core.migration_policy import MigrationPolicy, MIGRATION_POLICIES
from core.map_elites import MAPElites, create_map_elites
from config import settings

# Import existing agents
from prompt_designer.agent import PromptDesignerAgent
from code_generator.agent import CodeGeneratorAgent
from evaluator_agent.agent import EvaluatorAgent
from database_agent.agent import InMemoryDatabaseAgent
from selection_controller.agent import SelectionControllerAgent

logger = logging.getLogger(__name__)

class IslandTaskManager(TaskManagerInterface):
    """
    Enhanced Task Manager with Island Model, MAP-Elites, and parallel processing.
    Manages multiple islands evolving in parallel with periodic migration.
    """
    
    def __init__(self, task_definition: TaskDefinition, config: Optional[Dict[str, Any]] = None, evaluator=None):
        super().__init__(config)
        self.task_definition = task_definition
        
        # Island Model configuration
        self.island_config = settings.get_island_config()
        self.enable_island_model = self.island_config["enable_island_model"]
        self.num_islands = self.island_config["num_islands"]
        self.population_per_island = self.island_config["population_per_island"]
        
        # Initialize core agents
        self.prompt_designer = PromptDesignerAgent(task_definition=self.task_definition)
        self.code_generator = CodeGeneratorAgent()
        self.evaluator = evaluator or EvaluatorAgent(task_definition=self.task_definition)
        self.database = InMemoryDatabaseAgent()
        self.selection_controller = SelectionControllerAgent()
        
        # Initialize Island Model components
        if self.enable_island_model:
            self.island_manager = IslandManager()
            
            # Migration Policy
            migration_config = MIGRATION_POLICIES.get("conservative", {})
            migration_config.update({
                "migration_interval": self.island_config["migration_interval"],
                "migration_rate": self.island_config["migration_rate"],
                "topology": self.island_config["topology"],
                "elite_migration": self.island_config["elite_migration"]
            })
            self.migration_policy = MigrationPolicy(migration_config)
            
            # MAP-Elites
            map_elites_config = settings.get_map_elites_config()
            if map_elites_config["enable_map_elites"]:
                self.map_elites = create_map_elites(map_elites_config["config_name"])
            else:
                self.map_elites = None
        else:
            self.island_manager = None
            self.migration_policy = None
            self.map_elites = None
        
        # Evolution parameters
        self.num_generations = settings.GENERATIONS
        self.total_population_size = self.num_islands * self.population_per_island if self.enable_island_model else settings.POPULATION_SIZE
        
        logger.info(f"IslandTaskManager initialized:")
        logger.info(f"  Island Model: {'Enabled' if self.enable_island_model else 'Disabled'}")
        logger.info(f"  Islands: {self.num_islands}, Population per island: {self.population_per_island}")
        logger.info(f"  Total population: {self.total_population_size}")
        logger.info(f"  MAP-Elites: {'Enabled' if self.map_elites else 'Disabled'}")
    
    async def initialize_population(self) -> List[Island]:
        """Initialize population across islands"""
        if not self.enable_island_model:
            # Fallback to single population
            return await self._initialize_single_population()
        
        logger.info(f"Initializing {self.num_islands} islands with {self.population_per_island} programs each")
        
        # Initialize islands
        islands = await self.island_manager.initialize_islands(self.num_islands, self.population_per_island)
        
        # Generate initial programs for each island
        for island in islands:
            logger.info(f"Generating initial population for {island.id}")
            
            for i in range(self.population_per_island):
                program_id = f"{self.task_definition.id}_{island.id}_gen0_prog{i}"
                
                # Generate initial code using EVOLVE-BLOCK template
                generated_code = await self.code_generator.generate_initial_code_with_template(
                    task_description=self.task_definition.description,
                    function_name=self.task_definition.function_name_to_evolve,
                    generation=0
                )
                
                program = Program(
                    id=program_id,
                    code=generated_code,
                    generation=0,
                    status="unevaluated"
                )
                
                island.population.append(program)
                await self.database.save_program(program)
            
            logger.debug(f"Initialized {island.id} with {len(island.population)} programs")
        
        return islands
    
    async def _initialize_single_population(self) -> List[Island]:
        """Fallback: Initialize single island for non-island mode"""
        island = Island(
            id="single_island",
            population=[],
            strategy=EvolutionStrategy.BALANCED,
            generation=0
        )
        
        for i in range(settings.POPULATION_SIZE):
            program_id = f"{self.task_definition.id}_single_gen0_prog{i}"
            generated_code = await self.code_generator.generate_initial_code_with_template(
                task_description=self.task_definition.description,
                function_name=self.task_definition.function_name_to_evolve,
                generation=0
            )
            
            program = Program(
                id=program_id,
                code=generated_code,
                generation=0,
                status="unevaluated"
            )
            
            island.population.append(program)
            await self.database.save_program(program)
        
        return [island]
    
    async def evaluate_islands(self, islands: List[Island]) -> List[Island]:
        """Evaluate all programs across all islands"""
        logger.info(f"Evaluating programs across {len(islands)} islands")
        
        # Collect all unevaluated programs
        all_programs = []
        for island in islands:
            unevaluated = [p for p in island.population if p.status != "evaluated"]
            all_programs.extend(unevaluated)
        
        if not all_programs:
            logger.info("No programs to evaluate")
            return islands
        
        logger.info(f"Evaluating {len(all_programs)} programs")
        
        # Parallel evaluation with batching
        batch_size = settings.PARALLEL_EVALUATION_BATCH_SIZE
        evaluated_programs = []
        
        for i in range(0, len(all_programs), batch_size):
            batch = all_programs[i:i + batch_size]
            
            # Create evaluation tasks
            eval_tasks = [
                self.evaluator.evaluate_program(program, self.task_definition)
                for program in batch
            ]
            
            # Execute batch
            batch_results = await asyncio.gather(*eval_tasks, return_exceptions=True)
            
            # Process results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Error evaluating program {batch[j].id}: {result}")
                    batch[j].status = "failed_evaluation"
                    batch[j].errors.append(str(result))
                    evaluated_programs.append(batch[j])
                else:
                    evaluated_programs.append(result)
                    
                    # Update MAP-Elites archive if enabled
                    if self.map_elites and result.status == "evaluated":
                        self.map_elites.update_archive(result)
        
        # Update programs in islands
        program_dict = {p.id: p for p in evaluated_programs}
        
        for island in islands:
            for i, program in enumerate(island.population):
                if program.id in program_dict:
                    island.population[i] = program_dict[program.id]
                    await self.database.save_program(program_dict[program.id])
        
        logger.info(f"Evaluation completed for {len(evaluated_programs)} programs")
        return islands
    
    async def evolve_islands_generation(self, islands: List[Island], generation: int) -> List[Island]:
        """Evolve all islands for one generation"""
        if not self.enable_island_model:
            # Fallback to single island evolution
            return await self._evolve_single_island_generation(islands, generation)
        
        logger.info(f"Evolving generation {generation} across {len(islands)} islands")
        
        # Parallel island evolution
        evolved_islands = await self.island_manager.evolve_islands_parallel(islands, self.task_definition)
        
        # Generate offspring for each island
        for island in evolved_islands:
            if not island.population:
                continue
            
            # Select parents
            num_parents = max(1, len(island.population) // 2)
            parents = self.selection_controller.select_parents(island.population, num_parents)
            
            if not parents:
                continue
            
            # Generate offspring
            offspring = []
            offspring_per_parent = max(1, self.population_per_island // len(parents))
            
            for parent in parents:
                for i in range(offspring_per_parent):
                    if len(offspring) >= self.population_per_island:
                        break
                    
                    child_id = f"{self.task_definition.id}_{island.id}_gen{generation}_child{len(offspring)}"
                    child = await self.generate_offspring(parent, generation, child_id)
                    
                    if child:
                        offspring.append(child)
                        await self.database.save_program(child)
            
            # Evaluate offspring
            if offspring:
                eval_tasks = [
                    self.evaluator.evaluate_program(child, self.task_definition)
                    for child in offspring
                ]
                
                evaluated_offspring = await asyncio.gather(*eval_tasks, return_exceptions=True)
                
                # Process evaluation results
                valid_offspring = []
                for result in evaluated_offspring:
                    if isinstance(result, Exception):
                        logger.error(f"Error evaluating offspring: {result}")
                    else:
                        valid_offspring.append(result)
                        if self.map_elites and result.status == "evaluated":
                            self.map_elites.update_archive(result)
                
                # Select survivors
                island.population = self.selection_controller.select_survivors(
                    island.population, valid_offspring, self.population_per_island
                )
            
            # Update island generation
            island.generation = generation
            
            # Update fitness history
            avg_fitness = island.get_average_fitness()
            island.fitness_history.append(avg_fitness)
        
        return evolved_islands
    
    async def _evolve_single_island_generation(self, islands: List[Island], generation: int) -> List[Island]:
        """Fallback: Evolve single island"""
        if not islands:
            return islands
        
        island = islands[0]
        
        # Standard evolution process
        parents = self.selection_controller.select_parents(island.population, len(island.population) // 2)
        
        offspring = []
        for parent in parents:
            child_id = f"{self.task_definition.id}_single_gen{generation}_child{len(offspring)}"
            child = await self.generate_offspring(parent, generation, child_id)
            if child:
                offspring.append(child)
        
        # Evaluate offspring
        if offspring:
            eval_tasks = [self.evaluator.evaluate_program(child, self.task_definition) for child in offspring]
            evaluated_offspring = await asyncio.gather(*eval_tasks, return_exceptions=True)
            
            valid_offspring = [r for r in evaluated_offspring if not isinstance(r, Exception)]
            
            # Select survivors
            island.population = self.selection_controller.select_survivors(
                island.population, valid_offspring, settings.POPULATION_SIZE
            )
        
        island.generation = generation
        return islands
    
    async def generate_offspring(self, parent: Program, generation: int, child_id: str) -> Optional[Program]:
        """Generate offspring from parent program"""
        try:
            # Determine prompt type based on parent's performance
            if parent.errors and parent.fitness_scores.get("score", 0.0) < 0.1:
                prompt_type = "bug_fix"
                error_message = parent.errors[0] if parent.errors else "Unknown error"
                prompt = self.prompt_designer.design_bug_fix_prompt(parent, error_message)
            else:
                prompt_type = "mutation"
                evaluation_feedback = parent.fitness_scores if hasattr(parent, 'fitness_scores') else None
                prompt = self.prompt_designer.design_mutation_prompt(parent, evaluation_feedback)
            
            # Generate code with generation info for model selection
            generated_code = await self.code_generator.generate_code(prompt, temperature=0.7, output_format="diff", generation=generation)
            
            child = Program(
                id=child_id,
                code=generated_code,
                generation=generation,
                parent_id=parent.id,
                status="unevaluated"
            )
            
            return child
            
        except Exception as e:
            logger.error(f"Error generating offspring from {parent.id}: {e}")
            return None
    
    async def perform_migration(self, islands: List[Island], generation: int) -> List[Island]:
        """Perform migration between islands if conditions are met"""
        if not self.enable_island_model or not self.migration_policy:
            return islands
        
        if not self.migration_policy.should_migrate(generation):
            return islands
        
        logger.info(f"Performing migration at generation {generation}")
        
        # Perform migration for each island
        for source_island in islands:
            if not source_island.population:
                continue
            
            # Select destination islands
            target_islands = self.migration_policy.select_destination_islands(source_island, islands)
            
            if target_islands:
                # Perform migration
                migrated_programs = self.migration_policy.perform_migration(
                    source_island, target_islands, generation
                )
                
                # Update MAP-Elites with migrated programs
                if self.map_elites:
                    for program in migrated_programs:
                        if program.status == "evaluated":
                            self.map_elites.update_archive(program)
        
        return islands
    
    async def manage_evolutionary_cycle(self):
        """Main evolutionary cycle with island model"""
        logger.info(f"Starting island-based evolutionary cycle for task: {self.task_definition.description[:50]}...")
        
        start_time = time.time()
        
        # Initialize islands
        islands = await self.initialize_population()
        
        # Initial evaluation
        islands = await self.evaluate_islands(islands)
        
        # Evolution loop
        for generation in range(1, self.num_generations + 1):
            logger.info(f"--- Generation {generation}/{self.num_generations} ---")
            
            # Evolve islands
            islands = await self.evolve_islands_generation(islands, generation)
            
            # Perform migration
            islands = await self.perform_migration(islands, generation)
            
            # Log statistics
            self._log_generation_statistics(islands, generation)
            
            # Check for early termination
            if self._should_terminate_early(islands):
                logger.info(f"Early termination at generation {generation}")
                break
        
        # Final evaluation and results
        total_time = time.time() - start_time
        logger.info(f"Evolution completed in {total_time:.2f} seconds")
        
        # Get best programs
        best_programs = self._get_best_programs_from_islands(islands)
        
        # Log final statistics
        self._log_final_statistics(islands, best_programs)
        
        return best_programs
    
    def _log_generation_statistics(self, islands: List[Island], generation: int):
        """Log statistics for current generation"""
        total_population = sum(len(island.population) for island in islands)
        
        # Calculate fitness statistics
        all_fitnesses = []
        for island in islands:
            for program in island.population:
                fitness = program.fitness_scores.get("score", program.fitness_scores.get("correctness", 0.0))
                all_fitnesses.append(fitness)
        
        if all_fitnesses:
            avg_fitness = sum(all_fitnesses) / len(all_fitnesses)
            max_fitness = max(all_fitnesses)
            
            logger.info(f"Generation {generation}: Population={total_population}, "
                       f"Avg Fitness={avg_fitness:.3f}, Max Fitness={max_fitness:.3f}")
        
        # Log island-specific statistics
        for island in islands:
            island_avg = island.get_average_fitness()
            best_program = island.get_best_program()
            best_fitness = best_program.fitness_scores.get("score", 0.0) if best_program else 0.0
            
            logger.debug(f"  {island.id}: Strategy={island.strategy.value}, "
                        f"Pop={len(island.population)}, Avg={island_avg:.3f}, Best={best_fitness:.3f}")
        
        # Log MAP-Elites statistics
        if self.map_elites:
            map_stats = self.map_elites.get_archive_statistics()
            logger.info(f"MAP-Elites: Archive size={map_stats['archive_size']}, "
                       f"Coverage={map_stats['coverage']:.3f}")
    
    def _should_terminate_early(self, islands: List[Island]) -> bool:
        """Check if evolution should terminate early"""
        # Check if any island has achieved perfect score
        for island in islands:
            best_program = island.get_best_program()
            if best_program:
                score = best_program.fitness_scores.get("score", 0.0)
                if score >= 0.99:  # Near perfect
                    return True
        
        return False
    
    def _get_best_programs_from_islands(self, islands: List[Island]) -> List[Program]:
        """Get best programs from all islands"""
        all_programs = []
        for island in islands:
            all_programs.extend(island.population)
        
        if not all_programs:
            return []
        
        # Sort by fitness and return top programs
        sorted_programs = sorted(
            all_programs,
            key=lambda p: (
                p.fitness_scores.get("score", p.fitness_scores.get("correctness", 0.0)),
                -p.fitness_scores.get("runtime_ms", float('inf'))
            ),
            reverse=True
        )
        
        return sorted_programs[:5]  # Return top 5
    
    def _log_final_statistics(self, islands: List[Island], best_programs: List[Program]):
        """Log final evolution statistics"""
        logger.info("=== Final Evolution Statistics ===")
        
        # Island statistics
        if self.island_manager:
            island_stats = self.island_manager.get_island_statistics()
            logger.info(f"Islands: {island_stats['num_islands']}")
            logger.info(f"Total population: {island_stats['total_population']}")
            logger.info(f"Migration events: {island_stats['migration_events']}")
        
        # MAP-Elites statistics
        if self.map_elites:
            map_stats = self.map_elites.get_archive_statistics()
            logger.info(f"MAP-Elites archive size: {map_stats['archive_size']}")
            logger.info(f"Behavior space coverage: {map_stats['coverage']:.3f}")
            logger.info(f"Archive update rate: {map_stats['update_rate']:.3f}")
        
        # Best programs
        if best_programs:
            logger.info(f"Best programs found: {len(best_programs)}")
            for i, program in enumerate(best_programs[:3]):
                logger.info(f"  #{i+1}: {program.id}, Fitness: {program.fitness_scores}")
        
    async def execute(self) -> List[Program]:
        """Execute the island-based evolutionary process"""
        return await self.manage_evolutionary_cycle() 
"""
GPU Island Task Manager - RTX-3080 최적화 병렬 진화 관리자
"""
import asyncio
import logging
import time
import torch
from typing import List, Dict, Any, Optional

from core.interfaces import TaskManagerInterface, TaskDefinition, Program, Island
from core.gpu_island_manager import GPUIslandManager
from core.gpu_migration_policy import GPUMigrationPolicy, MigrationTopology
from config import settings

logger = logging.getLogger(__name__)

class GPUIslandTaskManager(TaskManagerInterface):
    """GPU Island 기반 고성능 Task Manager"""
    
    def __init__(self, task_definition: TaskDefinition, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.task_definition = task_definition
        
        # GPU 설정
        gpu_config = settings.get_gpu_config()
        self.num_generations = gpu_config.get("num_generations", 50)
        self.num_islands = gpu_config.get("num_islands", 10)
        self.population_per_island = gpu_config.get("population_per_island", 30)
        
        # 컴포넌트 초기화
        self.island_manager = GPUIslandManager(task_definition)
        
        migration_config = {
            "migration_interval": 5,
            "migration_rate": 0.15,
            "topology": "ring",
            "elite_migration": True
        }
        self.migration_policy = GPUMigrationPolicy(migration_config)
        
        # 통계
        self.evolution_stats = {
            "generations_completed": 0,
            "total_evaluations": 0,
            "best_fitness_history": [],
            "migration_events": []
        }
        
        logger.info(f"GPU Task Manager initialized - Islands: {self.num_islands}, "
                   f"Population: {self.population_per_island}")

    async def run_evolution(self) -> List[Program]:
        """메인 진화 실행"""
        logger.info("Starting GPU Island evolution...")
        
        try:
            # 1. Island 초기화
            islands = await self.island_manager.initialize_islands(
                self.num_islands, self.population_per_island
            )
            
            # 2. 진화 루프
            for generation in range(1, self.num_generations + 1):
                logger.info(f"Generation {generation}/{self.num_generations}")
                
                # Island 병렬 진화
                islands = await self.island_manager.evolve_islands_parallel(
                    islands, self.task_definition
                )
                
                # Migration
                if self.migration_policy.should_migrate(generation):
                    islands = await self.island_manager.migrate_between_islands(
                        islands, generation
                    )
                    self.evolution_stats["migration_events"].append(generation)
                
                # 통계 수집
                await self._collect_stats(islands, generation)
                
                # GPU 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # 3. 최고 개체 반환
            best_programs = self.island_manager.get_best_programs_from_islands(
                islands, top_k=5
            )
            
            self.evolution_stats["generations_completed"] = generation
            logger.info(f"Evolution completed. Best fitness: "
                       f"{best_programs[0].fitness_scores if best_programs else 'None'}")
            
            return best_programs
            
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            return []
        finally:
            await self.island_manager.cleanup()

    async def _collect_stats(self, islands: List[Island], generation: int):
        """통계 수집"""
        all_fitness = []
        for island in islands:
            for program in island.population:
                if program.fitness_scores:
                    all_fitness.append(program.fitness_scores.get("correctness", 0.0))
        
        if all_fitness:
            avg_fitness = sum(all_fitness) / len(all_fitness)
            max_fitness = max(all_fitness)
            self.evolution_stats["best_fitness_history"].append({
                "generation": generation,
                "avg_fitness": avg_fitness,
                "max_fitness": max_fitness
            })
            
            logger.debug(f"Gen {generation}: avg={avg_fitness:.3f}, max={max_fitness:.3f}")

    def get_stats(self) -> Dict[str, Any]:
        """진화 통계 반환"""
        return {
            "evolution_stats": self.evolution_stats,
            "island_stats": self.island_manager.get_performance_stats(),
            "migration_stats": self.migration_policy.get_migration_statistics()
        }

    async def execute(self) -> List[Program]:
        """실행 메소드"""
        return await self.run_evolution()


class MOLSGPUTaskManager(GPUIslandTaskManager):
    """MOLS 특화 GPU Task Manager"""
    
    def __init__(self, task_definition: TaskDefinition, mols_size: int = 8, 
                 config: Optional[Dict[str, Any]] = None):
        super().__init__(task_definition, config)
        self.mols_size = mols_size
        self.target_pairs = (mols_size * (mols_size - 1)) // 2
        
        logger.info(f"MOLS GPU Manager - {mols_size}x{mols_size}, target pairs: {self.target_pairs}")

    async def _collect_stats(self, islands: List[Island], generation: int):
        """MOLS 특화 통계"""
        await super()._collect_stats(islands, generation)
        
        # MOLS 메트릭
        orthogonal_pairs = []
        for island in islands:
            for program in island.population:
                if program.fitness_scores and "orthogonal_pairs" in program.fitness_scores:
                    orthogonal_pairs.append(program.fitness_scores["orthogonal_pairs"])
        
        if orthogonal_pairs:
            max_pairs = max(orthogonal_pairs)
            avg_pairs = sum(orthogonal_pairs) / len(orthogonal_pairs)
            completeness = max_pairs / self.target_pairs
            
            logger.info(f"MOLS Gen {generation}: pairs={max_pairs}/{self.target_pairs} "
                       f"({completeness:.1%}), avg={avg_pairs:.1f}")


def create_gpu_task_manager(task_definition: TaskDefinition) -> GPUIslandTaskManager:
    """GPU Task Manager 팩토리"""
    return GPUIslandTaskManager(task_definition)

def create_mols_gpu_manager(task_definition: TaskDefinition, 
                           mols_size: int = 8) -> MOLSGPUTaskManager:
    """MOLS GPU Manager 팩토리"""
    return MOLSGPUTaskManager(task_definition, mols_size) 
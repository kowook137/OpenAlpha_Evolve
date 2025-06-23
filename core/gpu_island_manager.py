"""
GPU Island Manager - RTX-3080 최적화된 병렬 진화 시스템
AlphaEvolve의 Island Model을 GPU 가속으로 구현
"""
import torch
import torch.multiprocessing as mp
import asyncio
import logging
import time
import uuid
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from dataclasses import asdict
import gc

from core.interfaces import (
    IslandManagerInterface, Island, Program, TaskDefinition, 
    EvolutionStrategy, MigrationEvent, BaseAgent
)
from config import settings

# Import concrete agent implementations
from prompt_designer.agent import PromptDesignerAgent
from code_generator.agent import CodeGeneratorAgent
from evaluator_agent.agent import EvaluatorAgent
from database_agent.agent import InMemoryDatabaseAgent
from selection_controller.agent import SelectionControllerAgent

logger = logging.getLogger(__name__)

class GPUIslandManager(IslandManagerInterface):
    """
    RTX-3080 최적화 GPU Island Manager
    
    주요 기능:
    - 병렬 Island 진화 (10개 Island)
    - GPU 메모리 최적화
    - 비동기 Migration
    - 성능 모니터링
    """
    
    def __init__(self, task_definition: TaskDefinition, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.task_definition = task_definition
        
        # GPU 설정 확인
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            logger.warning("CUDA not available! Falling back to CPU mode.")
        else:
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # GPU 설정 로드
        gpu_config = settings.get_gpu_config()
        self.num_islands = gpu_config.get("num_islands", 10)
        self.population_per_island = gpu_config.get("population_per_island", 30)
        self.migration_interval = gpu_config.get("migration_interval", 5)
        self.migration_rate = gpu_config.get("migration_rate", 0.15)
        self.batch_size = gpu_config.get("gpu_batch_size", 64)
        self.num_workers = gpu_config.get("num_gpu_workers", 6)
        
        # 에이전트 초기화
        self.prompt_designer = PromptDesignerAgent(task_definition=self.task_definition)
        self.code_generator = CodeGeneratorAgent()
        self.evaluator = EvaluatorAgent(task_definition=self.task_definition)
        self.database = InMemoryDatabaseAgent()
        self.selection_controller = SelectionControllerAgent()
        
        # 성능 모니터링
        self.performance_stats = {
            "total_evaluations": 0,
            "gpu_memory_peak": 0,
            "migration_events": 0,
            "evolution_time": 0
        }
        
        # ThreadPoolExecutor for CPU-bound tasks
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        
        logger.info(f"GPUIslandManager initialized:")
        logger.info(f"  Islands: {self.num_islands}")
        logger.info(f"  Population per island: {self.population_per_island}")
        logger.info(f"  Total population: {self.num_islands * self.population_per_island}")
        logger.info(f"  GPU Workers: {self.num_workers}")

    async def initialize_islands(self, num_islands: int, population_per_island: int) -> List[Island]:
        """GPU 최적화된 Island 초기화"""
        logger.info(f"Initializing {num_islands} islands with {population_per_island} individuals each...")
        
        start_time = time.time()
        islands = []
        
        # 전략적 Island 배치
        strategies = [
            EvolutionStrategy.EXPLORATION,    # 2개 - 탐색 중심
            EvolutionStrategy.EXPLOITATION,   # 2개 - 활용 중심
            EvolutionStrategy.BALANCED,       # 4개 - 균형
            EvolutionStrategy.RANDOM          # 2개 - 무작위
        ]
        
        strategy_counts = [2, 2, 4, 2]
        strategy_idx = 0
        current_count = 0
        
        for i in range(num_islands):
            # 전략 할당
            if current_count >= strategy_counts[strategy_idx]:
                strategy_idx += 1
                current_count = 0
            strategy = strategies[strategy_idx]
            current_count += 1
            
            island = Island(
                id=f"island_{i:02d}_{strategy.value}",
                population=[],
                strategy=strategy,
                generation=0
            )
            islands.append(island)
        
        # 병렬 개체 생성
        init_tasks = []
        for island in islands:
            task = self._initialize_island_population(island, population_per_island)
            init_tasks.append(task)
        
        # GPU 메모리 최적화를 위한 배치 처리
        batch_size = min(self.num_workers, len(init_tasks))
        for i in range(0, len(init_tasks), batch_size):
            batch = init_tasks[i:i + batch_size]
            await asyncio.gather(*batch)
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        init_time = time.time() - start_time
        total_individuals = sum(len(island.population) for island in islands)
        
        logger.info(f"Island initialization completed in {init_time:.2f}s")
        logger.info(f"Created {total_individuals} individuals across {len(islands)} islands")
        
        return islands

    async def _initialize_island_population(self, island: Island, target_size: int):
        """단일 Island 개체군 초기화"""
        logger.debug(f"Initializing population for {island.id} with strategy {island.strategy.value}")
        
        # 전략별 온도 조정
        temp_map = {
            EvolutionStrategy.EXPLORATION: 1.2,
            EvolutionStrategy.EXPLOITATION: 0.6,
            EvolutionStrategy.BALANCED: 0.8,
            EvolutionStrategy.RANDOM: 1.5
        }
        temperature = temp_map.get(island.strategy, 0.8)
        
        population = []
        for i in range(target_size):
            program_id = f"{self.task_definition.id}_{island.id}_init_{i:03d}"
            
            try:
                # 초기 프롬프트 생성
                initial_prompt = self.prompt_designer.design_initial_prompt()
                
                # 전략별 코드 생성
                generated_code = await self.code_generator.generate_code(
                    initial_prompt, 
                    temperature=temperature
                )
                
                program = Program(
                    id=program_id,
                    code=generated_code,
                    generation=0,
                    status="unevaluated"
                )
                
                population.append(program)
                await self.database.save_program(program)
                
            except Exception as e:
                logger.error(f"Failed to create individual {program_id}: {e}")
        
        island.population = population
        logger.debug(f"Island {island.id} initialized with {len(population)} individuals")

    async def evolve_islands_parallel(self, islands: List[Island], task: TaskDefinition) -> List[Island]:
        """GPU 병렬 Island 진화"""
        logger.info(f"Starting parallel evolution of {len(islands)} islands...")
        
        start_time = time.time()
        
        # GPU 메모리 모니터링 시작
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        
        # 병렬 진화 태스크 생성
        evolution_tasks = []
        for island in islands:
            task_coro = self._evolve_single_island(island, task)
            evolution_tasks.append(task_coro)
        
        # GPU 메모리 제한을 고려한 배치 처리
        evolved_islands = []
        batch_size = min(self.num_workers, len(evolution_tasks))
        
        for i in range(0, len(evolution_tasks), batch_size):
            batch = evolution_tasks[i:i + batch_size]
            logger.debug(f"Processing island batch {i//batch_size + 1}/{(len(evolution_tasks) + batch_size - 1)//batch_size}")
            
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Island evolution failed: {result}")
                else:
                    evolved_islands.append(result)
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
        # 성능 통계 업데이트
        evolution_time = time.time() - start_time
        self.performance_stats["evolution_time"] += evolution_time
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            self.performance_stats["gpu_memory_peak"] = max(
                self.performance_stats["gpu_memory_peak"], 
                peak_memory
            )
            logger.info(f"GPU Peak Memory: {peak_memory:.1f} MB")
        
        logger.info(f"Parallel evolution completed in {evolution_time:.2f}s")
        return evolved_islands

    async def _evolve_single_island(self, island: Island, task: TaskDefinition) -> Island:
        """단일 Island 진화 단계"""
        logger.debug(f"Evolving island {island.id} (strategy: {island.strategy.value})")
        
        try:
            # 1. 개체군 평가
            island.population = await self._evaluate_population_batch(island.population, task)
            
            # 2. 부모 선택 (전략별 차별화)
            num_parents = self._get_parent_count_by_strategy(island.strategy, len(island.population))
            parents = self.selection_controller.select_parents(island.population, num_parents)
            
            if not parents:
                logger.warning(f"No parents selected for island {island.id}")
                return island
            
            # 3. 자손 생성
            offspring = await self._generate_offspring_batch(parents, island, task)
            
            # 4. 자손 평가
            offspring = await self._evaluate_population_batch(offspring, task)
            
            # 5. 생존자 선택
            island.population = self.selection_controller.select_survivors(
                island.population, offspring, self.population_per_island
            )
            
            # 6. Island 통계 업데이트
            island.generation += 1
            avg_fitness = island.get_average_fitness()
            island.fitness_history.append(avg_fitness)
            
            # 성능 통계 업데이트
            self.performance_stats["total_evaluations"] += len(offspring)
            
            logger.debug(f"Island {island.id} evolution complete. Avg fitness: {avg_fitness:.3f}")
            
        except Exception as e:
            logger.error(f"Error evolving island {island.id}: {e}")
        
        return island

    async def _evaluate_population_batch(self, population: List[Program], task: TaskDefinition) -> List[Program]:
        """GPU 최적화 배치 평가"""
        if not population:
            return population
        
        logger.debug(f"Batch evaluating {len(population)} programs")
        
        # 평가되지 않은 개체만 필터링
        unevaluated = [p for p in population if p.status != "evaluated"]
        if not unevaluated:
            return population
        
        # 배치 크기 조정 (GPU 메모리 고려)
        effective_batch_size = min(self.batch_size, len(unevaluated))
        
        evaluated = []
        for i in range(0, len(unevaluated), effective_batch_size):
            batch = unevaluated[i:i + effective_batch_size]
            
            # 병렬 평가
            eval_tasks = [self.evaluator.evaluate_program(prog, task) for prog in batch]
            batch_results = await asyncio.gather(*eval_tasks, return_exceptions=True)
            
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Evaluation failed for {batch[j].id}: {result}")
                    batch[j].status = "failed_evaluation"
                    batch[j].fitness_scores = {"correctness": 0.0, "score": 0.0}
                    evaluated.append(batch[j])
                else:
                    evaluated.append(result)
                    await self.database.save_program(result)
        
        # 평가된 개체와 기존 평가된 개체 합치기
        already_evaluated = [p for p in population if p.status == "evaluated"]
        return already_evaluated + evaluated

    async def _generate_offspring_batch(self, parents: List[Program], island: Island, task: TaskDefinition) -> List[Program]:
        """배치 자손 생성"""
        logger.debug(f"Generating offspring for island {island.id}")
        
        target_offspring = self.population_per_island - len(parents)
        offspring_per_parent = max(1, target_offspring // len(parents))
        
        generation_tasks = []
        offspring_count = 0
        
        for parent in parents:
            for i in range(offspring_per_parent):
                if offspring_count >= target_offspring:
                    break
                
                child_id = f"{task.id}_{island.id}_gen{island.generation + 1}_child{offspring_count:03d}"
                task_coro = self._generate_single_offspring(parent, island, child_id)
                generation_tasks.append(task_coro)
                offspring_count += 1
        
        # 병렬 자손 생성
        offspring_results = await asyncio.gather(*generation_tasks, return_exceptions=True)
        
        offspring = []
        for result in offspring_results:
            if isinstance(result, Exception):
                logger.error(f"Offspring generation failed: {result}")
            elif result:
                offspring.append(result)
                await self.database.save_program(result)
        
        logger.debug(f"Generated {len(offspring)} offspring for island {island.id}")
        return offspring

    async def _generate_single_offspring(self, parent: Program, island: Island, child_id: str) -> Optional[Program]:
        """단일 자손 생성"""
        try:
            # 전략별 변이 확률 조정
            mutation_temp_map = {
                EvolutionStrategy.EXPLORATION: 1.0,
                EvolutionStrategy.EXPLOITATION: 0.5,
                EvolutionStrategy.BALANCED: 0.75,
                EvolutionStrategy.RANDOM: 1.2
            }
            temperature = mutation_temp_map.get(island.strategy, 0.75)
            
            # 부모의 피드백을 기반으로 변이 프롬프트 생성
            feedback = {
                "errors": parent.errors,
                "correctness": parent.fitness_scores.get("correctness", 0.0),
                "runtime_ms": parent.fitness_scores.get("runtime_ms", 0.0),
                "strategy": island.strategy.value
            }
            feedback = {k: v for k, v in feedback.items() if v is not None}
            
            mutation_prompt = self.prompt_designer.design_mutation_prompt(
                program=parent, 
                evaluation_feedback=feedback
            )
            
            # 코드 생성
            generated_code = await self.code_generator.execute(
                prompt=mutation_prompt,
                temperature=temperature,
                output_format="diff",
                parent_code_for_diff=parent.code
            )
            
            if not generated_code or generated_code == parent.code:
                return None
            
            offspring = Program(
                id=child_id,
                code=generated_code,
                generation=island.generation + 1,
                parent_id=parent.id,
                status="unevaluated"
            )
            
            return offspring
            
        except Exception as e:
            logger.error(f"Failed to generate offspring from {parent.id}: {e}")
            return None

    def _get_parent_count_by_strategy(self, strategy: EvolutionStrategy, population_size: int) -> int:
        """전략별 부모 선택 수 결정"""
        strategy_ratios = {
            EvolutionStrategy.EXPLORATION: 0.6,    # 60% 선택 (높은 다양성)
            EvolutionStrategy.EXPLOITATION: 0.3,   # 30% 선택 (엘리트 중심)
            EvolutionStrategy.BALANCED: 0.5,       # 50% 선택 (균형)
            EvolutionStrategy.RANDOM: 0.7          # 70% 선택 (높은 변이)
        }
        ratio = strategy_ratios.get(strategy, 0.5)
        return max(1, int(population_size * ratio))

    async def migrate_between_islands(self, islands: List[Island], generation: int) -> List[Island]:
        """Island 간 Migration 수행"""
        if generation % self.migration_interval != 0:
            return islands
        
        logger.info(f"Performing migration at generation {generation}")
        start_time = time.time()
        
        migration_events = []
        
        # Ring topology migration
        for i, source_island in enumerate(islands):
            if len(source_island.population) < 2:
                continue
            
            # 다음 Island를 대상으로 설정 (Ring topology)
            target_idx = (i + 1) % len(islands)
            target_island = islands[target_idx]
            
            # Migration 수행
            migration_event = await self._perform_migration(source_island, target_island, generation)
            if migration_event:
                migration_events.append(migration_event)
        
        migration_time = time.time() - start_time
        self.performance_stats["migration_events"] += len(migration_events)
        
        logger.info(f"Migration completed in {migration_time:.2f}s. {len(migration_events)} migrations performed.")
        
        return islands

    async def _perform_migration(self, source: Island, target: Island, generation: int) -> Optional[MigrationEvent]:
        """두 Island 간 실제 Migration 수행"""
        try:
            # Migration할 개체 수 계산
            num_migrants = max(1, int(len(source.population) * self.migration_rate))
            
            # 최고 개체들 선택 (Elite migration)
            migrants = sorted(
                source.population, 
                key=lambda p: p.fitness_scores.get("score", 0.0), 
                reverse=True
            )[:num_migrants]
            
            if not migrants:
                return None
            
            # Migration 수행
            migrant_ids = []
            for migrant in migrants:
                # 새로운 ID로 복사
                new_id = f"{migrant.id}_migrated_to_{target.id}_gen{generation}"
                migrated_program = Program(
                    id=new_id,
                    code=migrant.code,
                    fitness_scores=migrant.fitness_scores.copy(),
                    generation=generation,
                    parent_id=migrant.id,
                    errors=migrant.errors.copy(),
                    status=migrant.status
                )
                
                target.population.append(migrated_program)
                migrant_ids.append(new_id)
                await self.database.save_program(migrated_program)
            
            # Target island 개체수 조정
            if len(target.population) > self.population_per_island:
                # 낮은 적합도 개체 제거
                target.population.sort(
                    key=lambda p: p.fitness_scores.get("score", 0.0), 
                    reverse=True
                )
                target.population = target.population[:self.population_per_island]
            
            # Migration 기록
            target.migration_history.append(f"gen{generation}_from_{source.id}")
            
            migration_event = MigrationEvent(
                source_island_id=source.id,
                target_island_id=target.id,
                migrant_program_ids=migrant_ids,
                generation=generation,
                migration_type="elite"
            )
            
            logger.debug(f"Migrated {len(migrant_ids)} individuals from {source.id} to {target.id}")
            return migration_event
            
        except Exception as e:
            logger.error(f"Migration failed from {source.id} to {target.id}: {e}")
            return None

    def get_performance_stats(self) -> Dict[str, Any]:
        """성능 통계 반환"""
        stats = self.performance_stats.copy()
        
        if torch.cuda.is_available():
            stats.update({
                "gpu_available": True,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**2,
                "gpu_memory_cached": torch.cuda.memory_reserved() / 1024**2
            })
        else:
            stats["gpu_available"] = False
        
        return stats

    def get_best_programs_from_islands(self, islands: List[Island], top_k: int = 5) -> List[Program]:
        """모든 Island에서 최고 개체들 추출"""
        all_programs = []
        for island in islands:
            all_programs.extend(island.population)
        
        # 적합도 기준 정렬
        sorted_programs = sorted(
            all_programs,
            key=lambda p: (
                p.fitness_scores.get("correctness", 0.0),
                -p.fitness_scores.get("runtime_ms", float('inf'))
            ),
            reverse=True
        )
        
        return sorted_programs[:top_k]

    async def cleanup(self):
        """리소스 정리"""
        logger.info("Cleaning up GPU Island Manager resources...")
        
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 캐시 정리
        if hasattr(self.evaluator, '_cache'):
            self.evaluator._cache.clear()
        
        logger.info("GPU Island Manager cleanup completed")

    async def execute(self, action: str, **kwargs) -> Any:
        """메인 실행 메소드"""
        if action == "initialize":
            return await self.initialize_islands(
                kwargs.get("num_islands", self.num_islands),
                kwargs.get("population_per_island", self.population_per_island)
            )
        elif action == "evolve":
            return await self.evolve_islands_parallel(
                kwargs.get("islands"),
                kwargs.get("task", self.task_definition)
            )
        elif action == "migrate":
            return await self.migrate_between_islands(
                kwargs.get("islands"),
                kwargs.get("generation", 0)
            )
        elif action == "cleanup":
            return await self.cleanup()
        else:
            raise ValueError(f"Unknown action: {action}")


def create_gpu_island_manager(task_definition: TaskDefinition) -> GPUIslandManager:
    """GPU Island Manager 팩토리 함수"""
    return GPUIslandManager(task_definition)


# GPU 가속 설정 확인 함수
def check_gpu_setup():
    """GPU 설정 상태 확인"""
    print("=== GPU Setup Check ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"GPU compute capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("CUDA not available - will use CPU fallback")
    
    print("=======================")


if __name__ == "__main__":
    # GPU 설정 테스트
    check_gpu_setup() 
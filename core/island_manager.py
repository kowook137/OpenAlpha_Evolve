"""
Island Manager for managing multiple islands in parallel evolution.
Implements the core island model functionality with CPU-based parallelism.
"""
import asyncio
import logging
import uuid
import random
from typing import List, Dict, Any, Optional
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing

from core.interfaces import (
    IslandManagerInterface, Island, Program, TaskDefinition, 
    EvolutionStrategy, MigrationEvent
)
from config import settings

logger = logging.getLogger(__name__)

class IslandManager(IslandManagerInterface):
    """Manages multiple islands for parallel evolution"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.islands: List[Island] = []
        self.migration_history: List[MigrationEvent] = []
        self.max_workers = min(multiprocessing.cpu_count(), 8)  # CPU 코어 수에 맞춰 조정
        logger.info(f"IslandManager initialized with max_workers: {self.max_workers}")
        
    async def initialize_islands(self, num_islands: int, population_per_island: int) -> List[Island]:
        """Initialize islands with different evolution strategies"""
        logger.info(f"Initializing {num_islands} islands with {population_per_island} population each")
        
        self.islands = []
        strategies = list(EvolutionStrategy)
        
        for i in range(num_islands):
            # 각 섬에 다른 전략 할당 (순환)
            strategy = strategies[i % len(strategies)]
            
            island = Island(
                id=f"island_{i}",
                population=[],
                strategy=strategy,
                generation=0
            )
            self.islands.append(island)
            logger.debug(f"Created {island.id} with strategy {strategy.value}")
            
        return self.islands
    
    async def evolve_islands_parallel(self, islands: List[Island], task: TaskDefinition) -> List[Island]:
        """Evolve all islands in parallel using CPU cores"""
        logger.info(f"Starting parallel evolution of {len(islands)} islands")
        
        # ThreadPoolExecutor를 사용 (I/O bound 작업이 많음)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 각 섬의 진화를 병렬로 실행
            evolution_tasks = []
            for island in islands:
                task_future = executor.submit(self._evolve_single_island_sync, island, task)
                evolution_tasks.append(task_future)
            
            # 모든 섬의 진화 완료 대기
            evolved_islands = []
            for i, future in enumerate(evolution_tasks):
                try:
                    evolved_island = future.result(timeout=300)  # 5분 타임아웃
                    evolved_islands.append(evolved_island)
                    logger.debug(f"Island {islands[i].id} evolution completed")
                except Exception as e:
                    logger.error(f"Error evolving island {islands[i].id}: {e}")
                    # 실패한 경우 원본 섬 반환
                    evolved_islands.append(islands[i])
        
        logger.info(f"Parallel evolution completed for {len(evolved_islands)} islands")
        return evolved_islands
    
    def _evolve_single_island_sync(self, island: Island, task: TaskDefinition) -> Island:
        """단일 섬의 동기 진화 (ThreadPoolExecutor에서 실행)"""
        # 이 메서드는 동기적으로 실행되므로 asyncio를 사용하지 않음
        try:
            # 섬의 전략에 따라 다른 진화 방식 적용
            if island.strategy == EvolutionStrategy.EXPLOITATION:
                return self._evolve_exploitation_strategy(island, task)
            elif island.strategy == EvolutionStrategy.EXPLORATION:
                return self._evolve_exploration_strategy(island, task)
            elif island.strategy == EvolutionStrategy.RANDOM:
                return self._evolve_random_strategy(island, task)
            else:  # BALANCED
                return self._evolve_balanced_strategy(island, task)
                
        except Exception as e:
            logger.error(f"Error in island {island.id} evolution: {e}")
            return island
    
    def _evolve_exploitation_strategy(self, island: Island, task: TaskDefinition) -> Island:
        """착취 중심 전략: 최고 성능 개체들을 중심으로 진화"""
        logger.debug(f"Applying exploitation strategy to {island.id}")
        
        if not island.population:
            return island
            
        # 상위 50% 개체들만 선택하여 진화
        sorted_pop = sorted(island.population, 
                           key=lambda p: p.fitness_scores.get("score", 0.0), 
                           reverse=True)
        
        elite_count = max(1, len(sorted_pop) // 2)
        island.population = sorted_pop[:elite_count]
        
        # 평균 피트니스 기록
        avg_fitness = island.get_average_fitness()
        island.fitness_history.append(avg_fitness)
        
        island.generation += 1
        return island
    
    def _evolve_exploration_strategy(self, island: Island, task: TaskDefinition) -> Island:
        """탐색 중심 전략: 다양성을 중시하여 진화"""
        logger.debug(f"Applying exploration strategy to {island.id}")
        
        if not island.population:
            return island
            
        # 다양성을 위해 랜덤하게 일부 개체 교체
        if len(island.population) > 2:
            # 하위 30% 개체를 랜덤하게 교체
            sorted_pop = sorted(island.population, 
                               key=lambda p: p.fitness_scores.get("score", 0.0))
            
            replace_count = max(1, len(sorted_pop) // 3)
            # 상위 개체들은 유지하고 하위 개체들만 변경
            island.population = sorted_pop[replace_count:]
        
        avg_fitness = island.get_average_fitness()
        island.fitness_history.append(avg_fitness)
        
        island.generation += 1
        return island
    
    def _evolve_random_strategy(self, island: Island, task: TaskDefinition) -> Island:
        """랜덤 전략: 무작위 변이 중심"""
        logger.debug(f"Applying random strategy to {island.id}")
        
        if island.population:
            # 랜덤하게 개체들 섞기
            random.shuffle(island.population)
            
            avg_fitness = island.get_average_fitness()
            island.fitness_history.append(avg_fitness)
        
        island.generation += 1
        return island
    
    def _evolve_balanced_strategy(self, island: Island, task: TaskDefinition) -> Island:
        """균형 전략: 착취와 탐색의 균형"""
        logger.debug(f"Applying balanced strategy to {island.id}")
        
        if not island.population:
            return island
            
        # 상위 70% 유지, 하위 30% 교체
        sorted_pop = sorted(island.population, 
                           key=lambda p: p.fitness_scores.get("score", 0.0), 
                           reverse=True)
        
        keep_count = max(1, int(len(sorted_pop) * 0.7))
        island.population = sorted_pop[:keep_count]
        
        avg_fitness = island.get_average_fitness()
        island.fitness_history.append(avg_fitness)
        
        island.generation += 1
        return island
    
    async def migrate_between_islands(self, islands: List[Island], generation: int) -> List[Island]:
        """섬 간 개체 이주 수행"""
        if len(islands) < 2:
            return islands
            
        logger.info(f"Starting migration between {len(islands)} islands at generation {generation}")
        
        migration_events = []
        
        # 각 섬에서 최고 개체 1-2개를 다른 섬으로 이주
        for i, source_island in enumerate(islands):
            if not source_island.population:
                continue
                
            # 이주할 개체 선택 (최고 성능 개체)
            best_programs = sorted(source_island.population, 
                                 key=lambda p: p.fitness_scores.get("score", 0.0), 
                                 reverse=True)[:2]
            
            if not best_programs:
                continue
                
            # 목적지 섬 선택 (링 토폴로지: 다음 섬으로 이주)
            target_index = (i + 1) % len(islands)
            target_island = islands[target_index]
            
            # 이주 수행
            migrant_ids = []
            for program in best_programs[:1]:  # 1개만 이주
                # 프로그램 복사 생성
                migrant = Program(
                    id=f"{program.id}_migrant_gen{generation}",
                    code=program.code,
                    fitness_scores=program.fitness_scores.copy(),
                    generation=generation,
                    parent_id=program.id,
                    status=program.status
                )
                
                target_island.population.append(migrant)
                migrant_ids.append(migrant.id)
                
                # 이주 기록
                target_island.migration_history.append(source_island.id)
            
            if migrant_ids:
                migration_event = MigrationEvent(
                    source_island_id=source_island.id,
                    target_island_id=target_island.id,
                    migrant_program_ids=migrant_ids,
                    generation=generation,
                    migration_type="elite"
                )
                migration_events.append(migration_event)
        
        self.migration_history.extend(migration_events)
        logger.info(f"Migration completed: {len(migration_events)} migration events")
        
        return islands
    
    def get_island_statistics(self) -> Dict[str, Any]:
        """섬들의 통계 정보 반환"""
        stats = {
            "num_islands": len(self.islands),
            "total_population": sum(len(island.population) for island in self.islands),
            "migration_events": len(self.migration_history),
            "island_details": []
        }
        
        for island in self.islands:
            island_stats = {
                "id": island.id,
                "strategy": island.strategy.value,
                "population_size": len(island.population),
                "generation": island.generation,
                "average_fitness": island.get_average_fitness(),
                "best_fitness": island.get_best_program().fitness_scores.get("score", 0.0) if island.get_best_program() else 0.0,
                "migration_count": len(island.migration_history)
            }
            stats["island_details"].append(island_stats)
        
        return stats
    
    async def execute(self, action: str, **kwargs) -> Any:
        """Execute specific island management actions"""
        if action == "initialize":
            return await self.initialize_islands(kwargs["num_islands"], kwargs["population_per_island"])
        elif action == "evolve_parallel":
            return await self.evolve_islands_parallel(kwargs["islands"], kwargs["task"])
        elif action == "migrate":
            return await self.migrate_between_islands(kwargs["islands"], kwargs["generation"])
        elif action == "get_stats":
            return self.get_island_statistics()
        else:
            raise ValueError(f"Unknown action: {action}") 
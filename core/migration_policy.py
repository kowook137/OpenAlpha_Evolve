"""
Migration Policy implementations for island model.
Defines when and how individuals migrate between islands.
"""
import logging
import random
from typing import List, Dict, Any, Optional
from enum import Enum

from core.interfaces import MigrationPolicyInterface, Island, Program
from config import settings

logger = logging.getLogger(__name__)

class MigrationTopology(Enum):
    """Different migration topologies"""
    RING = "ring"              # Each island connects to next island in ring
    FULLY_CONNECTED = "fully_connected"  # All islands connected to all
    STAR = "star"              # Central hub island
    RANDOM = "random"          # Random connections each time

class MigrationPolicy(MigrationPolicyInterface):
    """Implements various migration policies for island model"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.migration_interval = self.config.get("migration_interval", 5)  # Every 5 generations
        self.migration_rate = self.config.get("migration_rate", 0.1)  # 10% of population
        self.topology = MigrationTopology(self.config.get("topology", "ring"))
        self.elite_migration = self.config.get("elite_migration", True)  # Migrate best individuals
        
        logger.info(f"MigrationPolicy initialized: interval={self.migration_interval}, "
                   f"rate={self.migration_rate}, topology={self.topology.value}")
    
    def should_migrate(self, generation: int) -> bool:
        """Determine if migration should occur at this generation"""
        return generation > 0 and generation % self.migration_interval == 0
    
    def select_migrants(self, source_island: Island, num_migrants: int) -> List[Program]:
        """Select individuals to migrate from source island"""
        if not source_island.population or num_migrants <= 0:
            return []
        
        # 실제 이주할 개체 수 계산
        actual_migrants = min(num_migrants, len(source_island.population))
        
        if self.elite_migration:
            # 엘리트 이주: 최고 성능 개체들 선택
            sorted_population = sorted(
                source_island.population,
                key=lambda p: p.fitness_scores.get("score", p.fitness_scores.get("correctness", 0.0)),
                reverse=True
            )
            migrants = sorted_population[:actual_migrants]
            logger.debug(f"Selected {len(migrants)} elite migrants from {source_island.id}")
        else:
            # 랜덤 이주: 무작위 개체들 선택
            migrants = random.sample(source_island.population, actual_migrants)
            logger.debug(f"Selected {len(migrants)} random migrants from {source_island.id}")
        
        return migrants
    
    def select_destination_islands(self, source_island: Island, all_islands: List[Island]) -> List[Island]:
        """Select destination islands based on topology"""
        if len(all_islands) <= 1:
            return []
        
        # 자기 자신 제외
        other_islands = [island for island in all_islands if island.id != source_island.id]
        
        if self.topology == MigrationTopology.RING:
            return self._select_ring_destinations(source_island, all_islands)
        elif self.topology == MigrationTopology.FULLY_CONNECTED:
            return other_islands  # 모든 다른 섬들
        elif self.topology == MigrationTopology.STAR:
            return self._select_star_destinations(source_island, all_islands)
        elif self.topology == MigrationTopology.RANDOM:
            # 랜덤하게 1-2개 섬 선택
            num_destinations = min(random.randint(1, 2), len(other_islands))
            return random.sample(other_islands, num_destinations)
        else:
            return other_islands[:1]  # 기본값: 첫 번째 다른 섬
    
    def _select_ring_destinations(self, source_island: Island, all_islands: List[Island]) -> List[Island]:
        """Ring topology: connect to next island in the ring"""
        try:
            source_index = next(i for i, island in enumerate(all_islands) if island.id == source_island.id)
            next_index = (source_index + 1) % len(all_islands)
            return [all_islands[next_index]]
        except StopIteration:
            logger.warning(f"Source island {source_island.id} not found in all_islands")
            return []
    
    def _select_star_destinations(self, source_island: Island, all_islands: List[Island]) -> List[Island]:
        """Star topology: central hub island (island_0) connects to all, others connect to hub"""
        hub_island = None
        for island in all_islands:
            if island.id == "island_0":  # 첫 번째 섬을 허브로 사용
                hub_island = island
                break
        
        if not hub_island:
            # 허브가 없으면 첫 번째 섬을 허브로 사용
            hub_island = all_islands[0]
        
        if source_island.id == hub_island.id:
            # 허브에서는 모든 다른 섬으로 이주
            return [island for island in all_islands if island.id != hub_island.id]
        else:
            # 다른 섬에서는 허브로만 이주
            return [hub_island]
    
    def calculate_migration_pressure(self, source_island: Island, target_island: Island) -> float:
        """Calculate migration pressure between two islands"""
        if not source_island.population or not target_island.population:
            return 0.0
        
        source_fitness = source_island.get_average_fitness()
        target_fitness = target_island.get_average_fitness()
        
        # 피트니스 차이가 클수록 이주 압력이 높음
        fitness_difference = max(0.0, source_fitness - target_fitness)
        
        # 인구 밀도 차이도 고려
        source_density = len(source_island.population)
        target_density = len(target_island.population)
        density_factor = source_density / (target_density + 1)  # +1 to avoid division by zero
        
        migration_pressure = fitness_difference * density_factor
        return migration_pressure
    
    def adaptive_migration_rate(self, source_island: Island, generation: int) -> float:
        """Calculate adaptive migration rate based on island performance"""
        base_rate = self.migration_rate
        
        # 섬의 성능 정체 시 이주율 증가
        if len(source_island.fitness_history) >= 3:
            recent_fitness = source_island.fitness_history[-3:]
            fitness_variance = max(recent_fitness) - min(recent_fitness)
            
            if fitness_variance < 0.01:  # 성능 정체
                adaptive_rate = min(base_rate * 2.0, 0.3)  # 최대 30%까지
                logger.debug(f"Increased migration rate for stagnant island {source_island.id}: {adaptive_rate}")
                return adaptive_rate
        
        return base_rate
    
    def perform_migration(self, source_island: Island, target_islands: List[Island], 
                         generation: int) -> List[Program]:
        """Perform actual migration between islands"""
        if not target_islands:
            return []
        
        # 적응적 이주율 계산
        migration_rate = self.adaptive_migration_rate(source_island, generation)
        num_migrants = max(1, int(len(source_island.population) * migration_rate))
        
        # 이주할 개체 선택
        migrants = self.select_migrants(source_island, num_migrants)
        if not migrants:
            return []
        
        # 각 목적지 섬에 이주자 분배
        migrants_per_island = len(migrants) // len(target_islands)
        remaining_migrants = len(migrants) % len(target_islands)
        
        migrated_programs = []
        migrant_index = 0
        
        for i, target_island in enumerate(target_islands):
            # 이 섬에 보낼 이주자 수 계산
            island_migrants = migrants_per_island
            if i < remaining_migrants:
                island_migrants += 1
            
            # 이주자들을 목적지 섬에 추가
            for j in range(island_migrants):
                if migrant_index < len(migrants):
                    original_migrant = migrants[migrant_index]
                    
                    # 이주자 복사본 생성
                    migrant_copy = Program(
                        id=f"{original_migrant.id}_migrant_to_{target_island.id}_gen{generation}",
                        code=original_migrant.code,
                        fitness_scores=original_migrant.fitness_scores.copy(),
                        generation=generation,
                        parent_id=original_migrant.id,
                        status=original_migrant.status
                    )
                    
                    target_island.population.append(migrant_copy)
                    migrated_programs.append(migrant_copy)
                    
                    # 이주 기록
                    target_island.migration_history.append(source_island.id)
                    
                    migrant_index += 1
        
        logger.info(f"Migrated {len(migrated_programs)} individuals from {source_island.id} "
                   f"to {len(target_islands)} target islands")
        
        return migrated_programs
    
    async def execute(self, action: str, **kwargs) -> Any:
        """Execute migration policy actions"""
        if action == "should_migrate":
            return self.should_migrate(kwargs["generation"])
        elif action == "select_migrants":
            return self.select_migrants(kwargs["source_island"], kwargs["num_migrants"])
        elif action == "select_destinations":
            return self.select_destination_islands(kwargs["source_island"], kwargs["all_islands"])
        elif action == "perform_migration":
            return self.perform_migration(
                kwargs["source_island"], 
                kwargs["target_islands"], 
                kwargs["generation"]
            )
        else:
            raise ValueError(f"Unknown action: {action}")

# 사전 정의된 Migration Policy 설정들
MIGRATION_POLICIES = {
    "conservative": {
        "migration_interval": 10,
        "migration_rate": 0.05,
        "topology": "ring",
        "elite_migration": True
    },
    "aggressive": {
        "migration_interval": 3,
        "migration_rate": 0.2,
        "topology": "fully_connected",
        "elite_migration": True
    },
    "exploratory": {
        "migration_interval": 5,
        "migration_rate": 0.15,
        "topology": "random",
        "elite_migration": False
    },
    "hub_based": {
        "migration_interval": 7,
        "migration_rate": 0.1,
        "topology": "star",
        "elite_migration": True
    }
} 
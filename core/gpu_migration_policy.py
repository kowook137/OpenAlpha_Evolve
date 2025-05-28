"""
GPU-accelerated Migration Policy using PyTorch for RTX 3080.
Implements high-performance migration strategies using CUDA.
"""
import torch
import torch.nn.functional as F
import logging
import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from core.interfaces import (
    MigrationPolicyInterface, Island, Program, MigrationEvent,
    MigrationTopology
)

logger = logging.getLogger(__name__)

class GPUMigrationPolicy(MigrationPolicyInterface):
    """GPU-accelerated Migration Policy for RTX 3080"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # GPU 설정
        self.device = self._setup_gpu()
        
        # 기본 이주 설정
        self.migration_interval = self.config.get("migration_interval", 5)
        self.base_migration_rate = self.config.get("migration_rate", 0.1)
        self.max_migration_rate = self.config.get("max_migration_rate", 0.3)
        self.diversity_threshold = self.config.get("diversity_threshold", 0.1)
        
        # 토폴로지 설정
        topology_str = self.config.get("topology", "ring")
        if isinstance(topology_str, str):
            self.topology = MigrationTopology(topology_str)
        else:
            self.topology = topology_str
        
        # GPU 최적화 설정
        self.batch_migration = self.config.get("batch_migration", True)
        self.use_gpu_sorting = self.config.get("use_gpu_sorting", True)
        self.migration_tensor_cache = {}
        
        logger.info(f"GPUMigrationPolicy initialized on {self.device}")
        logger.info(f"  Topology: {self.topology.value}")
        logger.info(f"  Migration interval: {self.migration_interval}")
        logger.info(f"  Migration rate: {self.base_migration_rate}")
    
    def _setup_gpu(self) -> torch.device:
        """GPU 설정"""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.empty_cache()
            return device
        else:
            logger.warning("CUDA not available, using CPU")
            return torch.device("cpu")
    
    async def should_migrate(self, islands: List[Island], generation: int) -> bool:
        """GPU 가속 이주 조건 확인"""
        if len(islands) < 2 or generation < self.migration_interval:
            return False
        
        # GPU에서 병렬로 다양성 계산
        diversity_scores = await self._calculate_gpu_diversity(islands)
        
        # 이주 조건: 다양성이 임계값 이하이거나 주기적 이주
        low_diversity = any(score < self.diversity_threshold for score in diversity_scores)
        periodic_migration = generation % self.migration_interval == 0
        
        should_migrate = low_diversity or periodic_migration
        
        if should_migrate:
            logger.info(f"Migration triggered at generation {generation}")
            logger.info(f"  Diversity scores: {diversity_scores}")
            logger.info(f"  Low diversity: {low_diversity}, Periodic: {periodic_migration}")
        
        return should_migrate
    
    async def _calculate_gpu_diversity(self, islands: List[Island]) -> List[float]:
        """GPU에서 병렬로 island 다양성 계산"""
        diversity_scores = []
        
        with torch.cuda.device(self.device):
            for island in islands:
                if len(island.population) < 2:
                    diversity_scores.append(0.0)
                    continue
                
                # 피트니스 점수를 GPU 텐서로 변환
                fitness_scores = []
                for program in island.population:
                    score = program.fitness_scores.get("score", program.fitness_scores.get("correctness", 0.0))
                    fitness_scores.append(score)
                
                if not fitness_scores:
                    diversity_scores.append(0.0)
                    continue
                
                # GPU에서 다양성 계산 (표준편차 사용)
                fitness_tensor = torch.tensor(fitness_scores, device=self.device, dtype=torch.float32)
                diversity = torch.std(fitness_tensor).item()
                diversity_scores.append(diversity)
        
        return diversity_scores
    
    async def select_migrants(self, islands: List[Island], generation: int) -> List[MigrationEvent]:
        """GPU 가속 이주자 선택"""
        if len(islands) < 2:
            return []
        
        logger.info(f"Selecting migrants using GPU acceleration for {len(islands)} islands")
        
        migration_events = []
        
        with torch.cuda.device(self.device):
            # 토폴로지별 GPU 가속 이주
            if self.topology == MigrationTopology.RING:
                migration_events = await self._gpu_ring_migration(islands, generation)
            elif self.topology == MigrationTopology.FULLY_CONNECTED:
                migration_events = await self._gpu_fully_connected_migration(islands, generation)
            elif self.topology == MigrationTopology.STAR:
                migration_events = await self._gpu_star_migration(islands, generation)
            else:  # RANDOM
                migration_events = await self._gpu_random_migration(islands, generation)
        
        logger.info(f"GPU migration selection completed: {len(migration_events)} events")
        return migration_events
    
    async def _gpu_ring_migration(self, islands: List[Island], generation: int) -> List[MigrationEvent]:
        """GPU 가속 링 토폴로지 이주"""
        migration_events = []
        
        # 모든 island의 피트니스를 GPU에서 병렬 처리
        island_fitness_tensors = []
        for island in islands:
            if not island.population:
                island_fitness_tensors.append(None)
                continue
            
            fitness_scores = [
                p.fitness_scores.get("score", p.fitness_scores.get("correctness", 0.0))
                for p in island.population
            ]
            
            if fitness_scores:
                fitness_tensor = torch.tensor(fitness_scores, device=self.device, dtype=torch.float32)
                island_fitness_tensors.append(fitness_tensor)
            else:
                island_fitness_tensors.append(None)
        
        # 링 토폴로지로 이주 수행
        for i in range(len(islands)):
            source_island = islands[i]
            target_index = (i + 1) % len(islands)
            target_island = islands[target_index]
            
            if island_fitness_tensors[i] is None:
                continue
            
            # GPU에서 최고 개체 선택
            fitness_tensor = island_fitness_tensors[i]
            _, best_idx = torch.max(fitness_tensor, 0)
            best_program = source_island.population[best_idx.item()]
            
            # 이주 수행
            migrant = self._create_migrant(best_program, generation)
            target_island.population.append(migrant)
            target_island.migration_history.append(source_island.id)
            
            migration_event = MigrationEvent(
                source_island_id=source_island.id,
                target_island_id=target_island.id,
                migrant_program_ids=[migrant.id],
                generation=generation,
                migration_type="gpu_ring_elite"
            )
            migration_events.append(migration_event)
        
        return migration_events
    
    async def _gpu_fully_connected_migration(self, islands: List[Island], generation: int) -> List[MigrationEvent]:
        """GPU 가속 완전 연결 토폴로지 이주"""
        migration_events = []
        
        # 모든 island 쌍에 대해 GPU에서 병렬 처리
        with torch.cuda.device(self.device):
            for i, source_island in enumerate(islands):
                if not source_island.population:
                    continue
                
                # 소스 island의 피트니스 계산
                fitness_scores = [
                    p.fitness_scores.get("score", p.fitness_scores.get("correctness", 0.0))
                    for p in source_island.population
                ]
                
                if not fitness_scores:
                    continue
                
                fitness_tensor = torch.tensor(fitness_scores, device=self.device, dtype=torch.float32)
                
                # 상위 N개 개체 선택 (N = 다른 island 수)
                num_migrants = min(len(islands) - 1, len(source_island.population))
                if num_migrants <= 0:
                    continue
                
                _, top_indices = torch.topk(fitness_tensor, num_migrants)
                top_indices_cpu = top_indices.cpu().numpy()
                
                # 각 다른 island로 이주
                for j, target_island in enumerate(islands):
                    if i == j:  # 자기 자신 제외
                        continue
                    
                    if j - (1 if j > i else 0) >= len(top_indices_cpu):
                        continue
                    
                    migrant_idx = top_indices_cpu[j - (1 if j > i else 0)]
                    best_program = source_island.population[migrant_idx]
                    
                    migrant = self._create_migrant(best_program, generation)
                    target_island.population.append(migrant)
                    target_island.migration_history.append(source_island.id)
                    
                    migration_event = MigrationEvent(
                        source_island_id=source_island.id,
                        target_island_id=target_island.id,
                        migrant_program_ids=[migrant.id],
                        generation=generation,
                        migration_type="gpu_fully_connected"
                    )
                    migration_events.append(migration_event)
        
        return migration_events
    
    async def _gpu_star_migration(self, islands: List[Island], generation: int) -> List[MigrationEvent]:
        """GPU 가속 스타 토폴로지 이주"""
        migration_events = []
        
        if not islands:
            return migration_events
        
        # 첫 번째 island를 허브로 사용
        hub_island = islands[0]
        satellite_islands = islands[1:]
        
        with torch.cuda.device(self.device):
            # 허브에서 위성들로 이주
            if hub_island.population:
                fitness_scores = [
                    p.fitness_scores.get("score", p.fitness_scores.get("correctness", 0.0))
                    for p in hub_island.population
                ]
                
                if fitness_scores:
                    fitness_tensor = torch.tensor(fitness_scores, device=self.device, dtype=torch.float32)
                    
                    # 상위 개체들을 각 위성으로 이주
                    num_migrants = min(len(satellite_islands), len(hub_island.population))
                    if num_migrants > 0:
                        _, top_indices = torch.topk(fitness_tensor, num_migrants)
                        top_indices_cpu = top_indices.cpu().numpy()
                        
                        for i, satellite_island in enumerate(satellite_islands):
                            if i >= len(top_indices_cpu):
                                break
                            
                            migrant_idx = top_indices_cpu[i]
                            best_program = hub_island.population[migrant_idx]
                            
                            migrant = self._create_migrant(best_program, generation)
                            satellite_island.population.append(migrant)
                            satellite_island.migration_history.append(hub_island.id)
                            
                            migration_event = MigrationEvent(
                                source_island_id=hub_island.id,
                                target_island_id=satellite_island.id,
                                migrant_program_ids=[migrant.id],
                                generation=generation,
                                migration_type="gpu_star_hub_to_satellite"
                            )
                            migration_events.append(migration_event)
            
            # 위성들에서 허브로 이주
            for satellite_island in satellite_islands:
                if not satellite_island.population:
                    continue
                
                fitness_scores = [
                    p.fitness_scores.get("score", p.fitness_scores.get("correctness", 0.0))
                    for p in satellite_island.population
                ]
                
                if not fitness_scores:
                    continue
                
                fitness_tensor = torch.tensor(fitness_scores, device=self.device, dtype=torch.float32)
                _, best_idx = torch.max(fitness_tensor, 0)
                best_program = satellite_island.population[best_idx.item()]
                
                migrant = self._create_migrant(best_program, generation)
                hub_island.population.append(migrant)
                hub_island.migration_history.append(satellite_island.id)
                
                migration_event = MigrationEvent(
                    source_island_id=satellite_island.id,
                    target_island_id=hub_island.id,
                    migrant_program_ids=[migrant.id],
                    generation=generation,
                    migration_type="gpu_star_satellite_to_hub"
                )
                migration_events.append(migration_event)
        
        return migration_events
    
    async def _gpu_random_migration(self, islands: List[Island], generation: int) -> List[MigrationEvent]:
        """GPU 가속 랜덤 토폴로지 이주"""
        migration_events = []
        
        with torch.cuda.device(self.device):
            for source_island in islands:
                if not source_island.population:
                    continue
                
                # 랜덤 목적지 선택
                possible_targets = [island for island in islands if island.id != source_island.id]
                if not possible_targets:
                    continue
                
                target_island = random.choice(possible_targets)
                
                # GPU에서 이주자 선택
                fitness_scores = [
                    p.fitness_scores.get("score", p.fitness_scores.get("correctness", 0.0))
                    for p in source_island.population
                ]
                
                if not fitness_scores:
                    continue
                
                fitness_tensor = torch.tensor(fitness_scores, device=self.device, dtype=torch.float32)
                
                # 확률적 선택 (피트니스 기반)
                probabilities = F.softmax(fitness_tensor, dim=0)
                selected_idx = torch.multinomial(probabilities, 1).item()
                selected_program = source_island.population[selected_idx]
                
                migrant = self._create_migrant(selected_program, generation)
                target_island.population.append(migrant)
                target_island.migration_history.append(source_island.id)
                
                migration_event = MigrationEvent(
                    source_island_id=source_island.id,
                    target_island_id=target_island.id,
                    migrant_program_ids=[migrant.id],
                    generation=generation,
                    migration_type="gpu_random_probabilistic"
                )
                migration_events.append(migration_event)
        
        return migration_events
    
    def _create_migrant(self, source_program: Program, generation: int) -> Program:
        """이주자 프로그램 생성"""
        return Program(
            id=f"{source_program.id}_gpu_migrant_gen{generation}",
            code=source_program.code,
            fitness_scores=source_program.fitness_scores.copy(),
            generation=generation,
            parent_id=source_program.id,
            status=source_program.status
        )
    
    async def calculate_migration_rate(self, islands: List[Island], generation: int) -> float:
        """GPU 가속 적응적 이주율 계산"""
        if not islands:
            return self.base_migration_rate
        
        with torch.cuda.device(self.device):
            # 모든 island의 다양성을 GPU에서 병렬 계산
            diversity_scores = await self._calculate_gpu_diversity(islands)
            
            if not diversity_scores:
                return self.base_migration_rate
            
            # GPU에서 평균 다양성 계산
            diversity_tensor = torch.tensor(diversity_scores, device=self.device, dtype=torch.float32)
            avg_diversity = torch.mean(diversity_tensor).item()
            
            # 적응적 이주율 계산
            if avg_diversity < self.diversity_threshold * 0.5:
                # 매우 낮은 다양성 -> 높은 이주율
                migration_rate = min(self.max_migration_rate, self.base_migration_rate * 2.0)
            elif avg_diversity < self.diversity_threshold:
                # 낮은 다양성 -> 증가된 이주율
                migration_rate = min(self.max_migration_rate, self.base_migration_rate * 1.5)
            else:
                # 충분한 다양성 -> 기본 이주율
                migration_rate = self.base_migration_rate
            
            logger.debug(f"GPU adaptive migration rate: {migration_rate:.3f} (diversity: {avg_diversity:.3f})")
            return migration_rate
    
    def get_gpu_migration_statistics(self) -> Dict[str, Any]:
        """GPU 이주 통계 반환"""
        base_stats = {
            "topology": self.topology.value,
            "migration_interval": self.migration_interval,
            "base_migration_rate": self.base_migration_rate,
            "diversity_threshold": self.diversity_threshold,
            "device": str(self.device),
            "batch_migration": self.batch_migration,
            "use_gpu_sorting": self.use_gpu_sorting
        }
        
        if torch.cuda.is_available():
            base_stats.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated(0) / 1e9,
                "gpu_memory_reserved": torch.cuda.memory_reserved(0) / 1e9,
                "cuda_version": torch.version.cuda
            })
        
        return base_stats
    
    def select_destination_islands(self, source_island: Island, all_islands: List[Island]) -> List[Island]:
        """GPU 가속 목적지 island 선택"""
        if not all_islands or len(all_islands) <= 1:
            return []
        
        # 소스 island 제외
        possible_destinations = [island for island in all_islands if island.id != source_island.id]
        
        if not possible_destinations:
            return []
        
        with torch.cuda.device(self.device):
            # 토폴로지에 따른 목적지 선택
            if self.topology == MigrationTopology.RING:
                # 링 토폴로지: 다음 island만 선택
                source_idx = next((i for i, island in enumerate(all_islands) if island.id == source_island.id), -1)
                if source_idx >= 0:
                    target_idx = (source_idx + 1) % len(all_islands)
                    return [all_islands[target_idx]]
                return []
            
            elif self.topology == MigrationTopology.FULLY_CONNECTED:
                # 완전 연결: 모든 다른 island
                return possible_destinations
            
            elif self.topology == MigrationTopology.STAR:
                # 스타 토폴로지: 허브 또는 위성들
                hub_island = all_islands[0]  # 첫 번째를 허브로 가정
                
                if source_island.id == hub_island.id:
                    # 허브에서 모든 위성으로
                    return [island for island in all_islands[1:]]
                else:
                    # 위성에서 허브로만
                    return [hub_island]
            
            else:  # RANDOM
                # 랜덤: GPU에서 확률적 선택
                if len(possible_destinations) == 1:
                    return possible_destinations
                
                # GPU에서 랜덤 선택 (1-3개 목적지)
                num_destinations = min(3, len(possible_destinations))
                selected_indices = torch.randperm(len(possible_destinations), device=self.device)[:num_destinations]
                selected_indices_cpu = selected_indices.cpu().numpy()
                
                return [possible_destinations[i] for i in selected_indices_cpu]
    
    def cleanup_gpu_cache(self):
        """GPU 캐시 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.migration_tensor_cache.clear()
            logger.debug("GPU migration cache cleaned up")
    
    async def execute(self, action: str, **kwargs) -> Any:
        """GPU Migration Policy 액션 실행"""
        try:
            if action == "should_migrate":
                return await self.should_migrate(kwargs["islands"], kwargs["generation"])
            elif action == "select_migrants":
                return await self.select_migrants(kwargs["islands"], kwargs["generation"])
            elif action == "calculate_rate":
                return await self.calculate_migration_rate(kwargs["islands"], kwargs["generation"])
            elif action == "get_stats":
                return self.get_gpu_migration_statistics()
            elif action == "cleanup":
                self.cleanup_gpu_cache()
                return True
            else:
                raise ValueError(f"Unknown action: {action}")
        except Exception as e:
            logger.error(f"Error executing GPU Migration Policy action '{action}': {e}")
            raise 
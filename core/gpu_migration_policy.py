"""
GPU Migration Policy - 고성능 Island 간 이주 정책
다양한 migration 전략과 GPU 최적화된 선택 알고리즘 구현
"""
import logging
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import torch

from core.interfaces import (
    MigrationPolicyInterface, Island, Program, EvolutionStrategy, BaseAgent
)
from config import settings

logger = logging.getLogger(__name__)

class MigrationTopology(Enum):
    """Migration 토폴로지 타입"""
    RING = "ring"                    # 원형 연결
    FULLY_CONNECTED = "fully_connected"  # 완전 연결
    STAR = "star"                    # 스타 형태
    RANDOM = "random"                # 무작위 연결
    ADAPTIVE = "adaptive"            # 적응적 연결

class GPUMigrationPolicy(MigrationPolicyInterface):
    """
    GPU 최적화된 Migration 정책
    
    특징:
    - 다양한 토폴로지 지원
    - 적응적 migration 빈도
    - 성능 기반 개체 선택
    - GPU 메모리 효율적 처리
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Migration 설정
        self.migration_interval = self.config.get("migration_interval", 5)
        self.migration_rate = self.config.get("migration_rate", 0.15)
        self.topology = MigrationTopology(self.config.get("topology", "ring"))
        self.elite_migration = self.config.get("elite_migration", True)
        
        # 적응적 설정
        self.adaptive_threshold = self.config.get("adaptive_threshold", 0.1)
        self.diversity_weight = self.config.get("diversity_weight", 0.3)
        self.performance_weight = self.config.get("performance_weight", 0.7)
        
        # GPU 설정
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_processing = self.config.get("batch_processing", True)
        
        # 통계 추적
        self.migration_history = []
        self.topology_performance = {topology.value: [] for topology in MigrationTopology}
        
        logger.info(f"GPUMigrationPolicy initialized:")
        logger.info(f"  Topology: {self.topology.value}")
        logger.info(f"  Migration rate: {self.migration_rate}")
        logger.info(f"  Elite migration: {self.elite_migration}")

    def should_migrate(self, generation: int) -> bool:
        """Migration 수행 여부 결정"""
        # 기본 주기 확인
        if generation % self.migration_interval != 0:
            return False
        
        # 첫 번째 generation은 skip
        if generation < 2:
            return False
        
        # 적응적 migration (성능 정체 시 더 자주)
        if self.topology == MigrationTopology.ADAPTIVE:
            return self._should_migrate_adaptive(generation)
        
        return True

    def _should_migrate_adaptive(self, generation: int) -> bool:
        """적응적 migration 결정"""
        if len(self.migration_history) < 2:
            return True
        
        # 최근 migration 성과 분석
        recent_migrations = self.migration_history[-3:]
        improvement_rate = sum(m.get("improvement", 0) for m in recent_migrations) / len(recent_migrations)
        
        # 성과가 좋지 않으면 더 자주 migration
        if improvement_rate < self.adaptive_threshold:
            return generation % max(2, self.migration_interval // 2) == 0
        
        return generation % self.migration_interval == 0

    def select_migrants(self, source_island: Island, num_migrants: int) -> List[Program]:
        """Migration할 개체 선택"""
        if not source_island.population or num_migrants <= 0:
            return []
        
        available_programs = [p for p in source_island.population 
                            if p.status == "evaluated" and p.fitness_scores]
        
        if not available_programs:
            return []
        
        num_migrants = min(num_migrants, len(available_programs))
        
        if self.elite_migration:
            return self._select_elite_migrants(available_programs, num_migrants, source_island)
        else:
            return self._select_diverse_migrants(available_programs, num_migrants, source_island)

    def _select_elite_migrants(self, programs: List[Program], num_migrants: int, source_island: Island) -> List[Program]:
        """엘리트 기반 개체 선택"""
        # 적합도 기준 정렬
        sorted_programs = sorted(
            programs,
            key=lambda p: (
                p.fitness_scores.get("correctness", 0.0),
                -p.fitness_scores.get("runtime_ms", float('inf')),
                -p.fitness_scores.get("complexity", 0.0)  # 낮은 복잡도 선호
            ),
            reverse=True
        )
        
        # 상위 개체 선택 (일부 랜덤 요소 추가)
        elite_count = max(1, int(num_migrants * 0.7))
        random_count = num_migrants - elite_count
        
        migrants = sorted_programs[:elite_count]
        
        # 랜덤 선택으로 다양성 확보
        if random_count > 0 and len(sorted_programs) > elite_count:
            remaining = sorted_programs[elite_count:]
            migrants.extend(random.sample(remaining, min(random_count, len(remaining))))
        
        logger.debug(f"Selected {len(migrants)} elite migrants from {source_island.id}")
        return migrants

    def _select_diverse_migrants(self, programs: List[Program], num_migrants: int, source_island: Island) -> List[Program]:
        """다양성 기반 개체 선택"""
        if len(programs) <= num_migrants:
            return programs
        
        # GPU 가속 다양성 계산
        if torch.cuda.is_available() and len(programs) > 50:
            return self._select_diverse_migrants_gpu(programs, num_migrants, source_island)
        
        # CPU 기반 다양성 선택
        selected = []
        remaining = programs.copy()
        
        # 첫 번째는 최고 성능 개체
        best = max(remaining, key=lambda p: p.fitness_scores.get("correctness", 0.0))
        selected.append(best)
        remaining.remove(best)
        
        # 나머지는 다양성 기준으로 선택
        while len(selected) < num_migrants and remaining:
            most_diverse = self._find_most_diverse_program(selected, remaining)
            selected.append(most_diverse)
            remaining.remove(most_diverse)
        
        logger.debug(f"Selected {len(selected)} diverse migrants from {source_island.id}")
        return selected

    def _select_diverse_migrants_gpu(self, programs: List[Program], num_migrants: int, source_island: Island) -> List[Program]:
        """GPU 가속 다양성 선택"""
        try:
            # 프로그램 특성 벡터 생성
            features = []
            for program in programs:
                feature_vector = [
                    program.fitness_scores.get("correctness", 0.0),
                    program.fitness_scores.get("runtime_ms", 0.0) / 1000.0,  # 정규화
                    program.fitness_scores.get("complexity", 0.0),
                    len(program.code),
                    program.generation
                ]
                features.append(feature_vector)
            
            # GPU 텐서로 변환
            feature_tensor = torch.tensor(features, device=self.device, dtype=torch.float32)
            
            # 거리 기반 다양성 선택
            selected_indices = self._gpu_diverse_selection(feature_tensor, num_migrants)
            
            return [programs[i] for i in selected_indices]
            
        except Exception as e:
            logger.warning(f"GPU diverse selection failed, falling back to CPU: {e}")
            return self._select_diverse_migrants(programs, num_migrants, source_island)

    def _gpu_diverse_selection(self, feature_tensor: torch.Tensor, num_migrants: int) -> List[int]:
        """GPU 기반 다양성 선택 알고리즘"""
        n_programs = feature_tensor.shape[0]
        selected_indices = []
        
        # 첫 번째는 최고 성능 (correctness 기준)
        best_idx = torch.argmax(feature_tensor[:, 0]).item()
        selected_indices.append(best_idx)
        
        # 나머지는 다양성 기준
        for _ in range(num_migrants - 1):
            if len(selected_indices) >= n_programs:
                break
            
            # 선택된 개체들과의 거리 계산
            selected_tensor = feature_tensor[selected_indices]
            distances = torch.cdist(feature_tensor, selected_tensor)
            min_distances = torch.min(distances, dim=1)[0]
            
            # 이미 선택된 개체는 제외
            for idx in selected_indices:
                min_distances[idx] = -1
            
            # 가장 먼 개체 선택
            next_idx = torch.argmax(min_distances).item()
            selected_indices.append(next_idx)
        
        return selected_indices

    def _find_most_diverse_program(self, selected: List[Program], candidates: List[Program]) -> Program:
        """가장 다양한 개체 찾기 (CPU 버전)"""
        best_program = None
        max_diversity = -1
        
        for candidate in candidates:
            diversity_score = self._calculate_diversity_score(candidate, selected)
            if diversity_score > max_diversity:
                max_diversity = diversity_score
                best_program = candidate
        
        return best_program or candidates[0]

    def _calculate_diversity_score(self, candidate: Program, selected: List[Program]) -> float:
        """개체의 다양성 점수 계산"""
        if not selected:
            return 1.0
        
        # 특성 기반 다양성 계산
        candidate_features = self._extract_program_features(candidate)
        
        min_distance = float('inf')
        for selected_prog in selected:
            selected_features = self._extract_program_features(selected_prog)
            distance = self._calculate_feature_distance(candidate_features, selected_features)
            min_distance = min(min_distance, distance)
        
        return min_distance

    def _extract_program_features(self, program: Program) -> List[float]:
        """프로그램의 특성 벡터 추출"""
        return [
            program.fitness_scores.get("correctness", 0.0),
            program.fitness_scores.get("runtime_ms", 0.0),
            program.fitness_scores.get("complexity", 0.0),
            len(program.code),
            len(program.code.split('\n'))
        ]

    def _calculate_feature_distance(self, features1: List[float], features2: List[float]) -> float:
        """특성 벡터 간 거리 계산"""
        distance = 0.0
        for f1, f2 in zip(features1, features2):
            distance += (f1 - f2) ** 2
        return distance ** 0.5

    def select_destination_islands(self, source_island: Island, all_islands: List[Island]) -> List[Island]:
        """Migration 대상 Island 선택"""
        if len(all_islands) <= 1:
            return []
        
        destinations = []
        
        if self.topology == MigrationTopology.RING:
            destinations = self._select_ring_destinations(source_island, all_islands)
        elif self.topology == MigrationTopology.FULLY_CONNECTED:
            destinations = self._select_fully_connected_destinations(source_island, all_islands)
        elif self.topology == MigrationTopology.STAR:
            destinations = self._select_star_destinations(source_island, all_islands)
        elif self.topology == MigrationTopology.RANDOM:
            destinations = self._select_random_destinations(source_island, all_islands)
        elif self.topology == MigrationTopology.ADAPTIVE:
            destinations = self._select_adaptive_destinations(source_island, all_islands)
        
        return destinations

    def _select_ring_destinations(self, source_island: Island, all_islands: List[Island]) -> List[Island]:
        """Ring 토폴로지 대상 선택"""
        source_idx = all_islands.index(source_island)
        next_idx = (source_idx + 1) % len(all_islands)
        return [all_islands[next_idx]]

    def _select_fully_connected_destinations(self, source_island: Island, all_islands: List[Island]) -> List[Island]:
        """완전 연결 토폴로지 대상 선택"""
        return [island for island in all_islands if island.id != source_island.id]

    def _select_star_destinations(self, source_island: Island, all_islands: List[Island]) -> List[Island]:
        """스타 토폴로지 대상 선택"""
        # 첫 번째 Island를 허브로 사용
        hub_island = all_islands[0]
        
        if source_island.id == hub_island.id:
            # 허브에서 모든 다른 Island로
            return [island for island in all_islands[1:]]
        else:
            # 다른 Island에서 허브로만
            return [hub_island]

    def _select_random_destinations(self, source_island: Island, all_islands: List[Island]) -> List[Island]:
        """무작위 토폴로지 대상 선택"""
        candidates = [island for island in all_islands if island.id != source_island.id]
        num_destinations = random.randint(1, min(3, len(candidates)))
        return random.sample(candidates, num_destinations)

    def _select_adaptive_destinations(self, source_island: Island, all_islands: List[Island]) -> List[Island]:
        """적응적 토폴로지 대상 선택"""
        candidates = [island for island in all_islands if island.id != source_island.id]
        
        # 성능 차이 기반 선택
        source_fitness = source_island.get_average_fitness()
        
        destinations = []
        for candidate in candidates:
            candidate_fitness = candidate.get_average_fitness()
            
            # 성능이 낮은 Island로 우선 migration
            if candidate_fitness < source_fitness:
                destinations.append(candidate)
        
        # 대상이 없으면 무작위 선택
        if not destinations:
            destinations = random.sample(candidates, min(2, len(candidates)))
        
        return destinations

    def calculate_migration_success(self, pre_migration_fitness: Dict[str, float], 
                                  post_migration_fitness: Dict[str, float]) -> float:
        """Migration 성공도 계산"""
        total_improvement = 0.0
        island_count = 0
        
        for island_id in pre_migration_fitness:
            if island_id in post_migration_fitness:
                improvement = post_migration_fitness[island_id] - pre_migration_fitness[island_id]
                total_improvement += improvement
                island_count += 1
        
        if island_count == 0:
            return 0.0
        
        return total_improvement / island_count

    def update_topology_performance(self, topology: str, success_rate: float):
        """토폴로지 성능 업데이트"""
        if topology in self.topology_performance:
            self.topology_performance[topology].append(success_rate)
            
            # 최근 5회 평균만 유지
            if len(self.topology_performance[topology]) > 5:
                self.topology_performance[topology].pop(0)

    def get_best_topology(self) -> MigrationTopology:
        """가장 성능이 좋은 토폴로지 반환"""
        best_topology = MigrationTopology.RING
        best_performance = -1
        
        for topology, performances in self.topology_performance.items():
            if performances:
                avg_performance = sum(performances) / len(performances)
                if avg_performance > best_performance:
                    best_performance = avg_performance
                    best_topology = MigrationTopology(topology)
        
        return best_topology

    def get_migration_statistics(self) -> Dict[str, Any]:
        """Migration 통계 반환"""
        return {
            "total_migrations": len(self.migration_history),
            "topology": self.topology.value,
            "migration_rate": self.migration_rate,
            "elite_migration": self.elite_migration,
            "topology_performance": dict(self.topology_performance),
            "recent_success_rate": self._calculate_recent_success_rate()
        }

    def _calculate_recent_success_rate(self) -> float:
        """최근 migration 성공률 계산"""
        if len(self.migration_history) < 5:
            return 0.0
        
        recent_migrations = self.migration_history[-5:]
        successful = sum(1 for m in recent_migrations if m.get("improvement", 0) > 0)
        return successful / len(recent_migrations)

    async def execute(self, action: str, **kwargs) -> Any:
        """메인 실행 메소드"""
        if action == "should_migrate":
            return self.should_migrate(kwargs.get("generation", 0))
        elif action == "select_migrants":
            return self.select_migrants(
                kwargs.get("source_island"),
                kwargs.get("num_migrants", 1)
            )
        elif action == "select_destinations":
            return self.select_destination_islands(
                kwargs.get("source_island"),
                kwargs.get("all_islands", [])
            )
        elif action == "calculate_success":
            return self.calculate_migration_success(
                kwargs.get("pre_fitness", {}),
                kwargs.get("post_fitness", {})
            )
        else:
            raise ValueError(f"Unknown action: {action}")


def create_gpu_migration_policy(config: Optional[Dict[str, Any]] = None) -> GPUMigrationPolicy:
    """GPU Migration Policy 팩토리 함수"""
    return GPUMigrationPolicy(config)


# 토폴로지 성능 테스트 함수
def test_topology_performance():
    """다양한 토폴로지 성능 테스트"""
    print("=== Migration Topology Performance Test ===")
    
    for topology in MigrationTopology:
        config = {
            "topology": topology.value,
            "migration_rate": 0.15,
            "elite_migration": True
        }
        
        policy = GPUMigrationPolicy(config)
        print(f"Topology: {topology.value}")
        print(f"  Migration rate: {policy.migration_rate}")
        print(f"  Elite migration: {policy.elite_migration}")
    
    print("============================================")


if __name__ == "__main__":
    test_topology_performance() 
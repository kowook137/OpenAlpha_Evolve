"""
GPU-accelerated Island Manager using PyTorch for RTX 3080.
Implements parallel evolution across multiple islands using CUDA.
"""
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import asyncio
import logging
import numpy as np
import random
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import time

from core.interfaces import (
    IslandManagerInterface, Island, Program, TaskDefinition, 
    EvolutionStrategy, MigrationEvent
)
from config import settings

logger = logging.getLogger(__name__)

class GPUIslandManager(IslandManagerInterface):
    """GPU-accelerated Island Manager for parallel evolution using RTX 3080"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # GPU 설정
        self.device = self._setup_gpu()
        self.islands: List[Island] = []
        self.migration_history: List[MigrationEvent] = []
        
        # GPU 병렬 처리 설정
        self.gpu_batch_size = self.config.get("gpu_batch_size", 32)
        self.use_mixed_precision = self.config.get("use_mixed_precision", True)
        
        # 멀티프로세싱 설정
        mp.set_start_method('spawn', force=True)
        self.num_gpu_workers = self.config.get("num_gpu_workers", 4)
        
        logger.info(f"GPUIslandManager initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        logger.info(f"  GPU Batch Size: {self.gpu_batch_size}")
        logger.info(f"  Mixed Precision: {self.use_mixed_precision}")
        logger.info(f"  GPU Workers: {self.num_gpu_workers}")
    
    def _setup_gpu(self) -> torch.device:
        """GPU 설정 및 확인"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return torch.device("cpu")
        
        # RTX 3080 확인
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU detected: {gpu_name}")
        
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        
        # CUDA 최적화 설정
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        return device
    
    async def initialize_islands(self, num_islands: int, population_per_island: int) -> List[Island]:
        """GPU를 활용하여 islands 초기화"""
        logger.info(f"Initializing {num_islands} islands with {population_per_island} population each on GPU")
        
        self.islands = []
        strategies = list(EvolutionStrategy)
        
        # GPU에서 병렬로 island 초기화
        with torch.cuda.device(self.device):
            for i in range(num_islands):
                strategy = strategies[i % len(strategies)]
                
                island = Island(
                    id=f"gpu_island_{i}",
                    population=[],
                    strategy=strategy,
                    generation=0
                )
                
                # GPU 메모리에 island 데이터 할당
                island.gpu_id = i % torch.cuda.device_count()
                self.islands.append(island)
                
                logger.debug(f"Created {island.id} with strategy {strategy.value} on GPU {island.gpu_id}")
        
        return self.islands
    
    async def evolve_islands_parallel(self, islands: List[Island], task: TaskDefinition) -> List[Island]:
        """GPU를 사용하여 모든 islands를 병렬로 진화"""
        logger.info(f"Starting GPU-accelerated parallel evolution of {len(islands)} islands")
        
        start_time = time.time()
        
        # GPU 메모리 최적화
        torch.cuda.empty_cache()
        
        # Islands를 GPU 배치로 그룹화
        island_batches = self._create_gpu_batches(islands)
        
        # 각 배치를 병렬로 처리
        evolved_islands = []
        
        with torch.cuda.device(self.device):
            # Mixed precision 사용
            if self.use_mixed_precision:
                scaler = torch.cuda.amp.GradScaler()
            
            # 병렬 처리를 위한 멀티프로세싱
            with mp.Pool(processes=self.num_gpu_workers) as pool:
                evolution_tasks = []
                
                for batch in island_batches:
                    task_args = (batch, task, self.device, self.use_mixed_precision)
                    future = pool.apply_async(self._evolve_island_batch_gpu, task_args)
                    evolution_tasks.append(future)
                
                # 모든 배치 완료 대기
                for i, future in enumerate(evolution_tasks):
                    try:
                        batch_result = future.get(timeout=600)  # 10분 타임아웃
                        evolved_islands.extend(batch_result)
                        logger.debug(f"GPU batch {i} evolution completed")
                    except Exception as e:
                        logger.error(f"Error in GPU batch {i}: {e}")
                        # 실패한 경우 원본 배치 반환
                        evolved_islands.extend(island_batches[i])
        
        evolution_time = time.time() - start_time
        logger.info(f"GPU parallel evolution completed in {evolution_time:.2f}s for {len(evolved_islands)} islands")
        
        return evolved_islands
    
    def _create_gpu_batches(self, islands: List[Island]) -> List[List[Island]]:
        """Islands를 GPU 처리에 최적화된 배치로 분할"""
        batch_size = max(1, len(islands) // self.num_gpu_workers)
        batches = []
        
        for i in range(0, len(islands), batch_size):
            batch = islands[i:i + batch_size]
            batches.append(batch)
        
        return batches
    
    @staticmethod
    def _evolve_island_batch_gpu(batch: List[Island], task: TaskDefinition, 
                                device: torch.device, use_mixed_precision: bool) -> List[Island]:
        """GPU에서 island 배치를 병렬로 진화 (정적 메서드로 멀티프로세싱 지원)"""
        try:
            # GPU 컨텍스트 설정
            torch.cuda.set_device(device)
            
            evolved_batch = []
            
            for island in batch:
                if use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        evolved_island = GPUIslandManager._evolve_single_island_gpu(island, task, device)
                else:
                    evolved_island = GPUIslandManager._evolve_single_island_gpu(island, task, device)
                
                evolved_batch.append(evolved_island)
            
            return evolved_batch
            
        except Exception as e:
            logger.error(f"Error in GPU batch evolution: {e}")
            return batch
    
    @staticmethod
    def _evolve_single_island_gpu(island: Island, task: TaskDefinition, device: torch.device) -> Island:
        """GPU에서 단일 island 진화"""
        try:
            if not island.population:
                return island
            
            # 피트니스 점수를 GPU 텐서로 변환
            fitness_scores = []
            for program in island.population:
                score = program.fitness_scores.get("score", program.fitness_scores.get("correctness", 0.0))
                fitness_scores.append(score)
            
            if not fitness_scores:
                return island
            
            # GPU에서 텐서 연산 수행
            fitness_tensor = torch.tensor(fitness_scores, device=device, dtype=torch.float32)
            
            # 전략별 GPU 가속 진화
            if island.strategy == EvolutionStrategy.EXPLOITATION:
                island = GPUIslandManager._gpu_exploitation_strategy(island, fitness_tensor, device)
            elif island.strategy == EvolutionStrategy.EXPLORATION:
                island = GPUIslandManager._gpu_exploration_strategy(island, fitness_tensor, device)
            elif island.strategy == EvolutionStrategy.RANDOM:
                island = GPUIslandManager._gpu_random_strategy(island, fitness_tensor, device)
            else:  # BALANCED
                island = GPUIslandManager._gpu_balanced_strategy(island, fitness_tensor, device)
            
            # 평균 피트니스 계산 (GPU)
            avg_fitness = torch.mean(fitness_tensor).item()
            island.fitness_history.append(avg_fitness)
            island.generation += 1
            
            return island
            
        except Exception as e:
            logger.error(f"Error in GPU island evolution: {e}")
            return island
    
    @staticmethod
    def _gpu_exploitation_strategy(island: Island, fitness_tensor: torch.Tensor, device: torch.device) -> Island:
        """GPU 가속 착취 전략"""
        # 상위 50% 선택
        top_k = max(1, len(island.population) // 2)
        _, top_indices = torch.topk(fitness_tensor, top_k)
        
        # CPU로 인덱스 변환하여 프로그램 선택
        top_indices_cpu = top_indices.cpu().numpy()
        island.population = [island.population[i] for i in top_indices_cpu]
        
        return island
    
    @staticmethod
    def _gpu_exploration_strategy(island: Island, fitness_tensor: torch.Tensor, device: torch.device) -> Island:
        """GPU 가속 탐색 전략"""
        if len(island.population) > 2:
            # 하위 30% 제거
            remove_count = max(1, len(island.population) // 3)
            _, bottom_indices = torch.topk(fitness_tensor, remove_count, largest=False)
            
            # 상위 개체들만 유지
            keep_indices = set(range(len(island.population))) - set(bottom_indices.cpu().numpy())
            island.population = [island.population[i] for i in keep_indices]
        
        return island
    
    @staticmethod
    def _gpu_random_strategy(island: Island, fitness_tensor: torch.Tensor, device: torch.device) -> Island:
        """GPU 가속 랜덤 전략"""
        # GPU에서 랜덤 순열 생성
        perm = torch.randperm(len(island.population), device=device)
        perm_cpu = perm.cpu().numpy()
        
        # 프로그램 순서 섞기
        island.population = [island.population[i] for i in perm_cpu]
        
        return island
    
    @staticmethod
    def _gpu_balanced_strategy(island: Island, fitness_tensor: torch.Tensor, device: torch.device) -> Island:
        """GPU 가속 균형 전략"""
        # 상위 70% 유지
        keep_count = max(1, int(len(island.population) * 0.7))
        _, top_indices = torch.topk(fitness_tensor, keep_count)
        
        top_indices_cpu = top_indices.cpu().numpy()
        island.population = [island.population[i] for i in top_indices_cpu]
        
        return island
    
    async def migrate_between_islands(self, islands: List[Island], generation: int) -> List[Island]:
        """GPU 가속 island 간 이주"""
        if len(islands) < 2:
            return islands
        
        logger.info(f"Starting GPU-accelerated migration between {len(islands)} islands at generation {generation}")
        
        with torch.cuda.device(self.device):
            migration_events = []
            
            # 각 island의 최고 개체를 GPU에서 병렬로 찾기
            for i, source_island in enumerate(islands):
                if not source_island.population:
                    continue
                
                # GPU에서 최고 개체 선택
                fitness_scores = [
                    p.fitness_scores.get("score", p.fitness_scores.get("correctness", 0.0))
                    for p in source_island.population
                ]
                
                if not fitness_scores:
                    continue
                
                fitness_tensor = torch.tensor(fitness_scores, device=self.device)
                _, best_idx = torch.max(fitness_tensor, 0)
                best_program = source_island.population[best_idx.item()]
                
                # 목적지 island (링 토폴로지)
                target_index = (i + 1) % len(islands)
                target_island = islands[target_index]
                
                # 이주 수행
                migrant = Program(
                    id=f"{best_program.id}_gpu_migrant_gen{generation}",
                    code=best_program.code,
                    fitness_scores=best_program.fitness_scores.copy(),
                    generation=generation,
                    parent_id=best_program.id,
                    status=best_program.status
                )
                
                target_island.population.append(migrant)
                target_island.migration_history.append(source_island.id)
                
                migration_event = MigrationEvent(
                    source_island_id=source_island.id,
                    target_island_id=target_island.id,
                    migrant_program_ids=[migrant.id],
                    generation=generation,
                    migration_type="gpu_elite"
                )
                migration_events.append(migration_event)
            
            self.migration_history.extend(migration_events)
            logger.info(f"GPU migration completed: {len(migration_events)} migration events")
        
        return islands
    
    def get_gpu_statistics(self) -> Dict[str, Any]:
        """GPU 사용 통계 반환"""
        if not torch.cuda.is_available():
            return {"gpu_available": False}
        
        gpu_stats = {
            "gpu_available": True,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated": torch.cuda.memory_allocated(0) / 1e9,  # GB
            "gpu_memory_reserved": torch.cuda.memory_reserved(0) / 1e9,    # GB
            "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1e9,  # GB
            "gpu_utilization": self._get_gpu_utilization(),
            "cuda_version": torch.version.cuda,
            "pytorch_version": torch.__version__
        }
        
        return gpu_stats
    
    def _get_gpu_utilization(self) -> float:
        """GPU 사용률 추정"""
        try:
            allocated = torch.cuda.memory_allocated(0)
            total = torch.cuda.get_device_properties(0).total_memory
            return (allocated / total) * 100
        except:
            return 0.0
    
    def get_island_statistics(self) -> Dict[str, Any]:
        """Island 통계와 GPU 통계 결합"""
        base_stats = {
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
                "migration_count": len(island.migration_history),
                "gpu_id": getattr(island, 'gpu_id', 0)
            }
            base_stats["island_details"].append(island_stats)
        
        # GPU 통계 추가
        base_stats["gpu_statistics"] = self.get_gpu_statistics()
        
        return base_stats
    
    def cleanup_gpu_memory(self):
        """GPU 메모리 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info("GPU memory cleaned up")
    
    async def execute(self, action: str, **kwargs) -> Any:
        """GPU Island Manager 액션 실행"""
        try:
            if action == "initialize":
                return await self.initialize_islands(kwargs["num_islands"], kwargs["population_per_island"])
            elif action == "evolve_parallel":
                return await self.evolve_islands_parallel(kwargs["islands"], kwargs["task"])
            elif action == "migrate":
                return await self.migrate_between_islands(kwargs["islands"], kwargs["generation"])
            elif action == "get_stats":
                return self.get_island_statistics()
            elif action == "get_gpu_stats":
                return self.get_gpu_statistics()
            elif action == "cleanup":
                self.cleanup_gpu_memory()
                return True
            else:
                raise ValueError(f"Unknown action: {action}")
        except Exception as e:
            logger.error(f"Error executing GPU Island Manager action '{action}': {e}")
            raise 
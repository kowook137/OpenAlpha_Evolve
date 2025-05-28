"""
GPU-accelerated Island Task Manager using PyTorch for RTX 3080.
Integrates GPU-based Island Model, Migration Policy, and MAP-Elites.
"""
import torch
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from core.interfaces import TaskDefinition, Program, Island
from core.gpu_island_manager import GPUIslandManager
from core.gpu_migration_policy import GPUMigrationPolicy
from core.gpu_map_elites import GPUMAPElites
from config import settings

logger = logging.getLogger(__name__)

@dataclass
class GPUEvolutionMetrics:
    """GPU 진화 메트릭"""
    generation: int
    total_time: float
    gpu_time: float
    cpu_time: float
    memory_usage: float
    island_stats: Dict[str, Any]
    migration_stats: Dict[str, Any]
    map_elites_stats: Dict[str, Any]
    gpu_utilization: float

class GPUIslandTaskManager:
    """GPU-accelerated Island Task Manager for RTX 3080"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # GPU 설정
        self.device = self._setup_gpu()
        
        # GPU 기반 컴포넌트 초기화
        self.island_manager = GPUIslandManager(self.config.get("island_manager", {}))
        self.migration_policy = GPUMigrationPolicy(self.config.get("migration_policy", {}))
        self.map_elites = GPUMAPElites(self.config.get("map_elites", {}))
        
        # 진화 설정
        self.max_generations = self.config.get("max_generations", 50)
        self.population_size = self.config.get("population_size", 32)
        self.num_islands = self.config.get("num_islands", 4)
        
        # GPU 최적화 설정
        self.gpu_batch_size = self.config.get("gpu_batch_size", 16)
        self.use_mixed_precision = self.config.get("use_mixed_precision", True)
        self.enable_gpu_profiling = self.config.get("enable_gpu_profiling", True)
        
        # 메트릭 저장
        self.evolution_metrics: List[GPUEvolutionMetrics] = []
        self.islands: List[Island] = []
        
        logger.info(f"GPUIslandTaskManager initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Islands: {self.num_islands}")
        logger.info(f"  Population per island: {self.population_size // self.num_islands}")
        logger.info(f"  Max generations: {self.max_generations}")
        logger.info(f"  GPU batch size: {self.gpu_batch_size}")
        logger.info(f"  Mixed precision: {self.use_mixed_precision}")
    
    def _setup_gpu(self) -> torch.device:
        """GPU 설정 및 확인"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            return torch.device("cpu")
        
        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        logger.info(f"GPU detected: {gpu_name}")
        logger.info(f"GPU memory: {gpu_memory:.1f} GB")
        
        # GPU 메모리 정리
        torch.cuda.empty_cache()
        
        # CUDA 최적화 설정
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        return device
    
    async def run_evolution(self, task: TaskDefinition, initial_programs: List[Program]) -> Dict[str, Any]:
        """GPU 가속 진화 실행"""
        logger.info(f"Starting GPU-accelerated evolution for task: {task.name}")
        logger.info(f"Initial programs: {len(initial_programs)}")
        
        start_time = time.time()
        
        try:
            # GPU 메모리 정리
            self._cleanup_gpu_memory()
            
            # Islands 초기화
            await self._initialize_islands(initial_programs)
            
            # 진화 루프
            for generation in range(self.max_generations):
                generation_start = time.time()
                
                logger.info(f"Generation {generation + 1}/{self.max_generations}")
                
                # GPU에서 병렬 진화
                gpu_start = time.time()
                self.islands = await self.island_manager.evolve_islands_parallel(self.islands, task)
                gpu_time = time.time() - gpu_start
                
                # 이주 수행
                migration_start = time.time()
                if await self.migration_policy.should_migrate(self.islands, generation):
                    self.islands = await self.migration_policy.migrate_between_islands(self.islands, generation)
                migration_time = time.time() - migration_start
                
                # MAP-Elites 업데이트
                map_elites_start = time.time()
                all_programs = []
                for island in self.islands:
                    all_programs.extend(island.population)
                
                if all_programs:
                    await self.map_elites.add_to_archive(all_programs)
                map_elites_time = time.time() - map_elites_start
                
                # 메트릭 수집
                generation_time = time.time() - generation_start
                await self._collect_generation_metrics(
                    generation, generation_time, gpu_time, migration_time + map_elites_time
                )
                
                # 진행 상황 로깅
                if (generation + 1) % 10 == 0:
                    await self._log_progress(generation + 1)
                
                # GPU 메모리 정리 (주기적)
                if (generation + 1) % 20 == 0:
                    self._cleanup_gpu_memory()
            
            # 최종 결과 수집
            final_results = await self._collect_final_results(task)
            
            total_time = time.time() - start_time
            logger.info(f"GPU evolution completed in {total_time:.2f}s")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Error in GPU evolution: {e}")
            raise
        finally:
            # GPU 메모리 정리
            self._cleanup_gpu_memory()
    
    async def _initialize_islands(self, initial_programs: List[Program]):
        """Islands 초기화"""
        logger.info(f"Initializing {self.num_islands} islands with GPU acceleration")
        
        # Island Manager로 islands 생성
        population_per_island = max(1, len(initial_programs) // self.num_islands)
        self.islands = await self.island_manager.initialize_islands(
            self.num_islands, population_per_island
        )
        
        # 초기 프로그램들을 islands에 분배
        for i, program in enumerate(initial_programs):
            island_idx = i % len(self.islands)
            self.islands[island_idx].population.append(program)
        
        logger.info(f"Islands initialized: {len(self.islands)} islands")
        for i, island in enumerate(self.islands):
            logger.info(f"  Island {i}: {len(island.population)} programs, strategy: {island.strategy.value}")
    
    async def _collect_generation_metrics(self, generation: int, total_time: float, 
                                        gpu_time: float, cpu_time: float):
        """세대별 메트릭 수집"""
        # GPU 사용률 계산
        gpu_utilization = 0.0
        memory_usage = 0.0
        
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated(self.device) / 1e9
            gpu_utilization = (torch.cuda.memory_allocated(self.device) / 
                             torch.cuda.get_device_properties(self.device).total_memory) * 100
        
        # 컴포넌트별 통계 수집
        island_stats = self.island_manager.get_island_statistics()
        migration_stats = self.migration_policy.get_gpu_migration_statistics()
        map_elites_stats = self.map_elites.get_gpu_archive_statistics()
        
        metrics = GPUEvolutionMetrics(
            generation=generation,
            total_time=total_time,
            gpu_time=gpu_time,
            cpu_time=cpu_time,
            memory_usage=memory_usage,
            island_stats=island_stats,
            migration_stats=migration_stats,
            map_elites_stats=map_elites_stats,
            gpu_utilization=gpu_utilization
        )
        
        self.evolution_metrics.append(metrics)
    
    async def _log_progress(self, generation: int):
        """진행 상황 로깅"""
        if not self.evolution_metrics:
            return
        
        latest_metrics = self.evolution_metrics[-1]
        
        logger.info(f"Generation {generation} Progress:")
        logger.info(f"  Total time: {latest_metrics.total_time:.2f}s")
        logger.info(f"  GPU time: {latest_metrics.gpu_time:.2f}s ({latest_metrics.gpu_time/latest_metrics.total_time*100:.1f}%)")
        logger.info(f"  GPU memory: {latest_metrics.memory_usage:.2f} GB")
        logger.info(f"  GPU utilization: {latest_metrics.gpu_utilization:.1f}%")
        
        # Island 통계
        island_stats = latest_metrics.island_stats
        logger.info(f"  Islands: {island_stats['num_islands']}, Total population: {island_stats['total_population']}")
        
        # MAP-Elites 통계
        map_stats = latest_metrics.map_elites_stats
        logger.info(f"  Archive size: {map_stats['archive_size']}, Coverage: {map_stats['coverage']:.3f}")
        
        # 최고 피트니스
        best_fitness = 0.0
        for island_detail in island_stats.get('island_details', []):
            best_fitness = max(best_fitness, island_detail.get('best_fitness', 0.0))
        logger.info(f"  Best fitness: {best_fitness:.4f}")
    
    async def _collect_final_results(self, task: TaskDefinition) -> Dict[str, Any]:
        """최종 결과 수집"""
        logger.info("Collecting final GPU evolution results")
        
        # 모든 island에서 최고 프로그램들 수집
        best_programs = []
        all_programs = []
        
        for island in self.islands:
            if island.population:
                best_program = island.get_best_program()
                if best_program:
                    best_programs.append(best_program)
                all_programs.extend(island.population)
        
        # MAP-Elites에서 다양한 프로그램들 수집
        diverse_programs = await self.map_elites.get_diverse_programs(20)
        
        # 전체 최고 프로그램 선택
        overall_best = None
        if best_programs:
            overall_best = max(best_programs, 
                             key=lambda p: p.fitness_scores.get("score", p.fitness_scores.get("correctness", 0.0)))
        
        # 최종 통계
        final_stats = {
            "task_name": task.name,
            "total_generations": len(self.evolution_metrics),
            "total_programs_evaluated": len(all_programs),
            "best_programs": best_programs,
            "diverse_programs": diverse_programs,
            "overall_best_program": overall_best,
            "island_statistics": self.island_manager.get_island_statistics(),
            "migration_statistics": self.migration_policy.get_gpu_migration_statistics(),
            "map_elites_statistics": self.map_elites.get_gpu_archive_statistics(),
            "gpu_performance": self._calculate_gpu_performance_stats(),
            "evolution_metrics": self.evolution_metrics
        }
        
        return final_stats
    
    def _calculate_gpu_performance_stats(self) -> Dict[str, Any]:
        """GPU 성능 통계 계산"""
        if not self.evolution_metrics:
            return {}
        
        total_time = sum(m.total_time for m in self.evolution_metrics)
        total_gpu_time = sum(m.gpu_time for m in self.evolution_metrics)
        total_cpu_time = sum(m.cpu_time for m in self.evolution_metrics)
        avg_memory_usage = sum(m.memory_usage for m in self.evolution_metrics) / len(self.evolution_metrics)
        avg_gpu_utilization = sum(m.gpu_utilization for m in self.evolution_metrics) / len(self.evolution_metrics)
        
        gpu_efficiency = (total_gpu_time / total_time) * 100 if total_time > 0 else 0
        
        return {
            "total_evolution_time": total_time,
            "total_gpu_time": total_gpu_time,
            "total_cpu_time": total_cpu_time,
            "gpu_efficiency_percent": gpu_efficiency,
            "average_memory_usage_gb": avg_memory_usage,
            "average_gpu_utilization_percent": avg_gpu_utilization,
            "generations_per_second": len(self.evolution_metrics) / total_time if total_time > 0 else 0,
            "device_info": {
                "device": str(self.device),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
                "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A",
                "pytorch_version": torch.__version__
            }
        }
    
    def _cleanup_gpu_memory(self):
        """GPU 메모리 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 컴포넌트별 메모리 정리
        self.island_manager.cleanup_gpu_memory()
        self.migration_policy.cleanup_gpu_cache()
        self.map_elites.cleanup_gpu_memory()
        
        logger.debug("GPU memory cleaned up")
    
    async def get_real_time_stats(self) -> Dict[str, Any]:
        """실시간 GPU 통계 반환"""
        current_stats = {
            "current_generation": len(self.evolution_metrics),
            "total_islands": len(self.islands),
            "total_population": sum(len(island.population) for island in self.islands),
            "gpu_info": {}
        }
        
        if torch.cuda.is_available():
            current_stats["gpu_info"] = {
                "device": str(self.device),
                "memory_allocated_gb": torch.cuda.memory_allocated(self.device) / 1e9,
                "memory_reserved_gb": torch.cuda.memory_reserved(self.device) / 1e9,
                "memory_total_gb": torch.cuda.get_device_properties(self.device).total_memory / 1e9,
                "utilization_percent": (torch.cuda.memory_allocated(self.device) / 
                                      torch.cuda.get_device_properties(self.device).total_memory) * 100
            }
        
        # 컴포넌트별 실시간 통계
        current_stats.update({
            "island_manager_stats": self.island_manager.get_island_statistics(),
            "migration_policy_stats": self.migration_policy.get_gpu_migration_statistics(),
            "map_elites_stats": self.map_elites.get_gpu_archive_statistics()
        })
        
        return current_stats
    
    async def save_gpu_profile(self, filepath: str):
        """GPU 프로파일링 결과 저장"""
        if not self.enable_gpu_profiling or not self.evolution_metrics:
            logger.warning("GPU profiling not enabled or no metrics available")
            return
        
        try:
            import json
            
            profile_data = {
                "device_info": self._calculate_gpu_performance_stats()["device_info"],
                "evolution_metrics": [
                    {
                        "generation": m.generation,
                        "total_time": m.total_time,
                        "gpu_time": m.gpu_time,
                        "cpu_time": m.cpu_time,
                        "memory_usage": m.memory_usage,
                        "gpu_utilization": m.gpu_utilization
                    }
                    for m in self.evolution_metrics
                ],
                "performance_summary": self._calculate_gpu_performance_stats()
            }
            
            with open(filepath, 'w') as f:
                json.dump(profile_data, f, indent=2)
            
            logger.info(f"GPU profile saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving GPU profile: {e}")
    
    def get_gpu_memory_usage(self) -> Dict[str, float]:
        """현재 GPU 메모리 사용량 반환"""
        if not torch.cuda.is_available():
            return {"available": False}
        
        return {
            "available": True,
            "allocated_gb": torch.cuda.memory_allocated(self.device) / 1e9,
            "reserved_gb": torch.cuda.memory_reserved(self.device) / 1e9,
            "total_gb": torch.cuda.get_device_properties(self.device).total_memory / 1e9,
            "utilization_percent": (torch.cuda.memory_allocated(self.device) / 
                                  torch.cuda.get_device_properties(self.device).total_memory) * 100
        } 
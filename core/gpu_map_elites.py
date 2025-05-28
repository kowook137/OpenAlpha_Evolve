"""
GPU-accelerated MAP-Elites using PyTorch for RTX 3080.
Implements high-performance behavioral diversity maintenance using CUDA.
"""
import torch
import torch.nn.functional as F
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import hashlib

from core.interfaces import MAPElitesInterface, Program, BehaviorCharacteristics

logger = logging.getLogger(__name__)

@dataclass
class GPUBehaviorSpace:
    """GPU에서 처리되는 행동 공간"""
    dimensions: List[str]
    resolution: List[int]
    bounds: List[Tuple[float, float]]
    total_cells: int
    device: torch.device

class GPUMAPElites(MAPElitesInterface):
    """GPU-accelerated MAP-Elites for RTX 3080"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # GPU 설정
        self.device = self._setup_gpu()
        
        # GPU 최적화 설정
        self.batch_size = self.config.get("gpu_batch_size", 64)
        self.use_mixed_precision = self.config.get("use_mixed_precision", True)
        self.cache_behavior_vectors = self.config.get("cache_behavior_vectors", True)
        
        # GPU 행동 공간 설정
        self.gpu_behavior_space = self._setup_gpu_behavior_space()
        
        # GPU 메모리에 아카이브 텐서 할당
        self.archive_tensor = torch.zeros(
            self.gpu_behavior_space.total_cells, 
            device=self.device, 
            dtype=torch.float32
        )
        self.archive_programs = {}  # 셀 인덱스 -> Program 매핑
        
        # 행동 벡터 캐시 (GPU)
        self.behavior_cache = {}
        
        logger.info(f"GPUMAPElites initialized:")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Behavior space: {self.behavior_space}")
        logger.info(f"  Total cells: {self.gpu_behavior_space.total_cells}")
        logger.info(f"  GPU batch size: {self.batch_size}")
        logger.info(f"  Mixed precision: {self.use_mixed_precision}")
    
    def _setup_gpu(self) -> torch.device:
        """GPU 설정"""
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            torch.cuda.empty_cache()
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            return device
        else:
            logger.warning("CUDA not available, using CPU")
            return torch.device("cpu")
    
    def _setup_gpu_behavior_space(self) -> GPUBehaviorSpace:
        """GPU 행동 공간 설정"""
        dimensions = list(self.behavior_space.keys())
        resolution = [self.behavior_space[dim]["resolution"] for dim in dimensions]
        bounds = [self.behavior_space[dim]["bounds"] for dim in dimensions]
        total_cells = np.prod(resolution)
        
        return GPUBehaviorSpace(
            dimensions=dimensions,
            resolution=resolution,
            bounds=bounds,
            total_cells=total_cells,
            device=self.device
        )
    
    async def add_to_archive(self, programs: List[Program]) -> Dict[str, Any]:
        """GPU 가속 아카이브 추가"""
        if not programs:
            return {"added": 0, "replaced": 0, "total_archive_size": len(self.archive_programs)}
        
        logger.debug(f"Adding {len(programs)} programs to GPU archive")
        
        added_count = 0
        replaced_count = 0
        
        # 프로그램들을 배치로 처리
        program_batches = self._create_program_batches(programs)
        
        with torch.cuda.device(self.device):
            for batch in program_batches:
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        batch_results = await self._process_program_batch_gpu(batch)
                else:
                    batch_results = await self._process_program_batch_gpu(batch)
                
                added_count += batch_results["added"]
                replaced_count += batch_results["replaced"]
        
        result = {
            "added": added_count,
            "replaced": replaced_count,
            "total_archive_size": len(self.archive_programs),
            "gpu_memory_usage": torch.cuda.memory_allocated(self.device) / 1e9 if torch.cuda.is_available() else 0
        }
        
        logger.debug(f"GPU archive update: {result}")
        return result
    
    def _create_program_batches(self, programs: List[Program]) -> List[List[Program]]:
        """프로그램들을 GPU 처리에 최적화된 배치로 분할"""
        batches = []
        for i in range(0, len(programs), self.batch_size):
            batch = programs[i:i + self.batch_size]
            batches.append(batch)
        return batches
    
    async def _process_program_batch_gpu(self, batch: List[Program]) -> Dict[str, int]:
        """GPU에서 프로그램 배치 처리"""
        added = 0
        replaced = 0
        
        # 배치의 모든 프로그램에 대해 행동 특성 계산
        behavior_vectors = []
        fitness_scores = []
        
        for program in batch:
            # 행동 특성 계산 (캐시 확인)
            behavior_vector = await self._calculate_gpu_behavior_vector(program)
            behavior_vectors.append(behavior_vector)
            
            # 피트니스 점수 추출
            fitness = program.fitness_scores.get("score", program.fitness_scores.get("correctness", 0.0))
            fitness_scores.append(fitness)
        
        if not behavior_vectors:
            return {"added": 0, "replaced": 0}
        
        # GPU 텐서로 변환
        behavior_tensor = torch.stack(behavior_vectors)  # [batch_size, num_dimensions]
        fitness_tensor = torch.tensor(fitness_scores, device=self.device, dtype=torch.float32)
        
        # 배치의 모든 프로그램에 대해 셀 인덱스 계산
        cell_indices = self._calculate_gpu_cell_indices(behavior_tensor)
        
        # 각 프로그램을 아카이브에 추가/교체
        for i, (program, cell_idx, fitness) in enumerate(zip(batch, cell_indices, fitness_tensor)):
            cell_idx_item = cell_idx.item()
            
            if cell_idx_item in self.archive_programs:
                # 기존 프로그램과 비교
                current_fitness = self.archive_tensor[cell_idx_item].item()
                if fitness.item() > current_fitness:
                    # 더 좋은 프로그램으로 교체
                    self.archive_tensor[cell_idx_item] = fitness
                    self.archive_programs[cell_idx_item] = program
                    replaced += 1
            else:
                # 새로운 셀에 추가
                self.archive_tensor[cell_idx_item] = fitness
                self.archive_programs[cell_idx_item] = program
                added += 1
        
        return {"added": added, "replaced": replaced}
    
    async def _calculate_gpu_behavior_vector(self, program: Program) -> torch.Tensor:
        """GPU에서 행동 벡터 계산"""
        # 캐시 확인
        if self.cache_behavior_vectors:
            cache_key = self._get_program_cache_key(program)
            if cache_key in self.behavior_cache:
                return self.behavior_cache[cache_key]
        
        # 행동 특성 계산
        behavior_chars = await self.calculate_behavior_characteristics(program)
        
        # GPU 텐서로 변환
        behavior_values = []
        for dim in self.gpu_behavior_space.dimensions:
            value = getattr(behavior_chars, dim, 0.0)
            behavior_values.append(value)
        
        behavior_vector = torch.tensor(behavior_values, device=self.device, dtype=torch.float32)
        
        # 캐시에 저장
        if self.cache_behavior_vectors:
            self.behavior_cache[cache_key] = behavior_vector
        
        return behavior_vector
    
    def _get_program_cache_key(self, program: Program) -> str:
        """프로그램 캐시 키 생성"""
        content = f"{program.code}_{program.generation}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _calculate_gpu_cell_indices(self, behavior_tensor: torch.Tensor) -> torch.Tensor:
        """GPU에서 행동 벡터들의 셀 인덱스 계산"""
        batch_size = behavior_tensor.shape[0]
        cell_indices = torch.zeros(batch_size, device=self.device, dtype=torch.long)
        
        for i, (dim, resolution, bounds) in enumerate(zip(
            self.gpu_behavior_space.dimensions,
            self.gpu_behavior_space.resolution,
            self.gpu_behavior_space.bounds
        )):
            # 정규화 및 셀 인덱스 계산
            min_val, max_val = bounds
            normalized = (behavior_tensor[:, i] - min_val) / (max_val - min_val)
            normalized = torch.clamp(normalized, 0.0, 1.0)
            
            dim_indices = (normalized * (resolution - 1)).long()
            
            # 다차원 인덱스를 1차원으로 변환
            if i == 0:
                cell_indices = dim_indices
            else:
                cell_indices = cell_indices * resolution + dim_indices
        
        return cell_indices
    
    async def get_diverse_programs(self, count: int) -> List[Program]:
        """GPU 가속 다양한 프로그램 선택"""
        if not self.archive_programs or count <= 0:
            return []
        
        logger.debug(f"Selecting {count} diverse programs from GPU archive")
        
        with torch.cuda.device(self.device):
            # 아카이브에서 유효한 프로그램들의 피트니스 점수
            valid_indices = list(self.archive_programs.keys())
            if not valid_indices:
                return []
            
            # GPU에서 피트니스 기반 선택
            fitness_values = [self.archive_tensor[idx].item() for idx in valid_indices]
            fitness_tensor = torch.tensor(fitness_values, device=self.device, dtype=torch.float32)
            
            # 상위 프로그램들 선택
            num_select = min(count, len(valid_indices))
            _, top_indices = torch.topk(fitness_tensor, num_select)
            
            selected_programs = []
            for idx in top_indices:
                archive_idx = valid_indices[idx.item()]
                program = self.archive_programs[archive_idx]
                selected_programs.append(program)
        
        logger.debug(f"Selected {len(selected_programs)} diverse programs")
        return selected_programs
    
    async def calculate_behavior_characteristics(self, program: Program) -> BehaviorCharacteristics:
        """GPU 가속 행동 특성 계산"""
        # 기본 특성들을 GPU에서 병렬 계산
        with torch.cuda.device(self.device):
            # 코드 복잡도 (GPU에서 계산)
            code_complexity = await self._calculate_gpu_code_complexity(program.code)
            
            # 실행 시간 추정 (GPU에서 계산)
            execution_time = await self._estimate_gpu_execution_time(program.code)
            
            # 메모리 사용량 추정 (GPU에서 계산)
            memory_usage = await self._estimate_gpu_memory_usage(program.code)
            
            # 솔루션 접근법 분류 (GPU에서 계산)
            solution_approach = await self._classify_gpu_solution_approach(program.code)
        
        return BehaviorCharacteristics(
            code_complexity=code_complexity,
            execution_time=execution_time,
            memory_usage=memory_usage,
            solution_approach=solution_approach
        )
    
    async def _calculate_gpu_code_complexity(self, code: str) -> float:
        """GPU에서 코드 복잡도 계산"""
        # 코드 특성을 텐서로 변환하여 GPU에서 처리
        features = torch.tensor([
            len(code),
            code.count('\n'),
            code.count('for'),
            code.count('while'),
            code.count('if'),
            code.count('def'),
            code.count('class'),
            len(set(code.split()))  # 고유 토큰 수
        ], device=self.device, dtype=torch.float32)
        
        # GPU에서 가중 합계 계산
        weights = torch.tensor([0.1, 0.2, 0.3, 0.3, 0.2, 0.4, 0.5, 0.15], device=self.device)
        complexity = torch.sum(features * weights).item()
        
        # 정규화 (0-1 범위)
        return min(1.0, complexity / 100.0)
    
    async def _estimate_gpu_execution_time(self, code: str) -> float:
        """GPU에서 실행 시간 추정"""
        # 실행 시간에 영향을 주는 요소들을 GPU에서 계산
        factors = torch.tensor([
            code.count('for') * 2.0,
            code.count('while') * 3.0,
            code.count('range') * 1.5,
            code.count('sort') * 2.5,
            code.count('append') * 0.5,
            len(code) * 0.001
        ], device=self.device, dtype=torch.float32)
        
        estimated_time = torch.sum(factors).item()
        
        # 정규화 (0-1 범위)
        return min(1.0, estimated_time / 50.0)
    
    async def _estimate_gpu_memory_usage(self, code: str) -> float:
        """GPU에서 메모리 사용량 추정"""
        # 메모리 사용에 영향을 주는 요소들을 GPU에서 계산
        memory_factors = torch.tensor([
            code.count('list') * 2.0,
            code.count('dict') * 3.0,
            code.count('set') * 2.5,
            code.count('append') * 1.0,
            code.count('range') * 1.5,
            len(code.split()) * 0.1
        ], device=self.device, dtype=torch.float32)
        
        estimated_memory = torch.sum(memory_factors).item()
        
        # 정규화 (0-1 범위)
        return min(1.0, estimated_memory / 100.0)
    
    async def _classify_gpu_solution_approach(self, code: str) -> float:
        """GPU에서 솔루션 접근법 분류"""
        # 다양한 접근법의 특성을 GPU에서 계산
        approach_scores = torch.tensor([
            # 반복적 접근법
            (code.count('for') + code.count('while')) * 0.3,
            # 재귀적 접근법
            code.count('return') * 0.4 if 'def' in code else 0.0,
            # 함수형 접근법
            (code.count('map') + code.count('filter') + code.count('lambda')) * 0.5,
            # 객체지향 접근법
            (code.count('class') + code.count('self.')) * 0.6,
            # 수학적 접근법
            (code.count('math.') + code.count('**') + code.count('//')) * 0.4
        ], device=self.device, dtype=torch.float32)
        
        # 가장 강한 접근법의 점수 반환
        max_score = torch.max(approach_scores).item()
        
        # 정규화 (0-1 범위)
        return min(1.0, max_score / 10.0)
    
    def get_gpu_archive_statistics(self) -> Dict[str, Any]:
        """GPU 아카이브 통계 반환"""
        if not self.archive_programs:
            return {
                "archive_size": 0,
                "coverage": 0.0,
                "gpu_memory_usage": 0.0
            }
        
        with torch.cuda.device(self.device):
            # 아카이브 커버리지 계산
            coverage = len(self.archive_programs) / self.gpu_behavior_space.total_cells
            
            # 피트니스 통계 (GPU에서 계산)
            valid_indices = list(self.archive_programs.keys())
            fitness_values = [self.archive_tensor[idx].item() for idx in valid_indices]
            
            if fitness_values:
                fitness_tensor = torch.tensor(fitness_values, device=self.device, dtype=torch.float32)
                avg_fitness = torch.mean(fitness_tensor).item()
                max_fitness = torch.max(fitness_tensor).item()
                min_fitness = torch.min(fitness_tensor).item()
                std_fitness = torch.std(fitness_tensor).item()
            else:
                avg_fitness = max_fitness = min_fitness = std_fitness = 0.0
        
        stats = {
            "archive_size": len(self.archive_programs),
            "total_cells": self.gpu_behavior_space.total_cells,
            "coverage": coverage,
            "avg_fitness": avg_fitness,
            "max_fitness": max_fitness,
            "min_fitness": min_fitness,
            "std_fitness": std_fitness,
            "behavior_space": {
                "dimensions": self.gpu_behavior_space.dimensions,
                "resolution": self.gpu_behavior_space.resolution,
                "bounds": self.gpu_behavior_space.bounds
            },
            "gpu_info": {
                "device": str(self.device),
                "memory_allocated": torch.cuda.memory_allocated(self.device) / 1e9 if torch.cuda.is_available() else 0,
                "memory_reserved": torch.cuda.memory_reserved(self.device) / 1e9 if torch.cuda.is_available() else 0,
                "cache_size": len(self.behavior_cache)
            }
        }
        
        return stats
    
    def cleanup_gpu_memory(self):
        """GPU 메모리 정리"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.behavior_cache.clear()
            logger.debug("GPU MAP-Elites memory cleaned up")
    
    async def execute(self, action: str, **kwargs) -> Any:
        """GPU MAP-Elites 액션 실행"""
        try:
            if action == "add_programs":
                return await self.add_to_archive(kwargs["programs"])
            elif action == "get_diverse":
                return await self.get_diverse_programs(kwargs["count"])
            elif action == "calculate_behavior":
                return await self.calculate_behavior_characteristics(kwargs["program"])
            elif action == "get_stats":
                return self.get_gpu_archive_statistics()
            elif action == "cleanup":
                self.cleanup_gpu_memory()
                return True
            else:
                raise ValueError(f"Unknown action: {action}")
        except Exception as e:
            logger.error(f"Error executing GPU MAP-Elites action '{action}': {e}")
            raise 
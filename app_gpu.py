"""
AlphaEvolve GPU Application - RTX-3080 최적화
GPU Island Manager를 활용한 고성능 진화 시스템
"""
import asyncio
import logging
import time
import torch
from typing import Optional, Dict, Any

from core.interfaces import TaskDefinition
from task_manager.gpu_island_task_manager import create_gpu_task_manager, create_mols_gpu_manager
from config import settings

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AlphaEvolveGPUApp:
    """
    AlphaEvolve GPU 애플리케이션
    RTX-3080에 최적화된 대규모 병렬 진화 시스템
    """
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        
        if self.gpu_available:
            self.device = torch.device("cuda:0")
            logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            logger.warning("No GPU detected! Performance will be limited.")
            self.device = torch.device("cpu")

    def create_mols_task(self, mols_size: int = 8) -> TaskDefinition:
        """MOLS 과업 정의 생성"""
        return TaskDefinition(
            id=f"mols_{mols_size}x{mols_size}_gpu_task",
            description=f"Generate {mols_size}x{mols_size} Mutually Orthogonal Latin Squares",
            input_output_examples=[
                {
                    "input": f"Generate {mols_size}x{mols_size} MOLS",
                    "output": f"Two {mols_size}x{mols_size} orthogonal Latin squares"
                }
            ],
            evaluation_criteria={
                "primary_metric": "orthogonal_pairs",
                "secondary_metrics": ["correctness", "efficiency"],
                "goal": "maximize"
            },
            initial_code_prompt=f"""
Create a Python function that generates {mols_size}x{mols_size} Mutually Orthogonal Latin Squares using Galois Field (Finite Field) structure.

Requirements:
1. Generate two {mols_size}x{mols_size} Latin squares that are mutually orthogonal
2. A Latin square uses each number 0 to {mols_size-1} exactly once in each row and column
3. Two Latin squares are orthogonal if all ordered pairs (a,b) from corresponding positions are unique
4. Return the squares as lists of lists

IMPORTANT: Use EVOLVE-BLOCK structure for evolvable code sections.

Advanced Example using Galois Field GF(8) structure:
```python
# EVOLVE-BLOCK-START
import numpy as np
# EVOLVE-BLOCK-END

def solve():
    \"\"\"
    Generate {mols_size}x{mols_size} Mutually Orthogonal Latin Squares using Galois Field
    \"\"\"
    # EVOLVE-BLOCK-START
    # GF(8) 생성: x^3 + x + 1 => binary 1011
    POLY = 0b1011  # GF(2^3)에서의 환산용 다항식

    def gf8_add(a, b):
        \"\"\"GF(8)에서 덧셈은 단순 XOR\"\"\"
        return a ^ b

    def gf8_mult(a, b):
        \"\"\"GF(8)에서 곱셈 구현: 다항식 곱 후 mod (x^3 + x + 1)\"\"\"
        result = 0
        temp_a = a
        i = 0
        while temp_a:
            if temp_a & 1:
                result ^= (b << i)
            temp_a >>= 1
            i += 1
        # 결과 다항식을 POLY로 나눠서 환산
        if result & 0b10000:
            result ^= (POLY << 1)
        if result & 0b1000:
            result ^= POLY
        return result  # 결과는 0~7 사이 값

    # 비-trivial한 생성원 α = x 를 선택
    alpha = 0b010  # GF(8)에서 x에 해당 (binary)

    n = {mols_size}
    latin1 = [[gf8_add(r, c) for c in range(n)] for r in range(n)]
    latin2 = [[gf8_add(r, gf8_mult(alpha, c)) for c in range(n)] for r in range(n)]
    
    return latin1, latin2
    # EVOLVE-BLOCK-END
```

The sections between # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END will be evolved.
You can improve the Galois Field operations, try different generators, or explore other mathematical structures.
            """.strip()
        )

    async def run_mols_evolution(self, mols_size: int = 8) -> Dict[str, Any]:
        """MOLS 진화 실행"""
        logger.info(f"=== Starting {mols_size}x{mols_size} MOLS Evolution ===")
        
        # MOLS 과업 생성
        task_definition = self.create_mols_task(mols_size)
        
        # GPU Task Manager 생성
        task_manager = create_mols_gpu_manager(task_definition, mols_size)
        
        start_time = time.time()
        
        try:
            # 진화 실행
            best_programs = await task_manager.execute()
            
            execution_time = time.time() - start_time
            
            # 결과 분석
            results = {
                "task_id": task_definition.id,
                "mols_size": mols_size,
                "execution_time": execution_time,
                "best_programs": len(best_programs),
                "evolution_stats": task_manager.get_stats()
            }
            
            if best_programs:
                best = best_programs[0]
                results["best_solution"] = {
                    "id": best.id,
                    "generation": best.generation,
                    "fitness_scores": best.fitness_scores,
                    "code_length": len(best.code)
                }
                
                logger.info("=== Evolution Complete ===")
                logger.info(f"Best solution: {best.id}")
                logger.info(f"Fitness: {best.fitness_scores}")
                logger.info(f"Generation: {best.generation}")
                
                # MOLS 특화 결과 출력
                if "orthogonal_pairs" in best.fitness_scores:
                    pairs_found = best.fitness_scores["orthogonal_pairs"]
                    target_pairs = (mols_size * (mols_size - 1)) // 2
                    completeness = pairs_found / target_pairs
                    
                    logger.info(f"MOLS Progress: {pairs_found}/{target_pairs} pairs ({completeness:.1%})")
                    results["mols_progress"] = {
                        "pairs_found": pairs_found,
                        "target_pairs": target_pairs,
                        "completeness": completeness
                    }
                
                # 코드 출력 (처음 500자)
                logger.info(f"Generated code preview:\n{best.code[:500]}...")
            else:
                logger.warning("No solutions found!")
                results["best_solution"] = None
            
            return results
            
        except Exception as e:
            logger.error(f"Evolution failed: {e}", exc_info=True)
            return {
                "error": str(e),
                "execution_time": time.time() - start_time
            }

    def create_general_task(self, description: str, examples: list) -> TaskDefinition:
        """일반적인 코딩 과업 생성"""
        return TaskDefinition(
            id=f"general_task_{int(time.time())}",
            description=description,
            input_output_examples=examples,
            evaluation_criteria={
                "primary_metric": "correctness",
                "secondary_metrics": ["efficiency", "simplicity"],
                "goal": "maximize"
            },
            initial_code_prompt=f"Write a Python function to solve: {description}"
        )

    async def run_general_evolution(self, description: str, examples: list) -> Dict[str, Any]:
        """일반 코딩 문제 진화"""
        logger.info(f"=== Starting General Evolution ===")
        logger.info(f"Problem: {description}")
        
        # 과업 생성
        task_definition = self.create_general_task(description, examples)
        
        # GPU Task Manager 생성
        task_manager = create_gpu_task_manager(task_definition)
        
        start_time = time.time()
        
        try:
            # 진화 실행
            best_programs = await task_manager.execute()
            
            execution_time = time.time() - start_time
            
            results = {
                "task_id": task_definition.id,
                "description": description,
                "execution_time": execution_time,
                "best_programs": len(best_programs),
                "evolution_stats": task_manager.get_stats()
            }
            
            if best_programs:
                best = best_programs[0]
                results["best_solution"] = {
                    "id": best.id,
                    "generation": best.generation,
                    "fitness_scores": best.fitness_scores,
                    "code": best.code
                }
                
                logger.info("=== Evolution Complete ===")
                logger.info(f"Best solution fitness: {best.fitness_scores}")
                logger.info(f"Generated code:\n{best.code}")
            
            return results
            
        except Exception as e:
            logger.error(f"Evolution failed: {e}", exc_info=True)
            return {"error": str(e), "execution_time": time.time() - start_time}

    def print_gpu_info(self):
        """GPU 정보 출력"""
        print("=== GPU 정보 ===")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"GPU 개수: {torch.cuda.device_count()}")
            print(f"현재 GPU: {torch.cuda.current_device()}")
            print(f"GPU 이름: {torch.cuda.get_device_name(0)}")
            print(f"GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"CUDA 버전: {torch.version.cuda}")
        else:
            print("GPU를 사용할 수 없습니다.")
        
        print("================")


async def main():
    """메인 실행 함수"""
    app = AlphaEvolveGPUApp()
    
    # GPU 정보 출력
    app.print_gpu_info()
    
    # MOLS 문제 해결
    print("\n🚀 MOLS 8x8 문제 실행 중...")
    mols_results = await app.run_mols_evolution(mols_size=8)
    
    print(f"\n📊 MOLS 결과:")
    print(f"실행 시간: {mols_results.get('execution_time', 0):.2f}초")
    
    if 'best_solution' in mols_results and mols_results['best_solution']:
        print(f"최고 적합도: {mols_results['best_solution']['fitness_scores']}")
        if 'mols_progress' in mols_results:
            progress = mols_results['mols_progress']
            print(f"MOLS 진행도: {progress['completeness']:.1%} "
                  f"({progress['pairs_found']}/{progress['target_pairs']} 쌍)")
    
    # 간단한 코딩 문제도 테스트
    print("\n🔍 간단한 코딩 문제 테스트...")
    
    simple_task_results = await app.run_general_evolution(
        description="리스트의 합을 구하는 함수를 작성하세요",
        examples=[
            {"input": [1, 2, 3], "output": 6},
            {"input": [], "output": 0},
            {"input": [10, -5, 3], "output": 8}
        ]
    )
    
    print(f"\n📈 간단한 문제 결과:")
    print(f"실행 시간: {simple_task_results.get('execution_time', 0):.2f}초")
    if 'best_solution' in simple_task_results and simple_task_results['best_solution']:
        print(f"최고 적합도: {simple_task_results['best_solution']['fitness_scores']}")


if __name__ == "__main__":
    # asyncio 실행
    asyncio.run(main()) 
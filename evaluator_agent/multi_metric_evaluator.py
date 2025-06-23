"""
Multi-Metric Evaluator for AlphaEvolve
다차원 평가 시스템 - 코드 품질, 효율성, 복잡도 등을 종합 평가
"""
import ast
import time
import logging
import asyncio
import radon.complexity as radon
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass

from core.interfaces import Program, TaskDefinition
from config import settings

logger = logging.getLogger(__name__)

@dataclass
class CodeMetrics:
    """코드 메트릭 데이터 클래스"""
    cyclomatic_complexity: float
    lines_of_code: int
    execution_time_ms: float
    memory_usage_mb: float
    operation_count: int
    readability_score: float
    mathematical_elegance: float
    efficiency_score: float

class MultiMetricEvaluator:
    """다차원 평가 시스템"""
    
    def __init__(self, llm_provider: Optional[str] = None):
        self.llm_provider = llm_provider or settings.EVALUATION_KEY
        self.weights = {
            'correctness': 0.4,      # 기능적 정확성 (가장 중요)
            'efficiency': 0.2,       # 연산 효율성
            'complexity': 0.15,      # 코드 복잡도 (낮을수록 좋음)
            'readability': 0.1,      # 가독성
            'elegance': 0.1,         # 수학적 우아함
            'performance': 0.05      # 실행 성능
        }
        
    def calculate_cyclomatic_complexity(self, code: str) -> float:
        """순환 복잡도 계산"""
        try:
            tree = ast.parse(code)
            complexity_blocks = radon.cc_visit(tree)
            
            if not complexity_blocks:
                return 1.0
            
            total_complexity = sum(block.complexity for block in complexity_blocks)
            avg_complexity = total_complexity / len(complexity_blocks)
            
            # 정규화: 1-10 범위로 변환
            normalized = min(avg_complexity, 10.0)
            return normalized
            
        except Exception as e:
            logger.warning(f"복잡도 계산 실패: {e}")
            return 5.0  # 기본값
    
    def count_operations(self, code: str) -> int:
        """코드 내 연산 횟수 추정"""
        try:
            tree = ast.parse(code)
            operation_count = 0
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare)):
                    operation_count += 1
                elif isinstance(node, ast.Call):
                    operation_count += 2  # 함수 호출은 더 비싼 연산
                elif isinstance(node, (ast.For, ast.While)):
                    operation_count += 5  # 반복문은 복합 연산
                elif isinstance(node, ast.ListComp):
                    operation_count += 3  # 리스트 컴프리헨션
            
            return operation_count
            
        except Exception as e:
            logger.warning(f"연산 횟수 계산 실패: {e}")
            return 100  # 기본값
    
    def calculate_lines_of_code(self, code: str) -> int:
        """유효 코드 라인 수 계산 (주석, 빈 줄 제외)"""
        lines = code.split('\n')
        effective_lines = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#'):
                effective_lines += 1
        
        return effective_lines
    
    async def evaluate_readability(self, code: str) -> float:
        """LLM을 통한 코드 가독성 평가"""
        try:
            from code_generator.agent import CodeGeneratorAgent
            
            generator = CodeGeneratorAgent()
            
            prompt = f"""
Rate the readability of this Python code on a scale of 1-10, where:
- 1 = Very hard to read (poor naming, no structure)
- 5 = Average readability
- 10 = Excellent readability (clear naming, good structure, well-organized)

Consider:
- Variable and function naming
- Code structure and organization
- Use of comments and docstrings
- Overall clarity

Code to evaluate:
```python
{code}
```

Respond with only a single number between 1 and 10.
"""
            
            response = await generator.generate_code(
                prompt=prompt,
                temperature=0.3,
                output_format="code"
            )
            
            # 숫자 추출
            try:
                score = float(response.strip())
                return max(1.0, min(10.0, score))
            except:
                return 5.0  # 기본값
                
        except Exception as e:
            logger.warning(f"가독성 평가 실패: {e}")
            return 5.0
    
    async def evaluate_mathematical_elegance(self, code: str, task: TaskDefinition) -> float:
        """LLM을 통한 수학적 우아함 평가"""
        try:
            from code_generator.agent import CodeGeneratorAgent
            
            generator = CodeGeneratorAgent()
            
            prompt = f"""
Rate the mathematical elegance of this algorithm for {task.description} on a scale of 1-10, where:
- 1 = Brute force, no mathematical insight
- 5 = Standard algorithmic approach
- 10 = Highly elegant, uses advanced mathematical concepts

Consider:
- Use of mathematical structures (group theory, finite fields, etc.)
- Algorithmic sophistication
- Theoretical foundations
- Innovation in approach

Code to evaluate:
```python
{code}
```

Respond with only a single number between 1 and 10.
"""
            
            response = await generator.generate_code(
                prompt=prompt,
                temperature=0.3,
                output_format="code"
            )
            
            try:
                score = float(response.strip())
                return max(1.0, min(10.0, score))
            except:
                return 5.0
                
        except Exception as e:
            logger.warning(f"수학적 우아함 평가 실패: {e}")
            return 5.0
    
    def calculate_efficiency_score(self, operation_count: int, lines_of_code: int) -> float:
        """효율성 점수 계산 (연산 대비 코드 길이)"""
        if lines_of_code == 0:
            return 1.0
        
        # 연산 밀도 계산 (라인당 연산 수)
        operation_density = operation_count / lines_of_code
        
        # 정규화: 높은 밀도일수록 비효율적
        # 적절한 밀도는 2-5 정도로 가정
        if operation_density <= 2:
            return 10.0
        elif operation_density <= 5:
            return 8.0 - (operation_density - 2) * 2
        else:
            return max(1.0, 8.0 - (operation_density - 5) * 0.5)
    
    async def evaluate_comprehensive(
        self, 
        program: Program, 
        task: TaskDefinition,
        execution_time_ms: float,
        memory_usage_mb: float,
        correctness_score: float
    ) -> Dict[str, Any]:
        """종합적 다차원 평가"""
        
        # 기본 메트릭 계산
        complexity = self.calculate_cyclomatic_complexity(program.code)
        lines_of_code = self.calculate_lines_of_code(program.code)
        operation_count = self.count_operations(program.code)
        efficiency_score = self.calculate_efficiency_score(operation_count, lines_of_code)
        
        # LLM 기반 평가 (병렬 실행)
        readability_task = self.evaluate_readability(program.code)
        elegance_task = self.evaluate_mathematical_elegance(program.code, task)
        
        readability_score, elegance_score = await asyncio.gather(
            readability_task, elegance_task
        )
        
        # 메트릭 정규화 (1-10 스케일)
        normalized_metrics = {
            'correctness': correctness_score * 10,  # 0-1 -> 0-10
            'efficiency': efficiency_score,         # 이미 1-10
            'complexity': max(1, 11 - complexity), # 복잡도는 낮을수록 좋음
            'readability': readability_score,       # 이미 1-10
            'elegance': elegance_score,            # 이미 1-10
            'performance': max(1, 11 - min(10, execution_time_ms / 100))  # 시간 기반
        }
        
        # 가중 평균 계산
        weighted_score = sum(
            normalized_metrics[metric] * self.weights[metric]
            for metric in self.weights.keys()
        )
        
        # 상세 메트릭 정보
        detailed_metrics = {
            'cyclomatic_complexity': complexity,
            'lines_of_code': lines_of_code,
            'operation_count': operation_count,
            'execution_time_ms': execution_time_ms,
            'memory_usage_mb': memory_usage_mb,
            'readability_score': readability_score,
            'mathematical_elegance': elegance_score,
            'efficiency_score': efficiency_score,
            'normalized_metrics': normalized_metrics,
            'weighted_total_score': weighted_score,
            'metric_weights': self.weights
        }
        
        logger.info(f"다차원 평가 완료 - 총점: {weighted_score:.2f}/10")
        logger.info(f"세부 점수: {normalized_metrics}")
        
        return detailed_metrics
    
    def compare_programs(self, prog1_metrics: Dict, prog2_metrics: Dict) -> Dict[str, Any]:
        """두 프로그램의 메트릭 비교"""
        comparison = {}
        
        for metric in self.weights.keys():
            score1 = prog1_metrics['normalized_metrics'][metric]
            score2 = prog2_metrics['normalized_metrics'][metric]
            
            comparison[f"{metric}_improvement"] = score2 - score1
            comparison[f"{metric}_improvement_pct"] = ((score2 - score1) / score1) * 100 if score1 > 0 else 0
        
        total_improvement = prog2_metrics['weighted_total_score'] - prog1_metrics['weighted_total_score']
        comparison['total_improvement'] = total_improvement
        comparison['is_better_overall'] = total_improvement > 0
        
        return comparison

# 사용 예시
async def example_usage():
    """다차원 평가 시스템 사용 예시"""
    evaluator = MultiMetricEvaluator()
    
    # 가상의 프로그램과 태스크
    program = Program(
        id="test_program",
        code="""
def solve():
    # GF(8) 구현
    n = 8
    latin1 = [[i ^ j for j in range(n)] for i in range(n)]
    latin2 = [[i ^ (j * 2) for j in range(n)] for i in range(n)]
    return latin1, latin2
        """,
        generation=1
    )
    
    task = TaskDefinition(
        id="mols_8x8",
        description="Generate 8x8 Mutually Orthogonal Latin Squares",
        input_output_examples=[],
        evaluation_criteria={}
    )
    
    # 종합 평가 실행
    metrics = await evaluator.evaluate_comprehensive(
        program=program,
        task=task,
        execution_time_ms=50.0,
        memory_usage_mb=2.5,
        correctness_score=0.95
    )
    
    print("📊 다차원 평가 결과:")
    print(f"총점: {metrics['weighted_total_score']:.2f}/10")
    print(f"세부 점수: {metrics['normalized_metrics']}")

if __name__ == "__main__":
    asyncio.run(example_usage()) 
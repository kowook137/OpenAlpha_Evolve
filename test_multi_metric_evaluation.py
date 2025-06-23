"""
다차원 평가 시스템 테스트 스크립트
AlphaEvolve의 새로운 Multi-Metric Evaluation 시스템을 테스트합니다
"""
import asyncio
import logging
from typing import List

from core.interfaces import Program, TaskDefinition
from evaluator_agent.multi_metric_evaluator import MultiMetricEvaluator

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_programs() -> List[Program]:
    """다양한 복잡도의 테스트 프로그램들 생성"""
    
    # 1. 간단한 브루트 포스 접근법
    simple_code = """
def solve():
    # 간단한 브루트 포스 접근
    n = 8
    square1 = []
    square2 = []
    
    for i in range(n):
        row1 = []
        row2 = []
        for j in range(n):
            row1.append((i + j) % n)
            row2.append((i * 2 + j) % n)
        square1.append(row1)
        square2.append(row2)
    
    return square1, square2
"""
    
    # 2. 유한체 기반 우아한 접근법 (제공해주신 코드)
    elegant_code = """
import numpy as np

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

def solve():
    # 비-trivial한 생성원 α = x 를 선택
    alpha = 0b010  # GF(8)에서 x에 해당 (binary)
    
    n = 8
    latin1 = [[gf8_add(r, c) for c in range(n)] for r in range(n)]
    latin2 = [[gf8_add(r, gf8_mult(alpha, c)) for c in range(n)] for r in range(n)]
    
    return latin1, latin2
"""
    
    # 3. 복잡하지만 비효율적인 접근법
    complex_code = """
import itertools
import random
import time

def solve():
    \"\"\"복잡하지만 비효율적인 MOLS 생성\"\"\"
    n = 8
    
    # 복잡한 초기화
    base_values = list(range(n))
    permutations = list(itertools.permutations(base_values))
    
    # 비효율적인 중첩 반복문
    best_square1 = None
    best_square2 = None
    best_score = -1
    
    for attempt in range(100):  # 많은 시도
        square1 = []
        square2 = []
        
        for i in range(n):
            row1 = []
            row2 = []
            for j in range(n):
                # 복잡한 계산
                val1 = 0
                val2 = 0
                for k in range(10):  # 불필요한 반복
                    temp = (i + j + k) % n
                    val1 ^= temp
                    val2 ^= (temp * 2) % n
                
                row1.append(val1 % n)
                row2.append(val2 % n)
            
            square1.append(row1)
            square2.append(row2)
        
        # 무의미한 점수 계산
        score = sum(sum(row) for row in square1) + sum(sum(row) for row in square2)
        if score > best_score:
            best_score = score
            best_square1 = square1
            best_square2 = square2
    
    return best_square1, best_square2
"""
    
    # 4. 매우 간결한 접근법
    concise_code = """
def solve():
    n = 8
    return [[(i+j)%n for j in range(n)] for i in range(n)], [[(i*2+j)%n for j in range(n)] for i in range(n)]
"""
    
    programs = [
        Program(id="simple_bruteforce", code=simple_code, generation=1),
        Program(id="elegant_galois_field", code=elegant_code, generation=1),
        Program(id="complex_inefficient", code=complex_code, generation=1),
        Program(id="concise_oneliner", code=concise_code, generation=1)
    ]
    
    return programs

async def test_multi_metric_evaluation():
    """다차원 평가 시스템 테스트"""
    print("🧪 다차원 평가 시스템 테스트 시작\n")
    
    # 테스트 프로그램들 생성
    programs = create_test_programs()
    
    # 태스크 정의
    task = TaskDefinition(
        id="mols_8x8",
        description="Generate 8x8 Mutually Orthogonal Latin Squares",
        input_output_examples=[],
        evaluation_criteria={}
    )
    
    # 다차원 평가자 초기화
    evaluator = MultiMetricEvaluator()
    
    print("📊 각 프로그램별 다차원 평가 결과:\n")
    
    results = []
    
    for program in programs:
        print(f"🔍 평가 중: {program.id}")
        print(f"코드 길이: {len(program.code)} 문자")
        
        try:
            # 가상의 실행 결과 (실제로는 evaluator_agent에서 실행)
            execution_time = {
                "simple_bruteforce": 45.0,
                "elegant_galois_field": 25.0,
                "complex_inefficient": 150.0,
                "concise_oneliner": 20.0
            }.get(program.id, 50.0)
            
            memory_usage = {
                "simple_bruteforce": 2.1,
                "elegant_galois_field": 2.5,
                "complex_inefficient": 4.2,
                "concise_oneliner": 1.8
            }.get(program.id, 2.0)
            
            correctness = {
                "simple_bruteforce": 0.85,
                "elegant_galois_field": 0.95,
                "complex_inefficient": 0.75,
                "concise_oneliner": 0.90
            }.get(program.id, 0.8)
            
            # 다차원 평가 실행
            metrics = await evaluator.evaluate_comprehensive(
                program=program,
                task=task,
                execution_time_ms=execution_time,
                memory_usage_mb=memory_usage,
                correctness_score=correctness
            )
            
            results.append((program.id, metrics))
            
            print(f"✅ 종합 점수: {metrics['weighted_total_score']:.2f}/10")
            print(f"   - 정확성: {metrics['normalized_metrics']['correctness']:.1f}/10")
            print(f"   - 효율성: {metrics['normalized_metrics']['efficiency']:.1f}/10")
            print(f"   - 복잡도: {metrics['normalized_metrics']['complexity']:.1f}/10")
            print(f"   - 가독성: {metrics['normalized_metrics']['readability']:.1f}/10")
            print(f"   - 우아함: {metrics['normalized_metrics']['elegance']:.1f}/10")
            print(f"   - 성능: {metrics['normalized_metrics']['performance']:.1f}/10")
            print(f"   - 코드 라인 수: {metrics['lines_of_code']}")
            print(f"   - 순환 복잡도: {metrics['cyclomatic_complexity']:.1f}")
            print(f"   - 연산 횟수: {metrics['operation_count']}")
            print()
            
        except Exception as e:
            print(f"❌ 평가 실패: {e}\n")
    
    # 결과 비교 및 순위
    print("🏆 프로그램 순위 (종합 점수 기준):")
    print("=" * 60)
    
    sorted_results = sorted(results, key=lambda x: x[1]['weighted_total_score'], reverse=True)
    
    for i, (program_id, metrics) in enumerate(sorted_results, 1):
        print(f"{i}. {program_id}")
        print(f"   종합 점수: {metrics['weighted_total_score']:.2f}/10")
        print(f"   특징: ", end="")
        
        # 각 프로그램의 강점 분석
        strengths = []
        if metrics['normalized_metrics']['correctness'] >= 9:
            strengths.append("높은 정확성")
        if metrics['normalized_metrics']['efficiency'] >= 8:
            strengths.append("우수한 효율성")
        if metrics['normalized_metrics']['complexity'] >= 8:
            strengths.append("낮은 복잡도")
        if metrics['normalized_metrics']['elegance'] >= 8:
            strengths.append("수학적 우아함")
        if metrics['normalized_metrics']['readability'] >= 8:
            strengths.append("높은 가독성")
        
        print(", ".join(strengths) if strengths else "개선 필요")
        print()
    
    # 프로그램 간 비교
    if len(sorted_results) >= 2:
        best = sorted_results[0]
        second = sorted_results[1]
        
        comparison = evaluator.compare_programs(second[1], best[1])
        
        print("🔄 1위와 2위 프로그램 비교:")
        print(f"{best[0]} vs {second[0]}")
        print(f"총점 개선: {comparison['total_improvement']:.2f}점")
        print(f"정확성 개선: {comparison['correctness_improvement']:.2f}점")
        print(f"효율성 개선: {comparison['efficiency_improvement']:.2f}점")
        print(f"복잡도 개선: {comparison['complexity_improvement']:.2f}점")

async def test_metric_weights():
    """메트릭 가중치 조정 테스트"""
    print("\n🎛️ 메트릭 가중치 조정 테스트")
    print("=" * 50)
    
    # 다른 가중치 설정들
    weight_configs = [
        {
            'name': '정확성 중심',
            'weights': {
                'correctness': 0.6,
                'efficiency': 0.15,
                'complexity': 0.1,
                'readability': 0.05,
                'elegance': 0.05,
                'performance': 0.05
            }
        },
        {
            'name': '효율성 중심',
            'weights': {
                'correctness': 0.3,
                'efficiency': 0.3,
                'complexity': 0.2,
                'readability': 0.05,
                'elegance': 0.05,
                'performance': 0.1
            }
        },
        {
            'name': '우아함 중심',
            'weights': {
                'correctness': 0.3,
                'efficiency': 0.15,
                'complexity': 0.15,
                'readability': 0.15,
                'elegance': 0.2,
                'performance': 0.05
            }
        }
    ]
    
    programs = create_test_programs()
    task = TaskDefinition(id="test", description="Test task", input_output_examples=[], evaluation_criteria={})
    
    for config in weight_configs:
        print(f"\n📊 {config['name']} 가중치 적용:")
        
        evaluator = MultiMetricEvaluator()
        evaluator.weights = config['weights']
        
        # 간단한 테스트 (첫 번째 프로그램만)
        program = programs[1]  # elegant_galois_field
        
        metrics = await evaluator.evaluate_comprehensive(
            program=program,
            task=task,
            execution_time_ms=25.0,
            memory_usage_mb=2.5,
            correctness_score=0.95
        )
        
        print(f"종합 점수: {metrics['weighted_total_score']:.2f}/10")
        print(f"가중치: {config['weights']}")

if __name__ == "__main__":
    asyncio.run(test_multi_metric_evaluation())
    asyncio.run(test_metric_weights()) 
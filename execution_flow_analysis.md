# Alpha Evolve 실행 흐름 분석

## 🚀 실행 시작: MOLS_generate_main.py

### 1단계: 초기화
```
MOLS_generate_main.py
├── config/settings.py (설정 로드)
├── core/interfaces.py (TaskDefinition 생성)
├── mols_task/evaluator_agent/agent.py (평가자 초기화)
└── task_manager/island_task_manager.py (Island 관리자 초기화)
```

### 2단계: Island Task Manager 초기화
```
task_manager/island_task_manager.py
├── core/island_manager.py (Island 관리)
├── core/migration_policy.py (이주 정책)
├── core/map_elites.py (MAP-Elites 아카이브)
├── prompt_designer/agent.py (프롬프트 설계)
├── code_generator/agent.py (코드 생성)
├── evaluator_agent/agent.py (일반 평가자 - 사용 안됨)
├── mols_task/evaluator_agent/agent.py (MOLS 전용 평가자 - 실제 사용)
├── database_agent/agent.py (데이터베이스)
└── selection_controller/agent.py (선택 제어)
```

### 3단계: 진화 사이클
```
IslandTaskManager.manage_evolutionary_cycle()
├── initialize_population() → 초기 개체군 생성
│   ├── PromptDesignerAgent.design_initial_prompt()
│   └── CodeGeneratorAgent.generate_code() → Gemini API 호출
├── evaluate_islands() → 평가
│   └── mols_task/evaluator_agent/agent.py.evaluate_program()
│       └── mols_task/evaluation.py.evaluate() ← 당신의 평가 함수!
├── evolve_islands_generation() → 진화
│   ├── IslandManager.evolve_islands_parallel()
│   ├── generate_offspring() → 자손 생성
│   │   ├── PromptDesignerAgent.design_mutation_prompt()
│   │   └── CodeGeneratorAgent.generate_code() → Gemini API 호출
│   └── evaluate_program() → 자손 평가
└── perform_migration() → 이주
    └── MigrationPolicy.perform_migration()
```

## 📊 평가 시스템

### 실제 사용되는 평가 체인:
```
Program.code (생성된 코드)
↓
mols_task/evaluator_agent/agent.py._execute_safely()
↓ (코드 실행하여 MOLS 생성)
squares = namespace['generate_MOLS_3']()
↓
mols_task/evaluation.py.evaluate(squares)
↓ (당신이 작성한 평가 함수)
{
  "score": 0.xx,
  "latin_score": 0.xx,
  "orthogonality_score": 0.xx,
  ...
}
```

### 평가 함수 의존성:
```
mols_task/evaluation.py
└── mols_task/reference_output.py (EXAMPLE_OUTPUT)
```

## 🧬 Import 진화 분석

### TaskDefinition에서 허용된 Import:
```python
# MOLS_generate_main.py 라인 120
allowed_imports=["random", "itertools", "numpy"]
```

### 실제 진화되는 코드 블록:
```python
# mols_task/program.py - 초기 템플릿
# EVOLVE-BLOCK-START
import random
import itertools
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
def generate_MOLS_3():
    # 이 부분이 진화됨
    square1 = [[(i + j) % 3 for j in range(3)] for i in range(3)]
    square2 = [[(i + 2*j) % 3 for j in range(3)] for i in range(3)]
    return [square1, square2]
# EVOLVE-BLOCK-END
```

### Import 진화 여부:
❌ **Import 부분은 진화하지 않습니다!**

**이유:**
1. `allowed_imports`는 **제약 조건**으로만 사용
2. 실제 코드 생성 시 LLM이 **함수 구현부만** 진화
3. Import 문은 **고정된 템플릿**으로 유지

### 진화되는 부분:
✅ **함수 구현부만 진화됩니다:**
- 알고리즘 로직
- 데이터 구조 사용법
- 계산 방식
- 허용된 라이브러리 활용법

## 🔄 코드 생성 과정

### 1. 초기 프롬프트:
```
PromptDesignerAgent.design_initial_prompt()
→ "3x3 MOLS를 생성하는 generate_MOLS_3() 함수를 작성하세요"
→ "사용 가능한 라이브러리: random, itertools, numpy"
```

### 2. 변이 프롬프트:
```
PromptDesignerAgent.design_mutation_prompt()
→ "기존 코드를 개선하여 더 나은 MOLS를 생성하세요"
→ "평가 결과: latin_score=0.8, orthogonality_score=0.3"
→ "diff 형식으로 수정사항을 제공하세요"
```

### 3. 코드 생성:
```
CodeGeneratorAgent.generate_code()
→ Gemini API 호출
→ 새로운 generate_MOLS_3() 함수 구현 반환
```

## 📈 성능 모니터링

### Island 통계:
- 3개 Island, 각각 6개 개체
- 3세대마다 이주 발생
- 각 Island는 다른 진화 전략 사용

### MAP-Elites 아카이브:
- 행동 차원: ["code_complexity", "execution_time", "solution_approach"]
- 다양성 유지를 위한 아카이브

### 최종 결과:
- 최고 성능 프로그램들 반환
- 완벽한 3x3 MOLS 발견 시 조기 종료 
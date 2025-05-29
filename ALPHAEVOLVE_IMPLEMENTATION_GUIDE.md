# 🚀 AlphaEvolve Implementation Guide

## 📋 **Overview**

이 코드베이스는 Google DeepMind의 **AlphaEvolve** 논문을 기반으로 하여 3x3 MOLS(Mutually Orthogonal Latin Squares) 문제를 해결하는 진화 알고리즘 시스템입니다.

## 🎯 **Key AlphaEvolve Features Implemented**

### ✅ **1. EVOLVE-BLOCK Based Evolution**
- **파일**: `core/evolve_block_parser.py`
- **기능**: 코드의 특정 부분만 진화하도록 `# EVOLVE-BLOCK-START`와 `# EVOLVE-BLOCK-END` 주석 지원
- **적용**: Import 문, 함수, 설정 파라미터 등을 개별적으로 진화 가능

### ✅ **2. Multi-Component Algorithm Evolution**
- **파일**: `mols_task/program.py`
- **구조**:
  ```python
  # EVOLVE-BLOCK-START (imports)
  import random, itertools, numpy as np
  # EVOLVE-BLOCK-END
  
  # EVOLVE-BLOCK-START (base_square_generator)
  def base_square_generator(size: int = 3):
      # Base Latin square generation strategy
  # EVOLVE-BLOCK-END
  
  # EVOLVE-BLOCK-START (orthogonality_transformer)
  def orthogonality_transformer(base_square, transformation_id):
      # Transform base square to create orthogonal partner
  # EVOLVE-BLOCK-END
  
  # EVOLVE-BLOCK-START (pair_optimizer)
  def pair_optimizer(square1, square2):
      # Optimize the pair for better orthogonality
  # EVOLVE-BLOCK-END
  
  # EVOLVE-BLOCK-START (algorithm_config)
  def algorithm_config():
      # Configuration parameters
  # EVOLVE-BLOCK-END
  
  # EVOLVE-BLOCK-START (main algorithm)
  def generate_MOLS_3():
      # Main orchestrating function
  # EVOLVE-BLOCK-END
  ```

### ✅ **3. Gemini Flash + Pro Ensemble**
- **파일**: `code_generator/agent.py`
- **기능**: 
  - Flash 모델: 빠른 탐색 및 다양성 (temperature=0.9)
  - Pro 모델: 깊이 있는 분석 및 정제 (temperature=0.3)
  - 두 결과를 비교하여 최적 선택

### ✅ **4. Advanced Diff-Based Mutations**
- **기능**: EVOLVE-BLOCK 인식 diff 적용
- **포맷**:
  ```
  <<<<<<< SEARCH
  # EVOLVE-BLOCK-START
  [기존 코드]
  # EVOLVE-BLOCK-END
  =======
  # EVOLVE-BLOCK-START
  [개선된 코드]
  # EVOLVE-BLOCK-END
  >>>>>>> REPLACE
  ```

### ✅ **5. Multi-Island Evolution with Migration**
- **파일**: `task_manager/island_task_manager.py`
- **기능**: 
  - 4개의 독립적인 island에서 병렬 진화
  - 주기적인 최고 개체 migration
  - 다양한 진화 전략 (exploitation, exploration, balanced)

## 🔧 **Setup & Installation**

### 1. **환경 설정**
```bash
# 프로젝트 디렉토리로 이동
cd OpenAlpha_Evolve

# Python 가상환경 생성 (권장)
python -m venv alphaevolve_env
source alphaevolve_env/bin/activate  # Linux/Mac
# 또는
alphaevolve_env\Scripts\activate     # Windows

# 필수 패키지 설치
pip install google-generativeai python-dotenv
```

### 2. **API 키 설정**
```bash
# .env 파일 생성
echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
```

### 3. **설정 확인**
- `config/settings.py`에서 `ALPHA_EVOLVE_MODE = True` 확인
- CPU 모드: `ENABLE_GPU_ACCELERATION = False` 설정됨

## 🚀 **Usage**

### **실행 방법**
```bash
cd OpenAlpha_Evolve
python MOLS_generate_main.py
```

### **실행 과정**
1. **초기화**: 4개 island에 각각 8개 개체 생성
2. **진화 사이클**: 40세대 동안 반복
   - 각 island에서 독립적 진화
   - EVOLVE-BLOCK 기반 변이
   - Flash+Pro 앙상블 코드 생성
   - 4세대마다 island 간 migration
3. **평가**: 3x3 MOLS 품질 평가
   - Latin square 유효성
   - 직교성 품질
   - 전체 성능 점수

## 📊 **Expected Output**

```
🏝️ Island Evolution Progress:
Generation 10/40
- Island 0 (EXPLOITATION): Best=0.89, Avg=0.72, Pop=8
- Island 1 (EXPLORATION): Best=0.85, Avg=0.68, Pop=8  
- Island 2 (BALANCED): Best=0.92, Avg=0.75, Pop=8
- Island 3 (RANDOM): Best=0.78, Avg=0.64, Pop=8

🔄 Migration Event: Elite migration completed between islands

🎯 BEST SOLUTION FOUND (Generation 35):
Score: 0.96 | Latin: 1.00 | Orthogonality: 0.89

📄 Generated Algorithm:
# EVOLVE-BLOCK-START
import random
import itertools  
import numpy as np
# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-START
def base_square_generator(size: int = 3):
    # Advanced cyclic generation with permutation
    square = []
    for i in range(size):
        row = [(i + j * 2) % size for j in range(size)]
        square.append(row)
    return square
# EVOLVE-BLOCK-END

[... 더 많은 진화된 코드 ...]
```

## 🔍 **Key Differences from Standard Evolution**

| Feature | Standard GP | AlphaEvolve Implementation |
|---------|-------------|---------------------------|
| **Evolution Target** | 단일 함수 | 다중 컴포넌트 동시 진화 |
| **Code Structure** | 전체 교체 | EVOLVE-BLOCK 부분 변경 |
| **Model Usage** | 단일 LLM | Flash+Pro 앙상블 |
| **Mutation Strategy** | 랜덤 변경 | 지능적 diff 기반 |
| **Component Awareness** | 없음 | imports, functions, configs 구분 |

## 🛠️ **Customization**

### **1. MOLS 크기 변경**
```python
# config/settings.py
MOLS_SIZE = 5  # 3x3 → 5x5 변경
```

### **2. 진화 컴포넌트 추가**
```python
# mols_task/program.py에 새 EVOLVE-BLOCK 추가
# EVOLVE-BLOCK-START
def new_optimization_component():
    # 새로운 최적화 로직
    pass
# EVOLVE-BLOCK-END
```

### **3. Island 설정 조정**
```python
# config/settings.py
NUM_ISLANDS = 6              # 더 많은 island
POPULATION_PER_ISLAND = 12   # 더 큰 개체군
MIGRATION_INTERVAL = 3       # 더 자주 migration
```

## 🎛️ **Advanced Configuration**

### **Evolution Strategy Probabilities**
```python
COMPONENT_EVOLUTION_PROBABILITY = {
    'imports': 0.3,     # Import 문 진화 확률
    'functions': 0.8,   # 함수 진화 확률  
    'configs': 0.6,     # 설정 진화 확률
    'classes': 0.4,     # 클래스 진화 확률
    'general': 0.5      # 기타 진화 확률
}
```

### **Temperature Settings**
```python
FLASH_TEMPERATURE = 0.9      # Flash 모델 창의성
PRO_TEMPERATURE = 0.3        # Pro 모델 안정성
MAX_EVOLVE_BLOCKS_PER_MUTATION = 3  # 동시 진화 블록 수
```

## 🧪 **Testing & Validation**

### **단위 테스트**
```bash
python test_3x3_mols.py
```

### **EVOLVE-BLOCK 파싱 테스트**
```bash
python -c "
from core.evolve_block_parser import EvolveBlockParser
from mols_task.program import generate_MOLS_3
import inspect

parser = EvolveBlockParser()
code = inspect.getsource(generate_MOLS_3)
result = parser.parse_code(code)
print(f'Found {len(result[\"evolve_blocks\"])} EVOLVE-BLOCKS')
"
```

## 📈 **Performance Monitoring**

실행 중 다음 메트릭들이 추적됩니다:
- **세대별 최고/평균 점수**
- **Island별 성능 분석**
- **Migration 효과**
- **EVOLVE-BLOCK별 진화 횟수**
- **Flash vs Pro 모델 사용 통계**

## 🚨 **Troubleshooting**

### **일반적인 문제들**

1. **API 키 오류**
   ```bash
   export GEMINI_API_KEY="your_actual_key"
   ```

2. **EVOLVE-BLOCK 파싱 실패**
   - 주석이 정확한지 확인: `# EVOLVE-BLOCK-START`, `# EVOLVE-BLOCK-END`
   - 블록 중첩이 없는지 확인

3. **메모리 부족**
   - `POPULATION_PER_ISLAND` 감소
   - `NUM_ISLANDS` 감소

4. **낮은 성능**
   - `GENERATIONS` 증가
   - `MUTATION_RATE` 조정 (0.6-0.9)
   - `FLASH_TEMPERATURE` 증가 (더 창의적)

## 🎯 **Expected Results for 3x3 MOLS**

완벽한 3x3 MOLS 쌍의 예:
```python
Square 1:          Square 2:
[0, 1, 2]         [0, 2, 1]  
[1, 2, 0]         [2, 1, 0]
[2, 0, 1]         [1, 0, 2]

# 직교성: 9개의 고유한 (i,j) 쌍
# 점수: 1.0 (완벽)
```

## 📚 **References**

- [AlphaEvolve Paper](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
- [MOLS Theory](https://en.wikipedia.org/wiki/Graeco-Latin_square)
- [Genetic Programming](https://en.wikipedia.org/wiki/Genetic_programming)

## 🔄 **Future Enhancements**

1. **GPU 가속 지원** (현재 CPU 전용)
2. **더 복잡한 MOLS 크기** (5x5, 7x7)
3. **실시간 시각화 대시보드**
4. **자동 하이퍼파라미터 튜닝**
5. **다중 목표 최적화**

---

**Happy Evolving! 🧬✨** 
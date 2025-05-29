# Import 진화 분석: 왜 Import 부분이 진화하지 않는가?

## 🔍 핵심 발견

### 1. EVOLVE-BLOCK 주석은 실제로 처리되지 않음
- `EVOLVE-BLOCK-START`와 `EVOLVE-BLOCK-END`는 **단순한 주석**
- 코드에서 이 주석을 파싱하거나 처리하는 로직이 **존재하지 않음**
- 개발자를 위한 표시일 뿐, 실제 진화 과정에는 영향 없음

### 2. 실제 진화 과정

#### 초기 프롬프트 (PromptDesignerAgent.design_initial_prompt):
```
"Allowed Standard Library Imports: ['random', 'itertools', 'numpy']. 
Do not use any other external libraries or packages.

Please provide *only* the complete Python code for the function generate_MOLS_3. 
The code should be self-contained or rely only on the allowed imports."
```

#### 변이 프롬프트 (PromptDesignerAgent.design_mutation_prompt):
```
"Allowed Standard Library Imports: ['random', 'itertools', 'numpy']. 
Do not use other external libraries or packages.

Current Code (Version from Generation X):
```python
[전체 코드 포함 - import 문도 포함]
```

Based on the task, the 'Current Code', and its 'Evaluation Feedback', 
your goal is to propose modifications to improve the function generate_MOLS_3."
```

### 3. LLM이 Import를 진화시키지 않는 이유

#### A. 프롬프트 제약:
- "Allowed Standard Library Imports: ['random', 'itertools', 'numpy']"
- "Do not use other external libraries or packages"
- → LLM이 허용된 라이브러리만 사용하도록 제한됨

#### B. 함수 중심 지시:
- "your goal is to propose modifications to improve the function generate_MOLS_3"
- → LLM이 **함수 구현부**에만 집중하도록 유도됨

#### C. Diff 형식의 한계:
- LLM은 diff 형식으로 변경사항을 제안
- 대부분의 경우 **알고리즘 로직 변경**에 집중
- Import 문 변경은 상대적으로 우선순위가 낮음

### 4. 실제 코드 생성 과정

```
1. PromptDesignerAgent → "함수 개선" 프롬프트 생성
2. CodeGeneratorAgent → Gemini API 호출 (diff 형식 요청)
3. Gemini → diff 블록 반환:
   <<<<<<< SEARCH
   def generate_MOLS_3():
       # 기존 구현
   =======
   def generate_MOLS_3():
       # 개선된 구현
   >>>>>>> REPLACE
4. CodeGeneratorAgent._apply_diff → 부모 코드에 diff 적용
```

### 5. Import가 진화하지 않는 구체적 이유

#### A. LLM 행동 패턴:
- 대부분의 경우 **알고리즘 로직**만 변경
- Import 문은 "이미 충분하다"고 판단
- `random`, `itertools`, `numpy`로 MOLS 생성 가능

#### B. 평가 함수의 영향:
- 평가는 **생성된 MOLS의 품질**에만 집중
- Import 문의 변경이 성능에 직접적 영향 없음
- → LLM이 Import 변경의 필요성을 느끼지 못함

#### C. Diff 적용 방식:
- 대부분의 diff는 함수 내부 로직 변경
- Import 문을 포함한 diff는 드물게 생성됨

## 🔧 Import 진화를 원한다면?

### 방법 1: 프롬프트 수정
```python
# prompt_designer/agent.py 수정
prompt += "\n\nIMPORTANT: Consider whether different import combinations might lead to better algorithms. You can modify import statements if it helps improve the solution."
```

### 방법 2: 더 다양한 라이브러리 허용
```python
# MOLS_generate_main.py
allowed_imports=["random", "itertools", "numpy", "math", "collections", "functools"]
```

### 방법 3: Import 중심 변이 프롬프트
```python
# 별도의 import_mutation_prompt 생성
"Focus specifically on improving the import statements and library usage to enhance the algorithm."
```

## 📊 결론

**EVOLVE-BLOCK 주석이 있어도 Import 부분이 진화하지 않는 이유:**

1. **주석은 처리되지 않음** - 단순한 표시일 뿐
2. **프롬프트가 함수 개선에 집중** - Import 변경 유도 없음  
3. **LLM의 행동 패턴** - 알고리즘 로직 변경 선호
4. **평가 함수의 특성** - Import 변경이 성능에 직접 영향 없음
5. **허용된 라이브러리로 충분** - 추가 Import 필요성 낮음

**Import 진화를 원한다면 프롬프트와 허용 라이브러리 목록을 수정해야 합니다.** 
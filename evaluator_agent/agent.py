# Evaluator Agent 
import time
import logging
import traceback
import subprocess
import tempfile
import os
import ast
import json
import asyncio
import sys
import psutil
import radon.complexity as radon
from typing import Optional, Dict, Any, Tuple, Union, List

from core.interfaces import EvaluatorAgentInterface, Program, TaskDefinition, BaseAgent
from evaluator_agent.multi_metric_evaluator import MultiMetricEvaluator
from config import settings

logger = logging.getLogger(__name__)

class EvaluatorAgent(EvaluatorAgentInterface, BaseAgent):
    def __init__(self, task_definition: Optional[TaskDefinition] = None):
        super().__init__()
        self.task_definition = task_definition
        self.evaluation_model_name = settings.GEMINI_EVALUATION_MODEL
        self.evaluation_timeout_seconds = settings.EVALUATION_TIMEOUT_SECONDS
        self._cache = {}  # Simple in-memory cache for MOLS evaluation
        self.multi_metric_evaluator = MultiMetricEvaluator()  # 다차원 평가 시스템
        logger.info(f"EvaluatorAgent initialized with model: {self.evaluation_model_name}, timeout: {self.evaluation_timeout_seconds}s")
        if self.task_definition:
            logger.info(f"EvaluatorAgent task_definition: {self.task_definition.id}")

    def _check_syntax(self, code: str) -> List[str]:
        errors = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"SyntaxError: {e.msg} at line {e.lineno}, offset {e.offset}")
        except Exception as e:
            errors.append(f"Unexpected error during syntax check: {str(e)}")
        return errors

    def _is_mols_task(self, task: TaskDefinition) -> bool:
        """MOLS 태스크인지 확인"""
        description_lower = task.description.lower()
        return any(keyword in description_lower for keyword in 
                  ['mols', 'mutually orthogonal latin squares', 'latin square', 'orthogonal'])

    def _validate_mols_result(self, result: Any) -> Tuple[float, Dict[str, Any]]:
        """MOLS 결과 검증 및 점수 계산"""
        try:
            if not isinstance(result, list) or len(result) != 2:
                return 0.0, {"error": "Result must be a list of 2 squares"}
            
            square1, square2 = result
            
            # 8x8 행렬인지 확인
            if not (isinstance(square1, list) and isinstance(square2, list) and
                    len(square1) == 8 and len(square2) == 8):
                return 0.0, {"error": "Each square must be 8x8"}
            
            for row in square1 + square2:
                if not (isinstance(row, list) and len(row) == 8):
                    return 0.0, {"error": "Each row must have 8 elements"}
            
            # Latin Square 유효성 검사
            latin1_score = self._validate_latin_square(square1)
            latin2_score = self._validate_latin_square(square2)
            
            # 직교성 검사
            orthogonality_score = self._validate_orthogonality(square1, square2)
            
            # 전체 점수 계산
            total_score = (latin1_score + latin2_score + orthogonality_score) / 3.0
            
            details = {
                "latin1_score": latin1_score,
                "latin2_score": latin2_score,
                "orthogonality_score": orthogonality_score,
                "total_score": total_score
            }
            
            return total_score, details
            
        except Exception as e:
            return 0.0, {"error": f"Validation error: {str(e)}"}

    def _validate_latin_square(self, square: List[List[int]]) -> float:
        """Latin Square 유효성 검사"""
        try:
            expected_set = set(range(8))  # {0, 1, 2, 3, 4, 5, 6, 7}
            
            # 행 검사
            row_score = 0
            for row in square:
                if set(row) == expected_set:
                    row_score += 1
            row_score = row_score / 8.0
            
            # 열 검사
            col_score = 0
            for col_idx in range(8):
                col = [square[row_idx][col_idx] for row_idx in range(8)]
                if set(col) == expected_set:
                    col_score += 1
            col_score = col_score / 8.0
            
            return (row_score + col_score) / 2.0
            
        except Exception:
            return 0.0

    def _validate_orthogonality(self, square1: List[List[int]], square2: List[List[int]]) -> float:
        """두 Latin Square의 직교성 검사"""
        try:
            pairs = set()
            for i in range(8):
                for j in range(8):
                    pair = (square1[i][j], square2[i][j])
                    pairs.add(pair)
            
            # 64개의 고유한 순서쌍이 있어야 함
            return len(pairs) / 64.0
            
        except Exception:
            return 0.0

    async def _execute_code_safely(
        self, 
        code: str, 
        task_for_examples: TaskDefinition,
        timeout_seconds: Optional[int] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        timeout = timeout_seconds if timeout_seconds is not None else self.evaluation_timeout_seconds
        
        # MOLS 태스크인 경우 특별 처리
        if self._is_mols_task(task_for_examples):
            return await self._execute_mols_code(code, task_for_examples, timeout)
        
        # 기존 코드 실행 로직 (일반 태스크용)
        results = {"test_outputs": [], "average_runtime_ms": 0.0}
        
        if not task_for_examples.input_output_examples:
            logger.warning("No input/output examples provided to _execute_code_safely.")
            return results, "No test cases to run."

        if not task_for_examples.function_name_to_evolve:
            logger.error(f"Task {task_for_examples.id} does not specify 'function_name_to_evolve'. Cannot execute code.")
            return None, "Task definition is missing 'function_name_to_evolve'."

        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "temp_script.py")

        def serialize_arg(arg):
            if isinstance(arg, (float, int)) and (arg == float('inf') or arg == float('-inf') or arg != arg):
                return f"float('{str(arg)}')"
            return json.dumps(arg)

        # Convert input_output_examples to a string with proper Python values for Infinity
        test_cases_str = json.dumps(task_for_examples.input_output_examples)
        test_cases_str = test_cases_str.replace('"Infinity"', 'float("inf")')
        test_cases_str = test_cases_str.replace('"NaN"', 'float("nan")')

        test_harness_code = f"""
import json
import time
import sys
import math  # Import math for inf/nan constants

# User's code (function to be tested)
{code}

# Test execution logic
results = []
total_execution_time = 0
num_tests = 0

# Special constants for test cases
Infinity = float('inf')
NaN = float('nan')

test_cases = {test_cases_str} 
function_to_test_name = "{task_for_examples.function_name_to_evolve}"

# Make sure the function_to_test is available in the global scope
if function_to_test_name not in globals():
    # Attempt to find it if it was defined inside a class (common for LLM output)
    # This is a simple heuristic and might need refinement.
    found_func = None
    for name, obj in list(globals().items()):
        if isinstance(obj, type):
            if hasattr(obj, function_to_test_name):
                method = getattr(obj, function_to_test_name)
                if callable(method):
                    globals()[function_to_test_name] = method
                    found_func = True
                    break
    if not found_func:
        print(json.dumps({{"error": f"Function '{{function_to_test_name}}' not found in the global scope or as a callable method of a defined class."}}))
        sys.exit(1)
        
function_to_test = globals()[function_to_test_name]

for i, test_case in enumerate(test_cases):
    input_args = test_case.get("input")
    
    start_time = time.perf_counter()
    try:
        if isinstance(input_args, list):
            actual_output = function_to_test(*input_args)
        elif isinstance(input_args, dict):
            actual_output = function_to_test(**input_args)
        elif input_args is None:
             actual_output = function_to_test()
        else:
            actual_output = function_to_test(input_args)
            
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        total_execution_time += execution_time_ms
        num_tests += 1
        results.append({{"test_case_id": i, "output": actual_output, "runtime_ms": execution_time_ms, "status": "success"}})
    except Exception as e:
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000
        error_output = {{
            "test_case_id": i,
            "error": str(e), 
            "error_type": type(e).__name__,
            "runtime_ms": execution_time_ms,
            "status": "error"
        }}
        try:
            json.dumps(error_output)
        except TypeError:
            error_output["error"] = "Unserializable error object"
        results.append(error_output)

final_output = {{"test_outputs": results}}
if num_tests > 0:
    final_output["average_runtime_ms"] = total_execution_time / num_tests

def custom_json_serializer(obj):
    if isinstance(obj, float):
        if obj == float('inf'):
            return 'Infinity'
        elif obj == float('-inf'):
            return '-Infinity'
        elif obj != obj:
            return 'NaN'
    raise TypeError(f"Object of type {{type(obj).__name__}} is not JSON serializable")

print(json.dumps(final_output, default=custom_json_serializer))
"""
        with open(temp_file_path, "w") as f:
            f.write(test_harness_code)

        cmd = [sys.executable, temp_file_path]
        
        proc = None
        try:
            logger.debug(f"Executing code: {' '.join(cmd)} in {temp_dir}")
            start_time = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=temp_dir
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            duration = time.monotonic() - start_time
            logger.debug(f"Code execution finished in {duration:.2f}s. Exit code: {proc.returncode}")

            stdout_str = stdout.decode('utf-8', errors='replace').strip()
            stderr_str = stderr.decode('utf-8', errors='replace').strip()

            if proc.returncode != 0:
                error_message = f"Execution failed with exit code {proc.returncode}. Stdout: '{stdout_str}'. Stderr: '{stderr_str}'"
                logger.warning(error_message)
                return None, error_message
            
            if not stdout_str:
                 logger.warning(f"Execution produced no stdout. Stderr: '{stderr_str}'")
                 return None, f"No output from script. Stderr: '{stderr_str}'"

            try:
                def json_loads_with_infinity(s):
                    s = s.replace('"Infinity"', 'float("inf")')
                    s = s.replace('"-Infinity"', 'float("-inf")')
                    s = s.replace('"NaN"', 'float("nan")')
                    return json.loads(s)

                parsed_output = json_loads_with_infinity(stdout_str)
                logger.debug(f"Parsed execution output: {parsed_output}")
                return parsed_output, None

            except json.JSONDecodeError as e:
                error_message = f"Failed to parse JSON output: {e}. Raw stdout: '{stdout_str}'"
                logger.error(error_message)
                return None, error_message

        except asyncio.TimeoutError:
            if proc:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    proc.kill()
            error_message = f"Code execution timed out after {timeout} seconds"
            logger.warning(error_message)
            return None, error_message

        except Exception as e:
            error_message = f"Unexpected error during code execution: {str(e)}"
            logger.error(error_message)
            return None, error_message

        finally:
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")

    async def _execute_mols_code(
        self, 
        code: str, 
        task: TaskDefinition,
        timeout_seconds: int
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """MOLS 태스크 전용 코드 실행"""
        # 캐시 확인
        cache_key = f"{code}_{task.id}"
        if cache_key in self._cache:
            logger.debug(f"Cache hit for MOLS evaluation")
            return self._cache[cache_key], None

        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "mols_script.py")

        # MOLS 실행 코드 작성
        mols_harness_code = f"""
import json
import time
import sys

# User's code (MOLS generation function)
{code}

try:
    function_name = "{task.function_name_to_evolve}"
    
    # 함수가 존재하는지 확인
    if function_name not in globals():
        print(json.dumps({{"error": f"Function '{{function_name}}' not found"}}))
        sys.exit(1)
    
    function_to_test = globals()[function_name]
    
    # 함수 실행
    start_time = time.perf_counter()
    result = function_to_test()
    end_time = time.perf_counter()
    
    execution_time_ms = (end_time - start_time) * 1000
    
    output = {{
        "result": result,
        "execution_time_ms": execution_time_ms,
        "status": "success"
    }}
    
    print(json.dumps(output))
    
except Exception as e:
    error_output = {{
        "error": str(e),
        "error_type": type(e).__name__,
        "status": "error"
    }}
    print(json.dumps(error_output))
"""

        with open(temp_file_path, "w") as f:
            f.write(mols_harness_code)

        cmd = [sys.executable, temp_file_path]
        
        proc = None
        try:
            start_time = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=temp_dir
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
            duration = time.monotonic() - start_time

            stdout_str = stdout.decode('utf-8', errors='replace').strip()
            stderr_str = stderr.decode('utf-8', errors='replace').strip()

            if proc.returncode != 0:
                error_message = f"MOLS execution failed. Stderr: '{stderr_str}'"
                return None, error_message
            
            if not stdout_str:
                return None, f"No output from MOLS script. Stderr: '{stderr_str}'"

            try:
                parsed_output = json.loads(stdout_str)
                
                # 성공적으로 실행된 경우 MOLS 검증 수행
                if parsed_output.get("status") == "success":
                    result = parsed_output.get("result")
                    score, details = self._validate_mols_result(result)
                    
                    final_output = {
                        "test_outputs": [{
                            "test_case_id": 0,
                            "output": result,
                            "runtime_ms": parsed_output.get("execution_time_ms", 0),
                            "status": "success",
                            "mols_score": score,
                            "mols_details": details
                        }],
                        "average_runtime_ms": parsed_output.get("execution_time_ms", 0),
                        "mols_validation": details
                    }
                    
                    # 캐시에 저장
                    self._cache[cache_key] = final_output
                    return final_output, None
                else:
                    return None, parsed_output.get("error", "Unknown error")
                
            except json.JSONDecodeError as e:
                return None, f"Failed to parse MOLS output: {e}. Raw: '{stdout_str}'"

        except asyncio.TimeoutError:
            if proc:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    proc.kill()
            return None, f"MOLS execution timed out after {timeout_seconds} seconds"

        except Exception as e:
            return None, f"Unexpected error during MOLS execution: {str(e)}"

        finally:
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

    def _assess_correctness(self, execution_results: Dict[str, Any], expected_outputs: List[Dict[str, Any]]) -> Tuple[float, int, int]:
        test_outputs = execution_results.get("test_outputs", [])
        
        # MOLS 태스크인 경우 특별 처리
        if len(test_outputs) == 1 and "mols_score" in test_outputs[0]:
            mols_score = test_outputs[0]["mols_score"]
            return mols_score, 1 if mols_score > 0.8 else 0, 1
        
        # 기존 correctness 평가 로직
        if not expected_outputs or not test_outputs:
            return 0.0, 0, len(expected_outputs) if expected_outputs else 0

        passed_tests = 0
        total_tests = len(expected_outputs)

        for i, expected in enumerate(expected_outputs):
            if i < len(test_outputs):
                test_result = test_outputs[i]
                if test_result.get("status") == "success":
                    actual_output = test_result.get("output")
                    expected_output = expected.get("output")
                    if self._compare_outputs(actual_output, expected_output):
                        passed_tests += 1

        correctness_score = passed_tests / total_tests if total_tests > 0 else 0.0
        return correctness_score, passed_tests, total_tests

    async def evaluate_program(self, program: Program, task: TaskDefinition) -> Program:
        logger.info(f"Evaluating program: {program.id} for task: {task.id}")
        
        program.status = "evaluating"
        program.errors = []
        program.fitness_scores = {}

        # 1. Syntax Check
        syntax_errors = self._check_syntax(program.code)
        if syntax_errors:
            program.status = "failed_evaluation"
            program.errors.extend(syntax_errors)
            program.fitness_scores = {"correctness": 0.0, "score": 0.0}
            logger.warning(f"Program {program.id} has syntax errors: {syntax_errors}")
            return program

        try:
            # 2. Code complexity analysis (추가)
            try:
                cc = radon.cc_visit(program.code)
                avg_complexity = sum(item.complexity for item in cc) / len(cc) if cc else 0
            except Exception:
                avg_complexity = 0

            # 3. Memory usage measurement (추가)
            try:
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
            except Exception:
                memory_before = 0

            # 4. Execute the code
            execution_results, error_message = await self._execute_code_safely(program.code, task)
            
            # 5. Memory usage after execution
            try:
                process = psutil.Process()
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = max(0, memory_after - memory_before)
            except Exception:
                memory_used = 0

            if execution_results is None:
                program.status = "failed_evaluation"
                program.errors.append(error_message or "Code execution failed")
                program.fitness_scores = {"correctness": 0.0, "score": 0.0}
                logger.warning(f"Program {program.id} execution failed: {error_message}")
                return program

            # 6. Assess correctness
            expected_outputs = task.input_output_examples or []
            correctness_score, passed_tests, total_tests = self._assess_correctness(execution_results, expected_outputs)

            # 7. Calculate runtime
            avg_runtime = execution_results.get("average_runtime_ms", 0.0)
            
            # 8. Assign fitness scores
            program.fitness_scores = {
                "correctness": correctness_score,
                "score": correctness_score,  # Primary score for selection
                "runtime_ms": avg_runtime,
                "passed_tests": passed_tests,
                "total_tests": total_tests,
                "complexity": avg_complexity,
                "memory_mb": memory_used,
                "code_length": len(program.code.split('\n'))
            }

            # MOLS 특별 점수 추가
            if self._is_mols_task(task) and execution_results.get("mols_validation"):
                mols_details = execution_results["mols_validation"]
                program.fitness_scores.update({
                    "latin1_score": mols_details.get("latin1_score", 0.0),
                    "latin2_score": mols_details.get("latin2_score", 0.0),
                    "orthogonality_score": mols_details.get("orthogonality_score", 0.0),
                    "mols_total_score": mols_details.get("total_score", 0.0)
                })

            # 🆕 다차원 평가 시스템 적용
            try:
                multi_metrics = await self.multi_metric_evaluator.evaluate_comprehensive(
                    program=program,
                    task=task,
                    execution_time_ms=avg_runtime,
                    memory_usage_mb=memory_used,
                    correctness_score=correctness_score
                )
                
                # 다차원 메트릭을 fitness_scores에 추가
                program.fitness_scores.update({
                    "multi_metric_score": multi_metrics["weighted_total_score"],
                    "code_complexity": multi_metrics["cyclomatic_complexity"],
                    "lines_of_code": multi_metrics["lines_of_code"],
                    "operation_count": multi_metrics["operation_count"],
                    "readability_score": multi_metrics["readability_score"],
                    "mathematical_elegance": multi_metrics["mathematical_elegance"],
                    "efficiency_score": multi_metrics["efficiency_score"],
                    "detailed_metrics": multi_metrics  # 전체 메트릭 정보 저장
                })
                
                logger.info(f"다차원 평가 완료 - 종합 점수: {multi_metrics['weighted_total_score']:.2f}/10")
                
            except Exception as e:
                logger.warning(f"다차원 평가 실패: {e}. 기본 평가만 사용합니다.")

            program.status = "evaluated"
            logger.info(f"Program {program.id} evaluated successfully. Correctness: {correctness_score:.2f}, Runtime: {avg_runtime:.2f}ms")

        except Exception as e:
            program.status = "failed_evaluation"
            program.errors.append(f"Evaluation error: {str(e)}")
            program.errors.append(traceback.format_exc())
            program.fitness_scores = {"correctness": 0.0, "score": 0.0}
            logger.error(f"Program {program.id} evaluation error: {str(e)}")

        return program

    async def execute(self, program: Program, task: TaskDefinition) -> Program:
        return await self.evaluate_program(program, task)

    def _compare_outputs(self, actual: Any, expected: Any) -> bool:
        try:
            if type(actual) != type(expected):
                return False
            
            # Handle floating point comparison
            if isinstance(actual, float) and isinstance(expected, float):
                import math
                if math.isnan(actual) and math.isnan(expected):
                    return True
                if math.isinf(actual) and math.isinf(expected):
                    return actual == expected
                return abs(actual - expected) < 1e-9
            
            # Handle lists/nested structures
            if isinstance(actual, list) and isinstance(expected, list):
                if len(actual) != len(expected):
                    return False
                return all(self._compare_outputs(a, e) for a, e in zip(actual, expected))
            
            # Direct comparison for other types
            return actual == expected
            
        except Exception:
            return actual == expected

# Removed the old __main__ block from this file, 
# as TaskManagerAgent should be the entry point for full runs.
# The more detailed __main__ from your version of EvaluatorAgent was good for unit testing it.
# For now, I am removing it to keep the agent file clean.
# If you need to unit test EvaluatorAgent specifically, 
# we can re-add a similar main block or create separate test files. 
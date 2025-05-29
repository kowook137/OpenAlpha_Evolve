# evaluator_agent/agent.py
import traceback
import asyncio
import time
import psutil
import radon.complexity as radon
from mols_task.evaluation import evaluate
from core.interfaces import EvaluatorAgentInterface, Program, TaskDefinition, BaseAgent
from typing import Optional, Dict, Any, List, Tuple
from config import settings
import logging
import subprocess
import tempfile
import os
import json

logger = logging.getLogger(__name__)

class EvaluatorAgent(EvaluatorAgentInterface, BaseAgent):
    def __init__(self, task_definition: Optional[TaskDefinition] = None):
        super().__init__()
        self.task_definition = task_definition
        self.evaluation_model_name = settings.GEMINI_EVALUATION_MODEL
        self.evaluation_timeout_seconds = settings.EVALUATION_TIMEOUT_SECONDS
        self._cache = {}  # Simple in-memory cache

    async def execute(self, program: Program, task: TaskDefinition) -> Program:
        return await self.evaluate_program(program, task)
    
    async def evaluate_program(self, program: Program, task: TaskDefinition) -> Program:
        logger.info(f"Evaluating MOLS program: {program.id}")
        
        # Check cache
        cache_key = f"{program.code}_{task.id}"
        if cache_key in self._cache:
            logger.info(f"Cache hit for program {program.id}")
            cached_result = self._cache[cache_key]
            program.status = cached_result["status"]
            program.errors = cached_result["errors"]
            program.fitness_scores = cached_result["fitness_scores"]
            return program

        program.status = "evaluating"
        program.errors = []
        program.fitness_scores = {}

        try:
            # 1. 코드 복잡도 분석
            cc = radon.cc_visit(program.code)
            avg_complexity = sum(item.complexity for item in cc) / len(cc) if cc else 0

            # 2. 안전한 환경에서 코드 실행
            start_time = time.time()
            squares = await self._execute_safely(program.code)
            execution_time = (time.time() - start_time) * 1000  # ms로 변환

            # 3. 메모리 사용량 측정
            process = psutil.Process()
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB로 변환

            # 4. 평가 함수 호출
            scores = evaluate(squares)

            if not isinstance(scores, dict) or "score" not in scores:
                raise RuntimeError(f"Invalid evaluation result: {scores}")

            # 5. 결과 기록
            program.fitness_scores = {
                **scores,  # 기존 점수들
                "runtime_ms": execution_time,
                "memory_mb": memory_usage,
                "complexity": avg_complexity,
                "code_length": len(program.code.split('\n'))
            }
            program.status = "evaluated"

            # Cache the result
            self._cache[cache_key] = {
                "status": program.status,
                "errors": program.errors,
                "fitness_scores": program.fitness_scores
            }

        except asyncio.TimeoutError:
            program.status = "failed_evaluation"
            program.errors.append("Evaluation timed out")
            program.fitness_scores = self._get_default_scores()

        except Exception as e:
            program.status = "failed_evaluation"
            program.errors.append(str(e))
            program.errors.append(traceback.format_exc())
            program.fitness_scores = self._get_default_scores()

        logger.info(f"Evaluation finished for {program.id}. Status: {program.status}, Fitness: {program.fitness_scores}")
        return program

    async def _execute_safely(self, code: str) -> List[Any]:
        """안전한 환경에서 코드 실행"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            # 실행할 코드 작성
            f.write(code + '\n')
            f.write('import json\n')
            f.write('result = generate_MOLS_3()\n')
            f.write('print(json.dumps(result))\n')
            temp_path = f.name

        try:
            # subprocess로 실행
            process = await asyncio.create_subprocess_exec(
                'python', temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.evaluation_timeout_seconds
                )
            except asyncio.TimeoutError:
                process.kill()
                raise

            if process.returncode != 0:
                raise RuntimeError(f"Process failed with error: {stderr.decode()}")

            return json.loads(stdout.decode())

        finally:
            # 임시 파일 삭제
            os.unlink(temp_path)

    def _get_default_scores(self) -> Dict[str, float]:
        """실패 시 기본 점수"""
        return {
            "score": 0.0,
            "latin_score": 0.0,
            "orthogonality_score": 0.0,
            "runtime_ms": float('inf'),
            "memory_mb": 0.0,
            "complexity": 0.0,
            "code_length": 0
        }
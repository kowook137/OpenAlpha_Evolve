# evaluator_agent/agent.py
import traceback
from mols_task.evaluation import evaluate  # 이 함수는 evaluate(squares) 형태여야 함
from core.interfaces import EvaluatorAgentInterface, Program, TaskDefinition, BaseAgent
from typing import Optional, Dict, Any, List, Tuple
from config import settings
import logging
logger = logging.getLogger(__name__)




class EvaluatorAgent(EvaluatorAgentInterface, BaseAgent):
    def __init__(self, task_definition: Optional[TaskDefinition] = None):
        super().__init__()
        self.task_definition = task_definition
        self.evaluation_model_name = settings.GEMINI_EVALUATION_MODEL
        self.evaluation_timeout_seconds = settings.EVALUATION_TIMEOUT_SECONDS

    async def execute(self, program: Program, task: TaskDefinition) -> Program:
        return await self.evaluate_program(program, task)
    
    async def evaluate_program(self, program: Program, task: TaskDefinition) -> Program:
        logger.info(f"Evaluating MOLS program: {program.id}")
        program.status = "evaluating"
        program.errors = []
        program.fitness_scores = {}

        try:
            # 1. 코드 실행 (동적 로딩)
            exec_globals = {}
            exec(program.code, exec_globals)
            logger.debug(f"Executed code. Available keys: {list(exec_globals.keys())}")

            if "generate_MOLS_10" not in exec_globals:
                raise RuntimeError("Function `generate_MOLS_10` not found in program code.")

            generate_func = exec_globals["generate_MOLS_10"]
            squares = generate_func()  # [square1, square2, square3]

            # 2. 평가 함수 호출
            scores = evaluate(squares)  # must return {"score": float, ...}

            if not isinstance(scores, dict) or "score" not in scores:
                raise RuntimeError(f"Invalid evaluation result: {scores}")

            # 3. 결과 기록
            program.fitness_scores = scores
            program.status = "evaluated"

        except Exception as e:
            program.status = "failed_evaluation"
            program.errors.append(str(e))
            program.errors.append(traceback.format_exec())
            program.fitness_scores = {"score": 0.0, "latin_score": 0.0, "orthogonality_score": 0.0}

        logger.info(f"Evaluation finished for {program.id}. Status: {program.status}, Fitness: {program.fitness_scores}")
        return program
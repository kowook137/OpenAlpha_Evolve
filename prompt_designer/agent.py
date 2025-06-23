# Prompt Designer Agent 
from typing import Optional, Dict, Any
import logging

from core.interfaces import PromptDesignerInterface, Program, TaskDefinition, BaseAgent
from core.evolve_block_parser import EvolveBlockParser

logger = logging.getLogger(__name__)

class PromptDesignerAgent(PromptDesignerInterface, BaseAgent):
    def __init__(self, task_definition: TaskDefinition):
        super().__init__()
        self.task_definition = task_definition
        self.evolve_parser = EvolveBlockParser()
        logger.info(f"PromptDesignerAgent initialized for task: {self.task_definition.id}")

    def design_initial_prompt(self) -> str:
        logger.info(f"Designing initial prompt for task: {self.task_definition.id}")
        # This prompt should request full code, not a diff.
        
        # 사용자 정의 초기 프롬프트가 있으면 사용
        if hasattr(self.task_definition, 'initial_code_prompt') and self.task_definition.initial_code_prompt:
            # TaskDefinition에 사용자 정의 초기 프롬프트가 있는 경우
            base_prompt = self.task_definition.initial_code_prompt
            
            # EVOLVE-BLOCK 템플릿 생성 (MOLS 문제의 경우)
            evolve_template = ""
            if 'latin square' in self.task_definition.description.lower() or 'mols' in self.task_definition.description.lower():
                evolve_template = f"\n\nIMPORTANT - Use EVOLVE-BLOCK structure:\n{self.evolve_parser.create_evolve_template(self.task_definition.function_name_to_evolve, self.task_definition.description)}\n\nThe parts between # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END will be evolved. Keep the structure but modify the implementation.\n"
            
            # 기본 정보 추가
            prompt = (
                f"You are an expert Python programmer and expert in the field of combinatorics.\n\n"
                f"Task Description: {self.task_definition.description}\n\n"
                f"Input/Output Examples:\n"
                f"{self._format_input_output_examples()}\n\n"
                f"Specific Instructions:\n{base_prompt}\n\n"
                f"{evolve_template}"
                f"Evaluation Criteria: {self.task_definition.evaluation_criteria}\n\n"
                f"Please provide *only* the complete Python code. "
                f"Do not include any surrounding text, explanations, or markdown code fences."
            )
        else:
            # 기본 프롬프트 사용
            prompt = (
                f"You are an expert Python programmer. Your task is to write a Python function based on the following specifications.\n\n"
                f"Task Description: {self.task_definition.description}\n\n"
                f"Function to Implement: `{self.task_definition.function_name_to_evolve}`\n\n"
                f"Input/Output Examples:\n"
                # Format examples for clarity
                f"{self._format_input_output_examples()}\n\n"
                f"Evaluation Criteria: {self.task_definition.evaluation_criteria}\n\n"
                f"Allowed Standard Library Imports: {self.task_definition.allowed_imports}. Do not use any other external libraries or packages.\n\n"
                f"Your Response Format:\n"
                f"Please provide *only* the complete Python code for the function `{self.task_definition.function_name_to_evolve}`. "
                f"The code should be self-contained or rely only on the allowed imports. "
                f"Do not include any surrounding text, explanations, comments outside the function, or markdown code fences (like ```python or ```)."
            )
        logger.debug(f"Designed initial prompt:\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    def _format_input_output_examples(self) -> str:
        if not self.task_definition.input_output_examples:
            return "No input/output examples provided."
        formatted_examples = []
        for i, example in enumerate(self.task_definition.input_output_examples):
            input_str = str(example.get('input'))
            output_str = str(example.get('output'))
            formatted_examples.append(f"Example {i+1}:\n  Input: {input_str}\n  Expected Output: {output_str}")
        return "\n".join(formatted_examples)

    def _format_evaluation_feedback(self, program: Program, evaluation_feedback: Optional[Dict[str, Any]]) -> str:
        if not evaluation_feedback:
            return "No detailed evaluation feedback is available for the previous version of this code. Attempt a general improvement or refinement."

        correctness = evaluation_feedback.get("correctness_score", None)
        runtime = evaluation_feedback.get("runtime_ms", None)
        errors = evaluation_feedback.get("errors", []) # Ensure errors is a list
        # stdout = evaluation_feedback.get("stdout", None) # Potentially useful but can be long
        stderr = evaluation_feedback.get("stderr", None)

        feedback_parts = []
        if correctness is not None:
            feedback_parts.append(f"- Correctness Score: {correctness*100:.2f}%")
        if runtime is not None:
            feedback_parts.append(f"- Runtime: {runtime:.2f} ms")
        
        if errors:
            error_messages = "\n".join([f"  - {e}" for e in errors])
            feedback_parts.append(f"- Errors Encountered During Evaluation:\n{error_messages}")
        elif stderr:
            feedback_parts.append(f"- Standard Error Output During Execution:\n{stderr}")
        elif correctness is not None and correctness < 1.0:
            feedback_parts.append("- The code did not achieve 100% correctness but produced no explicit errors or stderr. Review logic for test case failures.")
        elif correctness == 1.0:
            feedback_parts.append("- The code achieved 100% correctness. Consider optimizing for efficiency or exploring alternative correct solutions.")
        
        if not feedback_parts:
             return "The previous version was evaluated, but no specific feedback details were captured. Try a general improvement."

        return "Summary of the previous version's evaluation:\n" + "\n".join(feedback_parts)

    def design_mutation_prompt(self, program: Program, evaluation_feedback: Optional[Dict[str, Any]] = None) -> str:
        logger.info(f"Designing mutation prompt for program: {program.id} (Generation: {program.generation})")
        logger.debug(f"Parent program code (to be mutated):\n{program.code}")
        
        feedback_summary = self._format_evaluation_feedback(program, evaluation_feedback)
        logger.debug(f"Formatted evaluation feedback for prompt:\n{feedback_summary}")

        # Import 진화 제안
        import_suggestions = self._suggest_import_evolution(program, evaluation_feedback)

        # EVOLVE-BLOCK 인식 및 지시사항
        evolve_block_guidance = ""
        if self.evolve_parser.has_evolve_blocks(program.code):
            evolve_block_guidance = (
                "\nEVOLVE-BLOCK Guidance:\n"
                "The current code contains EVOLVE-BLOCK sections marked with # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END.\n"
                "Focus your improvements primarily on the content WITHIN these blocks, as these are the evolvable regions.\n"
                "You can modify imports in EVOLVE-BLOCK sections and the algorithm logic within function EVOLVE-BLOCKS.\n"
                "Keep the overall structure intact but evolve the implementation details.\n\n"
            )

        diff_instructions = (
            "Your Response Format:\n"
            "Propose improvements to the 'Current Code' below by providing your changes as a sequence of diff blocks. "
            "Each diff block must follow this exact format:\n"
            "<<<<<<< SEARCH\n"
            "# Exact original code lines to be found and replaced\n"
            "=======\n"
            "# New code lines to replace the original\n"
            ">>>>>>> REPLACE\n\n"
            "- The SEARCH block must be an *exact* segment from the 'Current Code'. Do not paraphrase or shorten it."
            "- If you are adding new code where nothing existed, the SEARCH block can be a comment indicating the location, or an adjacent existing line."
            "- If you are deleting code, the REPLACE block should be empty."
            "- You can also add import statements if they would help solve the problem."
            "- Provide all suggested changes as one or more such diff blocks. Do not include any other text, explanations, or markdown outside these blocks."
            f"{evolve_block_guidance}"
        )

        prompt = (
            f"You are an expert Python programmer. Your task is to improve an existing Python function based on its previous performance and the overall goal.\n\n"
            f"Overall Task Description: {self.task_definition.description}\n\n"
            f"Function to Improve: `{self.task_definition.function_name_to_evolve}`\n\n"
            f"Currently Allowed Imports: {self.task_definition.allowed_imports}. You can suggest additional standard library imports if needed.\n\n"
            f"{import_suggestions}\n\n"
            f"Current Code (Version from Generation {program.generation}):\n"
            f"```python\n{program.code}\n```\n\n"
            f"Evaluation Feedback on the 'Current Code':\n{feedback_summary}\n\n"
            f"Your Improvement Goal:\n"
            f"Based on the task, the 'Current Code', and its 'Evaluation Feedback', your goal is to propose modifications to improve the function `{self.task_definition.function_name_to_evolve}`. "
            f"Prioritize fixing any errors or correctness issues. If correct, focus on improving efficiency or exploring alternative robust logic. "
            f"Consider exploring different standard library modules that could help solve the problem more effectively. "
            f"Consider the original evaluation criteria: {self.task_definition.evaluation_criteria}\n\n"
            f"{diff_instructions}"
        )
        logger.debug(f"Designed mutation prompt (requesting diff):\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    def _suggest_import_evolution(self, program: Program, evaluation_feedback: Optional[Dict[str, Any]] = None) -> str:
        """Import 진화를 위한 제안 생성"""
        suggestions = []
        
        # 성능 문제가 있을 때 제안할 모듈들
        if evaluation_feedback and evaluation_feedback.get("runtime_ms", 0) > 1000:
            suggestions.append("Consider using 'itertools' for efficient iteration patterns")
            suggestions.append("Consider using 'collections' for specialized data structures")
        
        # 복잡한 수학 문제일 때
        if any(keyword in self.task_definition.description.lower() 
               for keyword in ['matrix', 'math', 'algorithm', 'combinatorial', 'optimization']):
            suggestions.append("Consider using 'math' for mathematical operations")
            suggestions.append("Consider using 'itertools' for permutations/combinations")
            suggestions.append("Consider using 'functools' for optimization")
        
        # MOLS 특별 처리
        if 'latin square' in self.task_definition.description.lower() or 'mols' in self.task_definition.description.lower():
            suggestions.append("Consider using 'itertools' for permutation generation")
            suggestions.append("Consider using 'collections' for efficient data handling")
            suggestions.append("Consider using 'copy' for deep copying matrices")
        
        if suggestions:
            return "Import Evolution Suggestions:\n" + "\n".join([f"- {s}" for s in suggestions])
        else:
            return "Consider if additional standard library imports could improve this solution."

    def design_bug_fix_prompt(self, program: Program, error_message: str, execution_output: Optional[str] = None) -> str:
        logger.info(f"Designing bug-fix prompt for program: {program.id} (Generation: {program.generation})")
        logger.debug(f"Buggy program code:\n{program.code}")
        logger.debug(f"Primary error message: {error_message}")
        if execution_output:
            logger.debug(f"Additional execution output (stdout/stderr): {execution_output}")

        output_segment = f"Execution Output (stdout/stderr that might be relevant):\n{execution_output}\n" if execution_output else "No detailed execution output was captured beyond the error message itself.\n"
        
        diff_instructions = (
            "Your Response Format:\n"
            "Propose fixes to the 'Buggy Code' below by providing your changes as a sequence of diff blocks. "
            "Each diff block must follow this exact format:\n"
            "<<<<<<< SEARCH\n"
            "# Exact original code lines to be found and replaced\n"
            "=======\n"
            "# New code lines to replace the original\n"
            ">>>>>>> REPLACE\n\n"
            "- The SEARCH block must be an *exact* segment from the 'Buggy Code'."
            "- Provide all suggested changes as one or more such diff blocks. Do not include any other text, explanations, or markdown outside these blocks."
        )

        prompt = (
            f"You are an expert Python programmer. Your task is to fix a bug in an existing Python function.\n\n"
            f"Overall Task Description: {self.task_definition.description}\n\n"
            f"Function to Fix: `{self.task_definition.function_name_to_evolve}`\n\n"
            f"Allowed Standard Library Imports: {self.task_definition.allowed_imports}. Do not use other external libraries or packages.\n\n"
            f"Buggy Code (Version from Generation {program.generation}):\n"
            f"```python\n{program.code}\n```\n\n"
            f"Error Encountered: {error_message}\n"
            f"{output_segment}\n"
            f"Your Goal:\n"
            f"Analyze the 'Buggy Code', the 'Error Encountered', and any 'Execution Output' to identify and fix the bug(s). "
            f"The corrected function must adhere to the overall task description and allowed imports.\n\n"
            f"{diff_instructions}"
        )
        logger.debug(f"Designed bug-fix prompt (requesting diff):\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    async def execute(self, *args, **kwargs) -> Any:
        raise NotImplementedError("PromptDesignerAgent.execute() is not the primary way to use this agent. Call specific design methods.")

# Example Usage:
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    sample_task_def = TaskDefinition(
        id="task_001_designer_test",
        description="Create a Python function `sum_list(numbers)` that returns the sum of a list of integers. Handle empty lists by returning 0.",
        function_name_to_evolve="sum_list",
        input_output_examples=[
            {"input": [1, 2, 3], "output": 6},
            {"input": [], "output": 0}
        ],
        evaluation_criteria="Ensure correctness for all cases, including empty lists.",
        allowed_imports=["math"]
    )
    designer = PromptDesignerAgent(task_definition=sample_task_def)

    print("--- Initial Prompt ---")
    initial_prompt = designer.design_initial_prompt()
    print(initial_prompt)

    sample_program_mutation = Program(
        id="prog_mut_001",
        code="def sum_list(numbers):\n  # Slightly off logic\n  s = 0\n  for n in numbers:\n    s += n\n  return s if numbers else 1", # Bug for empty list
        fitness_scores={"correctness_score": 0.5, "runtime_ms": 5.0},
        generation=1,
        errors=["Test case failed: Input [], Expected 0, Got 1"],
        status="evaluated"
    )
    mutation_feedback = {
        "correctness_score": sample_program_mutation.fitness_scores["correctness_score"],
        "runtime_ms": sample_program_mutation.fitness_scores["runtime_ms"],
        "errors": sample_program_mutation.errors,
        "stderr": None
    }
    print("\n--- Mutation Prompt (Requesting Diff) ---")
    mutation_prompt = designer.design_mutation_prompt(sample_program_mutation, evaluation_feedback=mutation_feedback)
    print(mutation_prompt)

    sample_program_buggy = Program(
        id="prog_bug_002",
        code="def sum_list(numbers):\n  # Buggy implementation causing TypeError\n  if not numbers:\n    return 0\n  return sum(numbers) + \"oops\"",
        fitness_scores={"correctness_score": 0.0, "runtime_ms": 2.0},
        generation=2,
        errors=["TypeError: unsupported operand type(s) for +: 'int' and 'str'"],
        status="evaluated"
    )
    print("\n--- Bug-Fix Prompt (Requesting Diff) ---")
    bug_fix_prompt = designer.design_bug_fix_prompt(sample_program_buggy, error_message=sample_program_buggy.errors[0], execution_output="TypeError occurred during summation.")
    print(bug_fix_prompt) 
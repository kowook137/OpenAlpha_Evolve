import os
import sys
import logging
from typing import Optional, Dict, Any, List

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.interfaces import PromptDesignerInterface, Program, TaskDefinition, BaseAgent
from core.evolve_block_parser import EvolveBlockParser, EvolveBlock

logger = logging.getLogger(__name__)

class PromptDesignerAgent(PromptDesignerInterface, BaseAgent):
    def __init__(self, task_definition: TaskDefinition):
        super().__init__()
        self.task_definition = task_definition
        self.parser = EvolveBlockParser()
        logger.info(f"AlphaEvolve PromptDesignerAgent initialized for task: {self.task_definition.id}")

    def design_initial_prompt(self) -> str:
        logger.info(f"Designing initial prompt for task: {self.task_definition.id}")
        
        # Provide template with EVOLVE-BLOCK structure
        template = f"""def {self.task_definition.function_name_to_evolve}():
    \"\"\"
    {self.task_definition.description}
    \"\"\"
    
    # EVOLVE-BLOCK-START
    # Generate first 3x3 latin square
    square1 = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    
    # Generate second 3x3 latin square  
    square2 = [[0, 1, 2], [2, 0, 1], [1, 2, 0]]
    
    # Return both squares as MOLS
    result = [square1, square2]
    # EVOLVE-BLOCK-END
    
    return result
"""
                                                           
        prompt = (
            f"🚨 CRITICAL: You MUST use the exact template provided below. DO NOT create new code structure!\n\n"
            
            f"You are completing this code template for: {self.task_definition.description}\n\n"
            
            f"MANDATORY TEMPLATE (YOU MUST USE THIS EXACT STRUCTURE):\n"
            f"```python\n{template}\n```\n\n"
            
            f"🚨 STRICT REQUIREMENTS:\n"
            f"1. KEEP the EVOLVE-BLOCK-START and EVOLVE-BLOCK-END markers EXACTLY as they are\n"
            f"2. ONLY modify the algorithm code BETWEEN the EVOLVE-BLOCK markers\n"
            f"3. DO NOT change the function signature or return statement location\n"
            f"4. DO NOT remove or modify the EVOLVE-BLOCK structure\n"
            f"5. Use only these imports: {self.task_definition.allowed_imports}\n\n"
            
            f"Task specifications:\n{self._format_input_output_examples()}\n\n"
            
            f"Return the COMPLETE code with EVOLVE-BLOCK structure preserved and your improved algorithm between the markers."
        )
        logger.debug(f"Designed initial prompt of length: {len(prompt)}")
        return prompt

    def design_mutation_prompt(self, program: Program, evaluation_feedback: Optional[Dict[str, Any]] = None) -> str:
        logger.info(f"Designing mutation prompt for program: {program.id}")
        
        feedback_summary = self._format_evaluation_feedback(program, evaluation_feedback)

        prompt = (
            f"You are an expert Python programmer. Improve this algorithm:\n\n"
            f"Current code:\n```python\n{program.code}\n```\n\n"
            f"Performance feedback:\n{feedback_summary}\n\n"
            f"Task: {self.task_definition.description}\n\n"
            f"🚨 CRITICAL: PRESERVE EVOLVE-BLOCK MARKERS!\n"
            f"- Keep all # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END markers exactly as they are\n"
            f"- Only modify the code INSIDE the EVOLVE-BLOCK markers\n"
            f"- Do NOT remove or change the EVOLVE-BLOCK structure\n\n"
            f"Provide improvements using this diff format:\n"
            f"<<<<<<< SEARCH\n"
            f"exact code to find and replace (including EVOLVE-BLOCK markers)\n"
            f"=======\n"
            f"improved replacement code (preserving EVOLVE-BLOCK markers)\n"
            f">>>>>>> REPLACE\n\n"
            f"Focus on algorithmic improvements INSIDE EVOLVE-BLOCKS to increase correctness and performance."
        )
        
        logger.debug(f"Designed mutation prompt of length: {len(prompt)}")
        return prompt

    def _select_components_for_evolution(self, components: Dict[str, List[EvolveBlock]], 
                                       evaluation_feedback: Optional[Dict[str, Any]] = None) -> List[str]:
        """Select which components should be evolved based on feedback"""
        to_evolve = []
        
        if not evaluation_feedback:
            # Default: evolve main functions and configs
            if components['functions']:
                to_evolve.append('functions')
            if components['configs']:
                to_evolve.append('configs')
        else:
            score = evaluation_feedback.get('score', 0.0)
            
            if score < 0.3:
                # Low score: major algorithmic changes needed
                to_evolve.extend(['imports', 'functions', 'configs'])
            elif score < 0.7:
                # Medium score: optimize functions and configs
                to_evolve.extend(['functions', 'configs'])
            else:
                # High score: fine-tune configurations
                to_evolve.append('configs')
        
        return to_evolve

    def _format_component_summary(self, components: Dict[str, List[EvolveBlock]]) -> str:
        """Format a summary of the algorithm components"""
        summary_lines = []
        
        for comp_type, blocks in components.items():
            if blocks:
                block_names = [f"'{block.id}'" for block in blocks]
                summary_lines.append(f"- {comp_type.title()}: {', '.join(block_names)}")
        
        return '\n'.join(summary_lines) if summary_lines else "No components found"

    def _format_input_output_examples(self) -> str:
        if not self.task_definition.input_output_examples:
            return "No input/output examples provided."
        
        formatted_examples = []
        for i, example in enumerate(self.task_definition.input_output_examples):
            input_str = str(example.get('input'))
            output_str = str(example.get('output'))
            
            example_text = f"Example {i+1}:\n  Input: {input_str}\n  Expected Output: {output_str}"
            
            # Add explanation if provided (AlphaEvolve style)
            explanation = example.get('explanation')
            if explanation:
                example_text += f"\n  Explanation: {explanation}"
            
            # Add skeleton code if provided (AlphaEvolve style)
            skeleton = example.get('skeleton')
            if skeleton:
                example_text += f"\n  Code Skeleton:\n{skeleton}"
            
            formatted_examples.append(example_text)
        
        return "\n\n".join(formatted_examples)

    def _format_evaluation_feedback(self, program: Program, evaluation_feedback: Optional[Dict[str, Any]]) -> str:
        if not evaluation_feedback:
            return "No detailed evaluation feedback available. Attempt general algorithmic improvements."

        feedback_parts = []
        
        # Core metrics
        score = evaluation_feedback.get("score", evaluation_feedback.get("correctness_score"))
        if score is not None:
            feedback_parts.append(f"- Overall Score: {score:.3f}")
        
        # Specific MOLS metrics
        latin_score = evaluation_feedback.get("latin_score")
        if latin_score is not None:
            feedback_parts.append(f"- Latin Square Quality: {latin_score:.3f}")
            
        orthogonality_score = evaluation_feedback.get("orthogonality_score")
        if orthogonality_score is not None:
            feedback_parts.append(f"- Orthogonality Quality: {orthogonality_score:.3f}")
        
        runtime = evaluation_feedback.get("runtime_ms")
        if runtime is not None:
            feedback_parts.append(f"- Runtime: {runtime:.2f} ms")
        
        errors = evaluation_feedback.get("errors", [])
        if errors:
            error_messages = "\n".join([f"  - {e}" for e in errors])
            feedback_parts.append(f"- Errors:\n{error_messages}")
        
        # Performance analysis
        if score is not None:
            if score < 0.3:
                feedback_parts.append("- Analysis: Major algorithmic improvements needed")
            elif score < 0.7:
                feedback_parts.append("- Analysis: Good foundation, needs optimization")
            else:
                feedback_parts.append("- Analysis: High-performing algorithm, fine-tuning opportunities")
        
        return "\n".join(feedback_parts) if feedback_parts else "Evaluation completed, no specific issues identified."

    def design_bug_fix_prompt(self, program: Program, error_message: str, execution_output: Optional[str] = None) -> str:
        logger.info(f"Designing AlphaEvolve bug-fix prompt for program: {program.id}")
        
        # Parse the current program to identify problematic components
        parsed_result = self.parser.parse_code(program.code)
        evolve_blocks = parsed_result['evolve_blocks']
        
        output_segment = f"Execution Output:\n{execution_output}\n" if execution_output else ""
        
        prompt = (
            f"You are an expert algorithm designer using AlphaEvolve methodology. "
            f"Your task is to fix bugs in a multi-component algorithmic solution.\n\n"
            
            f"**Task:** {self.task_definition.description}\n\n"
            
            f"**Buggy Algorithm (Generation {program.generation}):**\n"
            f"```python\n{program.code}\n```\n\n"
            
            f"**Error Encountered:** {error_message}\n"
            f"{output_segment}\n"
            
            f"**Algorithm Components:** {len(evolve_blocks)} EVOLVE-BLOCKS found\n\n"
            
            f"**Fix Strategy:**\n"
            f"1. Identify which EVOLVE-BLOCK(s) contain the bug\n"
            f"2. Consider component interactions that might cause the error\n"
            f"3. Apply targeted fixes while maintaining algorithm integrity\n"
            f"4. Ensure all components work together correctly\n\n"
            
            f"**Response Format:**\n"
            f"Provide fixes as diff blocks targeting specific EVOLVE-BLOCK sections:\n"
            f"<<<<<<< SEARCH\n"
            f"# EVOLVE-BLOCK-START\n"
            f"# Exact buggy code\n"
            f"# EVOLVE-BLOCK-END\n"
            f"=======\n"
            f"# EVOLVE-BLOCK-START\n"
            f"# Fixed code\n"
            f"# EVOLVE-BLOCK-END\n"
            f">>>>>>> REPLACE\n\n"
            f"Focus on the root cause and make minimal but effective changes."
        )
        
        logger.debug(f"Designed AlphaEvolve bug-fix prompt:\n--PROMPT START--\n{prompt}\n--PROMPT END--")
        return prompt

    async def execute(self, *args, **kwargs) -> Any:
        raise NotImplementedError("PromptDesignerAgent.execute() not implemented. Use specific design methods.")

                
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
        code="def sum_list(numbers):\n  # Slightly off logic\n  s = 0\n  for n in numbers:\n    s += n\n  return s if numbers else 1",                     
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
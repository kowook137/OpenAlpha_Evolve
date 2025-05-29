import google.generativeai as genai
from typing import Optional, Dict, Any
import logging
import asyncio                        
from google.api_core.exceptions import InternalServerError, GoogleAPIError, DeadlineExceeded                              
import time
import re                             

from core.interfaces import CodeGeneratorInterface, BaseAgent, Program
from config import settings
from core.evolve_block_parser import EvolveBlockParser, EvolveBlock

logger = logging.getLogger(__name__)

class GoogleAPIError(Exception):
    pass

class CodeGeneratorAgent(CodeGeneratorInterface, BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        if not settings.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in settings. Please set it in your .env file or config.")
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model_name = settings.GEMINI_PRO_MODEL_NAME                                            
        self.generation_config = genai.types.GenerationConfig(
            temperature=1.3, 
            top_p=0.9,
            top_k=40
        )
 
        self.use_pro_model = False
        logger.info(f"CodeGeneratorAgent initialized with model: {self.model_name}")
                                                                                                              
        # Explicitly ensure we use stable gemini-1.5-pro
        pro_model_name = "gemini-1.5-pro"  # Force stable Pro model
        flash_model_name = settings.GEMINI_FLASH_MODEL_NAME
        
        logger.info(f"Initializing models: Flash={flash_model_name}, Pro={pro_model_name}")
        
        self.flash_model = genai.GenerativeModel(flash_model_name)
        self.pro_model = genai.GenerativeModel(pro_model_name)
        
        self.parser = EvolveBlockParser()
        self.ensemble_mode = self.config.get("ensemble_mode", True)  # AlphaEvolve feature
        
        logger.info(f"AlphaEvolve CodeGenerator initialized:")
        logger.info(f"  Flash Model: {flash_model_name}")
        logger.info(f"  Pro Model: {pro_model_name}")
        logger.info(f"  Ensemble Mode: {self.ensemble_mode}")

    async def generate_code(self, prompt: str, model_name: Optional[str] = None, temperature: Optional[float] = None, output_format: str = "code", generation: int = 0) -> str:
        """
        AlphaEvolve-style code generation with generation-based model selection
        """
        if output_format == "code":
            # AlphaEvolve: Flash for early generations, Pro for later generations
            if generation < settings.FLASH_TO_PRO_TRANSITION_GENERATION:
                # Early generations: Use Flash model for exploration
                logger.info(f"Using Flash model for generation {generation} (exploration phase)")
                return await self._single_model_generate(prompt, settings.GEMINI_FLASH_MODEL_NAME, temperature or 1.0, output_format)
            else:
                # Later generations: Use Pro model for exploitation
                logger.info(f"Using Pro model for generation {generation} (exploitation phase)")  
                return await self._single_model_generate(prompt, "gemini-1.5-pro", temperature or 0.3, output_format)
        else:
            # Single model for diff-based mutations
            return await self._single_model_generate(prompt, model_name, temperature, output_format)
    
    async def _ensemble_generate(self, prompt: str, temperature: Optional[float] = None) -> str:
        """
        AlphaEvolve ensemble approach: Flash for exploration + Pro for depth
        """
        logger.info("Using AlphaEvolve ensemble generation (Flash + Pro)")
        
        # Flash model for breadth of exploration with higher temperature for stability
        flash_prompt = (prompt + 
                       "\n\n🚨 CRITICAL REQUIREMENT - EVOLVE-BLOCK MARKERS:" +
                       "\nYou MUST wrap ALL algorithm code with EVOLVE-BLOCK markers." +
                       "\nWithout these markers, the code cannot evolve!" +
                       "\n\nMARKER FORMAT (REQUIRED):" +
                       "\n# EVOLVE-BLOCK-START" +
                       "\n# ALL your algorithm code goes here" +
                       "\n# EVOLVE-BLOCK-END" +
                       "\n\nREQUIREMENT: Every function, every algorithm step MUST be inside EVOLVE-BLOCKS." +
                       "\nGenerate a creative, diverse algorithmic solution with multiple components.")
        flash_result = await self._single_model_generate(
            flash_prompt, settings.GEMINI_FLASH_MODEL_NAME, 
            temperature or 1.0, "code"
        )
        
        # Skip Pro model if Flash failed to generate valid code
        if not flash_result or len(flash_result.strip()) < 10:
            logger.warning("Flash model returned insufficient result, skipping Pro model")
            return flash_result
        
        # Pro model for depth and refinement - MUST PRESERVE EVOLVE-BLOCKS
        pro_prompt = (
            f"Improve this Python function for the task: {prompt[:300]}...\n\n"
            f"Current implementation:\n```python\n{flash_result[:1000]}\n```\n\n"
            f"🚨 CRITICAL: Your improved version MUST include EVOLVE-BLOCK markers:\n"
            f"# EVOLVE-BLOCK-START\n"
            f"# improved algorithm code here\n"
            f"# EVOLVE-BLOCK-END\n\n"
            f"Provide an improved and more efficient version that preserves the EVOLVE-BLOCK structure."
        )
        
        logger.info(f"Pro model prompt length: {len(pro_prompt)} characters")
        
        pro_result = await self._single_model_generate(
            pro_prompt, "gemini-1.5-pro",  # Explicitly use stable Pro model
            temperature or 0.3, "code"
        )
        
        # Choose result prioritizing EVOLVE-BLOCK presence (critical for evolution)
        flash_has_blocks = "EVOLVE-BLOCK-START" in flash_result and "EVOLVE-BLOCK-END" in flash_result
        pro_has_blocks = "EVOLVE-BLOCK-START" in pro_result and "EVOLVE-BLOCK-END" in pro_result if pro_result else False
        
        # Strongly prefer results with EVOLVE-BLOCKS
        if pro_has_blocks and not flash_has_blocks:
            logger.info("Using Pro model result from ensemble (has EVOLVE-BLOCKS)")
            return pro_result
        elif flash_has_blocks and not pro_has_blocks:
            logger.info("Using Flash model result from ensemble (has EVOLVE-BLOCKS)")
            return flash_result
        elif flash_has_blocks and pro_has_blocks:
            # Both have blocks, use length comparison
            if pro_result and len(pro_result.strip()) > len(flash_result.strip()) * 0.5:
                logger.info("Using Pro model result from ensemble (both have EVOLVE-BLOCKS, Pro preferred)")
                return pro_result
            else:
                logger.info("Using Flash model result from ensemble (both have EVOLVE-BLOCKS, Flash preferred)")
                return flash_result
        else:
            # Neither has blocks - serious problem but still need to return something
            logger.error("🚨 CRITICAL: Neither Flash nor Pro generated EVOLVE-BLOCKS! Evolution will fail!")
            if pro_result and len(pro_result.strip()) > len(flash_result.strip()) * 0.5:
                logger.info("Using Pro model result from ensemble (fallback: no EVOLVE-BLOCKS)")
                return pro_result
            else:
                logger.info("Using Flash model result from ensemble (fallback: no EVOLVE-BLOCKS)")
                return flash_result

    async def _single_model_generate(self, prompt: str, model_name: Optional[str] = None, 
                                   temperature: Optional[float] = None, output_format: str = "code") -> str:
        if model_name is None:
            if self.use_pro_model:
                effective_model_name = "gemini-1.5-pro"  # Force stable Pro
                model_to_use = self.pro_model
            else: 
                effective_model_name = settings.GEMINI_FLASH_MODEL_NAME
                model_to_use = self.flash_model
        else:
            if "pro" in model_name.lower():
                effective_model_name = "gemini-1.5-pro"  # Force stable Pro
                model_to_use = self.pro_model
            else:
                effective_model_name = model_name
                model_to_use = self.flash_model
        
        logger.info(f"Generating code using model: {effective_model_name}, output_format: {output_format}")
        
        # Enhanced diff instructions for EVOLVE-BLOCK support
        if output_format == "diff":
            prompt += self._get_evolve_diff_instructions()

        # Set generation config with higher temperature to avoid Flash model bugs
        generation_config = {
            "temperature": temperature or 1.0,  # Changed from 0.7 to 1.0 for Flash model stability
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 4096,
        }
        
        # Add safety settings for ALL models to avoid blocks
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        logger.debug(f"Using safety settings for model: {effective_model_name}")

        retries = 3
        for attempt in range(retries):
            try:
                logger.debug(f"API Call Attempt {attempt + 1} of {retries} to {effective_model_name}.")
                logger.debug(f"Prompt length: {len(prompt)} characters")
                
                # Make API call with safety settings
                response = await model_to_use.generate_content_async(
                    prompt, 
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                if not response.candidates:
                    logger.warning("Gemini API returned no candidates.")
                    if response.prompt_feedback and response.prompt_feedback.block_reason:
                        logger.error(f"Prompt blocked. Reason: {response.prompt_feedback.block_reason}")
                        # Try with simplified prompt
                        if attempt < retries - 1:
                            logger.info("Trying with simplified prompt")
                            simplified_prompt = f"Create Python code for: {self.task_definition.function_name_to_evolve if hasattr(self, 'task_definition') else 'function'}. Requirements: {prompt[:200]}..."
                            response = await model_to_use.generate_content_async(
                                simplified_prompt, 
                                generation_config=generation_config
                            )
                            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                                generated_text = response.candidates[0].content.parts[0].text
                                if generated_text and generated_text.strip():
                                    logger.info("Simplified prompt succeeded")
                                    return self._clean_llm_output(generated_text) if output_format == "code" else generated_text
                    continue

                # Safer access to response parts
                candidate = response.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    logger.warning(f"Gemini API returned empty content or parts. Attempt {attempt + 1}")
                    if attempt < retries - 1:
                        await asyncio.sleep(1)  # Brief pause before retry
                    continue
                
                if len(candidate.content.parts) == 0:
                    logger.warning(f"Gemini API returned empty parts list. Attempt {attempt + 1}")
                    if attempt < retries - 1:
                        await asyncio.sleep(1)
                    continue
                
                generated_text = candidate.content.parts[0].text
                if not generated_text or not generated_text.strip():
                    logger.warning(f"Gemini API returned empty text. Attempt {attempt + 1}")
                    if attempt < retries - 1:
                        await asyncio.sleep(1)
                    continue
                
                logger.debug(f"Raw response from Gemini API:\n--RESPONSE START--\n{generated_text}\n--RESPONSE END--")
                
                if output_format == "code":
                    cleaned_code = self._clean_llm_output(generated_text)
                    
                    # Validate EVOLVE-BLOCK structure
                    if "EVOLVE-BLOCK-START" in cleaned_code:
                        validation_result = self._validate_evolve_blocks(cleaned_code)
                        if not validation_result['valid']:
                            logger.warning(f"Invalid EVOLVE-BLOCK structure: {validation_result['errors']}")
                    
                    logger.debug(f"Cleaned code:\n--CLEANED CODE START--\n{cleaned_code}\n--CLEANED CODE END--")
                    return cleaned_code
                else:  # diff format
                    logger.debug(f"Returning raw diff text:\n--DIFF TEXT START--\n{generated_text}\n--DIFF TEXT END--")
                    return generated_text

            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == retries - 1:
                    logger.error(f"All {retries} attempts failed. Last error: {str(e)}")
                    logger.error(f"Prompt length: {len(prompt)}")
                    logger.error(f"Model: {effective_model_name}")
                    # Don't provide fallback - evolution requires real LLM generation
                    return ""
                await asyncio.sleep(2 ** attempt)

        # Don't provide fallback - this breaks evolution
        logger.error("All API attempts failed, returning empty string to maintain evolution integrity")
        return ""

    def _get_evolve_diff_instructions(self) -> str:
        """Enhanced diff instructions for EVOLVE-BLOCK support"""
        return '''

**AlphaEvolve EVOLVE-BLOCK Diff Format:**

When modifying code with EVOLVE-BLOCK sections, use this format:

<<<<<<< SEARCH
# EVOLVE-BLOCK-START
[exact original EVOLVE-BLOCK content]
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
[new improved EVOLVE-BLOCK content]
# EVOLVE-BLOCK-END
>>>>>>> REPLACE

**CRITICAL EVOLVE-BLOCK GUIDELINES:**
1. Always include the complete # EVOLVE-BLOCK-START and # EVOLVE-BLOCK-END markers
2. You can evolve multiple EVOLVE-BLOCKS in a single response
3. Each EVOLVE-BLOCK should be self-contained but can interact with others
4. Maintain the logical structure and dependencies between blocks
5. Focus on meaningful algorithmic improvements, not trivial changes
6. Consider the component interactions when making changes

**Multi-Component Evolution:**
You can evolve several components simultaneously:
- Import statements and dependencies
- Helper functions
- Configuration parameters  
- Main algorithm logic
- Optimization strategies

Make sure your changes work together cohesively!
'''

    def _validate_evolve_blocks(self, code: str) -> Dict[str, Any]:
        """Validate EVOLVE-BLOCK structure in generated code"""
        try:
            parsed_result = self.parser.parse_code(code)
            blocks = parsed_result['evolve_blocks']
            
            errors = []
            if len(blocks) == 0:
                errors.append("No EVOLVE-BLOCKS found in generated code")
            
            # Check for syntax errors
            is_valid, syntax_errors = self.parser.validate_code_structure(code)
            if not is_valid:
                errors.extend(syntax_errors)
            
            return {
                'valid': len(errors) == 0,
                'errors': errors,
                'block_count': len(blocks),
                'blocks': blocks
            }
        except Exception as e:
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'block_count': 0,
                'blocks': []
            }

    def _clean_llm_output(self, raw_output: str) -> str:
        """Enhanced cleaning for AlphaEvolve code generation"""
        cleaned = raw_output.strip()
        
        # First, try to extract Python code block from anywhere in the response
        python_code_pattern = r'```python\s*\n(.*?)\n```'
        python_match = re.search(python_code_pattern, cleaned, re.DOTALL)
        
        if python_match:
            # Extract the Python code block
            cleaned = python_match.group(1)
            logger.debug(f"Extracted Python code block from response")
        else:
            # Fallback: Remove markdown code fences from start/end
            if cleaned.startswith("```python"):
                cleaned = cleaned[9:]
            elif cleaned.startswith("```"):
                cleaned = cleaned[3:]
            
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        
        # Remove common LLM prefixes (only if they're at the start)
        prefixes_to_remove = [
            "Here's the solution:",
            "Here's the code:",
            "Here's the implementation:",
            "Here's the improved version:",
        ]
        
        for prefix in prefixes_to_remove:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Ensure proper EVOLVE-BLOCK formatting
        cleaned = self._normalize_evolve_blocks(cleaned)
        
        # Additional cleanup: remove any remaining non-Python text at the beginning
        lines = cleaned.split('\n')
        start_idx = 0
        
        # Find the first line that looks like Python code or EVOLVE-BLOCK
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if (stripped_line.startswith('# EVOLVE-BLOCK') or 
                stripped_line.startswith('def ') or 
                stripped_line.startswith('import ') or 
                stripped_line.startswith('from ') or
                stripped_line.startswith('class ') or
                (stripped_line.startswith('#') and 'EVOLVE' not in stripped_line) or
                stripped_line == ''):
                start_idx = i
                break
        
        if start_idx > 0:
            cleaned = '\n'.join(lines[start_idx:])
            logger.debug(f"Removed {start_idx} non-Python lines from start")
        
        return cleaned.strip()

    def _normalize_evolve_blocks(self, code: str) -> str:
        """Normalize EVOLVE-BLOCK formatting"""
        # Ensure proper spacing around EVOLVE-BLOCK markers
        code = re.sub(r'#\s*EVOLVE-BLOCK-START\s*', '# EVOLVE-BLOCK-START\n', code)
        code = re.sub(r'#\s*EVOLVE-BLOCK-END\s*', '# EVOLVE-BLOCK-END', code)
        
        return code

    def _apply_diff(self, parent_code: str, diff_text: str) -> str:
        """
        Enhanced diff application with EVOLVE-BLOCK awareness
        """
        logger.info("Applying AlphaEvolve diff with EVOLVE-BLOCK support")
        logger.debug(f"Parent code length: {len(parent_code)}")
        logger.debug(f"Diff text:\n{diff_text}")

        modified_code = parent_code
        
        # Enhanced pattern to handle EVOLVE-BLOCK diffs
        diff_pattern = re.compile(
            r"<<<<<<< SEARCH\s*?\n(.*?)\n=======\s*?\n(.*?)\n>>>>>>> REPLACE", 
            re.DOTALL
        )
        
        replacements_made = []
        
        for match in diff_pattern.finditer(diff_text):
            search_block = match.group(1).strip()
            replace_block = match.group(2).strip()
            
            logger.debug(f"Processing diff block:\nSEARCH:\n{search_block}\nREPLACE:\n{replace_block}")
            
            # Try exact match first
            if search_block in modified_code:
                logger.debug("Found exact match for SEARCH block")
                modified_code = modified_code.replace(search_block, replace_block, 1)
                logger.debug("Applied diff block successfully")
            else:
                # Enhanced fuzzy matching for EVOLVE-BLOCKS
                success = self._apply_evolve_block_diff(modified_code, search_block, replace_block)
                if success:
                    modified_code = success
                    logger.debug("Applied EVOLVE-BLOCK diff successfully")
                else:
                    logger.warning(f"Failed to apply diff block:\n{search_block}")
        
        # Validate the result
        validation_result = self._validate_evolve_blocks(modified_code)
        if not validation_result['valid']:
            logger.warning(f"Diff application resulted in invalid EVOLVE-BLOCK structure: {validation_result['errors']}")
        
        return modified_code

    def _apply_evolve_block_diff(self, code: str, search_block: str, replace_block: str) -> Optional[str]:
        """Apply diff specifically targeting EVOLVE-BLOCK sections"""
        try:
            # Check if this is an EVOLVE-BLOCK diff
            if "EVOLVE-BLOCK-START" not in search_block:
                return None
            
            # Parse current code to find EVOLVE-BLOCKS
            parsed_result = self.parser.parse_code(code)
            blocks = parsed_result['evolve_blocks']
            
            # Find the matching block by content similarity
            best_match = None
            best_similarity = 0
            
            for block in blocks:
                similarity = self._calculate_similarity(search_block, block.content)
                if similarity > best_similarity and similarity > 0.7:  # 70% similarity threshold
                    best_similarity = similarity
                    best_match = block
            
            if best_match:
                # Replace the matched block
                new_blocks = []
                for block in blocks:
                    if block.id == best_match.id:
                        new_block = EvolveBlock(
                            id=block.id,
                            content=replace_block.replace("# EVOLVE-BLOCK-START\n", "").replace("\n# EVOLVE-BLOCK-END", "").strip(),
                            start_line=block.start_line,
                            end_line=block.end_line,
                            block_type=block.block_type
                        )
                        new_blocks.append(new_block)
                    else:
                        new_blocks.append(block)
                
                # Reconstruct the code
                reconstructed = self.parser.reconstruct_code(new_blocks)
                return reconstructed
            
        except Exception as e:
            logger.error(f"Error in EVOLVE-BLOCK diff application: {e}")
        
        return None

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two code blocks"""
        # Simple similarity based on common lines
        lines1 = set(line.strip() for line in text1.split('\n') if line.strip())
        lines2 = set(line.strip() for line in text2.split('\n') if line.strip())
        
        if not lines1 or not lines2:
            return 0.0
        
        intersection = len(lines1.intersection(lines2))
        union = len(lines1.union(lines2))
        
        return intersection / union if union > 0 else 0.0

    async def execute(self, prompt: str, temperature: float = 0.7, 
                     output_format: str = "code", parent_code_for_diff: Optional[str] = None) -> str:
        """Execute code generation with AlphaEvolve enhancements"""
        logger.info(f"Executing AlphaEvolve code generation: format={output_format}")
        
        if output_format == "diff" and parent_code_for_diff:
            # Generate diff
            diff_result = await self.generate_code(prompt, temperature=temperature, output_format="diff")
            
            if diff_result and "<<<<<<< SEARCH" in diff_result:
                # Apply the diff
                final_code = self._apply_diff(parent_code_for_diff, diff_result)
                return final_code
            else:
                logger.warning("Diff generation failed, returning parent code")
                return parent_code_for_diff
        else:
            # Direct code generation
            return await self.generate_code(prompt, temperature=temperature, output_format=output_format)

    async def generate_initial_code_with_template(self, task_description: str, function_name: str, generation: int = 0) -> str:
        """Generate initial code by providing EVOLVE-BLOCK template and asking LLM to fill the algorithm only"""
        logger.info(f"Generating initial code with EVOLVE-BLOCK template for: {function_name}")
        
        # Provide fixed EVOLVE-BLOCK template
        template = f"""def {function_name}():
    \"\"\"
    {task_description}
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
        
        # Ask LLM to improve only the algorithm inside EVOLVE-BLOCK
        prompt = (
            f"You are improving the algorithm inside the EVOLVE-BLOCK markers.\n\n"
            f"Task: {task_description}\n\n"
            f"Current template:\n```python\n{template}\n```\n\n"
            f" REQUIREMENTS:\n"
            f"1. KEEP all EVOLVE-BLOCK-START and EVOLVE-BLOCK-END markers exactly as they are\n"
            f"2. ONLY improve the algorithm code BETWEEN the markers\n"
            f"3. DO NOT change function signature or return statement placement\n"
            f"4. Focus on creating a better MOLS generation algorithm\n\n"
            f"Return the COMPLETE code with improved algorithm between EVOLVE-BLOCK markers."
        )
        
        # For initial generation, directly use template to ensure EVOLVE-BLOCK structure
        # EVOLVE-BLOCK markers serve as position indicators only
        # LLM will modify only the algorithm code between markers in later generations
        logger.info("Using fixed EVOLVE-BLOCK template for initial generation to guarantee structure")
        logger.info("EVOLVE-BLOCK markers are position indicators - LLM will modify only the algorithm between markers during evolution")
        return template

                                                 
if __name__ == '__main__':
    import asyncio
    logging.basicConfig(level=logging.DEBUG)
    
    async def test_diff_application():
        agent = CodeGeneratorAgent()
        parent = """Line 1
Line 2 to be replaced
Line 3
Another block
To be changed
End of block
Final line"""

        diff = """Some preamble text from LLM...
<<<<<<< SEARCH
Line 2 to be replaced
=======
Line 2 has been successfully replaced
>>>>>>> REPLACE

Some other text...

<<<<<<< SEARCH
Another block
To be changed
End of block
=======
This
Entire
Block
Is New
>>>>>>> REPLACE
Trailing text..."""
        expected_output = """Line 1
Line 2 has been successfully replaced
Line 3
This
Entire
Block
Is New
Final line"""
        
        print("--- Testing _apply_diff directly ---")
        result = agent._apply_diff(parent, diff)
        print("Result of diff application:")
        print(result)
        assert result.strip() == expected_output.strip(), f"Direct diff application failed.\nExpected:\n{expected_output}\nGot:\n{result}"
        print("_apply_diff test passed.")

        print("\n--- Testing execute with output_format='diff' ---")
        async def mock_generate_code(prompt, model_name, temperature, output_format):
            return diff
        
        agent.generate_code = mock_generate_code 
        
        result_execute_diff = await agent.execute(
            prompt="doesn't matter for this mock", 
            parent_code_for_diff=parent,
            output_format="diff"
        )
        print("Result of execute with diff:")
        print(result_execute_diff)
        assert result_execute_diff.strip() == expected_output.strip(), f"Execute with diff failed.\nExpected:\n{expected_output}\nGot:\n{result_execute_diff}"
        print("Execute with diff test passed.")


    async def test_generation():
        agent = CodeGeneratorAgent()
        
        test_prompt_full_code = "Write a Python function that takes two numbers and returns their sum."
        generated_full_code = await agent.execute(test_prompt_full_code, temperature=0.6, output_format="code")
        print("\n--- Generated Full Code (via execute) ---")
        print(generated_full_code)
        print("----------------------")
        assert "def" in generated_full_code, "Full code generation seems to have failed."

        parent_code_for_llm_diff = '''
def greet(name):
    return f"Hello, {name}!"

def process_data(data):
    # TODO: Implement data processing
    return data * 2 # Simple placeholder
'''
        test_prompt_diff_gen = f'''
Current code:
```python
{parent_code_for_llm_diff}
```
Task: Modify the `process_data` function to add 5 to the result instead of multiplying by 2.
Also, change the greeting in `greet` to "Hi, {name}!!!".
'''
                                                                            
                                                           
                                          
                              
                                   
                                                           
           
                                                                       
                                           
                                         
                                                                                                               
                                                                                                           
        
        async def mock_generate_empty_diff(prompt, model_name, temperature, output_format):
            return "  \n  " 
        
        original_generate_code = agent.generate_code 
        agent.generate_code = mock_generate_empty_diff
        
        print("\n--- Testing execute with empty diff from LLM ---")
        result_empty_diff = await agent.execute(
            prompt="doesn't matter",
            parent_code_for_diff=parent_code_for_llm_diff,
            output_format="diff"
        )
        assert result_empty_diff == parent_code_for_llm_diff, "Empty diff should return parent code."
        print("Execute with empty diff test passed.")
        agent.generate_code = original_generate_code

    async def main_tests():
        await test_diff_application()
                                                                                     
        print("\nAll selected local tests in CodeGeneratorAgent passed.")

    asyncio.run(main_tests())

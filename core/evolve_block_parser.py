"""
EVOLVE-BLOCK Parser for AlphaEvolve
Parses and manages code blocks marked with EVOLVE-BLOCK-START/END comments
"""
import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EvolveBlock:
    """Represents a single EVOLVE-BLOCK"""
    id: str
    content: str
    start_line: int
    end_line: int
    block_type: str  # 'import', 'function', 'class', 'config'

class EvolveBlockParser:
    """Parser for EVOLVE-BLOCK marked code sections"""
    
    def __init__(self):
        self.blocks: List[EvolveBlock] = []
        self.static_code: str = ""
        
    def parse_code(self, code: str) -> Dict[str, any]:
        """
        Parse code and extract EVOLVE-BLOCK sections
        Returns: {
            'evolve_blocks': List[EvolveBlock],
            'static_code': str,
            'full_template': str
        }
        """
        lines = code.split('\n')
        blocks = []
        static_lines = []
        current_block = None
        current_block_lines = []
        block_counter = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            if stripped == "# EVOLVE-BLOCK-START":
                if current_block is not None:
                    logger.warning(f"Nested EVOLVE-BLOCK found at line {i+1}")
                
                current_block = {
                    'start_line': i,
                    'id': f"block_{block_counter}",
                    'content_lines': []
                }
                block_counter += 1
                static_lines.append(f"# EVOLVE-PLACEHOLDER-{current_block['id']}")
                
            elif stripped == "# EVOLVE-BLOCK-END":
                if current_block is None:
                    logger.warning(f"EVOLVE-BLOCK-END without START at line {i+1}")
                    continue
                
                # Determine block type
                content = '\n'.join(current_block['content_lines'])
                block_type = self._determine_block_type(content)
                
                block = EvolveBlock(
                    id=current_block['id'],
                    content=content,
                    start_line=current_block['start_line'],
                    end_line=i,
                    block_type=block_type
                )
                blocks.append(block)
                current_block = None
                
            elif current_block is not None:
                current_block['content_lines'].append(line)
            else:
                static_lines.append(line)
        
        self.blocks = blocks
        self.static_code = '\n'.join(static_lines)
        
        return {
            'evolve_blocks': blocks,
            'static_code': self.static_code,
            'full_template': code
        }
    
    def _determine_block_type(self, content: str) -> str:
        """Determine the type of EVOLVE-BLOCK based on content"""
        content_stripped = content.strip()
        
        if content_stripped.startswith('import ') or content_stripped.startswith('from '):
            return 'import'
        elif content_stripped.startswith('def '):
            return 'function'
        elif content_stripped.startswith('class '):
            return 'class'
        elif any(keyword in content_stripped for keyword in ['=', 'config', 'param']):
            return 'config'
        else:
            return 'general'
    
    def reconstruct_code(self, blocks: List[EvolveBlock]) -> str:
        """Reconstruct full code from static template and evolved blocks"""
        result = self.static_code
        
        for block in blocks:
            placeholder = f"# EVOLVE-PLACEHOLDER-{block.id}"
            if placeholder in result:
                replacement = f"# EVOLVE-BLOCK-START\n{block.content}\n# EVOLVE-BLOCK-END"
                result = result.replace(placeholder, replacement)
            else:
                logger.warning(f"Placeholder {placeholder} not found in static code")
        
        return result
    
    def get_evolvable_components(self) -> Dict[str, List[EvolveBlock]]:
        """Group blocks by type for targeted evolution"""
        components = {
            'imports': [],
            'functions': [],
            'classes': [],
            'configs': [],
            'general': []
        }
        
        for block in self.blocks:
            if block.block_type == 'import':
                components['imports'].append(block)
            elif block.block_type == 'function':
                components['functions'].append(block)
            elif block.block_type == 'class':
                components['classes'].append(block)
            elif block.block_type == 'config':
                components['configs'].append(block)
            else:
                components['general'].append(block)
        
        return components
    
    def validate_code_structure(self, code: str) -> Tuple[bool, List[str]]:
        """Validate that reconstructed code is syntactically correct"""
        errors = []
        
        try:
            compile(code, '<string>', 'exec')
            return True, []
        except SyntaxError as e:
            errors.append(f"Syntax Error: {e}")
        except Exception as e:
            errors.append(f"Compilation Error: {e}")
        
        return False, errors

    def extract_evolve_blocks_only(self, code: str) -> List[str]:
        """Extract only the content of EVOLVE-BLOCK sections"""
        blocks = []
        lines = code.split('\n')
        current_block_lines = []
        inside_block = False
        
        for line in lines:
            stripped = line.strip()
            
            if stripped == "# EVOLVE-BLOCK-START":
                inside_block = True
                current_block_lines = []
            elif stripped == "# EVOLVE-BLOCK-END":
                if inside_block:
                    blocks.append('\n'.join(current_block_lines))
                inside_block = False
            elif inside_block:
                current_block_lines.append(line)
        
        return blocks

    def has_evolve_blocks(self, code: str) -> bool:
        """Check if code contains EVOLVE-BLOCK markers"""
        return "# EVOLVE-BLOCK-START" in code and "# EVOLVE-BLOCK-END" in code

    def create_evolve_template(self, function_name: str, description: str) -> str:
        """Create a template with EVOLVE-BLOCK for MOLS problems"""
        return f'''# EVOLVE-BLOCK-START
import itertools
import random
import math
# EVOLVE-BLOCK-END

def {function_name}():
    """
    {description}
    """
    # EVOLVE-BLOCK-START
    # 초기 접근 방법 - 이 부분이 진화됩니다
    size = 8  # 기본 크기
    
    # 간단한 시작점
    square1 = [[(i + j) % size for j in range(size)] for i in range(size)]
    square2 = [[(i * 2 + j) % size for j in range(size)] for i in range(size)]
    
    return square1, square2
    # EVOLVE-BLOCK-END''' 
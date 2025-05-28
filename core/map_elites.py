"""
MAP-Elites (Multi-dimensional Archive of Phenotypic Elites) implementation.
Maintains diversity by storing the best individual for each behavior characteristic.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import hashlib

from core.interfaces import MAPElitesInterface, Program
from config import settings

logger = logging.getLogger(__name__)

class MAPElites(MAPElitesInterface):
    """
    MAP-Elites algorithm implementation for maintaining diverse solutions.
    Creates a multi-dimensional archive where each cell contains the best
    individual for that particular behavior characteristic combination.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Behavior dimensions configuration
        self.behavior_dimensions = self.config.get("behavior_dimensions", [
            "code_complexity",    # 코드 복잡도 (라인 수, 순환 복잡도 등)
            "execution_time",     # 실행 시간
            "memory_usage",       # 메모리 사용량 (추정)
            "solution_approach"   # 솔루션 접근 방식 (해시 기반)
        ])
        
        # Each dimension is discretized into bins
        self.dimension_bins = self.config.get("dimension_bins", {
            "code_complexity": 5,    # 5개 복잡도 레벨
            "execution_time": 5,     # 5개 시간 레벨  
            "memory_usage": 3,       # 3개 메모리 레벨
            "solution_approach": 10  # 10개 접근 방식 카테고리
        })
        
        # Archive: maps behavior descriptor tuple to best program
        self.archive: Dict[Tuple, Program] = {}
        
        # Statistics
        self.total_evaluations = 0
        self.archive_updates = 0
        
        logger.info(f"MAP-Elites initialized with dimensions: {self.behavior_dimensions}")
        logger.info(f"Dimension bins: {self.dimension_bins}")
    
    def get_behavior_descriptor(self, program: Program) -> tuple:
        """
        Calculate behavior descriptor for a program.
        Returns a tuple representing the program's position in behavior space.
        """
        descriptors = []
        
        for dimension in self.behavior_dimensions:
            if dimension == "code_complexity":
                descriptor = self._calculate_code_complexity(program)
            elif dimension == "execution_time":
                descriptor = self._calculate_execution_time_bin(program)
            elif dimension == "memory_usage":
                descriptor = self._calculate_memory_usage_bin(program)
            elif dimension == "solution_approach":
                descriptor = self._calculate_solution_approach(program)
            else:
                descriptor = 0  # Default
            
            descriptors.append(descriptor)
        
        return tuple(descriptors)
    
    def _calculate_code_complexity(self, program: Program) -> int:
        """Calculate code complexity bin (0-4)"""
        if not program.code:
            return 0
        
        lines = program.code.split('\n')
        non_empty_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
        line_count = len(non_empty_lines)
        
        # Bin by line count
        if line_count <= 5:
            return 0
        elif line_count <= 10:
            return 1
        elif line_count <= 20:
            return 2
        elif line_count <= 40:
            return 3
        else:
            return 4
    
    def _calculate_execution_time_bin(self, program: Program) -> int:
        """Calculate execution time bin (0-4)"""
        runtime_ms = program.fitness_scores.get("runtime_ms", 0.0)
        
        if runtime_ms <= 1.0:
            return 0  # Very fast
        elif runtime_ms <= 10.0:
            return 1  # Fast
        elif runtime_ms <= 100.0:
            return 2  # Medium
        elif runtime_ms <= 1000.0:
            return 3  # Slow
        else:
            return 4  # Very slow
    
    def _calculate_memory_usage_bin(self, program: Program) -> int:
        """Calculate estimated memory usage bin (0-2)"""
        if not program.code:
            return 0
        
        # Simple heuristic based on code patterns
        memory_indicators = 0
        
        # Check for memory-intensive patterns
        if 'list(' in program.code or '[' in program.code:
            memory_indicators += 1
        if 'dict(' in program.code or '{' in program.code:
            memory_indicators += 1
        if 'set(' in program.code:
            memory_indicators += 1
        if 'heapq' in program.code:
            memory_indicators += 1
        
        # Count variable assignments (rough estimate)
        assignments = program.code.count('=')
        if assignments > 10:
            memory_indicators += 1
        
        if memory_indicators <= 1:
            return 0  # Low memory
        elif memory_indicators <= 3:
            return 1  # Medium memory
        else:
            return 2  # High memory
    
    def _calculate_solution_approach(self, program: Program) -> int:
        """Calculate solution approach bin (0-9) based on code patterns"""
        if not program.code:
            return 0
        
        # Use hash of key algorithmic patterns to categorize approach
        approach_indicators = []
        
        # Check for different algorithmic patterns
        if 'heapq' in program.code:
            approach_indicators.append('heap')
        if 'queue' in program.code or 'deque' in program.code:
            approach_indicators.append('queue')
        if 'stack' in program.code or 'append' in program.code and 'pop' in program.code:
            approach_indicators.append('stack')
        if 'sort' in program.code:
            approach_indicators.append('sort')
        if 'recursive' in program.code or 'return ' in program.code and '(' in program.code:
            approach_indicators.append('recursive')
        if 'for' in program.code and 'in' in program.code:
            approach_indicators.append('iterative')
        if 'while' in program.code:
            approach_indicators.append('while_loop')
        if 'dict' in program.code or 'get(' in program.code:
            approach_indicators.append('hash_table')
        if 'set(' in program.code:
            approach_indicators.append('set_based')
        if 'lambda' in program.code or 'map(' in program.code:
            approach_indicators.append('functional')
        
        # Create a hash from the combination of patterns
        pattern_string = '_'.join(sorted(approach_indicators))
        if not pattern_string:
            pattern_string = 'basic'
        
        # Convert to bin (0-9)
        hash_value = int(hashlib.md5(pattern_string.encode()).hexdigest(), 16)
        return hash_value % self.dimension_bins["solution_approach"]
    
    def update_archive(self, program: Program) -> bool:
        """
        Update archive with new program if it's better than existing one in its cell.
        Returns True if archive was updated, False otherwise.
        """
        self.total_evaluations += 1
        
        # Get behavior descriptor
        behavior_desc = self.get_behavior_descriptor(program)
        
        # Get fitness score
        fitness = program.fitness_scores.get("score", program.fitness_scores.get("correctness", 0.0))
        
        # Check if this cell is empty or if new program is better
        if behavior_desc not in self.archive:
            # Empty cell - add program
            self.archive[behavior_desc] = program
            self.archive_updates += 1
            logger.debug(f"Added program {program.id} to empty cell {behavior_desc} with fitness {fitness}")
            return True
        else:
            # Cell occupied - check if new program is better
            existing_program = self.archive[behavior_desc]
            existing_fitness = existing_program.fitness_scores.get("score", 
                                                                 existing_program.fitness_scores.get("correctness", 0.0))
            
            if fitness > existing_fitness:
                # New program is better
                self.archive[behavior_desc] = program
                self.archive_updates += 1
                logger.debug(f"Replaced program in cell {behavior_desc}: {existing_fitness:.3f} -> {fitness:.3f}")
                return True
            else:
                logger.debug(f"Program {program.id} not better than existing in cell {behavior_desc}")
                return False
    
    def get_diverse_programs(self, num_programs: int) -> List[Program]:
        """
        Get diverse programs from the archive.
        Returns up to num_programs diverse individuals.
        """
        if not self.archive:
            return []
        
        # Get all programs from archive
        all_programs = list(self.archive.values())
        
        if len(all_programs) <= num_programs:
            return all_programs
        
        # Select diverse subset
        # Strategy: select programs with highest fitness from different cells
        sorted_programs = sorted(all_programs, 
                               key=lambda p: p.fitness_scores.get("score", 
                                                                p.fitness_scores.get("correctness", 0.0)), 
                               reverse=True)
        
        return sorted_programs[:num_programs]
    
    def get_archive_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current archive"""
        if not self.archive:
            return {
                "archive_size": 0,
                "total_evaluations": self.total_evaluations,
                "archive_updates": self.archive_updates,
                "update_rate": 0.0,
                "coverage": 0.0
            }
        
        # Calculate total possible cells
        total_cells = 1
        for dim in self.behavior_dimensions:
            total_cells *= self.dimension_bins[dim]
        
        # Calculate coverage
        coverage = len(self.archive) / total_cells
        
        # Calculate fitness statistics
        fitnesses = [p.fitness_scores.get("score", p.fitness_scores.get("correctness", 0.0)) 
                    for p in self.archive.values()]
        
        stats = {
            "archive_size": len(self.archive),
            "total_evaluations": self.total_evaluations,
            "archive_updates": self.archive_updates,
            "update_rate": self.archive_updates / max(1, self.total_evaluations),
            "coverage": coverage,
            "total_possible_cells": total_cells,
            "fitness_stats": {
                "mean": np.mean(fitnesses) if fitnesses else 0.0,
                "std": np.std(fitnesses) if fitnesses else 0.0,
                "min": np.min(fitnesses) if fitnesses else 0.0,
                "max": np.max(fitnesses) if fitnesses else 0.0
            }
        }
        
        return stats
    
    def get_behavior_space_distribution(self) -> Dict[str, Any]:
        """Get distribution of programs across behavior space"""
        if not self.archive:
            return {}
        
        # Count programs in each dimension bin
        dimension_counts = {}
        
        for i, dimension in enumerate(self.behavior_dimensions):
            counts = defaultdict(int)
            for behavior_desc in self.archive.keys():
                bin_value = behavior_desc[i]
                counts[bin_value] += 1
            dimension_counts[dimension] = dict(counts)
        
        return dimension_counts
    
    def clear_archive(self):
        """Clear the archive and reset statistics"""
        self.archive.clear()
        self.total_evaluations = 0
        self.archive_updates = 0
        logger.info("MAP-Elites archive cleared")
    
    async def execute(self, action: str, **kwargs) -> Any:
        """Execute MAP-Elites actions"""
        if action == "get_behavior_descriptor":
            return self.get_behavior_descriptor(kwargs["program"])
        elif action == "update_archive":
            return self.update_archive(kwargs["program"])
        elif action == "get_diverse_programs":
            return self.get_diverse_programs(kwargs["num_programs"])
        elif action == "get_statistics":
            return self.get_archive_statistics()
        elif action == "get_distribution":
            return self.get_behavior_space_distribution()
        elif action == "clear":
            self.clear_archive()
            return True
        else:
            raise ValueError(f"Unknown action: {action}")

# Helper function to create MAP-Elites with different configurations
def create_map_elites(config_name: str = "default") -> MAPElites:
    """Create MAP-Elites instance with predefined configuration"""
    
    configs = {
        "default": {
            "behavior_dimensions": ["code_complexity", "execution_time", "solution_approach"],
            "dimension_bins": {
                "code_complexity": 5,
                "execution_time": 5,
                "solution_approach": 8
            }
        },
        "detailed": {
            "behavior_dimensions": ["code_complexity", "execution_time", "memory_usage", "solution_approach"],
            "dimension_bins": {
                "code_complexity": 6,
                "execution_time": 6,
                "memory_usage": 4,
                "solution_approach": 12
            }
        },
        "simple": {
            "behavior_dimensions": ["code_complexity", "execution_time"],
            "dimension_bins": {
                "code_complexity": 4,
                "execution_time": 4
            }
        }
    }
    
    config = configs.get(config_name, configs["default"])
    return MAPElites(config) 
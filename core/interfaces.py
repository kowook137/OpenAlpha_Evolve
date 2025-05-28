from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
import enum

@dataclass
class Program:
    id: str
    code: str
    fitness_scores: Dict[str, float] = field(default_factory=dict)                                                 
    generation: int = 0
    parent_id: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    status: str = "unevaluated"                                                              

@dataclass
class TaskDefinition:
    id: str
    description: str                                              
    function_name_to_evolve: Optional[str] = None                                                      
    input_output_examples: Optional[List[Dict[str, Any]]] = None                                                    
    evaluation_criteria: Optional[Dict[str, Any]] = None                                                            
    initial_code_prompt: Optional[str] = "Provide an initial Python solution for the following problem:"
    allowed_imports: Optional[List[str]] = None                                  

class EvolutionStrategy(enum.Enum):
    """Different evolution strategies for islands"""
    EXPLOITATION = "exploitation"  # Focus on refining best solutions
    EXPLORATION = "exploration"    # Focus on diverse solutions
    BALANCED = "balanced"         # Mix of both
    RANDOM = "random"            # Random mutations

@dataclass
class Island:
    """Represents an island in the island model"""
    id: str
    population: List[Program] = field(default_factory=list)
    strategy: EvolutionStrategy = EvolutionStrategy.BALANCED
    generation: int = 0
    migration_history: List[str] = field(default_factory=list)  # Track migration sources
    fitness_history: List[float] = field(default_factory=list)  # Track average fitness over time
    
    def get_best_program(self) -> Optional[Program]:
        """Get the best program in this island"""
        if not self.population:
            return None
        return max(self.population, 
                  key=lambda p: p.fitness_scores.get("score", p.fitness_scores.get("correctness", 0.0)))
    
    def get_average_fitness(self) -> float:
        """Get average fitness of the island"""
        if not self.population:
            return 0.0
        scores = [p.fitness_scores.get("score", p.fitness_scores.get("correctness", 0.0)) 
                 for p in self.population]
        return sum(scores) / len(scores)

@dataclass
class MigrationEvent:
    """Represents a migration event between islands"""
    source_island_id: str
    target_island_id: str
    migrant_program_ids: List[str]
    generation: int
    migration_type: str = "standard"  # standard, elite, random

class BaseAgent(ABC):
    """Base class for all agents."""
    @abstractmethod
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Main execution method for an agent."""
        pass

class TaskManagerInterface(BaseAgent):
    @abstractmethod
    async def manage_evolutionary_cycle(self):
        pass

class PromptDesignerInterface(BaseAgent):
    @abstractmethod
    def design_initial_prompt(self, task: TaskDefinition) -> str:
        pass

    @abstractmethod
    def design_mutation_prompt(self, task: TaskDefinition, parent_program: Program, evaluation_feedback: Optional[Dict] = None) -> str:
        pass

    @abstractmethod
    def design_bug_fix_prompt(self, task: TaskDefinition, program: Program, error_info: Dict) -> str:
        pass

class CodeGeneratorInterface(BaseAgent):
    @abstractmethod
    async def generate_code(self, prompt: str, model_name: Optional[str] = None, temperature: Optional[float] = 0.7, output_format: str = "code") -> str:
        pass

class EvaluatorAgentInterface(BaseAgent):
    @abstractmethod
    async def evaluate_program(self, program: Program, task: TaskDefinition) -> Program:
        pass

class DatabaseAgentInterface(BaseAgent):
    @abstractmethod
    async def save_program(self, program: Program):
        pass

    @abstractmethod
    async def get_program(self, program_id: str) -> Optional[Program]:
        pass

    @abstractmethod
    async def get_best_programs(self, task_id: str, limit: int = 10, objective: Optional[str] = None) -> List[Program]:
        pass
    
    @abstractmethod
    async def get_programs_for_next_generation(self, task_id: str, generation_size: int) -> List[Program]:
        pass

class SelectionControllerInterface(BaseAgent):
    @abstractmethod
    def select_parents(self, evaluated_programs: List[Program], num_parents: int) -> List[Program]:
        pass

    @abstractmethod
    def select_survivors(self, current_population: List[Program], offspring_population: List[Program], population_size: int) -> List[Program]:
        pass

# New interfaces for Island Model
class IslandManagerInterface(BaseAgent):
    @abstractmethod
    async def initialize_islands(self, num_islands: int, population_per_island: int) -> List[Island]:
        pass
    
    @abstractmethod
    async def evolve_islands_parallel(self, islands: List[Island], task: TaskDefinition) -> List[Island]:
        pass
    
    @abstractmethod
    async def migrate_between_islands(self, islands: List[Island], generation: int) -> List[Island]:
        pass

class MigrationPolicyInterface(BaseAgent):
    @abstractmethod
    def should_migrate(self, generation: int) -> bool:
        pass
    
    @abstractmethod
    def select_migrants(self, source_island: Island, num_migrants: int) -> List[Program]:
        pass
    
    @abstractmethod
    def select_destination_islands(self, source_island: Island, all_islands: List[Island]) -> List[Island]:
        pass

class MAPElitesInterface(BaseAgent):
    @abstractmethod
    def get_behavior_descriptor(self, program: Program) -> tuple:
        pass
    
    @abstractmethod
    def update_archive(self, program: Program) -> bool:
        pass
    
    @abstractmethod
    def get_diverse_programs(self, num_programs: int) -> List[Program]:
        pass

class RLFineTunerInterface(BaseAgent):
    @abstractmethod
    async def update_policy(self, experience_data: List[Dict]):
        pass

class MonitoringAgentInterface(BaseAgent):
    @abstractmethod
    async def log_metrics(self, metrics: Dict):
        pass

    @abstractmethod
    async def report_status(self):
        pass
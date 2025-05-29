import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_FLASH_MODEL_NAME = "gemini-1.5-flash"               # Use stable 1.5 Flash instead of 2.5 Flash Preview
GEMINI_PRO_MODEL_NAME = "gemini-1.5-pro"                   # Use stable 1.5 Pro
GEMINI_EVALUATION_MODEL = "gemini-2.5-flash-preview-05-20" # Use preview flash for evaluation

# Alternative: Use stable models to avoid preview model bugs
# GEMINI_FLASH_MODEL_NAME = "gemini-2.5-flash-preview-05-20"  # Preview model with known issues

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY must be set in .env file. Please create a .env file with your valid API key.")

# AlphaEvolve Core Settings
EVOLVE_BLOCK_PARSING = True                        # Enable EVOLVE-BLOCK parsing
ENSEMBLE_CODE_GENERATION = False                   # Disable old ensemble, use generation-based selection
MULTI_COMPONENT_EVOLUTION = True                   # Evolve multiple components simultaneously
FLASH_TO_PRO_TRANSITION_GENERATION = 20            # Switch from Flash to Pro at generation 20
                                    
POPULATION_SIZE = 12                               # Increased for better diversity
GENERATIONS = 40                                   # More generations for complex evolution
ELITISM_COUNT = 2                                  # Keep top performers                      
MUTATION_RATE = 0.8                               # Higher mutation for exploration                          
CROSSOVER_RATE = 0.2                                                                          

# Island Model Settings
ENABLE_ISLAND_MODEL = True                         # Enable island-based evolution
NUM_ISLANDS = 4                                    # Increased island count
POPULATION_PER_ISLAND = 8                          # Balanced population
MIGRATION_INTERVAL = 4                             # Migration frequency
MIGRATION_RATE = 0.2                               # Migration proportion
MIGRATION_TOPOLOGY = "ring"                        # ring, fully_connected, star, random
ELITE_MIGRATION = True                             # Migrate best individuals

# AlphaEvolve Evolution Strategy
ALPHA_EVOLVE_MODE = True                          # Enable AlphaEvolve features
COMPONENT_EVOLUTION_PROBABILITY = {               # Per-component evolution rates
    'imports': 0.3,
    'functions': 0.8,
    'configs': 0.6,
    'classes': 0.4,
    'general': 0.5
}

# Code Generation Settings
CODE_GENERATION_TEMPERATURE = 0.7                # Default temperature
FLASH_TEMPERATURE = 0.9                          # Higher for exploration
PRO_TEMPERATURE = 0.3                            # Lower for refinement
MAX_EVOLVE_BLOCKS_PER_MUTATION = 3               # Max blocks to evolve simultaneously

# Evaluation Settings
EVALUATION_TIMEOUT_SECONDS = 30                  
USE_DETAILED_EVALUATION = True                   # Enhanced MOLS evaluation
MOLS_SIZE = 3                                    # Target MOLS size

# API Settings
API_MAX_RETRIES = 3                              
API_RETRY_DELAY_SECONDS = 2                      

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "alpha_evolve.log"
ENABLE_DEBUG_LOGGING = True

# GPU Settings (Disabled for CPU-only)
ENABLE_GPU_ACCELERATION = False                   # CPU mode
USE_MULTI_GPU = False                             
GPU_DEVICES = ["cuda:0", "cuda:1"]                
GPU_DEVICE = "cuda:0"                             
GPU_NUM_ISLANDS = 8                               
GPU_POPULATION_PER_ISLAND = 25                   
GPU_MIGRATION_INTERVAL = 4                       
GPU_MIGRATION_RATE = 0.2                         
GPU_MIGRATION_TOPOLOGY = "ring"                  
GPU_BATCH_SIZE = 64                              
GPU_USE_MIXED_PRECISION = True                   
GPU_NUM_WORKERS = 4
GPU_ENABLE_PROFILING = True
GPU_MEMORY_OPTIMIZATION = True
GPU_EMPTY_CACHE_INTERVAL = 20
GPU_CUDA_BENCHMARK = True
GPU_CUDA_DETERMINISTIC = False

# MAP-Elites Settings
ENABLE_MAP_ELITES = True                          # Enable MAP-Elites archive
MAP_ELITES_CONFIG = "default"                     # default, detailed, simple
BEHAVIOR_DIMENSIONS = ["code_complexity", "execution_time", "solution_approach"]
DIMENSION_BINS = {
    "code_complexity": 5,
    "execution_time": 5,
    "memory_usage": 3,
    "solution_approach": 8
}

# GPU MAP-Elites Settings
GPU_MAP_ELITES_ENABLE = True
GPU_MAP_ELITES_BATCH_SIZE = 128
GPU_BEHAVIOR_DIMENSIONS = ["code_complexity", "execution_time", "memory_usage", "solution_approach"]
GPU_DIMENSION_BINS = {
    "code_complexity": 15,
    "execution_time": 15,
    "memory_usage": 12,
    "solution_approach": 12
}
GPU_ARCHIVE_SIZE_LIMIT = 20000
GPU_CACHE_BEHAVIOR_VECTORS = True

# CPU Parallel Processing Settings
MAX_PARALLEL_ISLANDS = 4
ISLAND_EVOLUTION_TIMEOUT = 300
USE_PROCESS_POOL = False
PARALLEL_EVALUATION_BATCH_SIZE = 5

# Advanced Evolution Settings
ENABLE_EVALUATION_CASCADE = True
QUICK_EVALUATION_TIMEOUT = 5
FULL_EVALUATION_TIMEOUT = 30

# Multi-objective Optimization
ENABLE_MULTI_OBJECTIVE = True
OBJECTIVES = ["correctness", "efficiency", "simplicity"]
OBJECTIVE_WEIGHTS = {
    "correctness": 0.6,
    "efficiency": 0.3,
    "simplicity": 0.1
}

# Monitoring
MONITORING_DASHBOARD_URL = "http://localhost:8080"

def get_gpu_config():
    """Get GPU configuration dictionary"""
    return {
        "enabled": ENABLE_GPU_ACCELERATION,
        "use_multi_gpu": USE_MULTI_GPU,
        "devices": GPU_DEVICES,
        "primary_device": GPU_DEVICE,
        "num_islands": GPU_NUM_ISLANDS,
        "population_per_island": GPU_POPULATION_PER_ISLAND,
        "migration_interval": GPU_MIGRATION_INTERVAL,
        "migration_rate": GPU_MIGRATION_RATE,
        "batch_size": GPU_BATCH_SIZE,
        "mixed_precision": GPU_USE_MIXED_PRECISION,
        "num_workers": GPU_NUM_WORKERS,
        "enable_profiling": GPU_ENABLE_PROFILING,
        "memory_optimization": GPU_MEMORY_OPTIMIZATION,
        "empty_cache_interval": GPU_EMPTY_CACHE_INTERVAL,
        "cuda_benchmark": GPU_CUDA_BENCHMARK,
        "cuda_deterministic": GPU_CUDA_DETERMINISTIC
    }

def get_alpha_evolve_config():
    """Get AlphaEvolve-specific configuration"""
    return {
        "mode_enabled": ALPHA_EVOLVE_MODE,
        "evolve_block_parsing": EVOLVE_BLOCK_PARSING,
        "ensemble_generation": ENSEMBLE_CODE_GENERATION,
        "multi_component_evolution": MULTI_COMPONENT_EVOLUTION,
        "component_probabilities": COMPONENT_EVOLUTION_PROBABILITY,
        "max_blocks_per_mutation": MAX_EVOLVE_BLOCKS_PER_MUTATION,
        "flash_temperature": FLASH_TEMPERATURE,
        "pro_temperature": PRO_TEMPERATURE
    }

def get_island_config():
    """Get island model configuration"""
    return {
        "enable_island_model": ENABLE_ISLAND_MODEL,
        "num_islands": NUM_ISLANDS,
        "population_per_island": POPULATION_PER_ISLAND,
        "migration_interval": MIGRATION_INTERVAL,
        "migration_rate": MIGRATION_RATE,
        "topology": MIGRATION_TOPOLOGY,
        "elite_migration": ELITE_MIGRATION
    }

def get_map_elites_config():
    """Get MAP-Elites configuration"""
    return {
        "enable_map_elites": ENABLE_MAP_ELITES,
        "config_name": MAP_ELITES_CONFIG,
        "behavior_dimensions": BEHAVIOR_DIMENSIONS,
        "dimension_bins": DIMENSION_BINS
    }

def get_gpu_map_elites_config():
    """Get GPU MAP-Elites configuration"""
    return {
        "enable_map_elites": GPU_MAP_ELITES_ENABLE,
        "gpu_batch_size": GPU_MAP_ELITES_BATCH_SIZE,
        "behavior_dimensions": GPU_BEHAVIOR_DIMENSIONS,
        "dimension_bins": GPU_DIMENSION_BINS,
        "archive_size_limit": GPU_ARCHIVE_SIZE_LIMIT,
        "cache_behavior_vectors": GPU_CACHE_BEHAVIOR_VECTORS
    }

def get_parallel_config():
    """Get parallel processing configuration"""
    return {
        "max_parallel_islands": MAX_PARALLEL_ISLANDS,
        "island_evolution_timeout": ISLAND_EVOLUTION_TIMEOUT,
        "use_process_pool": USE_PROCESS_POOL,
        "parallel_evaluation_batch_size": PARALLEL_EVALUATION_BATCH_SIZE
    }

def get_setting(key, default=None):
    """Retrieves a setting value"""
    return globals().get(key, default)

def get_llm_model(model_type="pro"):
    """Get LLM model name"""
    if model_type == "pro":
        return GEMINI_PRO_MODEL_NAME
    elif model_type == "flash":
        return GEMINI_FLASH_MODEL_NAME
    return GEMINI_FLASH_MODEL_NAME

                                 

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# If no API key is found in environment variables,
# use a placeholder (non-functional)
if not GEMINI_API_KEY:
    # This is a placeholder. You MUST replace this with your actual API key
    # either by setting the GEMINI_API_KEY environment variable
    # or by creating a .env file in the project root with:
    # GEMINI_API_KEY="your_actual_api_key_here"
    # You can get your API key from: https://aistudio.google.com/app/apikey
    print("Warning: GEMINI_API_KEY not found in .env or environment. Using a NON-FUNCTIONAL placeholder. Please create a .env file with your valid API key.")
    GEMINI_API_KEY = "Your Gemini Api key"  # This will NOT work

# LLM Model Configuration
GEMINI_PRO_MODEL_NAME = "gemini-2.5-flash-preview-04-17"  # Primary model for generation
GEMINI_FLASH_MODEL_NAME = "gemini-2.5-flash-preview-04-17"  # Faster model for evaluation
GEMINI_EVALUATION_MODEL = "gemini-2.5-flash-preview-04-17"  # Model specifically for evaluation

# Evolution Algorithm Parameters
POPULATION_SIZE = 10  # Total population size
GENERATIONS = 50  # Number of generations to evolve (increased)
ELITISM_COUNT = 1  # Number of best individuals to preserve each generation
MUTATION_RATE = 0.7  # Probability of mutation
CROSSOVER_RATE = 0.2  # Probability of crossover (currently not implemented)

# Island Model Settings
ENABLE_ISLAND_MODEL = True  # Enable/disable island model
NUM_ISLANDS = 4  # Number of islands
POPULATION_PER_ISLAND = 8  # Population size per island
MIGRATION_INTERVAL = 5  # Migration every N generations
MIGRATION_RATE = 0.1  # Percentage of population to migrate
MIGRATION_TOPOLOGY = "ring"  # ring, fully_connected, star, random
ELITE_MIGRATION = True  # Migrate best individuals

# MAP-Elites Settings
ENABLE_MAP_ELITES = True  # Enable MAP-Elites archive
MAP_ELITES_CONFIG = "default"  # default, detailed, simple
BEHAVIOR_DIMENSIONS = ["code_complexity", "execution_time", "solution_approach"]
DIMENSION_BINS = {
    "code_complexity": 5,
    "execution_time": 5,
    "memory_usage": 3,
    "solution_approach": 8
}

# GPU Island Model Settings (RTX 3080 Optimized)
ENABLE_GPU_ACCELERATION = True  # Enable GPU acceleration
GPU_DEVICE = "cuda:0"  # GPU device
GPU_NUM_ISLANDS = 10  # Number of islands for GPU processing (increased)
GPU_POPULATION_PER_ISLAND = 30  # Population per island (increased)
GPU_MIGRATION_INTERVAL = 5  # Migration interval
GPU_MIGRATION_RATE = 0.15  # Migration rate (higher for GPU)
GPU_MIGRATION_TOPOLOGY = "ring"  # Migration topology
GPU_BATCH_SIZE = 64  # GPU batch processing size (increased for larger population)
GPU_USE_MIXED_PRECISION = True  # Use mixed precision training
GPU_NUM_WORKERS = 6  # Number of GPU workers (increased)

# GPU MAP-Elites Settings
GPU_MAP_ELITES_ENABLE = True  # Enable GPU MAP-Elites
GPU_MAP_ELITES_BATCH_SIZE = 128  # GPU batch size for MAP-Elites (increased)
GPU_BEHAVIOR_DIMENSIONS = ["code_complexity", "execution_time", "memory_usage", "solution_approach"]
GPU_DIMENSION_BINS = {
    "code_complexity": 15,  # Higher resolution with GPU
    "execution_time": 15,
    "memory_usage": 12,
    "solution_approach": 12
}
GPU_ARCHIVE_SIZE_LIMIT = 10000  # Larger archive with GPU memory (increased)
GPU_CACHE_BEHAVIOR_VECTORS = True  # Cache behavior vectors in GPU memory

# GPU Performance Settings
GPU_ENABLE_PROFILING = True  # Enable GPU performance profiling
GPU_MEMORY_OPTIMIZATION = True  # Enable memory optimization
GPU_EMPTY_CACHE_INTERVAL = 20  # Clean GPU cache every N generations
GPU_CUDA_BENCHMARK = True  # Enable cuDNN benchmark mode
GPU_CUDA_DETERMINISTIC = False  # Disable for better performance

# CPU Parallel Processing Settings (Fallback)
MAX_PARALLEL_ISLANDS = 4  # Max islands to process in parallel
ISLAND_EVOLUTION_TIMEOUT = 300  # Timeout for island evolution (seconds)
USE_PROCESS_POOL = False  # Use ProcessPool instead of ThreadPool
PARALLEL_EVALUATION_BATCH_SIZE = 5  # Batch size for parallel evaluation

# Timeout Settings
EVALUATION_TIMEOUT_SECONDS = 30
CODE_GENERATION_TIMEOUT_SECONDS = 120
DIFF_APPLICATION_TIMEOUT_SECONDS = 10

# Database Configuration (for storing programs and their fitness)
DATABASE_TYPE = "in_memory"  # "in_memory" or "file" (file not implemented yet)
DATABASE_PATH = "program_database.json"  # Path for file-based database

# Logging Configuration
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "alpha_evolve.log"

# API Configuration
API_MAX_RETRIES = 3
API_RETRY_DELAY_SECONDS = 5
MAX_CONCURRENT_EVALUATIONS = 5
MAX_CONCURRENT_GENERATIONS = 3  # Max concurrent generation requests

# RL Fine-tuner Configuration (placeholder for future implementation)
RL_TRAINING_INTERVAL_GENERATIONS = 50  # Train RL model every N generations
RL_MODEL_PATH = "rl_finetuner_model.pth"

# Monitoring Configuration
MONITORING_DASHBOARD_URL = "http://localhost:8080"  # Placeholder

# Advanced Evolution Settings
ENABLE_EVALUATION_CASCADE = True  # Enable two-stage evaluation
QUICK_EVALUATION_TIMEOUT = 5  # Quick evaluation timeout (seconds)
FULL_EVALUATION_TIMEOUT = 30  # Full evaluation timeout (seconds)

# Multi-objective Optimization
ENABLE_MULTI_OBJECTIVE = True  # Enable multi-objective optimization
OBJECTIVES = ["correctness", "efficiency", "simplicity"]  # List of objectives to optimize
OBJECTIVE_WEIGHTS = {  # Weights for combining objectives
    "correctness": 0.6,
    "efficiency": 0.3,
    "simplicity": 0.1
}

# Utility functions for accessing settings
def get_setting(key, default=None):
    """
    Retrieves a setting value.
    For LLM models, it specifically checks if the primary choice is available,
    otherwise falls back to a secondary/default if defined.
    """
    # This is a simple implementation. In the future, we might add
    # logic to check model availability or handle environment-specific overrides.
    return globals().get(key, default)

# Helper function to get the appropriate LLM model based on use case
def get_llm_model(model_type="pro"):
    if model_type == "pro":
        return GEMINI_PRO_MODEL_NAME
    elif model_type == "flash":
        return GEMINI_FLASH_MODEL_NAME
    return GEMINI_FLASH_MODEL_NAME  # Default fallback

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

def get_gpu_config():
    """Get GPU acceleration configuration"""
    return {
        "enable_gpu_acceleration": ENABLE_GPU_ACCELERATION,
        "device": GPU_DEVICE,
        "num_islands": GPU_NUM_ISLANDS,
        "population_per_island": GPU_POPULATION_PER_ISLAND,
        "migration_interval": GPU_MIGRATION_INTERVAL,
        "migration_rate": GPU_MIGRATION_RATE,
        "topology": GPU_MIGRATION_TOPOLOGY,
        "gpu_batch_size": GPU_BATCH_SIZE,
        "use_mixed_precision": GPU_USE_MIXED_PRECISION,
        "num_gpu_workers": GPU_NUM_WORKERS,
        "enable_profiling": GPU_ENABLE_PROFILING,
        "memory_optimization": GPU_MEMORY_OPTIMIZATION,
        "empty_cache_interval": GPU_EMPTY_CACHE_INTERVAL,
        "cuda_benchmark": GPU_CUDA_BENCHMARK,
        "cuda_deterministic": GPU_CUDA_DETERMINISTIC
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

# End of settings.py
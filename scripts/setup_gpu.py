#!/usr/bin/env python3
"""
GPU Setup and Test Script for RTX 3080
Tests GPU availability and performance for OpenAlpha_Evolve
"""
import sys
import subprocess
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """GPU 사용 가능성 확인"""
    try:
        import torch
        
        logger.info("PyTorch GPU Availability Check:")
        logger.info(f"  PyTorch version: {torch.__version__}")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"  CUDA version: {torch.version.cuda}")
            logger.info(f"  GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # RTX 3080 확인
            gpu_name = torch.cuda.get_device_name(0)
            if "3080" in gpu_name or "RTX 3080" in gpu_name:
                logger.info("✅ RTX 3080 detected!")
                return True
            else:
                logger.warning(f"⚠️  Expected RTX 3080, but found: {gpu_name}")
                return True  # 다른 GPU라도 CUDA 지원하면 사용 가능
        else:
            logger.error("❌ CUDA not available")
            return False
            
    except ImportError:
        logger.error("❌ PyTorch not installed")
        return False
    except Exception as e:
        logger.error(f"❌ Error checking GPU: {e}")
        return False

def test_gpu_performance():
    """GPU 성능 테스트"""
    try:
        import torch
        import time
        
        if not torch.cuda.is_available():
            logger.error("GPU not available for performance test")
            return False
        
        device = torch.device("cuda:0")
        logger.info(f"Testing GPU performance on {device}")
        
        # 메모리 정리
        torch.cuda.empty_cache()
        
        # 성능 테스트 1: 행렬 곱셈
        logger.info("Performance Test 1: Matrix Multiplication")
        size = 4096
        
        start_time = time.time()
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        logger.info(f"  GPU Matrix Multiplication ({size}x{size}): {gpu_time:.3f}s")
        
        # CPU 비교
        start_time = time.time()
        a_cpu = torch.randn(size, size)
        b_cpu = torch.randn(size, size)
        c_cpu = torch.matmul(a_cpu, b_cpu)
        cpu_time = time.time() - start_time
        
        logger.info(f"  CPU Matrix Multiplication ({size}x{size}): {cpu_time:.3f}s")
        logger.info(f"  GPU Speedup: {cpu_time/gpu_time:.1f}x")
        
        # 성능 테스트 2: 배치 처리
        logger.info("Performance Test 2: Batch Processing")
        batch_size = 64
        vector_size = 1024
        
        start_time = time.time()
        batch_data = torch.randn(batch_size, vector_size, device=device)
        
        # 여러 GPU 연산 수행
        for _ in range(100):
            result = torch.sum(batch_data, dim=1)
            result = torch.softmax(result, dim=0)
            result = torch.topk(result, k=10)
        
        torch.cuda.synchronize()
        batch_time = time.time() - start_time
        
        logger.info(f"  GPU Batch Processing (100 iterations): {batch_time:.3f}s")
        
        # 메모리 사용량 확인
        memory_allocated = torch.cuda.memory_allocated(device) / 1e9
        memory_reserved = torch.cuda.memory_reserved(device) / 1e9
        memory_total = torch.cuda.get_device_properties(device).total_memory / 1e9
        
        logger.info(f"GPU Memory Usage:")
        logger.info(f"  Allocated: {memory_allocated:.2f} GB")
        logger.info(f"  Reserved: {memory_reserved:.2f} GB")
        logger.info(f"  Total: {memory_total:.2f} GB")
        logger.info(f"  Utilization: {(memory_allocated/memory_total)*100:.1f}%")
        
        # 메모리 정리
        torch.cuda.empty_cache()
        
        logger.info("✅ GPU performance test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ GPU performance test failed: {e}")
        return False

def test_island_model_gpu():
    """Island Model GPU 기능 테스트"""
    try:
        logger.info("Testing Island Model GPU Components...")
        
        # GPU Island Manager 테스트
        from core.gpu_island_manager import GPUIslandManager
        
        config = {
            "gpu_batch_size": 16,
            "use_mixed_precision": True,
            "num_gpu_workers": 2
        }
        
        island_manager = GPUIslandManager(config)
        logger.info("✅ GPU Island Manager initialized")
        
        # GPU Migration Policy 테스트
        from core.gpu_migration_policy import GPUMigrationPolicy
        
        migration_config = {
            "migration_interval": 5,
            "base_migration_rate": 0.1,
            "topology": "RING"
        }
        
        migration_policy = GPUMigrationPolicy(migration_config)
        logger.info("✅ GPU Migration Policy initialized")
        
        # GPU MAP-Elites 테스트
        from core.gpu_map_elites import GPUMAPElites
        
        map_elites_config = {
            "behavior_space": {
                "code_complexity": {"bounds": (0.0, 1.0), "resolution": 10},
                "execution_time": {"bounds": (0.0, 1.0), "resolution": 10},
                "memory_usage": {"bounds": (0.0, 1.0), "resolution": 8},
                "solution_approach": {"bounds": (0.0, 1.0), "resolution": 8}
            },
            "gpu_batch_size": 32,
            "use_mixed_precision": True
        }
        
        map_elites = GPUMAPElites(map_elites_config)
        logger.info("✅ GPU MAP-Elites initialized")
        
        # GPU Task Manager 테스트
        from task_manager.gpu_island_task_manager import GPUIslandTaskManager
        
        task_manager_config = {
            "max_generations": 5,
            "population_size": 16,
            "num_islands": 4,
            "gpu_batch_size": 8,
            "use_mixed_precision": True,
            "enable_gpu_profiling": True
        }
        
        task_manager = GPUIslandTaskManager(task_manager_config)
        logger.info("✅ GPU Island Task Manager initialized")
        
        # GPU 통계 확인
        gpu_stats = island_manager.get_gpu_statistics()
        logger.info("GPU Statistics:")
        for key, value in gpu_stats.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("✅ All GPU components test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Island Model GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def install_pytorch_gpu():
    """PyTorch GPU 버전 설치"""
    try:
        logger.info("Installing PyTorch with CUDA support...")
        
        # CUDA 11.8 버전용 PyTorch 설치 (RTX 3080 호환)
        install_command = [
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ]
        
        result = subprocess.run(install_command, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ PyTorch GPU installation completed")
            return True
        else:
            logger.error(f"❌ PyTorch GPU installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error installing PyTorch GPU: {e}")
        return False

def main():
    """메인 실행 함수"""
    logger.info("=" * 60)
    logger.info("OpenAlpha_Evolve GPU Setup and Test")
    logger.info("=" * 60)
    
    # 1. GPU 사용 가능성 확인
    logger.info("Step 1: Checking GPU availability...")
    if not check_gpu_availability():
        logger.info("Attempting to install PyTorch with GPU support...")
        if install_pytorch_gpu():
            logger.info("Please restart the script after installation")
        return False
    
    # 2. GPU 성능 테스트
    logger.info("\nStep 2: Testing GPU performance...")
    if not test_gpu_performance():
        return False
    
    # 3. Island Model GPU 컴포넌트 테스트
    logger.info("\nStep 3: Testing Island Model GPU components...")
    if not test_island_model_gpu():
        return False
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 GPU setup and test completed successfully!")
    logger.info("Your RTX 3080 is ready for OpenAlpha_Evolve GPU acceleration")
    logger.info("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 
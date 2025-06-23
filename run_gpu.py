#!/usr/bin/env python3
"""
GPU Island Manager 실행 스크립트
RTX-3080에서 AlphaEvolve 테스트
"""
import asyncio
import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app_gpu import AlphaEvolveGPUApp

async def test_gpu_setup():
    """GPU 설정 테스트"""
    print("🔧 GPU Island Manager 테스트 시작...")
    
    app = AlphaEvolveGPUApp()
    app.print_gpu_info()
    
    # 간단한 문제로 테스트
    print("\n🧪 간단한 리스트 합계 문제로 테스트...")
    
    results = await app.run_general_evolution(
        description="리스트의 모든 요소의 합을 구하는 함수를 작성하세요",
        examples=[
            {"input": [1, 2, 3], "output": 6},
            {"input": [], "output": 0},
            {"input": [5, -2, 7], "output": 10}
        ]
    )
    
    print("\n📊 테스트 결과:")
    print(f"실행 시간: {results.get('execution_time', 0):.2f}초")
    
    if 'best_solution' in results and results['best_solution']:
        best = results['best_solution']
        print(f"최고 적합도: {best['fitness_scores']}")
        print(f"생성된 코드:")
        print("=" * 50)
        print(best['code'])
        print("=" * 50)
    else:
        print("❌ 해결책을 찾지 못했습니다.")
    
    print("\n✅ GPU Island Manager 테스트 완료!")

if __name__ == "__main__":
    print("🚀 AlphaEvolve GPU Island Manager")
    print("RTX-3080 최적화 진화 시스템")
    print("=" * 50)
    
    try:
        asyncio.run(test_gpu_setup())
    except KeyboardInterrupt:
        print("\n⏹️  사용자에 의해 중단됨")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc() 
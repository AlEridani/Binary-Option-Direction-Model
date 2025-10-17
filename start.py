import sys
import os
from datetime import datetime
import pandas as pd

# 시스템 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from main_pipe import MainPipeline
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def quick_start():
    """빠른 시작 - 시뮬레이션 모드"""
    print("="*60)
    print("바이너리 옵션 예측 시스템 - 빠른 시작")
    print("="*60)

    # 1. 디렉토리 생성
    Config.create_directories()
    print("✓ 디렉토리 구조 생성 완료")

    # 2. 파이프라인 초기화 (모델 없으면 초기학습까지 수행)
    pipeline = MainPipeline()
    print("✓ 파이프라인 인스턴스 생성")

    # 3. 시뮬레이션 데이터 생성
    print("\n초기 학습을 위한 시뮬레이션 데이터 생성 중...")
    initial_data = pipeline.generate_simulation_data(days=270)
    print(f"✓ {len(initial_data)} 개의 데이터 포인트 생성 완료")

    # 4. 초기 모델 학습
    print("\n초기 모델 학습 중... (약 1-2분 소요)")
    metrics = pipeline.model_optimizer.initial_training(initial_data)
    tr_wr = metrics['train'].get('win_rate', None)
    va_wr = metrics['validation'].get('win_rate', None)
    te_wr = metrics['test'].get('win_rate', None)

    print("\n학습 결과:")
    if tr_wr is not None: print(f"- 학습 승률: {tr_wr:.2%}")
    if va_wr is not None: print(f"- 검증 승률: {va_wr:.2%}")
    if te_wr is not None: print(f"- 테스트 승률: {te_wr:.2%}")

    # 5. 실시간/백테스트용 컴포넌트 연결
    ok = pipeline.initialize_system()
    if not ok:
        print("초기화 실패: 모델/트레이더 준비 중 오류")
        return pipeline

    # 6. 백테스팅 (파이프라인 내부 run_backtest로 수행)
    print("\n백테스팅 실행 중...")
    pipeline.run_backtest()  # ✨ 변경 포인트: real_trader.backtest() → run_backtest()

    print("\n시스템 준비 완료!")
    print("="*60)
    return pipeline


def main_menu():
    """메인 메뉴"""
    print("\n" + "="*60)
    print("바이너리 옵션 예측 시스템 v1.0")
    print("="*60)
    print("\n실행 모드를 선택하세요:")
    print("1. 빠른 시작 (시뮬레이션)")
    print("2. 실시간 거래 (API 연결)")
    print("3. 백테스팅")
    print("4. 모델 재학습")
    print("5. 성능 리포트")
    print("6. 하이퍼파라미터 최적화")
    print("0. 종료")
    
    choice = input("\n선택 (0-6): ").strip()
    
    if choice == "1":
        # 빠른 시작
        pipeline = quick_start()
        
        # 시뮬레이션 거래 시작
        print("\n시뮬레이션 거래를 시작하시겠습니까? (y/n): ", end="")
        if input().lower() == 'y':
            print("시뮬레이션 거래 시작 (1시간)...")
            pipeline.real_trader.run_live_trading(duration_hours=1)  # 필요시 amount=100.0 지정 가능
            
    elif choice == "2":
        # 실시간 거래
        print("\n⚠️  실제 API 연결이 필요합니다.")
        print("API 키를 설정하셨습니까? (y/n): ", end="")
        if input().lower() == 'y':
            pipeline = MainPipeline()
            pipeline.start()
        else:
            print("config.py에서 API 키를 설정해주세요.")
            
    elif choice == "3":
        # 백테스팅
        print("\n백테스팅 설정:")
        start_date = input("시작 날짜 (YYYY-MM-DD, Enter로 스킵): ").strip()
        end_date = input("종료 날짜 (YYYY-MM-DD, Enter로 스킵): ").strip()
        
        pipeline = MainPipeline()
        pipeline.run_backtest(
            start_date=pd.to_datetime(start_date) if start_date else None,
            end_date=pd.to_datetime(end_date) if end_date else None
        )
        
    elif choice == "4":
        # 모델 재학습
        pipeline = MainPipeline()
        if pipeline.initialize_system():
            pipeline.retrain_pipeline()
            
    elif choice == "5":
        # 성능 리포트
        pipeline = MainPipeline()
        pipeline.monitor.generate_report()
        
    elif choice == "6":
        # 하이퍼파라미터 최적화
        pipeline = MainPipeline()
        pipeline.optimize_hyperparameters()
        
    elif choice == "0":
        print("프로그램을 종료합니다.")
        sys.exit(0)
    else:
        print("잘못된 선택입니다.")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n프로그램이 중단되었습니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()

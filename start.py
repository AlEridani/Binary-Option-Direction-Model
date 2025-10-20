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
    
    # 2. 파이프라인 초기화
    pipeline = MainPipeline()
    print("✓ 파이프라인 초기화 완료")
    
    # 3. 초기 학습 데이터 생성 (시뮬레이션)
    print("\n초기 학습을 위한 시뮬레이션 데이터 생성 중...")
    initial_data = pipeline.generate_simulation_data(days=270)
    print(f"✓ {len(initial_data)} 개의 데이터 포인트 생성 완료")
    
    # 4. 초기 모델 학습
    print("\n초기 모델 학습 중... (약 1-2분 소요)")
    metrics = pipeline.model_optimizer.initial_training(initial_data)
    
    # 레짐별 결과 출력
    print("\n학습 결과 (레짐별):")
    print("="*60)
    
    for regime_name in ['UP', 'DOWN', 'FLAT']:
        if regime_name in metrics:
            regime_metrics = metrics[regime_name]
            print(f"\n[{regime_name} 레짐]")
            print(f"  - 학습 승률:   {regime_metrics['train']['win_rate']:.2%}")
            print(f"  - 검증 승률:   {regime_metrics['validation']['win_rate']:.2%}")
            print(f"  - 테스트 승률: {regime_metrics['test']['win_rate']:.2%}")
    
    # 전체 평균 계산
    all_test_rates = []
    for regime_name, regime_metrics in metrics.items():
        all_test_rates.append(regime_metrics['test']['win_rate'])
    
    if all_test_rates:
        avg_test_rate = sum(all_test_rates) / len(all_test_rates)
        print(f"\n전체 평균 테스트 승률: {avg_test_rate:.2%}")
    
    # 5. 백테스팅
    if pipeline.real_trader is not None:
        print("\n백테스팅 실행 중...")
        test_data = initial_data.tail(10000)  # 최근 10000개 데이터로 백테스트
        results = pipeline.real_trader.backtest(test_data)
    else:
        print("\n⚠️  실시간 거래 모듈이 초기화되지 않았습니다.")
        print("시스템 초기화 중...")
        if pipeline.initialize_system():
            test_data = initial_data.tail(10000)
            results = pipeline.real_trader.backtest(test_data)
    
    return pipeline

def main_menu():
    """메인 메뉴"""
    print("\n" + "="*60)
    print("바이너리 옵션 예측 시스템 v1.0 (레짐 스위칭)")
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
            duration = input("실행 시간 (시간, Enter=1): ").strip()
            duration = int(duration) if duration else 1
            print(f"시뮬레이션 거래 시작 ({duration}시간)...")
            pipeline.real_trader.run_live_trading(duration_hours=duration)
            
    elif choice == "2":
        # 실시간 거래
        print("\n⚠️  실제 API 연결이 필요합니다.")
        print("API 키를 설정하셨습니까? (y/n): ", end="")
        if input().lower() == 'y':
            pipeline = MainPipeline()
            
            # 모델이 없으면 먼저 학습
            if not os.path.exists(os.path.join(Config.MODEL_DIR, 'bundle_UP.pkl')):
                print("\n학습된 모델이 없습니다. 초기 학습을 진행합니다...")
                initial_data = pipeline.generate_simulation_data(days=270)
                pipeline.model_optimizer.initial_training(initial_data)
            
            duration = input("실행 시간 (시간, Enter=무제한): ").strip()
            duration = int(duration) if duration else 99999
            
            if pipeline.initialize_system():
                pipeline.real_trader.run_live_trading(duration_hours=duration)
        else:
            print("config.py에서 API 키를 설정해주세요.")
            
    elif choice == "3":
        # 백테스팅
        print("\n백테스팅 설정:")
        start_date = input("시작 날짜 (YYYY-MM-DD, Enter로 스킵): ").strip()
        end_date = input("종료 날짜 (YYYY-MM-DD, Enter로 스킵): ").strip()
        
        pipeline = MainPipeline()
        
        # 모델 체크
        if not pipeline.initialize_system():
            print("\n모델을 먼저 학습해주세요. (메뉴 1번)")
            return
        
        # 데이터 생성 또는 로드
        print("\n데이터 생성 중...")
        data = pipeline.generate_simulation_data(days=90)
        
        results = pipeline.real_trader.backtest(
            data,
            start_date=pd.to_datetime(start_date) if start_date else None,
            end_date=pd.to_datetime(end_date) if end_date else None
        )
        
    elif choice == "4":
        # 모델 재학습
        pipeline = MainPipeline()
        
        print("\n새로운 데이터 생성 중...")
        new_data = pipeline.generate_simulation_data(days=270)
        
        print("\n모델 재학습 중...")
        metrics = pipeline.model_optimizer.retrain_model(new_data)
        
        print("\n재학습 완료!")
        for regime_name in ['UP', 'DOWN', 'FLAT']:
            if regime_name in metrics:
                regime_metrics = metrics[regime_name]
                print(f"\n[{regime_name}]")
                print(f"  테스트 승률: {regime_metrics['test']['win_rate']:.2%}")

    elif choice == "5":
        # 성능 리포트
        pipeline = MainPipeline()
        
        if not pipeline.initialize_system():
            print("\n거래 기록이 없거나 모델이 없습니다.")
            return
        
        pipeline.monitor.generate_report()
        
    elif choice == "6":
        # 하이퍼파라미터 최적화
        print("\n하이퍼파라미터 최적화는 개발 중입니다.")
        print("현재는 config.py에서 수동으로 조정해주세요.")
        
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

"""
================================================================================
                        사용 가이드 (README)
================================================================================

# 바이너리 옵션 예측 시스템

## 시스템 개요
- 바이낸스 바이너리 옵션을 위한 자동 거래 시스템
- LightGBM 앙상블 모델을 사용한 10분 후 가격 예측
- 목표 승률: 60% 이상
- 자동 재학습 파이프라인 포함

## 설치 방법

### 1. 필요 패키지 설치
```bash
pip install pandas numpy lightgbm scikit-learn joblib requests schedule
```

### 2. 프로젝트 구조 생성
```bash
python start.py
# 옵션 1 선택 (빠른 시작)
```

## 사용 방법

### 1. 시뮬레이션 모드 (추천)
```bash
python start.py
# 옵션 1 선택
```
- 시뮬레이션 데이터로 학습 및 백테스팅
- API 키 없이 시스템 테스트 가능

### 2. 실시간 거래 모드
```bash
python main_pipe.py --mode live
```
- 실제 바이낸스 API 연결 필요
- config.py에서 API 키 설정 필수

### 3. 백테스팅
```bash
python main_pipe.py --mode backtest --start-date 2024-01-01 --end-date 2024-12-31
```

### 4. 모델 학습
```bash
python main_pipe.py --mode train
```

### 5. 하이퍼파라미터 최적화
```bash
python main_pipe.py --mode optimize
```

## 주요 기능

### 자동 재학습
- 50거래마다 승률 체크
- 55% 미만 시 자동 재학습
- 실패 패턴 분석 및 필터 생성

### 거래 필터
- 높은 변동성 필터
- 낮은 거래량 필터
- 시간대 기반 필터

### 성능 모니터링
- 실시간 승률 추적
- 일별/주별/월별 통계
- 자동 리포트 생성

## 설정 파일 (config.py)

주요 설정:
- PREDICTION_WINDOW: 10 (10분 예측)
- TARGET_WIN_RATE: 0.60 (목표 승률 60%)
- RETRAIN_THRESHOLD: 0.55 (재학습 임계값)
- EVALUATION_WINDOW: 50 (평가 주기)

## 폴더 구조

```
project/
├── models/           # 학습된 모델
├── price_data/       # 가격 데이터
├── feature_log/      # 피처 로그
├── trade_log/        # 거래 기록
├── result/           # 병합 데이터
├── main_pipe.py      # 메인 파이프라인
├── data_merge.py     # 데이터 병합
├── model_train.py    # 모델 학습
├── real_trade.py     # 실시간 거래
├── config.py         # 설정
└── start.py          # 시작 스크립트
```

## 주의사항

1. **리스크 관리**
   - 시뮬레이션으로 충분히 테스트 후 실거래
   - 소액으로 시작
   - 손실 한도 설정

2. **API 제한**
   - 바이낸스 API 레이트 리밋 준수
   - 과도한 요청 방지

3. **데이터 관리**
   - 90일 이상 오래된 데이터 자동 정리
   - 정기적인 백업 권장

## 트러블슈팅

### 모델이 로드되지 않음
```bash
# 초기 학습 실행
python main_pipe.py --mode train
```

### 승률이 낮음
```bash
# 하이퍼파라미터 최적화
python main_pipe.py --mode optimize
```

### 메모리 부족
- feature_selection의 top_k 값 감소
- ENSEMBLE_MODELS 수 감소

## 성능 향상 팁

1. **피처 엔지니어링**
   - 추가 기술지표 구현
   - 시장 미시구조 피처 추가

2. **모델 개선**
   - 더 많은 앙상블 모델
   - 딥러닝 모델 추가 (LSTM, Transformer)

3. **필터 최적화**
   - 거래 시간대 분석
   - 변동성 임계값 조정

================================================================================
"""
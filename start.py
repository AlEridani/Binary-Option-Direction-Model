"""
바이너리 옵션 트레이딩 시스템 - CLI 시작 스크립트
버전: 1.3.0
"""

import sys
import os
from datetime import datetime
import pandas as pd

# 시스템 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 상세한 임포트 디버깅
print("=" * 70)
print("모듈 임포트 디버깅")
print("=" * 70)

modules_to_check = [
    ('config', 'config'),
    ('main_pipe', 'MainPipeline'),
    ('monitor', 'Monitor'),
    ('log_manager', 'LogManager'),
    ('model_train', 'ModelTrainer'),
    ('real_trade', 'RealTrader'),
    ('feature_engineer', 'FeatureEngineer'),
    ('data_loader', 'DataLoader'),
    ('timeframe_manager', 'TimeframeManager'),
    ('data_merge', 'DataMerger'),
]

failed_imports = []

for module_name, class_name in modules_to_check:
    try:
        if class_name == module_name:  # config 같은 모듈
            exec(f"import {module_name}")
            print(f"✅ {module_name}")
        else:
            exec(f"from {module_name} import {class_name}")
            print(f"✅ {module_name}.{class_name}")
    except ImportError as e:
        print(f"❌ {module_name}.{class_name}: {e}")
        failed_imports.append((module_name, class_name, str(e)))
    except Exception as e:
        print(f"⚠️  {module_name}.{class_name}: {e}")
        failed_imports.append((module_name, class_name, str(e)))

print("=" * 70)

if failed_imports:
    print("\n❌ 임포트 실패한 모듈:")
    for module, cls, error in failed_imports:
        print(f"\n모듈: {module}")
        print(f"클래스: {cls}")
        print(f"오류: {error}")
    print("\n해결 방법:")
    print("1. 해당 파일이 존재하는지 확인")
    print("2. 파일 내부의 상대 임포트(from . import) 제거")
    print("3. 모든 임포트를 절대 임포트로 변경")
    sys.exit(1)

print("\n✅ 모든 모듈 임포트 성공!")
print("=" * 70 + "\n")

# 실제 임포트
try:
    import config
    from main_pipe import MainPipeline
    from monitor import Monitor
    from log_manager import LogManager
except ImportError as e:
    print(f"❌ 최종 임포트 실패: {e}")
    print("\n상세 오류:")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def quick_start():
    """빠른 시작 - 시뮬레이션 모드"""
    print("=" * 70)
    print("바이너리 옵션 예측 시스템 - 빠른 시작 (시뮬레이션)")
    print("=" * 70)
    
    # 1. 디렉토리 확인
    print("\n1️⃣  디렉토리 구조 확인")
    for dir_path in config.DIRS_TO_CREATE:
        if dir_path.exists():
            print(f"  ✅ {dir_path.name}")
        else:
            print(f"  ⚠️  {dir_path.name} 생성 중...")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # 2. 파이프라인 초기화
    print("\n2️⃣  파이프라인 초기화")
    pipeline = MainPipeline()
    
    # 3. 시뮬레이션 데이터 생성
    print("\n3️⃣  초기 학습 데이터 생성")
    days = int(input("  생성할 일수 (권장 90-270, 기본 180): ") or "180")
    initial_data = pipeline.generate_simulation_data(days=days)
    
    # 4. 초기 모델 학습
    print("\n4️⃣  초기 모델 학습 (1-3분 소요)")
    start_date = initial_data['timestamp'].min().strftime("%Y-%m-%d")
    end_date = initial_data['timestamp'].max().strftime("%Y-%m-%d")
    
    success = pipeline.retrain_pipeline(start_date, end_date)
    
    if not success:
        print("\n❌ 학습 실패")
        return None
    
    print("\n✅ 초기 학습 완료!")
    
    # 5. 백테스트
    print("\n5️⃣  백테스트 실행")
    test_data = initial_data.tail(5000)  # 최근 5000개 데이터
    
    results = pipeline.trader.backtest(
        test_data,
        start_date=None,
        end_date=None
    )
    
    # 6. 시뮬레이션 거래 선택
    print("\n" + "=" * 70)
    print("시뮬레이션 거래를 시작하시겠습니까?")
    print("  - 시뮬레이션 데이터로 실시간 거래 테스트")
    print("  - Ctrl+C로 언제든 중단 가능")
    print("=" * 70)
    
    choice = input("\n시작 (y/n): ").strip().lower()
    
    if choice == 'y':
        duration = input("실행 시간 (시간, 기본 1시간): ").strip()
        duration = int(duration) if duration else 1
        
        print(f"\n시뮬레이션 거래 시작 ({duration}시간)")
        # 실제로는 run() 메서드 사용
        # pipeline.start() 대신 간단한 루프
        print("  (실제 구현에서는 pipeline.start() 호출)")
    
    return pipeline


def live_trading():
    """실시간 거래 모드"""
    print("=" * 70)
    print("실시간 거래 모드")
    print("=" * 70)
    
    # API 키 확인
    if not config.BINANCE_API_KEY:
        print("\n⚠️  Binance API 키가 설정되지 않았습니다.")
        print("config.py에서 다음 항목을 설정해주세요:")
        print("  - BINANCE_API_KEY")
        print("  - BINANCE_API_SECRET")
        print("\n시뮬레이션 모드로 전환합니다.")
        return
    
    # 파이프라인 초기화
    pipeline = MainPipeline()
    
    # 모델 확인
    if not pipeline.trader.model_loaded:
        print("\n⚠️  학습된 모델이 없습니다.")
        print("먼저 모델을 학습해야 합니다. (메뉴 1번 또는 4번)")
        return
    
    # 초기화
    if not pipeline.initialize_system():
        print("\n❌ 시스템 초기화 실패")
        return
    
    # 실행 시간 설정
    duration = input("\n실행 시간 (시간, 무제한=Enter): ").strip()
    duration = int(duration) if duration else 99999
    
    print(f"\n🚀 실시간 거래 시작 ({duration}시간)")
    print("  Ctrl+C로 안전 종료")
    
    # 시작
    pipeline.start()


def backtest_mode():
    """백테스트 모드"""
    print("=" * 70)
    print("백테스트 모드")
    print("=" * 70)
    
    # 파이프라인 초기화
    pipeline = MainPipeline()
    
    # 모델 확인
    if not pipeline.trader.model_loaded:
        print("\n⚠️  학습된 모델이 없습니다.")
        print("먼저 모델을 학습해야 합니다. (메뉴 1번 또는 4번)")
        return
    
    # 날짜 설정
    print("\n백테스트 기간 설정:")
    start_date = input("  시작일 (YYYY-MM-DD, Enter=전체): ").strip()
    end_date = input("  종료일 (YYYY-MM-DD, Enter=전체): ").strip()
    
    # 데이터 생성
    print("\n📊 테스트 데이터 생성 중...")
    days = int(input("  생성할 일수 (기본 90): ") or "90")
    test_data = pipeline.generate_simulation_data(days=days)
    
    # 백테스트 실행
    print("\n🔄 백테스트 실행 중...")
    results = pipeline.trader.backtest(
        test_data,
        start_date=start_date if start_date else None,
        end_date=end_date if end_date else None
    )
    
    if not results.empty:
        print("\n✅ 백테스트 완료!")
        print(f"  결과 저장 위치: {config.REPORT_DIR}")


def retrain_model():
    """모델 재학습"""
    print("=" * 70)
    print("모델 재학습")
    print("=" * 70)
    
    # 파이프라인 초기화
    pipeline = MainPipeline()
    
    # 데이터 생성
    print("\n📊 학습 데이터 생성")
    days = int(input("  생성할 일수 (권장 90-270, 기본 180): ") or "180")
    new_data = pipeline.generate_simulation_data(days=days)
    
    # 날짜 설정
    start_date = new_data['timestamp'].min().strftime("%Y-%m-%d")
    end_date = new_data['timestamp'].max().strftime("%Y-%m-%d")
    
    # 재학습
    print("\n🔄 모델 재학습 중...")
    success = pipeline.retrain_pipeline(start_date, end_date)
    
    if success:
        print("\n✅ 재학습 완료!")
    else:
        print("\n❌ 재학습 실패")


def performance_report():
    """성능 리포트"""
    print("=" * 70)
    print("성능 리포트")
    print("=" * 70)
    
    # LogManager 초기화
    log_manager = LogManager()
    monitor = Monitor(log_manager)
    
    # 리포트 생성
    monitor.print_summary()
    
    # 트렌드 분석
    print("\n📈 24시간 트렌드 분석")
    trend = monitor.get_performance_trend(hours=24)
    
    if trend['win_rate_trend']:
        import numpy as np
        print(f"  평균 승률: {np.mean(trend['win_rate_trend']):.2%}")
        print(f"  평균 PnL: {np.mean(trend['pnl_trend']):.2f}")
        print(f"  평균 진입률: {np.mean(trend['entry_rate_trend']):.1f}회/시간")


def optimize_hyperparameters():
    """하이퍼파라미터 최적화"""
    print("=" * 70)
    print("하이퍼파라미터 최적화")
    print("=" * 70)
    
    # 파이프라인 초기화
    pipeline = MainPipeline()
    
    # 학습 기간 설정
    days = int(input("\n학습 데이터 일수 (기본 90): ") or "90")
    
    # 최적화 실행
    print(f"\n🔍 최적화 시작 ({days}일 데이터)")
    print("  ⚠️  시간이 오래 걸릴 수 있습니다 (30분~1시간)")
    
    best_params = pipeline.optimize_hyperparameters(lookback_days=days)
    
    if best_params:
        print("\n✅ 최적화 완료!")
        print("\n최적 파라미터:")
        for key, value in best_params.items():
            print(f"  - {key}: {value}")
        
        print("\nconfig.py에 수동으로 적용하거나,")
        print("다음 학습부터 자동으로 사용됩니다.")
    else:
        print("\n⚠️  최적화 실패 또는 개선 없음")


def main_menu():
    """메인 메뉴"""
    while True:
        print("\n" + "=" * 70)
        print("바이너리 옵션 트레이딩 시스템 v1.3.0")
        print("=" * 70)
        print("\n실행 모드를 선택하세요:")
        print()
        print("  1. 빠른 시작 (시뮬레이션)")
        print("  2. 실시간 거래 (API 연결)")
        print("  3. 백테스트")
        print("  4. 모델 재학습")
        print("  5. 성능 리포트")
        print("  6. 하이퍼파라미터 최적화")
        print()
        print("  0. 종료")
        print()
        
        choice = input("선택 (0-6): ").strip()
        
        try:
            if choice == "1":
                quick_start()
            elif choice == "2":
                live_trading()
            elif choice == "3":
                backtest_mode()
            elif choice == "4":
                retrain_model()
            elif choice == "5":
                performance_report()
            elif choice == "6":
                optimize_hyperparameters()
            elif choice == "0":
                print("\n👋 프로그램을 종료합니다.")
                sys.exit(0)
            else:
                print("\n⚠️  잘못된 선택입니다. 0-6 사이의 숫자를 입력하세요.")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  작업이 중단되었습니다.")
            print("메인 메뉴로 돌아갑니다...")
        
        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            print("\n메인 메뉴로 돌아갑니다...")


# ============================================================
# README / 사용 가이드
# ============================================================
README = """
================================================================================
                    바이너리 옵션 트레이딩 시스템 v1.3.0
================================================================================

## 시스템 개요

- 바이낸스 바이너리 옵션 자동 거래 시스템
- 멀티타임프레임 레짐 기반 예측 (4h/1h/15m)
- LightGBM 앙상블 + 캘리브레이션
- 동적 필터 + 자동 재학습
- 목표 승률: 55% 이상

## 빠른 시작

```bash
# 1. 패키지 설치
pip install pandas numpy lightgbm scikit-learn requests schedule psutil

# 2. 시스템 시작
python start.py

# 3. 메뉴에서 "1" 선택 (빠른 시작)
```

## 주요 기능

### 1. 빠른 시작 (시뮬레이션)
- 시뮬레이션 데이터로 모델 학습
- 백테스트로 성능 확인
- API 키 없이 시스템 테스트

### 2. 실시간 거래
- 실제 Binance API 연결
- 레짐 기반 자동 진입/청산
- 실시간 모니터링

### 3. 백테스트
- 과거 데이터로 전략 검증
- 레짐별 성과 분석
- 시간대별 승률 확인

### 4. 모델 재학습
- 새 데이터로 모델 업데이트
- 레짐별 모델 분리 학습
- 성능 평가 및 저장

### 5. 성능 리포트
- 승률/손익 통계
- 레짐별 성과
- 트렌드 분석

### 6. 하이퍼파라미터 최적화
- Grid Search 기반
- 최적 파라미터 자동 탐색
- 성능 개선

## 설정 파일 (config.py)

주요 설정값:

```python
# 타임프레임
BAR_MINUTES = 30                    # 30분 예측
REGIME_TIMEFRAMES = ['4h','1h','15m']

# 페이아웃
PAYOUT_30M_PLUS = 0.85              # 30분 승리 시 85%

# 히스테리시스
CUT_ON_DEFAULT = 0.60               # 진입 임계값
CUT_OFF_DEFAULT = 0.55              # 청산 임계값

# 재학습
RETRAIN_CHECK_INTERVAL = 50         # 50거래마다 체크
RETRAIN_WIN_RATE_THRESHOLD = 0.55   # 55% 미만 시 재학습
```

## 폴더 구조

```
project/
├── models/              # 학습된 모델
│   ├── lgbm_regime_UP.pkl
│   ├── lgbm_regime_DOWN.pkl
│   ├── lgbm_regime_FLAT.pkl
│   └── model_metadata.json
├── data/               # 가격 데이터
├── cache/              # 피처 캐시
├── logs/               # 로그 파일
│   ├── trade_log/     # 거래 기록
│   ├── feature_log/   # 피처 로그
│   ├── monitor/       # 모니터링
│   └── system/        # 시스템 로그
├── reports/            # 리포트
├── config.py           # 설정
├── main_pipe.py        # 메인 파이프라인
├── model_train.py      # 모델 학습
├── real_trade.py       # 실시간 거래
├── feature_engineer.py # 피처 엔지니어링
├── data_loader.py      # 데이터 로더
├── log_manager.py      # 로그 관리
├── monitor.py          # 모니터링
└── start.py            # CLI 시작
```

## API 키 설정

1. config.py 열기
2. 다음 항목 설정:

```python
BINANCE_API_KEY = "your_api_key_here"
BINANCE_API_SECRET = "your_api_secret_here"
```

3. 또는 환경 변수 사용:

```bash
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
```

## 주의사항

### 리스크 관리
- 시뮬레이션으로 충분히 테스트 후 실거래
- 소액으로 시작 권장
- 최대 동시 포지션 제한 (기본 5개)
- 손실 한도 설정

### API 사용
- Binance API 레이트 리밋 준수
- 과도한 요청 방지
- READ 권한만 사용 (안전)

### 데이터 관리
- 정기적인 로그 백업
- 오래된 캐시 자동 정리
- 거래 기록 보관

## 트러블슈팅

### 모델이 로드되지 않음
```bash
# 초기 학습 실행
python start.py
# 메뉴에서 1번 또는 4번 선택
```

### 승률이 낮음
```bash
# 하이퍼파라미터 최적화
python start.py
# 메뉴에서 6번 선택
```

### 메모리 부족
- config.py에서 앙상블 수 감소 (N_ENSEMBLE)
- 룩백 기간 단축
- 피처 수 제한

### API 연결 실패
- API 키 확인
- 인터넷 연결 확인
- 시뮬레이션 모드로 자동 전환됨

## 성능 향상 팁

### 1. 데이터 품질
- 충분한 학습 데이터 (최소 90일)
- 다양한 시장 상황 포함
- 정기적인 재학습

### 2. 피처 엔지니어링
- 레짐 타임프레임 조정
- ADX 임계값 최적화
- 추가 기술지표 구현

### 3. 필터 최적화
- 동적 필터 활용
- 패배 패턴 분석
- 시간대별 성과 확인

### 4. 모델 개선
- 하이퍼파라미터 최적화
- 앙상블 수 증가
- 캘리브레이션 조정

## FAQ

**Q: 실제 거래는 어떻게 하나요?**
A: config.py에 API 키 설정 후 메뉴 2번 선택

**Q: 시뮬레이션과 실거래의 차이는?**
A: 시뮬레이션은 랜덤 데이터, 실거래는 실제 시세 사용

**Q: 승률 목표는?**
A: 55% 이상 (페이아웃 85% 기준)

**Q: 재학습은 언제 하나요?**
A: 50거래마다 자동 체크, 승률 55% 미만 시 자동 재학습

**Q: 동시 포지션 제한은?**
A: 기본 5개, config.py에서 조정 가능

**Q: TTL은 무엇인가요?**
A: Time To Live - 포지션 만기 시간 (기본 30분)

**Q: 레짐이란?**
A: 시장 추세 방향 (UP/DOWN/FLAT)
   - 4h, 1h, 15m 타임프레임 가중 합산

**Q: 동적 필터란?**
A: 패배 패턴 학습하여 자동으로 진입 차단

## 고급 사용법

### 커스텀 전략
```python
from main_pipe import MainPipeline

pipeline = MainPipeline()

# 커스텀 컷오프 적용
pipeline.trader.cut_on = 0.65
pipeline.trader.cut_off = 0.60

# 실행
pipeline.start()
```

### 백테스트 분석
```python
from real_trade import RealTrader

trader = RealTrader()
results = trader.backtest(historical_data)

# 레짐별 분석
regime_stats = results.groupby('regime').agg({
    'correct': ['count', 'mean'],
    'pnl': 'sum'
})
```

### 모니터링 커스터마이징
```python
from monitor import Monitor
from log_manager import LogManager

log_mgr = LogManager()
monitor = Monitor(log_mgr)

# 승률 추적
win_rate_data = monitor.perf_monitor.track_win_rate(window=100)

# 레짐별 승률
regime_data = monitor.perf_monitor.track_win_rate_by_regime(window=200)
```

## 라이선스

MIT License

## 기여

Pull Request 환영합니다!

## 지원

이슈는 GitHub Issues에 등록해주세요.

================================================================================
"""


def print_readme():
    """README 출력"""
    print(README)


if __name__ == "__main__":
    try:
        # 환영 메시지
        print("\n" + "🎯 " * 25)
        print("바이너리 옵션 트레이딩 시스템 v1.3.0")
        print("🎯 " * 25)
        
        # 버전 정보
        print(f"\n시스템 버전: {config.SYSTEM_VERSION}")
        print(f"설정 파일: config.py")
        print(f"데이터 경로: {config.DATA_DIR}")
        
        # 메인 메뉴 실행
        main_menu()
        
    except KeyboardInterrupt:
        print("\n\n👋 프로그램을 종료합니다.")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ 치명적 오류: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
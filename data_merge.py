"""
데이터 병합 - 가격/로그/거래 통합 + 품질 검증
버전: 1.3.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import config
from data_loader import DataLoader
from log_manager import LogManager
from timeframe_manager import TimeframeManager


class DataMerger:
    """
    데이터 병합 및 품질 관리
    - 가격 + 로그 + 거래 병합
    - 타임프레임 정렬
    - 데이터 품질 검증
    - 재학습용 클린 데이터셋 생성
    """
    
    def __init__(self):
        """인자 없이 초기화 (Config 참조)"""
        self.data_loader = DataLoader()
        self.log_manager = LogManager()
        self.tf_manager = TimeframeManager()
    
    def merge_all_data(self, start_date: str, end_date: str, filename: str = None) -> pd.DataFrame:
        """
        전체 데이터 병합
        
        Args:
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
            filename: 파일명 (None이면 자동 탐색)
        
        Returns:
            병합된 데이터프레임
        """
        print(f"\n📊 데이터 병합 시작: {start_date} ~ {end_date}")
        
        # 1. 가격 데이터 로드
        if filename:
            df_price = self.data_loader.load_price_data(start_date, end_date, filename)
        else:
            # 파일 자동 탐색
            possible_files = ['btcusdt_1m.csv', 'test_merger.csv', 'test_data_loader.csv']
            df_price = pd.DataFrame()
            
            for fname in possible_files:
                try:
                    df_price = self.data_loader.load_price_data(start_date, end_date, fname)
                    if not df_price.empty:
                        print(f"  ✅ 파일 로드: {fname}")
                        break
                except:
                    continue
        
        if df_price.empty:
            print("⚠️  가격 데이터 없음")
            return pd.DataFrame()
        
        # 2. 1분봉 → 30분봉 집계
        df_30m = self.tf_manager.aggregate_1m_to_30m(df_price)
        
        if df_30m.empty:
            print("⚠️  30분봉 집계 실패")
            return pd.DataFrame()
        
        print(f"  ✅ 30분봉 집계: {len(df_30m)}개")
        
        # 3. 거래 로그 로드
        df_trades = self._load_trade_logs_range(start_date, end_date)
        
        if not df_trades.empty:
            print(f"  ✅ 거래 로그 로드: {len(df_trades)}개")
            
            # 4. 병합
            df_merged = self._merge_price_and_trades(df_30m, df_trades)
        else:
            print("  ⚠️  거래 로그 없음, 가격 데이터만 사용")
            df_merged = df_30m.copy()
        
        print(f"\n✅ 병합 완료: {len(df_merged)}개 레코드")
        
        return df_merged
    
    def _load_trade_logs_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """날짜 범위의 거래 로그 로드"""
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        dfs = []
        current_dt = start_dt
        
        while current_dt <= end_dt:
            date_str = current_dt.strftime("%Y%m%d")
            df_day = self.log_manager.load_trade_log(date_str)
            
            if not df_day.empty:
                dfs.append(df_day)
            
            current_dt += pd.Timedelta(days=1)
        
        if not dfs:
            return pd.DataFrame()
        
        df_combined = pd.concat(dfs, ignore_index=True)
        
        return df_combined
    
    def _merge_price_and_trades(self, df_price: pd.DataFrame, 
                                df_trades: pd.DataFrame) -> pd.DataFrame:
        """가격과 거래 로그 병합"""
        # bar30_start 기준으로 병합
        if 'bar30_start' not in df_price.columns:
            print("⚠️  bar30_start 컬럼 없음")
            return df_price
        
        if 'bar30_start' not in df_trades.columns:
            print("⚠️  거래 로그에 bar30_start 없음")
            return df_price
        
        # 타임스탬프 정규화
        df_price = df_price.copy()
        df_trades = df_trades.copy()
        
        df_price['bar30_start'] = pd.to_datetime(df_price['bar30_start'])
        df_trades['bar30_start'] = pd.to_datetime(df_trades['bar30_start'])
        
        # 거래 수 집계
        trade_counts = df_trades.groupby('bar30_start').size().reset_index(name='n_trades')
        
        # 승률 집계
        df_closed = df_trades[df_trades['status'] == 'CLOSED']
        if not df_closed.empty:
            win_rates = df_closed.groupby('bar30_start').apply(
                lambda x: (x['result'] == 'WIN').mean()
            ).reset_index(name='win_rate')
        else:
            win_rates = pd.DataFrame(columns=['bar30_start', 'win_rate'])
        
        # 병합
        df_merged = df_price.merge(trade_counts, on='bar30_start', how='left')
        df_merged = df_merged.merge(win_rates, on='bar30_start', how='left')
        
        # 결측치 처리
        df_merged['n_trades'] = df_merged['n_trades'].fillna(0).astype(int)
        df_merged['win_rate'] = df_merged['win_rate'].fillna(0.0)
        
        return df_merged
    
    def validate_merged_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        병합 데이터 검증
        
        Returns:
            (검증 통과 여부, 오류 목록)
        """
        errors = []
        
        if df.empty:
            errors.append("데이터가 비어있음")
            return False, errors
        
        # 1. 필수 컬럼 체크
        required_cols = ['bar30_start', 'bar30_end', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            errors.append(f"필수 컬럼 누락: {missing_cols}")
        
        # 2. 타임스탬프 정렬 체크
        if 'bar30_start' in df.columns:
            df_check = df.copy()
            df_check['bar30_start'] = pd.to_datetime(df_check['bar30_start'])
            
            if not df_check['bar30_start'].is_monotonic_increasing:
                errors.append("타임스탬프가 정렬되지 않음")
        
        # 3. OHLC 정합성 체크
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_hl = (df['high'] < df['low']).sum()
            if invalid_hl > 0:
                errors.append(f"High < Low: {invalid_hl}건")
            
            invalid_ohlc = (
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            ).sum()
            
            if invalid_ohlc > 0:
                errors.append(f"OHLC 범위 위반: {invalid_ohlc}건")
        
        # 4. 중복 체크
        if 'bar30_start' in df.columns:
            duplicates = df['bar30_start'].duplicated().sum()
            if duplicates > 0:
                errors.append(f"중복 타임스탬프: {duplicates}건")
        
        # 5. NaN 체크
        critical_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in critical_cols:
            if col in df.columns:
                nan_count = df[col].isnull().sum()
                if nan_count > 0:
                    errors.append(f"{col} 결측치: {nan_count}건")
        
        # 6. 이상치 체크 (가격이 0 이하)
        if 'close' in df.columns:
            invalid_price = (df['close'] <= 0).sum()
            if invalid_price > 0:
                errors.append(f"유효하지 않은 가격: {invalid_price}건")
        
        return len(errors) == 0, errors
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터 정제
        
        Args:
            df: 원본 데이터
        
        Returns:
            정제된 데이터
        """
        if df.empty:
            return df
        
        df_clean = df.copy()
        
        # 1. 중복 제거
        if 'bar30_start' in df_clean.columns:
            df_clean['bar30_start'] = pd.to_datetime(df_clean['bar30_start'])
            df_clean = df_clean.drop_duplicates(subset=['bar30_start'], keep='first')
        
        # 2. 정렬
        if 'bar30_start' in df_clean.columns:
            df_clean = df_clean.sort_values('bar30_start').reset_index(drop=True)
        
        # 3. OHLC 정합성 수정
        if all(col in df_clean.columns for col in ['open', 'high', 'low', 'close']):
            # High는 max(open, close, high)
            df_clean['high'] = df_clean[['open', 'high', 'close']].max(axis=1)
            
            # Low는 min(open, close, low)
            df_clean['low'] = df_clean[['open', 'low', 'close']].min(axis=1)
        
        # 4. 결측치 제거 (중요 컬럼)
        critical_cols = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in critical_cols if col in df_clean.columns]
        
        if available_cols:
            df_clean = df_clean.dropna(subset=available_cols)
        
        # 5. 이상치 제거 (가격 <= 0)
        if 'close' in df_clean.columns:
            df_clean = df_clean[df_clean['close'] > 0]
        
        df_clean = df_clean.reset_index(drop=True)
        
        return df_clean
    
    def align_timeframes(self, df_price: pd.DataFrame, 
                        df_trades: pd.DataFrame) -> pd.DataFrame:
        """타임프레임 정렬 (30분봉 기준)"""
        if df_price.empty:
            return pd.DataFrame()
        
        if df_trades.empty:
            return df_price
        
        # 양쪽 모두 정렬
        df_price, df_trades = self.tf_manager.align_dataframes(df_price, df_trades)
        
        # 병합
        df_aligned = self._merge_price_and_trades(df_price, df_trades)
        
        return df_aligned
    
    def create_retrain_dataset(self, start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        재학습용 데이터셋 생성
        
        Args:
            start_date: 시작일
            end_date: 종료일
        
        Returns:
            (X, y) 튜플
        """
        print(f"\n🔄 재학습 데이터셋 생성: {start_date} ~ {end_date}")
        
        # 데이터 병합
        df_merged = self.merge_all_data(start_date, end_date)
        
        if df_merged.empty:
            print("❌ 데이터 병합 실패")
            return pd.DataFrame(), pd.Series()
        
        # 데이터 정제
        print("🧹 데이터 정제 중...")
        df_clean = self.clean_data(df_merged)
        
        # 검증
        valid, errors = self.validate_merged_data(df_clean)
        
        if not valid:
            print("⚠️  데이터 검증 실패:")
            for error in errors:
                print(f"  - {error}")
        
        # 가격 데이터를 1분봉으로 변환 (피처 생성용)
        # 실제로는 원본 1분봉 로드 필요
        df_1m = self.data_loader.load_price_data(start_date, end_date)
        
        if df_1m.empty:
            print("❌ 1분봉 데이터 없음")
            return pd.DataFrame(), pd.Series()
        
        # 피처 생성 및 학습 데이터 준비
        X, y = self.data_loader.prepare_training_data(df_1m, use_cache=False)
        
        print(f"✅ 재학습 데이터셋 생성 완료: X={X.shape}, y={y.shape}")
        
        return X, y
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """데이터 품질 리포트"""
        if df.empty:
            return {
                'total_records': 0,
                'missing_ratio': 0.0,
                'duplicate_ratio': 0.0,
                'outlier_ratio': 0.0,
                'quality_score': 0.0
            }
        
        total_records = len(df)
        
        # 결측치 비율
        missing_count = df.isnull().sum().sum()
        total_values = df.shape[0] * df.shape[1]
        missing_ratio = missing_count / total_values if total_values > 0 else 0.0
        
        # 중복 비율
        if 'bar30_start' in df.columns:
            duplicate_count = df['bar30_start'].duplicated().sum()
            duplicate_ratio = duplicate_count / total_records
        else:
            duplicate_ratio = 0.0
        
        # 이상치 비율 (가격 <= 0)
        if 'close' in df.columns:
            outlier_count = (df['close'] <= 0).sum()
            outlier_ratio = outlier_count / total_records
        else:
            outlier_ratio = 0.0
        
        # 품질 점수 (0~100)
        quality_score = 100 * (1 - missing_ratio - duplicate_ratio - outlier_ratio)
        quality_score = max(0, min(100, quality_score))
        
        return {
            'total_records': total_records,
            'missing_ratio': missing_ratio,
            'duplicate_ratio': duplicate_ratio,
            'outlier_ratio': outlier_ratio,
            'quality_score': quality_score
        }


# ============================================================
# 테스트 및 검증
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DataMerger 테스트")
    print("=" * 60)
    
    # DataMerger 초기화
    merger = DataMerger()
    
    print("\n✅ DataMerger 초기화 성공")
    
    # 테스트 데이터 생성
    from datetime import timedelta
    
    print("\n📝 테스트 데이터 생성")
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    n_minutes = 1440  # 1일
    
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_minutes)]
    
    np.random.seed(42)
    price = 50000
    prices = []
    
    for _ in range(n_minutes):
        change = np.random.normal(0, 50)
        price = max(price + change, 45000)
        prices.append(price)
    
    df_1m_test = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices
    })
    
    df_1m_test['open'] = df_1m_test['close'].shift(1).fillna(df_1m_test['close'])
    df_1m_test['high'] = df_1m_test[['open', 'close']].max(axis=1) * 1.001
    df_1m_test['low'] = df_1m_test[['open', 'close']].min(axis=1) * 0.999
    df_1m_test['volume'] = np.random.uniform(1000, 5000, n_minutes)
    
    # 데이터 저장
    merger.data_loader.save_price_data(df_1m_test, "test_merge_data.csv")
    
    # 30분봉 집계 테스트
    print("\n📊 30분봉 집계 테스트")
    df_30m_test = merger.tf_manager.aggregate_1m_to_30m(df_1m_test)
    
    print(f"  1분봉: {len(df_1m_test)}개")
    print(f"  30분봉: {len(df_30m_test)}개")
    
    # 거래 로그 생성 (더미)
    print("\n📝 거래 로그 생성")
    for i in range(10):
        trade_time = start_time + timedelta(minutes=30*i)
        
        merger.log_manager.log_trade_entry_simple(
            trade_id=f"merge_test_{i:03d}",
            direction='UP' if i % 2 == 0 else 'DOWN',
            entry_price=50000 + np.random.uniform(-200, 200),
            entry_ts=trade_time,
            p_at_entry=np.random.uniform(0.6, 0.8),
            regime=np.random.choice([1, 0, -1]),
            bar30_start=trade_time,
            bar30_end=trade_time + timedelta(minutes=30)
        )
    
    print(f"  생성된 거래: 10개")
    
    # 데이터 병합 테스트
    print("\n🔗 데이터 병합 테스트")
    df_merged = merger.merge_all_data("2025-01-01", "2025-01-01")
    
    if not df_merged.empty:
        print(f"  ✅ 병합 성공: {len(df_merged)}개 레코드")
        print(f"\n  컬럼: {list(df_merged.columns)}")
        
        # 샘플 데이터
        sample_cols = ['bar30_start', 'open', 'close', 'volume', 'n_trades', 'win_rate']
        available_cols = [col for col in sample_cols if col in df_merged.columns]
        
        if available_cols:
            print(f"\n  샘플 데이터:")
            print(df_merged[available_cols].head(3))
    else:
        print("  ❌ 병합 실패")
    
    # 데이터 검증 테스트
    print("\n🔍 데이터 검증 테스트")
    valid, errors = merger.validate_merged_data(df_merged)
    
    if valid:
        print("  ✅ 검증 통과")
    else:
        print("  ⚠️  검증 실패:")
        for error in errors:
            print(f"    - {error}")
    
    # 데이터 정제 테스트
    print("\n🧹 데이터 정제 테스트")
    
    # 의도적으로 오류 추가
    df_dirty = df_merged.copy()
    if len(df_dirty) > 5:
        df_dirty.loc[2, 'close'] = -100  # 음수 가격
        df_dirty.loc[3, 'high'] = df_dirty.loc[3, 'low'] - 10  # High < Low
    
    print(f"  정제 전: {len(df_dirty)}개")
    
    df_clean = merger.clean_data(df_dirty)
    
    print(f"  정제 후: {len(df_clean)}개")
    
    valid_clean, errors_clean = merger.validate_merged_data(df_clean)
    
    if valid_clean:
        print("  ✅ 정제 후 검증 통과")
    else:
        print("  ⚠️  정제 후에도 오류 존재:")
        for error in errors_clean:
            print(f"    - {error}")
    
    # 품질 리포트 테스트
    print("\n📊 데이터 품질 리포트")
    quality_report = merger.get_data_quality_report(df_clean)
    
    print(f"  총 레코드: {quality_report['total_records']}")
    print(f"  결측치 비율: {quality_report['missing_ratio']:.2%}")
    print(f"  중복 비율: {quality_report['duplicate_ratio']:.2%}")
    print(f"  이상치 비율: {quality_report['outlier_ratio']:.2%}")
    print(f"  품질 점수: {quality_report['quality_score']:.1f}/100")
    
    # 재학습 데이터셋 생성 테스트
    print("\n🔄 재학습 데이터셋 생성 테스트")
    
    # 더 많은 테스트 데이터 생성 (피처 엔지니어링에 충분한 양)
    n_minutes_large = 5000
    timestamps_large = [start_time + timedelta(minutes=i) for i in range(n_minutes_large)]
    
    prices_large = []
    price = 50000
    for _ in range(n_minutes_large):
        change = np.random.normal(0, 30)
        price = max(price + change, 45000)
        prices_large.append(price)
    
    df_1m_large = pd.DataFrame({
        'timestamp': timestamps_large,
        'close': prices_large
    })
    
    df_1m_large['open'] = df_1m_large['close'].shift(1).fillna(df_1m_large['close'])
    df_1m_large['high'] = df_1m_large[['open', 'close']].max(axis=1) * 1.001
    df_1m_large['low'] = df_1m_large[['open', 'close']].min(axis=1) * 0.999
    df_1m_large['volume'] = np.random.uniform(1000, 5000, n_minutes_large)
    
    merger.data_loader.save_price_data(df_1m_large, "test_merge_data.csv")
    
    X, y = merger.create_retrain_dataset("2025-01-01", "2025-01-04")
    
    if not X.empty:
        print(f"  ✅ 데이터셋 생성 성공")
        print(f"    X shape: {X.shape}")
        print(f"    y shape: {y.shape}")
        print(f"    타겟 분포: UP={y.mean():.2%}")
        
        # 데이터 품질 체크
        valid_X, errors_X = merger.data_loader.validate_data_quality(X, y)
        
        if valid_X:
            print(f"  ✅ 데이터 품질 검증 통과")
        else:
            print(f"  ⚠️  데이터 품질 이슈:")
            for error in errors_X:
                print(f"    - {error}")
    else:
        print("  ❌ 데이터셋 생성 실패")
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)
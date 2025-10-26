"""
데이터 로더 - 입출력 및 캐싱 관리
버전: 1.3.1 (30분 옵션 최적화 + 미래 데이터 누수 방지)
"""

import pandas as pd
import numpy as np
import hashlib
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Optional
from sklearn.model_selection import TimeSeriesSplit

import config
from feature_engineer import FeatureEngineer


class DataLoader:
    """
    데이터 입출력 및 캐싱 관리
    - 1분봉 CSV 저장/로드 (UTC 강제)
    - SHA1 기반 피처 캐싱
    - 시계열 분할 (누수 방지)
    - 학습셋 준비
    """
    
    def __init__(self):
        """인자 없이 초기화 (Config 참조)"""
        self.data_dir = config.DATA_DIR
        self.cache_dir = config.CACHE_DIR
        self.feature_engineer = FeatureEngineer()
        
    def save_price_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        1분봉 데이터 저장 (UTC 강제)
        
        Args:
            df: 1분봉 데이터프레임
            filename: 파일명 (확장자 포함)
        """
        filepath = self.data_dir / filename
        
        # dtype 정규화 및 UTC 변환
        df_save = df.copy()
        
        for col, dtype in config.PRICE_DATA_DTYPES.items():
            if col in df_save.columns:
                if dtype == 'datetime64[ns]':
                    df_save[col] = pd.to_datetime(df_save[col], utc=True)
                    if df_save[col].dt.tz is None:
                        df_save[col] = df_save[col].dt.tz_localize(config.TZ)
                else:
                    df_save[col] = df_save[col].astype(dtype)
        
        df_save.to_csv(filepath, index=False)
        print(f"✅ 데이터 저장 완료: {filepath}")
    
    def load_price_data(self, start_date: str, end_date: str, 
                       filename: Optional[str] = None) -> pd.DataFrame:
        """
        1분봉 데이터 로드 (UTC 강제)
        
        Args:
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
            filename: 파일명 (None이면 기본 파일 사용)
        
        Returns:
            1분봉 데이터프레임 (UTC timezone-aware)
        """
        if filename is None:
            filename = "btcusdt_1m.csv"
        
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            print(f"⚠️  파일 없음: {filepath}")
            return pd.DataFrame()
        
        # 데이터 로드
        df = pd.read_csv(filepath)
        
        # dtype 정규화 및 UTC 변환
        for col, dtype in config.PRICE_DATA_DTYPES.items():
            if col in df.columns:
                if dtype == 'datetime64[ns]':
                    # UTC 변환 강제
                    df[col] = pd.to_datetime(df[col], utc=True)
                    if df[col].dt.tz is None:
                        df[col] = df[col].dt.tz_localize(config.TZ)
                    elif df[col].dt.tz != config.TZ:
                        df[col] = df[col].dt.tz_convert(config.TZ)
                else:
                    df[col] = df[col].astype(dtype)
        
        # 날짜 필터링 (timezone-aware 비교)
        if 'timestamp' in df.columns:
            start_dt = pd.to_datetime(start_date, utc=True).tz_localize(config.TZ)
            end_dt = pd.to_datetime(end_date, utc=True).tz_localize(config.TZ)
            
            # 하루 종료 시점까지 포함
            end_dt = end_dt + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            
            df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
        
        print(f"✅ 데이터 로드 완료: {len(df)}개 레코드 ({start_date} ~ {end_date})")
        
        return df.reset_index(drop=True)
    
    def get_cache_key(self, df: pd.DataFrame, params: dict) -> str:
        """
        캐시 키 생성 (SHA1 해시)
        
        Args:
            df: 원본 데이터프레임
            params: 파라미터 딕셔너리
        
        Returns:
            SHA1 해시 문자열
        """
        # 데이터 해시
        if 'timestamp' in df.columns:
            data_str = f"{df['timestamp'].min()}_{df['timestamp'].max()}_{len(df)}"
        else:
            data_str = f"{len(df)}_{df['close'].sum():.2f}"
        
        # 파라미터 해시
        params_str = json.dumps(params, sort_keys=True)
        
        # 결합 해시
        combined = f"{data_str}_{params_str}"
        hash_obj = hashlib.sha1(combined.encode())
        
        return hash_obj.hexdigest()
    
    def load_cached_features(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        캐시된 피처 로드
        
        Args:
            cache_key: 캐시 키
        
        Returns:
            캐시된 피처 데이터프레임 (없으면 None)
        """
        cache_file = self.cache_dir / f"features_{cache_key}.parquet"
        
        if not cache_file.exists():
            return None
        
        try:
            df = pd.read_parquet(cache_file)
            print(f"✅ 캐시 로드: {cache_key[:8]}... ({len(df)}개)")
            return df
        except Exception as e:
            print(f"⚠️  캐시 로드 실패: {e}")
            return None
    
    def save_cached_features(self, cache_key: str, df: pd.DataFrame) -> None:
        """
        피처 캐싱
        
        Args:
            cache_key: 캐시 키
            df: 피처 데이터프레임
        """
        cache_file = self.cache_dir / f"features_{cache_key}.parquet"
        
        try:
            df.to_parquet(cache_file, index=False)
            print(f"✅ 캐시 저장: {cache_key[:8]}... ({len(df)}개)")
        except Exception as e:
            print(f"⚠️  캐시 저장 실패: {e}")
    
    def prepare_training_data(self, df_1m: pd.DataFrame, 
                            use_cache: bool = True,
                            lookback_30m_bars: int = 100) -> Tuple[pd.DataFrame, pd.Series]:
        """
        학습 데이터 준비 (X, y)
        
        Args:
            df_1m: 1분봉 데이터
            use_cache: 캐시 사용 여부
            lookback_30m_bars: 룩백 기간 (30분봉 개수)
        
        Returns:
            (X, y) 튜플
        """
        # 최소 데이터 요구량 계산 (30분봉 기준)
        min_1m_required = lookback_30m_bars * config.BAR_MINUTES
        
        if df_1m.empty or len(df_1m) < min_1m_required:
            print(f"⚠️  데이터 부족 (최소 {min_1m_required}개 1분봉 필요, 현재 {len(df_1m)}개)")
            return pd.DataFrame(), pd.Series()
        
        # 캐시 키 생성
        cache_params = {
            'lookback_30m_bars': lookback_30m_bars,
            'bar_minutes': config.BAR_MINUTES,
            'feature_ver': config.FEATURE_VERSION,
            'data_ver': config.DATA_VERSION
        }
        cache_key = self.get_cache_key(df_1m, cache_params)
        
        # 캐시 확인
        if use_cache:
            df_features = self.load_cached_features(cache_key)
            if df_features is not None:
                return self._split_features_target(df_features)
        
        # 피처 생성
        print(f"🔄 피처 생성 중... (lookback={lookback_30m_bars}개 30분봉)")
        df_features = self.feature_engineer.create_feature_pool(df_1m, lookback_30m_bars)
        
        if df_features.empty:
            print("❌ 피처 생성 실패")
            return pd.DataFrame(), pd.Series()
        
        # 캐시 저장
        if use_cache:
            self.save_cached_features(cache_key, df_features)
        
        return self._split_features_target(df_features)
    
    def _split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """피처와 타겟 분리"""
        if 'target' not in df.columns:
            print("⚠️  타겟 컬럼 없음")
            return df, pd.Series()
        
        # 학습에 사용할 피처만 추출
        feature_cols = self.feature_engineer.get_feature_names(df)
        
        X = df[feature_cols].copy()
        y = df['target'].copy()
        
        return X, y
    
    def time_series_split(self, X: pd.DataFrame, y: pd.Series, 
                         n_splits: int = 5,
                         test_size: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        시계열 교차 검증 분할 (sklearn TimeSeriesSplit 사용)
        
        Args:
            X: 피처 데이터프레임
            y: 타겟 시리즈
            n_splits: 분할 수
            test_size: 테스트셋 크기 (None이면 자동 계산)
        
        Returns:
            [(train_idx, test_idx), ...] 리스트
        """
        if len(X) < config.MIN_TRAIN_SIZE * 2:
            print(f"⚠️  데이터 부족 (최소 {config.MIN_TRAIN_SIZE * 2}개 필요)")
            return []
        
        # sklearn TimeSeriesSplit 사용 (겹치지 않는 fold)
        if test_size is None:
            test_size = len(X) // (n_splits + 1)
        
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        
        splits = []
        for train_idx, test_idx in tscv.split(X):
            # 최소 학습 샘플 보장
            if len(train_idx) >= config.MIN_TRAIN_SIZE:
                splits.append((train_idx, test_idx))
        
        print(f"✅ 시계열 분할: {len(splits)}개 생성 (겹치지 않음)")
        for i, (train_idx, test_idx) in enumerate(splits, 1):
            train_start = train_idx[0] if len(train_idx) > 0 else 0
            train_end = train_idx[-1] if len(train_idx) > 0 else 0
            test_start = test_idx[0] if len(test_idx) > 0 else 0
            test_end = test_idx[-1] if len(test_idx) > 0 else 0
            
            print(f"  Split {i}:")
            print(f"    Train: [{train_start:5d}:{train_end:5d}] ({len(train_idx):5d}개)")
            print(f"    Test:  [{test_start:5d}:{test_end:5d}] ({len(test_idx):5d}개)")
        
        return splits
    
    def split_train_valid_test(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """
        학습/검증/테스트 분할 (시계열 순서 유지)
        
        Args:
            X: 피처
            y: 타겟
        
        Returns:
            (X_train, X_valid, X_test, y_train, y_valid, y_test)
        """
        n = len(X)
        
        train_end = int(n * config.TRAIN_RATIO)
        valid_end = int(n * (config.TRAIN_RATIO + config.VALID_RATIO))
        
        X_train = X.iloc[:train_end].copy()
        X_valid = X.iloc[train_end:valid_end].copy()
        X_test = X.iloc[valid_end:].copy()
        
        y_train = y.iloc[:train_end].copy()
        y_valid = y.iloc[train_end:valid_end].copy()
        y_test = y.iloc[valid_end:].copy()
        
        print(f"✅ 데이터 분할 (시계열 순서 유지):")
        print(f"  Train: {len(X_train):5d} ({len(X_train)/n*100:.1f}%) [0:{train_end}]")
        print(f"  Valid: {len(X_valid):5d} ({len(X_valid)/n*100:.1f}%) [{train_end}:{valid_end}]")
        print(f"  Test:  {len(X_test):5d} ({len(X_test)/n*100:.1f}%) [{valid_end}:{n}]")
        
        return X_train, X_valid, X_test, y_train, y_valid, y_test
    
    def validate_data_quality(self, X: pd.DataFrame, y: pd.Series) -> Tuple[bool, List[str]]:
        """
        데이터 품질 검증
        
        Returns:
            (검증 통과 여부, 오류 목록)
        """
        errors = []
        
        # 1. 샘플 수 일치
        if len(X) != len(y):
            errors.append(f"샘플 수 불일치: X={len(X)}, y={len(y)}")
        
        # 2. 결측치 체크
        X_nan = X.isnull().sum().sum()
        y_nan = y.isnull().sum()
        
        if X_nan > 0:
            X_nan_ratio = X_nan / (len(X) * len(X.columns))
            errors.append(f"X에 결측치 {X_nan}개 존재 ({X_nan_ratio:.2%})")
        
        if y_nan > 0:
            errors.append(f"y에 결측치 {y_nan}개 존재")
        
        # 3. 타겟 분포 체크 (30분 바이너리 옵션: 40~60% 권장)
        if len(y) > 0:
            target_ratio = y.mean()
            if target_ratio < 0.40 or target_ratio > 0.60:
                errors.append(f"타겟 불균형 (40~60% 권장): UP={target_ratio:.2%}")
        
        # 4. 피처 타입 체크
        object_cols = [col for col in X.columns if X[col].dtype == 'object']
        if object_cols:
            errors.append(f"피처가 object 타입: {object_cols[:5]}...")
        
        # 5. 무한대 체크
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X_inf = np.isinf(X[numeric_cols]).sum().sum()
            if X_inf > 0:
                errors.append(f"X에 무한대 {X_inf}개 존재")
        
        # 6. 상수 피처 체크
        constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_cols:
            errors.append(f"상수 피처 존재: {constant_cols[:5]}...")
        
        # 7. 타겟 값 체크 (0 또는 1만 허용)
        if len(y) > 0:
            unique_targets = y.unique()
            if not set(unique_targets).issubset({0, 1}):
                errors.append(f"타겟 값 이상: {unique_targets}")
        
        return len(errors) == 0, errors
    
    def clean_cache(self, older_than_days: int = 7) -> None:
        """
        오래된 캐시 파일 삭제
        
        Args:
            older_than_days: 삭제 기준 (일)
        """
        if not self.cache_dir.exists():
            return
        
        cutoff_time = datetime.now().timestamp() - (older_than_days * 86400)
        deleted_count = 0
        
        for cache_file in self.cache_dir.glob("features_*.parquet"):
            if cache_file.stat().st_mtime < cutoff_time:
                cache_file.unlink()
                deleted_count += 1
        
        if deleted_count > 0:
            print(f"🗑️  캐시 정리: {deleted_count}개 파일 삭제 ({older_than_days}일 이상)")
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        데이터 정보 요약
        
        Args:
            df: 데이터프레임
        
        Returns:
            정보 딕셔너리
        """
        info = {
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'dtypes': df.dtypes.value_counts().to_dict(),
            'null_counts': df.isnull().sum().to_dict(),
        }
        
        if 'timestamp' in df.columns:
            info['time_range'] = {
                'start': str(df['timestamp'].min()),
                'end': str(df['timestamp'].max()),
                'duration_days': (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400
            }
        
        return info


# ============================================================
# 테스트 및 검증
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DataLoader 테스트 (v1.3.1)")
    print("=" * 60)
    
    # DataLoader 초기화
    loader = DataLoader()
    
    print("\n✅ DataLoader 초기화 성공")
    print(f"  - DATA_DIR: {loader.data_dir}")
    print(f"  - CACHE_DIR: {loader.cache_dir}")
    
    # 테스트 데이터 생성
    from datetime import timedelta
    
    start_time = pd.Timestamp('2025-01-01 00:00:00', tz='UTC')
    n_minutes = 5000  # 약 3.5일
    
    timestamps = [start_time + pd.Timedelta(minutes=i) for i in range(n_minutes)]
    
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
    
    # 데이터 저장 테스트
    print("\n💾 데이터 저장 테스트")
    loader.save_price_data(df_1m_test, "test_data.csv")
    
    # 데이터 로드 테스트
    print("\n📂 데이터 로드 테스트")
    df_loaded = loader.load_price_data("2025-01-01", "2025-01-04", "test_data.csv")
    
    # 데이터 정보 확인
    print("\n📊 데이터 정보")
    info = loader.get_data_info(df_loaded)
    print(f"  - 행: {info['n_rows']:,}개")
    print(f"  - 열: {info['n_cols']}개")
    print(f"  - 메모리: {info['memory_mb']:.2f} MB")
    if 'time_range' in info:
        print(f"  - 기간: {info['time_range']['start']} ~ {info['time_range']['end']}")
        print(f"  - 일수: {info['time_range']['duration_days']:.1f}일")
    
    # 학습 데이터 준비 (캐시 사용)
    print("\n🎯 학습 데이터 준비 (캐시 사용)")
    X, y = loader.prepare_training_data(df_loaded, use_cache=True, lookback_30m_bars=100)
    
    if not X.empty:
        print(f"  - X shape: {X.shape}")
        print(f"  - y shape: {y.shape}")
        print(f"  - 피처 수: {X.shape[1]}개")
        
        # 데이터 품질 검증
        print("\n🔍 데이터 품질 검증")
        valid, errors = loader.validate_data_quality(X, y)
        
        if valid:
            print("  ✅ 검증 통과")
        else:
            print("  ⚠️  검증 경고:")
            for error in errors:
                print(f"    - {error}")
        
        # 데이터 분할 테스트
        print("\n✂️  데이터 분할 테스트")
        X_train, X_valid, X_test, y_train, y_valid, y_test = loader.split_train_valid_test(X, y)
        
        # 시계열 CV 분할
        print("\n🔄 시계열 교차 검증 분할 (sklearn TimeSeriesSplit)")
        splits = loader.time_series_split(X, y, n_splits=3)
        
        # 캐시 재사용 테스트
        print("\n♻️  캐시 재사용 테스트")
        X2, y2 = loader.prepare_training_data(df_loaded, use_cache=True, lookback_30m_bars=100)
        
        if X.equals(X2) and y.equals(y2):
            print("  ✅ 캐시 재사용 성공 (데이터 일치)")
        else:
            print("  ⚠️  캐시 데이터 불일치")
    
    else:
        print("  ❌ 학습 데이터 준비 실패")
    
    # 캐시 키 테스트
    print("\n🔑 캐시 키 생성 테스트")
    params1 = {'lookback': 100, 'version': '1.3.1'}
    params2 = {'lookback': 100, 'version': '1.3.1'}
    params3 = {'lookback': 200, 'version': '1.3.1'}
    
    key1 = loader.get_cache_key(df_loaded, params1)
    key2 = loader.get_cache_key(df_loaded, params2)
    key3 = loader.get_cache_key(df_loaded, params3)
    
    print(f"  - Key1: {key1[:16]}...")
    print(f"  - Key2: {key2[:16]}...")
    print(f"  - Key3: {key3[:16]}...")
    print(f"  - Key1 == Key2: {key1 == key2}")
    print(f"  - Key1 == Key3: {key1 == key3}")
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)
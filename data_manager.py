"""
data_loader.py
단순 데이터 로드/저장 (30분봉 시스템) - 보강 완료
- 1분봉 CSV 로드/저장 (경로 보장, dtype 정규화)
- 30분봉 피처 캐싱 (SHA1 해싱)
- 학습 데이터 분할 (최소 샘플 보장)
- 로그 병합 (피처 로그 포함)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Dict
import pickle
import hashlib

from config import Config
from timeframe_manager import TimeframeManager
from feature_engineer import FeatureEngineer


class DataLoader:
    """데이터 로더 (단순 입출력)"""
    
    def __init__(self):
        self.config = Config
        self.tf_manager = TimeframeManager()
        self.feature_engineer = FeatureEngineer()
        
        # ✅ 1) 디렉토리 보장
        self.raw_dir = self.config.PRICE_DATA_DIR / 'raw'
        self.cache_dir = self.config.PRICE_DATA_DIR / 'cache'
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================
    # 1분봉 로드/저장
    # ==========================================
    
    @staticmethod
    def _normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        """
        ✅ 3) dtype 정규화
        - timestamp: datetime64[ns, UTC]
        - OHLCV: float64
        - NaN 처리
        """
        df = df.copy()
        
        # timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        
        # OHLCV 숫자형 변환
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 결측치 확인
        na_count = df[numeric_cols].isna().sum().sum()
        if na_count > 0:
            print(f"  ⚠️ 결측치 {na_count}개 발견 → 제거")
            df = df.dropna(subset=numeric_cols)
        
        return df
    
    def load_1m_csv(
        self, 
        symbol: str = 'BTCUSDT',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        1분봉 CSV 로드 (dtype 정규화)
        
        Parameters:
        -----------
        symbol : str
            심볼
        start_date : str
            시작 날짜 (YYYYMMDD)
        end_date : str
            종료 날짜 (YYYYMMDD)
        
        Returns:
        --------
        DataFrame: 1분봉 데이터 (정규화됨)
        """
        pattern = f"{symbol}_*_1m.csv"
        files = list(self.raw_dir.glob(pattern))
        
        if len(files) == 0:
            print(f"⚠️ 파일 없음: {pattern}")
            return pd.DataFrame()
        
        dfs = []
        for f in files:
            df = pd.read_csv(f)
            # ✅ 3) dtype 정규화
            df = self._normalize_dtypes(df)
            dfs.append(df)
        
        df_all = pd.concat(dfs, ignore_index=True)
        df_all = df_all.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        
        # 날짜 필터링
        if start_date:
            start = pd.to_datetime(start_date, format='%Y%m%d', utc=True)
            df_all = df_all[df_all['timestamp'] >= start]
        
        if end_date:
            end = pd.to_datetime(end_date, format='%Y%m%d', utc=True) + pd.Timedelta(days=1)
            df_all = df_all[df_all['timestamp'] < end]
        
        print(f"✓ 1분봉 로드: {len(df_all):,}개 ({df_all['timestamp'].min()} ~ {df_all['timestamp'].max()})")
        return df_all
    
    def save_1m_csv(self, df_1m: pd.DataFrame, symbol: str = 'BTCUSDT'):
        """
        1분봉 CSV 저장 (경로 보장)
        """
        # ✅ 1) 디렉토리 보장 (이미 __init__에서 처리)
        
        df_1m = self._normalize_dtypes(df_1m)
        
        start_date = df_1m['timestamp'].min().strftime('%Y%m%d')
        end_date = df_1m['timestamp'].max().strftime('%Y%m%d')
        
        filename = f"{symbol}_{start_date}_{end_date}_1m.csv"
        filepath = self.raw_dir / filename
        
        df_1m.to_csv(filepath, index=False)
        print(f"✓ 1분봉 저장: {filepath.name} ({len(df_1m):,}개)")
    
    # ==========================================
    # 30분봉 피처 캐싱
    # ==========================================
    
    @staticmethod
    def _compute_hash_sha1(df_1m: pd.DataFrame) -> str:
        """
        ✅ 2) SHA1 해싱으로 캐시 키 생성
        - timestamp 범위 + 길이 → SHA1
        - 충돌 회피
        """
        start = df_1m['timestamp'].min().isoformat()
        end = df_1m['timestamp'].max().isoformat()
        length = len(df_1m)
        
        # 문자열 결합 → SHA1
        key_str = f"{start}_{end}_{length}"
        hash_obj = hashlib.sha1(key_str.encode('utf-8'))
        return hash_obj.hexdigest()[:16]  # 16자리만 사용
    
    def load_features_30m(
        self, 
        df_1m: pd.DataFrame, 
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        30분봉 피처 생성 (SHA1 캐싱)
        """
        # ✅ 2) SHA1 해싱
        cache_key = self._compute_hash_sha1(df_1m)
        cache_path = self.cache_dir / f"features_30m_{cache_key}.pkl"
        
        # 캐시 확인
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    features = pickle.load(f)
                print(f"✓ 캐시 로드: {cache_path.name}")
                return features
            except Exception as e:
                print(f"  ⚠️ 캐시 로드 실패: {e}")
        
        # 피처 생성
        print("피처 생성 중...")
        features = self.feature_engineer.create_feature_pool(df_1m, lookback_bars=100)
        
        # ✅ 6) 레짐 진단 로그
        self._print_regime_diagnostics(features)
        
        # 캐싱
        if use_cache:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"✓ 캐시 저장: {cache_path.name}")
            except Exception as e:
                print(f"  ⚠️ 캐싱 실패: {e}")
        
        return features
    
    @staticmethod
    def _print_regime_diagnostics(features: pd.DataFrame):
        """
        ✅ 6) 레짐·ADX 진단 로그
        """
        print("\n[레짐 진단]")
        
        # 레짐 컬럼 확인
        regime_cols = [c for c in features.columns if c.startswith('regime_')]
        
        if 'regime' in features.columns or 'regime_final' in features.columns:
            regime_col = 'regime_final' if 'regime_final' in features.columns else 'regime'
            regime_data = features[regime_col].dropna()
            
            if len(regime_data) > 0:
                total = len(features)
                valid = len(regime_data)
                valid_pct = (valid / total) * 100
                
                print(f"  최종 레짐: {valid:,}/{total:,}개 ({valid_pct:.1f}% 유효)")
                
                # 분포
                dist = regime_data.value_counts().sort_index()
                for val, count in dist.items():
                    pct = (count / valid) * 100
                    label = {1: "UP", -1: "DOWN", 0: "FLAT"}.get(val, f"REGIME-{val}")
                    print(f"    {label}: {count:,}개 ({pct:.1f}%)")
        
        # 타임프레임별 레짐
        for col in regime_cols:
            if col in features.columns:
                data = features[col].dropna()
                if len(data) > 0:
                    valid_pct = (len(data) / len(features)) * 100
                    tf = col.replace('regime_', '')
                    print(f"  {tf}: {valid_pct:.1f}% 유효")
        
        # ADX
        if 'adx_14' in features.columns:
            adx_data = features['adx_14'].dropna()
            if len(adx_data) > 0:
                valid_pct = (len(adx_data) / len(features)) * 100
                mean_adx = adx_data.mean()
                print(f"  ADX-14: {valid_pct:.1f}% 유효 (평균: {mean_adx:.1f})")
        
        print()
    
    # ==========================================
    # 학습 데이터 준비
    # ==========================================
    
    def prepare_train_data(
        self,
        df_1m: pd.DataFrame,
        use_cache: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """학습 데이터 준비 (X, y)"""
        # 피처 생성
        features = self.load_features_30m(df_1m, use_cache=use_cache)
        
        # 타겟 생성
        df_30m = self.tf_manager.aggregate_1m_to_30m(df_1m)
        target = self.feature_engineer.create_target_30m(df_30m)
        
        # 길이 맞추기
        target_aligned = target.iloc[100:].reset_index(drop=True)
        min_len = min(len(features), len(target_aligned))
        
        features = features.iloc[:min_len].copy()
        target = target_aligned.iloc[:min_len].copy()
        
        # NaN 제거
        valid_mask = target.notna()
        features = features[valid_mask].reset_index(drop=True)
        target = target[valid_mask].reset_index(drop=True)
        
        print(f"✓ 학습 데이터: {len(features):,}개")
        print(f"  타겟 분포: {target.value_counts().to_dict()}")
        
        return features, target
    
    def split_train_val_test(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        train_ratio: Optional[float] = None,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None
    ) -> Dict:
        """
        ✅ 4) 시계열 분할 (최소 샘플 보장)
        
        Parameters:
        -----------
        features : DataFrame
        target : Series
        train_ratio : float
            학습 비율 (None이면 Config 사용)
        val_ratio : float
            검증 비율
        test_ratio : float
            테스트 비율
        
        Returns:
        --------
        Dict: {'train': (X, y), 'val': (X, y), 'test': (X, y)}
        """
        if train_ratio is None:
            train_ratio = self.config.TRAIN_RATIO
        if val_ratio is None:
            val_ratio = self.config.VAL_RATIO
        if test_ratio is None:
            test_ratio = self.config.TEST_RATIO
        
        n = len(features)
        
        # ✅ 4) 최소 샘플 보장
        min_samples = 10  # 각 세트 최소 10개
        
        # 비율 자동 조정
        if n < min_samples * 3:
            print(f"  ⚠️ 샘플 부족 ({n}개) → train 전체 사용")
            return {
                'train': (features.copy(), target.copy()),
                'val': (features.iloc[:0].copy(), target.iloc[:0].copy()),
                'test': (features.iloc[:0].copy(), target.iloc[:0].copy())
            }
        
        # 최소 샘플 확보 후 비율 재조정
        n_test = max(min_samples, int(n * test_ratio))
        n_val = max(min_samples, int(n * val_ratio))
        n_train = n - n_test - n_val
        
        if n_train < min_samples:
            # 비율 재분배
            n_train = min_samples
            n_val = max(min_samples, (n - n_train) // 2)
            n_test = n - n_train - n_val
        
        # 분할
        X_train = features.iloc[:n_train].copy()
        y_train = target.iloc[:n_train].copy()
        
        X_val = features.iloc[n_train:n_train + n_val].copy()
        y_val = target.iloc[n_train:n_train + n_val].copy()
        
        X_test = features.iloc[n_train + n_val:].copy()
        y_test = target.iloc[n_train + n_val:].copy()
        
        print(f"✓ 분할:")
        print(f"  Train: {len(X_train):,}개 ({len(X_train)/n:.1%})")
        print(f"  Val:   {len(X_val):,}개 ({len(X_val)/n:.1%})")
        print(f"  Test:  {len(X_test):,}개 ({len(X_test)/n:.1%})")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
    
    # ==========================================
    # 로그 병합
    # ==========================================
    
    def merge_logs(
        self,
        start_date: str,
        end_date: str,
        include_features: bool = True
    ) -> pd.DataFrame:
        """
        ✅ 5) 거래 로그 + 피처 로그 병합
        
        Parameters:
        -----------
        start_date : str
            시작 날짜 (YYYYMMDD)
        end_date : str
            종료 날짜 (YYYYMMDD)
        include_features : bool
            피처 로그 포함 여부
        
        Returns:
        --------
        DataFrame: 병합된 로그
        """
        from log_manager import LogManager
        
        log_mgr = LogManager()
        
        # 거래 로그 로드
        trades = log_mgr.load_all_trades(start_date, end_date)
        
        if len(trades) == 0:
            print("⚠️ 거래 로그 없음")
            return pd.DataFrame()
        
        print(f"✓ 거래 로그: {len(trades):,}개")
        
        # ✅ 5) 피처 로그 병합 (선택적)
        if include_features:
            try:
                # bar30_start 기준으로 조인
                features_list = []
                
                current_date = pd.to_datetime(start_date, format='%Y%m%d')
                end_date_dt = pd.to_datetime(end_date, format='%Y%m%d')
                
                while current_date <= end_date_dt:
                    date_str = current_date.strftime('%Y%m%d')
                    feat_df = log_mgr.load_feature_log(date_str)
                    
                    if not feat_df.empty:
                        features_list.append(feat_df)
                    
                    current_date += pd.Timedelta(days=1)
                
                if features_list:
                    all_features = pd.concat(features_list, ignore_index=True)
                    
                    # bar30_start 기준 병합
                    if 'bar30_start' in trades.columns and 'bar30_start' in all_features.columns:
                        merged = trades.merge(
                            all_features,
                            on='bar30_start',
                            how='left',
                            suffixes=('', '_feat')
                        )
                        
                        feat_matched = merged['p_now'].notna().sum() if 'p_now' in merged.columns else 0
                        print(f"  ✓ 피처 병합: {feat_matched:,}/{len(trades):,}개 매칭")
                        
                        return merged
            
            except Exception as e:
                print(f"  ⚠️ 피처 병합 실패: {e}")
        
        return trades


# ==========================================
# 테스트
# ==========================================
if __name__ == "__main__":
    Config.create_directories()
    
    loader = DataLoader()
    
    # 샘플 데이터
    print("="*60)
    print("DataLoader 테스트")
    print("="*60)
    
    print("\n샘플 데이터 생성...")
    periods = 10000
    ts = pd.date_range(
        end=datetime.now(timezone.utc),
        periods=periods,
        freq='1min',
        tz='UTC'
    )
    
    np.random.seed(42)
    base = 42000 + np.random.randn(periods).cumsum() * 20
    
    rows = []
    for i, t in enumerate(ts):
        b = base[i]
        o = b + np.random.uniform(-20, 20)
        c = b + np.random.uniform(-20, 20)
        h = max(o, c) + np.random.uniform(0, 30)
        l = min(o, c) - np.random.uniform(0, 30)
        v = np.random.uniform(100, 1000)
        rows.append({
            'timestamp': t,
            'open': o,
            'high': h,
            'low': l,
            'close': c,
            'volume': v
        })
    
    df_1m = pd.DataFrame(rows)
    print(f"✓ 샘플: {len(df_1m):,}개")
    
    # 1. 저장/로드 (경로 보장)
    print("\n[1] CSV 저장/로드 (경로 보장)")
    loader.save_1m_csv(df_1m)
    df_loaded = loader.load_1m_csv(symbol='BTCUSDT')
    
    # 2. 피처 생성 (SHA1 캐싱)
    print("\n[2] 피처 생성 (SHA1 캐싱)")
    features = loader.load_features_30m(df_1m, use_cache=True)
    print(f"  피처: {features.shape}")
    
    # 3. 학습 데이터
    print("\n[3] 학습 데이터 준비")
    X, y = loader.prepare_train_data(df_1m)
    
    # 4. 분할 (최소 샘플 보장)
    print("\n[4] 데이터 분할 (최소 샘플 보장)")
    splits = loader.split_train_val_test(X, y)
    
    # 5. 로그 병합 (피처 포함)
    print("\n[5] 로그 병합 (피처 포함)")
    # 실제 로그가 없어서 테스트 스킵
    # merged = loader.merge_logs('20250101', '20250110', include_features=True)
    
    print("\n" + "="*60)
    print("✓ 모든 테스트 완료")
    print("="*60)
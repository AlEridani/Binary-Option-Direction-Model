"""
data_loader.py
단순 데이터 로드/저장 (30분봉 시스템)
- 1분봉 CSV 로드/저장
- 30분봉 피처 캐싱
- 학습 데이터 분할
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, Dict
import pickle

from config import Config
from timeframe_manager import TimeframeManager
from feature_engineer import FeatureEngineer


class DataLoader:
    """데이터 로더 (단순 입출력)"""
    
    def __init__(self):
        self.config = Config
        self.tf_manager = TimeframeManager()
        self.feature_engineer = FeatureEngineer()
        
        # 캐시 디렉토리
        self.cache_dir = self.config.PRICE_DATA_DIR / 'cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================
    # 1분봉 로드/저장
    # ==========================================
    
    def load_1m_csv(
        self, 
        symbol: str = 'BTCUSDT',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """1분봉 CSV 로드"""
        raw_dir = self.config.PRICE_DATA_DIR / 'raw'
        pattern = f"{symbol}_*_1m.csv"
        files = list(raw_dir.glob(pattern))
        
        if len(files) == 0:
            print(f"⚠️ 파일 없음: {pattern}")
            return pd.DataFrame()
        
        dfs = []
        for f in files:
            df = pd.read_csv(f)
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
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
        
        print(f"✓ 1분봉 로드: {len(df_all):,}개")
        return df_all
    
    def save_1m_csv(self, df_1m: pd.DataFrame, symbol: str = 'BTCUSDT'):
        """1분봉 CSV 저장"""
        df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], utc=True)
        start_date = df_1m['timestamp'].min().strftime('%Y%m%d')
        end_date = df_1m['timestamp'].max().strftime('%Y%m%d')
        
        filename = f"{symbol}_{start_date}_{end_date}_1m.csv"
        filepath = self.config.PRICE_DATA_DIR / 'raw' / filename
        
        df_1m.to_csv(filepath, index=False)
        print(f"✓ 1분봉 저장: {filepath.name}")
    
    # ==========================================
    # 30분봉 피처 캐싱
    # ==========================================
    
    def _get_cache_key(self, df_1m: pd.DataFrame) -> str:
        """캐시 키 생성"""
        start = df_1m['timestamp'].min()
        end = df_1m['timestamp'].max()
        length = len(df_1m)
        return f"{start}_{end}_{length}".replace(' ', '_').replace(':', '')
    
    def load_features_30m(
        self, 
        df_1m: pd.DataFrame, 
        use_cache: bool = True
    ) -> pd.DataFrame:
        """30분봉 피처 생성 (캐싱)"""
        cache_key = self._get_cache_key(df_1m)
        cache_path = self.cache_dir / f"features_30m_{cache_key}.pkl"
        
        # 캐시 확인
        if use_cache and cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    features = pickle.load(f)
                print(f"✓ 캐시 로드: {cache_path.name}")
                return features
            except:
                pass
        
        # 피처 생성
        print("피처 생성 중...")
        features = self.feature_engineer.create_feature_pool(df_1m, lookback_bars=100)
        
        # 캐싱
        if use_cache:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"✓ 캐시 저장: {cache_path.name}")
            except:
                pass
        
        return features
    
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
        target: pd.Series
    ) -> Dict:
        """시계열 분할"""
        n = len(features)
        n_train = int(n * self.config.TRAIN_RATIO)
        n_val = int(n * self.config.VAL_RATIO)
        
        X_train = features.iloc[:n_train].copy()
        y_train = target.iloc[:n_train].copy()
        
        X_val = features.iloc[n_train:n_train + n_val].copy()
        y_val = target.iloc[n_train:n_train + n_val].copy()
        
        X_test = features.iloc[n_train + n_val:].copy()
        y_test = target.iloc[n_train + n_val:].copy()
        
        print(f"✓ 분할: Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }


# ==========================================
# 테스트
# ==========================================
if __name__ == "__main__":
    Config.create_directories()
    
    loader = DataLoader()
    
    # 샘플 데이터
    print("샘플 데이터 생성...")
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
    
    # 1. 저장/로드
    print("\n[1] CSV 저장/로드")
    loader.save_1m_csv(df_1m)
    df_loaded = loader.load_1m_csv(symbol='BTCUSDT')
    
    # 2. 피처 생성 (캐싱)
    print("\n[2] 피처 생성")
    features = loader.load_features_30m(df_1m, use_cache=True)
    
    # 3. 학습 데이터
    print("\n[3] 학습 데이터 준비")
    X, y = loader.prepare_train_data(df_1m)
    
    # 4. 분할
    print("\n[4] 데이터 분할")
    splits = loader.split_train_val_test(X, y)
    
    print("\n✓ 테스트 완료")
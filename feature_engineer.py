"""
피처 엔지니어링 - 30분 바이너리 옵션 전용
버전: 1.3.1 (미래 데이터 누수 완전 제거)

30분 바이너리 옵션 구조:
- 진입 시점: N번째 봉 종료 직후 (예: 10:00:04)
- 진입 가격: N+1번째 봉 open (예: 10:00:04 시점 가격)
- 만기 시점: 진입 후 정확히 30분 (예: 10:30:04)
- 정산 가격: N+1번째 봉 close (예: 10:30:00 종가)
- 타겟: 정산 가격 > 진입 가격
"""

import pandas as pd
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

import config
from timeframe_manager import TimeframeManager


class FeatureEngineer:
    """
    30분봉 기반 피처 엔지니어링
    - 테크니컬 지표 계산
    - 멀티타임프레임 레짐 (4h, 1h, 15m)
    - ADX 가중 레짐 점수
    - 30분 바이너리 옵션 타겟 생성
    - 미래 누수 완전 방지
    """
    
    def __init__(self):
        """인자 없이 초기화 (Config 참조)"""
        self.tf_manager = TimeframeManager()
        self.regime_timeframes = config.REGIME_TIMEFRAMES
        self.regime_weights = config.REGIME_WEIGHTS
        self.adx_threshold = config.ADX_THRESHOLD
        self.regime_lookback = config.REGIME_LOOKBACK
    
    def create_feature_pool(self, df_1m: pd.DataFrame, lookback_30m_bars: int = 100) -> pd.DataFrame:
        """
        1분봉 → 30분봉 집계 및 전체 피처 생성
        
        Args:
            df_1m: 1분봉 데이터
            lookback_30m_bars: 계산에 사용할 과거 30분봉 수
        
        Returns:
            피처가 포함된 30분봉 데이터 (타겟 포함)
        """
        # 1분봉 → 30분봉 집계 (백테스트 모드: 완성된 봉만)
        df_30m = self.tf_manager.aggregate_1m_to_30m(df_1m, realtime_safe=False)
        
        if df_30m.empty:
            print("❌ 30분봉 집계 결과 없음")
            return pd.DataFrame()
        
        if len(df_30m) < lookback_30m_bars:
            print(f"❌ 30분봉 부족: {len(df_30m)} < {lookback_30m_bars}")
            return pd.DataFrame()
        
        df = df_30m.copy()
        
        print(f"📊 30분봉 집계 완료: {len(df)}개")
        
        # 기본 테크니컬 지표 (과거 데이터만 사용)
        df = self._add_moving_averages(df)
        df = self._add_rsi(df)
        df = self._add_macd(df)
        df = self._add_bollinger_bands(df)
        df = self._add_stochastic(df)
        df = self._add_adx(df)
        
        # 멀티타임프레임 레짐 (과거 데이터만 사용)
        df = self._add_multi_regime(df)
        
        # 레짐 점수 (ADX 가중)
        df = self._calculate_regime_score(df)
        
        # 30분 바이너리 옵션 타겟 생성 (핵심!)
        df = self._create_target_30m_binary(df)
        
        # 결측치 제거 (마지막 봉은 타겟이 NaN이므로 자동 제거됨)
        initial_len = len(df)
        df = df.dropna().reset_index(drop=True)
        
        if df.empty:
            print(f"❌ 결측치 제거 후 데이터 없음 (초기: {initial_len}개)")
            return pd.DataFrame()
        
        print(f"✅ 피처 생성 완료: {len(df)}개 (초기 {initial_len}개 → 결측 제거)")
        
        # 버전 정보 추가
        df['feature_ver'] = config.FEATURE_VERSION
        df['data_ver'] = config.DATA_VERSION
        
        return df
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """이동평균 추가 (과거 데이터만 사용)"""
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        return df
    
    def _add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """RSI 추가"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """MACD 추가"""
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, num_std: int = 2) -> pd.DataFrame:
        """볼린저 밴드 추가"""
        df['bb_middle'] = df['close'].rolling(window=period).mean()
        bb_std = df['close'].rolling(window=period).std()
        
        df['bb_upper'] = df['bb_middle'] + (bb_std * num_std)
        df['bb_lower'] = df['bb_middle'] - (bb_std * num_std)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        return df
    
    def _add_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Stochastic 추가"""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        return df
    
    def _add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """ADX 및 DI+/DI- 추가"""
        high_diff = df['high'].diff()
        low_diff = -df['low'].diff()
        
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        
        # DM+ / DM-
        dm_plus = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
        dm_minus = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
        
        # Smoothed DM
        dm_plus_smooth = dm_plus.rolling(window=period).mean()
        dm_minus_smooth = dm_minus.rolling(window=period).mean()
        
        # DI+ / DI-
        df['di_plus'] = 100 * (dm_plus_smooth / atr)
        df['di_minus'] = 100 * (dm_minus_smooth / atr)
        
        # DX
        di_sum = df['di_plus'] + df['di_minus']
        di_diff = abs(df['di_plus'] - df['di_minus'])
        dx = 100 * (di_diff / di_sum)
        
        # ADX
        df['adx'] = dx.rolling(window=period).mean()
        
        return df
    
    def calculate_regime(self, df: pd.DataFrame, timeframe: str) -> pd.Series:
        """
        단일 타임프레임 레짐 계산 (타임스탬프 기반 정렬)
        
        Args:
            df: 30분봉 데이터 (bar30_start 컬럼 필수)
            timeframe: '4h', '1h', '15m'
        
        Returns:
            레짐 시리즈 (1: UP, -1: DOWN, 0: FLAT)
        """
        # 해당 타임프레임으로 리샘플
        df_tf = self.tf_manager.resample_to_timeframe(df, timeframe)
        
        if df_tf.empty or len(df_tf) < self.regime_lookback.get(timeframe, 8):
            return pd.Series(0, index=df.index)
        
        # EMA 기반 추세 판단
        lookback = self.regime_lookback.get(timeframe, 8)
        ema_fast = df_tf['close'].ewm(span=max(lookback//2, 2), adjust=False).mean()
        ema_slow = df_tf['close'].ewm(span=lookback, adjust=False).mean()
        
        # 레짐 결정
        df_tf['regime'] = 0
        df_tf.loc[ema_fast > ema_slow, 'regime'] = 1   # UP
        df_tf.loc[ema_fast < ema_slow, 'regime'] = -1  # DOWN
        
        # 30분봉에 매핑 (타임스탬프 기반 merge)
        # df_tf의 시작 시각 컬럼 추출
        tf_start_col = f'bar_{timeframe}_start'
        if tf_start_col not in df_tf.columns:
            # 첫 번째 datetime 컬럼 사용
            datetime_cols = df_tf.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                tf_start_col = datetime_cols[0]
            else:
                return pd.Series(0, index=df.index)
        
        # 30분봉 각 행에 대해 해당하는 레짐 찾기 (forward fill)
        df_30m_aligned = df[['bar30_start']].copy()
        df_30m_aligned['regime'] = 0
        
        df_tf_sorted = df_tf.sort_values(tf_start_col).reset_index(drop=True)
        
        for idx, row in df_30m_aligned.iterrows():
            bar_start = row['bar30_start']
            
            # 해당 시점 이전의 가장 최근 레짐 값 찾기 (forward fill)
            mask = df_tf_sorted[tf_start_col] <= bar_start
            if mask.any():
                regime_val = df_tf_sorted[mask]['regime'].iloc[-1]
                df_30m_aligned.at[idx, 'regime'] = regime_val
        
        return df_30m_aligned['regime']
    
    def _add_multi_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """멀티타임프레임 레짐 추가"""
        print(f"🔄 멀티타임프레임 레짐 계산 중... ({self.regime_timeframes})")
        
        # 각 타임프레임별 레짐 계산
        df['regime_4h'] = self.calculate_regime(df, '4h')
        df['regime_1h'] = self.calculate_regime(df, '1h')
        df['regime_15m'] = self.calculate_regime(df, '15m')
        
        # 가중 합산
        df['regime_final'] = (
            df['regime_4h'] * self.regime_weights['4h'] +
            df['regime_1h'] * self.regime_weights['1h'] +
            df['regime_15m'] * self.regime_weights['15m']
        )
        
        # 최종 레짐 (반올림)
        df['regime'] = df['regime_final'].apply(lambda x: 1 if x > 0.3 else (-1 if x < -0.3 else 0))
        
        return df
    
    def _calculate_regime_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """ADX 가중 레짐 점수 계산"""
        if 'regime_final' in df.columns and 'adx' in df.columns:
            # ADX를 0-1 범위로 정규화
            adx_norm = df['adx'] / 100.0
            adx_norm = adx_norm.clip(0, 1)
            
            # 레짐 점수 = 레짐 방향 × ADX 신뢰도
            df['regime_score'] = df['regime_final'] * adx_norm
        else:
            df['regime_score'] = 0.0
        
        return df
    
    def _create_target_30m_binary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        30분 바이너리 옵션 타겟 생성
        
        타임라인:
        [봉 N]           [봉 N+1]
        09:30~10:00      10:00~10:30
        |                |        |
        피처 계산        진입     만기
        (과거 데이터)    (open)   (close)
        
        - N번째 봉: 피처 계산에만 사용 (학습용)
        - N+1번째 봉: 실제 진입/정산
          - 진입 시점: 10:00:04초 (N+1 봉 시작 직후)
          - 진입 가격: N+1 봉 open
          - 만기 시점: 10:30:04초 (진입 후 정확히 30분)
          - 정산 가격: N+1 봉 close
        - 타겟: N+1 봉의 (close > open) 여부
        
        Returns:
            df with 'target' column
        """
        # 30분 바이너리: 같은 봉의 close가 open보다 높은가?
        df['target_raw'] = (df['close'] > df['open']).astype(int)
        
        # 미래 누수 방지: N번째 봉에서 N+1번째 결과를 예측
        # N번째 피처 → N+1번째 타겟
        df['target'] = df['target_raw'].shift(-1)
        
        # 임시 컬럼 제거
        df = df.drop(['target_raw'], axis=1)
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> list:
        """학습에 사용할 피처 이름 반환"""
        exclude_cols = [
            'bar30_start', 'bar30_end', 'target',
            'feature_ver', 'data_ver', 'timestamp',
            'open', 'high', 'low', 'close', 'volume'  # OHLCV는 제외 (피처에서 파생된 지표만 사용)
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return feature_cols
    
    def validate_features(self, df: pd.DataFrame) -> tuple:
        """피처 품질 검증"""
        errors = []
        warnings_list = []
        
        # 1. 타겟 존재 여부
        if 'target' not in df.columns:
            errors.append("타겟 컬럼 없음")
        
        # 2. 타겟 분포 체크
        if 'target' in df.columns:
            target_ratio = df['target'].mean()
            if target_ratio < 0.40 or target_ratio > 0.60:
                warnings_list.append(f"타겟 불균형: UP={target_ratio:.2%} (40~60% 권장)")
        
        # 3. 필수 피처 체크
        required_features = ['regime', 'regime_score', 'adx', 'rsi_14', 'macd']
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            errors.append(f"필수 피처 누락: {missing_features}")
        
        # 4. NaN 체크
        nan_counts = df.isnull().sum()
        if nan_counts.sum() > 0:
            warnings_list.append(f"결측치 존재: {nan_counts[nan_counts > 0].to_dict()}")
        
        # 5. Inf 체크
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_counts = np.isinf(df[numeric_cols]).sum()
        if inf_counts.sum() > 0:
            errors.append(f"무한대 존재: {inf_counts[inf_counts > 0].to_dict()}")
        
        return (len(errors) == 0, errors, warnings_list)


# ============================================================
# 테스트 및 검증
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FeatureEngineer 테스트 (v1.3.1 - 30분 바이너리 옵션)")
    print("=" * 60)
    
    # 테스트 데이터 생성 (3일치)
    from datetime import datetime, timedelta
    
    start_time = pd.Timestamp('2025-01-01 00:00:00', tz='UTC')
    n_minutes = 4320  # 3일 = 4320분
    
    timestamps = [start_time + pd.Timedelta(minutes=i) for i in range(n_minutes)]
    
    # 랜덤 워크 시뮬레이션
    np.random.seed(42)
    price = 50000
    prices = [price]
    
    for _ in range(n_minutes - 1):
        change = np.random.normal(0, 100)
        price = max(price + change, 45000)  # 하한선
        prices.append(price)
    
    df_1m_test = pd.DataFrame({
        'timestamp': timestamps,
        'close': prices
    })
    
    # OHLC 생성
    df_1m_test['open'] = df_1m_test['close'].shift(1).fillna(df_1m_test['close'])
    df_1m_test['high'] = df_1m_test[['open', 'close']].max(axis=1) * 1.002
    df_1m_test['low'] = df_1m_test[['open', 'close']].min(axis=1) * 0.998
    df_1m_test['volume'] = np.random.uniform(1000, 5000, n_minutes)
    
    # FeatureEngineer 초기화
    fe = FeatureEngineer()
    
    print("\n✅ FeatureEngineer 초기화 성공")
    print(f"  - 레짐 타임프레임: {fe.regime_timeframes}")
    print(f"  - 레짐 가중치: {fe.regime_weights}")
    print(f"  - ADX 임계값: {fe.adx_threshold}")
    print(f"  - 레짐 룩백: {fe.regime_lookback}")
    
    # 피처 생성
    print("\n📊 피처 생성 테스트")
    print(f"  입력: {len(df_1m_test):,}개 1분봉")
    
    df_features = fe.create_feature_pool(df_1m_test, lookback_30m_bars=100)
    
    if not df_features.empty:
        print(f"  출력: {len(df_features)}개 30분봉 + 피처")
        print(f"  피처 수: {len(fe.get_feature_names(df_features))}개")
        
        # 피처 목록
        print("\n📋 생성된 피처 목록:")
        feature_names = fe.get_feature_names(df_features)
        for i, name in enumerate(feature_names, 1):
            print(f"  {i:2d}. {name}")
        
        # 샘플 데이터 (타겟 정합성 확인)
        print("\n📈 샘플 데이터 (타겟 정합성 확인):")
        sample_cols = ['bar30_start', 'open', 'close', 'regime', 'adx', 'rsi_14', 'target']
        sample_df = df_features[sample_cols].tail(5)
        
        # 타겟 계산 검증
        for idx, row in sample_df.iterrows():
            manual_target = 1 if row['close'] > row['open'] else 0
            print(f"  [{row['bar30_start']}]")
            print(f"    open={row['open']:.2f}, close={row['close']:.2f}")
            print(f"    실제 타겟={row['target']}, 검증={manual_target}")
        
        # 레짐 분포
        print("\n🎯 레짐 분포:")
        regime_counts = df_features['regime'].value_counts().sort_index()
        for regime, count in regime_counts.items():
            regime_name = {1: 'UP', 0: 'FLAT', -1: 'DOWN'}.get(regime, 'UNKNOWN')
            pct = count / len(df_features) * 100
            print(f"  {regime_name:5s} ({regime:2d}): {count:4d}개 ({pct:5.1f}%)")
        
        # 타겟 분포
        print("\n🎲 타겟 분포 (30분 바이너리 옵션):")
        target_counts = df_features['target'].value_counts().sort_index()
        for target, count in target_counts.items():
            target_name = {1: 'UP (close > open)', 0: 'DOWN (close <= open)'}.get(target, 'UNKNOWN')
            pct = count / len(df_features) * 100
            print(f"  {target_name}: {count:4d}개 ({pct:5.1f}%)")
        
        # 데이터 품질 검증
        print("\n🔍 피처 품질 검증:")
        valid, errors, warnings_list = fe.validate_features(df_features)
        
        if valid:
            print("  ✅ 검증 통과")
        else:
            print("  ❌ 오류 발견:")
            for error in errors:
                print(f"    - {error}")
        
        if warnings_list:
            print("  ⚠️  경고:")
            for warning in warnings_list:
                print(f"    - {warning}")
        
    else:
        print("  ❌ 피처 생성 실패")
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)
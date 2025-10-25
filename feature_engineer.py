"""
feature_engineer.py
30분봉 기반 피처 엔지니어링 (전체 재작성)
- 1분봉 → 30분봉 집계
- 30분봉 close 시점에 판단
- 다음 봉 open 진입 → close 만기 예측
- 멀티타임프레임 레짐 (4h, 1h, 30min, 15min) - 1분봉 기준 계산
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from config import Config
from timeframe_manager import TimeframeManager


class FeatureEngineer:
    """피처 엔지니어링 (30분봉 기준, 미래 누출 방지)"""
    
    def __init__(self):
        self.regime_timeframes = Config.REGIME_TIMEFRAMES
        self.regime_weights = Config.REGIME_WEIGHTS
        self.adx_threshold = Config.REGIME_ADX_THR
        self.adx_window = Config.REGIME_ADX_WINDOW
        self.tf_manager = TimeframeManager()
    
    # ==========================================
    # ADX 계산
    # ==========================================
    
    @staticmethod
    def _compute_adx_core(high, low, close, window=14):
        """ADX, +DI, -DI 계산 (핵심 로직)"""
        # True Range 계산
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high.diff()
        down_move = -low.diff()
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        
        # Smoothing
        alpha = 1.0 / window
        atr = tr.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
        plus_dm_smooth = pd.Series(plus_dm, index=high.index).ewm(
            alpha=alpha, adjust=False, min_periods=window
        ).mean()
        minus_dm_smooth = pd.Series(minus_dm, index=high.index).ewm(
            alpha=alpha, adjust=False, min_periods=window
        ).mean()
        
        # Directional Indicators
        plus_di = 100.0 * (plus_dm_smooth / (atr + 1e-9))
        minus_di = 100.0 * (minus_dm_smooth / (atr + 1e-9))
        
        # ADX
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        adx = dx.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
        
        return plus_di, minus_di, adx
    
    # ==========================================
    # 멀티타임프레임 레짐 (1분봉 기준 계산)
    # ==========================================
    
    def compute_regime_single_tf(self, df_1m, timeframe='15min'):
        """
        단일 타임프레임 레짐 계산 (1분봉 기준)
        
        Parameters:
        -----------
        df_1m : DataFrame
            1분봉 데이터 (timestamp, open, high, low, close, volume)
        timeframe : str
            리샘플 타임프레임 ('4h', '1h', '30min', '15min')
        
        Returns:
        --------
        Series: 레짐 (-1: DOWN, 0: FLAT, 1: UP)
        """
        df = df_1m.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp').sort_index()
        
        # 타임프레임 리샘플
        ohlc = pd.DataFrame({
            'open': df['open'].resample(timeframe).first(),
            'high': df['high'].resample(timeframe).max(),
            'low': df['low'].resample(timeframe).min(),
            'close': df['close'].resample(timeframe).last(),
            'volume': df['volume'].resample(timeframe).sum(),
        }).dropna()
        
        if len(ohlc) < self.adx_window:
            # 데이터 부족 시 FLAT 반환
            return pd.Series(0, index=df.index, name=f'regime_{timeframe}')
        
        # ADX 계산
        pdi, mdi, adx = self._compute_adx_core(
            ohlc['high'], ohlc['low'], ohlc['close'], 
            window=self.adx_window
        )
        
        # 레짐 판단
        strong_trend = adx > self.adx_threshold
        up_trend = pdi > mdi
        
        regime = np.select(
            [strong_trend & up_trend, strong_trend & ~up_trend],
            [1, -1],
            default=0
        )
        
        regime_series = pd.Series(regime, index=ohlc.index, name=f'regime_{timeframe}')
        
        # 1분봉 타임프레임으로 forward fill
        regime_reindexed = regime_series.reindex(df.index, method='ffill')
        
        return regime_reindexed
    
    def compute_multi_timeframe_regime(self, df_1m, target_30m_index):
        """
        멀티타임프레임 레짐 계산 (1분봉 → 30분봉 인덱스)
        ✅ 4h, 1h, 30min, 15min 모두 계산
        
        Parameters:
        -----------
        df_1m : DataFrame
            1분봉 원본 데이터
        target_30m_index : Index or Series
            목표 30분봉 인덱스 (bar30_start)
        
        Returns:
        --------
        DataFrame: 30분봉 인덱스 기준 레짐 정보
        """
        # 1분봉 인덱스 설정
        df_1m_indexed = df_1m.copy()
        df_1m_indexed['timestamp'] = pd.to_datetime(df_1m_indexed['timestamp'], utc=True)
        df_1m_indexed = df_1m_indexed.set_index('timestamp').sort_index()
        
        # 30분봉 타임스탬프 생성
        if isinstance(target_30m_index, pd.Series):
            bar30_timestamps = pd.to_datetime(target_30m_index.values, utc=True)
        else:
            bar30_timestamps = pd.to_datetime(target_30m_index, utc=True)
        
        result = pd.DataFrame(index=range(len(bar30_timestamps)))
        
        # 주말 체크
        result['is_weekend'] = bar30_timestamps.dayofweek.isin([5, 6]).astype(int)
        
        # 각 타임프레임별 레짐 계산
        regime_scores = []
        regime_diagnostics = {}  # 진단용
        
        print("  멀티타임프레임 레짐 계산:")
        
        for tf in self.regime_timeframes:
            print(f"    - {tf} 레짐 계산 중...", end=' ')
            
            # 1분봉에서 직접 타임프레임별 레짐 계산
            regime_1m = self.compute_regime_single_tf(df_1m, tf)
            
            # 30분봉 시작 시각으로 매핑
            regime_30m = []
            valid_count = 0

            # 벡터화된 매핑 처리
            regime_1m = self.compute_regime_single_tf(df_1m, tf)

            # 30분봉 타임스탬프 기준으로 forward-fill 매핑
            mapped = regime_1m.reindex(pd.to_datetime(bar30_timestamps, utc=True), method='ffill')

            # NaN 방어
            mapped = mapped.fillna(0)

            # numpy 변환 후 결과 저장
            regime_30m = mapped.to_numpy()
            result[f'regime_{tf}'] = regime_30m

            # 진단용 통계 (기존 valid_count 계산 대체)
            valid_count = np.count_nonzero(~np.isnan(regime_30m))
            total_count = len(regime_30m)
            valid_pct = (valid_count / total_count) * 100 if total_count > 0 else 0
            regime_diagnostics[tf] = (valid_count, total_count, valid_pct)
            print(f"✓ ({valid_pct:.1f}% 유효)")
            
            # 가중치 적용
            weight = self.regime_weights.get(tf, 0)
            regime_scores.append(np.array(regime_30m) * weight)
        
        # 가중 평균 점수
        result['regime_score'] = np.sum(regime_scores, axis=0)
        
        # 최종 레짐 결정
        threshold = 0.3
        result['regime_final'] = np.select(
            [result['regime_score'] > threshold, result['regime_score'] < -threshold],
            [1, -1],
            default=0
        )
        
        # ✅ 진단 출력
        self._print_regime_diagnostics(regime_diagnostics, result)
        
        return result
    
    @staticmethod
    def _print_regime_diagnostics(diagnostics: dict, result: pd.DataFrame):
        """
        ✅ 레짐 진단 로그 출력
        """
        print("\n[레짐 진단]")
        
        # 타임프레임별 유효율
        for tf, (valid, total, pct) in diagnostics.items():
            print(f"  {tf:6s}: {valid:5d}/{total:5d}개 ({pct:5.1f}% 유효)")
        
        # 최종 레짐 분포
        if 'regime_final' in result.columns:
            regime_data = result['regime_final']
            regime_dist = regime_data.value_counts().sort_index()
            
            print(f"\n  최종 레짐 분포:")
            regime_labels = {1: "UP 🟢", -1: "DOWN 🔴", 0: "FLAT ⚪"}
            total = len(regime_data)
            
            for val in [1, 0, -1]:
                count = regime_dist.get(val, 0)
                label = regime_labels.get(val, f"REGIME-{val}")
                pct = (count / total) * 100 if total > 0 else 0
                print(f"    {label:12s}: {count:5d}개 ({pct:5.1f}%)")
        
        # 레짐 스코어 통계
        if 'regime_score' in result.columns:
            score_mean = result['regime_score'].mean()
            score_std = result['regime_score'].std()
            score_min = result['regime_score'].min()
            score_max = result['regime_score'].max()
            print(f"\n  레짐 스코어:")
            print(f"    평균={score_mean:+.3f}, 표준편차={score_std:.3f}")
            print(f"    범위=[{score_min:+.3f}, {score_max:+.3f}]")
        
        print()
    
    # ==========================================
    # 30분봉 피처 생성
    # ==========================================
    
    def create_feature_pool(self, df_1m, lookback_bars=100):
        """
        30분봉 기반 피처 풀 생성
        
        Parameters:
        -----------
        df_1m : DataFrame
            1분봉 원본 데이터
        lookback_bars : int
            초기 룩백 제거 (30분봉 기준)
        
        Returns:
        --------
        DataFrame: 30분봉 피처 데이터프레임
        """
        # ======================================
        # 1. 1분봉 → 30분봉 집계
        # ======================================
        print("1분봉 → 30분봉 집계 중...")
        df_30m = self.tf_manager.aggregate_1m_to_30m(df_1m)

        # TimeframeManager 반환값 확인 및 보정
        if 'bar30_start' not in df_30m.columns:
            df_30m = df_30m.reset_index()
        if 'index' in df_30m.columns:
            df_30m = df_30m.rename(columns={'index': 'bar30_start'})

        if 'bar30_end' not in df_30m.columns:
            df_30m['bar30_end'] = pd.to_datetime(df_30m['bar30_start'], utc=True) + pd.Timedelta(minutes=30)
            
        print(f"✓ 30분봉 생성: {len(df_30m):,}개 바")
        
        # ======================================
        # 2. 피처 DataFrame 초기화
        # ======================================

        features = pd.DataFrame()
        features['bar30_start'] = df_30m['bar30_start'].values
        features['bar30_end'] = df_30m['bar30_end'].values
        features['timestamp'] = df_30m['bar30_start'].values
        
        # m1_index 계산
        if 'm1_index_entry' in df_30m.columns:
            features['m1_index_entry'] = df_30m['m1_index_entry'].values
        else:
            features['m1_index_entry'] = (
                pd.to_datetime(df_30m['bar30_end'], utc=True).astype('int64') // 10**9 // 60
            ).astype('int64').values

        if 'm1_index_label' in df_30m.columns:
            features['m1_index_label'] = df_30m['m1_index_label'].values
        else:
            features['m1_index_label'] = (
                (pd.to_datetime(df_30m['bar30_end'], utc=True) + pd.Timedelta(minutes=30)).astype('int64') // 10**9 // 60
            ).astype('int64').values
        
        # ======================================
        # 3. 과거 캔들 (shift로 미래 누수 방지)
        # ======================================
        features['prev_open'] = df_30m['open'].shift(1).values
        features['prev_high'] = df_30m['high'].shift(1).values
        features['prev_low'] = df_30m['low'].shift(1).values
        features['prev_close'] = df_30m['close'].shift(1).values
        features['prev_volume'] = df_30m['volume'].shift(1).values
        
        # ======================================
        # 4. 수익률 (30분봉 기준)
        # ======================================
        sc = df_30m['close'].shift(1)
        for period in [1, 2, 3, 5, 10, 20]:
            features[f'return_{period}'] = (sc / df_30m['close'].shift(period + 1) - 1).values
        
        # ======================================
        # 5. 거래량 변화 (30분봉 기준)
        # ======================================
        sv = df_30m['volume'].shift(1)
        for period in [1, 2, 3, 5, 10]:
            features[f'volume_change_{period}'] = (
                sv / df_30m['volume'].shift(period + 1) - 1
            ).values
        
        # ======================================
        # 6. 이동평균 (30분봉 기준)
        # ======================================
        for period in [5, 10, 20, 50, 100]:
            ma = sc.rolling(window=period, min_periods=period).mean()
            features[f'ma_{period}'] = ma.values
            features[f'price_to_ma_{period}'] = (sc / (ma + 1e-9) - 1).values
            features[f'ma_{period}_slope'] = (ma.diff(3) / (ma.shift(3) + 1e-9)).values
        
        # ======================================
        # 7. EMA (30분봉 기준)
        # ======================================
        for period in [12, 26, 50]:
            ema = sc.ewm(span=period, adjust=False, min_periods=period).mean()
            features[f'ema_{period}'] = ema.values
            features[f'price_to_ema_{period}'] = (sc / (ema + 1e-9) - 1).values
        
        # ======================================
        # 8. 볼린저 밴드 (30분봉 기준)
        # ======================================
        for period in [20, 50]:
            ma = sc.rolling(window=period, min_periods=period).mean()
            std = sc.rolling(window=period, min_periods=period).std()
            upper = ma + 2 * std
            lower = ma - 2 * std
            features[f'bb_upper_{period}'] = upper.values
            features[f'bb_lower_{period}'] = lower.values
            features[f'bb_width_{period}'] = (4 * std / (ma + 1e-9)).values
            features[f'bb_position_{period}'] = ((sc - lower) / ((upper - lower) + 1e-9)).values
        
        # ======================================
        # 9. RSI (30분봉 기준)
        # ======================================
        for period in [14, 28]:
            delta = sc.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
            rs = gain / (loss + 1e-9)
            features[f'rsi_{period}'] = (100 - (100 / (1 + rs))).values
        
        # ======================================
        # 10. MACD (30분봉 기준)
        # ======================================
        ema12 = sc.ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = sc.ewm(span=26, adjust=False, min_periods=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        features['macd'] = macd.values
        features['macd_signal'] = signal.values
        features['macd_histogram'] = (macd - signal).values
        
        # ======================================
        # 11. Stochastic (30분봉 기준)
        # ======================================
        for period in [14]:
            sh = df_30m['high'].shift(1)
            sl = df_30m['low'].shift(1)
            low_min = sl.rolling(window=period, min_periods=period).min()
            high_max = sh.rolling(window=period, min_periods=period).max()
            features[f'stoch_{period}'] = (
                (sc - low_min) / ((high_max - low_min) + 1e-9) * 100
            ).values
        
        # ======================================
        # 12. ATR (30분봉 기준)
        # ======================================
        for period in [14, 28]:
            sh = df_30m['high'].shift(1)
            sl = df_30m['low'].shift(1)
            sc_prev = df_30m['close'].shift(2)
            tr = pd.concat([
                sh - sl,
                (sh - sc_prev).abs(),
                (sl - sc_prev).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(window=period, min_periods=period).mean()
            features[f'atr_{period}'] = atr.values
            features[f'atr_ratio_{period}'] = (atr / (sc + 1e-9)).values
        
        # ======================================
        # 13. 거래량 피처 (30분봉 기준)
        # ======================================
        features['volume_sma_10'] = sv.rolling(window=10, min_periods=10).mean().values
        features['volume_sma_50'] = sv.rolling(window=50, min_periods=50).mean().values
        features['volume_ratio'] = (sv / (features['volume_sma_10'] + 1e-9)).values
        features['volume_trend'] = (features['volume_sma_10'] / (features['volume_sma_50'] + 1e-9)).values
        
        # OBV
        price_diff = df_30m['close'].diff().shift(1)
        obv = (np.sign(price_diff) * sv).cumsum()
        features['obv'] = obv.values
        features['obv_ema'] = obv.ewm(span=20, adjust=False, min_periods=20).mean().values
        features['obv_signal'] = (obv / (features['obv_ema'] + 1e-9) - 1).values

        # ======================================
        # 14. 캔들 패턴 (30분봉 기준)
        # ======================================
        po = df_30m['open'].shift(1)
        pc = df_30m['close'].shift(1)
        ph = df_30m['high'].shift(1)
        pl = df_30m['low'].shift(1)
        
        features['body_size'] = ((pc - po).abs() / (po + 1e-9)).values
        features['upper_shadow'] = ((ph - pd.concat([po, pc], axis=1).max(axis=1)) / (po + 1e-9)).values
        features['lower_shadow'] = ((pd.concat([po, pc], axis=1).min(axis=1) - pl) / (po + 1e-9)).values
        features['body_position'] = ((pc - po) / ((ph - pl) + 1e-9)).values
        
        for i in range(1, 4):
            features[f'candle_direction_{i}'] = np.sign(
                df_30m['close'].shift(i) - df_30m['open'].shift(i)
            ).values
            features[f'candle_size_{i}'] = (
                (df_30m['high'].shift(i) - df_30m['low'].shift(i)) / 
                (df_30m['close'].shift(i) + 1e-9)
            ).values
        
        # ======================================
        # 15. 시간 피처
        # ======================================
        dt = pd.to_datetime(df_30m['bar30_start'], utc=True)
        features['hour'] = dt.dt.hour.values
        features['minute'] = dt.dt.minute.values
        features['day_of_week'] = dt.dt.dayofweek.values
        features['day_of_month'] = dt.dt.day.values

        # 순환 인코딩
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24.0)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24.0)
        features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7.0)
        features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7.0)
        
        # ======================================
        # 16. 마이크로구조 (30분봉 기준)
        # ======================================
        for period in [3, 5, 10]:
            sh = df_30m['high'].shift(1)
            sl = df_30m['low'].shift(1)
            features[f'high_low_ratio_{period}'] = (
                (sh / (sl + 1e-9) - 1).rolling(period, min_periods=period).mean()
            ).values
            features[f'close_position_{period}'] = (
                ((sc - sl) / ((sh - sl) + 1e-9)).rolling(period, min_periods=period).mean()
            ).values
        
        # ======================================
        # 17. 추세 강도 (30분봉 기준)
        # ======================================
        for period in [10, 20, 50]:
            ma = sc.rolling(window=period, min_periods=period).mean()
            above = (sc > ma).astype(int)
            features[f'trend_strength_{period}'] = above.rolling(
                window=period, min_periods=period
            ).mean().values
        
        # ======================================
        # 18. 변동성 (30분봉 기준)
        # ======================================
        for period in [10, 30]:
            returns = sc.pct_change()
            vol_short = returns.rolling(window=period, min_periods=period).std()
            vol_long = returns.rolling(window=period * 3, min_periods=period * 3).std()
            
            features[f'volatility_{period}'] = vol_short.values
            features[f'volatility_ratio_{period}'] = (vol_short / (vol_long + 1e-9)).values
        
        # ======================================
        # 19. DI / ADX (30분봉 기준)
        # ======================================
        pdi, mdi, adx = self._compute_adx_core(
            df_30m['high'], df_30m['low'], df_30m['close'], 
            window=self.adx_window
        )
        features['di_plus_14'] = pdi.shift(1).values
        features['di_minus_14'] = mdi.shift(1).values
        features['adx_14'] = adx.shift(1).values
        
        # ======================================
        # 20. 멀티타임프레임 레짐 (1분봉 기준)
        # ======================================
        print("\n멀티타임프레임 레짐 계산 시작...")
        
        # 1분봉 전달, 30분봉 인덱스 기준으로 매핑
        regime_df = self.compute_multi_timeframe_regime(df_1m, df_30m['bar30_start'])
        
        # ======================================
        # 21. 통합 및 정리
        # ======================================
        # lookback 제거
        features = features.iloc[lookback_bars:].copy().reset_index(drop=True)
        regime_df = regime_df.iloc[lookback_bars:].copy().reset_index(drop=True)

        # 길이 맞추기
        min_len = min(len(features), len(regime_df))
        if len(features) != len(regime_df):
            print(f"⚠️ 길이 불일치: features={len(features)}, regime={len(regime_df)} → {min_len}으로 조정")
            features = features.iloc[:min_len].copy()
            regime_df = regime_df.iloc[:min_len].copy()

        # 안전한 concat
        features = pd.concat([features, regime_df], axis=1)
        
        # regime 컬럼 추가 (모델 학습용)
        features['regime'] = features['regime_final'].astype('int16')

        # ✅ NaN 처리 (pandas 2.0 호환)
        nan_count = features.isna().sum().sum()
        if nan_count > 0:
            print(f"⚠️ NaN: {nan_count}개 → forward fill")
            features = features.ffill().fillna(0)
        
        print(f"\n✓ 피처 생성 완료: {len(features):,}개 바, {len(features.columns)}개 피처")
        
        return features
    
    # ==========================================
    # 라벨 생성
    # ==========================================
    
    @staticmethod
    def create_target_30m(df_30m):
        """
        라벨 생성 (30분봉 기준)
        현재 봉 close 시점에서 다음 봉 예측
        
        Parameters:
        -----------
        df_30m : DataFrame
            30분봉 데이터 (open, close 컬럼 필요)
        
        Returns:
        --------
        Series: 1 (다음 봉 양봉), 0 (다음 봉 음봉)
        """
        # 다음 봉의 open (= 현재 봉 close 시점의 미래 가격)
        next_open = df_30m['open'].shift(-1)
        # 다음 봉의 close
        next_close = df_30m['close'].shift(-1)
        
        # 다음 봉이 양봉이면 1, 음봉이면 0
        return (next_close > next_open).astype(int)
    
    # ==========================================
    # 미래 누수 검증
    # ==========================================
    
    @staticmethod
    def validate_no_future_leak(features, target):
        """
        미래 데이터 누출 검증
        
        Parameters:
        -----------
        features : DataFrame
        target : Series
        
        Returns:
        --------
        bool: True (문제 없음), False (누출 의심)
        """
        issues = []
        exclude_cols = [
            'bar30_start', 'bar30_end', 'm1_index_entry', 'm1_index_label',
            'timestamp', 'regime', 'regime_final', 'is_weekend'
        ]
        for col in features.columns:
            if col in exclude_cols:
                continue
            
            try:
                corr = features[col].corr(target)
                if pd.notna(corr) and abs(corr) > 0.95:
                    issues.append(f"{col}: {corr:.3f}")
            except:
                continue
        
        if issues:
            print("⚠️ 미래 데이터 누출 의심:")
            for it in issues:
                print(f"  - {it}")
            return False
        
        print("✓ 미래 누수 검증 통과")
        return True


# ==========================================
# 테스트
# ==========================================
if __name__ == "__main__":
    from config import Config
    
    Config.create_directories()
    
    # 샘플 데이터 생성 (1분봉)
    print("="*60)
    print("FeatureEngineer 테스트 - 멀티타임프레임 (4h, 1h, 30min, 15min)")
    print("="*60)
    
    print("\n샘플 1분봉 데이터 생성...")
    periods = 10000  # 약 7일
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
    print(f"✓ 1분봉: {len(df_1m):,}개")
    
    # 피처 생성 (30분봉 기준)
    print("\n[1] 피처 생성")
    fe = FeatureEngineer()
    features = fe.create_feature_pool(df_1m, lookback_bars=100)
    
    print(f"\n[2] 30분봉 피처 샘플:")
    display_cols = ['bar30_start', 'bar30_end', 'prev_close', 'ma_20', 'regime']
    if all(c in features.columns for c in display_cols):
        print(features[display_cols].head())
    
    print(f"\n[3] 레짐 분포:")
    if 'regime' in features.columns:
        regime_dist = features['regime'].value_counts().sort_index()
        regime_labels = {1: "UP 🟢", -1: "DOWN 🔴", 0: "FLAT ⚪"}
        for val, count in regime_dist.items():
            label = regime_labels.get(val, f"REGIME-{val}")
            pct = (count / len(features)) * 100
            print(f"  {label:12s}: {count:5d}개 ({pct:5.1f}%)")
    
    print(f"\n[4] 타임프레임별 레짐:")
    regime_tf_cols = [c for c in features.columns if c.startswith('regime_') and c != 'regime_final']
    for col in regime_tf_cols:
        valid = features[col].notna().sum()
        valid_pct = (valid / len(features)) * 100
        print(f"  {col:15s}: {valid_pct:5.1f}% 유효")
    
    print(f"\n[5] 주말 비율:")
    if 'is_weekend' in features.columns:
        weekend_count = features['is_weekend'].sum()
        weekend_pct = (weekend_count / len(features)) * 100
        print(f"  주말: {weekend_count:5d}개 ({weekend_pct:5.1f}%)")
    
    # 라벨 생성
    print("\n[6] 타겟 생성")
    tf_manager = TimeframeManager()
    df_30m = tf_manager.aggregate_1m_to_30m(df_1m)
    target = fe.create_target_30m(df_30m)

    # lookback 맞추기
    target_aligned = target.iloc[100:].reset_index(drop=True)

    # 길이 맞추기
    min_len = min(len(features), len(target_aligned))
    features_valid = features.iloc[:min_len].copy()
    target_valid = target_aligned.iloc[:min_len].copy()

    # 유효 데이터만 (NaN 제거)
    valid_mask = target_valid.notna()
    features_final = features_valid[valid_mask].reset_index(drop=True)
    target_final = target_valid[valid_mask].reset_index(drop=True)
    
    print(f"\n[7] 최종 데이터:")
    print(f"  Features: {len(features_final):,}개 바 × {len(features_final.columns)}개 컬럼")
    print(f"  Target:   {len(target_final):,}개")
    if len(target_final) > 0:
        target_dist = target_final.value_counts().to_dict()
        print(f"  Target 분포: {target_dist}")
        if 0 in target_dist and 1 in target_dist:
            balance = min(target_dist[0], target_dist[1]) / max(target_dist[0], target_dist[1])
            print(f"  클래스 균형: {balance:.2%}")

    # 미래 누수 검증
    print("\n[8] 미래 누수 검증:")
    if len(features_final) > 0 and len(target_final) > 0:
        fe.validate_no_future_leak(features_final, target_final)
    
    # 피처 요약
    print("\n[9] 피처 카테고리:")
    feature_categories = {
        '타임스탬프': ['bar30_start', 'bar30_end', 'timestamp', 'm1_index_entry', 'm1_index_label'],
        '과거 캔들': ['prev_open', 'prev_high', 'prev_low', 'prev_close', 'prev_volume'],
        '수익률': [c for c in features_final.columns if c.startswith('return_')],
        '거래량': [c for c in features_final.columns if 'volume' in c],
        '이동평균': [c for c in features_final.columns if c.startswith('ma_') or c.startswith('ema_')],
        '볼린저밴드': [c for c in features_final.columns if c.startswith('bb_')],
        'RSI': [c for c in features_final.columns if c.startswith('rsi_')],
        'MACD': [c for c in features_final.columns if 'macd' in c],
        'ATR': [c for c in features_final.columns if c.startswith('atr_')],
        '캔들패턴': [c for c in features_final.columns if 'candle' in c or 'body' in c or 'shadow' in c],
        '시간피처': [c for c in features_final.columns if any(x in c for x in ['hour', 'minute', 'day', 'dow'])],
        '추세/변동성': [c for c in features_final.columns if 'trend' in c or 'volatility' in c],
        'ADX/DI': [c for c in features_final.columns if 'adx' in c or 'di_' in c],
        '레짐': [c for c in features_final.columns if 'regime' in c or 'is_weekend' in c],
    }
    
    for category, cols in feature_categories.items():
        matching = [c for c in cols if c in features_final.columns]
        if matching:
            print(f"  {category:15s}: {len(matching):3d}개")
    
    print("\n" + "="*60)
    print("✓ 테스트 완료")
    print("="*60)
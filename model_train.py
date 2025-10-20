# model_train.py
# 레짐별 스위칭 모델 (UP/DOWN/FLAT 각각 독립 학습)
# - 상위봉(기본 15분) ADX 레짐을 'regime' 컬럼으로 제공
# - 각 레짐마다 별도 LGBM 앙상블 + 캘리브레이터 학습
# - Isotonic/Platt 캘리브레이션 선택 가능

import os
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    _LGBM_ERR = None
except Exception as e:
    lgb, _LGBM_ERR = None, e

import numpy as np
import pandas as pd

from datetime import datetime, timezone

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
)
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _safe_prep(df):
    return df.fillna(method='ffill').fillna(0).replace([np.inf, -np.inf], 0)

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)


# ===============================
# Feature Engineering
# ===============================
class FeatureEngineer:
    """피처 엔지니어링 (미래 누출 방지)"""

    @staticmethod
    def _compute_adx_core(high, low, close, w: int = 14):
        up_move   = high.diff()
        down_move = (-low.diff())

        plus_dm  = np.where((up_move > down_move) & (up_move > 0),  up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr1 = (high - low)
        tr2 = (high - close.shift()).abs()
        tr3 = (low  - close.shift()).abs()
        tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        alpha = 1.0 / w
        tr_s       = tr.ewm(alpha=alpha, adjust=False, min_periods=w).mean()
        plus_dm_s  = pd.Series(plus_dm,  index=high.index).ewm(alpha=alpha, adjust=False, min_periods=w).mean()
        minus_dm_s = pd.Series(minus_dm, index=high.index).ewm(alpha=alpha, adjust=False, min_periods=w).mean()

        plus_di  = 100.0 * (plus_dm_s / (tr_s + 1e-9))
        minus_di = 100.0 * (minus_dm_s / (tr_s + 1e-9))
        dx       = 100.0 * (plus_di.subtract(minus_di).abs() / (plus_di + minus_di + 1e-9))
        adx      = dx.ewm(alpha=alpha, adjust=False, min_periods=w).mean()
        return plus_di, minus_di, adx

    @staticmethod
    def _compute_adx(df: pd.DataFrame, w: int = 14):
        high  = df['high'].astype(float)
        low   = df['low'].astype(float)
        close = df['close'].astype(float)
        return FeatureEngineer._compute_adx_core(high, low, close, w=w)

    @staticmethod
    def _htf_regime_features(df, tf='15min', adx_thr=20, w=14):
        """상위봉 레짐 계산"""
        if 'timestamp' in df.columns:
            idx = pd.to_datetime(df['timestamp'], utc=True)
        else:
            idx = pd.to_datetime(df.index, utc=True)
        src = df.set_index(idx).sort_index()
        ohlc = src[['open','high','low','close','volume']]

        htf = pd.DataFrame({
            'open':   ohlc['open'].resample(tf).first(),
            'high':   ohlc['high'].resample(tf).max(),
            'low':    ohlc['low'].resample(tf).min(),
            'close':  ohlc['close'].resample(tf).last(),
            'volume': ohlc['volume'].resample(tf).sum(),
        }).dropna()

        pdi, mdi, adx = FeatureEngineer._compute_adx_core(htf['high'], htf['low'], htf['close'], w=w)
        up   = (adx > adx_thr) & (pdi > mdi)
        down = (adx > adx_thr) & (mdi > pdi)
        regime = np.select([up, down], [1, -1], default=0).astype(int)

        out = pd.DataFrame({
            'htf_pdi_14': pdi,
            'htf_mdi_14': mdi,
            'htf_adx_14': adx,
            'htf_regime': regime,
        }, index=htf.index)

        out_1m = out.reindex(src.index, method='ffill')
        return out_1m.reset_index(drop=True)

    @staticmethod
    def create_feature_pool(df, lookback_window=200):
        features = pd.DataFrame(index=df.index)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            features['timestamp'] = df['timestamp']

        # 기본 과거 캔들
        features['prev_open']   = df['open'].shift(1)
        features['prev_high']   = df['high'].shift(1)
        features['prev_low']    = df['low'].shift(1)
        features['prev_close']  = df['close'].shift(1)
        features['prev_volume'] = df['volume'].shift(1)

        # 수익률/체결량 변화율
        for period in [1, 2, 3, 5, 10, 15, 30, 60]:
            features[f'return_{period}']        = df['close'].shift(1) / df['close'].shift(period+1) - 1
            features[f'volume_change_{period}'] = df['volume'].shift(1) / df['volume'].shift(period+1) - 1

        # 이동평균
        for period in [5, 10, 20, 50, 100, 200]:
            ma = df['close'].shift(1).rolling(window=period, min_periods=period).mean()
            features[f'ma_{period}'] = ma
            features[f'price_to_ma_{period}'] = df['close'].shift(1) / (ma + 1e-9) - 1
            features[f'ma_{period}_slope'] = ma.diff(5) / (ma.shift(5) + 1e-9)

        # EMA
        for period in [12, 26, 50]:
            ema = df['close'].shift(1).ewm(span=period, adjust=False, min_periods=period).mean()
            features[f'ema_{period}'] = ema
            features[f'price_to_ema_{period}'] = df['close'].shift(1) / (ema + 1e-9) - 1

        # 볼린저
        for period in [20, 50]:
            sc  = df['close'].shift(1)
            ma  = sc.rolling(window=period, min_periods=period).mean()
            std = sc.rolling(window=period, min_periods=period).std()
            upper = ma + 2 * std
            lower = ma - 2 * std
            features[f'bb_upper_{period}'] = upper
            features[f'bb_lower_{period}'] = lower
            features[f'bb_width_{period}'] = 4 * std / (ma + 1e-9)
            features[f'bb_position_{period}'] = (sc - lower) / ((upper - lower) + 1e-9)

        # RSI
        for period in [14, 28]:
            sc = df['close'].shift(1)
            delta = sc.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
            rs = gain / (loss + 1e-9)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # MACD
        sc = df['close'].shift(1)
        ema12 = sc.ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = sc.ewm(span=26, adjust=False, min_periods=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = macd - signal

        # Stochastic
        for period in [14]:
            sh = df['high'].shift(1); sl = df['low'].shift(1); sc = df['close'].shift(1)
            low_min  = sl.rolling(window=period, min_periods=period).min()
            high_max = sh.rolling(window=period, min_periods=period).max()
            features[f'stoch_{period}'] = (sc - low_min) / ((high_max - low_min) + 1e-9) * 100

        # ATR
        for period in [14, 28]:
            sh = df['high'].shift(1); sl = df['low'].shift(1); sc = df['close'].shift(1)
            tr = pd.concat([
                sh - sl,
                (sh - sc.shift(1)).abs(),
                (sl - sc.shift(1)).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(window=period, min_periods=period).mean()
            features[f'atr_{period}'] = atr
            features[f'atr_ratio_{period}'] = atr / (sc + 1e-9)

        # 거래량
        sv = df['volume'].shift(1)
        features['volume_sma_10'] = sv.rolling(window=10, min_periods=10).mean()
        features['volume_sma_50'] = sv.rolling(window=50, min_periods=50).mean()
        features['volume_ratio'] = sv / (features['volume_sma_10'] + 1e-9)
        features['volume_trend'] = features['volume_sma_10'] / (features['volume_sma_50'] + 1e-9)

        # OBV
        price_diff = df['close'].diff().shift(1)
        obv = (np.sign(price_diff) * df['volume'].shift(1)).cumsum()
        features['obv'] = obv
        features['obv_ema'] = obv.ewm(span=20, adjust=False, min_periods=20).mean()
        features['obv_signal'] = obv / (features['obv_ema'] + 1e-9) - 1

        # 캔들 패턴
        po = df['open'].shift(1); pc = df['close'].shift(1); ph = df['high'].shift(1); pl = df['low'].shift(1)
        features['body_size'] = (pc - po).abs() / (po + 1e-9)
        features['upper_shadow'] = (ph - pd.concat([po, pc], axis=1).max(axis=1)) / (po + 1e-9)
        features['lower_shadow'] = (pd.concat([po, pc], axis=1).min(axis=1) - pl) / (po + 1e-9)
        features['body_position'] = (pc - po) / ((ph - pl) + 1e-9)

        for i in range(1, 4):
            features[f'candle_direction_{i}'] = np.sign(df['close'].shift(i) - df['open'].shift(i))
            features[f'candle_size_{i}'] = (df['high'].shift(i) - df['low'].shift(i)) / (df['close'].shift(i) + 1e-9)

        # 시간
        if 'timestamp' in df.columns:
            dt = pd.to_datetime(df['timestamp'], utc=True)
            features['hour'] = dt.dt.hour
            features['minute'] = dt.dt.minute
            features['day_of_week'] = dt.dt.dayofweek
            features['day_of_month'] = dt.dt.day
            features['hour_sin'] = np.sin(2*np.pi*features['hour']/24.0)
            features['hour_cos'] = np.cos(2*np.pi*features['hour']/24.0)
            features['dow_sin'] = np.sin(2*np.pi*features['day_of_week']/7.0)
            features['dow_cos'] = np.cos(2*np.pi*features['day_of_week']/7.0)

        # 마이크로구조
        for period in [3, 5, 10]:
            sh = df['high'].shift(1); sl = df['low'].shift(1); sc = df['close'].shift(1)
            features[f'high_low_ratio_{period}'] = (sh / (sl + 1e-9) - 1).rolling(period, min_periods=period).mean()
            features[f'close_position_{period}'] = (((sc - sl) / ((sh - sl) + 1e-9))
                                                    .rolling(period, min_periods=period).mean())

        # 추세 강도
        for period in [10, 20, 50]:
            sc = df['close'].shift(1)
            ma = sc.rolling(window=period, min_periods=period).mean()
            above = (sc > ma).astype(int)
            features[f'trend_strength_{period}'] = above.rolling(window=period, min_periods=period).mean()

        # 변동성
        for period in [10, 30]:
            sr = df['close'].shift(1).pct_change()
            features[f'volatility_{period}'] = sr.rolling(window=period, min_periods=period).std()
            features[f'volatility_ratio_{period}'] = (
                features[f'volatility_{period}'] /
                (sr.rolling(window=period*3, min_periods=period*3).std() + 1e-9)
            )

        # lookback 제거
        features = features.iloc[lookback_window:]

        # DI/ADX
        pdi, mdi, adx = FeatureEngineer._compute_adx(df, w=14)
        pdi  = pdi.shift(1).reindex(features.index)
        mdi  = mdi.shift(1).reindex(features.index)
        adx  = adx.shift(1).reindex(features.index)

        features['di_plus_14']  = pdi
        features['di_minus_14'] = mdi
        features['adx_14']      = adx

        # 상위봉 레짐
        try:
            from config import Config
            tf = getattr(Config, 'REGIME_TF', '15min')
            adx_thr = getattr(Config, 'REGIME_ADX_THR', 20)
        except Exception:
            tf = '15min'
            adx_thr = 20

        htf = FeatureEngineer._htf_regime_features(df.copy(), tf=tf, adx_thr=adx_thr, w=14)
        htf = htf.iloc[lookback_window:].reset_index(drop=True)
        features = features.reset_index(drop=True)
        features = pd.concat([features, htf], axis=1)

        features['regime'] = features['htf_regime'].astype('int16')

        return features

    @staticmethod
    def create_target(df, window=10):
        current = df['close']
        future = df['close'].shift(-window)
        return (future > current).astype(int)

    @staticmethod
    def validate_no_future_leak(features, target):
        issues = []
        for col in features.columns:
            if col == 'timestamp':
                continue
            corr = features[col].corr(target)
            if pd.notna(corr) and abs(corr) > 0.95:
                issues.append(f"Suspicious correlation in {col}: {corr:.3f}")
        if issues:
            print("⚠️ 미래 데이터 누출 의심:")
            for it in issues:
                print("  -", it)
            return False
        return True


# ===============================
# Model Trainer
# ===============================
class ModelTrainer:
    """레짐별 독립 LGBM 앙상블 + 확률 캘리브레이션"""
    def __init__(self, config):
        self.config = config
        if lgb is None:
            raise ImportError(f"lightgbm import failed: {_LGBM_ERR}")

        # 레짐별 번들
        self.bundles = {'UP': None, 'DOWN': None, 'FLAT': None}

        # Legacy 호환
        self.models = []
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.selected_features = None
        self.calibrator = None
        self.calib_method = 'isotonic'

    def _regime_to_name(self, regime_val: int) -> str:
        if regime_val == 1:
            return 'UP'
        elif regime_val == -1:
            return 'DOWN'
        else:
            return 'FLAT'

    def _name_to_regime(self, name: str) -> int:
        if name == 'UP':
            return 1
        elif name == 'DOWN':
            return -1
        else:
            return 0

    # --- 결정 함수 (real_trade 호환) ---
    def decide_from_proba_simple(self, p_up: float) -> int:
        return 1 if p_up >= 0.5 else 0

    def decide_from_proba_regime(self, p_up, regime):
        """레짐 기반 진입 결정"""
        if regime == 1:
            return 1
        elif regime == -1:
            return 0
        else:
            return 1 if p_up >= 0.5 else 0

    # --- 피처 선택 ---
    def feature_selection_regime(self, X, y, regime_col='regime', top_k=50):
        """레짐별 피처 선택"""
        if regime_col not in X.columns:
            return self.feature_selection_temporal(X, y, top_k)

        # 인덱스 정렬 및 리셋 (핵심 수정)
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        # 길이 맞추기
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len]
        y = y.iloc[:min_len]

        all_selected = set()

        for rval in [1, -1, 0]:
            rname = self._regime_to_name(rval)
            mask = (X[regime_col] == rval).values  # .values로 numpy array 변환
            Xr = X[mask].copy()
            yr = y[mask].copy()

            if len(Xr) < 100:
                print(f"  레짐 {rname}: 샘플 부족 ({len(Xr)}개) → 스킵")
                continue

            print(f"\n레짐 {rname} 피처 선택 (n={len(Xr)})...")

            # 다시 인덱스 리셋
            Xr = Xr.reset_index(drop=True)
            yr = yr.reset_index(drop=True)

            train_size = int(len(Xr) * 0.7)
            X_tr, y_tr = Xr.iloc[:train_size], yr.iloc[:train_size]
            X_va, y_va = Xr.iloc[train_size:], yr.iloc[train_size:]

            cols = [c for c in X_tr.columns if c not in ['timestamp', 'regime', 'htf_regime']]
            X_tr = _safe_prep(X_tr[cols])
            X_va = _safe_prep(X_va[cols])

            n_tr = min(len(X_tr), len(y_tr))
            n_va = min(len(X_va), len(y_va))
            X_tr, y_tr = X_tr.iloc[:n_tr], y_tr.iloc[:n_tr]
            X_va, y_va = X_va.iloc[:n_va], y_va.iloc[:n_va]

            dtr = lgb.Dataset(X_tr, y_tr)
            dva = lgb.Dataset(X_va, y_va, reference=dtr)

            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'learning_rate': 0.03,
                'num_leaves': 31,
                'max_depth': 7,
                'min_data_in_leaf': 100,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.7,
                'bagging_freq': 1,
                'lambda_l2': 10.0,
                'verbose': -1
            }

            model = lgb.train(
                params,
                dtr,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )

            imp = pd.DataFrame({
                'feature': cols,
                'importance': model.feature_importance(importance_type='gain')
            }).sort_values('importance', ascending=False)

            top_feats = set(imp.head(top_k)['feature'].tolist())
            all_selected.update(top_feats)

            print(f"  → 상위 {len(top_feats)}개 피처 선택")

        self.selected_features = list(all_selected)
        print(f"\n전체 선택 피처: {len(self.selected_features)}개")

        return self.selected_features

    def feature_selection_temporal(self, X, y, top_k=50):
        """전체 데이터 피처 선택"""
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        train_size = int(len(X) * 0.7)
        X_tr, y_tr = X.iloc[:train_size], y.iloc[:train_size]
        X_va, y_va = X.iloc[train_size:], y.iloc[train_size:]

        cols = [c for c in X.columns if c not in ['timestamp', 'regime', 'htf_regime']]
        X_tr = _safe_prep(X_tr[cols])
        X_va = _safe_prep(X_va[cols])

        n_tr = min(len(X_tr), len(y_tr))
        n_va = min(len(X_va), len(y_va))
        X_tr, y_tr = X_tr.iloc[:n_tr], y_tr.iloc[:n_tr]
        X_va, y_va = X_va.iloc[:n_va], y_va.iloc[:n_va]

        dtr = lgb.Dataset(X_tr, y_tr)
        dva = lgb.Dataset(X_va, y_va, reference=dtr)

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'num_leaves': 31,
            'max_depth': 7,
            'min_data_in_leaf': 200,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 1,
            'lambda_l2': 10.0,
        }

        model = lgb.train(
            params,
            dtr,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        imp = pd.DataFrame({
            'feature': cols,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        self.feature_importance = imp
        self.selected_features = imp.head(top_k)['feature'].tolist()

        print("\n선택된 상위 피처(10):")
        for _, row in imp.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.2f}")

        return self.selected_features

    # --- 레짐별 학습 ---
    def train_ensemble_regime(self, X, y, regime_col='regime', test_size=0.2):
        """레짐별 독립 학습"""
        if regime_col not in X.columns:
            return self.train_ensemble_temporal(X, y, test_size)

        # 인덱스 정렬 및 길이 맞추기
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len]
        y = y.iloc[:min_len]

        if not self.selected_features:
            self.selected_features = [c for c in X.columns if c not in ['timestamp', 'regime', 'htf_regime']][:50]

        all_metrics = {}

        for rval in [1, -1, 0]:
            rname = self._regime_to_name(rval)
            print(f"\n{'='*60}")
            print(f"레짐 {rname} ({rval}) 학습 시작")
            print(f"{'='*60}")

            mask = (X[regime_col] == rval).values  # .values로 numpy array 변환
            Xr = X[mask].copy()
            yr = y[mask].copy()

            if len(Xr) < 200:
                print(f"  샘플 부족 ({len(Xr)}개) → 스킵")
                self.bundles[rname] = None
                continue

            # 인덱스 리셋
            Xr = Xr.reset_index(drop=True)
            yr = yr.reset_index(drop=True)

            Xs = Xr[self.selected_features]
            Xs = _safe_prep(Xs)

            total = len(Xs)
            val_size = int(total * test_size)
            train_size = total - 2 * val_size

            if train_size <= 50:
                print(f"  train 샘플 부족 → 스킵")
                self.bundles[rname] = None
                continue

            X_tr, y_tr = Xs.iloc[:train_size], yr.iloc[:train_size]
            X_va, y_va = Xs.iloc[train_size:train_size+val_size], yr.iloc[train_size:train_size+val_size]
            X_te, y_te = Xs.iloc[train_size+val_size:], yr.iloc[train_size+val_size:]

            print(f"  데이터 분할: train={len(X_tr)}, val={len(X_va)}, test={len(X_te)}")

            X_tr, y_tr = X_tr.reset_index(drop=True), y_tr.reset_index(drop=True)
            X_va, y_va = X_va.reset_index(drop=True), y_va.reset_index(drop=True)
            X_te, y_te = X_te.reset_index(drop=True), y_te.reset_index(drop=True)

            n_tr = min(len(X_tr), len(y_tr))
            n_va = min(len(X_va), len(y_va))
            n_te = min(len(X_te), len(y_te))
            X_tr, y_tr = X_tr.iloc[:n_tr], y_tr.iloc[:n_tr]
            X_va, y_va = X_va.iloc[:n_va], y_va.iloc[:n_va]
            X_te, y_te = X_te.iloc[:n_te], y_te.iloc[:n_te]

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)
            X_te_s = scaler.transform(X_te)

            pos, neg = int((y_tr == 1).sum()), int((y_tr == 0).sum())
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'learning_rate': 0.03,
                'num_leaves': 31,
                'max_depth': 7,
                'min_data_in_leaf': 100,
                'feature_fraction': 0.7,
                'bagging_fraction': 0.7,
                'bagging_freq': 1,
                'lambda_l2': 10.0,
                'verbose': -1
            }
            if pos > 0 and neg > 0:
                params['scale_pos_weight'] = neg / max(1, pos)

            n_models = int(getattr(self.config, 'ENSEMBLE_MODELS', 3))
            n_models = max(1, min(5, n_models))
            models = []

            for i in range(n_models):
                p = params.copy()
                p['random_state'] = 42 + i
                p['num_leaves'] = min(63, 31 + i * 8)
                p['feature_fraction'] = max(0.5, 0.7 - i * 0.05)

                start = int(i * len(X_tr_s) * 0.1)
                dtr = lgb.Dataset(X_tr_s[start:], y_tr.iloc[start:])
                dva = lgb.Dataset(X_va_s, y_va, reference=dtr)

                model = lgb.train(
                    p,
                    dtr,
                    valid_sets=[dva],
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
                models.append(model)
                print(f"  모델 {i+1}/{n_models} 학습 완료")

            # 캘리브레이션
            val_pred_raw = self._predict_proba_array_from_models(X_va_s, models)
            method = getattr(self.config, 'CALIBRATION_METHOD', 'isotonic')
            if method not in ('isotonic', 'platt'):
                method = 'isotonic'

            if method == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(val_pred_raw, y_va.values)
                calib_method = 'isotonic'
                print(f"  ✓ 캘리브레이션: IsotonicRegression")
            else:
                lr = LogisticRegression(max_iter=1000)
                lr.fit(val_pred_raw.reshape(-1, 1), y_va.values)
                calibrator = lr
                calib_method = 'platt'
                print(f"  ✓ 캘리브레이션: Platt")

            # 번들 저장
            self.bundles[rname] = {
                'models': models,
                'scaler': scaler,
                'calibrator': calibrator,
                'calib_method': calib_method,
                'selected_features': self.selected_features,
                'regime_value': rval
            }

            # 진단 저장
            self._save_calibration_diagnostics_regime(
                X_va_s, y_va, val_pred_raw, calibrator, calib_method, rname
            )

            # 성능 평가
            tr_proba = self._predict_proba_from_bundle(X_tr, rname)
            va_proba = self._predict_proba_from_bundle(X_va, rname)
            te_proba = self._predict_proba_from_bundle(X_te, rname)

            metrics = {
                'train': self._evaluate_block(y_tr, tr_proba),
                'validation': self._evaluate_block(y_va, va_proba),
                'test': self._evaluate_block(y_te, te_proba),
            }
            all_metrics[rname] = metrics

            print(f"\n레짐 {rname} 결과:")
            for split, m in metrics.items():
                print(f"  {split}: acc={m['accuracy']:.4f}, win={m['win_rate']:.4f}")

        # Legacy 호환: FLAT 번들을 기본으로
        if self.bundles['FLAT'] is not None:
            self.models = self.bundles['FLAT']['models']
            self.scaler = self.bundles['FLAT']['scaler']
            self.calibrator = self.bundles['FLAT']['calibrator']
            self.calib_method = self.bundles['FLAT']['calib_method']
        elif self.bundles['UP'] is not None:
            self.models = self.bundles['UP']['models']
            self.scaler = self.bundles['UP']['scaler']
            self.calibrator = self.bundles['UP']['calibrator']
            self.calib_method = self.bundles['UP']['calib_method']
        elif self.bundles['DOWN'] is not None:
            self.models = self.bundles['DOWN']['models']
            self.scaler = self.bundles['DOWN']['scaler']
            self.calibrator = self.bundles['DOWN']['calibrator']
            self.calib_method = self.bundles['DOWN']['calib_method']

        return all_metrics

    def train_ensemble_temporal(self, X, y, test_size=0.2):
        """단일 모델 학습 (fallback)"""
        self.models = []
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        if 'timestamp' in X.columns:
            X = X.drop('timestamp', axis=1)
        if 'regime' in X.columns:
            X = X.drop('regime', axis=1)
        if 'htf_regime' in X.columns:
            X = X.drop('htf_regime', axis=1)

        if not self.selected_features:
            self.selected_features = list(X.columns)[:50]

        Xs = _safe_prep(X[self.selected_features])

        pos, neg = int((y == 1).sum()), int((y == 0).sum())
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'num_leaves': 31,
            'max_depth': 7,
            'min_data_in_leaf': 200,
            'feature_fraction': 0.7,
            'bagging_fraction': 0.7,
            'bagging_freq': 1,
            'lambda_l2': 10.0,
        }
        if pos > 0 and neg > 0:
            params['scale_pos_weight'] = neg / max(1, pos)

        total = len(Xs)
        val_size = int(total * test_size)
        train_size = total - 2 * val_size

        X_tr, y_tr = Xs.iloc[:train_size], y.iloc[:train_size]
        X_va, y_va = Xs.iloc[train_size:train_size+val_size], y.iloc[train_size:train_size+val_size]
        X_te, y_te = Xs.iloc[train_size+val_size:], y.iloc[train_size+val_size:]

        print(f"\n데이터 분할: train={len(X_tr)}, val={len(X_va)}, test={len(X_te)}")

        X_tr, y_tr = X_tr.reset_index(drop=True), y_tr.reset_index(drop=True)
        X_va, y_va = X_va.reset_index(drop=True), y_va.reset_index(drop=True)
        X_te, y_te = X_te.reset_index(drop=True), y_te.reset_index(drop=True)

        n_tr = min(len(X_tr), len(y_tr))
        n_va = min(len(X_va), len(y_va))
        n_te = min(len(X_te), len(y_te))
        X_tr, y_tr = X_tr.iloc[:n_tr], y_tr.iloc[:n_tr]
        X_va, y_va = X_va.iloc[:n_va], y_va.iloc[:n_va]
        X_te, y_te = X_te.iloc[:n_te], y_te.iloc[:n_te]

        self.scaler = StandardScaler()
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_va_s = self.scaler.transform(X_va)
        X_te_s = self.scaler.transform(X_te)

        n_models = int(getattr(self.config, 'ENSEMBLE_MODELS', 3))
        for i in range(n_models):
            p = params.copy()
            p['random_state'] = 42 + i
            p['num_leaves'] = min(63, 31 + i * 8)
            p['feature_fraction'] = max(0.5, 0.7 - i * 0.05)

            start = int(i * len(X_tr_s) * 0.1)
            dtr = lgb.Dataset(X_tr_s[start:], y_tr.iloc[start:])
            dva = lgb.Dataset(X_va_s, y_va, reference=dtr)

            model = lgb.train(
                p,
                dtr,
                valid_sets=[dva],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            self.models.append(model)
            print(f"  모델 {i+1}/{n_models} 학습 완료")

        val_pred_raw = self._predict_proba_array(X_va_s)
        method = getattr(self.config, 'CALIBRATION_METHOD', 'isotonic')
        if method not in ('isotonic', 'platt'):
            method = 'isotonic'

        if method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(val_pred_raw, y_va.values)
            self.calib_method = 'isotonic'
            print("✓ 캘리브레이션: IsotonicRegression")
        else:
            lr = LogisticRegression(max_iter=1000)
            lr.fit(val_pred_raw.reshape(-1, 1), y_va.values)
            self.calibrator = lr
            self.calib_method = 'platt'
            print("✓ 캘리브레이션: Platt")

        self._save_calibration_diagnostics(X_va_s, y_va, val_pred_raw)

        tr_proba = self.predict_proba_df(pd.DataFrame(X_tr, columns=X_tr.columns))
        va_proba = self.predict_proba_df(pd.DataFrame(X_va, columns=X_va.columns))
        te_proba = self.predict_proba_df(pd.DataFrame(X_te, columns=X_te.columns))

        metrics = {
            'train': self._evaluate_block(y_tr, tr_proba),
            'validation': self._evaluate_block(y_va, va_proba),
            'test': self._evaluate_block(y_te, te_proba),
        }

        return metrics

    # --- Predict helpers ---
    def _predict_proba_array_from_models(self, X_nd, models):
        """특정 모델 리스트로 예측"""
        preds = []
        for m in models:
            preds.append(m.predict(X_nd, num_iteration=m.best_iteration))
        preds = np.vstack(preds)
        n = preds.shape[0]
        if n == 1:
            w = np.array([1.0])
        else:
            w = np.linspace(1.0, 1.5, n)
            w = w / w.sum()
        return np.average(preds, axis=0, weights=w)

    def _predict_proba_array(self, X_nd):
        """self.models로 예측"""
        return self._predict_proba_array_from_models(X_nd, self.models)

    def _apply_calibration_with(self, p_raw, calibrator, method):
        """특정 캘리브레이터 적용"""
        if calibrator is None:
            return p_raw
        if method == 'isotonic':
            return calibrator.predict(p_raw)
        return calibrator.predict_proba(p_raw.reshape(-1, 1))[:, 1]

    def _apply_calibration(self, p_raw):
        """self.calibrator 적용"""
        return self._apply_calibration_with(p_raw, self.calibrator, self.calib_method)

    def _predict_proba_from_bundle(self, X_df, regime_name):
        """특정 레짐 번들로 예측"""
        bundle = self.bundles.get(regime_name)
        if bundle is None:
            return np.full(len(X_df), 0.5)

        X_tmp = X_df[bundle['selected_features']].copy()
        X_tmp = _safe_prep(X_tmp)
        X_s = bundle['scaler'].transform(X_tmp)

        p_raw = self._predict_proba_array_from_models(X_s, bundle['models'])
        p = self._apply_calibration_with(p_raw, bundle['calibrator'], bundle['calib_method'])
        return np.clip(p, 0.02, 0.98)

    def predict_proba(self, X, regime=None):
        """레짐별 예측"""
        if isinstance(X, pd.DataFrame):
            return self.predict_proba_df(X, regime)

        if regime is not None:
            rname = self._regime_to_name(regime)
            bundle = self.bundles.get(rname)
            if bundle is not None:
                X_s = bundle['scaler'].transform(X)
                p_raw = self._predict_proba_array_from_models(X_s, bundle['models'])
                p = self._apply_calibration_with(p_raw, bundle['calibrator'], bundle['calib_method'])
                return np.clip(p, 0.02, 0.98)

        p_raw = self._predict_proba_array(X)
        p = self._apply_calibration(p_raw)
        return np.clip(p, 0.02, 0.98)

    def predict_proba_df(self, X_df, regime=None):
        """DataFrame으로 예측"""
        X_tmp = X_df.copy()
        if 'timestamp' in X_tmp.columns:
            X_tmp = X_tmp.drop('timestamp', axis=1)

        if regime is None and 'regime' in X_tmp.columns:
            regime = int(X_tmp['regime'].iloc[-1])

        if 'regime' in X_tmp.columns:
            X_tmp = X_tmp.drop('regime', axis=1)
        if 'htf_regime' in X_tmp.columns:
            X_tmp = X_tmp.drop('htf_regime', axis=1)

        if regime is not None:
            rname = self._regime_to_name(regime)
            bundle = self.bundles.get(rname)
            if bundle is not None:
                X_tmp = X_tmp[bundle['selected_features']]
                X_tmp = _safe_prep(X_tmp)
                X_s = bundle['scaler'].transform(X_tmp)
                p_raw = self._predict_proba_array_from_models(X_s, bundle['models'])
                p = self._apply_calibration_with(p_raw, bundle['calibrator'], bundle['calib_method'])
                return np.clip(p, 0.02, 0.98)

        if not self.selected_features:
            self.selected_features = list(X_tmp.columns)[:50]
        X_tmp = X_tmp[self.selected_features]
        X_tmp = _safe_prep(X_tmp)
        X_s = self.scaler.transform(X_tmp)
        p_raw = self._predict_proba_array(X_s)
        p = self._apply_calibration(p_raw)
        return np.clip(p, 0.02, 0.98)

    def predict(self, X, threshold=0.5, regime=None):
        p = self.predict_proba(X, regime)
        return (p > threshold).astype(int)

    # --- Evaluation ---
    def _evaluate_block(self, y_true, proba, th=0.5):
        y_pred = (proba > th).astype(int)
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'win_rate': float(np.mean(y_pred == y_true))
        }

    def _save_calibration_diagnostics(self, X_val_s, y_val, val_pred_raw):
        """단일 모델 진단"""
        try:
            val_pred_cal = self._apply_calibration(val_pred_raw)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            outdir = os.path.join(self.config.MODEL_DIR, "diag")
            _ensure_dir(outdir)

            plt.figure(figsize=(9, 4))
            plt.subplot(1, 2, 1)
            plt.hist(val_pred_raw, bins=50)
            plt.title("P(UP) raw (val)")
            plt.xlabel("p_up"); plt.ylabel("freq")

            plt.subplot(1, 2, 2)
            plt.hist(val_pred_cal, bins=50)
            plt.title(f"P(UP) calibrated ({self.calib_method})")
            plt.xlabel("p_up"); plt.ylabel("freq")

            hist_path = os.path.join(outdir, f"proba_hist_{ts}.png")
            plt.tight_layout(); plt.savefig(hist_path); plt.close()

            frac_raw, mean_raw = calibration_curve(y_val.values, val_pred_raw, n_bins=20, strategy="uniform")
            frac_cal, mean_cal = calibration_curve(y_val.values, val_pred_cal, n_bins=20, strategy="uniform")

            plt.figure(figsize=(5, 5))
            plt.plot([0, 1], [0, 1], linestyle="--", label="ideal")
            plt.plot(mean_raw, frac_raw, marker="o", label="raw")
            plt.plot(mean_cal, frac_cal, marker="o", label=f"calibrated ({self.calib_method})")
            plt.xlabel("predicted probability"); plt.ylabel("observed frequency")
            plt.title("Reliability (validation)"); plt.legend()
            rel_path = os.path.join(outdir, f"reliability_{ts}.png")
            plt.tight_layout(); plt.savefig(rel_path); plt.close()

            b_raw = brier_score_loss(y_val.values, val_pred_raw)
            b_cal = brier_score_loss(y_val.values, val_pred_cal)
            print(f"진단 저장: {hist_path}, {rel_path}")
            print(f"Brier raw={b_raw:.4f} → calibrated={b_cal:.4f}")
        except Exception as e:
            print(f"(진단 저장 생략) {e}")

    def _save_calibration_diagnostics_regime(self, X_val_s, y_val, val_pred_raw, calibrator, method, regime_name):
        """레짐별 진단"""
        try:
            val_pred_cal = self._apply_calibration_with(val_pred_raw, calibrator, method)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            outdir = os.path.join(self.config.MODEL_DIR, "diag", regime_name)
            _ensure_dir(outdir)

            plt.figure(figsize=(9, 4))
            plt.subplot(1, 2, 1)
            plt.hist(val_pred_raw, bins=50)
            plt.title(f"P(UP) raw - {regime_name}")
            plt.xlabel("p_up"); plt.ylabel("freq")

            plt.subplot(1, 2, 2)
            plt.hist(val_pred_cal, bins=50)
            plt.title(f"P(UP) calibrated - {regime_name}")
            plt.xlabel("p_up"); plt.ylabel("freq")

            hist_path = os.path.join(outdir, f"proba_hist_{ts}.png")
            plt.tight_layout(); plt.savefig(hist_path); plt.close()

            frac_raw, mean_raw = calibration_curve(y_val.values, val_pred_raw, n_bins=20, strategy="uniform")
            frac_cal, mean_cal = calibration_curve(y_val.values, val_pred_cal, n_bins=20, strategy="uniform")

            plt.figure(figsize=(5, 5))
            plt.plot([0, 1], [0, 1], linestyle="--", label="ideal")
            plt.plot(mean_raw, frac_raw, marker="o", label="raw")
            plt.plot(mean_cal, frac_cal, marker="o", label=f"calibrated")
            plt.xlabel("predicted probability"); plt.ylabel("observed frequency")
            plt.title(f"Reliability - {regime_name}"); plt.legend()
            rel_path = os.path.join(outdir, f"reliability_{ts}.png")
            plt.tight_layout(); plt.savefig(rel_path); plt.close()

            b_raw = brier_score_loss(y_val.values, val_pred_raw)
            b_cal = brier_score_loss(y_val.values, val_pred_cal)
            print(f"  진단 저장: {hist_path}, {rel_path}")
            print(f"  Brier raw={b_raw:.4f} → calibrated={b_cal:.4f}")
        except Exception as e:
            print(f"  (진단 저장 생략) {e}")

    # --- Save / Load ---
    def save_model(self, filepath=None):
        """레짐별 번들 저장"""
        if filepath is None:
            filepath = os.path.join(self.config.MODEL_DIR, 'current_model.pkl')

        # 레짐별 번들 개별 저장
        for rname, bundle in self.bundles.items():
            if bundle is not None:
                bpath = os.path.join(self.config.MODEL_DIR, f"bundle_{rname}.pkl")
                joblib.dump(bundle, bpath)
                print(f"레짐 {rname} 번들 저장: {bpath}")

        # Legacy 호환
        data = {
            'models': self.models,
            'scaler': self.scaler,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance,
            'calibration_method': self.calib_method,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        if self.calibrator is not None:
            data['calibrator'] = self.calibrator
        joblib.dump(data, filepath)

        if self.feature_importance is not None:
            _ensure_dir(self.config.FEATURE_LOG_DIR)
            fp = os.path.join(self.config.FEATURE_LOG_DIR,
                              f'feature_importance_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.csv')
            self.feature_importance.to_csv(fp, index=False)

        sel = os.path.join(self.config.FEATURE_LOG_DIR, 'selected_features.json')
        with open(sel, 'w') as f:
            json.dump({
                'features': self.selected_features,
                'count': len(self.selected_features) if self.selected_features else 0,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }, f, indent=2)

    def load_model(self, filepath=None):
        """레짐별 번들 로드"""
        for rname in ['UP', 'DOWN', 'FLAT']:
            bpath = os.path.join(self.config.MODEL_DIR, f"bundle_{rname}.pkl")
            if os.path.exists(bpath):
                self.bundles[rname] = joblib.load(bpath)
                print(f"레짐 {rname} 번들 로드: {bpath}")
            else:
                self.bundles[rname] = None

        if filepath is None:
            filepath = os.path.join(self.config.MODEL_DIR, 'current_model.pkl')
        if not os.path.exists(filepath):
            if any(b is not None for b in self.bundles.values()):
                for rname in ['FLAT', 'UP', 'DOWN']:
                    if self.bundles[rname] is not None:
                        self.models = self.bundles[rname]['models']
                        self.scaler = self.bundles[rname]['scaler']
                        self.calibrator = self.bundles[rname]['calibrator']
                        self.calib_method = self.bundles[rname]['calib_method']
                        self.selected_features = self.bundles[rname]['selected_features']
                        break
                return True
            return False

        data = joblib.load(filepath)
        self.models = data['models']
        self.scaler = data['scaler']
        self.selected_features = data['selected_features']
        self.feature_importance = data.get('feature_importance')
        self.calib_method = data.get('calibration_method', 'isotonic')
        self.calibrator = data.get('calibrator', None)
        return True


# ===============================
# Model Optimizer
# ===============================
class ModelOptimizer:
    def __init__(self, config):
        self.config = config
        self.trainer = ModelTrainer(config)

    def initial_training(self, df):
        print("="*60)
        print("초기 학습 시작 (레짐별 스위칭 모델)")
        print("="*60)

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            print(f"기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

        fe = FeatureEngineer()
        X = fe.create_feature_pool(df, lookback_window=200)
        y = fe.create_target(df, window=self.config.PREDICTION_WINDOW)

        valid = y.notna() & (y.index >= 200)
        X, y = X[valid], y[valid]
        print(f"유효 샘플: {len(X)}")

        print("\n미래 누출 점검...")
        if not fe.validate_no_future_leak(X, y):
            print("⚠️ 누출 의심 존재")
        else:
            print("✓ 누출 없음")

        print("\n레짐별 피처 선택...")
        topk = int(getattr(self.config, 'TOP_K_FEATURES', 30))
        self.trainer.feature_selection_regime(X, y, regime_col='regime', top_k=topk)

        print("\n레짐별 학습 + 캘리브레이션...")
        metrics = self.trainer.train_ensemble_regime(X, y, regime_col='regime', test_size=0.2)

        print("\n전체 결과:")
        for regime_name, regime_metrics in metrics.items():
            print(f"\n[{regime_name}]")
            for split, m in regime_metrics.items():
                print(f"  {split}: acc={m['accuracy']:.4f}, prec={m['precision']:.4f}, "
                      f"rec={m['recall']:.4f}, f1={m['f1']:.4f}, win={m['win_rate']:.4f}")

        self.trainer.save_model()
        print("\n모델 저장 완료")

        _ensure_dir(self.config.MODEL_DIR)
        all_rows = []
        for regime_name, regime_metrics in metrics.items():
            for split, m in regime_metrics.items():
                row = {'regime': regime_name, 'split': split}
                row.update(m)
                all_rows.append(row)
        
        results_df = pd.DataFrame(all_rows)
        results_df.to_csv(
            os.path.join(self.config.MODEL_DIR, 
                        f'training_results_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.csv'),
            index=False
        )
        
        return metrics

    def retrain_model(self, new_df):
        print("\n재학습 시작 (레짐별)...")
        if 'timestamp' in new_df.columns:
            new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], utc=True)

        fe = FeatureEngineer()
        X = fe.create_feature_pool(new_df, lookback_window=200)
        y = fe.create_target(new_df, window=self.config.PREDICTION_WINDOW)

        valid = y.notna() & (y.index >= 200)
        X, y = X[valid], y[valid]

        old = ModelTrainer(self.config)
        old.load_model()

        new = ModelTrainer(self.config)
        new.selected_features = old.selected_features

        new_metrics = new.train_ensemble_regime(X, y, regime_col='regime', test_size=0.2)

        n = len(X)
        if n > 0:
            start = int(n * 0.9)
            X_te, y_te = X.iloc[start:], y.iloc[start:]
            
            regime_accs = {'old': {}, 'new': {}}
            
            for rval in [1, -1, 0]:
                rname = self.trainer._regime_to_name(rval)
                mask = (X_te['regime'] == rval)
                if mask.sum() < 10:
                    continue
                    
                X_r = X_te[mask]
                y_r = y_te[mask]
                
                old_pred = old.predict(X_r, regime=rval)
                new_pred = new.predict(X_r, regime=rval)
                
                old_acc = accuracy_score(y_r, old_pred)
                new_acc = accuracy_score(y_r, new_pred)
                
                regime_accs['old'][rname] = old_acc
                regime_accs['new'][rname] = new_acc

            print("\n모델 비교(최근 10%, 레짐별):")
            for rname in ['UP', 'DOWN', 'FLAT']:
                if rname in regime_accs['old']:
                    print(f"  [{rname}] old: {regime_accs['old'][rname]:.4f}, "
                          f"new: {regime_accs['new'][rname]:.4f}")
            
            old_avg = np.mean(list(regime_accs['old'].values())) if regime_accs['old'] else 0
            new_avg = np.mean(list(regime_accs['new'].values())) if regime_accs['new'] else 0

            if new_avg > old_avg:
                print("→ 새 모델 채택")
                bkp = os.path.join(self.config.MODEL_DIR,
                                   f'backup/model_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.pkl')
                _ensure_dir(os.path.dirname(bkp))
                old.save_model(bkp)
                new.save_model()
                self.trainer = new
            else:
                print("→ 기존 모델 유지")
                self.trainer = old

        return new_metrics

    def analyze_failures(self, trade_log_df):
        """실패 거래 패턴 분석"""
        if 'result' not in trade_log_df.columns:
            return {}

        fails = trade_log_df[trade_log_df['result'] == 0].copy()
        if fails.empty:
            print("실패 거래 없음.")
            return {}

        print("\n" + "="*60)
        print(f"실패 패턴 분석 (n={len(fails)})")
        print("="*60)

        patterns = {}

        # 변동성 기반
        if 'atr_14' in fails.columns:
            f_atr = fails['atr_14'].dropna()
            s_atr = trade_log_df[trade_log_df['result'] == 1]['atr_14'].dropna()
            if len(f_atr) > 5 and len(s_atr) > 5:
                if f_atr.quantile(0.75) > s_atr.quantile(0.75) * 1.2:
                    th = float(f_atr.quantile(0.6))
                    patterns['high_volatility'] = {
                        'type': 'num', 'name': 'High ATR',
                        'field': 'atr_14', 'operator': '>',
                        'threshold': th,
                        'condition': f"atr_14>{th:.6f}",
                        'reason': f"실패 ATR↑ (median {f_atr.median():.6f} vs 성공 {s_atr.median():.6f})",
                        'improvement': 0.0
                    }
                    print(f"✓ 변동성 필터: ATR>{th:.6f}")

        # 거래량 기반
        if 'volume_ratio' in fails.columns:
            f_v = fails['volume_ratio'].dropna()
            s_v = trade_log_df[trade_log_df['result'] == 1]['volume_ratio'].dropna()
            if len(f_v) > 5 and len(s_v) > 5:
                high_total = len(trade_log_df[trade_log_df['volume_ratio'] > 1.5])
                if high_total > 0:
                    high_fail = len(f_v[f_v > 1.5])
                    rate = high_fail / high_total if high_total else 0
                    if rate > 0.6:
                        patterns['high_volume'] = {
                            'type': 'num', 'name': 'High Volume',
                            'field': 'volume_ratio', 'operator': '>',
                            'threshold': 1.5,
                            'condition': "volume_ratio>1.5",
                            'reason': f"고거래량 실패율 {rate:.1%}",
                            'improvement': 0.0
                        }
                        print(f"✓ 고거래량 필터: volume_ratio>1.5")

                low_total = len(trade_log_df[trade_log_df['volume_ratio'] < 0.5])
                if low_total > 0:
                    low_fail = len(f_v[f_v < 0.5])
                    rate = low_fail / low_total if low_total else 0
                    if rate > 0.6:
                        patterns['low_volume'] = {
                            'type': 'num', 'name': 'Low Volume',
                            'field': 'volume_ratio', 'operator': '<',
                            'threshold': 0.5,
                            'condition': "volume_ratio<0.5",
                            'reason': f"저거래량 실패율 {rate:.1%}",
                            'improvement': 0.0
                        }
                        print(f"✓ 저거래량 필터: volume_ratio<0.5")

        # RSI 극단
        if 'rsi_14' in fails.columns:
            ext_fail = fails[(fails['rsi_14'] < 30) | (fails['rsi_14'] > 70)]
            ext_all  = trade_log_df[(trade_log_df['rsi_14'] < 30) | (trade_log_df['rsi_14'] > 70)]
            if len(ext_all) > 5:
                rate = len(ext_fail) / len(ext_all)
                if rate > 0.6:
                    patterns['rsi_extreme'] = {
                        'type': 'range', 'name': 'RSI extreme',
                        'field': 'rsi_14', 'operator': 'extreme',
                        'lower_threshold': 30.0, 'upper_threshold': 70.0,
                        'condition': "rsi_14<30 or rsi_14>70",
                        'reason': f"극단 RSI에서 실패↑ ({rate:.1%})",
                        'improvement': 0.0
                    }
                    print("✓ RSI 극단값 필터")

        # 저장
        _ensure_dir(self.config.MODEL_DIR)
        fpath = os.path.join(self.config.MODEL_DIR, 'adaptive_filters.json')
        with open(fpath, 'w') as f:
            json.dump({'active_filters': [], 'filter_history': [patterns]}, f, indent=2)
        print(f"\n임시 필터 제안 저장: {fpath}\n")
        return patterns


# ===============================
# Quick sample run
# ===============================
if __name__ == "__main__":
    from config import Config
    Config.create_directories()

    print("샘플 데이터 생성...")
    periods = 1000
    ts = pd.date_range(end=datetime.now(timezone.utc), periods=periods, freq='1min', tz='UTC')
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
        rows.append({'timestamp': t, 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})
    df = pd.DataFrame(rows)
    df.to_csv('sample_data.csv', index=False)

    opt = ModelOptimizer(Config)
    metrics = opt.initial_training(df)

    print("\n학습 결과 요약:")
    for regime_name, regime_metrics in metrics.items():
        print(f"\n[{regime_name}]")
        for split, m in regime_metrics.items():
            print(f"  {split}: acc={m['accuracy']:.4f}, prec={m['precision']:.4f}, "
                  f"rec={m['recall']:.4f}, f1={m['f1']:.4f}, win={m['win_rate']:.4f}")
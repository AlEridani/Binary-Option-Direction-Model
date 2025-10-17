# model_train.py - 모델 학습 및 재학습 모듈 (시계열 기반, 미래 데이터 누출 방지)

import os
import json
from datetime import datetime, timezone
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')


# =========================
# Feature Engineering
# =========================
class FeatureEngineer:
    """피처 엔지니어링 클래스 - 미래 데이터 누출 방지"""

    @staticmethod
    def create_feature_pool(df: pd.DataFrame, lookback_window: int = 200) -> pd.DataFrame:
        """
        전체 피처 풀 생성 (모든 피처는 과거 데이터만 사용)
        - 반드시 현재 캔들 정보는 배제: 가격/거래량 등은 모두 shift(1) 적용
        - rolling/EMA/BB/RSI/MACD/ATR 등도 shift(1) 기반으로 계산
        """
        features = pd.DataFrame(index=df.index)

        # 시간 정보 UTC
        if 'timestamp' in df.columns:
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            features['timestamp'] = df['timestamp']

        # ====== 기본 가격/거래량 (모두 이전 캔들) ======
        features['prev_open'] = df['open'].shift(1)
        features['prev_high'] = df['high'].shift(1)
        features['prev_low'] = df['low'].shift(1)
        features['prev_close'] = df['close'].shift(1)
        features['prev_volume'] = df['volume'].shift(1)

        # ====== 변화율 기반 ======
        for period in [1, 2, 3, 5, 10, 15, 30, 60]:
            # 현재가 대신 이전가 / 더 과거가
            features[f'return_{period}'] = df['close'].shift(1) / df['close'].shift(period + 1) - 1
            features[f'volume_change_{period}'] = df['volume'].shift(1) / df['volume'].shift(period + 1) - 1

        # ====== 이동평균/가격 대비 ======
        for period in [5, 10, 20, 50, 100, 200]:
            shifted_close = df['close'].shift(1)
            ma = shifted_close.rolling(window=period, min_periods=period).mean()
            features[f'ma_{period}'] = ma
            features[f'price_to_ma_{period}'] = shifted_close / ma - 1
            # 기울기(모멘텀)
            features[f'ma_{period}_slope'] = ma.diff(5) / (ma.shift(5) + 1e-8)

        # ====== EMA ======
        for period in [12, 26, 50]:
            ema = df['close'].shift(1).ewm(span=period, adjust=False, min_periods=period).mean()
            features[f'ema_{period}'] = ema
            features[f'price_to_ema_{period}'] = df['close'].shift(1) / ema - 1

        # ====== 볼린저 ======
        for period in [20, 50]:
            shifted_close = df['close'].shift(1)
            ma = shifted_close.rolling(window=period, min_periods=period).mean()
            std = shifted_close.rolling(window=period, min_periods=period).std()
            upper = ma + 2 * std
            lower = ma - 2 * std
            features[f'bb_upper_{period}'] = upper
            features[f'bb_lower_{period}'] = lower
            features[f'bb_width_{period}'] = (4 * std) / (ma + 1e-8)
            features[f'bb_position_{period}'] = (shifted_close - lower) / (upper - lower + 1e-8)

        # ====== RSI ======
        for period in [14, 28]:
            shifted_close = df['close'].shift(1)
            delta = shifted_close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
            rs = gain / (loss + 1e-8)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # ====== MACD ======
        shifted_close = df['close'].shift(1)
        ema_12 = shifted_close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema_26 = shifted_close.ewm(span=26, adjust=False, min_periods=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
        features['macd'] = macd_line
        features['macd_signal'] = signal_line
        features['macd_histogram'] = macd_line - signal_line

        # ====== Stochastic ======
        for period in [14]:
            sh = df['high'].shift(1)
            sl = df['low'].shift(1)
            sc = df['close'].shift(1)
            low_min = sl.rolling(window=period, min_periods=period).min()
            high_max = sh.rolling(window=period, min_periods=period).max()
            features[f'stoch_{period}'] = (sc - low_min) / (high_max - low_min + 1e-8) * 100

        # ====== ATR ======
        for period in [14, 28]:
            sh = df['high'].shift(1)
            sl = df['low'].shift(1)
            sc = df['close'].shift(1)
            tr = pd.concat([
                (sh - sl),
                (sh - sc.shift(1)).abs(),
                (sl - sc.shift(1)).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(window=period, min_periods=period).mean()
            features[f'atr_{period}'] = atr
            features[f'atr_ratio_{period}'] = atr / (sc + 1e-8)

        # ====== 거래량 ======
        sv = df['volume'].shift(1)
        features['volume_sma_10'] = sv.rolling(window=10, min_periods=10).mean()
        features['volume_sma_50'] = sv.rolling(window=50, min_periods=50).mean()
        features['volume_ratio'] = sv / (features['volume_sma_10'] + 1e-8)
        features['volume_trend'] = (features['volume_sma_10'] / (features['volume_sma_50'] + 1e-8))

        # ====== OBV ======
        price_diff = df['close'].diff().shift(1)
        obv = (np.sign(price_diff) * df['volume'].shift(1)).cumsum()
        features['obv'] = obv
        features['obv_ema'] = obv.ewm(span=20, adjust=False, min_periods=20).mean()
        features['obv_signal'] = obv / (features['obv_ema'] + 1e-8) - 1

        # ====== 캔들 패턴(이전 캔들) ======
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        prev_high = df['high'].shift(1)
        prev_low = df['low'].shift(1)
        features['body_size'] = (prev_close - prev_open).abs() / (prev_open + 1e-8)
        features['upper_shadow'] = (prev_high - pd.concat([prev_open, prev_close], axis=1).max(axis=1)) / (prev_open + 1e-8)
        features['lower_shadow'] = (pd.concat([prev_open, prev_close], axis=1).min(axis=1) - prev_low) / (prev_open + 1e-8)
        features['body_position'] = (prev_close - prev_open) / (prev_high - prev_low + 1e-8)

        # 최근 3개 캔들 특성
        for i in range(1, 4):
            features[f'candle_direction_{i}'] = np.sign(df['close'].shift(i) - df['open'].shift(i))
            features[f'candle_size_{i}'] = (df['high'].shift(i) - df['low'].shift(i)) / (df['close'].shift(i) + 1e-8)

        # ====== 시간 피처(UTC) ======
        if 'timestamp' in df.columns:
            dt = pd.to_datetime(df['timestamp'], utc=True)
            features['hour'] = dt.dt.hour
            features['minute'] = dt.dt.minute
            features['day_of_week'] = dt.dt.dayofweek
            features['day_of_month'] = dt.dt.day
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
            features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)

        # ====== 마이크로 구조 ======
        for period in [3, 5, 10]:
            sh = df['high'].shift(1)
            sl = df['low'].shift(1)
            sc = df['close'].shift(1)
            features[f'high_low_ratio_{period}'] = ((sh / (sl + 1e-8) - 1)
                                                    .rolling(window=period, min_periods=period).mean())
            features[f'close_position_{period}'] = (
                ((sc - sl) / (sh - sl + 1e-8)).rolling(window=period, min_periods=period).mean()
            )

        # ====== 추세 강도 ======
        for period in [10, 20, 50]:
            sc = df['close'].shift(1)
            ma = sc.rolling(window=period, min_periods=period).mean()
            above_ma = (sc > ma).astype(int)
            features[f'trend_strength_{period}'] = above_ma.rolling(window=period, min_periods=period).mean()

        # ====== 변동성 ======
        for period in [10, 30]:
            shifted_returns = df['close'].shift(1).pct_change()
            vol = shifted_returns.rolling(window=period, min_periods=period).std()
            vol_long = shifted_returns.rolling(window=period * 3, min_periods=period * 3).std()
            features[f'volatility_{period}'] = vol
            features[f'volatility_ratio_{period}'] = vol / (vol_long + 1e-8)

        # 첫 lookback_window 제거 (지표 안정화)
        features = features.iloc[lookback_window:]
        return features

    @staticmethod
    def create_target(df: pd.DataFrame, window: int = 10) -> pd.Series:
        """
        타깃: window분 후 종가가 상승했는가 (ROC>0과 동일)
        """
        current_close = df['close']
        future_close = df['close'].shift(-window)
        target = (future_close > current_close).astype(int)
        return target

    @staticmethod
    def validate_no_future_leak(features: pd.DataFrame, target: pd.Series) -> bool:
        """
        미래 데이터 누출 간단 검증:
        - 동시 상관 vs 피처를 +1 시프트한 상관 비교(라그 상관)
        - 동시 상관이 과도하고 라그 상관 대비 유의하게 높으면 경고
        """
        issues = []
        t = target.dropna()
        f = features.loc[t.index].drop(columns=['timestamp'], errors='ignore')

        for col in f.columns:
            s0 = f[col].corr(t)
            s1 = f[col].shift(1).corr(t)  # 피처를 더 과거로 미룸
            if s0 is not None and s1 is not None:
                if (abs(s0) > 0.5) and (abs(s0) - abs(s1) > 0.2):
                    issues.append(f"{col}: corr={s0:.3f} vs lagged={s1:.3f}")

        if issues:
            print("⚠️ 가능 누수 신호(참고):")
            for it in issues:
                print(f"  - {it}")
            # 경고만, 하드 스톱하지는 않음
            return False
        return True


# =========================
# Trainer
# =========================
class TimeSeriesModelTrainer:
    """시계열 기반 모델 학습 클래스"""

    def __init__(self, config):
        self.config = config
        self.models: list[lgb.Booster] = []
        self.feature_importance: pd.DataFrame | None = None
        self.selected_features: list[str] | None = None
        self.scaler = StandardScaler()

    def feature_selection_temporal(self, X: pd.DataFrame, y: pd.Series, top_k: int = 50) -> list[str]:
        """
        시계열 고려한 피처 선택 (간단: 앞 70%로 학습, 뒤 30%로 검증)
        """
        # timestamp 제외
        feature_cols = [c for c in X.columns if c != 'timestamp']
        train_size = int(len(X) * 0.7)
        X_train, y_train = X.iloc[:train_size][feature_cols], y.iloc[:train_size]
        X_val, y_val = X.iloc[train_size:][feature_cols], y.iloc[train_size:]

        # 전처리
        def _clean(df):
            return (df.fillna(method='ffill').fillna(0).replace([np.inf, -np.inf], 0))

        X_train, X_val = _clean(X_train), _clean(X_val)

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

        params = self.config.LGBM_PARAMS.copy()
        params.update({'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt'})

        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )

        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        self.feature_importance = importance
        self.selected_features = importance.head(top_k)['feature'].tolist()

        print(f"\n선택된 상위 10개 피처:")
        for _, row in importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.2f}")

        return self.selected_features

    def _split_time_blocks(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """
        시간 순서 분할: 60/20/20 (train/val/test)
        """
        total = len(X)
        test = int(total * test_size)
        val = test
        train = total - test - val
        X_train, y_train = X.iloc[:train], y.iloc[:train]
        X_val, y_val = X.iloc[train:train + val], y.iloc[train:train + val]
        X_test, y_test = X.iloc[train + val:], y.iloc[train + val:]
        return (X_train, y_train, X_val, y_val, X_test, y_test)

    def train_ensemble_temporal(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> dict:
        """
        시계열 앙상블 모델 학습 (60/20/20) + 동적 가중 앙상블
        """
        self.models = []

        # timestamp 제외 & 선택 피처 적용
        if 'timestamp' in X.columns:
            X = X.drop('timestamp', axis=1)
        X = X[self.selected_features]

        # 시간 분할
        X_train, y_train, X_val, y_val, X_test, y_test = self._split_time_blocks(X, y, test_size)

        print(f"\n데이터 분할 (시간 순서 유지):")
        print(f"  학습: {len(X_train)} | 검증: {len(X_val)} | 테스트: {len(X_test)}")

        # 전처리
        def _prep(df):
            return df.fillna(method='ffill').fillna(0).replace([np.inf, -np.inf], 0)

        X_train, X_val, X_test = _prep(X_train), _prep(X_val), _prep(X_test)

        # 스케일러 (LightGBM엔 필수 아님이지만 기존 흐름 유지)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        # 앙상블 학습 (각 모델에 시간 다양성/파라미터 약간씩 변화)
        n_models = getattr(self.config, 'ENSEMBLE_MODELS', 5)
        for i in range(n_models):
            params = self.config.LGBM_PARAMS.copy()
            params.update({
                'random_state': 42 + i,
                'num_leaves': max(15, int(31 + i * 8)),
                'learning_rate': max(0.005, float(0.05 - i * 0.005)),
                'feature_fraction': min(1.0, float(0.9 - i * 0.05)),
                'bagging_fraction': min(1.0, float(0.8 + i * 0.03)),
            })

            start_idx = int(i * len(X_train_scaled) * 0.1)
            X_tr_sub = X_train_scaled[start_idx:]
            y_tr_sub = y_train.iloc[start_idx:]

            ds_tr = lgb.Dataset(X_tr_sub, y_tr_sub)
            ds_val = lgb.Dataset(X_val_scaled, y_val, reference=ds_tr)

            model = lgb.train(
                params,
                ds_tr,
                valid_sets=[ds_val],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            self.models.append(model)
            print(f"  모델 {i + 1} 학습 완료")

        # 성능 측정
        metrics = {
            'train': self.evaluate_predictions(y_train, self._predict_proba_from_scaled(self.models, X_train_scaled)),
            'validation': self.evaluate_predictions(y_val, self._predict_proba_from_scaled(self.models, X_val_scaled)),
            'test': self.evaluate_predictions(y_test, self._predict_proba_from_scaled(self.models, X_test_scaled)),
        }
        return metrics

    def _predict_proba_from_scaled(self, models: list, X_scaled: np.ndarray) -> np.ndarray:
        """스케일 완료 데이터를 입력받아 동적 가중 앙상블 확률 반환"""
        if not models:
            return np.zeros(X_scaled.shape[0])
        preds = []
        for m in models:
            p = m.predict(X_scaled, num_iteration=m.best_iteration)
            preds.append(p)
        P = np.vstack(preds)  # [n_models, n_samples]

        # 최근 모델 가중 ↑ (지수 가중)
        n = len(models)
        w = np.array([1.05 ** i for i in range(n)])  # older->small, newer->large
        w = w / w.sum()
        return (w[:, None] * P).sum(axis=0)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """원본 X로부터 확률 예측"""
        if 'timestamp' in X.columns:
            X = X.drop('timestamp', axis=1)
        X = X[self.selected_features]
        X = X.fillna(method='ffill').fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X)
        return self._predict_proba_from_scaled(self.models, X_scaled)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) > threshold).astype(int)

    def evaluate_predictions(self, y_true: pd.Series, y_pred_proba: np.ndarray, threshold: float = 0.5) -> dict:
        y_pred = (y_pred_proba > threshold).astype(int)
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'win_rate': float(np.mean(y_pred == y_true)),
        }

    # -----------------
    # Save / Load
    # -----------------
    def save_model(self, filepath: str | None = None):
        if filepath is None:
            filepath = os.path.join(self.config.MODEL_DIR, 'current_model.pkl')
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        joblib.dump(model_data, filepath)

        # 피처 중요도 & 선택 피처 저장
        if self.feature_importance is not None:
            os.makedirs(self.config.FEATURE_LOG_DIR, exist_ok=True)
            fpath = os.path.join(
                self.config.FEATURE_LOG_DIR,
                f'feature_importance_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.csv'
            )
            self.feature_importance.to_csv(fpath, index=False)

        sel_path = os.path.join(self.config.FEATURE_LOG_DIR, 'selected_features.json')
        with open(sel_path, 'w') as f:
            json.dump({
                'features': self.selected_features,
                'count': len(self.selected_features) if self.selected_features else 0,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }, f, indent=2)

    def load_model(self, filepath: str | None = None) -> bool:
        if filepath is None:
            filepath = os.path.join(self.config.MODEL_DIR, 'current_model.pkl')
        if not os.path.exists(filepath):
            return False
        data = joblib.load(filepath)
        self.models = data['models']
        self.scaler = data['scaler']
        self.selected_features = data['selected_features']
        self.feature_importance = data.get('feature_importance')
        return True


# =========================
# Optimizer / Retrain
# =========================
class ModelOptimizer:
    """모델 최적화 및 재학습 관리"""

    def __init__(self, config):
        self.config = config
        self.trainer = TimeSeriesModelTrainer(config)
        self.performance_history = []

    def _prepare_xy(self, df: pd.DataFrame):
        """공통: 피처/타깃 생성 + 유효 인덱스 필터"""
        fe = FeatureEngineer()
        X = fe.create_feature_pool(df, lookback_window=200)
        y = fe.create_target(df, window=self.config.PREDICTION_WINDOW)

        # 유효 인덱스: lookback 이후 + 타깃 NaN 제거 + 마지막 window 제거(명시)
        valid = y.notna() & (X.index >= X.index.min())
        # 마지막 window 구간 제거 (future_close가 없는 꼬리 제거)
        tail = self.config.PREDICTION_WINDOW
        if tail > 0:
            X = X.iloc[:-tail] if len(X) > tail else X.iloc[0:0]
            y = y.iloc[:-tail] if len(y) > tail else y.iloc[0:0]

        X = X[valid.iloc[:len(X)].values] if len(valid) >= len(X) else X
        y = y.loc[X.index]

        return X, y

    def initial_training(self, df: pd.DataFrame) -> dict:
        print("=" * 50)
        print("초기 모델 학습 시작 (시계열 기반)")
        print("=" * 50)

        if 'timestamp' in df.columns:
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            print(f"데이터 기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

        print("\n피처 생성 중...")
        X, y = self._prepare_xy(df)
        print(f"유효 데이터: {len(X)} 샘플")

        print("\n미래 데이터 누출 검증 중...")
        if not FeatureEngineer.validate_no_future_leak(X, y):
            print("⚠️ 참고: 잠재 누수 신호가 있습니다(라그 상관). 피처/라벨 정의 재검토 권장.")
        else:
            print("✓ 미래 데이터 누출 없음")

        print("\n피처 선택 중...")
        selected = self.trainer.feature_selection_temporal(X, y, top_k=50)
        print(f"선택된 피처 수: {len(selected)}")

        print("\n앙상블 모델 학습 중...")
        metrics = self.trainer.train_ensemble_temporal(X, y, test_size=0.2)

        print("\n학습 결과:")
        for split, metric in metrics.items():
            print(f"\n{split.upper()}:")
            for k, v in metric.items():
                print(f"  {k}: {v:.4f}")

        self.trainer.save_model()
        print("\n모델 저장 완료")

        # 결과 CSV
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
        pd.DataFrame(metrics).T.to_csv(
            os.path.join(self.config.MODEL_DIR, f'training_results_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.csv')
        )
        return metrics

    def retrain_model(self, new_data_df: pd.DataFrame) -> dict:
        """
        새로운 데이터로 재학습
        - 재학습 시에도 **새로 최적 피처를 다시 선택**
        - 새/구 모델을 동일한 테스트 구간(마지막 20%)으로 비교
        """
        print("\n모델 재학습 시작...")

        if 'timestamp' in new_data_df.columns:
            new_data_df = new_data_df.copy()
            new_data_df['timestamp'] = pd.to_datetime(new_data_df['timestamp'], utc=True)

        X, y = self._prepare_xy(new_data_df)
        if len(X) == 0:
            print("재학습 데이터가 충분하지 않습니다.")
            return {}

        # --------- (A) 기존 모델 로드 ---------
        old_tr = TimeSeriesModelTrainer(self.config)
        has_old = old_tr.load_model()

        # --------- (B) 새 모델: 새로 피처 선택 ---------
        new_tr = TimeSeriesModelTrainer(self.config)
        print("\n(재학습) 피처 선택 중...")
        new_tr.feature_selection_temporal(X, y, top_k=50)

        print("\n(재학습) 앙상블 학습 중...")
        new_metrics = new_tr.train_ensemble_temporal(X, y, test_size=0.2)

        # --------- (C) 공정 비교: 마지막 20% ---------
        test_start = int(len(X) * 0.8)
        X_test = X.iloc[test_start:]
        y_test = y.iloc[test_start:]

        if len(X_test) > 0 and has_old:
            old_tr.selected_features = old_tr.selected_features or new_tr.selected_features
            # old_tr가 사용했던 스케일러/피처 그대로 예측
            try:
                old_pred = old_tr.predict(X_test)
                new_pred = new_tr.predict(X_test)
            except Exception:
                # 혹시 피처 미스매치 발생 시, 교집합 피처로 리매핑
                shared = list(set(new_tr.selected_features).intersection(set(old_tr.selected_features or [])))
                if not shared:
                    shared = new_tr.selected_features
                    old_tr.selected_features = new_tr.selected_features
                Xs = X_test.drop(columns=[c for c in X_test.columns if c not in shared])
                old_tr.selected_features = shared
                new_tr.selected_features = shared
                old_pred = old_tr.predict(Xs)
                new_pred = new_tr.predict(Xs)

            old_acc = accuracy_score(y_test, old_pred)
            new_acc = accuracy_score(y_test, new_pred)

            print(f"\n모델 비교 (마지막 20%):")
            print(f"기존 모델 정확도: {old_acc:.4f}")
            print(f"새 모델 정확도: {new_acc:.4f}")

            if new_acc > old_acc:
                print("새 모델이 더 우수 → 교체 진행")
                # 백업
                backup_path = os.path.join(
                    self.config.MODEL_DIR,
                    f'backup/model_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.pkl'
                )
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                old_tr.save_model(backup_path)
                # 저장 & 바인딩
                new_tr.save_model()
                self.trainer = new_tr
                print("모델 교체 완료")
            else:
                print("기존 모델 유지")
                self.trainer = old_tr
        else:
            # 기존 모델이 없거나 테스트셋이 너무 작으면 그냥 신규 저장
            new_tr.save_model()
            self.trainer = new_tr
            print("신규 모델 저장 완료")

        return new_metrics

    def analyze_failures(self, trade_log_df: pd.DataFrame) -> dict:
        """실패 거래 분석 및 필터 제안"""
        if 'result' not in trade_log_df.columns:
            return {}

        failures = trade_log_df[trade_log_df['result'] == 0].copy()
        if len(failures) == 0:
            return {}

        patterns = {'high_volatility': {}, 'low_volume': {}, 'time_based': {}, 'technical_signals': {}}

        # 변동성
        if 'atr_14' in failures.columns:
            thr = float(failures['atr_14'].quantile(0.75))
            patterns['high_volatility'] = {'atr_14_threshold': thr, 'filter': f"atr_14 > {thr}"}

        # 거래량
        if 'volume_ratio' in failures.columns:
            thr = float(failures['volume_ratio'].quantile(0.25))
            patterns['low_volume'] = {'volume_ratio_threshold': thr, 'filter': f"volume_ratio < {thr}"}

        # 시간대(UTC)
        if 'entry_time' in failures.columns:
            failures['hour'] = pd.to_datetime(failures['entry_time'], utc=True).dt.hour
            by_hour = failures.groupby('hour').size()
            if len(by_hour) > 0:
                high_hours = by_hour[by_hour > by_hour.mean() + by_hour.std()].index.tolist()
                if high_hours:
                    patterns['time_based'] = {'avoid_hours': high_hours, 'filter': f"hour not in {high_hours}"}

        # RSI 극값
        if 'rsi_14' in failures.columns:
            rsi_fail = failures[(failures['rsi_14'] < 30) | (failures['rsi_14'] > 70)]
            if len(rsi_fail) > len(failures) * 0.3:
                patterns['technical_signals']['rsi_extreme'] = {
                    'condition': "RSI < 30 or RSI > 70",
                    'failure_rate': float(len(rsi_fail) / len(failures))
                }

        # 결과 저장(원하면 그대로 분석 재활용)
        os.makedirs(self.config.TRADE_LOG_DIR, exist_ok=True)
        failures.to_csv(
            os.path.join(self.config.TRADE_LOG_DIR, f'failure_analysis_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.csv'),
            index=False
        )
        return patterns


# 외부에서 import 편의상 별칭 유지
ModelTrainer = TimeSeriesModelTrainer


# =========================
# Standalone quick test
# =========================
if __name__ == "__main__":
    from config import Config

    Config.create_directories()

    # 샘플 시계열 생성 (UTC)
    periods = 500
    ts = pd.date_range(end=datetime.now(timezone.utc), periods=periods, freq='1min', tz='UTC')
    rng = np.random.default_rng(42)
    base = 42000 + rng.standard_normal(periods).cumsum() * 100

    rows = []
    for i, t in enumerate(ts):
        o = base[i] + rng.uniform(-50, 50)
        c = base[i] + rng.uniform(-50, 50)
        h = max(o, c) + rng.uniform(0, 100)
        l = min(o, c) - rng.uniform(0, 100)
        v = rng.uniform(100, 1000)
        rows.append({'timestamp': t, 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})
    df = pd.DataFrame(rows)

    # 저장
    df.to_csv('sample_data.csv', index=False)
    print("샘플 데이터 저장: sample_data.csv")

    # 초기 학습
    opt = ModelOptimizer(Config)
    metrics = opt.initial_training(df)

    print("\n테스트 완료!")

# model_train.py - 시계열 안전 학습/재학습 + 실패패턴 필터 (timestamp=close_time)
import os
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import warnings

warnings.filterwarnings("ignore")


# ---------------- Feature Engineering ----------------
class FeatureEngineer:
    @staticmethod
    def create_feature_pool(df: pd.DataFrame, lookback_window: int = 200) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)

        # timestamp는 이미 close_time으로 세팅되어 들어옴 (UTC 가정)
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            f["timestamp"] = ts

        # 과거 값만 사용 (shift(1))
        f["prev_open"] = df["open"].shift(1)
        f["prev_high"] = df["high"].shift(1)
        f["prev_low"]  = df["low"].shift(1)
        f["prev_close"] = df["close"].shift(1)
        f["prev_volume"] = df["volume"].shift(1)

        for p in [1,2,3,5,10,15,30,60]:
            f[f"return_{p}"] = (df["close"].shift(1) / df["close"].shift(p+1) - 1)
            f[f"volume_change_{p}"] = (df["volume"].shift(1) / df["volume"].shift(p+1) - 1)

        for p in [5,10,20,50,100,200]:
            ma = df["close"].shift(1).rolling(p, min_periods=p).mean()
            f[f"ma_{p}"] = ma
            f[f"price_to_ma_{p}"] = (df["close"].shift(1) / ma - 1)
            f[f"ma_{p}_slope"] = ma.diff(5) / ma.shift(5)

        for p in [12,26,50]:
            ema = df["close"].shift(1).ewm(span=p, adjust=False, min_periods=p).mean()
            f[f"ema_{p}"] = ema
            f[f"price_to_ema_{p}"] = (df["close"].shift(1) / ema - 1)

        for p in [20,50]:
            sc = df["close"].shift(1)
            ma = sc.rolling(p, min_periods=p).mean()
            std = sc.rolling(p, min_periods=p).std()
            f[f"bb_upper_{p}"] = ma + 2*std
            f[f"bb_lower_{p}"] = ma - 2*std
            f[f"bb_width_{p}"] = (4*std) / ma
            f[f"bb_position_{p}"] = (sc - f[f"bb_lower_{p}"]) / (f[f"bb_upper_{p}"] - f[f"bb_lower_{p}"] + 1e-4)

        for p in [14,28]:
            sc = df["close"].shift(1)
            delta = sc.diff()
            gain = delta.where(delta>0,0).rolling(p, min_periods=p).mean()
            loss = (-delta.where(delta<0,0)).rolling(p, min_periods=p).mean()
            rs = gain / (loss + 1e-4)
            f[f"rsi_{p}"] = 100 - (100/(1+rs))

        sc = df["close"].shift(1)
        ema12 = sc.ewm(span=12, adjust=False, min_periods=12).mean()
        ema26 = sc.ewm(span=26, adjust=False, min_periods=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
        f["macd"] = macd
        f["macd_signal"] = signal
        f["macd_histogram"] = macd - signal

        sh, sl, sc2 = df["high"].shift(1), df["low"].shift(1), df["close"].shift(1)
        low14  = sl.rolling(14, min_periods=14).min()
        high14 = sh.rolling(14, min_periods=14).max()
        f["stoch_14"] = (sc2 - low14) / (high14 - low14 + 1e-4) * 100

        for p in [14,28]:
            sh, sl, sc3 = df["high"].shift(1), df["low"].shift(1), df["close"].shift(1)
            tr = pd.concat([sh-sl, (sh - sc3.shift(1)).abs(), (sl - sc3.shift(1)).abs()], axis=1).max(axis=1)
            f[f"atr_{p}"] = tr.rolling(p, min_periods=p).mean()
            f[f"atr_ratio_{p}"] = f[f"atr_{p}"] / sc3

        sv = df["volume"].shift(1)
        f["volume_sma_10"] = sv.rolling(10, min_periods=10).mean()
        f["volume_sma_50"] = sv.rolling(50, min_periods=50).mean()
        f["volume_ratio"]  = sv / f["volume_sma_10"]
        f["volume_trend"]  = f["volume_sma_10"] / f["volume_sma_50"]

        price_diff = df["close"].diff().shift(1)
        obv = (np.sign(price_diff) * df["volume"].shift(1)).cumsum()
        f["obv"] = obv
        f["obv_ema"] = obv.ewm(span=20, adjust=False, min_periods=20).mean()
        f["obv_signal"] = obv / f["obv_ema"] - 1

        po, pc, ph, pl = df["open"].shift(1), df["close"].shift(1), df["high"].shift(1), df["low"].shift(1)
        f["body_size"] = (pc - po).abs() / po
        f["upper_shadow"] = (ph - pd.concat([po, pc], axis=1).max(axis=1)) / po
        f["lower_shadow"] = (pd.concat([po, pc], axis=1).min(axis=1) - pl) / po
        f["body_position"] = (pc - po) / (ph - pl + 1e-4)
        for i in range(1,4):
            f[f"candle_direction_{i}"] = np.sign(df["close"].shift(i) - df["open"].shift(i))
            f[f"candle_size_{i}"] = (df["high"].shift(i) - df["low"].shift(i)) / df["close"].shift(i)

        # 시간 피처 (timestamp=close_time)
        if "timestamp" in f.columns:
            dt = pd.to_datetime(f["timestamp"], utc=True, errors="coerce")
            f["hour"] = dt.dt.hour
            f["minute"] = dt.dt.minute
            f["day_of_week"] = dt.dt.dayofweek
            f["day_of_month"] = dt.dt.day
            f["hour_sin"] = np.sin(2*np.pi*f["hour"]/24)
            f["hour_cos"] = np.cos(2*np.pi*f["hour"]/24)
            f["dow_sin"] = np.sin(2*np.pi*f["day_of_week"]/7)
            f["dow_cos"] = np.cos(2*np.pi*f["day_of_week"]/7)

        # 마이크로/추세/변동성
        for p in [3,5,10]:
            sh, sl, sc4 = df["high"].shift(1), df["low"].shift(1), df["close"].shift(1)
            f[f"high_low_ratio_{p}"] = ((sh/sl - 1).rolling(p, min_periods=p).mean())
            f[f"close_position_{p}"] = (((sc4 - sl) / (sh - sl + 1e-4)).rolling(p, min_periods=p).mean())

        for p in [10,20,50]:
            sc5 = df["close"].shift(1)
            ma = sc5.rolling(p, min_periods=p).mean()
            f[f"trend_strength_{p}"] = (sc5 > ma).astype(int).rolling(p, min_periods=p).mean()

        for p in [10,30]:
            sr = df["close"].shift(1).pct_change()
            f[f"volatility_{p}"] = sr.rolling(p, min_periods=p).std()
            f[f"volatility_ratio_{p}"] = f[f"volatility_{p}"] / sr.rolling(p*3, min_periods=p*3).std()

        return f.iloc[lookback_window:]

    @staticmethod
    def create_target(df: pd.DataFrame, window: int = 10) -> pd.Series:
        # timestamp=close_time 기준 → 현재 닫힌 봉 대비 10분 뒤 닫힌 봉
        cur = df["close"]
        fut = df["close"].shift(-window)
        return (fut > cur).astype(float)


# ---------------- Trainer ----------------
class TimeSeriesModelTrainer:
    def __init__(self, config):
        self.config = config
        self.models = []
        self.scaler = StandardScaler()
        self.selected_features = None
        self.feature_importance = None

    def _prepare_xy(self, df: pd.DataFrame):
        if "timestamp" in df.columns:
            df = df.copy()
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

        fe = FeatureEngineer()
        X = fe.create_feature_pool(df, lookback_window=200)
        y = fe.create_target(df, window=self.config.PREDICTION_WINDOW)

        y = y.dropna()
        common = X.index.intersection(y.index)
        X = X.loc[common]
        y = y.loc[common].astype(int)

        X = X.fillna(method="ffill").fillna(0).replace([np.inf, -np.inf], 0)
        return X, y

    def feature_selection_temporal(self, X: pd.DataFrame, y: pd.Series, top_k: int = 50):
        cols = [c for c in X.columns if c != "timestamp"]
        n = len(X)
        cut = int(n * 0.7)
        Xtr, ytr = X.iloc[:cut][cols], y.iloc[:cut]
        Xva, yva = X.iloc[cut:][cols], y.iloc[cut:]

        dtr = lgb.Dataset(Xtr, ytr)
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        params = self.config.LGBM_PARAMS.copy()
        params.update({"objective": "binary", "metric": "binary_logloss"})

        model = lgb.train(
            params, dtr, valid_sets=[dva],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )

        imp = pd.DataFrame({
            "feature": cols,
            "importance": model.feature_importance("gain")
        }).sort_values("importance", ascending=False)

        self.feature_importance = imp
        self.selected_features = imp.head(top_k)["feature"].tolist()
        print("\n[피처선택 상위 10]")
        for _, r in imp.head(10).iterrows():
            print(f"  {r['feature']}: {r['importance']:.2f}")
        return self.selected_features

    def _split_ts(self, X: pd.DataFrame, y: pd.Series, test_size=0.2):
        tot = len(X); val_n = int(tot * test_size); tr_n = tot - 2*val_n
        Xtr, ytr = X.iloc[:tr_n], y.iloc[:tr_n]
        Xva, yva = X.iloc[tr_n:tr_n+val_n], y.iloc[tr_n:tr_n+val_n]
        Xte, yte = X.iloc[tr_n+val_n:], y.iloc[tr_n+val_n:]
        return Xtr, ytr, Xva, yva, Xte, yte

    def _eval(self, y_true, p, thr=0.5):
        y_pred = (p > thr).astype(int)
        return dict(
            accuracy=float(accuracy_score(y_true, y_pred)),
            precision=float(precision_score(y_true, y_pred, zero_division=0)),
            recall=float(recall_score(y_true, y_pred, zero_division=0)),
            f1=float(f1_score(y_true, y_pred, zero_division=0)),
            win_rate=float(np.mean(y_pred == y_true)),
        )

    def train_ensemble_temporal(self, X: pd.DataFrame, y: pd.Series, test_size=0.2):
        self.models.clear()
        if "timestamp" in X.columns:
            X = X.drop(columns=["timestamp"])
        if self.selected_features:
            X = X[self.selected_features]

        Xtr, ytr, Xva, yva, Xte, yte = self._split_ts(X, y, test_size=test_size)
        Xtr_s = self.scaler.fit_transform(Xtr)
        Xva_s = self.scaler.transform(Xva)
        Xte_s = self.scaler.transform(Xte)

        for i in range(self.config.ENSEMBLE_MODELS):
            params = self.config.LGBM_PARAMS.copy()
            params.update(dict(
                random_state=42+i,
                num_leaves=31 + i*8,
                learning_rate=max(0.01, 0.05 - i*0.005),
                feature_fraction=max(0.6, 0.9 - i*0.05),
                bagging_fraction=min(0.95, 0.8 + i*0.03),
                objective="binary", metric="binary_logloss"
            ))
            start = int(i * len(Xtr_s) * 0.1)
            dtr = lgb.Dataset(Xtr_s[start:], ytr.iloc[start:])
            dva = lgb.Dataset(Xva_s, yva, reference=dtr)
            model = lgb.train(
                params, dtr, valid_sets=[dva],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            self.models.append(model)
            print(f"  - 모델 {i+1} 학습 완료")

        tr_p = self.predict_proba_raw(Xtr_s)
        va_p = self.predict_proba_raw(Xva_s)
        te_p = self.predict_proba_raw(Xte_s)
        return dict(
            train=self._eval(ytr, tr_p),
            validation=self._eval(yva, va_p),
            test=self._eval(yte, te_p),
        )

    def predict_proba_raw(self, Xs: np.ndarray) -> np.ndarray:
        if not self.models:
            return np.zeros(Xs.shape[0])
        preds = np.vstack([m.predict(Xs, num_iteration=m.best_iteration) for m in self.models])
        w = np.array([0.15,0.15,0.2,0.25,0.25])[:preds.shape[0]]
        w = w / w.sum()
        return np.average(preds, axis=0, weights=w)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xc = X.copy()
        if "timestamp" in Xc.columns:
            Xc = Xc.drop(columns=["timestamp"])
        if self.selected_features:
            # 실시간 누락 피처가 있으면 0으로 채워도 OK (real_trade에서 이미 처리)
            missing = [c for c in self.selected_features if c not in Xc.columns]
            for c in missing:
                Xc[c] = 0.0
            Xc = Xc[self.selected_features]
        Xc = Xc.fillna(method="ffill").fillna(0).replace([np.inf, -np.inf], 0)
        Xs = self.scaler.transform(Xc)
        return self.predict_proba_raw(Xs)

    def predict(self, X: pd.DataFrame, threshold=0.5):
        return (self.predict_proba(X) > threshold).astype(int)

    # -------- I/O --------
    def save_model(self, filepath=None):
        if filepath is None:
            filepath = os.path.join(self.config.MODEL_DIR, "current_model.pkl")
        data = dict(
            models=self.models,
            scaler=self.scaler,
            selected_features=self.selected_features,
            feature_importance=self.feature_importance,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        joblib.dump(data, filepath)
        if self.feature_importance is not None:
            fi_path = os.path.join(
                self.config.FEATURE_LOG_DIR,
                f"feature_importance_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"
            )
            self.feature_importance.to_csv(fi_path, index=False)
        sf = os.path.join(self.config.FEATURE_LOG_DIR, "selected_features.json")
        with open(sf, "w", encoding="utf-8") as f:
            json.dump(
                {"features": self.selected_features or [], "count": len(self.selected_features or [])},
                f, indent=2
            )

    def load_model(self, filepath=None) -> bool:
        if filepath is None:
            filepath = os.path.join(self.config.MODEL_DIR, "current_model.pkl")
        if not os.path.exists(filepath):
            return False
        data = joblib.load(filepath)
        self.models = data["models"]
        self.scaler = data["scaler"]
        self.selected_features = data["selected_features"]
        self.feature_importance = data.get("feature_importance")
        return True


# ---------------- Optimizer (선택) ----------------
class ModelOptimizer:
    def __init__(self, config):
        self.config = config
        self.trainer = TimeSeriesModelTrainer(config)

    def initial_training(self, df: pd.DataFrame):
        print("="*50); print("초기 모델 학습 시작"); print("="*50)
        X, y = self.trainer._prepare_xy(df)
        print(f"유효 샘플: {len(X)}")
        self.trainer.feature_selection_temporal(X, y, top_k=50)
        metrics = self.trainer.train_ensemble_temporal(X, y)
        self.trainer.save_model()
        pd.DataFrame(metrics).T.to_csv(
            os.path.join(self.config.MODEL_DIR, f"training_results_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv")
        )
        return metrics

    def retrain_model(self, df: pd.DataFrame):
        print("\n모델 재학습 시작...")
        # old/new 비교 후 더 좋은 쪽 유지 로직은 필요 시 이전 버전 그대로 사용 가능
        new = TimeSeriesModelTrainer(self.config)
        X, y = new._prepare_xy(df)
        if new.selected_features is None:
            # 기존 모델 피처 재사용
            old = TimeSeriesModelTrainer(self.config)
            if old.load_model():
                new.selected_features = old.selected_features
        metrics_new = new.train_ensemble_temporal(X, y)
        new.save_model()  # 간단화(필요시 A/B 비교 추가)
        self.trainer = new
        return metrics_new

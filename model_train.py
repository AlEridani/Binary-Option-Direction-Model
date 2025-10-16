# model_train.py - 모델 학습 및 재학습 모듈 (시계열 기반, 미래 데이터 누출 방지)

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
from datetime import datetime, timedelta, timezone
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """피처 엔지니어링 클래스 - 미래 데이터 누출 방지"""
    
    @staticmethod
    def create_feature_pool(df, lookback_window=200):
        """
        전체 피처 풀 생성
        중요: 모든 피처는 과거 데이터만 사용 (shift 필수)
        lookback_window: 피처 계산에 필요한 최대 과거 데이터 수
        """
        features = pd.DataFrame(index=df.index)
        
        # 시간 정보 (UTC)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            features['timestamp'] = df['timestamp']
        
        # ===== 중요: 모든 가격 정보는 shift(1) 적용 =====
        # 현재 시점에서는 이전 캔들의 정보만 알 수 있음
        
        # 기본 가격 데이터 (이전 캔들)
        features['prev_open'] = df['open'].shift(1)
        features['prev_high'] = df['high'].shift(1)
        features['prev_low'] = df['low'].shift(1)
        features['prev_close'] = df['close'].shift(1)
        features['prev_volume'] = df['volume'].shift(1)
        
        # 가격 변화율 (모두 과거 데이터 기준)
        for period in [1, 2, 3, 5, 10, 15, 30, 60]:
            # shift(1)을 먼저 적용 후 변화율 계산
            features[f'return_{period}'] = (df['close'].shift(1) / df['close'].shift(period + 1) - 1)
            features[f'volume_change_{period}'] = (df['volume'].shift(1) / df['volume'].shift(period + 1) - 1)
        
        # 이동평균 (MA) - 과거 데이터만 사용
        for period in [5, 10, 20, 50, 100, 200]:
            # 현재 캔들을 제외한 과거 period개의 평균
            ma = df['close'].shift(1).rolling(window=period, min_periods=period).mean()
            features[f'ma_{period}'] = ma
            features[f'price_to_ma_{period}'] = (df['close'].shift(1) / ma - 1)
            
            # 이동평균 기울기 (momentum)
            features[f'ma_{period}_slope'] = ma.diff(5) / ma.shift(5)
        
        # 지수이동평균 (EMA) - 과거 데이터만
        for period in [12, 26, 50]:
            ema = df['close'].shift(1).ewm(span=period, adjust=False, min_periods=period).mean()
            features[f'ema_{period}'] = ema
            features[f'price_to_ema_{period}'] = (df['close'].shift(1) / ema - 1)
        
        # 볼린저 밴드 - 과거 데이터만
        for period in [20, 50]:
            # shift를 먼저 적용
            shifted_close = df['close'].shift(1)
            ma = shifted_close.rolling(window=period, min_periods=period).mean()
            std = shifted_close.rolling(window=period, min_periods=period).std()
            
            features[f'bb_upper_{period}'] = ma + (std * 2)
            features[f'bb_lower_{period}'] = ma - (std * 2)
            features[f'bb_width_{period}'] = (std * 4) / ma  # 정규화된 밴드 폭
            features[f'bb_position_{period}'] = (shifted_close - features[f'bb_lower_{period}']) / (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'] + 0.0001)
        
        # RSI - 과거 데이터만
        for period in [14, 28]:
            # shift를 먼저 적용
            shifted_close = df['close'].shift(1)
            delta = shifted_close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
            rs = gain / (loss + 0.0001)
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD - 과거 데이터만
        shifted_close = df['close'].shift(1)
        ema_12 = shifted_close.ewm(span=12, adjust=False, min_periods=12).mean()
        ema_26 = shifted_close.ewm(span=26, adjust=False, min_periods=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
        
        features['macd'] = macd_line
        features['macd_signal'] = signal_line
        features['macd_histogram'] = macd_line - signal_line
        
        # Stochastic Oscillator - 과거 데이터만
        for period in [14]:
            # shift를 먼저 적용
            shifted_high = df['high'].shift(1)
            shifted_low = df['low'].shift(1)
            shifted_close = df['close'].shift(1)
            
            low_min = shifted_low.rolling(window=period, min_periods=period).min()
            high_max = shifted_high.rolling(window=period, min_periods=period).max()
            features[f'stoch_{period}'] = ((shifted_close - low_min) / (high_max - low_min + 0.0001)) * 100
        
        # ATR (Average True Range) - 과거 데이터만
        for period in [14, 28]:
            # shift를 먼저 적용
            shifted_high = df['high'].shift(1)
            shifted_low = df['low'].shift(1)
            shifted_close = df['close'].shift(1)
            
            high_low = shifted_high - shifted_low
            high_close = np.abs(shifted_high - shifted_close.shift(1))
            low_close = np.abs(shifted_low - shifted_close.shift(1))
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            features[f'atr_{period}'] = true_range.rolling(window=period, min_periods=period).mean()
            features[f'atr_ratio_{period}'] = features[f'atr_{period}'] / shifted_close
        
        # 거래량 지표 - 과거 데이터만
        shifted_volume = df['volume'].shift(1)
        features['volume_sma_10'] = shifted_volume.rolling(window=10, min_periods=10).mean()
        features['volume_sma_50'] = shifted_volume.rolling(window=50, min_periods=50).mean()
        features['volume_ratio'] = shifted_volume / features['volume_sma_10']
        features['volume_trend'] = features['volume_sma_10'] / features['volume_sma_50']
        
        # OBV (On Balance Volume) - 과거 데이터만
        price_diff = df['close'].diff().shift(1)  # 이전 캔들의 가격 변화
        obv = (np.sign(price_diff) * df['volume'].shift(1)).cumsum()
        features['obv'] = obv
        features['obv_ema'] = obv.ewm(span=20, adjust=False, min_periods=20).mean()
        features['obv_signal'] = obv / features['obv_ema'] - 1
        
        # 캔들 패턴 - 이전 캔들 기준
        prev_open = df['open'].shift(1)
        prev_close = df['close'].shift(1)
        prev_high = df['high'].shift(1)
        prev_low = df['low'].shift(1)
        
        features['body_size'] = np.abs(prev_close - prev_open) / prev_open
        features['upper_shadow'] = (prev_high - pd.concat([prev_open, prev_close], axis=1).max(axis=1)) / prev_open
        features['lower_shadow'] = (pd.concat([prev_open, prev_close], axis=1).min(axis=1) - prev_low) / prev_open
        features['body_position'] = (prev_close - prev_open) / (prev_high - prev_low + 0.0001)
        
        # 다중 캔들 패턴 (과거 3개 캔들)
        for i in range(1, 4):
            features[f'candle_direction_{i}'] = np.sign(df['close'].shift(i) - df['open'].shift(i))
            features[f'candle_size_{i}'] = (df['high'].shift(i) - df['low'].shift(i)) / df['close'].shift(i)
        
        # 시간 피처 (UTC 기준)
        if 'timestamp' in df.columns:
            dt = pd.to_datetime(df['timestamp'], utc=True)
            features['hour'] = dt.dt.hour
            features['minute'] = dt.dt.minute
            features['day_of_week'] = dt.dt.dayofweek
            features['day_of_month'] = dt.dt.day
            
            # 순환 인코딩
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
            features['dow_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
            features['dow_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        # 마이크로 구조 피처 - 과거 데이터만
        for period in [3, 5, 10]:
            shifted_high = df['high'].shift(1)
            shifted_low = df['low'].shift(1)
            shifted_close = df['close'].shift(1)
            
            features[f'high_low_ratio_{period}'] = (
                (shifted_high / shifted_low - 1).rolling(window=period, min_periods=period).mean()
            )
            features[f'close_position_{period}'] = (
                ((shifted_close - shifted_low) / (shifted_high - shifted_low + 0.0001))
                .rolling(window=period, min_periods=period).mean()
            )
        
        # 추세 강도 지표 - 과거 데이터만
        for period in [10, 20, 50]:
            shifted_close = df['close'].shift(1)
            ma = shifted_close.rolling(window=period, min_periods=period).mean()
            
            # 가격이 이동평균 위에 있던 비율
            above_ma = (shifted_close > ma).astype(int)
            features[f'trend_strength_{period}'] = above_ma.rolling(window=period, min_periods=period).mean()
        
        # 변동성 지표 - 과거 데이터만
        for period in [10, 30]:
            shifted_returns = df['close'].shift(1).pct_change()
            features[f'volatility_{period}'] = shifted_returns.rolling(window=period, min_periods=period).std()
            features[f'volatility_ratio_{period}'] = (
                features[f'volatility_{period}'] / 
                shifted_returns.rolling(window=period*3, min_periods=period*3).std()
            )
        
        # NaN 처리 - 첫 lookback_window 행은 제거
        features = features.iloc[lookback_window:]
        
        return features
    
    @staticmethod
    def create_target(df, window=10):
        """
        타겟 변수 생성: window분 후 가격이 올랐는지(1) 내렸는지(0)
        중요: 미래 데이터를 정확히 참조
        """
        current_close = df['close']
        future_close = df['close'].shift(-window)  # window분 후의 종가
        
        # 가격이 올랐으면 1, 내렸으면 0
        target = (future_close > current_close).astype(int)
        
        return target
    
    @staticmethod
    def validate_no_future_leak(features, target):
        """
        미래 데이터 누출 검증
        """
        # 피처에 미래 정보가 포함되어 있는지 확인
        issues = []
        
        # shift가 제대로 적용되었는지 확인
        for col in features.columns:
            if 'timestamp' in col:
                continue
                
            # 피처와 타겟의 상관관계가 너무 높으면 의심
            if target.notna().sum() > 0:
                corr = features[col].corr(target)
                if abs(corr) > 0.95:  # 상관관계가 너무 높음
                    issues.append(f"Suspicious correlation in {col}: {corr:.3f}")
        
        if issues:
            print("⚠️ 미래 데이터 누출 가능성 발견:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        
        return True


class TimeSeriesModelTrainer:
    """시계열 기반 모델 학습 클래스"""
    
    def __init__(self, config):
        self.config = config
        self.models = []
        self.feature_importance = None
        self.selected_features = None
        self.scaler = StandardScaler()
        
    def time_series_split(self, X, y, n_splits=5, test_size=0.2):
        """
        시계열 데이터를 위한 Walk-Forward 분할
        미래 데이터가 학습에 사용되지 않도록 보장
        """
        total_size = len(X)
        test_samples = int(total_size * test_size)
        
        splits = []
        for i in range(n_splits):
            # 각 분할마다 학습 데이터 크기 증가
            train_end = int(total_size * (0.3 + 0.1 * i))
            val_start = train_end
            val_end = min(train_end + test_samples, total_size)
            
            if val_end <= total_size:
                train_idx = list(range(train_end))
                val_idx = list(range(val_start, val_end))
                splits.append((train_idx, val_idx))
        
        return splits
    
    def feature_selection_temporal(self, X, y, top_k=50):
        """
        시계열 고려한 피처 선택
        """
        # 시간 순서 보장
        train_size = int(len(X) * 0.7)
        X_train = X.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_val = X.iloc[train_size:]
        y_val = y.iloc[train_size:]
        
        # timestamp 제외
        feature_cols = [col for col in X.columns if col != 'timestamp']
        X_train = X_train[feature_cols]
        X_val = X_val[feature_cols]
        
        # NaN 처리 (forward fill만 사용 - 미래 데이터 방지)
        X_train = X_train.fillna(method='ffill').fillna(0)
        X_val = X_val.fillna(method='ffill').fillna(0)
        
        # 무한값 처리
        X_train = X_train.replace([np.inf, -np.inf], 0)
        X_val = X_val.replace([np.inf, -np.inf], 0)
        
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        
        # 시계열에 적합한 파라미터
        params = self.config.LGBM_PARAMS.copy()
        params['objective'] = 'binary'
        params['metric'] = 'binary_logloss'
        params['boosting_type'] = 'gbdt'
        
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        # 피처 중요도 추출
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance
        self.selected_features = importance.head(top_k)['feature'].tolist()
        
        print(f"\n선택된 상위 10개 피처:")
        for idx, row in importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.2f}")
        
        return self.selected_features
    
    def train_ensemble_temporal(self, X, y, test_size=0.2):
        """
        시계열 앙상블 모델 학습
        """
        self.models = []
        
        # timestamp 제외
        if 'timestamp' in X.columns:
            X = X.drop('timestamp', axis=1)
        
        X_selected = X[self.selected_features]
        
        # 시계열 순서 유지한 분할
        total_size = len(X_selected)
        train_size = int(total_size * (1 - test_size * 2))
        val_size = int(total_size * test_size)
        
        X_train = X_selected.iloc[:train_size]
        y_train = y.iloc[:train_size]
        X_val = X_selected.iloc[train_size:train_size + val_size]
        y_val = y.iloc[train_size:train_size + val_size]
        X_test = X_selected.iloc[train_size + val_size:]
        y_test = y.iloc[train_size + val_size:]
        
        print(f"\n데이터 분할 (시간 순서 유지):")
        print(f"  학습: {len(X_train)} 샘플")
        print(f"  검증: {len(X_val)} 샘플")
        print(f"  테스트: {len(X_test)} 샘플")
        
        # 전처리
        X_train = X_train.fillna(method='ffill').fillna(0).replace([np.inf, -np.inf], 0)
        X_val = X_val.fillna(method='ffill').fillna(0).replace([np.inf, -np.inf], 0)
        X_test = X_test.fillna(method='ffill').fillna(0).replace([np.inf, -np.inf], 0)
        
        # 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Walk-Forward Validation으로 앙상블 학습
        for i in range(self.config.ENSEMBLE_MODELS):
            # 각 모델마다 다른 파라미터와 다른 학습 구간 사용
            params = self.config.LGBM_PARAMS.copy()
            params['random_state'] = 42 + i
            params['num_leaves'] = 31 + i * 10
            params['learning_rate'] = 0.05 - i * 0.005
            params['feature_fraction'] = 0.9 - i * 0.05
            params['bagging_fraction'] = 0.8 + i * 0.03
            
            # 학습 데이터의 다른 부분 사용 (시간 다양성)
            start_idx = int(i * len(X_train_scaled) * 0.1)
            end_idx = len(X_train_scaled)
            
            X_train_subset = X_train_scaled[start_idx:end_idx]
            y_train_subset = y_train.iloc[start_idx:end_idx]
            
            lgb_train = lgb.Dataset(X_train_subset, y_train_subset)
            lgb_val = lgb.Dataset(X_val_scaled, y_val, reference=lgb_train)
            
            model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            self.models.append(model)
            print(f"  모델 {i+1} 학습 완료")
        
        # 앙상블 성능 평가
        train_pred = self.predict_proba(X_train)
        val_pred = self.predict_proba(X_val)
        test_pred = self.predict_proba(X_test)
        
        metrics = {
            'train': self.evaluate_predictions(y_train, train_pred),
            'validation': self.evaluate_predictions(y_val, val_pred),
            'test': self.evaluate_predictions(y_test, test_pred)
        }
        
        return metrics
    
    def predict_proba(self, X):
        """앙상블 예측 (확률)"""
        if 'timestamp' in X.columns:
            X = X.drop('timestamp', axis=1)
            
        X_selected = X[self.selected_features]
        X_selected = X_selected.fillna(method='ffill').fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.transform(X_selected)
        
        predictions = []
        for model in self.models:
            pred = model.predict(X_scaled, num_iteration=model.best_iteration)
            predictions.append(pred)
        
        # 가중 평균 (최근 모델에 더 높은 가중치)
        weights = np.array([0.15, 0.15, 0.2, 0.25, 0.25])[:len(self.models)]
        weights = weights / weights.sum()
        
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred
    
    def predict(self, X, threshold=0.5):
        """이진 분류 예측"""
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)
    
    def evaluate_predictions(self, y_true, y_pred_proba, threshold=0.5):
        """예측 성능 평가"""
        y_pred = (y_pred_proba > threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'win_rate': np.mean(y_pred == y_true)
        }
        
        return metrics
    
    def save_model(self, filepath=None):
        """모델 저장"""
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
        
        # 피처 정보 CSV로 저장
        if self.feature_importance is not None:
            feature_csv_path = os.path.join(self.config.FEATURE_LOG_DIR, 
                                           f'feature_importance_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.csv')
            self.feature_importance.to_csv(feature_csv_path, index=False)
        
        # 선택된 피처 목록 저장
        selected_features_path = os.path.join(self.config.FEATURE_LOG_DIR, 'selected_features.json')
        with open(selected_features_path, 'w') as f:
            json.dump({
                'features': self.selected_features,
                'count': len(self.selected_features),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }, f, indent=2)
    
    def load_model(self, filepath=None):
        """모델 로드"""
        if filepath is None:
            filepath = os.path.join(self.config.MODEL_DIR, 'current_model.pkl')
        
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.scaler = model_data['scaler']
            self.selected_features = model_data['selected_features']
            self.feature_importance = model_data.get('feature_importance')
            return True
        return False


class ModelOptimizer:
    """모델 최적화 및 재학습 관리"""
    
    def __init__(self, config):
        self.config = config
        self.trainer = TimeSeriesModelTrainer(config)
        self.performance_history = []
        
    def initial_training(self, df):
        """
        초기 모델 학습 (시계열 기반)
        """
        print("=" * 50)
        print("초기 모델 학습 시작 (시계열 기반)")
        print("=" * 50)
        
        # UTC 시간 확인
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            print(f"데이터 기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
        
        # 피처 생성
        print("\n피처 생성 중...")
        feature_engineer = FeatureEngineer()
        features = feature_engineer.create_feature_pool(df, lookback_window=200)
        target = feature_engineer.create_target(df, window=self.config.PREDICTION_WINDOW)
        
        # 유효한 데이터만 사용
        valid_idx = target.notna() & (target.index >= 200)  # lookback_window 이후 데이터
        features = features[valid_idx]
        target = target[valid_idx]
        
        print(f"유효 데이터: {len(features)} 샘플")
        
        # 미래 데이터 누출 검증
        print("\n미래 데이터 누출 검증 중...")
        if not feature_engineer.validate_no_future_leak(features, target):
            print("⚠️ 경고: 미래 데이터 누출 가능성이 감지되었습니다!")
        else:
            print("✓ 미래 데이터 누출 없음")
        
        # 피처 선택 (시계열 기반)
        print("\n피처 선택 중...")
        selected_features = self.trainer.feature_selection_temporal(features, target, top_k=50)
        print(f"선택된 피처 수: {len(selected_features)}")
        
        # 앙상블 학습 (시계열 기반)
        print("\n앙상블 모델 학습 중...")
        metrics = self.trainer.train_ensemble_temporal(features, target)
        
        print("\n학습 결과:")
        for split, metric in metrics.items():
            print(f"\n{split.upper()}:")
            for key, value in metric.items():
                print(f"  {key}: {value:.4f}")
        
        # 모델 저장
        self.trainer.save_model()
        print("\n모델 저장 완료")
        
        # 학습 결과 CSV 저장
        results_df = pd.DataFrame(metrics).T
        results_df.to_csv(os.path.join(self.config.MODEL_DIR, 
                          f'training_results_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.csv'))
        
        return metrics
    
    def retrain_model(self, new_data_df):
        """
        새로운 데이터로 모델 재학습 (시계열 기반)
        """
        print("\n모델 재학습 시작...")
        
        # UTC 시간 변환
        if 'timestamp' in new_data_df.columns:
            new_data_df['timestamp'] = pd.to_datetime(new_data_df['timestamp'], utc=True)
        
        # 피처 생성
        feature_engineer = FeatureEngineer()
        features = feature_engineer.create_feature_pool(new_data_df, lookback_window=200)
        target = feature_engineer.create_target(new_data_df, window=self.config.PREDICTION_WINDOW)
        
        # 유효한 데이터만 사용
        valid_idx = target.notna() & (target.index >= 200)
        features = features[valid_idx]
        target = target[valid_idx]
        
        # 기존 모델 로드
        old_model = TimeSeriesModelTrainer(self.config)
        old_model.load_model()
        
        # 새 모델 학습
        new_model = TimeSeriesModelTrainer(self.config)
        new_model.selected_features = old_model.selected_features  # 같은 피처 사용
        
        # 시계열 순서 유지한 데이터 분할
        train_size = int(len(features) * 0.7)
        val_size = int(len(features) * 0.2)
        
        # 새 모델 학습
        new_metrics = new_model.train_ensemble_temporal(features, target)
        
        # 마지막 10% 데이터로 두 모델 비교
        test_start = train_size + val_size
        X_test = features.iloc[test_start:]
        y_test = target.iloc[test_start:]
        
        if len(X_test) > 0:
            old_pred = old_model.predict(X_test)
            new_pred = new_model.predict(X_test)
            
            old_accuracy = accuracy_score(y_test, old_pred)
            new_accuracy = accuracy_score(y_test, new_pred)
            
            print(f"\n모델 비교:")
            print(f"기존 모델 정확도: {old_accuracy:.4f}")
            print(f"새 모델 정확도: {new_accuracy:.4f}")
            
            # 새 모델이 더 좋으면 교체
            if new_accuracy > old_accuracy:
                print("새 모델이 더 우수함. 모델 교체...")
                # 기존 모델 백업
                backup_path = os.path.join(self.config.MODEL_DIR, 
                                          f'backup/model_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.pkl')
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                old_model.save_model(backup_path)
                
                # 새 모델 저장
                new_model.save_model()
                self.trainer = new_model
                print("모델 교체 완료")
            else:
                print("기존 모델 유지")
                self.trainer = old_model
        
        return new_metrics
    
    def analyze_failures(self, trade_log_df):
        """
        실패 거래 분석 및 필터 생성
        """
        if 'result' not in trade_log_df.columns:
            return {}
            
        failures = trade_log_df[trade_log_df['result'] == 0].copy()
        
        if len(failures) == 0:
            return {}
        
        # 실패 패턴 분석
        failure_patterns = {
            'high_volatility': {},
            'low_volume': {},
            'time_based': {},
            'technical_signals': {}
        }
        
        # 변동성 기반 필터
        if 'atr_14' in failures.columns:
            high_vol_threshold = failures['atr_14'].quantile(0.75)
            failure_patterns['high_volatility'] = {
                'atr_14_threshold': float(high_vol_threshold),
                'filter': f"atr_14 > {high_vol_threshold}"
            }
        
        # 거래량 기반 필터
        if 'volume_ratio' in failures.columns:
            low_vol_threshold = failures['volume_ratio'].quantile(0.25)
            failure_patterns['low_volume'] = {
                'volume_ratio_threshold': float(low_vol_threshold),
                'filter': f"volume_ratio < {low_vol_threshold}"
            }
        
        # 시간대 기반 필터 (UTC 기준)
        if 'entry_time' in failures.columns:
            failures['hour'] = pd.to_datetime(failures['entry_time'], utc=True).dt.hour
            hour_failure_rate = failures.groupby('hour').size()
            if len(hour_failure_rate) > 0:
                high_failure_hours = hour_failure_rate[
                    hour_failure_rate > hour_failure_rate.mean() + hour_failure_rate.std()
                ].index.tolist()
                
                if high_failure_hours:
                    failure_patterns['time_based'] = {
                        'avoid_hours': high_failure_hours,
                        'filter': f"hour not in {high_failure_hours}"
                    }
        
        # RSI 극단값 필터
        if 'rsi_14' in failures.columns:
            rsi_failures = failures[(failures['rsi_14'] < 30) | (failures['rsi_14'] > 70)]
            if len(rsi_failures) > len(failures) * 0.3:  # 30% 이상이 RSI 극단값에서 실패
                failure_patterns['technical_signals']['rsi_extreme'] = {
                    'condition': "RSI < 30 or RSI > 70",
                    'failure_rate': len(rsi_failures) / len(failures)
                }
        
        # 결과를 CSV로 저장
        analysis_path = os.path.join(self.config.TRADE_LOG_DIR, 
                                    f'failure_analysis_{datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")}.csv')
        failures.to_csv(analysis_path, index=False)
        
        return failure_patterns


# 사용 예시
if __name__ == "__main__":
    from config import Config
    
    # 설정 초기화
    Config.create_directories()
    
    # 시뮬레이션 데이터 생성
    print("시뮬레이션 데이터 생성 중...")
    
    # UTC 시간으로 데이터 생성
    periods = 500
    timestamps = pd.date_range(end=datetime.now(timezone.utc), periods=periods, freq='1min', tz='UTC')
    
    np.random.seed(42)
    prices = 42000 + np.random.randn(periods).cumsum() * 100
    
    data = []
    for i, ts in enumerate(timestamps):
        base = prices[i]
        o = base + np.random.uniform(-50, 50)
        c = base + np.random.uniform(-50, 50)
        h = max(o, c) + np.random.uniform(0, 100)
        l = min(o, c) - np.random.uniform(0, 100)
        v = np.random.uniform(100, 1000)
        
        data.append({
            'timestamp': ts,
            'open': o,
            'high': h,
            'low': l,
            'close': c,
            'volume': v
        })
    
    df = pd.DataFrame(data)
    
    # CSV로 저장
    df.to_csv('sample_data.csv', index=False)
    print(f"샘플 데이터 저장 완료: sample_data.csv")
    
    # 모델 학습 테스트
    optimizer = ModelOptimizer(Config)
    metrics = optimizer.initial_training(df)
    
    print("\n테스트 완료!")
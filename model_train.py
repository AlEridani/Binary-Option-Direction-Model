# model_train.py - 모델 학습 및 재학습 모듈

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """피처 엔지니어링 클래스"""
    
    @staticmethod
    def create_feature_pool(df):
        """
        전체 피처 풀 생성
        미래 데이터 사용 방지를 위해 shift 활용
        """
        features = pd.DataFrame(index=df.index)
        
        # 기본 가격 데이터 (현재 캔들 기준)
        features['open'] = df['open']
        features['high'] = df['high'] 
        features['low'] = df['low']
        features['close'] = df['close']
        features['volume'] = df['volume']
        
        # 가격 변화율 (이전 캔들 대비)
        for period in [1, 3, 5, 10, 15, 30]:
            features[f'return_{period}'] = df['close'].pct_change(period).shift(1)
            features[f'volume_change_{period}'] = df['volume'].pct_change(period).shift(1)
        
        # 이동평균 (MA)
        for period in [5, 10, 20, 50, 100, 200]:
            ma = df['close'].rolling(window=period).mean().shift(1)
            features[f'ma_{period}'] = ma
            features[f'price_to_ma_{period}'] = (df['close'] / ma - 1).shift(1)
        
        # 지수이동평균 (EMA)
        for period in [12, 26, 50]:
            ema = df['close'].ewm(span=period, adjust=False).mean().shift(1)
            features[f'ema_{period}'] = ema
            features[f'price_to_ema_{period}'] = (df['close'] / ema - 1).shift(1)
        
        # 볼린저 밴드
        for period in [20, 50]:
            ma = df['close'].rolling(window=period).mean().shift(1)
            std = df['close'].rolling(window=period).std().shift(1)
            features[f'bb_upper_{period}'] = ma + (std * 2)
            features[f'bb_lower_{period}'] = ma - (std * 2)
            features[f'bb_width_{period}'] = features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']
            features[f'bb_position_{period}'] = ((df['close'] - features[f'bb_lower_{period}']) / 
                                                  features[f'bb_width_{period}']).shift(1)
        
        # RSI (Relative Strength Index)
        for period in [14, 28]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            features[f'rsi_{period}'] = (100 - (100 / (1 + rs))).shift(1)
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        features['macd'] = macd_line.shift(1)
        features['macd_signal'] = signal_line.shift(1)
        features['macd_histogram'] = (macd_line - signal_line).shift(1)
        
        # Stochastic Oscillator
        for period in [14]:
            low_min = df['low'].rolling(window=period).min()
            high_max = df['high'].rolling(window=period).max()
            features[f'stoch_{period}'] = (((df['close'] - low_min) / 
                                           (high_max - low_min)) * 100).shift(1)
        
        # ATR (Average True Range) - 변동성 지표
        for period in [14, 28]:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift(1))
            low_close = np.abs(df['low'] - df['close'].shift(1))
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            features[f'atr_{period}'] = true_range.rolling(window=period).mean().shift(1)
        
        # 거래량 지표
        features['volume_sma_10'] = df['volume'].rolling(window=10).mean().shift(1)
        features['volume_ratio'] = (df['volume'] / features['volume_sma_10']).shift(1)
        
        # OBV (On Balance Volume)
        obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        features['obv'] = obv.shift(1)
        features['obv_change'] = obv.pct_change(10).shift(1)
        
        # 캔들 패턴
        features['body_size'] = (df['close'] - df['open']).abs().shift(1)
        features['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)).shift(1)
        features['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']).shift(1)
        features['body_to_shadow'] = (features['body_size'] / 
                                     (features['upper_shadow'] + features['lower_shadow'] + 0.0001)).shift(1)
        
        # 시간 피처
        if 'timestamp' in df.columns:
            dt = pd.to_datetime(df['timestamp'])
            features['hour'] = dt.dt.hour
            features['minute'] = dt.dt.minute
            features['day_of_week'] = dt.dt.dayofweek
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        
        # 마이크로 구조 피처 (고빈도 거래 패턴)
        for period in [3, 5, 10]:
            features[f'high_low_ratio_{period}'] = ((df['high'] / df['low'] - 1) * 100).rolling(window=period).mean().shift(1)
            features[f'close_position_{period}'] = ((df['close'] - df['low']) / 
                                                   (df['high'] - df['low'] + 0.0001)).rolling(window=period).mean().shift(1)
        
        return features
    
    @staticmethod
    def create_target(df, window=10):
        """
        타겟 변수 생성: window분 후 가격이 올랐는지(1) 내렸는지(0)
        """
        future_price = df['close'].shift(-window)
        target = (future_price > df['close']).astype(int)
        return target


class ModelTrainer:
    """모델 학습 클래스"""
    
    def __init__(self, config):
        self.config = config
        self.models = []
        self.feature_importance = None
        self.selected_features = None
        self.scaler = StandardScaler()
        
    def feature_selection(self, X, y, top_k=50):
        """
        피처 중요도 기반 피처 선택
        """
        # 초기 모델로 피처 중요도 파악
        train_idx = int(len(X) * 0.8)
        X_train, X_val = X[:train_idx], X[train_idx:]
        y_train, y_val = y[:train_idx], y[train_idx:]
        
        # NaN 처리
        X_train = X_train.fillna(method='ffill').fillna(0)
        X_val = X_val.fillna(method='ffill').fillna(0)
        
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
        
        model = lgb.train(
            self.config.LGBM_PARAMS,
            lgb_train,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        # 피처 중요도 추출
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance
        self.selected_features = importance.head(top_k)['feature'].tolist()
        
        return self.selected_features
    
    def train_ensemble(self, X, y):
        """
        앙상블 모델 학습
        """
        self.models = []
        X_selected = X[self.selected_features]
        
        # 데이터 분할 (시계열 고려)
        train_size = int(len(X) * self.config.TRAIN_RATIO)
        val_size = int(len(X) * self.config.VAL_RATIO)
        
        X_train = X_selected[:train_size]
        y_train = y[:train_size]
        X_val = X_selected[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X_selected[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        # NaN 처리 및 스케일링
        X_train = X_train.fillna(method='ffill').fillna(0)
        X_val = X_val.fillna(method='ffill').fillna(0)
        X_test = X_test.fillna(method='ffill').fillna(0)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 앙상블 모델 학습
        for i in range(self.config.ENSEMBLE_MODELS):
            # 각 모델마다 다른 파라미터 사용
            params = self.config.LGBM_PARAMS.copy()
            params['random_state'] = 42 + i
            params['num_leaves'] = 31 + i * 5
            params['learning_rate'] = 0.05 - i * 0.005
            
            lgb_train = lgb.Dataset(X_train_scaled, y_train)
            lgb_val = lgb.Dataset(X_val_scaled, y_val, reference=lgb_train)
            
            model = lgb.train(
                params,
                lgb_train,
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            self.models.append(model)
        
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
        """
        앙상블 예측 (확률)
        """
        X_selected = X[self.selected_features]
        X_selected = X_selected.fillna(method='ffill').fillna(0)
        X_scaled = self.scaler.transform(X_selected)
        
        predictions = []
        for model in self.models:
            pred = model.predict(X_scaled, num_iteration=model.best_iteration)
            predictions.append(pred)
        
        # 앙상블 평균
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def predict(self, X, threshold=0.5):
        """
        이진 분류 예측
        """
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)
    
    def evaluate_predictions(self, y_true, y_pred_proba, threshold=0.5):
        """
        예측 성능 평가
        """
        y_pred = (y_pred_proba > threshold).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'win_rate': np.mean(y_pred == y_true)
        }
        
        return metrics
    
    def save_model(self, filepath=None):
        """
        모델 저장
        """
        if filepath is None:
            filepath = os.path.join(self.config.MODEL_DIR, 'current_model.pkl')
        
        model_data = {
            'models': self.models,
            'scaler': self.scaler,
            'selected_features': self.selected_features,
            'feature_importance': self.feature_importance,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        
        # 피처 정보 저장
        feature_log_path = os.path.join(self.config.FEATURE_LOG_DIR, 'selected_features.json')
        with open(feature_log_path, 'w') as f:
            json.dump({
                'features': self.selected_features,
                'importance': self.feature_importance.to_dict() if self.feature_importance is not None else None
            }, f, indent=2)
    
    def load_model(self, filepath=None):
        """
        모델 로드
        """
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
        self.trainer = ModelTrainer(config)
        self.performance_history = []
        
    def initial_training(self, df):
        """
        초기 모델 학습 (8개월 학습, 1개월 검증)
        """
        print("=" * 50)
        print("초기 모델 학습 시작")
        print("=" * 50)
        
        # 피처 생성
        feature_engineer = FeatureEngineer()
        features = feature_engineer.create_feature_pool(df)
        target = feature_engineer.create_target(df, window=self.config.PREDICTION_WINDOW)
        
        # 유효한 데이터만 사용 (NaN 제거)
        valid_idx = target.notna()
        features = features[valid_idx]
        target = target[valid_idx]
        
        # 시간 기준 분할 (8개월 학습, 1개월 검증)
        total_months = 9
        train_months = 8
        train_size = int(len(features) * (train_months / total_months))
        
        X_train = features[:train_size]
        y_train = target[:train_size]
        X_test = features[train_size:]
        y_test = target[train_size:]
        
        print(f"학습 데이터: {len(X_train)} 샘플")
        print(f"검증 데이터: {len(X_test)} 샘플")
        
        # 피처 선택
        print("\n피처 선택 중...")
        selected_features = self.trainer.feature_selection(X_train, y_train, top_k=50)
        print(f"선택된 피처 수: {len(selected_features)}")
        print(f"상위 10개 피처: {selected_features[:10]}")
        
        # 앙상블 학습
        print("\n앙상블 모델 학습 중...")
        metrics = self.trainer.train_ensemble(features, target)
        
        print("\n학습 결과:")
        for split, metric in metrics.items():
            print(f"\n{split.upper()}:")
            for key, value in metric.items():
                print(f"  {key}: {value:.4f}")
        
        # 모델 저장
        self.trainer.save_model()
        print("\n모델 저장 완료")
        
        return metrics
    
    def retrain_model(self, new_data_df):
        """
        새로운 데이터로 모델 재학습
        """
        print("\n모델 재학습 시작...")
        
        # 피처 생성
        feature_engineer = FeatureEngineer()
        features = feature_engineer.create_feature_pool(new_data_df)
        target = feature_engineer.create_target(new_data_df, window=self.config.PREDICTION_WINDOW)
        
        # 유효한 데이터만 사용
        valid_idx = target.notna()
        features = features[valid_idx]
        target = target[valid_idx]
        
        # 기존 모델 로드
        old_model = ModelTrainer(self.config)
        old_model.load_model()
        
        # 새 모델 학습
        new_model = ModelTrainer(self.config)
        new_model.selected_features = old_model.selected_features  # 같은 피처 사용
        
        # 데이터 분할 (70% 학습, 20% 검증, 10% 테스트)
        train_size = int(len(features) * 0.7)
        val_size = int(len(features) * 0.2)
        
        X_train = features[:train_size]
        y_train = target[:train_size]
        X_val = features[train_size:train_size+val_size]
        y_val = target[train_size:train_size+val_size]
        X_test = features[train_size+val_size:]
        y_test = target[train_size+val_size:]
        
        # 새 모델 학습
        new_metrics = new_model.train_ensemble(features, target)
        
        # 테스트 데이터에서 두 모델 비교
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
            backup_path = os.path.join(self.config.MODEL_DIR, f'backup/model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
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
        failures = trade_log_df[trade_log_df['result'] == 0]
        
        if len(failures) == 0:
            return {}
        
        # 실패 패턴 분석
        failure_patterns = {
            'high_volatility': [],
            'low_volume': [],
            'time_based': [],
            'technical_signals': []
        }
        
        # 변동성 기반 필터
        if 'atr_14' in failures.columns:
            high_vol_threshold = failures['atr_14'].quantile(0.75)
            failure_patterns['high_volatility'] = {
                'atr_14_threshold': high_vol_threshold,
                'filter': f"atr_14 > {high_vol_threshold}"
            }
        
        # 거래량 기반 필터
        if 'volume_ratio' in failures.columns:
            low_vol_threshold = failures['volume_ratio'].quantile(0.25)
            failure_patterns['low_volume'] = {
                'volume_ratio_threshold': low_vol_threshold,
                'filter': f"volume_ratio < {low_vol_threshold}"
            }
        
        # 시간대 기반 필터
        if 'hour' in failures.columns:
            hour_failure_rate = failures.groupby('hour').size()
            high_failure_hours = hour_failure_rate[hour_failure_rate > hour_failure_rate.mean() + hour_failure_rate.std()].index.tolist()
            failure_patterns['time_based'] = {
                'avoid_hours': high_failure_hours,
                'filter': f"hour not in {high_failure_hours}"
            }
        
        return failure_patterns


# 사용 예시
if __name__ == "__main__":
    from config import Config
    
    # 설정 초기화
    Config.create_directories()
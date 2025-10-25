"""
model_trainer.py
모델 학습/저장/로드 (30분봉 시스템)
- LightGBM 기반 학습
- 시계열 CV
- 앙상블 (3개 모델)
- 피처 선택 (Top K)
- 캘리브레이션 (Isotonic/Platt)
- 레짐별 모델 분리 학습
- 모델 버전 관리
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import pickle
import json
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, log_loss, classification_report
)

from config import Config


class ModelTrainer:
    """모델 학습 및 관리 (레짐별 분리 학습 지원)"""
    
    def __init__(self):
        self.config = Config
        self.models = {}  # {'overall': [...], 'regime_1': [...], 'regime_-1': [...], 'regime_0': [...]}
        self.feature_importance = {}
        self.selected_features = None
        self.calibrators = {}
        self.metadata = {}
        
        # 버전 정보
        self.model_version = self.config.get_version_string()
        self.feature_version = self.config.get_version_string()
    
    # ==========================================
    # 피처 선택
    # ==========================================
    
    def select_features(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        top_k: Optional[int] = None
    ) -> List[str]:
        """
        피처 선택 (Feature Importance 기반)
        
        Parameters:
        -----------
        X : DataFrame
            피처
        y : Series
            타겟
        top_k : int
            선택할 피처 개수 (None이면 Config 사용)
        
        Returns:
        --------
        List[str]: 선택된 피처 리스트
        """
        if top_k is None:
            top_k = self.config.TOP_K_FEATURES
        
        # 제외할 메타 컬럼
        exclude_cols = [
            'bar30_start', 'bar30_end', 'timestamp', 
            'm1_index_entry', 'm1_index_label',
            'regime', 'regime_final', 'is_weekend'
        ]
        
        feature_cols = [c for c in X.columns if c not in exclude_cols]
        
        if len(feature_cols) <= top_k:
            print(f"  전체 피처 사용: {len(feature_cols)}개")
            return feature_cols
        
        # LightGBM으로 Feature Importance 계산
        print(f"  피처 중요도 계산 중... (전체 {len(feature_cols)}개)")
        
        train_data = lgb.Dataset(
            X[feature_cols].values, 
            label=y.values
        )
        
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbose': -1,
            'num_leaves': 31,
            'learning_rate': 0.05,
        }
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            verbose_eval=False
        )
        
        # Feature Importance 추출
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        # Top K 선택
        selected = importance.head(top_k)['feature'].tolist()
        
        print(f"  ✓ Top {top_k} 피처 선택 완료")
        print(f"    상위 5개: {selected[:5]}")
        
        self.feature_importance = importance
        self.selected_features = selected
        
        return selected
    
    # ==========================================
    # 앙상블 학습
    # ==========================================
    
    def train_ensemble(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        features: Optional[List[str]] = None,
        n_models: Optional[int] = None
    ) -> List:
        """
        앙상블 모델 학습
        
        Parameters:
        -----------
        X_train, y_train : 학습 데이터
        X_val, y_val : 검증 데이터
        features : 사용할 피처 (None이면 전체)
        n_models : 앙상블 모델 개수 (None이면 Config 사용)
        
        Returns:
        --------
        List: 학습된 모델 리스트
        """
        if n_models is None:
            n_models = self.config.ENSEMBLE_MODELS
        
        if features is None:
            features = self.selected_features or list(X_train.columns)
        
        print(f"  앙상블 학습 시작: {n_models}개 모델, {len(features)}개 피처")
        
        models = []
        
        train_data = lgb.Dataset(
            X_train[features].values,
            label=y_train.values
        )
        
        val_data = lgb.Dataset(
            X_val[features].values,
            label=y_val.values,
            reference=train_data
        )
        
        for i in range(n_models):
            print(f"    [{i+1}/{n_models}] 학습 중...", end=' ')
            
            # 파라미터 (seed만 변경)
            params = self.config.LGBM_PARAMS.copy()
            params['random_state'] = 42 + i
            params['verbose'] = -1
            
            # 학습
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=params.get('early_stopping_rounds', 10)),
                    lgb.log_evaluation(period=0)
                ]
            )
            
            # 검증 성능
            y_pred = model.predict(X_val[features].values)
            y_pred_binary = (y_pred > 0.5).astype(int)
            acc = accuracy_score(y_val, y_pred_binary)
            auc = roc_auc_score(y_val, y_pred)
            
            print(f"✓ (Acc: {acc:.4f}, AUC: {auc:.4f})")
            
            models.append(model)
        
        return models
    
    # ==========================================
    # 캘리브레이션
    # ==========================================
    
    def calibrate_models(
        self,
        models: List,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        features: List[str],
        method: Optional[str] = None
    ) -> Dict:
        """
        모델 캘리브레이션
        
        Parameters:
        -----------
        models : List
            학습된 모델 리스트
        X_val, y_val : 검증 데이터
        features : 피처 리스트
        method : str
            'isotonic' or 'platt' (None이면 Config 사용)
        
        Returns:
        --------
        Dict: {'models': List, 'calibrators': List}
        """
        if method is None:
            method = self.config.CALIBRATION_METHOD
        
        print(f"  캘리브레이션 ({method}) 시작...")
        
        calibrators = []
        
        for i, model in enumerate(models):
            print(f"    [{i+1}/{len(models)}] 캘리브레이션 중...", end=' ')
            
            # 예측 확률
            y_pred = model.predict(X_val[features].values).reshape(-1, 1)
            
            # Dummy estimator (이미 확률이므로)
            class DummyEstimator:
                def predict(self, X):
                    return X.ravel()
            
            dummy = DummyEstimator()
            
            # CalibratedClassifierCV
            calibrator = CalibratedClassifierCV(
                dummy,
                method=method,
                cv='prefit'
            )
            
            calibrator.fit(y_pred, y_val.values)
            
            # 캘리브레이션 후 성능
            y_cal = calibrator.predict_proba(y_pred)[:, 1]
            y_cal_binary = (y_cal > 0.5).astype(int)
            acc = accuracy_score(y_val, y_cal_binary)
            
            print(f"✓ (Acc: {acc:.4f})")
            
            calibrators.append(calibrator)
        
        return {
            'models': models,
            'calibrators': calibrators
        }
    
    # ==========================================
    # 레짐별 모델 학습
    # ==========================================
    
    def train_regime_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        regime_col: str = 'regime'
    ) -> Dict:
        """
        레짐별 모델 분리 학습
        
        Parameters:
        -----------
        X : DataFrame
            전체 피처
        y : Series
            전체 타겟
        regime_col : str
            레짐 컬럼명
        
        Returns:
        --------
        Dict: {
            'overall': {...},  # 전체 모델
            'regime_1': {...},  # UP 모델
            'regime_-1': {...}, # DOWN 모델
            'regime_0': {...}   # FLAT 모델
        }
        """
        print("\n" + "="*60)
        print("레짐별 모델 학습")
        print("="*60)
        
        if regime_col not in X.columns:
            print(f"⚠️ {regime_col} 컬럼 없음 → 전체 모델만 학습")
            return self._train_single_model(X, y, name='overall')
        
        results = {}
        
        # 1. 전체 모델 (Overall)
        print("\n[1] 전체 모델 학습")
        results['overall'] = self._train_single_model(X, y, name='overall')
        
        # 2. 레짐별 모델
        regimes = X[regime_col].unique()
        regime_names = {1: 'UP', -1: 'DOWN', 0: 'FLAT'}
        
        for regime_val in sorted(regimes):
            if pd.isna(regime_val):
                continue
            
            regime_name = regime_names.get(regime_val, f'REGIME-{regime_val}')
            print(f"\n[{int(regime_val)+2}] {regime_name} 레짐 모델 학습")
            
            # 해당 레짐 데이터만 필터링
            mask = X[regime_col] == regime_val
            X_regime = X[mask].copy()
            y_regime = y[mask].copy()
            
            if len(X_regime) < 100:
                print(f"  ⚠️ 데이터 부족 ({len(X_regime)}개) → 스킵")
                continue
            
            results[f'regime_{int(regime_val)}'] = self._train_single_model(
                X_regime, y_regime, name=regime_name
            )
        
        self.models = results
        
        return results
    
    def _train_single_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        name: str = 'model'
    ) -> Dict:
        """
        단일 모델 학습 (내부 헬퍼)
        
        Returns:
        --------
        Dict: {
            'models': List,
            'calibrators': List,
            'features': List[str],
            'metrics': Dict
        }
        """
        print(f"  데이터: {len(X):,}개 (클래스 분포: {y.value_counts().to_dict()})")
        
        # 데이터 분할
        train_ratio = self.config.TRAIN_RATIO
        val_ratio = self.config.VAL_RATIO
        
        n = len(X)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        X_train = X.iloc[:n_train].copy()
        y_train = y.iloc[:n_train].copy()
        
        X_val = X.iloc[n_train:n_train + n_val].copy()
        y_val = y.iloc[n_train:n_train + n_val].copy()
        
        X_test = X.iloc[n_train + n_val:].copy()
        y_test = y.iloc[n_train + n_val:].copy()
        
        print(f"  분할: Train={len(X_train):,} | Val={len(X_val):,} | Test={len(X_test):,}")
        
        # 피처 선택
        if self.selected_features is None:
            features = self.select_features(X_train, y_train)
        else:
            features = self.selected_features
        
        # 앙상블 학습
        models = self.train_ensemble(X_train, y_train, X_val, y_val, features)
        
        # 캘리브레이션
        calibrated = self.calibrate_models(models, X_val, y_val, features)
        
        # 테스트 성능 평가
        print(f"  테스트 성능 평가...")
        metrics = self._evaluate_ensemble(
            calibrated['models'],
            calibrated['calibrators'],
            X_test,
            y_test,
            features
        )
        
        return {
            'models': calibrated['models'],
            'calibrators': calibrated['calibrators'],
            'features': features,
            'metrics': metrics,
            'name': name
        }
    
    def _evaluate_ensemble(
        self,
        models: List,
        calibrators: List,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        features: List[str]
    ) -> Dict:
        """앙상블 성능 평가"""
        # 앙상블 예측 (평균)
        preds = []
        for model, calibrator in zip(models, calibrators):
            y_pred = model.predict(X_test[features].values).reshape(-1, 1)
            y_cal = calibrator.predict_proba(y_pred)[:, 1]
            preds.append(y_cal)
        
        y_ensemble = np.mean(preds, axis=0)
        y_ensemble_binary = (y_ensemble > 0.5).astype(int)
        
        # 메트릭 계산
        metrics = {
            'accuracy': accuracy_score(y_test, y_ensemble_binary),
            'precision': precision_score(y_test, y_ensemble_binary, zero_division=0),
            'recall': recall_score(y_test, y_ensemble_binary, zero_division=0),
            'f1': f1_score(y_test, y_ensemble_binary, zero_division=0),
            'auc': roc_auc_score(y_test, y_ensemble),
            'log_loss': log_loss(y_test, y_ensemble),
            'n_test': len(y_test)
        }
        
        print(f"    ✓ Acc: {metrics['accuracy']:.4f} | "
              f"AUC: {metrics['auc']:.4f} | "
              f"F1: {metrics['f1']:.4f}")
        
        return metrics
    
    # ==========================================
    # 모델 저장/로드
    # ==========================================
    
    def save_models(self, suffix: str = ''):
        """
        모델 저장
        
        Parameters:
        -----------
        suffix : str
            파일명 접미사 (예: '_retrain')
        """
        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        
        for model_key, model_data in self.models.items():
            # 파일명
            filename = f"model_{model_key}_{timestamp}{suffix}.pkl"
            filepath = self.config.MODEL_DIR / filename
            
            # 저장 데이터
            save_data = {
                'models': model_data['models'],
                'calibrators': model_data['calibrators'],
                'features': model_data['features'],
                'metrics': model_data['metrics'],
                'model_version': self.model_version,
                'feature_version': self.feature_version,
                'created_at': timestamp,
                'config': {
                    'LGBM_PARAMS': self.config.LGBM_PARAMS,
                    'ENSEMBLE_MODELS': self.config.ENSEMBLE_MODELS,
                    'TOP_K_FEATURES': self.config.TOP_K_FEATURES,
                }
            }
            
            # 저장
            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"  ✓ {model_key}: {filepath.name}")
        
        # 메타데이터 저장
        metadata_path = self.config.MODEL_DIR / f'model_metadata_{timestamp}.json'
        metadata = {
            'timestamp': timestamp,
            'model_version': self.model_version,
            'feature_version': self.feature_version,
            'models': {k: v['metrics'] for k, v in self.models.items()},
            'selected_features': self.selected_features,
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ 모델 저장 완료: {timestamp}")
    
    def load_models(self, timestamp: Optional[str] = None):
        """
        모델 로드
        
        Parameters:
        -----------
        timestamp : str
            모델 타임스탬프 (None이면 최신)
        """
        if timestamp is None:
            # 최신 모델 찾기
            model_files = list(self.config.MODEL_DIR.glob('model_overall_*.pkl'))
            if not model_files:
                print("⚠️ 저장된 모델 없음")
                return False
            
            latest = max(model_files, key=lambda x: x.stem)
            timestamp = latest.stem.split('_')[2]
        
        print(f"모델 로드 중: {timestamp}")
        
        # 모델 파일 찾기
        pattern = f"model_*_{timestamp}*.pkl"
        model_files = list(self.config.MODEL_DIR.glob(pattern))
        
        if not model_files:
            print(f"⚠️ 모델 파일 없음: {pattern}")
            return False
        
        self.models = {}
        
        for filepath in model_files:
            # 모델 키 추출
            parts = filepath.stem.split('_')
            if len(parts) >= 3:
                model_key = '_'.join(parts[1:-1])  # model_overall_timestamp → overall
            else:
                model_key = 'overall'
            
            # 로드
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models[model_key] = model_data
            
            metrics = model_data.get('metrics', {})
            print(f"  ✓ {model_key}: Acc={metrics.get('accuracy', 0):.4f}")
        
        # 메타데이터 로드
        metadata_path = self.config.MODEL_DIR / f'model_metadata_{timestamp}.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            self.model_version = self.metadata.get('model_version', 'unknown')
            self.feature_version = self.metadata.get('feature_version', 'unknown')
            self.selected_features = self.metadata.get('selected_features')
        
        print(f"✓ 모델 로드 완료: {len(self.models)}개")
        return True
    
    # ==========================================
    # 예측
    # ==========================================
    
    def predict(
        self,
        X: pd.DataFrame,
        regime: Optional[int] = None,
        use_regime_model: bool = True
    ) -> np.ndarray:
        """
        예측
        
        Parameters:
        -----------
        X : DataFrame
            피처
        regime : int
            레짐 값 (1, -1, 0)
        use_regime_model : bool
            레짐별 모델 사용 여부
        
        Returns:
        --------
        np.ndarray: 예측 확률
        """
        # 모델 선택
        if use_regime_model and regime is not None:
            model_key = f'regime_{int(regime)}'
            if model_key in self.models:
                model_data = self.models[model_key]
            else:
                # 레짐 모델 없으면 전체 모델 사용
                model_data = self.models.get('overall')
        else:
            model_data = self.models.get('overall')
        
        if model_data is None:
            raise ValueError("모델이 로드되지 않았습니다")
        
        models = model_data['models']
        calibrators = model_data['calibrators']
        features = model_data['features']
        
        # 앙상블 예측
        preds = []
        for model, calibrator in zip(models, calibrators):
            y_pred = model.predict(X[features].values).reshape(-1, 1)
            y_cal = calibrator.predict_proba(y_pred)[:, 1]
            preds.append(y_cal)
        
        return np.mean(preds, axis=0)
    

class ModelOptimizer:
    """
    호환용 얇은 래퍼.
    - main_pipe.py가 기대하는 인터페이스(initial_training / retrain_model / analyze_failures)를 제공
    - 내부적으로 현재 ModelTrainer + FeatureEngineer를 사용
    """
    def __init__(self, config):
        self.config = config
        self.trainer = ModelTrainer(config)

    def initial_training(self, df):
        """초기 학습: 피처 생성 → 레짐별 피처선택 → 레짐별 학습(+캘리브레이션) → 저장"""
        # 1) 피처/타겟
        fe = FeatureEngineer()
        df = df.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

        X = fe.create_feature_pool(df, lookback_window=200)
        y = fe.create_target(df, window=self.config.PREDICTION_WINDOW)

        valid = y.notna() & (y.index >= 200)
        X, y = X[valid], y[valid]
        if len(X) < 500:
            print("⚠️ 학습 표본이 적습니다. (500 미만)")

        # 2) 레짐별 피처 선택
        topk = int(getattr(self.config, 'TOP_K_FEATURES', 30))
        self.trainer.feature_selection_regime(X, y, regime_col='regime', top_k=topk)

        # 3) 레짐별 학습
        metrics = self.trainer.train_ensemble_regime(X, y, regime_col='regime', test_size=0.2)

        # 4) 저장
        self.trainer.save_model()
        return metrics

    def retrain_model(self, new_df):
        """
        간단 재학습: 기존 모델 유무와 상관없이 initial_training을 호출.
        (정교한 구버전 비교/채택 로직이 필요하면 여기서 확장 가능)
        """
        print("↻ 재학습(간단 모드): 새 데이터로 다시 학습합니다.")
        return self.initial_training(new_df)

    def analyze_failures(self, trade_log_df):
        """
        실패 패턴 분석(간이). trade_log_df에 필요한 피처가 없으면 빈 dict 반환.
        main_pipe의 적응형 필터 생성 단계에서 사용.
        """
        if trade_log_df is None or trade_log_df.empty or 'result' not in trade_log_df.columns:
            return {}

        out = {}

        # 예: ATR 기반 고변동성 실패 패턴
        if 'atr_14' in trade_log_df.columns:
            q = trade_log_df['atr_14'].quantile(0.8)
            sub = trade_log_df[trade_log_df['atr_14'] >= q]
            if len(sub) >= 50:
                fail_rate = (sub['result'] == 0).mean()
                if fail_rate > 0.55:
                    out['high_volatility'] = {
                        'atr_14_threshold': float(q),
                        'filter': f'atr_14 > {float(q):.6f}',
                        'reason': f'고변동성(atr_14 상위 20%) 구간 실패율 {fail_rate:.1%}'
                    }

        # 예: 거래량 기반 실패 패턴
        if 'volume_ratio' in trade_log_df.columns:
            qv = trade_log_df['volume_ratio'].quantile(0.8)
            subv = trade_log_df[trade_log_df['volume_ratio'] >= qv]
            if len(subv) >= 50:
                fail_rate = (subv['result'] == 0).mean()
                if fail_rate > 0.55:
                    out['high_volume'] = {
                        'volume_ratio_threshold': float(qv),
                        'filter': f'volume_ratio > {float(qv):.6f}',
                        'reason': f'고거래량(volume_ratio 상위 20%) 구간 실패율 {fail_rate:.1%}'
                    }

            ql = trade_log_df['volume_ratio'].quantile(0.2)
            subl = trade_log_df[trade_log_df['volume_ratio'] <= ql]
            if len(subl) >= 50:
                fail_rate = (subl['result'] == 0).mean()
                if fail_rate > 0.55:
                    out['low_volume'] = {
                        'volume_ratio_threshold': float(ql),
                        'filter': f'volume_ratio < {float(ql):.6f}',
                        'reason': f'저거래량(volume_ratio 하위 20%) 구간 실패율 {fail_rate:.1%}'
                    }

        # 예: 시간대 기반 실패 패턴
        if 'hour' in trade_log_df.columns:
            bad_hours = []
            for h, g in trade_log_df.groupby('hour'):
                if len(g) >= 30 and (g['result'] == 0).mean() > 0.6:
                    bad_hours.append(int(h))
            if bad_hours:
                out['time_based'] = {
                    'avoid_hours': bad_hours,
                    'filter': f'hour in {bad_hours}',
                    'reason': f'특정 시간대 실패율 높음(>60%)'
                }

        # 예: RSI 극단값 실패 패턴
        if 'rsi_14' in trade_log_df.columns:
            extremes = trade_log_df[(trade_log_df['rsi_14'] <= 30) | (trade_log_df['rsi_14'] >= 70)]
            if len(extremes) >= 50 and (extremes['result'] == 0).mean() > 0.55:
                out['rsi_extreme'] = {
                    'condition': 'RSI < 30 or RSI > 70',
                    'reason': 'RSI 극단값 근처에서 실패율 증가'
                }

        return out


# ==========================================
# 테스트
# ==========================================
if __name__ == "__main__":
    from data_loader import DataLoader
    
    Config.create_directories()
    
    print("="*60)
    print("ModelTrainer 테스트")
    print("="*60)
    
    # 데이터 로드
    loader = DataLoader()
    
    # 샘플 데이터 생성
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
    
    # 피처 + 타겟 생성
    print("\n피처 생성...")
    X, y = loader.prepare_train_data(df_1m, use_cache=False)
    
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}, 분포={y.value_counts().to_dict()}")
    
    # 모델 학습
    print("\n모델 학습...")
    trainer = ModelTrainer()
    
    # 레짐별 모델 학습
    results = trainer.train_regime_models(X, y)
    
    # 모델 저장
    print("\n모델 저장...")
    trainer.save_models()
    
    # 모델 로드 테스트
    print("\n모델 로드 테스트...")
    trainer2 = ModelTrainer()
    trainer2.load_models()
    
    # 예측 테스트
    print("\n예측 테스트...")
    X_test = X.tail(10).copy()
    
    # 전체 모델로 예측
    y_pred_overall = trainer2.predict(X_test, use_regime_model=False)
    print(f"  전체 모델 예측: {y_pred_overall}")
    
    # 레짐별 모델로 예측 (regime=1)
    if 'regime' in X_test.columns:
        y_pred_regime = trainer2.predict(X_test, regime=1, use_regime_model=True)
        print(f"  레짐(UP) 모델 예측: {y_pred_regime}")
    
    print("\n" + "="*60)
    print("✓ 테스트 완료")
    print("="*60)
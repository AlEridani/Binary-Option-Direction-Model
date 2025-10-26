"""
모델 학습 - LightGBM 기반 + 레짐별 모델
버전: 1.3.0
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import hashlib

import config


class ModelTrainer:
    """
    모델 학습 및 관리
    - LightGBM 기반 이진 분류
    - 레짐별 모델 분리
    - 앙상블 및 캘리브레이션
    - 버전 관리
    """
    
    def __init__(self):
        """인자 없이 초기화 (Config 참조)"""
        self.model_dir = config.MODEL_DIR
        self.models = {}
        self.feature_importance = {}
        self.metadata = {}
        
        # 레짐별 모델 저장 구조
        self.regime_models = {
            'UP': None,     # regime = 1
            'DOWN': None,   # regime = -1
            'FLAT': None    # regime = 0
        }
        
        # LightGBM 파라미터 준비 (verbose 중복 방지)
        self.lgbm_params = config.LIGHTGBM_PARAMS.copy()
        self.lgbm_params.pop('verbose', None)  # 기존 verbose 제거
        self.lgbm_params['verbose'] = -1  # 새로 설정
        
    def feature_selection_regime(self, X: pd.DataFrame, y: pd.Series, 
                                 regime_col: str, top_k: int = 30) -> List[str]:
        """
        레짐별 중요 피처 선택
        
        Args:
            X: 피처 데이터 (regime 컬럼 포함)
            y: 타겟
            regime_col: 레짐 컬럼명
            top_k: 상위 K개 피처
        
        Returns:
            중요 피처 리스트
        """
        if regime_col not in X.columns:
            print(f"⚠️  레짐 컬럼 없음: {regime_col}")
            return list(X.columns)
        
        # 임시 모델 학습 (regime 컬럼 포함)
        model = lgb.LGBMClassifier(**self.lgbm_params)
        model.fit(X, y)
        
        # 피처 중요도
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_features = importance.head(top_k)['feature'].tolist()
        
        print(f"✅ 피처 선택 완료: 상위 {top_k}개")
        print(f"  상위 5개: {top_features[:5]}")
        
        return top_features
    
    def train_ensemble_regime(self, X: pd.DataFrame, y: pd.Series, 
                             regime_col: str = 'regime',
                             test_size: float = 0.2) -> Dict:
        """
        레짐별 앙상블 모델 학습
        
        Args:
            X: 피처 데이터 (regime 컬럼 포함)
            y: 타겟
            regime_col: 레짐 컬럼명
            test_size: 테스트 비율
        
        Returns:
            학습 결과 딕셔너리
        """
        results = {}
        
        if regime_col not in X.columns:
            print(f"⚠️  레짐 컬럼 없음: {regime_col}, 통합 모델 학습")
            return self.train_single_model(X, y, test_size)
        
        # 레짐별 데이터 분할
        regimes = X[regime_col].unique()
        print(f"\n🎯 레짐별 모델 학습 시작: {sorted(regimes)}")
        
        for regime_value in sorted(regimes):
            regime_name = {1: 'UP', -1: 'DOWN', 0: 'FLAT'}.get(regime_value, f'REGIME_{regime_value}')
            
            # 레짐 필터링 (regime 컬럼은 피처로 유지)
            mask = X[regime_col] == regime_value
            X_regime = X[mask]  # regime 컬럼 유지!
            y_regime = y[mask]
            
            if len(X_regime) < 100:
                print(f"  ⚠️  {regime_name}: 데이터 부족 ({len(X_regime)}개), 스킵")
                continue
            
            print(f"\n  📊 {regime_name} 레짐 학습 ({len(X_regime)}개)")
            
            # Train/Test 분할
            X_train, X_test, y_train, y_test = train_test_split(
                X_regime, y_regime, test_size=test_size, shuffle=False
            )
            
            # 앙상블 학습
            models = []
            predictions_test = []
            
            for i in range(config.N_ENSEMBLE):
                # 부트스트랩 샘플링
                n_samples = len(X_train)
                indices = np.random.choice(n_samples, n_samples, replace=True)
                
                X_boot = X_train.iloc[indices]
                y_boot = y_train.iloc[indices]
                
                # 모델 학습
                model = lgb.LGBMClassifier(**self.lgbm_params)
                model.fit(X_boot, y_boot)
                
                # 예측
                pred_proba = model.predict_proba(X_test)[:, 1]
                predictions_test.append(pred_proba)
                
                models.append(model)
            
            # 앙상블 예측 (평균)
            y_pred_proba = np.mean(predictions_test, axis=0)
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # 성능 평가
            accuracy = (y_pred == y_test).mean()
            win_rate = accuracy
            
            print(f"    Accuracy: {accuracy:.4f}")
            print(f"    Win Rate: {win_rate:.4f}")
            
            # 캘리브레이션
            calibrated_model = self._calibrate_ensemble(models, X_test, y_test)
            
            # 저장
            self.regime_models[regime_name] = {
                'models': models,
                'calibrated': calibrated_model,
                'n_samples': len(X_regime),
                'accuracy': accuracy,
                'win_rate': win_rate
            }
            
            results[regime_name] = {
                'n_samples': len(X_regime),
                'accuracy': accuracy,
                'win_rate': win_rate
            }
        
        print("\n✅ 레짐별 모델 학습 완료")
        return results
    
    def train_single_model(self, X: pd.DataFrame, y: pd.Series, 
                          test_size: float = 0.2) -> Dict:
        """
        단일 통합 모델 학습
        
        Args:
            X: 피처 데이터
            y: 타겟
            test_size: 테스트 비율
        
        Returns:
            학습 결과 딕셔너리
        """
        print("\n🎯 통합 모델 학습 시작")
        
        # Train/Test 분할
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # 앙상블 학습
        models = []
        predictions_test = []
        
        for i in range(config.N_ENSEMBLE):
            # 부트스트랩
            n_samples = len(X_train)
            indices = np.random.choice(n_samples, n_samples, replace=True)
            
            X_boot = X_train.iloc[indices]
            y_boot = y_train.iloc[indices]
            
            # 모델 학습
            model = lgb.LGBMClassifier(**self.lgbm_params)
            model.fit(X_boot, y_boot)
            
            # 예측
            pred_proba = model.predict_proba(X_test)[:, 1]
            predictions_test.append(pred_proba)
            
            models.append(model)
        
        # 앙상블 예측
        y_pred_proba = np.mean(predictions_test, axis=0)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # 성능 평가
        accuracy = (y_pred == y_test).mean()
        
        print(f"  Accuracy: {accuracy:.4f}")
        
        # 캘리브레이션
        calibrated_model = self._calibrate_ensemble(models, X_test, y_test)
        
        # 저장
        self.models['unified'] = {
            'models': models,
            'calibrated': calibrated_model,
            'n_samples': len(X),
            'accuracy': accuracy
        }
        
        print("✅ 통합 모델 학습 완료")
        
        return {
            'unified': {
                'n_samples': len(X),
                'accuracy': accuracy
            }
        }
    
    def _calibrate_ensemble(self, models: List, X_cal: pd.DataFrame, 
                           y_cal: pd.Series):
        """앙상블 모델 캘리브레이션"""
        # 앙상블 예측
        predictions = []
        for model in models:
            pred = model.predict_proba(X_cal)[:, 1]
            predictions.append(pred)
        
        ensemble_pred = np.mean(predictions, axis=0)
        
        # 단순 캘리브레이션 (isotonic regression)
        from sklearn.isotonic import IsotonicRegression
        
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(ensemble_pred, y_cal)
        
        return calibrator
    
    def save_models(self) -> None:
        """모델 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 레짐별 모델 저장
        for regime_name, regime_data in self.regime_models.items():
            if regime_data is None:
                continue
            
            model_file = self.model_dir / f"lgbm_regime_{regime_name}.pkl"
            
            with open(model_file, 'wb') as f:
                pickle.dump(regime_data, f)
            
            print(f"✅ 모델 저장: {model_file.name}")
        
        # 통합 모델 저장 (있으면)
        if 'unified' in self.models:
            model_file = self.model_dir / "lgbm_unified.pkl"
            
            with open(model_file, 'wb') as f:
                pickle.dump(self.models['unified'], f)
            
            print(f"✅ 모델 저장: {model_file.name}")
        
        # 메타데이터 저장
        metadata = {
            'model_version': config.MODEL_VERSION,
            'feature_version': config.FEATURE_VERSION,
            'trained_at': timestamp,
            'lightgbm_params': self.lgbm_params,
            'n_ensemble': config.N_ENSEMBLE,
            'regime_models': {
                regime: {
                    'n_samples': data['n_samples'],
                    'accuracy': data.get('accuracy', 0),
                    'win_rate': data.get('win_rate', 0)
                } if data else None
                for regime, data in self.regime_models.items()
            }
        }
        
        # 해시 계산
        model_hash = self._calculate_model_hash()
        metadata['model_hash'] = model_hash
        
        metadata_file = self.model_dir / "model_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ 메타데이터 저장: {metadata_file.name}")
        
        self.metadata = metadata
    
    def load_models(self) -> bool:
        """모델 로드"""
        loaded_count = 0
        
        # 레짐별 모델 로드
        for regime_name in ['UP', 'DOWN', 'FLAT']:
            model_file = self.model_dir / f"lgbm_regime_{regime_name}.pkl"
            
            if model_file.exists():
                try:
                    with open(model_file, 'rb') as f:
                        self.regime_models[regime_name] = pickle.load(f)
                    
                    loaded_count += 1
                    print(f"✅ 모델 로드: {model_file.name}")
                except Exception as e:
                    print(f"⚠️  모델 로드 실패 ({model_file.name}): {e}")
        
        # 통합 모델 로드
        unified_file = self.model_dir / "lgbm_unified.pkl"
        if unified_file.exists():
            try:
                with open(unified_file, 'rb') as f:
                    self.models['unified'] = pickle.load(f)
                
                loaded_count += 1
                print(f"✅ 모델 로드: {unified_file.name}")
            except Exception as e:
                print(f"⚠️  모델 로드 실패 (unified): {e}")
        
        # 메타데이터 로드
        metadata_file = self.model_dir / "model_metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
                
                print(f"✅ 메타데이터 로드: {metadata_file.name}")
                print(f"  - 모델 버전: {self.metadata.get('model_version')}")
                print(f"  - 학습 시각: {self.metadata.get('trained_at')}")
            except Exception as e:
                print(f"⚠️  메타데이터 로드 실패: {e}")
        
        return loaded_count > 0
    
    def predict(self, X: pd.DataFrame, regime: Optional[int] = None, 
                use_regime_model: bool = True) -> np.ndarray:
        """
        예측 수행
        
        Args:
            X: 피처 데이터 (regime 컬럼 포함 권장)
            regime: 레짐 값 (1, 0, -1) - 모델 선택용
            use_regime_model: 레짐별 모델 사용 여부
        
        Returns:
            예측 확률 배열
        """
        # X를 복사하여 수정
        X_pred = X.copy()
        
        # regime 컬럼이 없으면 추가 (예측 시 regime 값 사용)
        if 'regime' not in X_pred.columns and regime is not None:
            X_pred['regime'] = regime
        
        # 레짐별 모델 사용
        if use_regime_model and regime is not None:
            regime_name = {1: 'UP', -1: 'DOWN', 0: 'FLAT'}.get(regime)
            
            if regime_name and self.regime_models.get(regime_name):
                return self._predict_with_regime(X_pred, regime_name)
        
        # 통합 모델 사용
        if 'unified' in self.models:
            return self._predict_with_unified(X_pred)
        
        # 폴백: 사용 가능한 첫 번째 모델
        for regime_name, regime_data in self.regime_models.items():
            if regime_data is not None:
                print(f"⚠️  폴백: {regime_name} 모델 사용")
                return self._predict_with_regime(X_pred, regime_name)
        
        print("❌ 사용 가능한 모델 없음")
        return np.full(len(X), 0.5)
    
    def _predict_with_regime(self, X: pd.DataFrame, regime_name: str) -> np.ndarray:
        """레짐별 모델로 예측"""
        regime_data = self.regime_models[regime_name]
        
        if regime_data is None:
            return np.full(len(X), 0.5)
        
        models = regime_data['models']
        calibrator = regime_data.get('calibrated')
        
        # 앙상블 예측
        predictions = []
        for model in models:
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)
        
        ensemble_pred = np.mean(predictions, axis=0)
        
        # 캘리브레이션 적용
        if calibrator is not None:
            ensemble_pred = calibrator.predict(ensemble_pred)
        
        return ensemble_pred
    
    def _predict_with_unified(self, X: pd.DataFrame) -> np.ndarray:
        """통합 모델로 예측"""
        unified_data = self.models['unified']
        
        models = unified_data['models']
        calibrator = unified_data.get('calibrated')
        
        # 앙상블 예측
        predictions = []
        for model in models:
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)
        
        ensemble_pred = np.mean(predictions, axis=0)
        
        # 캘리브레이션 적용
        if calibrator is not None:
            ensemble_pred = calibrator.predict(ensemble_pred)
        
        return ensemble_pred
    
    def _calculate_model_hash(self) -> str:
        """모델 해시 계산"""
        hash_data = {
            'model_version': config.MODEL_VERSION,
            'feature_version': config.FEATURE_VERSION,
            'params': self.lgbm_params,
            'n_ensemble': config.N_ENSEMBLE,
            'regimes': list(self.regime_models.keys())
        }
        
        hash_str = json.dumps(hash_data, sort_keys=True)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]
    
    def get_model_meta(self) -> Dict:
        """모델 메타데이터 반환"""
        if not self.metadata:
            return {
                'model_version': config.MODEL_VERSION,
                'trained_at': 'unknown',
                'model_hash': 'unknown'
            }
        
        return self.metadata


# ============================================================
# 테스트 및 검증
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ModelTrainer 테스트")
    print("=" * 60)
    
    # ModelTrainer 초기화
    trainer = ModelTrainer()
    
    print("\n✅ ModelTrainer 초기화 성공")
    print(f"  - MODEL_DIR: {trainer.model_dir}")
    print(f"  - N_ENSEMBLE: {config.N_ENSEMBLE}")
    
    # 테스트 데이터 생성
    np.random.seed(42)
    n_samples = 2000
    n_features = 20
    
    X_test = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # 레짐 추가
    X_test['regime'] = np.random.choice([1, 0, -1], n_samples)
    
    # 타겟 생성 (레짐에 따라 약간의 경향성 부여)
    y_test = np.random.rand(n_samples)
    y_test = (y_test + X_test['regime'] * 0.1 + 0.5).clip(0, 1)
    y_test = (y_test > 0.5).astype(int)
    
    print(f"\n📊 테스트 데이터:")
    print(f"  - 샘플 수: {n_samples}")
    print(f"  - 피처 수: {n_features + 1} (regime 포함)")
    print(f"  - 타겟 분포: UP={y_test.mean():.2%}")
    
    # 레짐별 모델 학습
    print("\n🎯 레짐별 모델 학습")
    results = trainer.train_ensemble_regime(X_test, y_test, regime_col='regime', test_size=0.2)
    
    print("\n📊 학습 결과:")
    for regime_name, result in results.items():
        print(f"  {regime_name}:")
        print(f"    - 샘플 수: {result['n_samples']}")
        print(f"    - Accuracy: {result['accuracy']:.4f}")
        print(f"    - Win Rate: {result.get('win_rate', 0):.4f}")
    
    # 모델 저장
    print("\n💾 모델 저장")
    trainer.save_models()
    
    # 모델 로드
    print("\n📂 모델 로드")
    trainer_new = ModelTrainer()
    loaded = trainer_new.load_models()
    
    if loaded:
        print("✅ 모델 로드 성공")
        
        # 예측 테스트
        print("\n🔮 예측 테스트")
        
        X_pred = X_test.head(10).drop(columns=['regime'])
        
        for regime_value in [1, 0, -1]:
            regime_name = {1: 'UP', -1: 'DOWN', 0: 'FLAT'}.get(regime_value)
            pred = trainer_new.predict(X_pred, regime=regime_value, use_regime_model=True)
            
            print(f"  {regime_name} 레짐 예측:")
            print(f"    평균 확률: {pred.mean():.4f}")
            print(f"    확률 범위: [{pred.min():.4f}, {pred.max():.4f}]")
        
        # 메타데이터 확인
        print("\n📋 모델 메타데이터:")
        meta = trainer_new.get_model_meta()
        for key, value in meta.items():
            if key != 'lightgbm_params' and key != 'regime_models':
                print(f"  - {key}: {value}")
    
    else:
        print("❌ 모델 로드 실패")
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)
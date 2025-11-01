"""
모델 학습 - 레짐별 앙상블 + 캘리브레이션 + 버전 관리
버전: 1.4.1 (인덱스 중복 수정)

핵심 기능:
- 레짐별(UP/FLAT/DOWN) 독립 앙상블 모델
- 각 모델당 N개 learner (VotingClassifier)
- 캘리브레이션 (Isotonic/Sigmoid)
- 버전 관리 (bundle_up_20250127_v1.0.0)
- 레짐 피처 하이브리드 (regime 제외, 나머지 사용)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import joblib
import json
import shutil
import hashlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb

import config


class EnsembleModel:
    """
    앙상블 모델 (N개 LightGBM learner + Calibration)
    
    - 다른 seed로 학습
    - VotingClassifier로 통합
    - Isotonic/Sigmoid 캘리브레이션
    - 안정적인 예측
    """
    
    def __init__(self, n_learners: int = 3, params: Optional[Dict] = None):
        """
        초기화
        
        Args:
            n_learners: learner 개수
            params: LightGBM 파라미터
        """
        self.n_learners = n_learners
        self.params = params or config.LIGHTGBM_PARAMS.copy()
        self.params['verbose'] = -1
        
        self.ensemble = None
        self.calibrator = None
        self.feature_names = []
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """
        앙상블 학습 + 캘리브레이션
        
        Args:
            X_train: 학습 데이터
            y_train: 학습 타겟
            X_val: 검증 데이터
            y_val: 검증 타겟
        
        Returns:
            학습 결과
        """
        print(f"    🔄 앙상블 학습 시작 ({self.n_learners}개 learner)")
        
        self.feature_names = list(X_train.columns)
        
        # learner 생성
        estimators = []
        
        for i in range(self.n_learners):
            params = self.params.copy()
            params['random_state'] = 42 + i  # seed 다양화
            
            learner = lgb.LGBMClassifier(**params)
            estimators.append((f'lgb_{i}', learner))
        
        # VotingClassifier 생성
        self.ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',  # 확률 평균
            n_jobs=-1
        )
        
        # 학습
        self.ensemble.fit(X_train, y_train)
        
        # 검증 (캘리브레이션 전)
        y_pred_raw = self.ensemble.predict(X_val)
        y_proba_raw = self.ensemble.predict_proba(X_val)[:, 1]
        
        # 캘리브레이션
        print(f"      🔧 캘리브레이션 ({config.CALIBRATION_METHOD})")
        self.calibrator = self._calibrate(y_proba_raw, y_val)
        
        # 캘리브레이션 후 예측
        y_proba_cal = self._apply_calibration(y_proba_raw)
        y_pred_cal = (y_proba_cal >= 0.5).astype(int)
        
        # 메트릭 계산
        metrics = {
            'accuracy_raw': accuracy_score(y_val, y_pred_raw),
            'accuracy_cal': accuracy_score(y_val, y_pred_cal),
            'precision': precision_score(y_val, y_pred_cal, zero_division=0),
            'recall': recall_score(y_val, y_pred_cal, zero_division=0),
            'f1': f1_score(y_val, y_pred_cal, zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_proba_cal)
        }
        
        print(f"      ✅ 학습 완료")
        print(f"         Accuracy (Raw): {metrics['accuracy_raw']:.4f}")
        print(f"         Accuracy (Cal): {metrics['accuracy_cal']:.4f}")
        print(f"         Precision: {metrics['precision']:.4f}")
        print(f"         Recall: {metrics['recall']:.4f}")
        print(f"         F1: {metrics['f1']:.4f}")
        print(f"         ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def _calibrate(self, y_proba: np.ndarray, y_true: pd.Series):
        """
        캘리브레이션 학습
        
        Args:
            y_proba: 원시 예측 확률
            y_true: 실제 타겟
        
        Returns:
            캘리브레이터
        """
        method = config.CALIBRATION_METHOD
        
        if method == 'isotonic':
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(y_proba, y_true)
        
        elif method == 'sigmoid':
            # Platt scaling
            calibrator = LogisticRegression()
            calibrator.fit(y_proba.reshape(-1, 1), y_true)
        
        else:
            print(f"      ⚠️  알 수 없는 캘리브레이션: {method}, isotonic 사용")
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(y_proba, y_true)
        
        return calibrator
    
    def _apply_calibration(self, y_proba: np.ndarray) -> np.ndarray:
        """캘리브레이션 적용"""
        if self.calibrator is None:
            return y_proba
        
        if config.CALIBRATION_METHOD == 'sigmoid':
            return self.calibrator.predict_proba(y_proba.reshape(-1, 1))[:, 1]
        else:  # isotonic
            return self.calibrator.predict(y_proba)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """확률 예측 (캘리브레이션 적용)"""
        if self.ensemble is None:
            raise ValueError("모델이 학습되지 않았습니다")
        
        # 피처 순서 확인
        if list(X.columns) != self.feature_names:
            X = X[self.feature_names]
        
        # 원시 예측
        y_proba_raw = self.ensemble.predict_proba(X)[:, 1]
        
        # 캘리브레이션 적용
        y_proba_cal = self._apply_calibration(y_proba_raw)
        
        return np.column_stack([1 - y_proba_cal, y_proba_cal])
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """예측"""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    def get_feature_importance(self) -> Dict:
        """피처 중요도"""
        if self.ensemble is None:
            return {}
        
        importances = []
        
        for name, estimator in self.ensemble.estimators_:
            importances.append(estimator.feature_importances_)
        
        avg_importance = np.mean(importances, axis=0)
        
        importance_dict = dict(zip(self.feature_names, avg_importance))
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict


class ModelTrainer:
    """
    모델 학습 관리자
    
    - 레짐별 앙상블 학습
    - 캘리브레이션
    - 버전 관리
    - 백업 및 롤백
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.regime_models = {}  # {regime_name: EnsembleModel}
        self.version = config.MODEL_VERSION
        self.metadata = {}
    
    def _debug_feature_stats(self, X: pd.DataFrame, tag: str = ""):
        """ADX/RSI 등 핵심 피처 분포와 NaN율을 출력"""
        print(f"\n[DEBUG] Feature stats {tag}")
        print(f"  - shape: {X.shape}")
        # 후보 컬럼 자동 탐색
        key_patterns = ["adx", "rsi", "atr", "momentum", "roc", "vol", "volume", "ma", "sma", "ema"]
        sel = [c for c in X.columns if any(p in c.lower() for p in key_patterns)]
        sel = sel[:20]  # 너무 길어지지 않게
        if not sel:
            print("  - (no key features found to summarize)")
            return
        # NaN율
        nan_rate = X[sel].isna().mean().sort_values(ascending=False).head(10)
        print("  - NaN rate(top10):")
        print(nan_rate.to_string())

        # 분위수(ADX/RSI가 0~100 스케일인지, 0~1로 정규화됐는지 즉시 파악)
        q = X[sel].quantile([0.0, 0.1, 0.5, 0.9, 1.0]).T
        print("\n  - quantiles:")
        print(q.to_string())

    def _debug_regime_counts(self, df_like: pd.DataFrame, where: str):
        """regime 분포 및 NaN 개수"""
        if "regime" not in df_like.columns:
            print(f"[DEBUG] {where}: 'regime' column NOT found")
            return
        vc = df_like["regime"].value_counts(dropna=False).to_dict()
        print(f"[DEBUG] {where}: regime counts = {vc}")
        n_nan = df_like["regime"].isna().sum()
        if n_nan:
            print(f"[DEBUG] {where}: regime NaNs = {n_nan}")
    def train_regime_ensemble(self, X: pd.DataFrame, y: pd.Series, 
                                n_learners: int = 3, test_size: float = 0.2) -> Dict:
        """
        레짐별 앙상블 학습
        
        Args:
            X: 피처 데이터 (regime 컬럼 포함)
            y: 타겟 데이터
            n_learners: learner 개수
            test_size: 테스트 비율
        
        Returns:
            레짐별 결과
        """
        print("\n" + "=" * 60)
        print("레짐별 앙상블 학습 + 캘리브레이션")
        print("=" * 60)
        
        if 'regime' not in X.columns:
            print("❌ regime 컬럼 없음")
            return {}
        self._debug_regime_counts(X, where="ENTRY")
        self._debug_feature_stats(X.drop(columns=['regime'], errors='ignore'), tag="[ENTRY]")

        # 전체 인덱스 리셋 (중복 방지)
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        
        
        # 스케일러 학습 (전체 데이터, regime 제외)
        X_no_regime = X.drop(columns=['regime'])
        self._debug_feature_stats(X_no_regime, tag="[BEFORE SCALER FIT]")
        self.scaler.fit(X_no_regime)
        
        results = {}
        
        # 레짐별 학습
        for regime_val in [1, 0, -1]:
            regime_name = {1: 'up', 0: 'flat', -1: 'down'}[regime_val]
            
            print(f"\n📚 {regime_name.upper()} 레짐 학습")
            
            # 1) numpy boolean으로 변환해 '위치기반' 마스킹을 강제
            mask_np = (X['regime'].to_numpy() == regime_val)

            # 2) 반드시 .loc[mask_np] 사용 (reindex 회피)
            X_regime = X.loc[mask_np].copy()
            y_regime = y.loc[mask_np].copy()
            
            
            print(f"  데이터: {len(X_regime)}건")
            if len(X_regime) == 0:
                # 레짐별 피처 NaN율/분위수도 같이 보자
                X_no_regime = X.drop(columns=['regime'], errors='ignore')
                nan_rate = X_no_regime.isna().mean().sort_values(ascending=False).head(10)
                print("  [DEBUG] top NaN rate features:\n", nan_rate.to_string())
                print("  ⚠️ FLAT(또는 해당 레짐) 0건 → 라벨링/정렬/임계값 점검 필요")
                continue
            
            # 레짐 인덱스 리셋 (중복 방지)
            X_regime = X_regime.reset_index(drop=True)
            y_regime = y_regime.reset_index(drop=True)
            
            # regime 컬럼 제거 (모델 선택용, 피처 아님)
            X_regime_clean = X_regime.drop(columns=['regime'])
            
            self._debug_feature_stats(X_regime_clean, tag=f"[REGIME={regime_name.upper()}]")

            # 시계열 분할 (순서 유지)
            n = len(X_regime_clean)
            split_idx = int(n * (1 - test_size))
            # 최소 1/1 보장
            if split_idx <= 0:
                split_idx = 1
            elif split_idx >= n:
                split_idx = n - 1
            
            X_train = X_regime_clean.iloc[:split_idx].copy()
            X_test = X_regime_clean.iloc[split_idx:].copy()
            y_train = y_regime.iloc[:split_idx].copy()
            y_test = y_regime.iloc[split_idx:].copy()
            
            # 인덱스 리셋 (reindex 오류 방지)
            X_train = X_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)
            
            print(f"  분할: Train={len(X_train)}, Test={len(X_test)}")
            
            # 스케일링
            try:
                X_train_scaled = pd.DataFrame(self.scaler.transform(X_train), columns=X_train.columns)
                X_test_scaled  = pd.DataFrame(self.scaler.transform(X_test),  columns=X_test.columns)
            except Exception as e:   # ← e 추가
                # 혹시 전체 스케일러가 안 맞으면 해당 레짐 피처로 재적합
                print(f"  [DEBUG] global scaler.transform 실패 → 레짐별 재적합: {e}")
                self.scaler.fit(X_regime_clean)
                X_train_scaled = pd.DataFrame(self.scaler.transform(X_train), columns=X_train.columns)
                X_test_scaled  = pd.DataFrame(self.scaler.transform(X_test),  columns=X_test.columns)
            
            # 앙상블 모델 생성 및 학습
            ensemble = EnsembleModel(n_learners=n_learners)
            metrics = ensemble.train(X_train_scaled, y_train, X_test_scaled, y_test)
            
            # 저장
            self.regime_models[regime_name] = ensemble
            results[regime_name] = metrics
        
        print("\n" + "=" * 60)
        print("✅ 레짐별 앙상블 학습 완료")
        print("=" * 60)
        
        # 메타데이터 저장
        self.metadata = {
            'version': self.version,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'n_learners': n_learners,
            'test_size': test_size,
            'calibration_method': config.CALIBRATION_METHOD,
            'results': results,
            'feature_count': X_no_regime.shape[1],
            'total_samples': len(X)
        }
        
        return results
    
    def save_bundle(self, bundle_name: Optional[str] = None) -> Path:
        """
        모델 번들 저장 (버전 관리)
        
        Args:
            bundle_name: 번들 이름 (None이면 자동 생성)
        
        Returns:
            번들 경로
        
        번들 구조:
        bundle_20250127_v1.0.0/
        ├─ model_up.pkl
        ├─ model_flat.pkl
        ├─ model_down.pkl
        ├─ scaler.pkl
        └─ metadata.json
        """
        # 번들명 생성
        if bundle_name is None:
            date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
            bundle_name = f"bundle_{date_str}_{self.version}"
        
        bundle_dir = config.MODEL_DIR / bundle_name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n💾 모델 번들 저장: {bundle_name}")
        
        # 레짐별 모델 저장
        for regime_name, model in self.regime_models.items():
            model_path = bundle_dir / f"model_{regime_name}.pkl"
            joblib.dump(model, model_path)
            print(f"  ✅ {regime_name.upper()} 모델 저장")
        
        # 스케일러 저장
        scaler_path = bundle_dir / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"  ✅ 스케일러 저장")
        
        # 메타데이터 저장
        metadata_path = bundle_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"  ✅ 메타데이터 저장")
        
        # 심볼릭 링크 생성 (최신 버전)
        latest_link = config.MODEL_DIR / "latest"
        # 링크면 unlink, 디렉터리면 rmtree
        if latest_link.is_symlink():
            latest_link.unlink()
        elif latest_link.exists():
            shutil.rmtree(latest_link, ignore_errors=True)

        try:
            latest_link.symlink_to(bundle_dir.name)
            print(f"  ✅ 최신 버전 링크 생성")
        except Exception as e:
            print(f"  ⚠️  심볼릭 링크 생성 실패: {e}")
            # 윈도우 등에서 실패 시 복사본 생성
            try:
                shutil.copytree(bundle_dir, latest_link, dirs_exist_ok=False)
                print(f"  ✅ 최신 버전 복사본 생성")
            except Exception as e2:
                print(f"  ❌ 최신 버전 복사본 생성 실패: {e2}")
        
        return bundle_dir
    
    def load_bundle(self, bundle_name: str = "latest") -> bool:
        """
        모델 번들 로드
        
        Args:
            bundle_name: 번들 이름 또는 "latest"
        
        Returns:
            로드 성공 여부
        """
        bundle_dir = config.MODEL_DIR / bundle_name
        
        if not bundle_dir.exists():
            print(f"❌ 번들 없음: {bundle_name}")
            return False
        
        print(f"\n📂 모델 번들 로드: {bundle_name}")
        
        try:
            # 메타데이터 로드
            metadata_path = bundle_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                print(f"  ✅ 메타데이터 로드")
            
            # 스케일러 로드
            scaler_path = bundle_dir / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                print(f"  ✅ 스케일러 로드")
            
            # 레짐별 모델 로드
            self.regime_models = {}
            
            for regime_name in ['up', 'flat', 'down']:
                model_path = bundle_dir / f"model_{regime_name}.pkl"
                
                if model_path.exists():
                    self.regime_models[regime_name] = joblib.load(model_path)
                    print(f"  ✅ {regime_name.upper()} 모델 로드")
                else:
                    print(f"  ⚠️  {regime_name.upper()} 모델 없음")
            
            if not self.regime_models:
                print("  ❌ 유효한 모델 없음")
                return False
            
            print(f"\n✅ 번들 로드 완료: {bundle_name}")
            
            return True
            
        except Exception as e:
            print(f"❌ 번들 로드 실패: {e}")
            return False
    
    def list_bundles(self) -> List[Dict]:
        """
        저장된 번들 목록
        
        Returns:
            번들 정보 리스트
        """
        bundles = []
        
        for bundle_dir in config.MODEL_DIR.glob("bundle_*"):
            if not bundle_dir.is_dir():
                continue
            
            metadata_path = bundle_dir / "metadata.json"
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            bundles.append({
                'name': bundle_dir.name,
                'path': bundle_dir,
                'timestamp': metadata.get('timestamp', 'unknown'),
                'version': metadata.get('version', 'unknown'),
                'results': metadata.get('results', {})
            })
        
        # 최신순 정렬
        bundles.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return bundles
    
    def backup_bundle(self, bundle_name: str, backup_dir: Optional[Path] = None) -> Path:
        """
        번들 백업
        
        Args:
            bundle_name: 번들 이름
            backup_dir: 백업 디렉토리 (None이면 기본 위치)
        
        Returns:
            백업 경로
        """
        if backup_dir is None:
            backup_dir = config.BASE_DIR / "backups"
        
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        bundle_dir = config.MODEL_DIR / bundle_name
        
        if not bundle_dir.exists():
            raise ValueError(f"번들 없음: {bundle_name}")
        
        # 타임스탬프 추가
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_name = f"{bundle_name}_backup_{timestamp}"
        backup_path = backup_dir / backup_name
        
        # 복사
        shutil.copytree(bundle_dir, backup_path)
        
        print(f"✅ 백업 완료: {backup_path}")
        
        return backup_path
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        """
        🔁 하위호환 어댑터:
        예전 호출부의 train_model(X, y, ...)을 받아서
        내부 표준 메서드 train_regime_ensemble(...)로 위임한다.

        요구사항:
        - X에 'regime' 컬럼이 없으면 단일 레짐(UP=1)으로 가정
        - n_learners/test_size 등 추가 인자는 kwargs에서 읽음
        반환:
        - train_regime_ensemble와 동일한 결과 dict
        """
        X_ = X.copy()
        if "regime" not in X_.columns:
            X_["regime"] = 1  # 단일 레짐으로 가정(UP)

        n_learners = kwargs.get("n_learners", 3)
        test_size  = kwargs.get("test_size", 0.2)

        return self.train_regime_ensemble(
            X=X_,
            y=y,
            n_learners=n_learners,
            test_size=test_size,
        )
    
    def get_feature_names_for_regime(self, regime: int):
        """
        학습 시 사용된 '정확한 피처 순서'를 반환.
        EnsembleModel.feature_names 를 그대로 꺼낸다.
        """
        regime_name = {1: 'up', 0: 'flat', -1: 'down'}.get(regime)
        if not regime_name:
            return None
        mdl = self.regime_models.get(regime_name)
        if mdl is None:
            return None
        feats = getattr(mdl, "feature_names", None)
        return list(feats) if feats else None
    
    def predict_with_regime(self, X: pd.DataFrame, regime: int) -> np.ndarray:
        """
        레짐별 예측 (캘리브레이션 적용)
        
        Args:
            X: 피처 데이터 (regime 컬럼 제외)
            regime: 레짐 값 (1, 0, -1)
        
        Returns:
            예측 확률
        """
        regime_name = {1: 'up', 0: 'flat', -1: 'down'}.get(regime)
        
        if regime_name not in self.regime_models:
            raise ValueError(f"모델 없음: {regime_name}")
        
        # 스케일링
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns
        )
        
        # 예측 (캘리브레이션 자동 적용)
        model = self.regime_models[regime_name]
        proba = model.predict_proba(X_scaled)
        
        return proba


# ============================================================
# 테스트 및 검증
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ModelTrainer 테스트 (레짐별 앙상블 + 캘리브레이션 + 버전 관리)")
    print("=" * 60)
    
    # 테스트 데이터 생성
    np.random.seed(42)
    n_samples = 3000
    
    X = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples),
        'regime_4h': np.random.choice([1, 0, -1], n_samples),
        'regime_1h': np.random.choice([1, 0, -1], n_samples),
        'regime_score': np.random.randn(n_samples),
        'regime': np.random.choice([1, 0, -1], n_samples)
    })
    
    y = pd.Series((X['feature_1'] + X['feature_2'] > 0).astype(int))
    
    print(f"\n📊 테스트 데이터: {len(X)}개")
    print(f"  타겟 분포: UP={y.mean():.2%}")
    print(f"  레짐 분포: {X['regime'].value_counts().to_dict()}")
    
    # 트레이너 초기화
    trainer = ModelTrainer()
    
    # 레짐별 앙상블 학습
    results = trainer.train_regime_ensemble(X, y, n_learners=3, test_size=0.2)
    
    # 번들 저장
    bundle_path = trainer.save_bundle()
    
    # 번들 목록
    print("\n📋 저장된 번들:")
    bundles = trainer.list_bundles()
    for bundle in bundles:
        print(f"  - {bundle['name']} ({bundle['version']})")
    
    # 번들 로드 테스트
    print("\n🔄 번들 로드 테스트:")
    trainer2 = ModelTrainer()
    success = trainer2.load_bundle("latest")
    
    if success:
        # 예측 테스트
        X_test = X.head(10).drop(columns=['regime'])
        regime_test = 1
        
        proba = trainer2.predict_with_regime(X_test, regime_test)
        print(f"\n✅ 예측 테스트 (regime={regime_test}):")
        print(f"  확률: {proba[:5, 1]}")
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)
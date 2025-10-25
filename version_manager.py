"""
version_manager.py
버전 관리 및 롤백
- 재학습 시 자동 버전 생성
- 모델/피처/필터/데이터 버전 관리
- 백업 및 롤백 기능
- 버전 간 성능 비교
"""

import os
import json
import joblib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from config import Config


class VersionManager:
    """버전 관리 시스템"""
    
    def __init__(self):
        self.config = Config
        self.version_dir = self.config.VERSION_DIR
        self.model_dir = self.config.MODEL_DIR
        
        # 디렉토리 생성
        self._ensure_dirs()
        
        # 버전 메타데이터 파일
        self.metadata_file = self.version_dir / 'versions_metadata.json'
        self.metadata = self._load_metadata()
    
    def _ensure_dirs(self):
        """디렉토리 생성"""
        self.version_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_metadata(self) -> Dict:
        """버전 메타데이터 로드"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {'versions': [], 'current_version': None}
    
    def _save_metadata(self):
        """버전 메타데이터 저장"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
    
    # ==========================================
    # 버전 생성
    # ==========================================
    
    def create_version(
        self,
        model_trainer,
        metrics: Dict,
        description: str = "",
        tags: List[str] = None
    ) -> str:
        """
        새 버전 생성
        
        Parameters:
        -----------
        model_trainer : ModelTrainer
            학습된 모델
        metrics : dict
            성능 메트릭 (레짐별)
        description : str
            버전 설명
        tags : list of str
            태그 (예: ['retrain', 'emergency'])
        
        Returns:
        --------
        str: 버전 ID
        """
        # 버전 ID 생성
        version_id = self.config.get_version_string()
        version_path = self.version_dir / version_id
        version_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"버전 생성: {version_id}")
        print(f"{'='*60}")
        
        # ======================================
        # 1. 모델 저장
        # ======================================
        print("1. 모델 저장 중...")
        
        # 레짐별 번들 저장
        for rname, bundle in model_trainer.bundles.items():
            if bundle is not None:
                bundle_path = version_path / f"bundle_{rname}.pkl"
                joblib.dump(bundle, bundle_path)
                print(f"  [{rname}] 번들: {bundle_path}")
        
        # Legacy 모델 저장
        legacy_data = {
            'models': model_trainer.models,
            'scaler': model_trainer.scaler,
            'selected_features': model_trainer.selected_features,
            'feature_importance': model_trainer.feature_importance,
            'calibration_method': model_trainer.calib_method,
            'calibrator': model_trainer.calibrator,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        legacy_path = version_path / 'model.pkl'
        joblib.dump(legacy_data, legacy_path)
        print(f"  Legacy 모델: {legacy_path}")
        
        # ======================================
        # 2. 피처 정보 저장
        # ======================================
        print("2. 피처 정보 저장 중...")
        
        feature_info = {
            'selected_features': model_trainer.selected_features,
            'feature_count': len(model_trainer.selected_features) if model_trainer.selected_features else 0,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        feature_path = version_path / 'features.json'
        with open(feature_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        print(f"  피처 정보: {feature_path}")
        
        # 피처 중요도 저장
        if model_trainer.feature_importance is not None:
            importance_path = version_path / 'feature_importance.csv'
            model_trainer.feature_importance.to_csv(importance_path, index=False)
            print(f"  피처 중요도: {importance_path}")
        
        # ======================================
        # 3. 성능 메트릭 저장
        # ======================================
        print("3. 성능 메트릭 저장 중...")
        
        metrics_path = version_path / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  메트릭: {metrics_path}")
        
        # ======================================
        # 4. 설정 스냅샷 저장
        # ======================================
        print("4. 설정 스냅샷 저장 중...")
        
        config_snapshot = {
            'CUT_ON': self.config.CUT_ON,
            'CUT_OFF': self.config.CUT_OFF,
            'TTL_MIN_SECONDS': self.config.TTL_MIN_SECONDS,
            'TTL_MAX_SECONDS': self.config.TTL_MAX_SECONDS,
            'DP_MIN': self.config.DP_MIN,
            'REFRACTORY_MINUTES': self.config.REFRACTORY_MINUTES,
            'REGIME_TIMEFRAMES': self.config.REGIME_TIMEFRAMES,
            'REGIME_WEIGHTS': self.config.REGIME_WEIGHTS,
            'REGIME_ADX_THR': self.config.REGIME_ADX_THR,
            'CALIBRATION_METHOD': self.config.CALIBRATION_METHOD,
            'ENSEMBLE_MODELS': self.config.ENSEMBLE_MODELS,
            'TOP_K_FEATURES': self.config.TOP_K_FEATURES,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        config_path = version_path / 'config_snapshot.json'
        with open(config_path, 'w') as f:
            json.dump(config_snapshot, f, indent=2)
        print(f"  설정: {config_path}")
        
        # ======================================
        # 5. 메타데이터 업데이트
        # ======================================
        print("5. 메타데이터 업데이트 중...")
        
        # 평균 성능 계산
        avg_acc = 0
        count = 0
        for regime_metrics in metrics.values():
            if 'test' in regime_metrics:
                avg_acc += regime_metrics['test']['accuracy']
                count += 1
        avg_acc = avg_acc / count if count > 0 else 0
        
        version_meta = {
            'version_id': version_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'description': description,
            'tags': tags or [],
            'metrics': {
                'avg_test_accuracy': avg_acc,
                'regime_count': len(metrics)
            },
            'config': {
                'cut_on': self.config.CUT_ON,
                'cut_off': self.config.CUT_OFF,
                'regime_timeframes': self.config.REGIME_TIMEFRAMES
            }
        }
        
        self.metadata['versions'].append(version_meta)
        self.metadata['current_version'] = version_id
        self._save_metadata()
        
        print(f"  메타데이터 업데이트 완료")
        
        print(f"\n✓ 버전 생성 완료: {version_id}")
        print(f"  경로: {version_path}")
        print(f"  평균 테스트 정확도: {avg_acc:.4f}")
        print(f"{'='*60}\n")
        
        return version_id
    
    # ==========================================
    # 버전 로드
    # ==========================================
    
    def load_version(self, version_id: str, model_trainer) -> bool:
        """
        특정 버전 로드
        
        Parameters:
        -----------
        version_id : str
            버전 ID
        model_trainer : ModelTrainer
            로드할 모델 트레이너
        
        Returns:
        --------
        bool: 성공 여부
        """
        version_path = self.version_dir / version_id
        
        if not version_path.exists():
            print(f"❌ 버전 없음: {version_id}")
            return False
        
        print(f"\n{'='*60}")
        print(f"버전 로드: {version_id}")
        print(f"{'='*60}")
        
        try:
            # 레짐별 번들 로드
            for rname in ['UP', 'DOWN', 'FLAT']:
                bundle_path = version_path / f"bundle_{rname}.pkl"
                if bundle_path.exists():
                    model_trainer.bundles[rname] = joblib.load(bundle_path)
                    print(f"  [{rname}] 번들 로드: {bundle_path}")
                else:
                    model_trainer.bundles[rname] = None
            
            # Legacy 모델 로드
            legacy_path = version_path / 'model.pkl'
            if legacy_path.exists():
                data = joblib.load(legacy_path)
                model_trainer.models = data['models']
                model_trainer.scaler = data['scaler']
                model_trainer.selected_features = data['selected_features']
                model_trainer.feature_importance = data.get('feature_importance')
                model_trainer.calib_method = data.get('calibration_method', 'isotonic')
                model_trainer.calibrator = data.get('calibrator')
                print(f"  Legacy 모델 로드: {legacy_path}")
            
            # 메타데이터 업데이트
            self.metadata['current_version'] = version_id
            self._save_metadata()
            
            print(f"✓ 버전 로드 완료: {version_id}")
            print(f"{'='*60}\n")
            
            return True
            
        except Exception as e:
            print(f"❌ 버전 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # ==========================================
    # 롤백
    # ==========================================
    
    def rollback(self, model_trainer, steps: int = 1) -> bool:
        """
        이전 버전으로 롤백
        
        Parameters:
        -----------
        model_trainer : ModelTrainer
            모델 트레이너
        steps : int
            되돌릴 단계 수 (1 = 바로 이전)
        
        Returns:
        --------
        bool: 성공 여부
        """
        if len(self.metadata['versions']) < steps + 1:
            print(f"❌ 롤백 불가: {steps}단계 이전 버전 없음")
            return False
        
        target_version = self.metadata['versions'][-(steps + 1)]
        version_id = target_version['version_id']
        
        print(f"\n{'='*60}")
        print(f"롤백: {steps}단계 → {version_id}")
        print(f"{'='*60}")
        
        success = self.load_version(version_id, model_trainer)
        
        if success:
            print(f"✓ 롤백 완료: {version_id}")
        
        return success
    
    # ==========================================
    # 버전 비교
    # ==========================================
    
    def compare_versions(self, version_id1: str, version_id2: str) -> Dict:
        """
        두 버전 비교
        
        Parameters:
        -----------
        version_id1 : str
            첫 번째 버전
        version_id2 : str
            두 번째 버전
        
        Returns:
        --------
        dict: 비교 결과
        """
        path1 = self.version_dir / version_id1 / 'metrics.json'
        path2 = self.version_dir / version_id2 / 'metrics.json'
        
        if not path1.exists() or not path2.exists():
            return {'error': '버전 메트릭 없음'}
        
        with open(path1, 'r') as f:
            metrics1 = json.load(f)
        
        with open(path2, 'r') as f:
            metrics2 = json.load(f)
        
        comparison = {
            'version1': version_id1,
            'version2': version_id2,
            'regime_comparison': {}
        }
        
        for regime in metrics1.keys():
            if regime in metrics2:
                m1 = metrics1[regime].get('test', {})
                m2 = metrics2[regime].get('test', {})
                
                comparison['regime_comparison'][regime] = {
                    'accuracy_diff': m2.get('accuracy', 0) - m1.get('accuracy', 0),
                    'win_rate_diff': m2.get('win_rate', 0) - m1.get('win_rate', 0),
                    'v1_accuracy': m1.get('accuracy', 0),
                    'v2_accuracy': m2.get('accuracy', 0)
                }
        
        return comparison
    
    def print_comparison(self, version_id1: str, version_id2: str):
        """버전 비교 결과 출력"""
        comp = self.compare_versions(version_id1, version_id2)
        
        if 'error' in comp:
            print(f"❌ {comp['error']}")
            return
        
        print(f"\n{'='*60}")
        print(f"버전 비교")
        print(f"{'='*60}")
        print(f"Version 1: {comp['version1']}")
        print(f"Version 2: {comp['version2']}")
        print(f"\n레짐별 성능 차이:")
        
        for regime, diff in comp['regime_comparison'].items():
            print(f"\n[{regime}]")
            print(f"  V1 정확도: {diff['v1_accuracy']:.4f}")
            print(f"  V2 정확도: {diff['v2_accuracy']:.4f}")
            print(f"  차이: {diff['accuracy_diff']:+.4f} ({diff['accuracy_diff']*100:+.2f}%p)")
            print(f"  승률 차이: {diff['win_rate_diff']:+.4f}")
    
    # ==========================================
    # 버전 목록 및 정보
    # ==========================================
    
    def list_versions(self) -> List[Dict]:
        """버전 목록 반환"""
        return self.metadata['versions']
    
    def print_versions(self, limit: int = 10):
        """버전 목록 출력"""
        versions = self.metadata['versions'][-limit:]
        
        print(f"\n{'='*60}")
        print(f"버전 목록 (최근 {len(versions)}개)")
        print(f"{'='*60}")
        
        for v in reversed(versions):
            current = " [현재]" if v['version_id'] == self.metadata['current_version'] else ""
            print(f"\n버전: {v['version_id']}{current}")
            print(f"  시간: {v['timestamp']}")
            print(f"  설명: {v.get('description', '-')}")
            print(f"  태그: {', '.join(v.get('tags', []))}")
            print(f"  평균 정확도: {v['metrics']['avg_test_accuracy']:.4f}")
    
    def get_current_version(self) -> Optional[str]:
        """현재 버전 ID 반환"""
        return self.metadata.get('current_version')
    
    def get_version_info(self, version_id: str) -> Optional[Dict]:
        """특정 버전 정보 반환"""
        for v in self.metadata['versions']:
            if v['version_id'] == version_id:
                return v
        return None
    
    # ==========================================
    # 백업 및 복원
    # ==========================================
    
    def backup_current_model(self, reason: str = "manual_backup") -> str:
        """
        현재 모델 백업
        
        Parameters:
        -----------
        reason : str
            백업 사유
        
        Returns:
        --------
        str: 백업 경로
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = self.config.BACKUP_DIR / f"backup_{timestamp}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n현재 모델 백업 중...")
        print(f"  사유: {reason}")
        print(f"  경로: {backup_path}")
        
        # 모델 파일 복사
        for rname in ['UP', 'DOWN', 'FLAT']:
            src = self.model_dir / f"bundle_{rname}.pkl"
            if src.exists():
                dst = backup_path / f"bundle_{rname}.pkl"
                shutil.copy2(src, dst)
        
        src = self.model_dir / 'current_model.pkl'
        if src.exists():
            dst = backup_path / 'current_model.pkl'
            shutil.copy2(src, dst)
        
        # 백업 정보 저장
        backup_info = {
            'timestamp': timestamp,
            'reason': reason,
            'current_version': self.metadata.get('current_version'),
            'backup_time': datetime.now(timezone.utc).isoformat()
        }
        
        info_path = backup_path / 'backup_info.json'
        with open(info_path, 'w') as f:
            json.dump(backup_info, f, indent=2)
        
        print(f"✓ 백업 완료: {backup_path}")
        
        return str(backup_path)
    
    def list_backups(self) -> List[Dict]:
        """백업 목록 반환"""
        backups = []
        
        if not self.config.BACKUP_DIR.exists():
            return backups
        
        for backup_dir in sorted(self.config.BACKUP_DIR.iterdir(), reverse=True):
            if not backup_dir.is_dir():
                continue
            
            info_path = backup_dir / 'backup_info.json'
            if info_path.exists():
                with open(info_path, 'r') as f:
                    info = json.load(f)
                info['path'] = str(backup_dir)
                backups.append(info)
        
        return backups
    
    def print_backups(self, limit: int = 10):
        """백업 목록 출력"""
        backups = self.list_backups()[:limit]
        
        print(f"\n{'='*60}")
        print(f"백업 목록 (최근 {len(backups)}개)")
        print(f"{'='*60}")
        
        for b in backups:
            print(f"\n타임스탬프: {b['timestamp']}")
            print(f"  사유: {b['reason']}")
            print(f"  버전: {b.get('current_version', '-')}")
            print(f"  경로: {b['path']}")


# ==========================================
# 테스트
# ==========================================
if __name__ == "__main__":
    from model_train import ModelTrainer
    from feature_engineer import FeatureEngineer
    import numpy as np
    
    Config.create_directories()
    
    # 샘플 데이터 생성
    print("샘플 데이터 생성...")
    periods = 3000
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
    
    # 피처 생성
    print("\n피처 생성...")
    fe = FeatureEngineer()
    features = fe.create_feature_pool(df_1m, lookback_bars=50)
    
    # 라벨 생성
    from timeframe_manager import TimeframeManager
    tf_manager = TimeframeManager()
    df_30m = tf_manager.aggregate_1m_to_30m(df_1m)
    target = fe.create_target_30m(df_30m)
    
    valid = target.notna() & (target.index >= 50)
    X = features
    y = target[valid].reset_index(drop=True)
    
    min_len = min(len(X), len(y))
    X = X.iloc[:min_len]
    y = y.iloc[:min_len]
    
    # 모델 학습
    print("\n모델 학습...")
    trainer = ModelTrainer(Config)
    trainer.feature_selection_regime(X, y, regime_col='regime', top_k=20)
    metrics = trainer.train_ensemble_regime(X, y, regime_col='regime', test_size=0.2)
    
    # 버전 관리자 생성
    print("\n버전 관리 테스트...")
    vm = VersionManager()
    
    # 버전 생성
    version_id = vm.create_version(
        trainer, 
        metrics, 
        description="초기 모델",
        tags=['initial', 'test']
    )
    
    # 백업
    backup_path = vm.backup_current_model(reason="테스트 백업")
    
    # 버전 목록
    vm.print_versions()
    
    # 백업 목록
    vm.print_backups()
    
    print("\n✓ 버전 관리 테스트 완료")
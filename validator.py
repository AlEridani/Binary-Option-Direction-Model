"""
validator.py
재학습 전후 자동 검증
- 재학습 전: PSI, KS, 샘플 수, NaN 검사
- 재학습 후: 회귀 테스트 (체결률 ±10% 이내)
- 데이터 분포 변화 감지
- 모델 분별력 검증
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from config import Config


class Validator:
    """재학습 전후 검증"""
    
    def __init__(self):
        self.config = Config
        
        # 임계값
        self.psi_threshold = self.config.RETRAIN_PSI_THRESHOLD
        self.ks_threshold = self.config.RETRAIN_KS_THRESHOLD
        self.min_samples = self.config.RETRAIN_MIN_SAMPLES_PER_REGIME
        self.entry_rate_tolerance = self.config.RETRAIN_ENTRY_RATE_TOLERANCE
    
    # ==========================================
    # PSI (Population Stability Index)
    # ==========================================
    
    def calculate_psi(
        self, 
        expected: np.ndarray, 
        actual: np.ndarray, 
        bins: int = 10
    ) -> float:
        """
        PSI 계산 (데이터 분포 변화)
        
        Parameters:
        -----------
        expected : array
            기준 데이터 (이전)
        actual : array
            비교 데이터 (현재)
        bins : int
            히스토그램 구간 수
        
        Returns:
        --------
        float: PSI 값 (< 0.1: 안정, 0.1~0.2: 주의, > 0.2: 재학습 필요)
        """
        # NaN 제거
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]
        
        if len(expected) == 0 or len(actual) == 0:
            return np.inf
        
        # 동일한 구간으로 히스토그램 생성
        breakpoints = np.linspace(
            min(expected.min(), actual.min()),
            max(expected.max(), actual.max()),
            bins + 1
        )
        
        expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)
        
        # 0으로 나누기 방지
        expected_percents = np.where(expected_percents == 0, 0.0001, expected_percents)
        actual_percents = np.where(actual_percents == 0, 0.0001, actual_percents)
        
        # PSI 계산
        psi = np.sum((actual_percents - expected_percents) * 
                     np.log(actual_percents / expected_percents))
        
        return float(psi)
    
    def calculate_feature_psi(
        self, 
        df_old: pd.DataFrame, 
        df_new: pd.DataFrame, 
        feature_cols: list,
        top_k: int = 10
    ) -> Dict:
        """
        피처별 PSI 계산
        
        Parameters:
        -----------
        df_old : DataFrame
            이전 데이터
        df_new : DataFrame
            새 데이터
        feature_cols : list
            피처 컬럼 목록
        top_k : int
            PSI 상위 몇 개 출력
        
        Returns:
        --------
        dict: {
            'psi_scores': dict,
            'max_psi': float,
            'unstable_features': list
        }
        """
        psi_scores = {}
        
        for col in feature_cols:
            if col not in df_old.columns or col not in df_new.columns:
                continue
            
            try:
                old_vals = df_old[col].values
                new_vals = df_new[col].values
                
                psi = self.calculate_psi(old_vals, new_vals)
                psi_scores[col] = psi
            except:
                continue
        
        # 상위 PSI 피처
        sorted_psi = sorted(psi_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 불안정 피처 (PSI > 0.2)
        unstable = [k for k, v in psi_scores.items() if v > 0.2]
        
        max_psi = max(psi_scores.values()) if psi_scores else 0.0
        
        return {
            'psi_scores': dict(sorted_psi[:top_k]),
            'max_psi': max_psi,
            'unstable_features': unstable,
            'total_features': len(psi_scores)
        }
    
    # ==========================================
    # KS (Kolmogorov-Smirnov)
    # ==========================================
    
    def calculate_ks(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> float:
        """
        KS 통계량 계산 (모델 분별력)
        
        Parameters:
        -----------
        y_true : array
            실제 라벨 (0/1)
        y_pred : array
            예측 확률 (0~1)
        
        Returns:
        --------
        float: KS 통계량 (> 0.2: 좋음, > 0.4: 매우 좋음)
        """
        # NaN 제거
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return 0.0
        
        # 양성/음성 분포
        pos_pred = y_pred[y_true == 1]
        neg_pred = y_pred[y_true == 0]
        
        if len(pos_pred) == 0 or len(neg_pred) == 0:
            return 0.0
        
        # KS 통계량
        ks_stat, _ = stats.ks_2samp(pos_pred, neg_pred)
        
        return float(ks_stat)
    
    # ==========================================
    # 재학습 전 검증
    # ==========================================
    
    def validate_before_retrain(
        self,
        X_new: pd.DataFrame,
        y_new: pd.Series,
        X_old: Optional[pd.DataFrame] = None,
        y_old: Optional[pd.Series] = None,
        model_trainer = None
    ) -> Dict:
        """
        재학습 전 검증
        
        Parameters:
        -----------
        X_new : DataFrame
            새 피처 데이터
        y_new : Series
            새 라벨
        X_old : DataFrame or None
            이전 피처 데이터 (PSI 계산용)
        y_old : Series or None
            이전 라벨
        model_trainer : ModelTrainer or None
            현재 모델 (KS 계산용)
        
        Returns:
        --------
        dict: {
            'passed': bool,
            'checks': dict,
            'errors': list,
            'warnings': list
        }
        """
        print(f"\n{'='*60}")
        print(f"재학습 전 검증")
        print(f"{'='*60}")
        
        checks = {}
        errors = []
        warnings = []
        
        # ======================================
        # 1. 샘플 수 검사
        # ======================================
        print("\n1. 샘플 수 검사...")
        
        total_samples = len(X_new)
        checks['total_samples'] = total_samples
        
        print(f"  전체 샘플: {total_samples:,}건")
        
        if total_samples < self.min_samples:
            errors.append(f"전체 샘플 부족: {total_samples} < {self.min_samples}")
        
        # 레짐별 샘플 수
        if 'regime' in X_new.columns:
            regime_counts = X_new['regime'].value_counts()
            checks['regime_samples'] = regime_counts.to_dict()
            
            print(f"  레짐별 샘플:")
            for regime, count in regime_counts.items():
                regime_name = ['FLAT', 'UP', 'DOWN'][int(regime) + 1] if regime in [-1, 0, 1] else 'UNKNOWN'
                print(f"    {regime_name:5s}: {count:6,}건", end='')
                
                if count < self.min_samples:
                    errors.append(f"레짐 {regime_name} 샘플 부족: {count} < {self.min_samples}")
                    print(" ❌")
                else:
                    print(" ✓")
        
        # ======================================
        # 2. NaN 검사
        # ======================================
        print("\n2. NaN 검사...")
        
        nan_cols = X_new.columns[X_new.isna().any()].tolist()
        checks['nan_columns'] = nan_cols
        
        if nan_cols:
            nan_counts = {col: X_new[col].isna().sum() for col in nan_cols}
            checks['nan_counts'] = nan_counts
            errors.append(f"NaN 발견: {len(nan_cols)}개 컬럼")
            
            print(f"  ❌ NaN 발견:")
            for col, count in list(nan_counts.items())[:10]:
                print(f"    {col}: {count:,}건")
        else:
            print(f"  ✓ NaN 없음")
        
        # ======================================
        # 3. PSI 검사 (데이터 분포 변화)
        # ======================================
        if X_old is not None and len(X_old) > 0:
            print("\n3. PSI 검사 (데이터 분포 변화)...")
            
            # 피처 컬럼 추출
            feature_cols = [c for c in X_new.columns 
                            if c not in ['timestamp', 'bar30_start', 'bar30_end',
                                          'm1_index_entry', 'm1_index_label',
                                          'regime', 'regime_final', 'is_weekend']]
            
            psi_result = self.calculate_feature_psi(
                X_old, X_new, feature_cols, top_k=10
            )
            
            checks['psi'] = psi_result
            
            print(f"  최대 PSI: {psi_result['max_psi']:.4f} (임계값: {self.psi_threshold})")
            
            if psi_result['max_psi'] > self.psi_threshold:
                warnings.append(f"PSI 높음: {psi_result['max_psi']:.4f} > {self.psi_threshold}")
                print(f"  ⚠️ 데이터 분포 변화 감지")
            else:
                print(f"  ✓ 데이터 분포 안정")
            
            print(f"\n  상위 PSI 피처:")
            for feat, psi in list(psi_result['psi_scores'].items())[:5]:
                status = "⚠️" if psi > 0.2 else "✓"
                print(f"    {status} {feat}: {psi:.4f}")
        else:
            print("\n3. PSI 검사 스킵 (이전 데이터 없음)")
            checks['psi'] = None
        
        # ======================================
        # 4. KS 검사 (모델 분별력)
        # ======================================
        if model_trainer is not None and y_new is not None:
            print("\n4. KS 검사 (현재 모델 분별력)...")
            
            try:
                # 예측
                y_pred = model_trainer.predict_proba_df(X_new)
                
                # KS 계산
                ks_stat = self.calculate_ks(y_new.values, y_pred)
                checks['ks_stat'] = ks_stat
                
                print(f"  KS 통계량: {ks_stat:.4f} (임계값: {self.ks_threshold})")
                
                if ks_stat < self.ks_threshold:
                    warnings.append(f"KS 낮음: {ks_stat:.4f} < {self.ks_threshold}")
                    print(f"  ⚠️ 모델 분별력 저하")
                else:
                    print(f"  ✓ 모델 분별력 양호")
                
            except Exception as e:
                print(f"  ⚠️ KS 계산 실패: {e}")
                checks['ks_stat'] = None
        else:
            print("\n4. KS 검사 스킵")
            checks['ks_stat'] = None
        
        # ======================================
        # 5. 결과 정리
        # ======================================
        passed = len(errors) == 0
        
        print(f"\n{'='*60}")
        if passed:
            print(f"✓ 재학습 전 검증 통과")
            if warnings:
                print(f"  경고: {len(warnings)}개")
                for w in warnings:
                    print(f"    - {w}")
        else:
            print(f"❌ 재학습 전 검증 실패")
            print(f"  오류: {len(errors)}개")
            for e in errors:
                print(f"    - {e}")
        print(f"{'='*60}\n")
        
        return {
            'passed': passed,
            'checks': checks,
            'errors': errors,
            'warnings': warnings,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    # ==========================================
    # 재학습 후 검증
    # ==========================================
    
    def validate_after_retrain(
        self,
        old_trainer,
        new_trainer,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """
        재학습 후 회귀 테스트
        
        Parameters:
        -----------
        old_trainer : ModelTrainer
            이전 모델
        new_trainer : ModelTrainer
            새 모델
        X_test : DataFrame
            테스트 데이터
        y_test : Series
            테스트 라벨
        
        Returns:
        --------
        dict: {
            'passed': bool,
            'old_metrics': dict,
            'new_metrics': dict,
            'improvement': float,
            'errors': list
        }
        """
        print(f"\n{'='*60}")
        print(f"재학습 후 회귀 테스트")
        print(f"{'='*60}")
        
        errors = []
        
        # ======================================
        # 1. 이전 모델 성능
        # ======================================
        print("\n1. 이전 모델 성능 측정...")
        
        try:
            old_pred = old_trainer.predict_proba_df(X_test)
            old_acc = np.mean((old_pred > 0.5).astype(int) == y_test.values)
            old_entry_rate = np.mean((old_pred > self.config.CUT_ON) | 
                                      (old_pred < (1 - self.config.CUT_ON)))
            
            old_metrics = {
                'accuracy': float(old_acc),
                'entry_rate': float(old_entry_rate)
            }
            
            print(f"  정확도: {old_acc:.4f}")
            print(f"  체결률: {old_entry_rate:.4f}")
            
        except Exception as e:
            print(f"  ❌ 측정 실패: {e}")
            old_metrics = {'accuracy': 0.0, 'entry_rate': 0.0}
            errors.append(f"이전 모델 평가 실패: {e}")
        
        # ======================================
        # 2. 새 모델 성능
        # ======================================
        print("\n2. 새 모델 성능 측정...")
        
        try:
            new_pred = new_trainer.predict_proba_df(X_test)
            new_acc = np.mean((new_pred > 0.5).astype(int) == y_test.values)
            new_entry_rate = np.mean((new_pred > self.config.CUT_ON) | 
                                      (new_pred < (1 - self.config.CUT_ON)))
            
            new_metrics = {
                'accuracy': float(new_acc),
                'entry_rate': float(new_entry_rate)
            }
            
            print(f"  정확도: {new_acc:.4f}")
            print(f"  체결률: {new_entry_rate:.4f}")
            
        except Exception as e:
            print(f"  ❌ 측정 실패: {e}")
            new_metrics = {'accuracy': 0.0, 'entry_rate': 0.0}
            errors.append(f"새 모델 평가 실패: {e}")
        
        # ======================================
        # 3. 체결률 회귀 테스트
        # ======================================
        print("\n3. 체결률 회귀 테스트...")
        
        entry_rate_change = abs(new_metrics['entry_rate'] - old_metrics['entry_rate'])
        entry_rate_change_pct = entry_rate_change / (old_metrics['entry_rate'] + 1e-9)
        
        print(f"  이전 체결률: {old_metrics['entry_rate']:.4f}")
        print(f"  새 체결률: {new_metrics['entry_rate']:.4f}")
        print(f"  변화율: {entry_rate_change_pct:.2%} (허용: ±{self.entry_rate_tolerance:.0%})")
        
        if entry_rate_change_pct > self.entry_rate_tolerance:
            errors.append(
                f"체결률 급변: {entry_rate_change_pct:.2%} > {self.entry_rate_tolerance:.0%}"
            )
            print(f"  ❌ 체결률 급변 감지")
        else:
            print(f"  ✓ 체결률 안정")
        
        # ======================================
        # 4. 성능 개선 확인
        # ======================================
        print("\n4. 성능 개선 확인...")
        
        improvement = new_metrics['accuracy'] - old_metrics['accuracy']
        
        print(f"  정확도 개선: {improvement:+.4f} ({improvement*100:+.2f}%p)")
        
        if improvement < 0:
            print(f"  ⚠️ 성능 저하")
        else:
            print(f"  ✓ 성능 개선")
        
        # ======================================
        # 5. 결과 정리
        # ======================================
        passed = len(errors) == 0
        
        print(f"\n{'='*60}")
        if passed:
            print(f"✓ 재학습 후 회귀 테스트 통과")
        else:
            print(f"❌ 재학습 후 회귀 테스트 실패")
            print(f"  오류: {len(errors)}개")
            for e in errors:
                print(f"    - {e}")
        print(f"{'='*60}\n")
        
        return {
            'passed': passed,
            'old_metrics': old_metrics,
            'new_metrics': new_metrics,
            'improvement': float(improvement),
            'entry_rate_change_pct': float(entry_rate_change_pct),
            'errors': errors,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    # ==========================================
    # 전체 검증 파이프라인
    # ==========================================
    
    def full_validation_pipeline(
        self,
        X_new: pd.DataFrame,
        y_new: pd.Series,
        old_trainer,
        new_trainer,
        X_old: Optional[pd.DataFrame] = None,
        y_old: Optional[pd.Series] = None,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None
    ) -> Dict:
        """
        전체 검증 파이프라인 (재학습 전후)
        
        Returns:
        --------
        dict: {
            'before_retrain': dict,
            'after_retrain': dict,
            'overall_passed': bool
        }
        """
        print(f"\n{'='*60}")
        print(f"전체 검증 파이프라인")
        print(f"{'='*60}")
        
        # 재학습 전 검증
        before_result = self.validate_before_retrain(
            X_new, y_new, X_old, y_old, old_trainer
        )
        
        # 재학습 후 검증
        after_result = None
        if X_test is not None and y_test is not None:
            after_result = self.validate_after_retrain(
                old_trainer, new_trainer, X_test, y_test
            )
        
        overall_passed = before_result['passed']
        if after_result is not None:
            overall_passed = overall_passed and after_result['passed']
        
        print(f"\n{'='*60}")
        print(f"전체 검증 결과")
        print(f"{'='*60}")
        print(f"재학습 전: {'✓ 통과' if before_result['passed'] else '❌ 실패'}")
        if after_result:
            print(f"재학습 후: {'✓ 통과' if after_result['passed'] else '❌ 실패'}")
        print(f"종합: {'✓ 통과' if overall_passed else '❌ 실패'}")
        print(f"{'='*60}\n")
        
        return {
            'before_retrain': before_result,
            'after_retrain': after_result,
            'overall_passed': overall_passed,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }


# ==========================================
# 테스트
# ==========================================
if __name__ == "__main__":
    from feature_engineer import FeatureEngineer
    from model_train import ModelTrainer
    
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
    
    # 데이터 분할
    split = int(len(X) * 0.7)
    X_old = X.iloc[:split]
    y_old = y.iloc[:split]
    X_new = X.iloc[split:]
    y_new = y.iloc[split:]
    
    # 이전 모델 학습
    print("\n이전 모델 학습...")
    old_trainer = ModelTrainer(Config)
    old_trainer.feature_selection_regime(X_old, y_old, regime_col='regime', top_k=20)
    old_trainer.train_ensemble_regime(X_old, y_old, regime_col='regime', test_size=0.2)
    
    # Validator 생성
    validator = Validator()
    
    # 재학습 전 검증
    before_result = validator.validate_before_retrain(
        X_new, y_new, X_old, y_old, old_trainer
    )
    
    # 새 모델 학습
    if before_result['passed']:
        print("\n새 모델 학습...")
        new_trainer = ModelTrainer(Config)
        new_trainer.selected_features = old_trainer.selected_features
        new_trainer.train_ensemble_regime(X_new, y_new, regime_col='regime', test_size=0.2)
        
        # 재학습 후 검증
        test_split = int(len(X_new) * 0.8)
        X_test = X_new.iloc[test_split:]
        y_test = y_new.iloc[test_split:]
        
        after_result = validator.validate_after_retrain(
            old_trainer, new_trainer, X_test, y_test
        )
    
    print("\n✓ 검증 테스트 완료")
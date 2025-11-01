"""
모니터링 시스템 - 캘리브레이션 검증 중심 + 성능 추적 + 버그 탐지
버전: 1.4.0

핵심 기능:
1. 캘리브레이션 검증 (최우선)
   - 예측 확률 vs 실제 승률
   - ECE (Expected Calibration Error)
   - 확률 구간별 정확도
   - Brier Score, Log Loss
2. 방향별 성과 (UP/DOWN)
3. 확률 구간별 분석
4. 레짐별/모델별 승률
5. 시스템 모니터링
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import json
import psutil
from pathlib import Path
from collections import defaultdict
import json

import config
from log_manager import LogManager


class CalibrationMetrics:
    """
    캘리브레이션 메트릭 계산
    
    - ECE (Expected Calibration Error)
    - Brier Score
    - Log Loss
    - 확률 구간별 정확도
    """
    
    @staticmethod
    def calculate_ece(probabilities: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> Tuple[float, Dict]:
        """
        ECE (Expected Calibration Error) 계산
        
        Args:
            probabilities: 예측 확률 배열
            outcomes: 실제 결과 (1: 승리, 0: 패배)
            n_bins: 구간 수
        
        Returns:
            (ece_value, bin_details)
        """
        if len(probabilities) == 0:
            return 0.0, {}
        
        # 구간 정의
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        bin_details = {}
        total_samples = len(probabilities)
        ece = 0.0
        
        for i in range(n_bins):
            # 현재 구간에 속하는 샘플
            in_bin = (probabilities >= bin_edges[i]) & (probabilities < bin_edges[i + 1])
            
            if i == n_bins - 1:  # 마지막 구간은 상한 포함
                in_bin = (probabilities >= bin_edges[i]) & (probabilities <= bin_edges[i + 1])
            
            n_samples = in_bin.sum()
            
            if n_samples == 0:
                continue
            
            # 평균 예측 확률
            avg_prob = probabilities[in_bin].mean()
            
            # 실제 승률
            actual_rate = outcomes[in_bin].mean()
            
            # 가중 오차
            weight = n_samples / total_samples
            error = abs(avg_prob - actual_rate)
            ece += weight * error
            
            bin_details[f"bin_{i}"] = {
                'range': f"[{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f})",
                'n_samples': int(n_samples),
                'avg_prob': float(avg_prob),
                'actual_rate': float(actual_rate),
                'error': float(error)
            }
        
        return float(ece), bin_details
    
    @staticmethod
    def calculate_brier_score(probabilities: np.ndarray, outcomes: np.ndarray) -> float:
        """
        Brier Score 계산 (낮을수록 좋음)
        
        BS = (1/N) * Σ(p_i - y_i)^2
        """
        if len(probabilities) == 0:
            return 0.0
        
        return float(np.mean((probabilities - outcomes) ** 2))
    
    @staticmethod
    def calculate_log_loss(probabilities: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
        """
        Log Loss 계산 (낮을수록 좋음)
        
        LL = -(1/N) * Σ[y_i*log(p_i) + (1-y_i)*log(1-p_i)]
        """
        if len(probabilities) == 0:
            return 0.0
        
        # Clip to avoid log(0)
        p = np.clip(probabilities, eps, 1 - eps)
        
        log_loss = -np.mean(
            outcomes * np.log(p) + (1 - outcomes) * np.log(1 - p)
        )
        
        return float(log_loss)


class PerformanceMonitor:
    """
    성능 모니터링
    - 캘리브레이션 검증
    - 승률/손익/진입률 추적
    - 레짐별/모델별/방향별 분석
    - 확률 구간별 분석
    - 연속 손실 추적
    """
    
    def __init__(self, log_manager: LogManager):
        self.log_manager = log_manager
        self.snapshot_history = []
    
    def validate_calibration(self, window: int = 200) -> Dict:
        """
        캘리브레이션 검증 (핵심 기능)
        
        예측 확률 vs 실제 승률 비교
        """
        df = self.log_manager.load_recent_trades(n=window)
        
        if df.empty:
            return {
                'valid': False,
                'reason': '데이터 없음',
                'n_samples': 0
            }
        
        df_closed = df[df['status'] == 'CLOSED'].copy()
        
        if df_closed.empty or len(df_closed) < 20:
            return {
                'valid': False,
                'reason': f'샘플 부족 ({len(df_closed)}건)',
                'n_samples': len(df_closed)
            }
        
        # 확률 데이터 추출 (p_cal_at_entry 우선, 없으면 p_raw_at_entry)
        if 'p_cal_at_entry' in df_closed.columns:
            prob_col = 'p_cal_at_entry'
        elif 'p_raw_at_entry' in df_closed.columns:
            prob_col = 'p_raw_at_entry'
        else:
            return {
                'valid': False,
                'reason': '확률 컬럼 없음',
                'n_samples': len(df_closed)
            }
        
        # 결측치 제거
        df_valid = df_closed[[prob_col, 'result']].dropna()
        
        if df_valid.empty:
            return {
                'valid': False,
                'reason': '유효한 데이터 없음',
                'n_samples': 0
            }
        
        # 확률과 결과 추출
        probabilities = df_valid[prob_col].values
        outcomes = (df_valid['result'] == 'WIN').astype(int).values
        
        # 메트릭 계산
        ece, bin_details = CalibrationMetrics.calculate_ece(probabilities, outcomes, n_bins=10)
        brier = CalibrationMetrics.calculate_brier_score(probabilities, outcomes)
        log_loss = CalibrationMetrics.calculate_log_loss(probabilities, outcomes)
        
        # 전체 통계
        avg_prob = float(probabilities.mean())
        actual_wr = float(outcomes.mean())
        calibration_gap = abs(avg_prob - actual_wr)
        
        return {
            'valid': True,
            'n_samples': len(df_valid),
            'prob_column': prob_col,
            'metrics': {
                'ece': ece,
                'brier_score': brier,
                'log_loss': log_loss,
                'avg_predicted_prob': avg_prob,
                'actual_win_rate': actual_wr,
                'calibration_gap': calibration_gap
            },
            'bins': bin_details,
            'is_well_calibrated': ece < 0.05 and calibration_gap < 0.03
        }
    
    def analyze_by_probability_bins(self, window: int = 200, bin_width: float = 0.02) -> Dict:
        """
        확률 구간별 성과 분석
        
        Args:
            window: 분석할 거래 수
            bin_width: 구간 폭 (기본 2%)
        """
        df = self.log_manager.load_recent_trades(n=window)
        
        if df.empty:
            return {}
        
        df_closed = df[df['status'] == 'CLOSED'].copy()
        
        if df_closed.empty:
            return {}
        
        # 확률 컬럼 선택
        prob_col = 'p_cal_at_entry' if 'p_cal_at_entry' in df_closed.columns else 'p_raw_at_entry'
        
        if prob_col not in df_closed.columns:
            return {}
        
        df_valid = df_closed[[prob_col, 'result', 'side']].dropna()
        
        if df_valid.empty:
            return {}
        
        # 구간 생성
        bins = np.arange(0.5, 1.0 + bin_width, bin_width)
        df_valid['prob_bin'] = pd.cut(df_valid[prob_col], bins=bins, include_lowest=True)
        
        results = {}
        
        for bin_range in df_valid['prob_bin'].unique():
            if pd.isna(bin_range):
                continue
            
            df_bin = df_valid[df_valid['prob_bin'] == bin_range]
            
            n_trades = len(df_bin)
            n_wins = (df_bin['result'] == 'WIN').sum()
            win_rate = n_wins / n_trades if n_trades > 0 else 0.0
            
            avg_prob = df_bin[prob_col].mean()
            
            results[str(bin_range)] = {
                'n_trades': int(n_trades),
                'n_wins': int(n_wins),
                'win_rate': float(win_rate),
                'avg_prob': float(avg_prob),
                'error': float(abs(avg_prob - win_rate))
            }
        
        return results
    
    def analyze_by_direction(self, window: int = 200) -> Dict:
        """
        방향별(UP/DOWN) 성과 분석
        """
        df = self.log_manager.load_recent_trades(n=window)
        
        if df.empty or 'side' not in df.columns:
            return {}
        
        df_closed = df[df['status'] == 'CLOSED'].copy()
        
        if df_closed.empty:
            return {}
        
        results = {}
        
        for direction in ['LONG', 'SHORT']:
            df_dir = df_closed[df_closed['side'] == direction]
            
            if df_dir.empty:
                continue
            
            n_trades = len(df_dir)
            n_wins = (df_dir['result'] == 'WIN').sum()
            win_rate = n_wins / n_trades if n_trades > 0 else 0.0
            
            # 확률 통계
            prob_col = 'p_cal_at_entry' if 'p_cal_at_entry' in df_dir.columns else 'p_raw_at_entry'
            
            if prob_col in df_dir.columns:
                avg_prob = df_dir[prob_col].mean()
            else:
                avg_prob = 0.0
            
            results[direction] = {
                'n_trades': int(n_trades),
                'n_wins': int(n_wins),
                'n_losses': int(n_trades - n_wins),
                'win_rate': float(win_rate),
                'avg_prob': float(avg_prob),
                'calibration_gap': float(abs(avg_prob - win_rate))
            }
        
        return results
    
    def track_win_rate(self, window: int = 50) -> Dict:
        """
        승률 추적
        
        Args:
            window: 윈도우 크기
        
        Returns:
            승률 통계
        """
        df = self.log_manager.load_recent_trades(n=window)
        
        if df.empty:
            return {
                'win_rate': 0.0,
                'n_trades': 0,
                'n_wins': 0,
                'n_losses': 0
            }
        
        df_closed = df[df['status'] == 'CLOSED']
        
        if df_closed.empty:
            return {
                'win_rate': 0.0,
                'n_trades': 0,
                'n_wins': 0,
                'n_losses': 0
            }
        
        n_wins = (df_closed['result'] == 'WIN').sum()
        n_losses = (df_closed['result'] == 'LOSS').sum()
        n_total = len(df_closed)
        
        win_rate = n_wins / n_total if n_total > 0 else 0.0
        
        return {
            'win_rate': win_rate,
            'n_trades': n_total,
            'n_wins': n_wins,
            'n_losses': n_losses,
            'window': window
        }
    
    def track_win_rate_by_regime(self, window: int = 100) -> Dict:
        """레짐별 승률 추적"""
        df = self.log_manager.load_recent_trades(n=window)
        
        if df.empty:
            return {}
        
        df_closed = df[df['status'] == 'CLOSED']
        
        if df_closed.empty:
            return {}
        
        results = {}
        
        for regime in [1, 0, -1]:
            df_regime = df_closed[df_closed['regime'] == regime]
            
            if df_regime.empty:
                continue
            
            n_wins = (df_regime['result'] == 'WIN').sum()
            n_total = len(df_regime)
            
            regime_name = {1: 'UP', 0: 'FLAT', -1: 'DOWN'}.get(regime, f'REGIME_{regime}')
            
            results[regime_name] = {
                'win_rate': n_wins / n_total if n_total > 0 else 0.0,
                'n_trades': n_total,
                'n_wins': n_wins
            }
        
        return results
    
    def track_win_rate_by_model(self, window: int = 200) -> Dict:
        """모델 버전별 승률 추적"""
        df = self.log_manager.load_recent_trades(n=window)
        
        if df.empty or 'model_ver' not in df.columns:
            return {}
        
        df_closed = df[df['status'] == 'CLOSED']
        
        if df_closed.empty:
            return {}
        
        results = {}
        
        for model_ver in df_closed['model_ver'].unique():
            if pd.isna(model_ver):
                continue
            
            df_model = df_closed[df_closed['model_ver'] == model_ver]
            
            n_wins = (df_model['result'] == 'WIN').sum()
            n_total = len(df_model)
            
            results[model_ver] = {
                'win_rate': n_wins / n_total if n_total > 0 else 0.0,
                'n_trades': n_total,
                'n_wins': n_wins
            }
        
        return results
    
    def track_profit(self, window: int = 50) -> Dict:
        """손익 추적"""
        df = self.log_manager.load_recent_trades(n=window)
        
        if df.empty:
            return {
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'pnl_ratio': 0.0
            }
        
        df_closed = df[df['status'] == 'CLOSED']
        
        if df_closed.empty:
            return {
                'total_pnl': 0.0,
                'avg_pnl': 0.0,
                'pnl_ratio': 0.0
            }
        
        # 간단한 PnL 계산 (WIN: +payout, LOSS: -1)
        wins = df_closed[df_closed['result'] == 'WIN']
        losses = df_closed[df_closed['result'] == 'LOSS']
        
        pnl_wins = wins['payout'].sum() if 'payout' in wins.columns else len(wins) * 0.85
        pnl_losses = -len(losses)
        
        total_pnl = pnl_wins + pnl_losses
        avg_pnl = total_pnl / len(df_closed)
        pnl_ratio = total_pnl / len(df_closed) if len(df_closed) > 0 else 0.0
        
        return {
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'pnl_ratio': pnl_ratio,
            'n_trades': len(df_closed)
        }
    
    def track_entry_rate(self, hours: int = 24) -> Dict:
        """진입률 추적 (시간당 거래 수)"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        df = self.log_manager.load_recent_trades(n=1000)
        
        if df.empty or 'entry_ts' not in df.columns:
            return {
                'entry_rate': 0.0,
                'n_entries': 0,
                'hours': hours
            }
        
        df['entry_ts'] = pd.to_datetime(df['entry_ts'], utc=True)
        df_recent = df[df['entry_ts'] >= cutoff_time]
        
        n_entries = len(df_recent)
        entry_rate = n_entries / hours if hours > 0 else 0.0
        
        return {
            'entry_rate': entry_rate,
            'n_entries': n_entries,
            'hours': hours
        }
    
    def track_consecutive_losses(self) -> Dict:
        """연속 손실 추적"""
        df = self.log_manager.load_recent_trades(n=100)
        
        if df.empty:
            return {
                'current_streak': 0,
                'max_streak': 0
            }
        
        df_closed = df[df['status'] == 'CLOSED'].sort_values('entry_ts')
        
        if df_closed.empty:
            return {
                'current_streak': 0,
                'max_streak': 0
            }
        
        results = df_closed['result'].values
        
        current_streak = 0
        max_streak = 0
        streak = 0
        
        for result in reversed(results):
            if result == 'LOSS':
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                if current_streak == 0:
                    current_streak = streak
                streak = 0
        
        if current_streak == 0:
            current_streak = streak
        
        return {
            'current_streak': current_streak,
            'max_streak': max_streak
        }


class SystemMonitor:
    """
    시스템 모니터링
    - 메모리/CPU 사용량
    - 지연시간
    - NaN 체크
    """
    
    @staticmethod
    def check_memory() -> Dict:
        """메모리 사용량 체크"""
        mem = psutil.virtual_memory()
        
        return {
            'total_gb': mem.total / (1024**3),
            'used_gb': mem.used / (1024**3),
            'percent': mem.percent,
            'available_gb': mem.available / (1024**3)
        }
    
    @staticmethod
    def check_cpu() -> Dict:
        """CPU 사용량 체크"""
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        
        return {
            'cpu_percent': cpu_percent,
            'cpu_count': cpu_count,
            'per_cpu': psutil.cpu_percent(interval=1, percpu=True)
        }
    
    @staticmethod
    def check_nan(df: pd.DataFrame) -> Dict:
        """DataFrame NaN 체크"""
        if df.empty:
            return {
                'total_nans': 0,
                'nan_ratio': 0.0,
                'nan_columns': {}
            }
        
        nan_counts = df.isnull().sum()
        total_nans = nan_counts.sum()
        total_values = df.shape[0] * df.shape[1]
        nan_ratio = total_nans / total_values if total_values > 0 else 0.0
        
        nan_columns = nan_counts[nan_counts > 0].to_dict()
        
        return {
            'total_nans': int(total_nans),
            'nan_ratio': nan_ratio,
            'nan_columns': nan_columns
        }


class BugDetector:
    """
    버그 탐지기
    - 캘리브레이션 불량
    - 낮은 승률
    - 높은 연속 손실
    - 진입 없음
    - 레짐 편향
    - 방향 편향
    - NaN 급증
    """
    
    def __init__(self, log_manager: LogManager):
        self.log_manager = log_manager
        self.perf_monitor = PerformanceMonitor(log_manager)
    
    def detect_poor_calibration(self, ece_threshold: float = 0.10, gap_threshold: float = 0.05) -> Optional[Dict]:
        """캘리브레이션 불량 탐지 (최우선)"""
        cal_data = self.perf_monitor.validate_calibration(window=200)
        
        if not cal_data['valid']:
            return None
        
        metrics = cal_data['metrics']
        
        issues = []
        
        if metrics['ece'] > ece_threshold:
            issues.append(f"ECE 높음 ({metrics['ece']:.4f} > {ece_threshold})")
        
        if metrics['calibration_gap'] > gap_threshold:
            issues.append(f"캘리브레이션 갭 ({metrics['calibration_gap']:.4f} > {gap_threshold})")
        
        if issues:
            return {
                'severity': 'CRITICAL',
                'type': 'POOR_CALIBRATION',
                'message': ', '.join(issues),
                'data': cal_data
            }
        
        return None
    
    def detect_low_win_rate(self, threshold: float = 0.45, window: int = 50) -> Optional[Dict]:
        """낮은 승률 탐지"""
        win_rate_data = self.perf_monitor.track_win_rate(window=window)
        
        if win_rate_data['n_trades'] < 20:
            return None
        
        if win_rate_data['win_rate'] < threshold:
            return {
                'severity': 'CRITICAL',
                'type': 'LOW_WIN_RATE',
                'message': f"승률 {win_rate_data['win_rate']:.2%} < {threshold:.0%}",
                'data': win_rate_data
            }
        
        return None
    
    def detect_high_consecutive_losses(self, threshold: int = 5) -> Optional[Dict]:
        """높은 연속 손실 탐지"""
        streak_data = self.perf_monitor.track_consecutive_losses()
        
        if streak_data['current_streak'] >= threshold:
            return {
                'severity': 'WARN',
                'type': 'HIGH_CONSECUTIVE_LOSSES',
                'message': f"연속 손실 {streak_data['current_streak']}회",
                'data': streak_data
            }
        
        return None
    
    def detect_no_entries(self, hours: int = 2) -> Optional[Dict]:
        """진입 없음 탐지"""
        entry_data = self.perf_monitor.track_entry_rate(hours=hours)
        
        if entry_data['n_entries'] == 0:
            return {
                'severity': 'WARN',
                'type': 'NO_ENTRIES',
                'message': f"최근 {hours}시간 동안 진입 없음",
                'data': entry_data
            }
        
        return None
    
    def detect_regime_bias(self, threshold: float = 0.8, window: int = 50) -> Optional[Dict]:
        """레짐 편향 탐지"""
        regime_data = self.perf_monitor.track_win_rate_by_regime(window=window)
        
        if not regime_data:
            return None
        
        total_trades = sum(d['n_trades'] for d in regime_data.values())
        
        for regime_name, data in regime_data.items():
            ratio = data['n_trades'] / total_trades if total_trades > 0 else 0
            
            if ratio > threshold:
                return {
                    'severity': 'INFO',
                    'type': 'REGIME_BIAS',
                    'message': f"{regime_name} 레짐 편향 {ratio:.0%}",
                    'data': {'regime': regime_name, 'ratio': ratio}
                }
        
        return None
    
    def detect_direction_bias(self, threshold: float = 0.8, window: int = 50) -> Optional[Dict]:
        """방향 편향 탐지"""
        dir_data = self.perf_monitor.analyze_by_direction(window=window)
        
        if not dir_data or len(dir_data) < 2:
            return None
        
        total_trades = sum(d['n_trades'] for d in dir_data.values())
        
        for direction, data in dir_data.items():
            ratio = data['n_trades'] / total_trades if total_trades > 0 else 0
            
            if ratio > threshold:
                return {
                    'severity': 'INFO',
                    'type': 'DIRECTION_BIAS',
                    'message': f"{direction} 방향 편향 {ratio:.0%}",
                    'data': {'direction': direction, 'ratio': ratio}
                }
        
        return None
    
    def detect_nan_in_logs(self) -> Optional[Dict]:
        """로그 NaN 탐지"""
        df = self.log_manager.load_recent_trades(n=100)
        
        if df.empty:
            return None
        
        nan_data = SystemMonitor.check_nan(df)
        
        if nan_data['nan_ratio'] > config.ALERT_NAN_RATIO:
            return {
                'severity': 'WARN',
                'type': 'NAN_IN_LOGS',
                'message': f"NaN 비율 {nan_data['nan_ratio']:.2%}",
                'data': nan_data
            }
        
        return None
    
    def check_all(self) -> List[Dict]:
        """모든 버그 체크"""
        alerts = []
        
        detectors = [
            (self.detect_poor_calibration, ()),  # 최우선
            (self.detect_low_win_rate, (config.ALERT_WIN_RATE_LOW,)),
            (self.detect_high_consecutive_losses, (config.ALERT_CONSECUTIVE_LOSSES,)),
            (self.detect_no_entries, (config.ALERT_NO_ENTRY_HOURS,)),
            (self.detect_regime_bias, (config.ALERT_REGIME_BIAS,)),
            (self.detect_direction_bias, (0.8,)),
            (self.detect_nan_in_logs, ())
        ]
        
        for detector_func, args in detectors:
            try:
                result = detector_func(*args) if args else detector_func()
                if result:
                    alerts.append(result)
            except Exception as e:
                print(f"⚠️  탐지기 실행 실패 ({detector_func.__name__}): {e}")
        
        return alerts


class Monitor:
    """
    통합 모니터
    - 캘리브레이션 검증 (최우선)
    - 성능 + 시스템 + 버그 탐지
    - 스냅샷 저장
    """
    
    def __init__(self, log_manager: LogManager):
        self.log_manager = log_manager
        self.perf_monitor = PerformanceMonitor(log_manager)
        self.bug_detector = BugDetector(log_manager)
        self.snapshot_file = config.MONITOR_LOG_DIR / "snapshots.jsonl"
    
    def update(self) -> Dict:
        """모니터링 업데이트"""
        # 캘리브레이션 검증 (최우선)
        calibration = self.perf_monitor.validate_calibration(window=200)
        prob_bins = self.perf_monitor.analyze_by_probability_bins(window=200)
        direction_analysis = self.perf_monitor.analyze_by_direction(window=200)
        
        # 성능 지표 수집
        win_rate_data = self.perf_monitor.track_win_rate(window=50)
        regime_data = self.perf_monitor.track_win_rate_by_regime(window=100)
        model_data = self.perf_monitor.track_win_rate_by_model(window=200)
        profit_data = self.perf_monitor.track_profit(window=50)
        entry_data = self.perf_monitor.track_entry_rate(hours=24)
        streak_data = self.perf_monitor.track_consecutive_losses()
        
        # 시스템 지표 수집
        mem_data = SystemMonitor.check_memory()
        cpu_data = SystemMonitor.check_cpu()
        
        # 버그 탐지
        alerts = self.bug_detector.check_all()
        
        # 스냅샷 구성
        snapshot = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'calibration': calibration,
            'probability_bins': prob_bins,
            'direction_analysis': direction_analysis,
            'performance': {
                'win_rate': win_rate_data,
                'regime': regime_data,
                'model': model_data,
                'profit': profit_data,
                'entry': entry_data,
                'streak': streak_data
            },
            'system': {
                'memory': mem_data,
                'cpu': cpu_data
            },
            'alerts': alerts
        }
        
        # 스냅샷 저장
        self._save_snapshot(snapshot)
        
        return snapshot
    
    def print_summary(self) -> None:
        """요약 출력"""
        snapshot = self.update()
        
        print("\n" + "=" * 70)
        print("시스템 모니터링 요약")
        print("=" * 70)
        
        # 캘리브레이션 (최우선)
        cal = snapshot['calibration']
        print(f"\n🎯 캘리브레이션 검증:")
        if cal['valid']:
            metrics = cal['metrics']
            print(f"  샘플 수: {cal['n_samples']}건")
            print(f"  ECE: {metrics['ece']:.4f} {'✅' if metrics['ece'] < 0.05 else '⚠️'}")
            print(f"  Brier Score: {metrics['brier_score']:.4f}")
            print(f"  Log Loss: {metrics['log_loss']:.4f}")
            print(f"  예측 평균: {metrics['avg_predicted_prob']:.4f}")
            print(f"  실제 승률: {metrics['actual_win_rate']:.4f}")
            print(f"  캘리브레이션 갭: {metrics['calibration_gap']:.4f} {'✅' if metrics['calibration_gap'] < 0.03 else '⚠️'}")
            print(f"  판정: {'✅ 잘 캘리브레이션됨' if cal['is_well_calibrated'] else '⚠️  재캘리브레이션 필요'}")
        else:
            print(f"  ⚠️  {cal['reason']}")
        
        # 확률 구간별 분석
        if snapshot['probability_bins']:
            print(f"\n📊 확률 구간별 분석:")
            for bin_name, data in sorted(snapshot['probability_bins'].items()):
                if data['n_trades'] >= 5:  # 5건 이상만 표시
                    print(f"  {bin_name}: {data['n_trades']:3d}건, "
                          f"예측={data['avg_prob']:.3f}, "
                          f"실제={data['win_rate']:.3f}, "
                          f"오차={data['error']:.3f}")
        
        # 방향별 분석
        if snapshot['direction_analysis']:
            print(f"\n🔄 방향별 성과:")
            for direction, data in snapshot['direction_analysis'].items():
                print(f"  {direction}: {data['n_trades']}건 "
                      f"({data['n_wins']}W/{data['n_losses']}L), "
                      f"승률={data['win_rate']:.2%}, "
                      f"갭={data['calibration_gap']:.4f}")
        
        # 성능
        perf = snapshot['performance']
        print(f"\n📈 전체 성과:")
        print(f"  승률: {perf['win_rate']['win_rate']:.2%} "
              f"({perf['win_rate']['n_wins']}W / {perf['win_rate']['n_losses']}L)")
        print(f"  손익: {perf['profit']['total_pnl']:.2f} "
              f"(평균 {perf['profit']['avg_pnl']:.3f})")
        print(f"  진입률: {perf['entry']['entry_rate']:.2f}/시간 "
              f"({perf['entry']['n_entries']}건/{perf['entry']['hours']}시간)")
        print(f"  연속 손실: {perf['streak']['current_streak']}회 "
              f"(최대 {perf['streak']['max_streak']}회)")
        
        # 레짐별
        if perf['regime']:
            print(f"\n🌐 레짐별 승률:")
            for regime_name, data in perf['regime'].items():
                print(f"  {regime_name:5s}: {data['win_rate']:.2%} ({data['n_trades']}건)")
        
        # 모델별
        if perf['model']:
            print(f"\n🤖 모델별 승률:")
            for model_ver, data in perf['model'].items():
                print(f"  {model_ver}: {data['win_rate']:.2%} ({data['n_trades']}건)")
        
        # 시스템
        sys = snapshot['system']
        print(f"\n💻 시스템:")
        print(f"  메모리: {sys['memory']['used_gb']:.1f}GB / "
              f"{sys['memory']['total_gb']:.1f}GB "
              f"({sys['memory']['percent']:.1f}%)")
        print(f"  CPU: {sys['cpu']['cpu_percent']:.1f}% "
              f"({sys['cpu']['cpu_count']}코어)")
        
        # 알림
        if snapshot['alerts']:
            print(f"\n⚠️  알림 ({len(snapshot['alerts'])}건):")
            for alert in snapshot['alerts']:
                severity_icon = {
                    'CRITICAL': '🔴',
                    'WARN': '🟡',
                    'INFO': '🔵'
                }.get(alert['severity'], '⚪')
                print(f"  {severity_icon} [{alert['severity']}] {alert['message']}")
        else:
            print(f"\n✅ 알림 없음")
        
        print("=" * 70 + "\n")
    
    def print_calibration_detail(self) -> None:
        """캘리브레이션 상세 출력"""
        cal = self.perf_monitor.validate_calibration(window=200)
        
        if not cal['valid']:
            print(f"\n⚠️  캘리브레이션 검증 불가: {cal['reason']}")
            return
        
        print("\n" + "=" * 70)
        print("캘리브레이션 상세 분석")
        print("=" * 70)
        
        metrics = cal['metrics']
        
        print(f"\n📊 전체 메트릭:")
        print(f"  샘플 수: {cal['n_samples']}건")
        print(f"  사용 컬럼: {cal['prob_column']}")
        print(f"  ECE (Expected Calibration Error): {metrics['ece']:.4f}")
        print(f"  Brier Score: {metrics['brier_score']:.4f}")
        print(f"  Log Loss: {metrics['log_loss']:.4f}")
        print(f"  평균 예측 확률: {metrics['avg_predicted_prob']:.4f}")
        print(f"  실제 승률: {metrics['actual_win_rate']:.4f}")
        print(f"  캘리브레이션 갭: {metrics['calibration_gap']:.4f}")
        
        print(f"\n📈 구간별 분석 (10개 구간):")
        print(f"  {'구간':<20} {'샘플':>6} {'예측':>8} {'실제':>8} {'오차':>8}")
        print("  " + "-" * 60)
        
        for bin_name, bin_data in sorted(cal['bins'].items()):
            print(f"  {bin_data['range']:<20} "
                  f"{bin_data['n_samples']:>6} "
                  f"{bin_data['avg_prob']:>8.4f} "
                  f"{bin_data['actual_rate']:>8.4f} "
                  f"{bin_data['error']:>8.4f}")
        
        print(f"\n💡 판정: ", end="")
        if cal['is_well_calibrated']:
            print("✅ 모델이 잘 캘리브레이션되어 있습니다.")
        else:
            print("⚠️  재캘리브레이션이 필요합니다.")
            
            if metrics['ece'] >= 0.05:
                print("  - ECE가 높습니다 (>0.05)")
            if metrics['calibration_gap'] >= 0.03:
                print("  - 전체 캘리브레이션 갭이 큽니다 (>0.03)")
        
        print("=" * 70 + "\n")
    
    def _save_snapshot(self, snapshot: Dict) -> None:
        try:
            self.snapshot_file.parent.mkdir(parents=True, exist_ok=True)
            
            # numpy 타입을 native Python 타입으로 변환
            def convert_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(v) for v in obj]
                return obj
            
            snapshot_clean = convert_types(snapshot)
            
            with open(self.snapshot_file, 'a') as f:
                f.write(json.dumps(snapshot_clean, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"⚠️  스냅샷 저장 실패: {e}")
    
    def load_snapshots(self, n: int = 100) -> List[Dict]:
        """저장된 스냅샷 로드"""
        if not self.snapshot_file.exists():
            return []
        
        snapshots = []
        
        try:
            with open(self.snapshot_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines[-n:]:
                snapshot = json.loads(line.strip())
                snapshots.append(snapshot)
        
        except Exception as e:
            print(f"⚠️  스냅샷 로드 실패: {e}")
        
        return snapshots
    
    def analyze_calibration_trend(self, n_snapshots: int = 20) -> Dict:
        """캘리브레이션 추세 분석"""
        snapshots = self.load_snapshots(n=n_snapshots)
        
        if not snapshots:
            return {
                'valid': False,
                'reason': '스냅샷 없음'
            }
        
        ece_history = []
        gap_history = []
        timestamps = []
        
        for snap in snapshots:
            if 'calibration' not in snap:
                continue
            
            cal = snap['calibration']
            if not cal.get('valid', False):
                continue
            
            metrics = cal['metrics']
            ece_history.append(metrics['ece'])
            gap_history.append(metrics['calibration_gap'])
            timestamps.append(snap['timestamp'])
        
        if len(ece_history) < 2:
            return {
                'valid': False,
                'reason': '충분한 데이터 없음'
            }
        
        # 추세 계산
        ece_trend = 'improving' if ece_history[-1] < ece_history[0] else 'degrading'
        gap_trend = 'improving' if gap_history[-1] < gap_history[0] else 'degrading'
        
        return {
            'valid': True,
            'n_snapshots': len(ece_history),
            'ece': {
                'current': ece_history[-1],
                'initial': ece_history[0],
                'min': min(ece_history),
                'max': max(ece_history),
                'trend': ece_trend
            },
            'gap': {
                'current': gap_history[-1],
                'initial': gap_history[0],
                'min': min(gap_history),
                'max': max(gap_history),
                'trend': gap_trend
            },
            'timestamps': timestamps
        }


# ============================================================
# 테스트 및 검증
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print("Monitor 테스트 (v1.4.0 - 캘리브레이션 검증 중심)")
    print("=" * 70)
    
    from log_manager import LogManager
    
    lm = LogManager()
    monitor = Monitor(lm)
    
    print("\n📊 전체 요약:")
    monitor.print_summary()
    
    print("\n🎯 캘리브레이션 상세:")
    monitor.print_calibration_detail()
    
    print("\n📈 캘리브레이션 추세:")
    trend = monitor.analyze_calibration_trend(n_snapshots=20)
    
    if trend['valid']:
        print(f"  스냅샷 수: {trend['n_snapshots']}개")
        print(f"  ECE 추세: {trend['ece']['trend']}")
        print(f"    현재: {trend['ece']['current']:.4f}")
        print(f"    초기: {trend['ece']['initial']:.4f}")
        print(f"    범위: [{trend['ece']['min']:.4f}, {trend['ece']['max']:.4f}]")
        print(f"  갭 추세: {trend['gap']['trend']}")
        print(f"    현재: {trend['gap']['current']:.4f}")
        print(f"    초기: {trend['gap']['initial']:.4f}")
        print(f"    범위: [{trend['gap']['min']:.4f}, {trend['gap']['max']:.4f}]")
    else:
        print(f"  ⚠️  {trend['reason']}")
    
    print("\n" + "=" * 70)
    print("테스트 완료")
    print("=" * 70)
"""
모니터링 시스템 - 성능 추적 + 버그 탐지
버전: 1.3.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
import json
import psutil
from pathlib import Path

import config
from log_manager import LogManager


class PerformanceMonitor:
    """
    성능 모니터링
    - 승률/손익/진입률 추적
    - 레짐별/모델별 분석
    - 연속 손실 추적
    """
    
    def __init__(self, log_manager: LogManager):
        self.log_manager = log_manager
        self.snapshot_history = []
    
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
        # UTC 타임존으로 cutoff_time 생성
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        # 최근 거래 로드
        df = self.log_manager.load_recent_trades(n=1000)
        
        if df.empty or 'entry_ts' not in df.columns:
            return {
                'entry_rate': 0.0,
                'n_entries': 0,
                'hours': hours
            }
        
        # UTC 타임존으로 변환
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
        
        # 연속 손실 계산
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
    - 낮은 승률
    - 높은 연속 손실
    - 진입 없음
    - 레짐 편향
    - NaN 급증
    """
    
    def __init__(self, log_manager: LogManager):
        self.log_manager = log_manager
        self.perf_monitor = PerformanceMonitor(log_manager)
    
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
        
        # 각 탐지기 실행
        detectors = [
            (self.detect_low_win_rate, (config.ALERT_WIN_RATE_LOW,)),
            (self.detect_high_consecutive_losses, (config.ALERT_CONSECUTIVE_LOSSES,)),
            (self.detect_no_entries, (config.ALERT_NO_ENTRY_HOURS,)),
            (self.detect_regime_bias, (config.ALERT_REGIME_BIAS,)),
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
    - 성능 + 시스템 + 버그 탐지
    - 스냅샷 저장
    - 알림
    """
    
    def __init__(self, log_manager: LogManager):
        self.log_manager = log_manager
        self.perf_monitor = PerformanceMonitor(log_manager)
        self.bug_detector = BugDetector(log_manager)
        self.snapshot_file = config.MONITOR_LOG_DIR / "snapshots.jsonl"
    
    def update(self) -> Dict:
        """모니터링 업데이트"""
        # 성능 지표 수집
        win_rate_data = self.perf_monitor.track_win_rate(window=50)
        regime_data = self.perf_monitor.track_win_rate_by_regime(window=100)
        profit_data = self.perf_monitor.track_profit(window=50)
        entry_data = self.perf_monitor.track_entry_rate(hours=24)
        streak_data = self.perf_monitor.track_consecutive_losses()
        
        # 시스템 지표 수집
        mem_data = SystemMonitor.check_memory()
        cpu_data = SystemMonitor.check_cpu()
        
        # 버그 탐지
        alerts = self.bug_detector.check_all()
        
        # 스냅샷 구성 - UTC 타임존으로 타임스탬프 생성
        snapshot = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'performance': {
                'win_rate': win_rate_data,
                'regime': regime_data,
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
        
        # 알림 전송 (심각한 경고만)
        critical_alerts = [a for a in alerts if a['severity'] == 'CRITICAL']
        if critical_alerts:
            self._send_alerts(critical_alerts)
        
        return snapshot
    
    def print_summary(self) -> None:
        """요약 출력"""
        snapshot = self.update()
        
        print("\n" + "=" * 60)
        print("시스템 모니터링 요약")
        print("=" * 60)
        
        # 성능
        perf = snapshot['performance']
        print(f"\n📊 성능:")
        print(f"  승률: {perf['win_rate']['win_rate']:.2%} ({perf['win_rate']['n_wins']}W / {perf['win_rate']['n_losses']}L)")
        print(f"  손익: {perf['profit']['total_pnl']:.2f} (평균 {perf['profit']['avg_pnl']:.3f})")
        print(f"  진입률: {perf['entry']['entry_rate']:.2f}/시간 ({perf['entry']['n_entries']}건/{perf['entry']['hours']}시간)")
        print(f"  연속 손실: {perf['streak']['current_streak']}회 (최대 {perf['streak']['max_streak']}회)")
        
        # 레짐별
        if perf['regime']:
            print(f"\n📈 레짐별 승률:")
            for regime_name, data in perf['regime'].items():
                print(f"  {regime_name}: {data['win_rate']:.2%} ({data['n_trades']}건)")
        
        # 시스템
        sys = snapshot['system']
        print(f"\n💻 시스템:")
        print(f"  메모리: {sys['memory']['used_gb']:.1f}GB / {sys['memory']['total_gb']:.1f}GB ({sys['memory']['percent']:.1f}%)")
        print(f"  CPU: {sys['cpu']['cpu_percent']:.1f}% ({sys['cpu']['cpu_count']}코어)")
        
        # 알림
        if snapshot['alerts']:
            print(f"\n⚠️  알림 ({len(snapshot['alerts'])}건):")
            for alert in snapshot['alerts']:
                print(f"  [{alert['severity']}] {alert['message']}")
        else:
            print(f"\n✅ 알림 없음")
        
        print("=" * 60)
    
    def _save_snapshot(self, snapshot: Dict) -> None:
        """스냅샷 저장"""
        try:
            self.snapshot_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.snapshot_file, 'a') as f:
                f.write(json.dumps(snapshot, ensure_ascii=False) + '\n')
        
        except Exception as e:
            print(f"⚠️  스냅샷 저장 실패: {e}")
    
    def _send_alerts(self, alerts: List[Dict]) -> None:
        """알림 전송 (placeholder)"""
        # TODO: 텔레그램/이메일 연동
        for alert in alerts:
            print(f"🚨 [ALERT] {alert['message']}")
    
    def load_snapshots(self, n: int = 100) -> List[Dict]:
        """저장된 스냅샷 로드"""
        if not self.snapshot_file.exists():
            return []
        
        snapshots = []
        
        try:
            with open(self.snapshot_file, 'r') as f:
                lines = f.readlines()
            
            # 최근 n개만
            for line in lines[-n:]:
                snapshot = json.loads(line.strip())
                snapshots.append(snapshot)
        
        except Exception as e:
            print(f"⚠️  스냅샷 로드 실패: {e}")
        
        return snapshots
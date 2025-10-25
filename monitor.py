import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import psutil
import time

from config import Config
from log_manager import LogManager


# =========================
# 공용 유틸: 안전한 CLOSED 마스크
# =========================
def _closed_mask(df: pd.DataFrame) -> pd.Series:
    """status가 없으면 result 유무로 CLOSED 추정"""
    if 'status' in df.columns:
        return df['status'].astype(str).str.upper().eq('CLOSED')
    return df['result'].isin([0, 1]) if 'result' in df.columns else pd.Series(False, index=df.index)


class PerformanceMonitor:
    """성능 모니터링 (승률/손익/진입률/모델별)"""

    def __init__(self, log_manager: LogManager):
        self.config = Config
        self.log_manager = log_manager

        # 성능 기록
        self.performance_history = []
        self.performance_path = self.config.RESULT_DIR / "performance_history.jsonl"

    # ==========================================
    # 승률 추적
    # ==========================================
    def track_win_rate(self, window: int = 50) -> Dict:
        """
        최근 N거래 승률 추적
        """
        recent_trades = self.log_manager.load_recent_trades(n=window)

        if recent_trades.empty or 'result' not in recent_trades.columns:
            return {'win_rate': None, 'total_trades': 0, 'wins': 0, 'losses': 0, 'window': window}

        closed = recent_trades[_closed_mask(recent_trades)].copy()
        if len(closed) == 0:
            return {'win_rate': None, 'total_trades': 0, 'wins': 0, 'losses': 0, 'window': window}

        wins = int((closed['result'] == 1).sum())
        losses = int((closed['result'] == 0).sum())
        total = len(closed)
        win_rate = wins / total if total > 0 else 0.0

        return {'win_rate': float(win_rate), 'total_trades': int(total), 'wins': int(wins), 'losses': int(losses), 'window': window}

    def track_win_rate_by_regime(self, window: int = 100) -> Dict:
        """레짐별 승률 추적"""
        recent_trades = self.log_manager.load_recent_trades(n=window)
        if recent_trades.empty or 'regime' not in recent_trades.columns:
            return {}

        closed = recent_trades[_closed_mask(recent_trades)].copy()
        if closed.empty:
            return {}

        regime_stats = {}
        for regime_val in closed['regime'].dropna().unique():
            regime_data = closed[closed['regime'] == regime_val]
            if 'result' in regime_data.columns and len(regime_data) > 0:
                wins = int((regime_data['result'] == 1).sum())
                total = len(regime_data)
                win_rate = wins / total if total > 0 else 0.0
                regime_name = {1: "UP", -1: "DOWN", 0: "FLAT"}.get(regime_val, f"REGIME-{regime_val}")
                regime_stats[regime_name] = {'win_rate': float(win_rate), 'total': int(total), 'wins': int(wins)}
        return regime_stats

    # ==========================================
    # 손익 추적
    # ==========================================
    def track_profit(self, window: int = 50) -> Dict:
        """
        손익 추적 (시뮬레이션)
        """
        recent_trades = self.log_manager.load_recent_trades(n=window)
        if recent_trades.empty or 'result' not in recent_trades.columns:
            return {'total_profit': 0.0, 'avg_profit_per_trade': 0.0, 'max_drawdown': 0.0, 'profit_factor': 0.0}

        closed = recent_trades[_closed_mask(recent_trades)].copy()
        if len(closed) == 0:
            return {'total_profit': 0.0, 'avg_profit_per_trade': 0.0, 'max_drawdown': 0.0, 'profit_factor': 0.0}

        payout = float(self.config.PAYOUT_RATIO)
        profits = [(payout if int(r) == 1 else -1.0) for r in closed['result'].tolist()]

        total_profit = float(sum(profits))
        avg_profit = float(total_profit / len(profits)) if len(profits) > 0 else 0.0

        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = running_max - cumulative
        max_drawdown = float(drawdown.max()) if len(drawdown) > 0 else 0.0

        gross_profit = sum(p for p in profits if p > 0)
        gross_loss = abs(sum(p for p in profits if p < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

        return {'total_profit': total_profit, 'avg_profit_per_trade': avg_profit, 'max_drawdown': max_drawdown, 'profit_factor': float(profit_factor)}

    # ==========================================
    # 진입률 추적
    # ==========================================
    def track_entry_rate(self, hours: int = 24) -> Dict:
        """
        시간당 진입률 추적
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y%m%d")

        trades = []
        for date in [today, yesterday]:
            df = self.log_manager.load_trade_log(date)
            if not df.empty:
                trades.append(df)

        if not trades:
            return {'entries_per_hour': 0.0, 'total_entries': 0, 'hours': hours}

        all_trades = pd.concat(trades, ignore_index=True)
        if 'entry_ts' in all_trades.columns:
            all_trades['entry_ts'] = pd.to_datetime(all_trades['entry_ts'], utc=True, errors='coerce')
            recent = all_trades[all_trades['entry_ts'] >= cutoff]
        else:
            recent = all_trades

        total_entries = int(len(recent))
        entries_per_hour = float(total_entries / hours) if hours > 0 else 0.0
        return {'entries_per_hour': entries_per_hour, 'total_entries': total_entries, 'hours': int(hours)}

    # ==========================================
    # 연속 손실 추적
    # ==========================================
    def track_consecutive_losses(self) -> Dict:
        """
        현재 연속 손실 및 최대 연속 손실
        """
        recent_trades = self.log_manager.load_recent_trades(n=200)
        if recent_trades.empty or 'result' not in recent_trades.columns:
            return {'current_streak': 0, 'max_streak': 0}

        closed = recent_trades[_closed_mask(recent_trades)].copy()
        if len(closed) == 0:
            return {'current_streak': 0, 'max_streak': 0}

        if 'entry_ts' in closed.columns:
            closed = closed.sort_values('entry_ts', ascending=False)

        results = closed['result'].astype(int).tolist()

        current_streak = 0
        for r in results:
            if r == 0:
                current_streak += 1
            else:
                break

        max_streak = 0
        temp_streak = 0
        for r in results:
            if r == 0:
                temp_streak += 1
                max_streak = max(max_streak, temp_streak)
            else:
                temp_streak = 0

        return {'current_streak': int(current_streak), 'max_streak': int(max_streak)}

    # ==========================================
    # 모델별 승률
    # ==========================================
    def track_win_rate_by_model(self, window: int = 200) -> Dict:
        trades = self.log_manager.load_recent_trades(n=window)
        if trades.empty or 'result' not in trades.columns:
            return {}

        closed = trades[_closed_mask(trades)].copy()
        if closed.empty:
            return {}

        # meta 조인 (model_version)
        # 필요 시 여러 일자 결합 가능. 우선 오늘자만.
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        meta = self.log_manager.load_trade_meta(today)
        if not meta.empty:
            meta = meta[['trade_id', 'model_version']].dropna()
            closed = closed.merge(meta, on='trade_id', how='left')

        if 'model_version' not in closed.columns:
            return {}

        out = {}
        for mv, g in closed.groupby('model_version'):
            if g.empty:
                continue
            wins = int((g['result'] == 1).sum())
            total = int(len(g))
            out[str(mv) if pd.notna(mv) else 'N/A'] = {
                'win_rate': float(wins / total) if total else 0.0,
                'total': total,
                'wins': wins
            }
        return out

    # ==========================================
    # 성능 스냅샷 저장 & 출력
    # ==========================================
    def save_snapshot(self):
        """현재 성능 스냅샷 저장 (JSONL)"""
        snapshot = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'win_rate': self.track_win_rate(window=50),
            'win_rate_by_regime': self.track_win_rate_by_regime(window=100),
            'profit': self.track_profit(window=50),
            'entry_rate': self.track_entry_rate(hours=24),
            'consecutive_losses': self.track_consecutive_losses()
        }
        with open(self.performance_path, 'a', encoding='utf-8') as f:
            json.dump(snapshot, f, ensure_ascii=False, default=str)
            f.write('\n')

    def print_summary(self):
        """성능 요약 출력"""
        print(f"\n{'='*60}")
        print("성능 모니터링 요약")
        print(f"{'='*60}")

        # 승률
        wr = self.track_win_rate(window=50)
        if wr['total_trades'] > 0 and wr['win_rate'] is not None:
            print(f"\n[승률] (최근 {wr['window']}거래)")
            print(f"  전체: {wr['win_rate']:.2%} ({wr['wins']}/{wr['total_trades']})")
        else:
            print(f"\n[승률] 거래 없음")

        # 레짐별 승률
        wr_regime = self.track_win_rate_by_regime(window=100)
        if wr_regime:
            print(f"\n[레짐별 승률]")
            for regime, stats in wr_regime.items():
                print(f"  {regime}: {stats['win_rate']:.2%} ({stats['wins']}/{stats['total']})")

        # 손익
        profit = self.track_profit(window=50)
        print(f"\n[손익] (최근 50거래)")
        print(f"  총 손익: {profit['total_profit']:+.2f}")
        print(f"  평균 손익: {profit['avg_profit_per_trade']:+.3f}")
        print(f"  최대 낙폭: {profit['max_drawdown']:.2f}")
        pf = profit['profit_factor']
        pf_str = f"{pf:.2f}" if np.isfinite(pf) else "∞"
        print(f"  Profit Factor: {pf_str}")

        # 진입률
        entry = self.track_entry_rate(hours=24)
        print(f"\n[진입률] (최근 24시간)")
        print(f"  시간당: {entry['entries_per_hour']:.2f}개")
        print(f"  총 진입: {entry['total_entries']}개")

        # 연속 손실
        streak = self.track_consecutive_losses()
        print(f"\n[연속 손실]")
        print(f"  현재: {streak['current_streak']}연속")
        print(f"  최대: {streak['max_streak']}연속")

        # 모델별 승률
        wr_model = self.track_win_rate_by_model(window=200)
        if wr_model:
            print(f"\n[모델별 승률]")
            for mv, s in wr_model.items():
                print(f"  {mv}: {s['win_rate']:.2%} ({s['wins']}/{s['total']})")

        print(f"{'='*60}\n")


class SystemMonitor:
    """시스템 상태 모니터링"""

    def __init__(self):
        self.config = Config

        # 시스템 기록
        self.system_history = []
        self.system_path = self.config.RESULT_DIR / "system_history.jsonl"

    # ==========================================
    # 메모리 사용량
    # ==========================================
    def check_memory(self) -> Dict:
        """메모리 사용량 체크"""
        mem = psutil.virtual_memory()
        return {
            'percent': float(mem.percent),
            'used_mb': float(mem.used / 1024 / 1024),
            'available_mb': float(mem.available / 1024 / 1024),
            'total_mb': float(mem.total / 1024 / 1024)
        }

    def check_memory_alert(self, threshold: float = 85.0) -> bool:
        """메모리 경고 체크 (임계값 초과 시 True)"""
        mem = self.check_memory()
        return mem['percent'] > threshold

    # ==========================================
    # CPU 사용량
    # ==========================================
    def check_cpu(self) -> Dict:
        """CPU 사용량 체크"""
        cpu_percent = psutil.cpu_percent(interval=0.2)  # 블로킹 시간 단축
        cpu_count = psutil.cpu_count()
        return {'percent': float(cpu_percent), 'count': int(cpu_count)}

    # ==========================================
    # 레이턴시 체크
    # ==========================================
    def check_latency(self, func, *args, **kwargs) -> Dict:
        """함수 실행 시간 측정"""
        start = time.time()
        success = True
        try:
            func(*args, **kwargs)
        except Exception:
            success = False
        latency = (time.time() - start) * 1000  # ms
        return {'latency_ms': float(latency), 'success': bool(success)}

    # ==========================================
    # NaN 체크
    # ==========================================
    def check_nan(self, df: pd.DataFrame) -> Dict:
        """DataFrame NaN 체크"""
        if df is None or df.empty:
            return {'has_nan': False, 'nan_count': 0, 'nan_columns': []}
        nan_count = int(df.isna().sum().sum())
        nan_columns = df.columns[df.isna().any()].tolist()
        return {'has_nan': nan_count > 0, 'nan_count': nan_count, 'nan_columns': nan_columns}

    # ==========================================
    # 시스템 스냅샷
    # ==========================================
    def save_snapshot(self):
        """시스템 상태 스냅샷 저장"""
        snapshot = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'memory': self.check_memory(),
            'cpu': self.check_cpu()
        }
        with open(self.system_path, 'a', encoding='utf-8') as f:
            json.dump(snapshot, f, ensure_ascii=False, default=str)
            f.write('\n')

    def print_summary(self):
        """시스템 상태 요약 출력"""
        print(f"\n{'='*60}")
        print("시스템 모니터링 요약")
        print(f"{'='*60}")

        # 메모리
        mem = self.check_memory()
        print(f"\n[메모리]")
        print(f"  사용률: {mem['percent']:.1f}%")
        print(f"  사용량: {mem['used_mb']:.0f} MB / {mem['total_mb']:.0f} MB")
        print(f"  가용량: {mem['available_mb']:.0f} MB")

        # CPU
        cpu = self.check_cpu()
        print(f"\n[CPU]")
        print(f"  사용률: {cpu['percent']:.1f}%")
        print(f"  코어 수: {cpu['count']}")

        print(f"{'='*60}\n")


class BugDetector:
    """버그 및 이상 패턴 탐지"""

    def __init__(self, log_manager: LogManager, perf_monitor: PerformanceMonitor):
        self.config = Config
        self.log_manager = log_manager
        self.perf_monitor = perf_monitor

        # 알림 기록
        self.alerts = []
        self.alert_path = self.config.RESULT_DIR / "alerts.jsonl"

    # ==========================================
    # 이상 탐지
    # ==========================================
    def detect_low_win_rate(self, threshold: float = 0.45, window: int = 50) -> Optional[Dict]:
        """승률 급락 탐지"""
        wr = self.perf_monitor.track_win_rate(window=window)
        if wr['total_trades'] >= 20 and wr['win_rate'] is not None:
            if wr['win_rate'] < threshold:
                return {
                    'type': 'LOW_WIN_RATE',
                    'severity': 'CRITICAL',
                    'message': f"승률 급락: {wr['win_rate']:.1%} < {threshold:.1%} (최근 {window}거래)",
                    'data': wr
                }
        return None

    def detect_high_consecutive_losses(self, threshold: int = 5) -> Optional[Dict]:
        """연속 손실 탐지"""
        streak = self.perf_monitor.track_consecutive_losses()
        if streak['current_streak'] >= threshold:
            return {
                'type': 'HIGH_CONSECUTIVE_LOSSES',
                'severity': 'WARNING',
                'message': f"연속 손실: {streak['current_streak']}연속",
                'data': streak
            }
        return None

    def detect_no_entries(self, hours: int = 2) -> Optional[Dict]:
        """진입 중단 탐지"""
        entry = self.perf_monitor.track_entry_rate(hours=hours)
        if entry['total_entries'] == 0:
            return {
                'type': 'NO_ENTRIES',
                'severity': 'WARNING',
                'message': f"진입 중단: 최근 {hours}시간 동안 진입 없음",
                'data': entry
            }
        return None

    def detect_regime_bias(self, threshold: float = 0.8, window: int = 100) -> Optional[Dict]:
        """레짐 편향 탐지"""
        wr_regime = self.perf_monitor.track_win_rate_by_regime(window=window)
        if not wr_regime:
            return None

        for regime, stats in wr_regime.items():
            if stats['total'] >= 10:
                loss_rate = 1 - stats['win_rate']
                if loss_rate > threshold:
                    return {
                        'type': 'REGIME_BIAS',
                        'severity': 'WARNING',
                        'message': f"{regime} 레짐 손실률 높음: {loss_rate:.1%}",
                        'data': {'regime': regime, 'loss_rate': loss_rate, 'stats': stats}
                    }
        return None

    def detect_nan_in_logs(self) -> Optional[Dict]:
        """로그 NaN 탐지"""
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        trades = self.log_manager.load_trade_log(today)
        if trades.empty:
            return None

        closed = trades[_closed_mask(trades)].copy()
        if closed.empty:
            return None

        critical_cols = ['entry_price', 'label_price', 'result', 'p_at_entry']
        for col in critical_cols:
            if col in closed.columns:
                nan_count = int(closed[col].isna().sum())
                if nan_count > 0:
                    return {
                        'type': 'NAN_IN_LOGS',
                        'severity': 'CRITICAL',
                        'message': f"로그 NaN 발견: {col} 컬럼 {nan_count}개",
                        'data': {'column': col, 'nan_count': nan_count}
                    }
        return None

    # ==========================================
    # 종합 체크
    # ==========================================
    def check_all(self) -> List[Dict]:
        """모든 이상 패턴 체크"""
        alerts: List[Dict] = []

        for alert in [
            self.detect_low_win_rate(threshold=0.45, window=50),
            self.detect_high_consecutive_losses(threshold=5),
            self.detect_no_entries(hours=2),
            self.detect_regime_bias(threshold=0.75, window=100),
            self.detect_nan_in_logs()
        ]:
            if alert:
                alerts.append(alert)

        return alerts

    def save_alert(self, alert: Dict):
        """알림 저장"""
        record = {'timestamp': datetime.now(timezone.utc).isoformat(), **alert}
        with open(self.alert_path, 'a', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, default=str)
            f.write('\n')
        self.alerts.append(record)

    def print_alerts(self, alerts: List[Dict]):
        """알림 출력"""
        if not alerts:
            print("✓ 이상 없음")
            return

        print(f"\n{'='*60}")
        print(f"⚠️  {len(alerts)}개 알림 발생")
        print(f"{'='*60}")

        for alert in alerts:
            severity = alert['severity']
            emoji = "🔴" if severity == 'CRITICAL' else "🟡"
            print(f"\n{emoji} [{severity}] {alert['type']}")
            print(f"  {alert['message']}")

        print(f"{'='*60}\n")


class AlertManager:
    """알림 관리 (콘솔/Slack)"""

    def __init__(self):
        self.config = Config
        self.slack_webhook = None  # Slack Webhook URL (선택)

    def send_console(self, message: str, severity: str = 'INFO'):
        """콘솔 알림"""
        emoji = {'INFO': 'ℹ️', 'WARNING': '⚠️', 'CRITICAL': '🔴'}.get(severity, 'ℹ️')
        print(f"{emoji} [{severity}] {message}")

    def send_slack(self, message: str, severity: str = 'INFO'):
        """Slack 알림 (선택)"""
        if not self.slack_webhook:
            return
        # TODO: Slack Webhook 구현
        pass


# ==========================================
# 통합 모니터
# ==========================================
class Monitor:
    """통합 모니터링 시스템"""

    def __init__(self, log_manager: LogManager):
        self.log_manager = log_manager
        self.perf_monitor = PerformanceMonitor(log_manager)
        self.sys_monitor = SystemMonitor()
        self.bug_detector = BugDetector(log_manager, self.perf_monitor)
        self.alert_manager = AlertManager()

    def update(self):
        """모니터링 업데이트 (주기적 호출)"""
        # 스냅샷 저장
        self.perf_monitor.save_snapshot()
        self.sys_monitor.save_snapshot()

        # 이상 탐지
        alerts = self.bug_detector.check_all()

        # 알림 처리
        for alert in alerts:
            self.bug_detector.save_alert(alert)
            self.alert_manager.send_console(alert['message'], severity=alert['severity'])

    def print_summary(self):
        """전체 요약 출력"""
        self.perf_monitor.print_summary()
        self.sys_monitor.print_summary()

        # 최근 알림
        alerts = self.bug_detector.check_all()
        self.bug_detector.print_alerts(alerts)


# ==========================================
# 테스트
# ==========================================
if __name__ == "__main__":
    Config.create_directories()
    log_manager = LogManager()
    monitor = Monitor(log_manager)

    print("="*60)
    print("Monitor 테스트")
    print("="*60)

    # 업데이트
    monitor.update()

    # 요약 출력
    monitor.print_summary()

    print("\n✓ 테스트 완료")

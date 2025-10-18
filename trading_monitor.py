# trading_monitor.py - 데이터 무결성 감시 + 성능 모니터 + 재학습 보호(롤백)

import os
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

from data_merge import DataMerger, DataValidator

class TradingMonitor:
    """
    목적:
      - 재학습 전: 데이터 무결성 검사(잘못된 데이터로 학습 방지)
      - 재학습 후: old/new 모델 성능 비교, 나빠지면 롤백
      - 운영 중: 최근 거래 통계/리포트
    """

    def __init__(self, config):
        self.config = config
        self.merger = DataMerger(config)

        # 교체 기준 (원하면 조정)
        self.replace_metric = "accuracy"     # 비교 기준: "accuracy" 또는 "win_rate"
        self.min_improve = 0.0               # 교체 최소 향상폭 (음수면 '동일 이상'도 허용)

    # ---------------------------
    # 1) 데이터 무결성 검사
    # ---------------------------
    def validate_data_integrity(self) -> bool:
        """가격/거래로그 기본 무결성 검사. 문제가 있으면 False."""
        price_df = self.merger.load_price_data()
        trade_df = self.merger.load_trade_logs()

        price_ok, price_issues = DataValidator.validate_price_data(price_df) if not price_df.empty else (False, ["가격 데이터 없음"])
        trade_ok, trade_issues = DataValidator.validate_trade_logs(trade_df) if not trade_df.empty else (True, [])  # 거래로그는 초기에 없을 수 있음

        if not price_ok or not trade_ok:
            print("\n" + "="*60)
            print("⚠️ 데이터 무결성 문제 감지됨! 재학습 중단")
            print("="*60)
            if price_issues:
                print("[가격 데이터 이슈]")
                for i in price_issues:
                    print(" -", i)
            if trade_issues:
                print("[거래 로그 이슈]")
                for i in trade_issues:
                    print(" -", i)
            print("="*60)
            return False

        # 타임스탬프 연속성 간단 체크(1분봉 기준, 큰 결손 구간 경고)
        if 'timestamp' in price_df.columns:
            ts = pd.to_datetime(price_df['timestamp'], utc=True, errors='coerce').dropna().sort_values()
            gaps = ts.diff().dt.total_seconds().fillna(60)
            big_gaps = (gaps > 60 * 5).sum()  # 5분 이상 끊긴 구간 개수
            if big_gaps > 0:
                print(f"⚠️ 시계열 불연속 경고: 5분 이상 결손 구간 {int(big_gaps)}개 발견 (재학습은 계속 가능)")

        return True

    # ---------------------------
    # 2) 최근 거래 통계/리포트
    # ---------------------------
    def analyze_recent_trades(self, days=7):
        trades = self.merger.load_trade_logs()
        if trades.empty:
            return None

        if 'entry_time' in trades.columns:
            trades['entry_time'] = pd.to_datetime(trades['entry_time'], utc=True, errors='coerce')
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent = trades[trades['entry_time'] >= cutoff] if 'entry_time' in trades.columns else trades
        if recent.empty:
            return None

        wins = int((recent.get('result') == 1).sum()) if 'result' in recent.columns else 0
        losses = int((recent.get('result') == 0).sum()) if 'result' in recent.columns else 0
        total = len(recent)
        total_pl = float(recent.get('profit_loss', pd.Series(dtype=float)).sum()) if 'profit_loss' in recent.columns else 0.0

        stats = {
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'total_profit': total_pl,
            'win_rate': (wins / total) if total > 0 else 0.0,
            'avg_profit': (total_pl / total) if total > 0 else 0.0,
        }

        if 'entry_time' in recent.columns and 'result' in recent.columns:
            recent = recent.copy()
            recent['hour'] = recent['entry_time'].dt.hour
            stats['hourly_performance'] = recent.groupby('hour')['result'].agg(['count', 'mean']).round(3)

        return stats

    def generate_report(self):
        print("\n" + "="*60)
        print("거래 시스템 종합 리포트")
        print("="*60)

        for window, title in [(7, "최근 7일 성과"), (30, "최근 30일 성과")]:
            s = self.analyze_recent_trades(window)
            if s:
                print(f"\n[{title}]")
                print(f"총 거래: {s['total_trades']}")
                print(f"승/패: {s['wins']}/{s['losses']}")
                print(f"승률: {s['win_rate']:.2%}")
                print(f"총 손익: ${s['total_profit']:.2f}")
                print(f"평균 손익: ${s['avg_profit']:.2f}")

        print("\n" + "="*60)

    # ---------------------------
    # 3) 모델 스냅샷 & 비교/교체
    # ---------------------------
    def snapshot_model(self, trainer, tag="pre"):
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = os.path.join(self.config.MODEL_DIR, f"backup/model_{tag}_{ts}.pkl")
        os.makedirs(os.path.dirname(backup), exist_ok=True)
        trainer.save_model(backup)
        return backup

    def _evaluate_on_tail(self, trainer, X, y, tail_ratio=0.1):
        """마지막 tail_ratio 구간으로 성능 계산"""
        n = len(X)
        if n == 0:
            return None
        start = int(n * (1 - tail_ratio))
        Xte, yte = X.iloc[start:], y.iloc[start:]
        if len(Xte) == 0:
            return None

        # 확률로 평가 → threshold 0.5
        p = trainer.predict_proba(Xte)
        pred = (p > 0.5).astype(int)
        acc = float((pred == yte.values).mean())
        win_rate = acc  # 동일 개념
        # 보수적으로 precision/recall은 간단 계산 생략 가능 (원하면 추가)
        return {"accuracy": acc, "win_rate": win_rate}

    def compare_and_apply(self, old_trainer, new_trainer, X, y, min_improve=None):
        """
        old/new 모델 tail 구간 성능 비교하여 교체/롤백 결정.
        return: (applied: bool, metrics_old, metrics_new)
        """
        if min_improve is None:
            min_improve = self.min_improve

        m_old = self._evaluate_on_tail(old_trainer, X, y) if old_trainer and old_trainer.models else None
        m_new = self._evaluate_on_tail(new_trainer, X, y)

        print("\n" + "-"*60)
        print("모델 A/B 비교 (tail 구간)")
        if m_old:
            print(f"OLD  {self.replace_metric}: {m_old[self.replace_metric]:.4f}")
        else:
            print("OLD  없음")
        if m_new:
            print(f"NEW  {self.replace_metric}: {m_new[self.replace_metric]:.4f}")
        print("-"*60)

        # old가 없으면 new 적용
        if not m_old:
            new_trainer.save_model()
            print("→ 기존 모델 없음: 새 모델 저장")
            return True, m_old, m_new

        # 비교 기준 충족 여부
        if (m_new[self.replace_metric] - m_old[self.replace_metric]) >= min_improve:
            # 교체
            self.snapshot_model(old_trainer, tag="prev")  # 백업
            new_trainer.save_model()
            print("→ 새 모델로 교체 (임계 충족)")
            return True, m_old, m_new
        else:
            print("→ 기존 모델 유지 (임계 미충족)")
            return False, m_old, m_new

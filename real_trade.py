# real_trade.py - 실시간 거래 및 검증 (동시 보유/수동 매매 보조 로그 최적화)

import os
import json
import time
import uuid
import warnings
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings('ignore')


# -----------------------------
# Binance API (시뮬레이션 fallback)
# -----------------------------
class BinanceAPIClient:
    """바이낸스 API 클라이언트 (실패 시 시뮬레이션)"""

    def __init__(self, api_key=None, api_secret=None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.binance.com"

    def get_current_price(self, symbol="BTCUSDT"):
        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            r = requests.get(url, params={"symbol": symbol}, timeout=5)
            r.raise_for_status()
            return float(r.json()["price"])
        except Exception:
            # 시뮬레이션
            return float(np.random.uniform(40000, 45000))

    def get_klines(self, symbol="BTCUSDT", interval="1m", limit=500):
        try:
            url = f"{self.base_url}/api/v3/klines"
            r = requests.get(
                url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=7
            )
            r.raise_for_status()
            data = r.json()
            df = pd.DataFrame(
                data,
                columns=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = df[c].astype(float)
            return df[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception:
            # 시뮬레이션
            limit = int(limit)
            ts = pd.date_range(end=datetime.now(timezone.utc), periods=limit, freq="1min", tz="UTC")
            prices = np.random.randn(limit).cumsum() + 42000
            rows = []
            for i, t in enumerate(ts):
                base = prices[i]
                o = base + np.random.uniform(-50, 50)
                c = base + np.random.uniform(-50, 50)
                h = max(o, c) + np.random.uniform(0, 100)
                l = min(o, c) - np.random.uniform(0, 100)
                v = np.random.uniform(100, 1000)
                rows.append({"timestamp": t, "open": o, "high": h, "low": l, "close": c, "volume": v})
            return pd.DataFrame(rows)


# -----------------------------
# 데이터 구조
# -----------------------------
@dataclass
class TradePosition:
    trade_id: str
    entry_time: datetime
    entry_price: float
    prediction: int         # 1: UP(LONG), 0: DOWN(SHORT)
    confidence: float       # 예측 확률
    amount: float
    status: str = "open"    # open/closed
    exit_time: datetime | None = None
    exit_price: float | None = None
    result: int | None = None
    profit_loss: float | None = None

    def as_log_row(self):
        return {
            "trade_id": self.trade_id,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "amount": self.amount,
            "status": self.status,
            "exit_time": self.exit_time.isoformat() if self.exit_time else "",
            "exit_price": self.exit_price if self.exit_price is not None else "",
            "result": self.result if self.result is not None else "",
            "profit_loss": self.profit_loss if self.profit_loss is not None else "",
        }


# -----------------------------
# 실시간 트레이더
# -----------------------------
class RealTimeTrader:
    """
    - 최대 동시 보유 5개 (기본)
    - 각 포지션은 '진입시각 + 10분'에 자동 종료
    - 신뢰도(확률) 필터 없음 (로그만 출력)
    - 1분마다 거래 통계 요약 로그 출력
    - 모델 교체 시 승률 초기화 지원(on_model_refreshed)
    """

    def __init__(self, config, model_trainer, api_client=None, max_concurrent=5, symbol="BTCUSDT"):
        self.config = config
        self.model_trainer = model_trainer
        self.api_client = api_client or BinanceAPIClient(
            api_key=getattr(config, "BINANCE_API_KEY", None),
            api_secret=getattr(config, "BINANCE_API_SECRET", None),
        )
        self.symbol = symbol

        # 상태
        self.is_running = False
        self.open_positions: dict[str, TradePosition] = {}  # trade_id -> TradePosition
        self.max_concurrent = max_concurrent

        # 최근 거래 결과(승/패)만 저장하여 창 승률 계산
        self.trade_history = deque(maxlen=self.config.EVALUATION_WINDOW)

        # 누적 성능 메트릭
        self.metrics = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "total_profit": 0.0,
            "win_rate": 0.0,
        }

        # 실패 패턴 필터 로드(있으면)
        self.trade_filters = {}
        self.load_filters()

    # ------------- 필터 로드/저장 -------------
    def load_filters(self):
        path = os.path.join(self.config.FEATURE_LOG_DIR, "trade_filters.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.trade_filters = json.load(f)
                print(f"거래 필터 로드: {len(self.trade_filters)}개")
            except Exception:
                self.trade_filters = {}

    # ------------- 피처 생성 -------------
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        from model_train import FeatureEngineer
        fe = FeatureEngineer()
        feats = fe.create_feature_pool(df)
        return feats

    # ------------- 예측 -------------
    def _predict(self, features: pd.DataFrame) -> tuple[int | None, float]:
        if not getattr(self.model_trainer, "models", None):
            print("모델이 로드되지 않음. 예측 스킵")
            return None, 0.5

        try:
            X = features.iloc[[-1]]  # 마지막 시점
            proba = float(self.model_trainer.predict_proba(X)[0])
            pred = 1 if proba > 0.5 else 0
            return pred, proba
        except Exception as e:
            print(f"예측 오류: {e}")
            return None, 0.5

    # ------------- 진입 실행 -------------
    def _enter_position(self, prediction: int, confidence: float, amount: float = 100.0) -> str:
        trade_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc)
        entry_price = float(self.api_client.get_current_price(self.symbol))

        pos = TradePosition(
            trade_id=trade_id,
            entry_time=now,
            entry_price=entry_price,
            prediction=prediction,
            confidence=confidence,
            amount=amount,
        )
        self.open_positions[trade_id] = pos
        self._append_trade_log(pos)

        # 수동 매매 보조용 진입 로그
        self._log_entry_signal(pos)

        return trade_id

    # ------------- 포지션 종료 체크/정산 -------------
    def _close_due_positions(self):
        """만기(10분) 도래한 포지션 종료"""
        if not self.open_positions:
            return

        now = datetime.now(timezone.utc)
        to_close = []
        for tid, pos in self.open_positions.items():
            # entry_time + 10분 도달 시 종료
            if now >= pos.entry_time + timedelta(minutes=self.config.PREDICTION_WINDOW):
                to_close.append(tid)

        for tid in to_close:
            self._close_position(tid)

    def _close_position(self, trade_id: str):
        pos = self.open_positions.get(trade_id)
        if not pos:
            return

        exit_price = float(self.api_client.get_current_price(self.symbol))
        now = datetime.now(timezone.utc)

        actual_direction = 1 if exit_price > pos.entry_price else 0
        is_win = 1 if pos.prediction == actual_direction else 0
        profit = pos.amount * self.config.WIN_RATE if is_win else -pos.amount

        # 업데이트
        pos.exit_time = now
        pos.exit_price = exit_price
        pos.result = is_win
        pos.profit_loss = profit
        pos.status = "closed"

        # 통계 반영
        self._update_metrics(is_win, profit)

        # 파일 로그 업데이트
        self._update_trade_log_result(pos)

        # 종료 로그(간결)
        direction = "LONG" if pos.prediction == 1 else "SHORT"
        print(
            "\n"
            "-----------------------------------------------\n"
            f"🔻 포지션 종료\n"
            f"  • 거래 ID     : {pos.trade_id}\n"
            f"  • 방향        : {direction}\n"
            f"  • 진입가/종가 : {pos.entry_price:.2f}  →  {pos.exit_price:.2f}\n"
            f"  • 결과        : {'승' if is_win else '패'}  ({pos.profit_loss:+.2f} USD)\n"
            f"  • 기간        : 10분 고정 만기\n"
            "-----------------------------------------------\n"
        )

        # 메모리 정리
        self.open_positions.pop(trade_id, None)

    # ------------- 통계 -------------
    def _update_metrics(self, is_win: int, profit: float):
        self.metrics["total_trades"] += 1
        if is_win:
            self.metrics["wins"] += 1
        else:
            self.metrics["losses"] += 1
        self.metrics["total_profit"] += float(profit)
        total = self.metrics["wins"] + self.metrics["losses"]
        self.metrics["win_rate"] = (self.metrics["wins"] / total) if total > 0 else 0.0

        self.trade_history.append(is_win)

    def _should_retrain(self) -> bool:
        n = len(self.trade_history)
        if n < self.config.EVALUATION_WINDOW:
            return False
        wr = sum(self.trade_history) / n
        print(f"[평가] 최근 {n}건 승률: {wr*100:.2f}% (임계 {self.config.RETRAIN_THRESHOLD*100:.0f}%)")
        return wr < self.config.RETRAIN_THRESHOLD

    # ------------- 파일 로그 I/O -------------
    def _append_trade_log(self, pos: TradePosition):
        path = os.path.join(self.config.TRADE_LOG_DIR, "trades.csv")
        new_row = pd.DataFrame([pos.as_log_row()])
        if os.path.exists(path):
            try:
                old = pd.read_csv(path)
                df = pd.concat([old, new_row], ignore_index=True)
            except Exception:
                df = new_row
        else:
            df = new_row
        df.to_csv(path, index=False)

    def _update_trade_log_result(self, pos: TradePosition):
        from data_merge import DataMerger
        merger = DataMerger(self.config)
        merger.update_trade_result(pos.trade_id, pos.result, pos.profit_loss)

    def _save_feature_snapshot(self, features: pd.DataFrame, trade_id: str):
        # 현재 시점 한 행 저장
        snap = features.iloc[[-1]].copy()
        snap["trade_id"] = trade_id
        snap["timestamp"] = datetime.now(timezone.utc)
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = os.path.join(self.config.FEATURE_LOG_DIR, f"features_{today}.csv")
        if os.path.exists(path):
            try:
                old = pd.read_csv(path)
                df = pd.concat([old, snap], ignore_index=True)
            except Exception:
                df = snap
        else:
            df = snap
        df.to_csv(path, index=False)

    # ------------- 콘솔 로그(수동 매매 보조) -------------
    def _log_entry_signal(self, pos: TradePosition):
        direction = "LONG" if pos.prediction == 1 else "SHORT"
        print(
            "\n"
            "-----------------------------------------------\n"
            f"🟢 진입 신호 감지\n"
            f"  • 진입 시간   : {pos.entry_time.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"  • 포지션      : {direction}\n"
            f"  • 진입 가격   : {pos.entry_price:.2f}\n"
            f"  • 예측 확률   : {pos.confidence*100:.2f}%\n"
            "-----------------------------------------------\n"
        )

    def _log_summary_stats(self):
        m = self.metrics
        winrate_pct = m["win_rate"] * 100.0
        print(
            "\n"
            "===============================================\n"
            "                 📊 거래 통계 요약\n"
            "-----------------------------------------------\n"
            f"  • 진입 포지션 수 : {len(self.open_positions)} 개\n"
            f"  • 승리 | 패배    : {m['wins']} | {m['losses']}\n"
            f"  • 현재 승률      : {winrate_pct:.2f}%\n"
            "===============================================\n"
        )

    # ------------- 모델 새로고침 훅 -------------
    def on_model_refreshed(self):
        """
        메인 파이프라인 재학습 후 호출할 것:
        - 새 모델 로드 (외부에서 load_model 완료되어 있을 것)
        - 승률/통계 초기화 (요청사항)
        - 오픈 포지션은 계속 진행 (만기 시 정상 청산)
        """
        self.trade_history.clear()
        self.metrics.update({"total_trades": 0, "wins": 0, "losses": 0, "total_profit": 0.0, "win_rate": 0.0})
        self.load_filters()
        print("[모델 교체] 성능 통계 초기화 완료. 오픈 포지션은 유지됩니다.")

    # ------------- 재학습 트리거 파일 -------------
    def _trigger_retrain(self):
        flag = os.path.join(self.config.BASE_DIR, ".retrain_required")
        with open(flag, "w", encoding="utf-8") as f:
            f.write(datetime.now(timezone.utc).isoformat())
        print("[재학습] 플래그 파일 생성 완료")

    # ------------- 메인 루프 -------------
    def run_live_trading(self, duration_hours: float = 1.0, entry_interval_minutes: int = 1, amount: float = 100.0):
        """
        - duration_hours 동안 루프 실행
        - entry_interval_minutes 간격으로 신규 진입 시도
        - 포지션 만기는 10분 고정(Config.PREDICTION_WINDOW)
        - 1분마다 요약 통계 출력
        """
        print("\n실시간 거래 시작")
        print("===============================================")

        if not getattr(self.model_trainer, "models", None):
            print("모델 로딩 시도...")
            if not self.model_trainer.load_model():
                print("모델을 찾지 못했습니다. 먼저 학습을 진행하세요.")
                return

        self.is_running = True
        start = datetime.now(timezone.utc)
        end = start + timedelta(hours=duration_hours)
        last_entry_try: datetime | None = None
        next_summary_at = start + timedelta(minutes=1)

        try:
            while self.is_running and datetime.now(timezone.utc) < end:
                now = datetime.now(timezone.utc)

                # 1) 만기 포지션 종료
                self._close_due_positions()

                # 2) 신규 진입 시도 (간격 + 동시보유 한도 체크)
                if (last_entry_try is None) or ((now - last_entry_try).total_seconds() >= entry_interval_minutes * 60):
                    last_entry_try = now

                    if len(self.open_positions) < self.max_concurrent:
                        # 최신 데이터 수집
                        df = self.api_client.get_klines(symbol=self.symbol, interval="1m", limit=500)
                        if not df.empty:
                            feats = self._prepare_features(df)
                            pred, conf = self._predict(feats)
                            if pred is not None:
                                tid = self._enter_position(pred, conf, amount=amount)
                                # 피처 스냅샷 로그
                                self._save_feature_snapshot(feats, tid)

                # 3) 1분 요약 로그
                if now >= next_summary_at:
                    self._log_summary_stats()
                    next_summary_at = now + timedelta(minutes=1)

                # 4) 재학습 필요 시 플래그 (오픈 포지션 유지)
                if self._should_retrain():
                    self._trigger_retrain()

                time.sleep(1.0)

        except KeyboardInterrupt:
            print("\n거래 중단 요청(Ctrl+C)")
        finally:
            self.is_running = False
            # 루프 종료 시 남은 포지션은 자연 만기 처리(강제청산 없음)
            self._log_summary_stats()
            print("실시간 거래 종료")
            print("===============================================")


# -----------------------------
# 간단 모니터 (main_pipe 호환)
# -----------------------------
class TradingMonitor:
    """거래 통계/리포트"""

    def __init__(self, config):
        self.config = config

    def analyze_recent_trades(self, days=7):
        from data_merge import DataMerger
        merger = DataMerger(self.config)
        trades = merger.load_trade_logs()
        if trades.empty:
            return None

        trades = trades.copy()
        if "entry_time" in trades.columns:
            trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            trades = trades[trades["entry_time"] >= cutoff]

        if trades.empty:
            return None

        stats = {
            "total_trades": int(trades.shape[0]),
            "wins": int((trades.get("result") == 1).sum()) if "result" in trades.columns else 0,
            "losses": int((trades.get("result") == 0).sum()) if "result" in trades.columns else 0,
            "total_profit": float(trades.get("profit_loss", pd.Series(dtype=float)).fillna(0).sum()),
        }
        denom = max(1, stats["wins"] + stats["losses"])
        stats["win_rate"] = stats["wins"] / denom
        if stats["total_trades"] > 0:
            stats["avg_profit"] = stats["total_profit"] / stats["total_trades"]
        return stats

    def generate_report(self):
        print("\n" + "=" * 60)
        print("거래 시스템 종합 리포트")
        print("=" * 60)
        for days, label in [(7, "최근 7일"), (30, "최근 30일")]:
            s = self.analyze_recent_trades(days)
            if not s:
                print(f"{label}: 데이터 없음")
                continue
            print(f"\n[{label}]")
            print(f"총 거래: {s['total_trades']}")
            print(f"승/패: {s['wins']}/{s['losses']}")
            print(f"승률: {s.get('win_rate', 0):.2%}")
            print(f"총 손익: ${s['total_profit']:.2f}")
            print(f"평균 손익: ${s.get('avg_profit', 0):.2f}")
        print("\n" + "=" * 60)


# 모듈 import alias (main_pipe 호환)
RealTrader = RealTimeTrader

# real_trade.py - 실시간 거래/만기 10분 고정/동시 포지션/CSV 기록(요청 형식)

import os
import csv
import json
import time
import uuid
import warnings
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")


# -------------------------
# 파일 스키마(요청 형식)
# -------------------------
TRADE_COLUMNS = [
    "trade_id",
    "entry_time",
    "entry_price",
    "prediction",
    "confidence",
    "amount",
    "status",
    "result",
    "profit_loss",
    "exit_time",
    "entry_hour",
]


def _fmt_ts_naive(dt: datetime) -> str:
    """CSV 요구 포맷: naive datetime (예: 2025-10-17 21:59:28.975434)"""
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")


# =========================
# Binance API (현물, 단순화)
# =========================
class BinanceAPIClient:
    """바이낸스 API 클라이언트 (시뮬레이션 폴백 포함)"""

    def __init__(self, api_key=None, api_secret=None, base_url="https://api.binance.com"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url

    def get_current_price(self, symbol="BTCUSDT") -> float:
        """현재가 조회 (실패 시 시뮬레이션)"""
        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            r = requests.get(url, params={"symbol": symbol}, timeout=3)
            r.raise_for_status()
            return float(r.json()["price"])
        except Exception:
            # 시뮬레이션: 40k~45k 범위 랜덤
            return float(np.random.uniform(40000, 45000))

    def get_klines(self, symbol="BTCUSDT", interval="1m", limit=500, start_time=None, end_time=None) -> pd.DataFrame:
        """캔들 조회 (start_time/end_time는 ms단위, 옵션)"""
        try:
            url = f"{self.base_url}/api/v3/klines"
            params = {"symbol": symbol, "interval": interval, "limit": int(limit)}
            if start_time is not None:
                params["startTime"] = int(start_time)
            if end_time is not None:
                params["endTime"] = int(end_time)
            r = requests.get(url, params=params, timeout=4)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                raise ValueError("Invalid kline response")

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
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = df[c].astype(float)
            return df[["open_time", "close_time", "open", "high", "low", "close", "volume"]]
        except Exception:
            # 시뮬레이션 1분봉
            now = datetime.now()
            idx = pd.date_range(end=now, periods=limit, freq="1min")
            base = 42000 + np.cumsum(np.random.randn(limit)) * 10
            o = base + np.random.uniform(-10, 10, size=limit)
            c = base + np.random.uniform(-10, 10, size=limit)
            h = np.maximum(o, c) + np.random.uniform(0, 20, size=limit)
            l = np.minimum(o, c) - np.random.uniform(0, 20, size=limit)
            v = np.random.uniform(100, 1000, size=limit)
            sim = pd.DataFrame(
                {
                    "open_time": idx,
                    "close_time": idx + pd.to_timedelta(59, unit="s"),
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                }
            )
            return sim

    # ---- 핵심: 만기 시점 가격 ----
    def get_price_at(self, ts: datetime, symbol="BTCUSDT") -> float:
        """
        만기(정확히 10분 후) 시점 가격.
        - 우선 해당 시점을 포함하는 1분봉의 'close'를 사용(현실적 폴백).
        - 더 정밀한 초 단위 체결가가 필요하면, 여기를 트레이드/틱 데이터로 교체.
        """
        try:
            # ts가 포함될 캔들을 확보(앞뒤 2분)
            ts_ms = int(ts.timestamp() * 1000)
            start_ms = ts_ms - 120_000
            end_ms = ts_ms + 1_000  # 살짝 여유
            df = self.get_klines(symbol=symbol, interval="1m", start_time=start_ms, end_time=end_ms, limit=5)
            if df.empty:
                raise ValueError("No klines returned")
            # ts가 포함되는 캔들: open_time <= ts <= close_time
            row = df[(df["open_time"] <= ts) & (df["close_time"] >= ts)]
            if not row.empty:
                return float(row.iloc[-1]["close"])
            # 없다면 ts 이전 가장 최근 캔들의 종가
            row2 = df[df["close_time"] <= ts]
            if not row2.empty:
                return float(row2.iloc[-1]["close"])
            # 최후 폴백: 현재가
            return self.get_current_price(symbol=symbol)
        except Exception:
            return self.get_current_price(symbol=symbol)


# =========================
# 트레이더
# =========================
class RealTimeTrader:
    """
    - 동시 포지션 최대 N개
    - 만기 '정확히' 10분: exit_time = entry_time + 10분 (파일/로그 모두 동일)
    - CSV 기록은 청산 시 1줄
    - 예측 임계치(기본 0.6 이상) 필터
    """

    def __init__(self, config, model_trainer, api_client=None):
        self.config = config
        self.model_trainer = model_trainer
        self.api = api_client or BinanceAPIClient(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)

        # 동시 포지션 관리
        self.max_positions = 5
        self.open_positions: list[dict] = []  # 각 포지션: dict(entry_time, expire_time, ...)
        self.is_running = False

        # 성능 추적
        self.performance_metrics = {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "long_wins": 0,
            "long_losses": 0,
            "short_wins": 0,
            "short_losses": 0,
            "total_profit": 0.0,
        }
        self.recent_results = deque(maxlen=self.config.EVALUATION_WINDOW)

        # 임계치(기본 0.6)
        self.conf_threshold = getattr(self.config, "CONF_THRESHOLD", 0.60)

        # 폴더 준비
        os.makedirs(self.config.TRADE_LOG_DIR, exist_ok=True)
        os.makedirs(self.config.FEATURE_LOG_DIR, exist_ok=True)

    # ---------- 내부 유틸 ----------
    def _append_trade_row(self, row_dict: dict):
        # 요청: tades.csv (오타 그대로) + trades.csv 동시 기록
        for fname in ("tades.csv", "trades.csv"):
            path = os.path.join(self.config.TRADE_LOG_DIR, fname)
            exists = os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=TRADE_COLUMNS)
                if not exists:
                    w.writeheader()
                complete = {k: row_dict.get(k, "") for k in TRADE_COLUMNS}
                w.writerow(complete)

    def _print_periodic_stats(self):
        m = self.performance_metrics
        total = m["wins"] + m["losses"]
        winrate = (m["wins"] / total * 100) if total > 0 else 0.0
        long_total = m["long_wins"] + m["long_losses"]
        short_total = m["short_wins"] + m["short_losses"]
        long_wr = (m["long_wins"] / long_total * 100) if long_total > 0 else 0.0
        short_wr = (m["short_wins"] / short_total * 100) if short_total > 0 else 0.0
        long_ratio = (long_total / total * 100) if total > 0 else 0.0
        short_ratio = (short_total / total * 100) if total > 0 else 0.0

        print(
            "\n===============================================\n"
            "거래통계\n"
            f"진입 포지션갯수 {len(self.open_positions)}개\n"
            f"승 {m['wins']} | 패 {m['losses']}  (승률 {winrate:.2f}%)\n"
            f"롱 비중 {long_ratio:.1f}% (승률 {long_wr:.2f}%) | "
            f"숏 비중 {short_ratio:.1f}% (승률 {short_wr:.2f}%)\n"
            "===============================================\n"
        )

    def _update_performance(self, is_win: bool, prediction: int, profit: float):
        self.performance_metrics["total_trades"] += 1
        if is_win:
            self.performance_metrics["wins"] += 1
            if prediction == 1:
                self.performance_metrics["long_wins"] += 1
            else:
                self.performance_metrics["short_wins"] += 1
        else:
            self.performance_metrics["losses"] += 1
            if prediction == 1:
                self.performance_metrics["long_losses"] += 1
            else:
                self.performance_metrics["short_losses"] += 1
        self.performance_metrics["total_profit"] += float(profit)

        self.recent_results.append(1 if is_win else 0)
        # 평가 창 도달 시 승률 체크(재학습 트리거 파일만 세팅)
        if len(self.recent_results) == self.config.EVALUATION_WINDOW:
            wr = sum(self.recent_results) / self.config.EVALUATION_WINDOW
            print(f"[최근 {self.config.EVALUATION_WINDOW}건 승률] {wr:.2%}")
            if wr < self.config.RETRAIN_THRESHOLD:
                flag_path = os.path.join(self.config.BASE_DIR, ".retrain_required")
                with open(flag_path, "w") as f:
                    f.write(datetime.now().isoformat())
                print(f"→ 승률 {wr:.2%} < {self.config.RETRAIN_THRESHOLD:.0%} → 재학습 플래그 생성")

    # ---------- 피처/예측 ----------
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        from model_train import FeatureEngineer  # 순환 import 방지용 내부 임포트
        fe = FeatureEngineer()
        feats = fe.create_feature_pool(df)
        return feats

    def _predict_one(self, feats: pd.DataFrame) -> tuple[int | None, float]:
        if not self.model_trainer.models:
            try:
                self.model_trainer.load_model()
            except Exception:
                print("모델이 없습니다. 학습 후 실행하세요.")
                return None, 0.5

        try:
            x = feats.iloc[[-1]]
            proba = float(self.model_trainer.predict_proba(x)[0])
            pred = 1 if proba >= 0.5 else 0
            return pred, proba
        except Exception as e:
            print(f"예측 오류: {e}")
            return None, 0.5

    # ---------- 진입/청산 ----------
    def execute_trade(self, prediction: int, confidence: float, amount: float = 100.0) -> str:
        trade_id = str(uuid.uuid4())[:8]
        entry_dt = datetime.now()  # naive
        entry_price = float(self.api.get_current_price())

        # 만기 딱 10분 고정
        expire_dt = entry_dt + timedelta(minutes=self.config.PREDICTION_WINDOW)

        pos = {
            "trade_id": trade_id,
            "entry_time": entry_dt,
            "entry_price": entry_price,
            "prediction": int(prediction),   # 1:UP, 0:DOWN
            "confidence": float(confidence),
            "amount": float(amount),
            "expire_time": expire_dt,        # 만기 고정
        }

        self.open_positions.append(pos)

        # 진입 로그
        direction = "UP" if prediction == 1 else "DOWN"
        print("\n-----------------------------------------------")
        print("진입")
        print(f"- id: {trade_id}")
        print(f"- 시간: {_fmt_ts_naive(entry_dt)}")
        print(f"- 가격: {entry_price:.2f}")
        print(f"- 방향: {direction}")
        print(f"- 예측확률: {confidence:.2%}")
        print("-----------------------------------------------\n")

        return trade_id

    def _close_trade(self, pos: dict):
        """만기 정확히 10분에 청산, 파일 1줄 기록"""
        entry_dt = pos["entry_time"]
        exit_dt = pos["expire_time"]  # 만기 시간 그대로 기록
        entry_price = float(pos["entry_price"])
        amount = float(pos["amount"])
        pred = int(pos["prediction"])

        # 만기 시점 가격 가져오기 (가능하면 정확 시점, 폴백: 해당 캔들 종가)
        exit_price = float(self.api.get_price_at(exit_dt))

        actual_dir = 1 if exit_price > entry_price else 0
        is_win = (pred == actual_dir)

        if is_win:
            profit = amount * self.config.WIN_RATE
            result = 1.0
        else:
            profit = -amount
            result = 0.0

        # 통계 업데이트
        self._update_performance(is_win, pred, profit)

        # CSV 1줄 기록(요청 형식)
        row = {
            "trade_id": pos["trade_id"],
            "entry_time": _fmt_ts_naive(entry_dt),
            "entry_price": round(entry_price, 2),
            "prediction": pred,
            "confidence": float(pos["confidence"]),
            "amount": int(amount),
            "status": "open",  # 기존 파일에서 쓰던 그대로 유지
            "result": result,
            "profit_loss": round(float(profit), 1) if profit % 1 != 0 else round(float(profit), 1),
            "exit_time": _fmt_ts_naive(exit_dt),
            "entry_hour": entry_dt.hour,
        }
        self._append_trade_row(row)

        # 청산 로그
        print("\n-----------------------------------------------")
        print("청산")
        print(f"- id: {pos['trade_id']}")
        print(f"- 결과: {'승' if is_win else '패'}")
        print(f"- 진입가/만기가: {entry_price:.2f} -> {exit_price:.2f}")
        print(f"- 손익: {profit:.2f}")
        print("-----------------------------------------------\n")

    # ---------- 메인 루프 ----------
    def run_live_trading(self, duration_hours=None, entry_interval_minutes=1, poll_seconds=5, amount=100.0):
        """
        duration_hours=None → 무제한
        entry_interval_minutes → 신규 진입 간격(분)
        poll_seconds → 만기 도래 체크 주기(초)
        """
        if duration_hours is None:
            print(f"\n실시간 거래 시작: 무제한 모드 (임계 {self.conf_threshold:.2f})")
        else:
            print(f"\n실시간 거래 시작: 기간 {duration_hours}h (임계 {self.conf_threshold:.2f})")

        self.is_running = True
        start = datetime.now()
        end = None if duration_hours is None else start + timedelta(hours=duration_hours)

        last_entry_at: datetime | None = None
        last_stats_print = datetime.now()

        # 모델 준비
        if not self.model_trainer.models:
            ok = self.model_trainer.load_model()
            if not ok:
                print("모델을 먼저 학습/저장하세요.")
                return

        try:
            while self.is_running and (end is None or datetime.now() < end):
                now = datetime.now()

                # 1) 만기 도래 포지션 청산 (정확히 10분)
                to_close = [p for p in self.open_positions if now >= p["expire_time"]]
                if to_close:
                    for p in to_close:
                        self._close_trade(p)
                    # 리스트에서 제거
                    self.open_positions = [p for p in self.open_positions if p not in to_close]

                # 2) 신규 진입(간격/동시보유/임계 충족)
                can_enter = (
                    (last_entry_at is None or (now - last_entry_at).total_seconds() >= entry_interval_minutes * 60)
                    and len(self.open_positions) < self.max_positions
                )
                if can_enter:
                    # 최신 1분봉 기반 피처
                    df = self.api.get_klines(limit=500)
                    feats = self._prepare_features(df)
                    pred, proba = self._predict_one(feats)

                    if pred is not None and proba >= self.conf_threshold:
                        self.execute_trade(prediction=pred, confidence=proba, amount=amount)
                        last_entry_at = now
                    # (요청) 임계 미달 스킵 로그는 출력하지 않음

                # 3) 1분마다 요약 통계
                if (now - last_stats_print).total_seconds() >= 60:
                    self._print_periodic_stats()
                    last_stats_print = now

                time.sleep(poll_seconds)

        except KeyboardInterrupt:
            print("\n[중단] 사용자 인터럽트")
        finally:
            # 종료 시 미청산 포지션은 전부 '만기 시각'으로 청산 처리
            if self.open_positions:
                print(f"[정리] 미청산 {len(self.open_positions)}건 만기 처리...")
                for p in list(self.open_positions):
                    if datetime.now() < p["expire_time"]:
                        # 만기 시각까지 기다리지 않고, '만기 시각' 기준으로 가격을 조회해 청산
                        self._close_trade(p)
                self.open_positions.clear()

            # 마지막 통계
            self._print_periodic_stats()
            self.is_running = False

    # 재학습 후 승률 초기화 요청 시 사용할 수 있게 제공
    def reset_performance(self):
        self.performance_metrics.update(
            dict(
                total_trades=0,
                wins=0,
                losses=0,
                long_wins=0,
                long_losses=0,
                short_wins=0,
                short_losses=0,
                total_profit=0.0,
            )
        )
        self.recent_results.clear()
        print("[성능 초기화] 누적 승패 및 최근 창을 초기화했습니다.")

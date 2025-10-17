# real_trade.py - 실시간 거래 및 검증 (동시보유 + 개별 만기 + 1분 요약 로그 + 재학습 일시정지 + 그레이스 클로즈)
import os
import json
import uuid
import time
import logging
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from collections import deque

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------
# 로깅 설정
# ---------------------------------------------------------------------
LOGGER_NAME = "RealTrade"
logger = logging.getLogger(LOGGER_NAME)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
# 파일 로깅은 main_pipe에서 pipeline.log 핸들러가 있으면 공유됨.


# ---------------------------------------------------------------------
# API Client (시뮬레이션 fallback)
# ---------------------------------------------------------------------
class BinanceAPIClient:
    """바이낸스 API 클라이언트 (시뮬레이션 fallback)"""

    def __init__(self, api_key=None, api_secret=None, base_url="https://api.binance.com"):
        from config import Config
        self.api_key = api_key or Config.BINANCE_API_KEY
        self.api_secret = api_secret or Config.BINANCE_API_SECRET
        self.base_url = base_url

    def get_current_price(self, symbol="BTCUSDT"):
        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            resp = requests.get(url, params={"symbol": symbol}, timeout=3)
            data = resp.json()
            return float(data['price'])
        except Exception:
            return float(np.random.uniform(40000, 45000))

    def get_klines(self, symbol="BTCUSDT", interval="1m", limit=500):
        try:
            url = f"{self.base_url}/api/v3/klines"
            resp = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": limit}, timeout=5)
            data = resp.json()
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            for c in ['open', 'high', 'low', 'close', 'volume']:
                df[c] = df[c].astype(float)
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception:
            return self._generate_simulation_data(limit)

    def _generate_simulation_data(self, limit=500):
        ts = pd.date_range(end=datetime.utcnow(), periods=limit, freq='1min', tz='UTC')
        prices = np.random.randn(limit).cumsum() + 42000
        rows = []
        for i, t in enumerate(ts):
            base = prices[i]
            o = base + np.random.uniform(-50, 50)
            c = base + np.random.uniform(-50, 50)
            h = max(o, c) + np.random.uniform(0, 100)
            l = min(o, c) - np.random.uniform(0, 100)
            v = np.random.uniform(100, 1000)
            rows.append({'timestamp': t, 'open': o, 'high': h, 'low': l, 'close': c, 'volume': v})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Real-time Trader
# ---------------------------------------------------------------------
class RealTimeTrader:
    """
    실시간 거래 실행:
      - 동시 보유 최대 N개 (Config.MAX_CONCURRENT_POS)
      - 각 포지션은 '진입+10분'에 정확히 만기
      - SHADOW_MODE=True면 항상 페이퍼 체결
      - 신뢰도 임계값 (Config.CONF_THRESHOLD) 미달은 조용히 스킵(로그 없음)
      - ENTRY_COOLDOWN_SEC(쿨다운) 준수
      - 1분마다 요약 로그 출력
      - 재학습 중 신규진입 일시정지(pause_new_entries)
      - 세션 종료 시 잔여 포지션 그레이스 클로즈
    """

    def __init__(self, config, model_trainer, api_client=None):
        self.config = config
        self.model_trainer = model_trainer
        self.api_client = api_client or BinanceAPIClient()

        # 상태
        self.is_running = False
        self.open_positions: dict[str, dict] = {}     # trade_id -> trade_info
        self.trade_history = deque(maxlen=self.config.EVALUATION_WINDOW)

        # 성과 메트릭 (PnL은 누적 유지, 모델 교체 시 승률만 초기화)
        self.performance_metrics = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'current_streak': 0,
            'max_streak': 0,
            'total_profit': 0.0,
            'win_rate': 0.0
        }

        self.last_entry_time = None
        self.last_minute_log = datetime.utcnow()

        # 재학습 중 신규진입 일시정지 플래그
        self.pause_new_entries = False

        # 필터
        self.trade_filters = {}
        self.load_filters()

        logger.info("RealTimeTrader 초기화 완료")

    # ---------------- Config helpers ----------------
    def _max_concurrent(self) -> int:
        return getattr(self.config, "MAX_CONCURRENT_POS", 5)

    def _conf_threshold(self) -> float:
        return float(getattr(self.config, "CONF_THRESHOLD", 0.60))

    def _entry_cooldown_sec(self) -> int:
        return int(getattr(self.config, "ENTRY_COOLDOWN_SEC", 60))

    def _shadow_mode(self) -> bool:
        return bool(getattr(self.config, "SHADOW_MODE", True))

    # ---------------- Filters ----------------
    def load_filters(self):
        path = os.path.join(self.config.FEATURE_LOG_DIR, 'trade_filters.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.trade_filters = json.load(f)
            logger.info(f"거래 필터 로드: {list(self.trade_filters.keys())}")

    def save_filters(self):
        path = os.path.join(self.config.FEATURE_LOG_DIR, 'trade_filters.json')
        with open(path, 'w') as f:
            json.dump(self.trade_filters, f, indent=2)

    def apply_filters(self, features: pd.DataFrame):
        """
        True/False, reasons
        """
        ok = True
        reasons = []

        # 변동성 필터
        if 'high_volatility' in self.trade_filters:
            cfg = self.trade_filters['high_volatility']
            th = cfg.get('atr_14_threshold', float('inf'))
            if 'atr_14' in features.columns and float(features['atr_14'].iloc[-1]) > th:
                ok = False; reasons.append("high_volatility")

        # 거래량 필터
        if 'low_volume' in self.trade_filters:
            cfg = self.trade_filters['low_volume']
            th = cfg.get('volume_ratio_threshold', 0)
            if 'volume_ratio' in features.columns and float(features['volume_ratio'].iloc[-1]) < th:
                ok = False; reasons.append("low_volume")

        # 시간대 필터 (UTC)
        if 'time_based' in self.trade_filters:
            cfg = self.trade_filters['time_based']
            hour = datetime.utcnow().hour
            if hour in cfg.get('avoid_hours', []):
                ok = False; reasons.append("time_based")

        return ok, reasons

    # ---------------- Features / Predict ----------------
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        from model_train import FeatureEngineer
        fe = FeatureEngineer()
        feats = fe.create_feature_pool(df)
        return feats

    def make_prediction(self, features: pd.DataFrame):
        if not self.model_trainer.models:
            return None, 0.5

        ok, _ = self.apply_filters(features)
        if not ok:
            return None, 0.5

        try:
            X_cur = features.iloc[[-1]]
            proba = float(self.model_trainer.predict_proba(X_cur)[0])
            pred = 1 if proba > 0.5 else 0
            return pred, proba
        except Exception:
            return None, 0.5

    # ---------------- Trade lifecycle ----------------
    def can_open_new_trade(self):
        if self.pause_new_entries:
            return False, "retrain_pause"
        if len(self.open_positions) >= self._max_concurrent():
            return False, "max_concurrent_reached"
        if self.last_entry_time is not None:
            if (datetime.utcnow() - self.last_entry_time).total_seconds() < self._entry_cooldown_sec():
                return False, "cooldown"
        return True, ""

    def execute_trade(self, prediction: int, confidence: float, amount: float = 100.0):
        trade_id = str(uuid.uuid4())[:8]
        entry_time = datetime.utcnow()
        entry_price = self.api_client.get_current_price()

        info = {
            'trade_id': trade_id,
            'entry_time': entry_time.replace(tzinfo=None).isoformat(),
            'entry_price': entry_price,
            'prediction': int(prediction),
            'confidence': float(confidence),
            'amount': float(amount),
            'status': 'open'
        }
        self.open_positions[trade_id] = info
        self.save_trade_log(info)

        direction = "UP" if prediction == 1 else "DOWN"
        mode = "SHADOW" if self._shadow_mode() else "LIVE"
        logger.info(f"[OPEN] id={trade_id} mode={mode} dir={direction} conf={confidence:.3f} "
                    f"entry={entry_price:.2f} amt=${amount:.2f} open={info['entry_time']}")

        self.last_entry_time = entry_time
        return trade_id

    def check_trade_result(self, trade_info: dict):
        entry_time = datetime.fromisoformat(trade_info['entry_time'])
        now = datetime.utcnow()
        if (now - entry_time).total_seconds() < self.config.PREDICTION_WINDOW * 60:
            return None  # 아직 만기 아님

        entry_price = trade_info['entry_price']
        exit_price = self.api_client.get_current_price()

        actual_dir = 1 if exit_price > entry_price else 0
        pred = trade_info['prediction']
        win = (pred == actual_dir)

        amt = trade_info['amount']
        pnl = amt * self.config.WIN_RATE if win else -amt
        result = 1 if win else 0

        trade_info.update({
            'exit_time': now.replace(tzinfo=None).isoformat(),
            'exit_price': exit_price,
            'result': result,
            'profit_loss': pnl,
            'status': 'closed'
        })

        self.update_performance(win, pnl)
        self.trade_history.append(result)
        self.update_trade_log(trade_info['trade_id'], result, pnl)

        direction = "UP" if pred == 1 else "DOWN"
        logger.info(f"[CLOSE] id={trade_info['trade_id']} dir={direction} "
                    f"entry={entry_price:.2f} exit={exit_price:.2f} "
                    f"res={'WIN' if win else 'LOSS'} pnl=${pnl:.2f} "
                    f"open={trade_info['entry_time']} close={trade_info['exit_time']}")

        return result

    def sweep_positions(self):
        if not self.open_positions:
            return
        to_close = []
        for tid, info in list(self.open_positions.items()):
            res = self.check_trade_result(info)
            if res is not None:
                to_close.append(tid)
        for tid in to_close:
            self.open_positions.pop(tid, None)

    # ---------------- Stats / Retrain ----------------
    def update_performance(self, is_win: bool, profit: float):
        m = self.performance_metrics
        m['total_trades'] += 1
        if is_win:
            m['wins'] += 1
            m['current_streak'] += 1
            m['max_streak'] = max(m['max_streak'], m['current_streak'])
        else:
            m['losses'] += 1
            m['current_streak'] = 0
        m['total_profit'] += float(profit)
        if m['total_trades'] > 0:
            m['win_rate'] = m['wins'] / m['total_trades']

    def reset_winrate(self, reason: str = "model_retrained"):
        """모델 교체 시 승률만 초기화 + 평가 히스토리 초기화"""
        m = self.performance_metrics
        m['total_trades'] = 0
        m['wins'] = 0
        m['losses'] = 0
        m['current_streak'] = 0
        m['max_streak'] = 0
        m['win_rate'] = 0.0
        self.trade_history.clear()  # ★ 평가 히스토리도 초기화
        logger.info(f"[RESET] winrate counters reset (reason={reason})")

    def check_retrain_needed(self):
        if len(self.trade_history) >= self.config.EVALUATION_WINDOW:
            wr = sum(self.trade_history) / len(self.trade_history)
            logger.info(f"[EVAL] recent_{self.config.EVALUATION_WINDOW} winrate={wr:.3f}")
            if wr < self.config.RETRAIN_THRESHOLD:
                logger.info(f"[RETRAIN] need retrain: {wr:.3f} < {self.config.RETRAIN_THRESHOLD:.2f}")
                return True
        return False

    def trigger_retrain(self):
        flag = os.path.join(self.config.BASE_DIR, '.retrain_required')
        with open(flag, 'w') as f:
            f.write(datetime.utcnow().isoformat())
        logger.info("[RETRAIN] retrain flag written")

    # ---------------- Logs / I/O ----------------
    def save_trade_log(self, trade_info: dict):
        path = os.path.join(self.config.TRADE_LOG_DIR, 'trades.csv')
        df_new = pd.DataFrame([trade_info])
        if os.path.exists(path):
            df = pd.read_csv(path)
            df = pd.concat([df, df_new], ignore_index=True)
        else:
            df = df_new
        df.to_csv(path, index=False)

    def update_trade_log(self, trade_id: str, result: int, profit_loss: float):
        from data_merge import DataMerger
        merger = DataMerger(self.config)
        merger.update_trade_result(trade_id, result, profit_loss)

    def save_feature_log(self, features: pd.DataFrame, trade_id: str):
        cur = features.iloc[[-1]].copy()
        cur['trade_id'] = trade_id
        cur['timestamp'] = datetime.utcnow().isoformat()
        fname = f'features_{datetime.utcnow().strftime("%Y%m%d")}.csv'
        path = os.path.join(self.config.FEATURE_LOG_DIR, fname)
        if os.path.exists(path):
            ex = pd.read_csv(path)
            df = pd.concat([ex, cur], ignore_index=True)
        else:
            df = cur
        df.to_csv(path, index=False)

    # ---------------- Monitoring / Pretty ----------------
    def _log_minute_summary(self):
        """
        1분마다 간결한 요약 로그:
        거래통계
        진입 포지션갯수 n개
        승 n | 패 n
        승률 xx.xx%
        """
        now = datetime.utcnow()
        if (now - self.last_minute_log).total_seconds() >= 60:
            m = self.performance_metrics
            winrate = (m['wins'] / m['total_trades'] * 100) if m['total_trades'] > 0 else 0.0
            summary = (
                "==============================================="
                f"거래통계\n"
                f"진입 포지션갯수 {len(self.open_positions)}개\n"
                f"승 {m['wins']} | 패 {m['losses']}\n"
                f"승률 {winrate:.2f}%"
                "==============================================="
            )
            logger.info(summary)
            self.last_minute_log = now

    def print_performance_summary(self):
        m = self.performance_metrics
        winrate = (m['wins'] / m['total_trades'] * 100) if m['total_trades'] > 0 else 0.0
        pretty = (
            "거래통계\n"
            f"진입 포지션갯수 {len(self.open_positions)}개\n"
            f"승 {m['wins']} | 패 {m['losses']}\n"
            f"승률 {winrate:.2f}%"
        )
        logger.info(pretty)

    # ---------------- Main loop ----------------
    def run_live_trading(self, duration_hours=1, amount=100.0):
        """
        실시간 거래 실행
        - 루프주기 5초: 만기 스윕 + 신호 시 진입
        - 신뢰도 임계 미달은 조용히 스킵 (로그 없음)
        - 1분마다 요약 로그 출력
        - 세션 종료 시 open 포지션 그레이스 클로즈
        """
        logger.info(f"실시간 거래 시작 (duration={duration_hours}h, thr={self._conf_threshold():.2f}, "
                    f"max_pos={self._max_concurrent()}, cooldown={self._entry_cooldown_sec()}s)")
        self.is_running = True

        if not self.model_trainer.models:
            if not self.model_trainer.load_model():
                logger.error("모델을 찾을 수 없습니다. 먼저 학습을 진행하세요.")
                return

        end_time = datetime.utcnow() + timedelta(hours=duration_hours)

        try:
            while datetime.utcnow() < end_time and self.is_running:
                # 1) 만기 도달 포지션 정리
                self.sweep_positions()

                # 2) 신규 진입 시도
                ok, _ = self.can_open_new_trade()
                if ok:
                    df = self.api_client.get_klines(limit=500)
                    feats = self.prepare_features(df)
                    pred, conf = self.make_prediction(feats)
                    # 신뢰도 미달은 조용히 스킵
                    if pred is not None and conf >= self._conf_threshold():
                        tid = self.execute_trade(pred, conf, amount=amount)
                        # 피처 로그 저장(선택)
                        self.save_feature_log(feats, tid)

                # 3) 1분 요약 로그
                self._log_minute_summary()

                # 4) 자동 재학습 체크
                if self.check_retrain_needed():
                    self.trigger_retrain()

                time.sleep(5)  # 루프 주기(초)

        except KeyboardInterrupt:
            logger.info("거래 중단(KeyboardInterrupt)")
        finally:
            # ★ 그레이스 클로즈: 남아있는 포지션 최대 prediction_window 분 동안 정리
            grace_deadline = datetime.utcnow() + timedelta(minutes=self.config.PREDICTION_WINDOW)
            while self.open_positions and datetime.utcnow() < grace_deadline:
                self.sweep_positions()
                time.sleep(5)

            self.is_running = False
            self.print_performance_summary()
            logger.info("실시간 거래 종료")


# ---------------------------------------------------------------------
# Trading Monitor
# ---------------------------------------------------------------------
class TradingMonitor:
    """거래 모니터링/리포트"""

    def __init__(self, config):
        self.config = config

    def analyze_recent_trades(self, days=7):
        from data_merge import DataMerger
        trades = DataMerger(self.config).load_trade_logs()
        if trades is None or trades.empty:
            logger.info("거래 기록이 없습니다.")
            return None

        cutoff = datetime.utcnow() - timedelta(days=days)
        if 'entry_time' in trades.columns:
            trades['entry_time'] = pd.to_datetime(trades['entry_time'], utc=True, errors='coerce')
            recent = trades[trades['entry_time'] >= cutoff]
        else:
            recent = trades

        if recent.empty:
            logger.info(f"최근 {days}일 기록이 없습니다.")
            return None

        stats = {
            'total_trades': int(len(recent)),
            'wins': int((recent.get('result') == 1).sum()) if 'result' in recent.columns else 0,
            'losses': int((recent.get('result') == 0).sum()) if 'result' in recent.columns else 0,
            'total_profit': float(recent.get('profit_loss', pd.Series(dtype=float)).sum() if 'profit_loss' in recent.columns else 0.0)
        }
        if stats['total_trades'] > 0:
            stats['win_rate'] = stats['wins'] / stats['total_trades']
            stats['avg_profit'] = stats['total_profit'] / stats['total_trades']

        # 시간대별 성과(UTC)
        if 'entry_time' in recent.columns:
            recent['hour'] = recent['entry_time'].dt.hour
            hour_tab = recent.groupby('hour').agg(result_count=('result', 'count'),
                                                  win_rate=('result', 'mean')).round(3)
            stats['hourly_performance'] = hour_tab

        return stats

    def generate_report(self):
        logger.info("=" * 60)
        logger.info("거래 시스템 종합 리포트")
        logger.info("=" * 60)

        for label, days in [("최근 7일", 7), ("최근 30일", 30)]:
            st = self.analyze_recent_trades(days)
            if st:
                logger.info(f"[{label}] trades={st['total_trades']} win={st['wins']} "
                            f"loss={st['losses']} winrate={st.get('win_rate', 0):.3f} "
                            f"pnl=${st['total_profit']:.2f} avg=${st.get('avg_profit', 0):.2f}")
        logger.info("=" * 60)


# 모듈 편의 별칭
RealTrader = RealTimeTrader

if __name__ == "__main__":
    from config import Config
    from model_train import ModelTrainer

    Config.create_directories()
    trainer = ModelTrainer(Config)
    trainer.load_model()

    trader = RealTimeTrader(Config, trainer)
    trader.run_live_trading(duration_hours=1, amount=100.0)

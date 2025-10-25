"""
real_trade.py (패치 완전판)
실시간 신호 생성 시스템 (30분봉 바이너리 옵션)
- 30분봉 완성 시점에만 예측 (슬롯 중복 방지)
- 패배 기반 유동성/연패/레짐/확률 필터 (동적)
- 동적 컷오프 히스테리시스 + Δp 방어 + TTL + 포지션 제한
- 켈리 추천 베팅금액 (5~250 제한) *로깅 전용
- 재학습 트리거 (50번마다 승률 체크) + 패배 원인 분석
- 모델 메타(버전/학습시각/해시) 로깅
- 모니터 스냅샷 JSON 주기 출력
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
import json
import time
from pathlib import Path

from binance.client import Client
from binance.exceptions import BinanceAPIException

from config import Config
from model_train import ModelTrainer
from feature_engineer import FeatureEngineer
from log_manager import LogManager
from timeframe_manager import TimeframeManager


class RealTrader:
    """실시간 신호 생성 시스템 (30분봉 기반)"""

    def __init__(self, symbol: str = 'BTCUSDT'):
        self.config = Config
        self.symbol = symbol

        # =============================
        # Binance Client (시세 조회 전용)
        # =============================
        api_key = self.config.BINANCE_API_KEY
        api_secret = self.config.BINANCE_API_SECRET
        if not api_key or not api_secret:
            raise ValueError(
                "Binance API 키 없음!\n"
                "환경변수 설정:\n"
                "  export BINANCE_API_KEY='your_key'\n"
                "  export BINANCE_API_SECRET='your_secret'"
            )
        self.client = Client(api_key, api_secret)
        print("✓ Binance API 연결 (시세 조회 전용)")

        # 모듈
        self.model_trainer = ModelTrainer()
        self.feature_engineer = FeatureEngineer()
        self.log_manager = LogManager()
        self.tf_manager = TimeframeManager()

        # 모델 로드 + 메타정보
        self.model_loaded = self.model_trainer.load_models()
        if not self.model_loaded:
            raise ValueError("모델 로드 실패! 학습된 모델이 필요합니다.")
        try:
            self.model_meta = self.model_trainer.get_model_meta()  # {version, trained_at, hash} 가정
        except Exception:
            self.model_meta = {}
        print(f"✓ 모델 메타: {self.model_meta or 'N/A'}")

        # =============================
        # 상태 관리
        # =============================
        self.active_positions: List[Dict] = []
        self.last_trade_time: Dict[str, datetime] = {}

        # 히스테리시스/TTL
        self.p_prev: Optional[float] = None
        self.direction_prev: Optional[int] = None
        self.signal_start_time: Optional[datetime] = None
        self.signal_direction: Optional[int] = None

        # 통계/사이클
        self.total_trades = 0
        self.cycle_count = 0

        # 1분봉 버퍼
        self.price_buffer: List[Dict] = []
        self.buffer_size = 500

        # 동적 필터 상태
        self.filter_state = self._load_filter_state()

        # 재학습 트리거
        self.retrain_check_interval = 50
        self.retrain_threshold = 0.55
        self.needs_retrain = False

        # 슬롯 중복 방지
        self.last_decision_slot: Optional[datetime] = None

        # 자본(추천금액 계산용, 거래는 수동)
        self.bankroll_path = self.config.RESULT_DIR / "bankroll.json"
        self.bankroll = self._load_bankroll()

        # 모니터 스냅샷 경로
        self.monitor_snapshot_path = self.config.RESULT_DIR / "monitor_snapshot.json"

    # =============================
    # 공용 유틸
    # =============================
    def _load_filter_state(self) -> Dict:
        p = self.config.RESULT_DIR / 'filter_state.json'
        if p.exists():
            try:
                return json.load(open(p, 'r'))
            except Exception:
                pass
        return {
            'cutoff_dynamic': self.config.CUT_OFF,
            'liquidity_threshold': self.config.LIQUIDITY_FILTER_THRESHOLD,
            'max_consecutive_losses': self.config.MAX_CONSECUTIVE_LOSSES,
            'last_updated': datetime.now(timezone.utc).isoformat(),
            'loss_patterns': {}
        }

    def _save_filter_state(self):
        p = self.config.RESULT_DIR / 'filter_state.json'
        self.filter_state['last_updated'] = datetime.now(timezone.utc).isoformat()
        json.dump(self.filter_state, open(p, 'w'), indent=2, default=str)

    def _load_bankroll(self) -> float:
        p = self.bankroll_path
        if p.exists():
            try:
                data = json.load(open(p, 'r'))
                return float(data.get("bankroll", 1000.0))
            except Exception:
                return 1000.0
        return 1000.0

    def _save_bankroll(self):
        json.dump(
            {"bankroll": float(self.bankroll), "updated": datetime.now(timezone.utc).isoformat()},
            open(self.bankroll_path, "w"),
            indent=2
        )

    @staticmethod
    def _current_30m_slot(dt_utc: datetime) -> datetime:
        dt_utc = dt_utc.replace(second=0, microsecond=0)
        return dt_utc.replace(minute=(0 if dt_utc.minute < 30 else 30))

    # =============================
    # 켈리 추천 베팅금액 (로깅용)
    # =============================
    def _kelly_stake(self, p_up: float, rr: float = 0.8) -> float:
        """
        바이옵 페이오프 가정: 승 +0.8, 패 -1.0 → b = 0.8
        f* = (b*p - (1-p))/b.  음수면 0으로.
        결과는 bankroll*f 를 [5, 250]로 클립.
        """
        b = rr
        p = float(p_up)
        q = 1.0 - p
        f_star = (b * p - q) / b
        f_star = max(0.0, f_star)
        raw = self.bankroll * f_star
        return float(np.clip(raw, 5.0, 250.0))

    # =============================
    # 패배 원인 분석(필터 업데이트)
    # =============================
    def analyze_loss_patterns(self):
        recent_trades = self.log_manager.load_recent_trades(n=100)
        if len(recent_trades) < 20 or 'result' not in recent_trades.columns:
            return
        losses = recent_trades[recent_trades['result'] == 0].copy()
        if len(losses) == 0:
            return

        print("\n[패배 원인 분석]")

        # (1) 레짐별 손실률
        if 'regime' in recent_trades.columns:
            regime_total = recent_trades['regime'].value_counts()
            regime_loss = losses['regime'].value_counts()
            for regime_val in regime_total.index:
                total_count = int(regime_total.get(regime_val, 0))
                loss_count = int(regime_loss.get(regime_val, 0))
                loss_rate = (loss_count / total_count) if total_count > 0 else 0
                name = {1: "UP", -1: "DOWN", 0: "FLAT"}.get(regime_val, f"REGIME-{regime_val}")
                print(f"  {name}: 손실률 {loss_rate:.1%} ({loss_count}/{total_count})")
                if loss_rate > 0.7 and total_count >= 10:
                    self.filter_state['loss_patterns'][f'regime_{regime_val}'] = {
                        'loss_rate': loss_rate, 'count': total_count
                    }
                    print(f"    ⚠️ {name} 레짐 손실률 높음!")

        # (2) 확률 구간별 손실률
        if 'p_up' in recent_trades.columns:
            bins = [0.5, 0.55, 0.6, 0.65, 0.7, 1.0]
            labels = ['0.50-0.55', '0.55-0.60', '0.60-0.65', '0.65-0.70', '0.70+']
            recent_trades['p_bin'] = pd.cut(recent_trades['p_up'].abs(), bins=bins, labels=labels, include_lowest=True)
            for label in labels:
                sub = recent_trades[recent_trades['p_bin'] == label]
                if len(sub) > 0 and 'result' in sub.columns:
                    loss_rate = (sub['result'] == 0).mean()
                    print(f"  확률 {label}: 손실률 {loss_rate:.1%} ({int((sub['result']==0).sum())}/{len(sub)})")
                    if loss_rate > 0.6 and len(sub) >= 10:
                        self.filter_state['loss_patterns'][f'p_bin_{label}'] = {
                            'loss_rate': float(loss_rate), 'count': int(len(sub))
                        }

        # (3) 동적 컷오프 보정
        if 'result' in recent_trades.columns:
            recent_win_rate = (recent_trades['result'] == 1).mean()
            if recent_win_rate < 0.50:
                new_cutoff = min(self.filter_state['cutoff_dynamic'] + 0.02, 0.65)
                if new_cutoff != self.filter_state['cutoff_dynamic']:
                    print(f"  ⚠️ Cutoff 조정: {self.filter_state['cutoff_dynamic']:.2f} → {new_cutoff:.2f}")
                    self.filter_state['cutoff_dynamic'] = new_cutoff
            elif recent_win_rate > 0.60:
                new_cutoff = max(self.filter_state['cutoff_dynamic'] - 0.01, self.config.CUT_OFF)
                if new_cutoff != self.filter_state['cutoff_dynamic']:
                    print(f"  ✓ Cutoff 조정: {self.filter_state['cutoff_dynamic']:.2f} → {new_cutoff:.2f}")
                    self.filter_state['cutoff_dynamic'] = new_cutoff

        self._save_filter_state()

    # =============================
    # 재학습 체크
    # =============================
    def check_retrain_trigger(self):
        if self.total_trades == 0:
            return
        if self.total_trades % self.retrain_check_interval != 0:
            return
        print(f"\n{'='*60}\n[재학습 체크] 거래 {self.total_trades}번\n{'='*60}")
        recent = self.log_manager.load_recent_trades(n=self.retrain_check_interval)
        if len(recent) < self.retrain_check_interval or 'result' not in recent.columns:
            print("  데이터 부족/결과 없음")
            return
        wins = int((recent['result'] == 1).sum())
        wr = wins / len(recent)
        print(f"  최근 {self.retrain_check_interval}번 승률: {wr:.1%} ({wins}/{len(recent)})")
        if wr < self.retrain_threshold:
            print(f"  ⚠️ 승률 {wr:.1%} < {self.retrain_threshold:.1%} → 🔄 재학습 필요")
            self.needs_retrain = True
            self.analyze_loss_patterns()
        else:
            print("  ✓ 승률 양호")
            self.needs_retrain = False

    # =============================
    # 데이터 수집
    # =============================
    def fetch_latest_klines(self, limit: int = 500) -> pd.DataFrame:
        try:
            klines = self.client.get_klines(
                symbol=self.symbol,
                interval=Client.KLINE_INTERVAL_1MINUTE,
                limit=limit
            )
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df.dropna()
        except BinanceAPIException as e:
            print(f"⚠️ Binance API 에러: {e}")
            return pd.DataFrame()

    def update_price_buffer(self, retries: int = 2, sleep_sec: float = 1.0) -> bool:
        """1분봉 버퍼 업데이트(약한 재시도)"""
        for attempt in range(retries + 1):
            df_new = self.fetch_latest_klines(limit=10)
            if not df_new.empty:
                for _, row in df_new.iterrows():
                    self.price_buffer.append(row.to_dict())
                # 최신 타임스탬프 기준 중복 제거
                seen = set(); unique = []
                for item in reversed(self.price_buffer):
                    ts = item['timestamp']
                    if ts not in seen:
                        seen.add(ts); unique.append(item)
                self.price_buffer = list(reversed(unique))[-self.buffer_size:]
                return True
            if attempt < retries:
                time.sleep(sleep_sec)
        return False

    def get_buffer_as_df(self) -> pd.DataFrame:
        if not self.price_buffer:
            return pd.DataFrame()
        df = pd.DataFrame(self.price_buffer).sort_values('timestamp').reset_index(drop=True)
        return df

    # =============================
    # 30분봉 체크/예측
    # =============================
    def is_30m_bar_complete(self) -> bool:
        now = datetime.now(timezone.utc)
        return (now.minute in [0, 30]) and (now.second < 10)

    def predict_next_30m(self) -> Optional[Tuple[float, int]]:
        df_1m = self.get_buffer_as_df()
        if len(df_1m) < 100:
            print(f"  ⚠️ 데이터 부족: {len(df_1m)}개")
            return None
        try:
            features = self.feature_engineer.create_feature_pool(df_1m, lookback_bars=100)
        except Exception as e:
            print(f"  ⚠️ 피처 생성 실패: {e}")
            return None
        if features.empty:
            print("  ⚠️ 피처 없음")
            return None
        X_latest = features.tail(1).copy()
        regime = int(X_latest['regime'].iloc[0]) if 'regime' in X_latest.columns else None
        try:
            p_up = self.model_trainer.predict(
                X_latest, regime=regime, use_regime_model=True
            )[0]
        except Exception as e:
            print(f"  ⚠️ 예측 실패: {e}")
            return None
        return p_up, regime

    # =============================
    # 동적 필터
    # =============================
    def check_liquidity_filter(self) -> bool:
        n = self.config.LIQUIDITY_FILTER_WINDOW
        recent = self.log_manager.load_recent_trades(n=n)
        if len(recent) < n or 'result' not in recent.columns:
            return True
        wr = (recent['result'] == 1).mean()
        thr = self.filter_state['liquidity_threshold']
        if wr < thr:
            print(f"  ❌ 유동성 필터: 승률 {wr:.1%} < {thr:.1%}")
            return False
        return True

    def check_consecutive_losses(self) -> bool:
        m = self.filter_state['max_consecutive_losses']
        recent = self.log_manager.load_recent_trades(n=m)
        if len(recent) < m or 'result' not in recent.columns:
            return True
        if (recent['result'] == 0).all():
            print(f"  ❌ 연속 패배: {m}연속 손실")
            return False
        return True

    def check_regime_filter(self, regime: Optional[int]) -> bool:
        if regime is None:
            return True
        key = f'regime_{int(regime)}'
        if key in self.filter_state['loss_patterns']:
            loss_rate = self.filter_state['loss_patterns'][key]['loss_rate']
            if loss_rate > 0.7:
                name = {1: "UP", -1: "DOWN", 0: "FLAT"}.get(regime, f"REGIME-{regime}")
                print(f"  ❌ 레짐 필터: {name} 손실률 {loss_rate:.1%}")
                return False
        return True

    def check_probability_filter(self, p_up: float) -> bool:
        p = abs(p_up)
        if 0.5 <= p < 0.55: label = '0.50-0.55'
        elif 0.55 <= p < 0.6: label = '0.55-0.60'
        elif 0.6 <= p < 0.65: label = '0.60-0.65'
        elif 0.65 <= p < 0.7: label = '0.65-0.70'
        else: label = '0.70+'
        key = f'p_bin_{label}'
        if key in self.filter_state['loss_patterns']:
            loss_rate = self.filter_state['loss_patterns'][key]['loss_rate']
            if loss_rate > 0.6:
                print(f"  ❌ 확률 필터: {label} 손실률 {loss_rate:.1%}")
                return False
        return True

    # =============================
    # 히스테리시스 / Δp / TTL / 포지션
    # =============================
    def check_hysteresis(self, p_now: float, direction_now: int) -> bool:
        cut_on = self.config.CUT_ON
        cut_off = self.filter_state['cutoff_dynamic']
        if self.p_prev is None:
            self.p_prev = p_now
            if p_now >= cut_on:
                self.direction_prev = 1; return True
            if p_now <= (1 - cut_on):
                self.direction_prev = 0; return True
            self.direction_prev = None; return False

        if self.direction_prev == 1:
            self.p_prev = p_now
            if p_now >= cut_off: return True
            self.direction_prev = None; return False

        if self.direction_prev == 0:
            self.p_prev = p_now
            if p_now <= (1 - cut_off): return True
            self.direction_prev = None; return False

        self.p_prev = p_now
        if p_now >= cut_on:
            self.direction_prev = 1; return True
        if p_now <= (1 - cut_on):
            self.direction_prev = 0; return True
        return False

    def check_ttl(self) -> bool:
        if self.signal_start_time is None:
            return False
        ttl_minutes = self.config.SIGNAL_TTL_MINUTES
        elapsed = (datetime.now(timezone.utc) - self.signal_start_time).total_seconds() / 60.0
        if elapsed > ttl_minutes:
            print(f"  ⏰ TTL 만료: {elapsed:.1f}분 경과")
            self.signal_start_time = None
            self.signal_direction = None
            return False
        return True

    def start_signal(self, direction: int):
        self.signal_start_time = datetime.now(timezone.utc)
        self.signal_direction = direction

    def check_delta_p(self, p_now: float) -> bool:
        if self.p_prev is None:
            return True
        delta_p = abs(p_now - self.p_prev)
        if delta_p > self.config.MAX_DELTA_P:
            print(f"  ⚠️ Δp 초과: {delta_p:.3f} > {self.config.MAX_DELTA_P:.3f}")
            return False
        return True

    def can_enter_new_position(self) -> bool:
        if len(self.active_positions) >= self.config.MAX_POSITIONS:
            print(f"  ❌ 최대 포지션: {len(self.active_positions)}/{self.config.MAX_POSITIONS}")
            return False
        last_time = self.last_trade_time.get(self.symbol)
        if last_time:
            elapsed = (datetime.now(timezone.utc) - last_time).total_seconds() / 60.0
            if elapsed < self.config.REFRACTORY_WINDOW_MINUTES:
                print(f"  ⏱️ 리프랙토리: {elapsed:.1f}분 < {self.config.REFRACTORY_WINDOW_MINUTES}분")
                return False
        return True

    def add_position(self, position: Dict):
        self.active_positions.append(position)
        self.last_trade_time[self.symbol] = datetime.now(timezone.utc)

    def remove_position(self, trade_id: str):
        self.active_positions = [p for p in self.active_positions if p['trade_id'] != trade_id]

    # =============================
    # 시그널 로깅/청산
    # =============================
    def log_trade_signal(self, direction: int, p_up: float, regime: Optional[int] = None) -> bool:
        now = datetime.now(timezone.utc)
        df_1m = self.get_buffer_as_df()
        if df_1m.empty:
            print("  ⚠️ 가격 데이터 없음")
            return False

        entry_price = float(df_1m.iloc[-1]['close'])
        trade_id = f"{self.symbol}_{now.strftime('%Y%m%d_%H%M%S')}"

        bar30_start = self._current_30m_slot(now)
        bar30_end = bar30_start + timedelta(minutes=30)
        expiry_time = bar30_end + timedelta(minutes=30)

        # 켈리 추천 베팅금액
        stake = self._kelly_stake(p_up)

        position = {
            'trade_id': trade_id,
            'symbol': self.symbol,
            'direction': direction,
            'entry_price': entry_price,
            'entry_time': now,
            'expiry_time': expiry_time,
            'p_up': p_up,
            'regime': regime,
            'bar30_start': bar30_start,
            'bar30_end': bar30_end,
            'stake_recommended': stake,
            'model_version': self.model_meta.get("version") if self.model_meta else None,
        }
        self.add_position(position)

        # ✅ 피처에서 추가 정보 추출
        features_dict = {}
        try:
            feats = self.feature_engineer.create_feature_pool(df_1m, lookback_bars=100)
            if not feats.empty:
                latest = feats.tail(1).iloc[0]
                features_dict = {
                    'regime_score': latest.get('regime_score', 0.0),
                    'adx_14': latest.get('adx_14', 0.0),
                    'di_plus_14': latest.get('di_plus_14', 0.0),
                    'di_minus_14': latest.get('di_minus_14', 0.0),
                }
                
                # 피처 로그도 저장
                self.log_manager.log_feature(
                    bar30_start=bar30_start,
                    bar30_end=bar30_end,
                    pred_ts=now,
                    entry_ts=now,
                    label_ts=expiry_time,
                    m1_index_entry=self._minute_index(now),
                    m1_index_label=self._minute_index(expiry_time),
                    cut_on=self.config.CUT_ON,
                    cut_off=self.filter_state['cutoff_dynamic'],
                    p_prev=self.p_prev,
                    p_now=p_up,
                    p_cal=p_up,
                    dp=abs(p_up - (self.p_prev or 0.5)),
                    dmin=(now - bar30_end).total_seconds() / 60.0,
                    regime=regime if regime is not None else 0,
                    vol_ratio=latest.get('volume_ratio', 1.0),
                    spread_bps=0.0,
                    vwap_gap_bps=0.0,
                    filters_passed="all",
                    signal_id=trade_id
                )
        except Exception as e:
            print(f"  ⚠️ 피처 추출 실패: {e}")

        # ✅ 간소화 로깅 호출
        self.log_manager.log_trade_entry_simple(
            trade_id=trade_id,
            direction=direction,
            entry_price=entry_price,
            entry_time=now,
            expiry_time=expiry_time,
            p_up=p_up,
            regime=regime,
            bar30_start=bar30_start,
            bar30_end=bar30_end,
            stake_recommended=stake,
            model_version=position['model_version'],
            features_dict=features_dict
        )

        direction_str = "UP 🟢" if direction == 1 else "DOWN 🔴"
        print(f"\n✅ 신호 발생: {direction_str}")
        print(f"  ID: {trade_id}")
        print(f"  가격: ${entry_price:,.2f}")
        print(f"  확률: {p_up:.3f}")
        print(f"  추천 베팅금액: ${stake:,.2f}")
        print(f"  모델버전: {position['model_version'] or 'N/A'}")
        print(f"  만기: {expiry_time.strftime('%H:%M:%S')}")

        self.total_trades += 1
        self.check_retrain_trigger()
        return True
    
    @staticmethod
    def _minute_index(dt: datetime) -> int:
        """1분 인덱스 계산"""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() // 60)

    def check_and_close_positions(self):
        now = datetime.now(timezone.utc)
        for position in self.active_positions.copy():
            if now >= position['expiry_time']:
                self.close_position(position)

    def close_position(self, position: Dict):
        df_1m = self.get_buffer_as_df()
        if df_1m.empty:
            print("  ⚠️ 청산 가격 없음")
            return
        exit_price = float(df_1m.iloc[-1]['close'])
        entry_price = position['entry_price']
        direction = position['direction']

        if direction == 1:
            result = 1 if exit_price > entry_price else 0
        else:
            result = 1 if exit_price < entry_price else 0

        profit_loss = 0.80 if result == 1 else -1.00  # 페이오프 가정

        # (선택) bankroll 시뮬레이션 업데이트를 원하면 주석 해제
        # self.bankroll += position.get("stake_recommended", 0) * profit_loss
        # self._save_bankroll()

        self.log_manager.update_trade_result(
            trade_id=position['trade_id'],
            exit_price=exit_price,
            result=result,
            profit_loss=profit_loss
        )
        self.remove_position(position['trade_id'])

        result_str = "승 ✅" if result == 1 else "패 ❌"
        pl_str = f"+{profit_loss:.0%}" if profit_loss > 0 else f"{profit_loss:.0%}"
        print(f"\n🏁 청산: {result_str} ({pl_str})  ID: {position['trade_id']}")
        print(f"  진입: ${entry_price:,.2f} → 청산: ${exit_price:,.2f}")

    # =============================
    # 모니터 스냅샷
    # =============================
    def _emit_monitor_snapshot(self):
        recent = self.log_manager.load_recent_trades(n=200)
        wr = float((recent['result'] == 1).mean()) if 'result' in recent.columns and len(recent) > 0 else None
        # 연패 계산
        cur_streak = 0; max_streak = 0
        if 'result' in recent.columns:
            for r in recent['result']:
                cur_streak = (cur_streak + 1) if r == 0 else 0
                max_streak = max(max_streak, cur_streak)
        snap = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "recent_win_rate": wr,
            "current_losses": cur_streak,
            "max_consecutive_losses": max_streak,
            "cutoff_dynamic": self.filter_state.get("cutoff_dynamic"),
            "liquidity_threshold": self.filter_state.get("liquidity_threshold"),
            "bankroll": self.bankroll,
            "model_version": (self.model_meta.get("version") if self.model_meta else None),
            "total_trades": self.total_trades,
            "active_positions": len(self.active_positions),
        }
        json.dump(snap, open(self.monitor_snapshot_path, "w"), indent=2, default=str)

    # =============================
    # 메인 루프(1분마다)
    # =============================
    def run(self):
        print(f"\n{'='*60}")
        print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] 사이클 #{self.cycle_count}")
        print(f"{'='*60}")
        self.cycle_count += 1

        # 1) 가격 업데이트
        if not self.update_price_buffer():
            print("⚠️ 가격 업데이트 실패")
            return
        print(f"✓ 버퍼: {len(self.price_buffer)}개")

        # 2) 만기 청산
        self.check_and_close_positions()
        print(f"✓ 활성 포지션: {len(self.active_positions)}개")

        # 3) 재학습 상태
        if self.needs_retrain:
            print("🔄 재학습 필요! (main_pipe에서 처리)")

        # 4) 30분봉 완성 + 슬롯 중복 방지
        if not self.is_30m_bar_complete():
            print("⏱️ 30분봉 대기 중...")
            self._emit_monitor_snapshot()
            return
        slot = self._current_30m_slot(datetime.now(timezone.utc))
        if self.last_decision_slot == slot:
            print("⛔ 동일 슬롯 중복 방지")
            self._emit_monitor_snapshot()
            return
        self.last_decision_slot = slot
        print("🔔 30분봉 완성!")

        # 5) 예측
        pred = self.predict_next_30m()
        if pred is None:
            print("⚠️ 예측 실패")
            self._emit_monitor_snapshot()
            return
        p_up, regime = pred
        direction = 1 if p_up > 0.5 else 0
        rname = {1: "UP", -1: "DOWN", 0: "FLAT"}.get(regime, "N/A")
        print(f"📊 예측: p_up={p_up:.3f}, 레짐={rname}")

        # 6) 필터(동적)
        if not self.check_liquidity_filter(): self._emit_monitor_snapshot(); return
        if not self.check_consecutive_losses(): self._emit_monitor_snapshot(); return
        if not self.check_regime_filter(regime): self._emit_monitor_snapshot(); return
        if not self.check_probability_filter(p_up): self._emit_monitor_snapshot(); return

        # 7) Δp
        if not self.check_delta_p(p_up):
            self._emit_monitor_snapshot(); return

        # 8) 히스테리시스(동적 컷오프)
        if not self.check_hysteresis(p_up, direction):
            print("⏸️ 히스테리시스: 신호 없음")
            self._emit_monitor_snapshot(); return

        # 9) 신호 발생 → TTL 시작
        if self.signal_start_time is None:
            self.start_signal(direction)
            print(f"🔔 신호 발생: {'UP' if direction == 1 else 'DOWN'}")

        # 10) TTL 체크
        if not self.check_ttl():
            self._emit_monitor_snapshot(); return

        # 11) 포지션 제한
        if not self.can_enter_new_position():
            self._emit_monitor_snapshot(); return

        # 12) 신호 로그 기록
        self.log_trade_signal(direction, p_up, regime)

        # 13) 모니터 스냅샷
        self._emit_monitor_snapshot()


# =============================
# 테스트 실행
# =============================
if __name__ == "__main__":
    print("="*60)
    print("RealTrader 테스트 (패치 완전판)")
    print("="*60)
    try:
        trader = RealTrader(symbol='BTCUSDT')
        print("\n✓ 초기화 완료")
        print(f"  심볼: {trader.symbol}")
        print(f"  모델: 로드됨")
        print(f"  필터 상태: {trader.filter_state}")
        print(f"  초기 Bankroll: {trader.bankroll}")

        # 단일 사이클
        print("\n단일 사이클 테스트...")
        trader.run()

        # 패배 원인 분석
        print("\n패배 원인 분석 테스트...")
        trader.analyze_loss_patterns()

    except Exception as e:
        print(f"❌ 에러: {e}")
        import traceback; traceback.print_exc()

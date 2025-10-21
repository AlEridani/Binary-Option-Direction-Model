# real_trade.py - 실시간 거래/백테스트 (ADX 레짐 + 캘리브레이션 확률, 10분 칼만기)

import pandas as pd
import numpy as np
import json
import time
import os
from datetime import datetime, timedelta, timezone
import requests
import uuid
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# ================================
# 바이낸스 API (실사용/시뮬 공용)
# ================================
class BinanceAPIClient:
    """바이낸스 API 클라이언트 (시뮬레이션 fallback)"""

    def __init__(self, api_key=None, api_secret=None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.binance.com"

    def get_current_price(self, symbol="BTCUSDT"):
        """현재 가격 조회"""
        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            params = {"symbol": symbol}
            response = requests.get(url, params=params, timeout=3)
            data = response.json()
            return float(data['price'])
        except Exception:
            # API 실패 시 시뮬레이션 값
            return float(np.random.uniform(40000, 45000))

    def get_klines(self, symbol="BTCUSDT", interval="1m", limit=500):
        """캔들스틱 데이터 조회"""
        try:
            url = f"{self.base_url}/api/v3/klines"
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            response = requests.get(url, params=params, timeout=5)
            data = response.json()

            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception:
            return self._generate_simulation_data(limit)

    def _generate_simulation_data(self, limit=500):
        """시뮬레이션용 데이터 생성"""
        timestamps = pd.date_range(end=datetime.now(timezone.utc), periods=limit, freq='1min', tz='UTC')
        prices = np.random.randn(limit).cumsum() + 42000

        data = []
        for i, ts in enumerate(timestamps):
            base_price = prices[i]
            o = base_price + np.random.uniform(-50, 50)
            c = base_price + np.random.uniform(-50, 50)
            h = max(o, c) + np.random.uniform(0, 100)
            l = min(o, c) - np.random.uniform(0, 100)
            v = np.random.uniform(100, 1000)

            data.append({
                'timestamp': ts,
                'open': o,
                'high': h,
                'low': l,
                'close': c,
                'volume': v
            })

        return pd.DataFrame(data)


# ===========================================
# 실시간 트레이더 (적응형 필터 + 칼만기 + ADX 레짐)
# ===========================================
class RealTimeTrader:
    """실시간 거래 실행 클래스"""

    def __init__(self, config, model_trainer, api_client=None):
        self.config = config
        self.model_trainer = model_trainer
        self.api_client = api_client or BinanceAPIClient()

        # 스로틀/디바운스
        self._last_pred_log_time = None
        self._last_skip_log_time = None

        # 거래 상태
        self.is_running = False
        self.active_positions = {}           # trade_id -> position dict
        self.max_positions = 999             # 데이터 수집용 무제한
        self.trade_history = deque(maxlen=self.config.EVALUATION_WINDOW)

        # 재학습 관련
        self.pending_retrain = False
        self.trades_since_last_check = 0

        # 성능 통계
        self.performance_metrics = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'long_trades': 0,
            'long_wins': 0,
            'short_trades': 0,
            'short_wins': 0,
            'current_streak': 0,
            'max_streak': 0,
            'total_profit': 0
        }

        # 적응형 필터
        self.adaptive_filters = self.load_adaptive_filters()

        # 쿨다운/로그 디바운스 상태
        self._cool_seconds = 60
        self.next_entry_after = None
        self.last_attempt_time = None
        self._last_pred_log = {"t": None, "sig": None}
        self._last_signal = {"dir": None, "p": None, "t": None}  # 같은 방향/확률 반복 차단

        self._retrain = {
            "active": False,
            "triggered": False,
            "trigger_time": None,
            "last_progress_print": 0
        }

    # ---------- 적응형 필터 ----------
    def load_adaptive_filters(self):
        """적응형 필터 로드 (model_train.py가 생성)"""
        filter_path = os.path.join(self.config.MODEL_DIR, 'adaptive_filters.json')

        if os.path.exists(filter_path):
            with open(filter_path, 'r', encoding='utf-8') as f:
                filters = json.load(f)
                active_filters = filters.get('active_filters', [])
                if active_filters:
                    print(f"\n{'='*70}")
                    print(f"✓ 적응형 필터 로드: {len(active_filters)}개")
                    print(f"{'='*70}")
                    for i, flt in enumerate(active_filters, 1):
                        print(f"{i}. [{flt['type'].upper()}] {flt['name']}: {flt['condition']}")
                        print(f"   개선: +{flt['improvement']:.1%} | {flt['reason']}")
                    print(f"{'='*70}\n")
                else:
                    print("\n⚠️  적응형 필터 없음 - 재학습 후 자동 생성됩니다.\n")
                return filters

        print("\n⚠️  적응형 필터 파일 없음 - 초기 학습 후 생성됩니다.\n")
        return {'active_filters': [], 'filter_history': []}

    def apply_adaptive_filters(self, features_row):
        """적응형 필터를 현재 피처에 적용"""
        active_filters = self.adaptive_filters.get('active_filters', [])
        if not active_filters:
            return True, []

        blocked_reasons = []
        for fc in active_filters:
            field = fc.get('field')
            if not field or field not in features_row:
                continue
            value = features_row[field]
            if pd.isna(value):
                continue

            op = fc.get('operator')
            if op == '>':
                th = fc.get('threshold')
                if value > th:
                    blocked_reasons.append(f"{fc['name']}: {field}={value:.4f} > {th:.4f}")
            elif op == '<':
                th = fc.get('threshold')
                if value < th:
                    blocked_reasons.append(f"{fc['name']}: {field}={value:.4f} < {th:.4f}")
            elif op == 'in':
                bad_values = fc.get('bad_hours', [])
                if value in bad_values:
                    blocked_reasons.append(f"{fc['name']}: {field}={value} (저승률)")
            elif op == 'extreme':
                lo = fc.get('lower_threshold')
                hi = fc.get('upper_threshold')
                if (lo is not None and value < lo) or (hi is not None and value > hi):
                    blocked_reasons.append(f"{fc['name']}: {field}={value:.4f} (극단값)")
            elif op == 'between':
                lo = fc.get('lower_threshold')
                hi = fc.get('upper_threshold')
                if lo is not None and hi is not None and lo < value < hi:
                    blocked_reasons.append(f"{fc['name']}: {lo:.2f} < {field}={value:.4f} < {hi:.2f}")

        return (len(blocked_reasons) == 0), blocked_reasons

    # ---------- 피처/예측 ----------
    def prepare_features(self, df):
        """실시간 피처 준비"""
        from model_train import FeatureEngineer
        fe = FeatureEngineer()
        features = fe.create_feature_pool(df)
        return features

    def _maybe_log_signal(self, p_up, side, force=False):
        """예측 로그 과다 방지(1분 디바운스). side: 1=LONG, 0=SHORT, None"""
        now = datetime.now(timezone.utc)
        sig = (side, round(float(p_up), 4))
        if not force:
            last = self._last_pred_log
            if last["sig"] == sig and last["t"] and (now - last["t"]).total_seconds() < 60:
                return
        side_txt = "LONG" if side == 1 else ("SHORT" if side == 0 else "NO-TRADE")
        print("\n[신호]  P(UP)={:.6f} → 결정: {}".format(p_up, side_txt))
        self._last_pred_log_time = now
        self._last_pred_log = {"t": now, "sig": sig}

    def make_prediction(self, features, debug=False):
        """
        레짐 기반 진입 결정:
          - features에서 regime 추출
          - model_trainer.predict_proba(features, regime) → 레짐별 모델로 p_up 계산
          - model_trainer.decide_from_proba_regime(p_up, regime) → 1/0/None
          - 적응형 필터 통과 시에만 side 반환
        """
        if not self.model_trainer.models:
            print("❌ 모델이 로드되지 않았습니다.")
            return None, 0.5

        try:
            # 현재 행에서 regime 추출
            regime = int(features['regime'].iloc[-1]) if 'regime' in features.columns else 0
            
            # 레짐별 모델로 확률 예측
            X_cur = features.iloc[[-1]]
            p_up = self.model_trainer.predict_proba(X_cur, regime=regime)
            
            # numpy array면 스칼라로 변환
            if isinstance(p_up, np.ndarray):
                p_up = float(p_up[-1]) if len(p_up) > 0 else 0.5
            else:
                p_up = float(p_up)
            
            if not np.isfinite(p_up):
                return None, 0.5

            # 레짐 기반 진입 결정
            side = self.model_trainer.decide_from_proba_regime(p_up, regime)

            if debug:
                self._maybe_log_signal(p_up, side, force=False)

            # 적응형 필터 체크 (trade 여부만 막음)
            if side is not None:
                ok, reasons = self.apply_adaptive_filters(features.iloc[-1])
                if not ok:
                    if debug:
                        print("  ❌ [필터 차단]")
                        for r in reasons:
                            print(f"     - {r}")
                    return None, p_up

            # 동일 신호 디바운스: 같은 방향 & 거의 같은 확률이 3~5분 내 반복되면 무시
            if side is not None:
                EPS = 1e-4
                now = datetime.now(timezone.utc)
                last = self._last_signal
                if (last["dir"] == side and last["p"] is not None
                    and abs(last["p"] - p_up) < EPS
                    and last["t"] is not None
                    and (now - last["t"]).total_seconds() < 300):
                    return None, p_up

            return side, p_up

        except Exception as e:
            print(f"❌ 예측 오류: {e}")
            import traceback; traceback.print_exc()
            return None, 0.5

    # ---------- 거래/청산/통계 ----------
    def _binary_payout(self, direction, entry_price, exit_price):
        """바이너리 옵션 페이아웃: 방향 적중시 +WIN_RATE, 미적중시 -1 (배팅액 기준)"""
        hit = (exit_price > entry_price) if direction == 1 else (exit_price < entry_price)
        return self.config.WIN_RATE if hit else -1.0

    def execute_trade(self, side, p_up, amount=100):
        """거래 실행 - 레짐 정보 포함"""
        if len(self.active_positions) >= self.max_positions:
            return None

        trade_id = str(uuid.uuid4())[:8]
        entry_time = datetime.now(timezone.utc)
        expiry_time = entry_time + timedelta(minutes=self.config.PREDICTION_WINDOW)
        entry_price = self.api_client.get_current_price()
        
        # ★ 현재 레짐 정보 추출
        try:
            df = self.api_client.get_klines(limit=500)
            features = self.prepare_features(df)
            current_regime = int(features['regime'].iloc[-1]) if 'regime' in features.columns else None
        except Exception as e:
            print(f"⚠️ 레짐 정보 추출 실패: {e}")
            current_regime = None

        info = {
            'trade_id': trade_id,
            'entry_time': entry_time.isoformat(),
            'expiry_time': expiry_time.isoformat(),
            'entry_price': entry_price,
            'direction': int(side),                         # 1=LONG, 0=SHORT
            'p_up': float(p_up),                            # 예측 확률
            'regime': current_regime,                        # ★ 레짐 정보 (0:UP, 1:DOWN, 2:FLAT)
            'amount': amount,
            'status': 'open'
        }
        self.active_positions[trade_id] = info
        self.save_trade_log(info)

        # 쿨다운
        self.next_entry_after = entry_time + timedelta(seconds=self._cool_seconds)

        # 동일 신호 디바운스 메모
        self._last_signal = {"dir": side, "p": p_up, "t": entry_time}

        # 알림 로그
        direction = "롱 (UP)" if side == 1 else "숏 (DOWN)"
        emoji = "🟢⬆️" if side == 1 else "🔴⬇️"
        
        # 레짐 표시
        regime_labels = {0: "UP 트렌드🟢", 1: "DOWN 트렌드🔴", 2: "FLAT 횡보⚪", None: "알 수 없음❓"}
        regime_text = regime_labels.get(current_regime, f"REGIME-{current_regime}")

        print("\n" + "="*70)
        print("🔔" * 35)
        print("="*70)
        print(f"{'💰 거래 진입!':^70}")
        print("="*70)
        print(f"  🆔 거래 ID     : {trade_id}")
        print(f"  ⏰ 진입 시간   : {entry_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"  ⏱️  만기 시간   : {expiry_time.strftime('%H:%M:%S')} UTC ({self.config.PREDICTION_WINDOW}분 후, 칼만기)")
        print(f"  📊 방향        : {direction} {emoji}")
        print(f"  🎯 레짐        : {regime_text}")  # ★ 레짐 정보 표시
        print(f"  📈 P(UP)       : {p_up:.2%}")
        print(f"  💰 진입가      : ${entry_price:,.2f}")
        print(f"  💵 배팅 금액   : ${amount}")
        print(f"  📈 활성 포지션 : {len(self.active_positions)}/{self.max_positions}")
        afc = len(self.adaptive_filters.get('active_filters', []))
        if afc > 0:
            print(f"  🛡️  활성 필터   : {afc}개 통과 ✓")
        print("="*70)
        print("🔔" * 35)
        print("="*70 + "\n")

        try:
            import winsound
            winsound.Beep(1000, 500)
        except Exception:
            pass

        return trade_id

    def check_trade_result(self, trade_id):
        """거래 결과 확인 — ★ 엔트리+10분 '칼만기' 기준으로만 청산"""
        pos = self.active_positions.get(trade_id)
        if not pos:
            return None

        entry_time = datetime.fromisoformat(pos['entry_time'].replace("Z",""))
        expiry_time = datetime.fromisoformat(pos['expiry_time'].replace("Z",""))
        now = datetime.now(timezone.utc)

        if now < expiry_time:
            return None  # 아직 만기 전

        entry_price = pos['entry_price']
        exit_price = self.api_client.get_current_price()

        direction = pos['direction']  # 1=LONG, 0=SHORT
        pnl_unit = self._binary_payout(direction, entry_price, exit_price)
        amount = pos['amount']
        profit = amount * pnl_unit
        result = 1 if profit > 0 else 0

        pos['exit_time'] = now.isoformat()
        pos['exit_price'] = exit_price
        pos['result'] = result
        pos['profit_loss'] = profit
        pos['status'] = 'closed'

        # 성능/로그 업데이트
        self.update_performance(result == 1, profit, direction)
        self.trade_history.append(result)
        self.update_trade_log(trade_id, result, profit)

        actual_dir = "상승" if exit_price > entry_price else "하락"
        result_emoji = "✅ 승리!" if result == 1 else "❌ 패배"
        result_color = "🟢" if result == 1 else "🔴"
        
        # 레짐 정보
        regime_labels = {0: "UP🟢", 1: "DOWN🔴", 2: "FLAT⚪", None: "N/A"}
        regime_text = regime_labels.get(pos.get('regime'), "N/A")

        print("\n" + "="*70)
        print(f"{result_color} 거래 청산: {trade_id}")
        print("="*70)
        print(f"  ⏰ 진입시각    : {entry_time.strftime('%H:%M:%S')} UTC")
        print(f"  ⏱️  만기시각    : {expiry_time.strftime('%H:%M:%S')} UTC  (칼만기)")
        print(f"  ⏳ 청산시각    : {now.strftime('%H:%M:%S')} UTC")
        print(f"  🎯 레짐        : {regime_text}")  # ★ 레짐 정보 표시
        print(f"  📊 예측 방향   : {'롱 (UP)' if direction==1 else '숏 (DOWN)'}")
        print(f"  📈 실제 방향   : {actual_dir}")
        print(f"  💰 진입가      : ${entry_price:,.2f}")
        print(f"  💰 종료가      : ${exit_price:,.2f}")
        print(f"  💵 손익        : ${profit:+,.2f}")
        print(f"  🎯 결과        : {result_emoji}")
        print("="*70 + "\n")

        # 활성 포지션 제거
        del self.active_positions[trade_id]

        # 재학습 평가 트리거
        if self.trades_since_last_check >= self.config.EVALUATION_WINDOW:
            if self.check_retrain_needed():
                self.pending_retrain = True
                print(f"\n⚠️  재학습 모드: 신규 진입 중단 (활성 {len(self.active_positions)}개 대기)")

        return result

    def update_performance(self, is_win, profit, direction):
        """성능 통계 업데이트"""
        self.performance_metrics['total_trades'] += 1
        self.trades_since_last_check += 1

        if direction == 1:
            self.performance_metrics['long_trades'] += 1
            if is_win:
                self.performance_metrics['long_wins'] += 1
        else:
            self.performance_metrics['short_trades'] += 1
            if is_win:
                self.performance_metrics['short_wins'] += 1

        if is_win:
            self.performance_metrics['wins'] += 1
            self.performance_metrics['current_streak'] += 1
            self.performance_metrics['max_streak'] = max(
                self.performance_metrics['max_streak'],
                self.performance_metrics['current_streak']
            )
        else:
            self.performance_metrics['losses'] += 1
            self.performance_metrics['current_streak'] = 0

        self.performance_metrics['total_profit'] += profit

        if self.performance_metrics['total_trades'] > 0:
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['wins'] / self.performance_metrics['total_trades']
            )
        self.performance_metrics['long_win_rate'] = (
            self.performance_metrics['long_wins'] / self.performance_metrics['long_trades']
            if self.performance_metrics['long_trades'] > 0 else 0
        )
        self.performance_metrics['short_win_rate'] = (
            self.performance_metrics['short_wins'] / self.performance_metrics['short_trades']
            if self.performance_metrics['short_trades'] > 0 else 0
        )

    def check_retrain_needed(self):
        """재학습 필요 여부 — 윌슨 하한 기반"""
        if self.trades_since_last_check < self.config.EVALUATION_WINDOW:
            return False

        n = len(self.trade_history)
        if n >= self.config.EVALUATION_WINDOW:
            wins = sum(self.trade_history)

            def _wilson(w, nn, z=1.96):
                if nn <= 0: return 0.0
                p = w / nn
                denom = 1 + (z * z) / nn
                center = p + (z * z) / (2 * nn)
                margin = z * ((p * (1 - p) + (z * z) / (4 * nn)) / nn) ** 0.5
                return (center - margin) / denom

            L = _wilson(wins, n, z=1.96)
            thresh = self.config.RETRAIN_THRESHOLD

            print(f"\n{'='*60}")
            print(f"재학습 평가(최근 {n}건) — 윌슨 하한 기반")
            print(f"{'='*60}")
            print(f"승: {wins} / 총: {n}  | 평균승률: {wins/n:.2%} | 윌슨 하한: {L:.2%} | 임계: {thresh:.2%}")

            if L < thresh:
                print(f"✗ 재학습 필요")
                return True
            else:
                print(f"✓ 거래 계속")
                self.trades_since_last_check = 0
                return False

        return False

    # ---------- 로깅/저장 ----------
    def save_trade_log(self, trade_info):
        """거래 로그 저장 (append) — regime 컬럼 포함"""
        try:
            log_path = os.path.join(self.config.TRADE_LOG_DIR, 'trades.csv')
            os.makedirs(self.config.TRADE_LOG_DIR, exist_ok=True)
            
            df_new = pd.DataFrame([trade_info])
            write_header = not os.path.exists(log_path)
            df_new.to_csv(log_path, mode='a', header=write_header, index=False, encoding='utf-8-sig')
            
        except Exception as e:
            print(f"  ❌ 거래 로그 저장 실패: {e}")
            import traceback
            traceback.print_exc()

    def update_trade_log(self, trade_id, result, profit_loss):
        """거래 결과 업데이트 (exit_time/exit_price/result/profit_loss 등)"""
        from data_merge import DataMerger
        merger = DataMerger(self.config)
        merger.update_trade_result(trade_id, result, profit_loss)

    def save_feature_log(self, features, trade_id):
        """피처 로그 저장 (append)"""
        current_features = features.iloc[[-1]].copy()
        current_features['trade_id'] = trade_id
        current_features['timestamp'] = datetime.now(timezone.utc)

        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        log_path = os.path.join(self.config.FEATURE_LOG_DIR, f'features_{today}.csv')

        os.makedirs(self.config.FEATURE_LOG_DIR, exist_ok=True)
        write_header = not os.path.exists(log_path)
        current_features.to_csv(log_path, mode='a', header=write_header, index=False, encoding='utf-8-sig')

    def print_trading_statistics(self):
        """1분마다 거래 통계"""
        metrics = self.performance_metrics
        active_longs = sum(1 for p in self.active_positions.values() if p['direction'] == 1)
        active_shorts = len(self.active_positions) - active_longs

        print("\n" + "┏" + "━"*68 + "┓")
        print(f"┃{'📊 거래 통계':^70}┃")
        print(f"┃{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC{'':^40}┃")

        if self.pending_retrain:
            print("┃" + " "*68 + "┃")
            print(f"┃{'⚠️  재학습 대기 중 - 신규 진입 중단':^70}┃")

        print("┣" + "━"*68 + "┫")
        print(f"┃  활성 포지션: {len(self.active_positions):2d}/{self.max_positions}  " + " "*10 +
              f"🟢 롱: {active_longs:2d}   🔴 숏: {active_shorts:2d}" + " "*20 + "┃")

        afc = len(self.adaptive_filters.get('active_filters', []))
        print(f"┃  적응형 필터: {afc:2d}개 활성" + " "*46 + "┃")

        progress = self.trades_since_last_check
        bar_len = 50
        filled = progress * bar_len // max(1, self.config.EVALUATION_WINDOW)
        progress_bar = "█" * filled + "░" * (bar_len - filled)
        print(f"┃  재학습 체크: [{progress_bar}] {progress}/{self.config.EVALUATION_WINDOW}" + " "*5 + "┃")

        print("┣" + "━"*68 + "┫")

        if metrics['total_trades'] > 0:
            win_rate = metrics.get('win_rate', 0)
            win_rate_bar = "█" * int(win_rate * 50) + "░" * (50 - int(win_rate * 50))

            if win_rate >= 0.60:
                rate_emoji = "🔥"
            elif win_rate >= 0.56:
                rate_emoji = "✅"
            elif win_rate >= 0.50:
                rate_emoji = "⚠️"
            else:
                rate_emoji = "❌"

            print(f"┃  전체: {metrics['total_trades']:3d}회  승: {metrics['wins']:3d}  패: {metrics['losses']:3d}  " +
                  f"승률: {win_rate:.1%} {rate_emoji}" + " "*15 + "┃")
            print(f"┃  [{win_rate_bar}]" + " "*10 + "┃")

            if metrics['long_trades'] > 0:
                long_wr = metrics.get('long_win_rate', 0)
                print(f"┃  🟢 롱:  {metrics['long_trades']:3d}회  승: {metrics['long_wins']:3d}  " +
                      f"패: {metrics['long_trades'] - metrics['long_wins']:3d}  승률: {long_wr:.1%}" +
                      " "*15 + "┃")

            if metrics['short_trades'] > 0:
                short_wr = metrics.get('short_win_rate', 0)
                print(f"┃  🔴 숏:  {metrics['short_trades']:3d}회  승: {metrics['short_wins']:3d}  " +
                      f"패: {metrics['short_trades'] - metrics['short_wins']:3d}  승률: {short_wr:.1%}" +
                      " "*15 + "┃")

            print("┣" + "━"*68 + "┫")

            avg_profit = metrics['total_profit'] / max(1, metrics['total_trades'])
            profit_emoji = "💰" if metrics['total_profit'] > 0 else "💸"
            print(f"┃  {profit_emoji} 총 손익: ${metrics['total_profit']:+,.2f}   " +
                  f"평균: ${avg_profit:+,.2f}" + " "*25 + "┃")
        else:
            print(f"┃{'아직 완료된 거래가 없습니다.':^70}┃")

        print("┗" + "━"*68 + "┛\n")

    def print_performance_summary(self):
        """최종 성능 요약"""
        self.print_trading_statistics()

    # ---------- 재학습 ----------
    def execute_retrain_process(self):
        """재학습 프로세스 (자동 실행 + 플래그 기록)"""
        print(f"\n{'='*60}")
        print("재학습 및 필터 업데이트 프로세스 시작")
        print(f"{'='*60}")

        # ★ 트리거 기록 (로깅/추적용)
        self.trigger_retrain()  # 플래그 파일 생성 (언제 재학습했는지 기록)

        try:
            # ★ 1. 데이터 병합
            print("\n[1/4] 데이터 병합 중...")
            from data_merge import DataMerger
            merger = DataMerger(self.config)
            merged_data = merger.merge_all_data()
            
            if merged_data is None or merged_data.empty:
                print("❌ 병합 데이터 없음 - 재학습 중단")
                self.pending_retrain = False
                return
            
            merged_data = merged_data.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            
            if len(merged_data) < 1000:
                print(f"❌ 데이터 부족 ({len(merged_data)}건) - 재학습 중단")
                self.pending_retrain = False
                return
            
            print(f"✓ 병합 완료: {len(merged_data):,}건")
            
            # ★ 2. 모델 재학습
            print("\n[2/4] 모델 재학습 중...")
            from model_train import ModelOptimizer
            
            optimizer = ModelOptimizer(self.config)
            metrics = optimizer.retrain_model(merged_data)
            
            print("✓ 재학습 완료!")
            
            # ★ 3. 적응형 필터 생성
            print("\n[3/4] 적응형 필터 생성 중...")
            
            trades_log = merger.load_trade_logs()
            
            if not trades_log.empty and 'result' in trades_log.columns:
                if 'regime' in trades_log.columns:
                    trades_log = trades_log[trades_log['regime'].notna()]
                
                if len(trades_log) >= 50:
                    features_log = merger.load_feature_logs()
                    
                    if not features_log.empty and 'trade_id' in features_log.columns:
                        trade_features = pd.merge(
                            trades_log,
                            features_log,
                            on='trade_id',
                            how='inner',
                            suffixes=('', '_feat')
                        )
                        
                        if len(trade_features) >= 50:
                            print(f"  분석 대상: {len(trade_features)}건")
                            patterns = optimizer.analyze_failures(trade_features)
                            
                            if patterns:
                                print(f"✓ {len(patterns)}개 패턴 발견")
                            else:
                                print("  새로운 패턴 없음")
                        else:
                            print("  ⚠️ 거래-피처 병합 데이터 부족")
                    else:
                        print("  ⚠️ 피처 로그 없음")
                else:
                    print(f"  ⚠️ 거래 데이터 부족 ({len(trades_log)}건)")
            else:
                print("  ⚠️ 거래 로그 없음")
            
            # ★ 4. 모델 및 필터 리로드
            print("\n[4/4] 모델 및 필터 리로드 중...")
            self.model_trainer.load_model()
            self.adaptive_filters = self.load_adaptive_filters()
            
            # ★ 완료 플래그 생성 (추적용)
            flag_path = os.path.join(self.config.BASE_DIR, '.retrain_complete')
            with open(flag_path, 'w', encoding='utf-8') as f:
                f.write(datetime.now(timezone.utc).isoformat())
            
            self.trades_since_last_check = 0
            self.pending_retrain = False
            
            print("\n" + "="*60)
            print("✓ 재학습 완료 - 거래 재개")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"\n❌ 재학습 실패: {e}")
            import traceback
            traceback.print_exc()
            
            self.pending_retrain = False
            print("\n⚠️ 기존 모델로 거래 계속")

    # ---------- 메인 루프 ----------
    def run_live_trading(self, duration_hours=99999, trade_interval_minutes=1):
        """실시간 거래 실행"""
        print(f"\n실시간 거래 시작 (레짐 기반 + 캘리, ADX 레짐, 적응형 필터)")
        print(f"- 실행 기간: {'무제한 (Ctrl+C로 종료)' if duration_hours >= 99999 else f'{duration_hours}시간'}")
        print(f"- 진입 간격: {trade_interval_minutes}분 (만기 {self.config.PREDICTION_WINDOW}분, 칼만기)")
        print(f"- 최대 포지션: {self.max_positions} (데이터 수집용)")
        print(f"- 재학습 평가: {self.config.EVALUATION_WINDOW}회마다")
        afc = len(self.adaptive_filters.get('active_filters', []))
        print(f"- 적응형 필터: {afc}개 {'활성' if afc > 0 else '대기중 (재학습 후 생성)'}")
        print("="*60 + "\n")

        self.is_running = True
        end_time = datetime.now(timezone.utc) + timedelta(hours=duration_hours)
        self._cool_seconds = int(trade_interval_minutes * 60)


        if not self.model_trainer.models:
            print("모델 로딩 중...")
            success = self.model_trainer.load_model()
            if not success:
                print("모델을 찾을 수 없습니다.")
                return

        last_stat_time = datetime.now(timezone.utc)
        self.last_attempt_time = None
        self.next_entry_after = None

        try:
            while datetime.now(timezone.utc) < end_time and self.is_running:
                now = datetime.now(timezone.utc)

                # 1) 1분마다 통계 출력
                if (now - last_stat_time).total_seconds() >= 60:
                    self.print_trading_statistics()
                    last_stat_time = now

                # 2) 열린 포지션 결과 체크 (만기 도달 시 칼청산)
                for trade_id in list(self.active_positions.keys()):
                    self.check_trade_result(trade_id)

                # 3) 재학습 대기: 모든 포지션 종료 시 재학습 수행
                if self.pending_retrain:
                    if len(self.active_positions) == 0:
                        print(f"\n모든 포지션 종료 → 재학습 진행")
                        self.execute_retrain_process()
                    time.sleep(5)
                    continue

                # 4) 신규 진입 시도 (쿨다운 + 최대 포지션 체크)
                if len(self.active_positions) < self.max_positions:
                    cooled = (self.next_entry_after is None) or (now >= self.next_entry_after)
                    throttled = (self.last_attempt_time is not None
                                 and (now - self.last_attempt_time).total_seconds() < 5)

                    if cooled and not throttled:
                        self.last_attempt_time = now
                        # 데이터/피처/예측
                        try:
                            df = self.api_client.get_klines(limit=500)
                            features = self.prepare_features(df)
                            side, p_up = self.make_prediction(features, debug=True)
                        except Exception as e:
                            print(f"  [오류] 예측 준비/수행 실패: {e}")
                            time.sleep(1)
                            continue

                        if side is not None:
                            trade_id = self.execute_trade(side, p_up)
                            if trade_id:
                                self.save_feature_log(features, trade_id)
                                # 다음 진입 허용 시각은 execute_trade에서 설정됨
                                self.last_attempt_time = None
                        else:
                            # 미진입 안내(1분 스로틀)
                            now2 = datetime.now(timezone.utc)
                            if (self._last_skip_log_time is None) or ((now2 - self._last_skip_log_time).total_seconds() >= 60):
                                print(f"  [미진입] 레짐/필터 미통과 (활성 {len(self.active_positions)}/{self.max_positions})")
                                self._last_skip_log_time = now2
                    else:
                        # 쿨다운 남은 시간 매분 안내
                        if not cooled and now.second == 0:
                            remain = int((self.next_entry_after - now).total_seconds())
                            if remain > 0:
                                print(f"  [대기] 다음 진입까지 {remain}초 "
                                      f"(활성: {len(self.active_positions)}/{self.max_positions})")

                time.sleep(1)

        except KeyboardInterrupt:
            print("\n거래 중단...")
        finally:
            self.is_running = False

            # 남은 포지션 정리(만기까지 대기)
            if self.active_positions:
                print(f"\n남은 포지션 {len(self.active_positions)}개 대기...")
                while self.active_positions:
                    for tid in list(self.active_positions.keys()):
                        self.check_trade_result(tid)
                    time.sleep(5)

            self.print_performance_summary()

    # ---------- 백테스트 ----------
    def backtest(self, historical_data, start_date=None, end_date=None):
        """레짐 기반 백테스트"""
        print("\n백테스팅 시작...")

        if start_date:
            historical_data = historical_data[historical_data['timestamp'] >= start_date]
        if end_date:
            historical_data = historical_data[historical_data['timestamp'] <= end_date]

        from model_train import FeatureEngineer
        fe = FeatureEngineer()
        
        # 피처 생성
        features = fe.create_feature_pool(historical_data)
        target = fe.create_target(historical_data, window=self.config.PREDICTION_WINDOW)

        # ★ 인덱스 정렬 후 필터링
        features = features.reset_index(drop=True)
        target = target.reset_index(drop=True)
        
        # 길이 맞추기
        min_len = min(len(features), len(target))
        features = features.iloc[:min_len]
        target = target.iloc[:min_len]
        
        # 유효한 데이터만 필터링
        valid_idx = target.notna()
        valid_indices = valid_idx[valid_idx].index.tolist()
        
        if len(valid_indices) == 0:
            print("❌ 유효한 타겟 데이터가 없습니다.")
            return pd.DataFrame()
        
        features = features.loc[valid_indices].reset_index(drop=True)
        target = target.loc[valid_indices].reset_index(drop=True)
        
        print(f"백테스트 데이터: {len(features):,}건")

        trades = []

        for i in range(len(features) - self.config.PREDICTION_WINDOW):
            try:
                X_current = features.iloc[[i]]
                
                # 예측
                p_up_arr = np.ravel(self.model_trainer.predict_proba(X_current))
                if len(p_up_arr) == 0 or not np.isfinite(p_up_arr[-1]):
                    continue
                p_up = float(p_up_arr[-1])

                # 레짐 추출
                regime = int(features['regime'].iloc[i]) if 'regime' in features.columns else 0
                
                # 진입 결정
                side = self.model_trainer.decide_from_proba_regime(p_up, regime)
                if side is None:
                    continue

                # 실제 결과
                actual = int(target.iloc[i])

                # 타임스탬프 추출
                if 'timestamp' in features.columns:
                    ts = features['timestamp'].iloc[i]
                elif i < len(historical_data):
                    ts = historical_data['timestamp'].iloc[i]
                else:
                    ts = pd.Timestamp.now()

                trades.append({
                    'timestamp': ts,
                    'p_up': p_up,
                    'regime': regime,
                    'decision': side,
                    'actual': actual,
                    'correct': int(side == actual)
                })
                
            except Exception as e:
                # 개별 거래 실패는 스킵
                continue

        if not trades:
            print("❌ 백테스트 거래가 생성되지 않았습니다.")
            return pd.DataFrame()

        trades_df = pd.DataFrame(trades)

        total_trades = len(trades_df)
        correct_trades = int(trades_df['correct'].sum()) if total_trades > 0 else 0
        win_rate = correct_trades / total_trades if total_trades > 0 else 0.0

        wins = correct_trades
        losses = total_trades - wins
        profit = (wins * 100 * self.config.WIN_RATE) - (losses * 100)

        print(f"\n백테스팅 결과 (레짐 기반):")
        if total_trades > 0:
            print(f"- 기간: {trades_df['timestamp'].min()} ~ {trades_df['timestamp'].max()}")
        print(f"- 총 거래: {total_trades}")
        print(f"- 승: {wins} / 패: {losses}")
        print(f"- 승률: {win_rate:.2%}")
        print(f"- 총 손익(가정): ${profit:.2f}")
        print(f"- 평균 손익/거래: ${profit/total_trades if total_trades > 0 else 0:.2f}")
        
        # 레짐별 성과
        if 'regime' in trades_df.columns and total_trades > 0:
            print(f"\n[레짐별 성과]")
            regime_labels = {0: "UP", 1: "DOWN", 2: "FLAT"}
            for regime_val in sorted(trades_df['regime'].unique()):
                regime_trades = trades_df[trades_df['regime'] == regime_val]
                regime_wins = regime_trades['correct'].sum()
                regime_total = len(regime_trades)
                regime_wr = regime_wins / regime_total if regime_total > 0 else 0
                regime_name = regime_labels.get(regime_val, f"REGIME-{regime_val}")
                print(f"  {regime_name}: {regime_total}건, 승률 {regime_wr:.2%}")

        return trades_df


# =================================
# 모니터 + 레짐별 분석
# =================================
class TradingMonitor:
    """거래 모니터링 클래스 (레짐별 분석 포함)"""

    def __init__(self, config):
        self.config = config

    def analyze_recent_trades(self, days=7):
        """최근 거래 분석 (레짐별 포함)"""
        from data_merge import DataMerger
        merger = DataMerger(self.config)
        trades = merger.load_trade_logs()

        if trades.empty:
            print("거래 기록이 없습니다.")
            return None

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        if 'entry_time' in trades.columns:
            trades['entry_time'] = pd.to_datetime(trades['entry_time'], utc=True, errors='coerce')
            recent_trades = trades[trades['entry_time'] >= cutoff]
        else:
            recent_trades = trades

        if recent_trades.empty:
            print(f"최근 {days}일간 거래 기록이 없습니다.")
            return None

        # result, profit_loss를 숫자로 변환
        if 'result' in recent_trades.columns:
            recent_trades['result'] = pd.to_numeric(recent_trades['result'], errors='coerce')
        
        if 'profit_loss' in recent_trades.columns:
            recent_trades['profit_loss'] = pd.to_numeric(recent_trades['profit_loss'], errors='coerce')

        stats = {
            'total_trades': len(recent_trades),
            'wins': int((recent_trades['result'] == 1).sum()) if 'result' in recent_trades.columns else 0,
            'losses': int((recent_trades['result'] == 0).sum()) if 'result' in recent_trades.columns else 0,
            'total_profit': float(recent_trades['profit_loss'].sum()) if 'profit_loss' in recent_trades.columns else 0.0
        }

        if stats['total_trades'] > 0:
            stats['win_rate'] = stats['wins'] / stats['total_trades']
            stats['avg_profit'] = stats['total_profit'] / stats['total_trades']

        # ★ 레짐별 성과 분석
        if 'regime' in recent_trades.columns and 'result' in recent_trades.columns:
            try:
                # regime 컬럼을 숫자로 변환
                recent_trades['regime'] = pd.to_numeric(recent_trades['regime'], errors='coerce')
                
                # 레짐 정보가 있는 데이터만 필터링
                trades_with_regime = recent_trades[recent_trades['regime'].notna()]
                
                if len(trades_with_regime) > 0:
                    regime_stats = trades_with_regime.groupby('regime').agg({
                        'result': ['count', 'mean', 'sum'],
                        'profit_loss': 'sum'
                    }).round(3)
                    
                    regime_stats.columns = ['_'.join(col).strip() for col in regime_stats.columns.values]
                    stats['regime_performance'] = regime_stats
                    
                    # 레짐별 롱/숏 성과
                    if 'direction' in trades_with_regime.columns:
                        trades_with_regime['direction'] = pd.to_numeric(trades_with_regime['direction'], errors='coerce')
                        regime_direction_stats = trades_with_regime.groupby(['regime', 'direction']).agg({
                            'result': ['count', 'mean'],
                            'profit_loss': 'sum'
                        }).round(3)
                        stats['regime_direction_performance'] = regime_direction_stats
                        
            except Exception as e:
                print(f"⚠️ 레짐별 통계 생성 실패: {e}")
                stats['regime_performance'] = None

        # 시간대별 성과
        if 'entry_time' in recent_trades.columns and 'result' in recent_trades.columns:
            try:
                recent_trades['hour'] = recent_trades['entry_time'].dt.hour
                valid_results = recent_trades[recent_trades['result'].notna()]
                
                if len(valid_results) > 0:
                    hourly_stats = valid_results.groupby('hour').agg({
                        'result': ['count', 'mean']
                    }).round(3)
                    stats['hourly_performance'] = hourly_stats
            except Exception as e:
                print(f"⚠️ 시간대별 통계 생성 실패: {e}")
                stats['hourly_performance'] = None

        return stats

    def generate_report(self):
        """종합 리포트 생성 (레짐 분석 포함)"""
        print("\n" + "="*60)
        print("거래 시스템 종합 리포트")
        print("="*60)

        week_stats = self.analyze_recent_trades(7)
        if week_stats:
            print("\n[최근 7일 성과]")
            print(f"총 거래: {week_stats['total_trades']}")
            print(f"승/패: {week_stats['wins']}/{week_stats['losses']}")
            print(f"승률: {week_stats.get('win_rate', 0):.2%}")
            print(f"총 손익: ${week_stats['total_profit']:.2f}")
            print(f"평균 손익: ${week_stats.get('avg_profit', 0):.2f}")
            
            # ★ 레짐별 성과
            if week_stats.get('regime_performance') is not None:
                print("\n[레짐별 성과]")
                regime_labels = {0: "UP 트렌드", 1: "DOWN 트렌드", 2: "FLAT 횡보"}
                rp = week_stats['regime_performance']
                for regime_idx in rp.index:
                    regime_name = regime_labels.get(regime_idx, f"REGIME-{regime_idx}")
                    count = int(rp.loc[regime_idx, 'result_count'])
                    win_rate = rp.loc[regime_idx, 'result_mean']
                    profit = rp.loc[regime_idx, 'profit_loss_sum']
                    print(f"  {regime_name}: {count}회, 승률 {win_rate:.1%}, 손익 ${profit:+.2f}")

        month_stats = self.analyze_recent_trades(30)
        if month_stats:
            print("\n[최근 30일 성과]")
            print(f"총 거래: {month_stats['total_trades']}")
            print(f"승/패: {month_stats['wins']}/{month_stats['losses']}")
            print(f"승률: {month_stats.get('win_rate', 0):.2%}")
            print(f"총 손익: ${month_stats['total_profit']:.2f}")
            print(f"평균 손익: ${month_stats.get('avg_profit', 0):.2f}")

        print("\n" + "="*60)


# --------------------
# 사용 예시 (단독 실행)
# --------------------
if __name__ == "__main__":
    from config import Config
    from model_train import ModelTrainer

    Config.create_directories()

    trainer = ModelTrainer(Config)
    trainer.load_model()

    trader = RealTimeTrader(Config, trainer)
    # PREDICTION_WINDOW=10 (옵션 칼만기), 진입 간격은 1분 권장
    trader.run_live_trading(duration_hours=99999, trade_interval_minutes=1)
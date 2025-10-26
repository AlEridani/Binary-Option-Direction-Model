"""
실시간 거래 시스템 - Binance API + 백테스트 + 동적 필터
버전: 1.3.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple, List
import json
from collections import defaultdict, deque
import requests
import uuid
import time
import warnings
warnings.filterwarnings('ignore')

import config
from model_train import ModelTrainer
from feature_engineer import FeatureEngineer
from log_manager import LogManager
from timeframe_manager import TimeframeManager

# ================================
# 바이낸스 API 클라이언트
# ================================
class BinanceAPIClient:
    """바이낸스 API 클라이언트 (시뮬레이션 폴백)"""
    
    def __init__(self, api_key=None, api_secret=None):
        self.api_key = api_key or config.BINANCE_API_KEY
        self.api_secret = api_secret or config.BINANCE_API_SECRET
        self.base_url = "https://api.binance.com"
        self.simulation_mode = False
        self._sim_price = 50000.0

    def get_current_price(self, symbol="BTCUSDT"):
        """현재 가격 조회"""
        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            params = {"symbol": symbol}
            response = requests.get(url, params=params, timeout=3)
            data = response.json()
            return float(data['price'])
        except Exception:
            if not self.simulation_mode:
                self.simulation_mode = True
                print("⚙️  시뮬레이션 모드 활성화")
            change = np.random.normal(0, 50)
            self._sim_price = max(self._sim_price + change, 10000)
            return float(self._sim_price)

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
            if not self.simulation_mode:
                self.simulation_mode = True
                print("⚙️  시뮬레이션 데이터 생성")
            return self._generate_simulation_data(limit)

    def _generate_simulation_data(self, limit=500):
        """시뮬레이션 데이터 생성"""
        end_time = datetime.now(timezone.utc)
        timestamps = [end_time - timedelta(minutes=i) for i in range(limit-1, -1, -1)]
        
        base_price = 50000.0
        returns = np.random.normal(0, 0.002, limit)
        prices = base_price * (1 + returns).cumprod()

        data = []
        for i, ts in enumerate(timestamps):
            close = prices[i]
            open_price = close + np.random.uniform(-50, 50)
            high = max(open_price, close) + np.random.uniform(0, 100)
            low = min(open_price, close) - np.random.uniform(0, 100)
            volume = np.random.uniform(100, 1000)

            data.append({
                'timestamp': ts,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        return pd.DataFrame(data)


# ===========================================
# 실시간 트레이더 (통합 버전)
# ===========================================
class RealTrader:
    """
    실시간 거래 시스템 + 백테스트
    - Binance API 연동
    - 레짐 기반 진입
    - 동적 필터
    - 재학습 트리거
    - 백테스트 엔진
    """
    
    def __init__(self, symbol: str = 'BTCUSDT', api_client=None):
        """초기화"""
        self.symbol = symbol
        
        # 컴포넌트
        self.model_trainer = ModelTrainer()
        self.feature_engineer = FeatureEngineer()
        self.log_manager = LogManager()
        self.tf_manager = TimeframeManager()
        self.api_client = api_client or BinanceAPIClient()
        
        # 모델 로드
        self.model_loaded = self.model_trainer.load_models()
        if not self.model_loaded:
            print("⚠️  모델 미로드")
        
        # 거래 상태
        self.is_running = False
        self.active_positions = {}
        self.max_positions = config.MAX_CONCURRENT_POSITIONS
        self.trade_history = deque(maxlen=config.RETRAIN_CHECK_INTERVAL)
        
        # 재학습
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
        
        # 동적 필터
        self.filter_state = self._load_filter_state()
        
        # 쿨다운
        self.next_entry_after = None
        self.last_attempt_time = None
        
    # ---------- 동적 필터 ----------
    def _load_filter_state(self):
        """동적 필터 로드"""
        filter_path = config.FILTER_STATE_FILE
        
        if filter_path.exists():
            try:
                with open(filter_path, 'r') as f:
                    filters = json.load(f)
                    print(f"✅ 동적 필터 로드: {len(filters.get('active_filters', []))}개")
                    return filters
            except Exception as e:
                print(f"⚠️  필터 로드 실패: {e}")
        
        return {'active_filters': [], 'filter_history': []}
    
    def _save_filter_state(self):
        """동적 필터 저장"""
        try:
            with open(config.FILTER_STATE_FILE, 'w') as f:
                json.dump(self.filter_state, f, indent=2)
        except Exception as e:
            print(f"⚠️  필터 저장 실패: {e}")
    
    def apply_adaptive_filters(self, features_row):
        """적응형 필터 적용"""
        active_filters = self.filter_state.get('active_filters', [])
        if not active_filters:
            return True, []
        
        blocked_reasons = []
        
        for fc in active_filters:
            filter_type = fc.get('type', 'num')
            
            # 복합 조건 필터
            if filter_type == 'compound':
                conditions = fc.get('conditions', [])
                all_met = True
                
                for cond in conditions:
                    field = cond.get('field')
                    operator = cond.get('operator')
                    threshold = cond.get('threshold')
                    
                    if field not in features_row or pd.isna(features_row[field]):
                        all_met = False
                        break
                    
                    value = features_row[field]
                    
                    if operator == '<=' and not (value <= threshold):
                        all_met = False
                        break
                    elif operator == '>' and not (value > threshold):
                        all_met = False
                        break
                
                if all_met:
                    blocked_reasons.append(f"{fc['name']}: {fc['reason']}")
                    continue
            
            # 단일 조건 필터
            field = fc.get('field')
            if not field or field not in features_row or pd.isna(features_row[field]):
                continue
            
            value = features_row[field]
            op = fc.get('operator')
            
            if op == '>':
                th = fc.get('threshold')
                if value > th:
                    blocked_reasons.append(f"{fc['name']}: {field}={value:.4f} > {th:.4f}")
            elif op == '<':
                th = fc.get('threshold')
                if value < th:
                    blocked_reasons.append(f"{fc['name']}: {field}={value:.4f} < {th:.4f}")
        
        return (len(blocked_reasons) == 0), blocked_reasons
    
    # ---------- 예측/진입 ----------
    def predict_next_30m(self) -> Tuple[Optional[float], Optional[int]]:
        """다음 30분봉 예측"""
        if not self.model_loaded:
            return None, None
        
        try:
            # 최근 데이터 로드
            df_1m = self.api_client.get_klines(limit=3000)
            
            # 피처 생성
            df_features = self.feature_engineer.create_feature_pool(df_1m, lookback_bars=100)
            
            if df_features.empty:
                return None, None
            
            # 최신 봉
            latest = df_features.iloc[-1]
            regime = int(latest['regime']) if 'regime' in latest else 0
            
            # 예측
            X = df_features.iloc[[-1]]
            feature_names = self.feature_engineer.get_feature_names(df_features)
            X = X[feature_names]
            
            p_up = self.model_trainer.predict(X, regime=regime, use_regime_model=True)
            
            if isinstance(p_up, np.ndarray):
                p_up = float(p_up[0]) if len(p_up) > 0 else 0.5
            else:
                p_up = float(p_up)
            
            return p_up, regime
            
        except Exception as e:
            print(f"❌ 예측 실패: {e}")
            return None, None
    
    def check_hysteresis(self, p_now: float, direction: str) -> bool:
        """히스테리시스 체크"""
        if direction == 'UP':
            return p_now >= config.CUT_ON_DEFAULT
        else:  # DOWN
            return p_now <= (1 - config.CUT_ON_DEFAULT)
    
    def check_ttl(self, position: Dict, now: datetime) -> bool:
        """TTL 체크"""
        elapsed = (now - position['ttl_start']).total_seconds()
        return elapsed < position['ttl_seconds']
    
    def check_delta_p(self, p_now: float, p_entry: float) -> bool:
        """delta_p 체크"""
        delta = abs(p_now - p_entry)
        return delta >= config.DELTA_P_THRESHOLD
    
    def execute_trade(self, side, p_up, amount=100):
        """거래 실행"""
        if len(self.active_positions) >= self.max_positions:
            return None
        
        trade_id = str(uuid.uuid4())[:8]
        entry_time = datetime.now(timezone.utc)
        expiry_time = entry_time + timedelta(minutes=config.BAR_MINUTES)
        entry_price = self.api_client.get_current_price(self.symbol)
        
        # 레짐 정보
        try:
            df = self.api_client.get_klines(limit=500)
            features = self.feature_engineer.create_feature_pool(df, 100)
            current_regime = int(features['regime'].iloc[-1]) if 'regime' in features.columns else None
        except:
            current_regime = None
        
        info = {
            'trade_id': trade_id,
            'entry_time': entry_time.isoformat(),
            'expiry_time': expiry_time.isoformat(),
            'entry_price': entry_price,
            'direction': int(side),
            'p_up': float(p_up),
            'regime': current_regime,
            'amount': amount,
            'status': 'open',
            'ttl_start': entry_time,
            'ttl_seconds': config.TTL_SECONDS,
            'p_at_entry': float(p_up)
        }
        
        self.active_positions[trade_id] = info
        
        # 로그 기록
        self.log_manager.log_trade_entry_simple(
            trade_id=trade_id,
            direction='UP' if side == 1 else 'DOWN',
            entry_price=entry_price,
            entry_ts=entry_time,
            p_at_entry=p_up,
            regime=current_regime or 0
        )
        
        # 쿨다운
        self.next_entry_after = entry_time + timedelta(seconds=config.REFRACTORY_WINDOW_SECONDS)
        
        direction_str = "롱🟢⬆️" if side == 1 else "숏🔴⬇️"
        regime_labels = {1: "UP🟢", -1: "DOWN🔴", 0: "FLAT⚪", None: "N/A"}
        regime_str = regime_labels.get(current_regime, "N/A")
        
        print("\n" + "="*70)
        print("💰 거래 진입!")
        print("="*70)
        print(f"  🆔 ID: {trade_id}")
        print(f"  📊 방향: {direction_str}")
        print(f"  🎯 레짐: {regime_str}")
        print(f"  📈 P(UP): {p_up:.2%}")
        print(f"  💰 진입가: ${entry_price:,.2f}")
        print(f"  💵 금액: ${amount}")
        print(f"  📈 활성: {len(self.active_positions)}/{self.max_positions}")
        print("="*70 + "\n")
        
        return trade_id
    
    def check_trade_result(self, trade_id):
        """거래 결과 확인"""
        pos = self.active_positions.get(trade_id)
        if not pos:
            return None
        
        entry_time = datetime.fromisoformat(pos['entry_time'].replace("Z",""))
        expiry_time = datetime.fromisoformat(pos['expiry_time'].replace("Z",""))
        now = datetime.now(timezone.utc)
        
        if now < expiry_time:
            return None
        
        entry_price = pos['entry_price']
        exit_price = self.api_client.get_current_price(self.symbol)
        
        direction = pos['direction']
        is_win = (exit_price > entry_price) if direction == 1 else (exit_price < entry_price)
        
        amount = pos['amount']
        profit = amount * config.PAYOUT_30M_PLUS if is_win else -amount
        result = 1 if is_win else 0
        
        pos['exit_time'] = now.isoformat()
        pos['exit_price'] = exit_price
        pos['result'] = result
        pos['profit_loss'] = profit
        pos['status'] = 'closed'
        
        self.update_performance(is_win, profit, direction)
        self.trade_history.append(result)
        
        # 로그 업데이트
        self.log_manager.update_trade_result(
            trade_id=trade_id,
            result='WIN' if is_win else 'LOSS',
            label_price=exit_price,
            payout=profit
        )
        
        result_emoji = "✅ 승리" if is_win else "❌ 패배"
        print(f"\n{result_emoji}: {trade_id}")
        print(f"  진입: ${entry_price:,.2f} → 청산: ${exit_price:,.2f}")
        print(f"  손익: ${profit:+,.2f}\n")
        
        del self.active_positions[trade_id]
        
        # 재학습 체크
        self.trades_since_last_check += 1
        if self.trades_since_last_check >= config.RETRAIN_CHECK_INTERVAL:
            if self.check_retrain_trigger():
                self.pending_retrain = True
                print("⚠️  재학습 필요 - 신규 진입 중단")
        
        return result
    
    def update_performance(self, is_win, profit, direction):
        """성능 통계 업데이트"""
        self.performance_metrics['total_trades'] += 1
        
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
    
    def check_retrain_trigger(self) -> None:
        """재학습 트리거 체크 (윌슨 하한)"""
        n = len(self.trade_history)
        if n < config.MIN_TRADES_FOR_RETRAIN:
            return False
        
        wins = sum(self.trade_history)
        p_hat = wins / n
        
        # 윌슨 하한
        z = 1.96
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2*n)) / denominator
        margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4*n**2)) / denominator
        lower_bound = center - margin
        
        print(f"\n재학습 체크: 승률={p_hat:.2%}, 하한={lower_bound:.2%}, 임계={config.RETRAIN_WIN_RATE_THRESHOLD:.0%}")
        
        return lower_bound < config.RETRAIN_WIN_RATE_THRESHOLD
    
    # ---------- 메인 루프 ----------

    def should_generate_signal(self) -> bool:
        """신호 생성 시점: 29분 40~50초"""
        now = datetime.now(timezone.utc)
        minute = now.minute
        second = now.second
        
        # 29분 40~50초 또는 59분 40~50초
        if minute in [29, 59] and 40 <= second <= 50:
            return True
        
        return False
    def run(self):
        """메인 루프 (1분마다 호출)"""
        
        # 신호 생성 시점 체크
        if not self.should_generate_signal():
            # 기존 포지션만 체크
            for trade_id in list(self.active_positions.keys()):
                self.check_trade_result(trade_id)
            return
        
        # 예측 수행
        p_up, regime = self.predict_next_30m()
        if p_up is None:
            return
        
        # 진입 판단
        if len(self.active_positions) < self.max_positions:
            side = None
            
            if regime == 1 and p_up >= config.CUT_ON_DEFAULT:
                side = 1
            elif regime == -1 and (1 - p_up) >= config.CUT_ON_DEFAULT:
                side = 0
            
            if side is not None:
                # 쿨다운 체크
                if self.next_entry_after is None or now >= self.next_entry_after:
                    # 필터 체크
                    try:
                        df = self.api_client.get_klines(limit=500)
                        features = self.feature_engineer.create_feature_pool(df, 100)
                        ok, reasons = self.apply_adaptive_filters(features.iloc[-1])
                        
                        if ok:
                            self.execute_trade(side, p_up)
                        else:
                            print(f"  ❌ 필터 차단: {reasons}")
                    except Exception as e:
                        print(f"  ⚠️  필터 체크 실패: {e}")
        
        # 기존 포지션 관리
        for trade_id in list(self.active_positions.keys()):
            self.check_trade_result(trade_id)
    
    # ---------- 백테스트 ----------
    def backtest(self, historical_data, start_date=None, end_date=None):
        """백테스트 실행"""
        print("\n" + "="*70)
        print("백테스트 시작")
        print("="*70)
        
        # 날짜 필터
        if start_date:
            historical_data = historical_data[historical_data['timestamp'] >= pd.to_datetime(start_date, utc=True)]
        if end_date:
            historical_data = historical_data[historical_data['timestamp'] <= pd.to_datetime(end_date, utc=True)]
        
        # 피처 생성
        features = self.feature_engineer.create_feature_pool(historical_data, lookback_bars=100)
        
        if features.empty:
            print("❌ 피처 생성 실패")
            return pd.DataFrame()
        
        # 타겟 생성
        features['next_open'] = features['open'].shift(-1)
        features['next_close'] = features['close'].shift(-1)
        features['target'] = (features['next_close'] > features['next_open']).astype(int)
        features = features.dropna(subset=['target']).reset_index(drop=True)
        
        print(f"백테스트 데이터: {len(features):,}건")
        
        trades = []
        
        for i in range(len(features) - 1):
            try:
                X_current = features.iloc[[i]]
                feature_names = self.feature_engineer.get_feature_names(features)
                X = X_current[feature_names]
                
                regime = int(features['regime'].iloc[i]) if 'regime' in features.columns else 0
                p_up = self.model_trainer.predict(X, regime=regime, use_regime_model=True)
                
                if isinstance(p_up, np.ndarray):
                    p_up = float(p_up[0])
                
                # 진입 결정
                side = None
                if regime == 1 and p_up >= config.CUT_ON_DEFAULT:
                    side = 1
                elif regime == -1 and (1 - p_up) >= config.CUT_ON_DEFAULT:
                    side = 0
                
                if side is None:
                    continue
                
                actual = int(features['target'].iloc[i])
                correct = int(side == actual)
                
                trades.append({
                    'timestamp': features['bar30_start'].iloc[i] if 'bar30_start' in features.columns else i,
                    'p_up': p_up,
                    'regime': regime,
                    'decision': side,
                    'actual': actual,
                    'correct': correct
                })
                
            except Exception:
                continue
        
        trades_df = pd.DataFrame(trades)
        
        if trades_df.empty:
            print("❌ 거래 없음")
            return pd.DataFrame()
        
        # 결과 출력
        total = len(trades_df)
        wins = trades_df['correct'].sum()
        win_rate = wins / total
        
        profit = (wins * 100 * config.PAYOUT_30M_PLUS) - ((total - wins) * 100)
        
        print(f"\n백테스트 결과:")
        print(f"  총 거래: {total}")
        print(f"  승/패: {wins}/{total-wins}")
        print(f"  승률: {win_rate:.2%}")
        print(f"  손익: ${profit:+,.2f}")
        print(f"  평균: ${profit/total:.2f}/거래")
        
        return trades_df
    
    def get_status(self) -> Dict:
        """현재 상태 반환"""
        return {
            'symbol': self.symbol,
            'model_loaded': self.model_loaded,
            'active_positions': len(self.active_positions),
            'max_positions': self.max_positions,
            'trade_count': self.performance_metrics['total_trades'],
            'win_rate': self.performance_metrics['wins'] / max(1, self.performance_metrics['total_trades']),
            'filter_patterns': len(self.filter_state.get('active_filters', []))
        }


# ============================================================
# 테스트 및 검증
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("RealTrader 테스트 (API + 백테스트)")
    print("=" * 60)
    
    # 초기화
    trader = RealTrader(symbol='BTCUSDT')
    
    status = trader.get_status()
    print(f"\n✅ 초기화 완료")
    print(f"  Symbol: {status['symbol']}")
    print(f"  Model Loaded: {status['model_loaded']}")
    
    # API 테스트
    print("\n💰 API 테스트")
    price = trader.api_client.get_current_price('BTCUSDT')
    print(f"  현재가: ${price:,.2f}")
    
    df = trader.api_client.get_klines(limit=10)
    print(f"  캔들 로드: {len(df)}개")
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)
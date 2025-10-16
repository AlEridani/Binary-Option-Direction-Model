import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class StrategyTradingSystem:
    def __init__(self, model_path, threshold_override=None, reverse_mode=False):
        """실시간 거래 시스템 초기화 (정상 모드 기본 설정)"""
        print("="*80)
        print("📊 ML 기반 실시간 거래 시스템 (정상/역발상 모드 선택)")
        print("="*80)
        
        self.reverse_mode = reverse_mode 
        if self.reverse_mode:
            print("⚠️  역발상 모드 활성화: SHORT 신호 → LONG 진입, LONG 신호 → SHORT 진입")
        else:
            print("✅ 정상 모드 활성화: LONG 신호 → LONG 진입, SHORT 신호 → SHORT 진입")
        
        # 바이낸스 API 초기화
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            print("⚠️  API 키 없음 - 데모 모드 (실제 거래 불가)")
            self.client = None
        else:
            try:
                self.client = Client(api_key, api_secret) 
                print("✅ 바이낸스 API 연결 성공")
            except Exception as e:
                print(f"❌ 바이낸스 API 연결 실패: {e}")
                self.client = None
        
        # 모델 로드
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        
        print(f"\n📦 모델 로딩: {os.path.basename(model_path)}")
        self.model_package = joblib.load(model_path)
        self.long_model = self.model_package['long_model']
        self.short_model = self.model_package['short_model']
        self.feature_columns = self.model_package['feature_columns']
        self.config = self.model_package['config']
        
        # 임계값 오버라이드
        if threshold_override is not None:
            self.config['threshold'] = threshold_override
            print(f"⚠️  임계값 오버라이드: {threshold_override}")
        
        print(f"✅ 모델 로드 완료")
        print(f"   룩백: {self.config.get('lookback', 30)}분")
        print(f"   임계값: {self.config.get('threshold', 0.65)}")
        print(f"   옵션 기간: {self.config.get('option_duration', 10)}분")
        
        # 변동성 필터 임계값 (기존 값 유지)
        self.volatility_thresholds = {
            'atr_90': 0.001075,
            'atr_95': 0.001454
        }
        print(f"\n🛡️  변동성 필터 활성화")

        # 🚨 [추가] 비정상 캔들 감지 및 거래 중단 설정
        self.lockout_time_minutes = 5  # 거래 중단 시간 (분)
        self.last_lockout_end_time = datetime.min # 마지막 잠금 해제 시간 추적
        self.vol_surge_thresholds = {
            'range_multiplier': 2.5, # 캔들 범위가 20-ATR%의 2.5배 이상
            'volume_multiplier': 2.0 # 거래량이 10-MA의 2.0배 이상
        }
        print(f"🛡️ 비정상 변동성 감지: 잠금 시간 {self.lockout_time_minutes}분 (ATR>{self.vol_surge_thresholds['range_multiplier']}x, Vol>{self.vol_surge_thresholds['volume_multiplier']}x)")
        
        # 데이터 버퍼
        self.lookback = self.config.get('lookback', 30)
        self.option_duration = self.config.get('option_duration', 10)
        buffer_size = self.lookback + self.option_duration + 100
        self.data_buffer = deque(maxlen=buffer_size)
        
        # 활성 거래
        self.active_trades = []
        # === 여기서 최대 동시 포지션 수를 설정합니다 ===
        self.max_active_trades = 5 
        
        # CSV 파일 설정
        mode_suffix = "REVERSE" if self.reverse_mode else "NORMAL"
        self.csv_filename = f"trades_{mode_suffix}_{datetime.now().strftime('%Y%m%d')}.csv"
        self.features_log_filename = f"features_log_{mode_suffix}_{datetime.now().strftime('%Y%m%d')}.csv"
        self._init_csv()
        
        # 통계
        self.stats = {
            'total': 0, 'wins': 0, 'losses': 0, 'pending': 0,
            'long_trades': 0, 'short_trades': 0
        }
        
        self.running = False
        
    def _init_csv(self):
        """CSV 파일 초기화"""
        # 거래 결과 CSV 초기화 (이 파일은 거래 완료 시점에 데이터가 추가됨)
        if not os.path.exists(self.csv_filename):
            df = pd.DataFrame(columns=[
                'entry_time', 'exit_time', 'direction', 
                'entry_price', 'exit_price', 'result', 
                'probability', 'profit_pct', 'original_signal'
            ])
            df.to_csv(self.csv_filename, index=False)
            print(f"📝 거래 결과 CSV: {self.csv_filename}")
        
        # 피처 로그 CSV 초기화
        if not os.path.exists(self.features_log_filename):
            print(f"📝 피처 로그 CSV: {self.features_log_filename}")
        else:
            print(f"📝 기존 피처 로그 사용: {self.features_log_filename}")

    def fetch_initial_data(self, symbol='BTCUSDT'):
        """초기 데이터 로드"""
        print(f"\n📊 {symbol} 초기 데이터 로딩...")
        
        if self.client is None:
            print("⚠️  데모 모드 - 실제 데이터 로드 없이 더미 데이터 사용")
            return False 
        
        try:
            # 실제 바이낸스 API 호출
            klines = self.client.get_klines(
                symbol=symbol,
                interval='1m',
                limit=self.lookback + 100
            )
            
            for kline in klines:
                candle = {
                    'timestamp': pd.to_datetime(kline[0], unit='ms'),
                    'open': float(kline[1]), 'high': float(kline[2]), 
                    'low': float(kline[3]), 'close': float(kline[4]), 
                    'volume': float(kline[5])
                }
                self.data_buffer.append(candle)
            
            print(f"✅ {len(self.data_buffer)}개 캔들 로드 완료")
            return True
            
        except BinanceAPIException as e:
            print(f"❌ 데이터 로드 실패: {e} (API 키 또는 권한 확인 필요)")
            return False
    
    def calculate_indicators_and_features(self, df):
        """훈련 시 사용된 지표 및 피처를 실시간 데이터에 맞게 계산하는 통합 함수"""
        required_data = max(self.lookback, 50) + 10
        if len(df) < required_data:
            return None, None

        df = df.copy()
        df.set_index('timestamp', inplace=True)

        # 1. RSI 계산 보조 함수 정의
        def calculate_rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        # 2. 15분봉 지표 계산 및 머지
        df_15m = df.resample('15T').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna()
        df_15m['rsi_14'] = calculate_rsi(df_15m['close'], period=14)
        df_15m['ema_20_15m'] = df_15m['close'].ewm(span=20, adjust=False).mean()
        df_15m['ema_50_15m'] = df_15m['close'].ewm(span=50, adjust=False).mean()
        df_15m['trend_15m'] = (df_15m['ema_20_15m'] > df_15m['ema_50_15m']).astype(int)
        df_15m_reindex = df_15m.reindex(df.index, method='ffill')
        df['rsi_14'] = df_15m_reindex['rsi_14'].values
        df['trend_15m'] = df_15m_reindex['trend_15m'].values
        
        # 3. 1시간봉 장기 추세 필터 (🚨 요청에 따라 계산 제거 - 10분 거래에 불필요)
        # 4. 1분봉 지표 계산
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['trend_ema'] = (df['ema_20'] > df['ema_50']).astype(int)
        df['body'] = df['close'] - df['open']
        df['range'] = df['high'] - df['low']
        df['body_ratio'] = abs(df['body']) / (df['range'] + 1e-8)
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['upper_shadow_ratio'] = df['upper_shadow'] / (df['range'] + 1e-8)
        df['lower_shadow_ratio'] = df['lower_shadow'] / (df['range'] + 1e-8)
        df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
        df['volume_surge'] = (df['volume'] > df['volume_ma_10'] * 1.5).astype(int)
        df['recent_high_20'] = df['high'].rolling(window=20).max()
        df['recent_low_20'] = df['low'].rolling(window=20).min()
        df['distance_to_high'] = (df['recent_high_20'] - df['close']) / df['close']
        df['distance_to_low'] = (df['close'] - df['recent_low_20']) / df['close']
        df['is_hammer'] = ((df['lower_shadow_ratio'] > 0.6) & (df['upper_shadow_ratio'] < 0.15) & (df['body_ratio'] < 0.3)).astype(int)
        df['is_shooting_star'] = ((df['upper_shadow_ratio'] > 0.6) & (df['lower_shadow_ratio'] < 0.15) & (df['body_ratio'] < 0.3)).astype(int)
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_low_with_hammer'] = (df['rsi_oversold'] & df['is_hammer']).astype(int)
        df['rsi_high_with_shooting'] = (df['rsi_overbought'] & df['is_shooting_star']).astype(int)
        
        # 🌟 [반등 필터 계산] 1분봉 RSI 3기간 계산을 위한 보조 함수 (Calculate RSI 재사용)
        df['rsi_3'] = calculate_rsi(df['close'], period=3) 

        # 5. 룩백 피처 추출
        i = len(df) - 1 
        if i < self.lookback:
            return None, None
            
        window_df = df.iloc[i - self.lookback : i]
        
        feature_dict = {}
        current_candle = df.iloc[i]
        
        feature_dict['current_rsi_14'] = current_candle['rsi_14']
        feature_dict['rsi_oversold'] = current_candle['rsi_oversold']
        feature_dict['rsi_overbought'] = current_candle['rsi_overbought']
        feature_dict['rsi_low_with_hammer'] = current_candle['rsi_low_with_hammer']
        feature_dict['rsi_high_with_shooting'] = current_candle['rsi_high_with_shooting']
        feature_dict['is_hammer'] = current_candle['is_hammer']
        feature_dict['is_shooting_star'] = current_candle['is_shooting_star']
        feature_dict['volume_surge'] = current_candle['volume_surge']
        feature_dict['trend_ema'] = current_candle['trend_ema']
        feature_dict['trend_15m'] = current_candle['trend_15m']
        feature_dict['distance_to_high'] = current_candle['distance_to_high']
        feature_dict['distance_to_low'] = current_candle['distance_to_low']
        
        # 🌟 [반등 필터 피처] 1분봉 RSI 3기간 값 추가
        feature_dict['current_rsi_3'] = current_candle['rsi_3'] 
        
        returns = (window_df['close'] / window_df['open'] - 1).values
        for j in range(self.lookback):
            if j < 15:
                feature_dict[f'ret_{j}'] = returns[-(j+1)]
        
        current_features = pd.Series(feature_dict)
        # 모델 학습 시 사용된 피처 목록으로 재정렬
        X_df = pd.DataFrame([current_features]).reindex(columns=self.feature_columns, fill_value=0.0)
        
        # 🚨 1시간봉 추세 필터 제거에 따라 더미값 1 할당 (check_entry_signal과의 형식 유지를 위함)
        trend_filter = 1 
        
        return X_df.iloc[0].squeeze(), trend_filter 

    
    def calculate_current_volatility(self, df):
        """현재 변동성 계산 (ATR)"""
        if len(df) < 21:
            return None
        
        recent_20 = df.iloc[-20:].copy()
        recent_20['tr'] = np.maximum(
            recent_20['high'] - recent_20['low'],
            np.maximum(
                abs(recent_20['high'] - recent_20['close'].shift(1)),
                abs(recent_20['low'] - recent_20['close'].shift(1))
            )
        )
        
        atr = recent_20['tr'].mean()
        current_price = df.iloc[-1]['close']
        atr_pct = atr / current_price
        
        return atr_pct
    
    def get_dynamic_threshold(self, atr_pct):
        """변동성에 따른 동적 임계값 계산"""
        base_threshold = self.config.get('threshold', 0.65)
        
        if atr_pct > self.volatility_thresholds['atr_95']:
            return None, 'BLOCKED'
        elif atr_pct > self.volatility_thresholds['atr_90']:
            adjusted = min(base_threshold + 0.05, 0.85)
            return adjusted, 'ADJUSTED'
        else:
            return base_threshold, 'NORMAL'
    
    def check_entry_signal(self):
        """진입 신호 확인 (정상 모드/역발상 적용)"""
        current_time = datetime.now()
        
        # 🚨 [1. 잠금 상태 확인]
        if current_time < self.last_lockout_end_time:
            lock_remaining = (self.last_lockout_end_time - current_time).total_seconds()
            print(f"🚫 거래 잠금 중 ({lock_remaining:.0f}초 남음). 비정상 캔들 발생 후 대기 중.")
            return None, None, {
                'atr_pct': None,
                'status': 'LOCKOUT',
                'threshold': None
            }

        if len(self.data_buffer) < self.lookback + 21: # ATR 계산을 위해 최소 21개 필요
            return None, None, None
        
        df = pd.DataFrame(list(self.data_buffer))
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        atr_pct = self.calculate_current_volatility(df)
        
        if atr_pct is None:
            return None, None, None
        
        # 🚨 [2. 비정상 변동성 캔들 감지 로직] (방금 닫힌 봉 기준)
        if len(df) >= 2:
            current_closed_candle = df.iloc[-2]
            current_range = current_closed_candle['high'] - current_closed_candle['low']
            current_price = current_closed_candle['close']
            
            # 1. 캔들 범위가 ATR의 X배 이상인가?
            range_pct = current_range / current_price
            atr_multiple = range_pct / (atr_pct + 1e-8)
            
            # 2. 거래량이 MA의 X배 이상인가?
            # df['volume_ma_10']를 계산하기 위해 df 전체를 사용해야 하지만, 여기서는 간소화를 위해 마지막 캔들 기준으로 계산합니다.
            volume_ma_10 = df['volume'].rolling(window=10).mean().iloc[-2]
            volume_multiple = current_closed_candle['volume'] / (volume_ma_10 + 1e-8)

            if (atr_multiple >= self.vol_surge_thresholds['range_multiplier'] and
                volume_multiple >= self.vol_surge_thresholds['volume_multiplier']):
                
                self.last_lockout_end_time = current_time + timedelta(minutes=self.lockout_time_minutes)
                print(f"\n🔥🔥 비정상 캔들 감지! {self.lockout_time_minutes}분간 거래 잠금 발동! 🔥🔥")
                print(f"   ATR 배율: {atr_multiple:.1f}x | 거래량 배율: {volume_multiple:.1f}x")
                print(f"   잠금 해제 시간: {self.last_lockout_end_time.strftime('%H:%M:%S')}")
                
                # 잠금 발동 즉시 신호 확인 중단
                return None, None, {
                    'atr_pct': atr_pct,
                    'status': 'LOCKOUT_TRIGGERED',
                    'threshold': None
                }
        
        # 3. [기존 동적 임계값 및 신호 확인 로직]
        dynamic_threshold, volatility_status = self.get_dynamic_threshold(atr_pct)
        
        if dynamic_threshold is None:
            return None, None, {
                'atr_pct': atr_pct,
                'status': volatility_status,
                'threshold': None
            }
        
        features_series, trend_filter = self.calculate_indicators_and_features(df.iloc[:-1].copy())
        
        if features_series is None:
            return None, None, None
            
        rsi_3 = features_series.get('current_rsi_3', 50.0) 
        
        try:
            X = features_series[self.feature_columns].values.reshape(1, -1)
            long_prob = self.long_model.predict_proba(X)[0][1] 
            short_prob = self.short_model.predict_proba(X)[0][1]
            
            threshold = dynamic_threshold
            
            long_signal = long_prob > threshold
            short_signal = short_prob > threshold
            
            # 피처 로그 저장
            self._save_features_log(features_series, long_prob, short_prob, long_signal, short_signal, trend_filter)
            
            signals = []
            
            if not self.reverse_mode:
                # 🌟🌟🌟 정상 모드 로직 🌟🌟🌟
                
                # 1. LONG 진입 로직: 장기 추세 필터 제거
                if long_signal and not short_signal:
                    signals.append({'direction': 'LONG', 'probability': long_prob, 'original_signal': 'LONG'})
                    print(f"   ⬆️  정상: LONG 신호({long_prob:.2%}) → LONG 진입")
                
                # 2. SHORT 진입 로직: Long 모델 역필터 및 1분봉 RSI 필터 추가
                elif short_signal and not long_signal and long_prob < 0.60 and rsi_3 < 90:
                    signals.append({'direction': 'SHORT', 'probability': short_prob, 'original_signal': 'SHORT'})
                    print(f"   ⬇️  정상: SHORT 신호({short_prob:.2%}) → SHORT 진입 (L-Prob:<0.60, RSI3:<90)")
                
                # 3. 신호/필터 충돌 및 차단 로직
                elif (long_signal and short_signal) or (short_signal and long_prob >= 0.60) or (short_signal and rsi_3 >= 90):
                    is_l_filtered = long_prob >= 0.60
                    is_rsi_filtered = rsi_3 >= 90
                    print(f"   ⚠️  신호/필터 충돌/차단 → 관망 (L:{long_prob:.2%}, S:{short_prob:.2%}, L필터:{is_l_filtered}, RSI3필터:{is_rsi_filtered})")
                else:
                    print(f"   ➖  신호 없음 (L:{long_prob:.2%}, S:{short_prob:.2%})")

            else:
                # 역발상 모드 (기존 로직 유지)
                if short_signal and not long_signal:
                    signals.append({'direction': 'LONG', 'probability': short_prob, 'original_signal': 'SHORT'})
                    print(f"   🔄 역발상: SHORT 신호({short_prob:.2%}) → LONG 진입")
                elif long_signal and not short_signal:
                    signals.append({'direction': 'SHORT', 'probability': long_prob, 'original_signal': 'LONG'})
                    print(f"   🔄 역발상: LONG 신호({long_prob:.2%}) → SHORT 진입")
                elif (long_signal and short_signal):
                    print(f"   ⚠️  신호 충돌 → 관망 (롱:{long_prob:.2%}, 숏:{short_prob:.2%})")
                else:
                    print(f"   ➖  신호 없음 (롱:{long_prob:.2%}, 숏:{short_prob:.2%})")
            
            volatility_info = {
                'atr_pct': atr_pct,
                'status': volatility_status,
                'threshold': threshold
            }
            
            return signals if signals else None, (long_prob, short_prob), volatility_info
            
        except Exception as e:
            print(f"❌ 신호 확인 오류: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def enter_trade(self, signal, entry_price):
        """거래 진입 (파일 수정 없이 활성 거래 목록에 추가만 함)
        반환: True(진입 성공) / False(진입 거부: 최대 포지션 도달)
        """
        # 동시 활성 거래 제한 확인
        if len(self.active_trades) >= self.max_active_trades:
            print(f"⚠️ 진입 거부: 활성 거래가 최대({self.max_active_trades})에 도달했습니다.")
            return False
            
        # 🚨 [추가] 잠금 상태 재확인 (혹시 모를 지연 실행 방지)
        if datetime.now() < self.last_lockout_end_time:
            print(f"🚫 진입 거부: 잠금 시간 ({self.last_lockout_end_time.strftime('%H:%M:%S')}) 이전입니다.")
            return False

        entry_time = datetime.now()
        exit_time = entry_time + timedelta(minutes=self.option_duration)
        
        trade = {
            'direction': signal['direction'],
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': None,
            'probability': signal['probability'],
            'original_signal': signal.get('original_signal', signal['direction']),
            'status': 'ACTIVE',
            'result': None
        }
        
        self.active_trades.append(trade)
        self.stats['total'] += 1
        self.stats['pending'] += 1
        
        if signal['direction'] == 'LONG':
            self.stats['long_trades'] += 1
        else:
            self.stats['short_trades'] += 1
        
        emoji = '🔺' if signal['direction'] == 'LONG' else '🔻'
        orig_signal_text = signal.get('original_signal', signal['direction'])
        
        if self.reverse_mode:
            print(f"   {emoji} {trade['direction']} 진입 (원래: {orig_signal_text}) | 종료: {exit_time.strftime('%H:%M:%S')}")
        else:
            print(f"   {emoji} {trade['direction']} 진입 | 종료: {exit_time.strftime('%H:%M:%S')}")
        return True
    
    def check_exits(self):
        """만료된 거래 확인 및 청산 (이 시점에 CSV 파일에 추가됨)"""
        if len(self.data_buffer) < 2:
            return
        
        current_time = datetime.now()
        completed_indices = []
        
        # 청산 가격은 현재 닫힌 봉의 종가
        exit_price = self.data_buffer[-2]['close']
        
        for idx, trade in enumerate(self.active_trades):
            if current_time >= trade['exit_time']:
                trade['exit_price'] = exit_price
                trade['status'] = 'COMPLETED'
                
                # 결과 판단
                if trade['direction'] == 'LONG':
                    trade['result'] = 'WIN' if exit_price > trade['entry_price'] else 'LOSS'
                else:
                    trade['result'] = 'WIN' if exit_price < trade['entry_price'] else 'LOSS'
                
                profit_pct = ((exit_price - trade['entry_price']) / trade['entry_price']) * 100
                if trade['direction'] == 'SHORT':
                    profit_pct = -profit_pct
                
                trade['profit_pct'] = profit_pct
                
                self.stats['pending'] -= 1
                if trade['result'] == 'WIN':
                    self.stats['wins'] += 1
                else:
                    self.stats['losses'] += 1
                
                # ✅ 거래 완료 시점에 CSV 파일에 추가
                self._save_trade_to_csv(trade)
                
                emoji = '✅' if trade['result'] == 'WIN' else '❌'
                orig_signal = f" (원래: {trade['original_signal']})" if self.reverse_mode else ""
                print(f"\n{emoji} 거래 종료 - {trade['direction']}{orig_signal}")
                print(f"   진입: {trade['entry_price']:.2f} → 종료: {exit_price:.2f}")
                print(f"   결과: {trade['result']}")
                
                completed_indices.append(idx)
        
        for idx in reversed(completed_indices):
            self.active_trades.pop(idx)
    
    def _save_features_log(self, features, long_prob, short_prob, long_signal, short_signal, trend_filter):
        """피처와 예측 확률 로그 저장 (매 분마다 추가)"""
        log_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'long_prob': long_prob,
            'short_prob': short_prob,
            'long_signal': int(long_signal),
            'short_signal': int(short_signal),
            'is_uptrend_1h': trend_filter # 🚨 이제 더미값이지만 로깅을 위해 필드 유지
        }
        
        for col in self.feature_columns:
            log_data[col] = features.get(col, np.nan)
        
        # 🌟 1분봉 RSI 3기간 값도 로그에 포함
        log_data['current_rsi_3'] = features.get('current_rsi_3', np.nan)
        
        df_log = pd.DataFrame([log_data])
        
        if not os.path.exists(self.features_log_filename):
            df_log.to_csv(self.features_log_filename, index=False)
        else:
            df_log.to_csv(self.features_log_filename, mode='a', header=False, index=False)
    
    def _save_trade_to_csv(self, trade):
        """거래 내역 CSV 저장 (거래 완료 시점에 추가)"""
        df = pd.DataFrame([{
            'entry_time': trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
            'exit_time': trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S'),
            'direction': trade['direction'],
            'entry_price': trade['entry_price'],
            'exit_price': trade['exit_price'],
            'result': trade['result'],
            'probability': trade['probability'],
            'profit_pct': f"{trade['profit_pct']:+.2f}%",
            'original_signal': trade['original_signal']
        }])
        
        df.to_csv(self.csv_filename, mode='a', header=False, index=False)
    
    def print_statistics(self):
        """통계 출력 (화면을 지우지 않고 누적 출력)"""
        print("\n" + "#"*80)
        mode_text = "🔄 역발상 전략" if self.reverse_mode else "📊 정상 전략"
        print(f"{mode_text} 실시간 거래 통계")
        print("#"*80)
        
        if self.stats['total'] > 0:
            completed = self.stats['wins'] + self.stats['losses']
            win_rate = (self.stats['wins'] / completed * 100) if completed > 0 else 0
            expected_return = (win_rate/100 * 0.80) - ((100-win_rate)/100 * 1)
            
            print(f"총 거래: {self.stats['total']} (완료: {completed}, 대기: {self.stats['pending']})")
            
            if completed > 0:
                print(f"승리: {self.stats['wins']} | 패배: {self.stats['losses']}")
                print(f"승률: {win_rate:.1f}%")
                print(f"롱/숏: {self.stats['long_trades']}/{self.stats['short_trades']}")
                print(f"기대 수익률 (80% 페이아웃 기준): {expected_return*100:+.1f}% per trade")
                print(f"손익분기: 55.56% | 차이: {win_rate - 55.56:+.2f}%p")
                
                if win_rate >= 55.56:
                    print(f"✅ 수익성 기준 충족")
                else:
                    print(f"⚠️  손익분기 미달")
        else:
            print("거래 기록 없음")
        print("="*80)

    
    def run_realtime(self, symbol='BTCUSDT', check_interval=1):
        """실시간 거래 실행"""
        print(f"\n{'='*80}")
        mode_text = "🔄 역발상 전략" if self.reverse_mode else "📊 정상 전략"
        print(f"{mode_text} 실시간 거래 시작")
        print(f"{'='*80}")
        print(f"심볼: {symbol}")
        print(f"저장 파일: {self.csv_filename}")
        print(f"Ctrl+C로 중단\n")
        
        self.running = True
        last_check_minute = None
        pending_signals = None
        
        try:
            while self.running:
                try:
                    current_time = datetime.now()
                    current_minute = current_time.replace(second=0, microsecond=0)
                    current_second = current_time.second
                    
                    # 최신 데이터 업데이트
                    if self.client:
                        klines = self.client.get_klines(symbol=symbol, interval='1m', limit=2)
                        latest_candle = {
                            'timestamp': pd.to_datetime(klines[-1][0], unit='ms'),
                            'open': float(klines[-1][1]), 'high': float(klines[-1][2]), 
                            'low': float(klines[-1][3]), 'close': float(klines[-1][4]), 
                            'volume': float(klines[-1][5])
                        }
                        
                        if len(self.data_buffer) == 0 or self.data_buffer[-1]['timestamp'] < latest_candle['timestamp']:
                            self.data_buffer.append(latest_candle)
                        elif len(self.data_buffer) > 0 and self.data_buffer[-1]['timestamp'] == latest_candle['timestamp']:
                            self.data_buffer[-1] = latest_candle
                        
                    # 매 분 58~59초: 신호 판단
                    if 58 <= current_second <= 59:
                        if last_check_minute is None or current_minute > last_check_minute:
                            
                            # 🚨 거래 잠금 상태인 경우 메시지만 출력하고 스킵
                            if current_time < self.last_lockout_end_time:
                                lock_remaining = (self.last_lockout_end_time - current_time).total_seconds()
                                print(f"🚫 [{current_time.strftime('%H:%M:%S')}] 잠금 중 ({lock_remaining:.0f}초 남음)...")
                                last_check_minute = current_minute
                                time.sleep(1)
                                continue

                            print(f"\n{'='*60}")
                            print(f"🔍 [{current_time.strftime('%H:%M:%S')}] 신호 판단 중...")
                            
                            signals, probs, vol_info = self.check_entry_signal()
                            
                            # 🚨 잠금 발동 시 루프를 다시 시작
                            if vol_info and vol_info['status'] == 'LOCKOUT_TRIGGERED':
                                pending_signals = None
                                last_check_minute = current_minute
                                continue
                            
                            if vol_info:
                                atr_pct = vol_info['atr_pct']
                                status = vol_info['status']
                                threshold = vol_info['threshold']
                                
                                if status == 'BLOCKED':
                                    print(f"🚫 변동성 과다! ATR: {atr_pct:.4%}")
                                    pending_signals = None
                                elif status == 'ADJUSTED':
                                    print(f"⚠️  변동성 높음! 임계값: {threshold:.2f}")
                                    pending_signals = signals
                                else:
                                    pending_signals = signals
                            
                            if signals:
                                long_prob, short_prob = probs
                                print(f"✅ 신호 감지! {len(signals)}개")
                                for idx, signal in enumerate(signals, 1):
                                    orig = f" ← {signal['original_signal']}" if self.reverse_mode else ""
                                    print(f"   [{idx}] {signal['direction']}{orig} | 확률: {signal['probability']:.2%}")
                                print(f"⏳ 다음 봉 open 진입 대기...")
                            
                            print(f"{'='*60}")
                            last_check_minute = current_minute
                    
                    # 매 분 정각 0~2초: 진입 실행
                    elif 0 <= current_second <= 2:
                        if pending_signals:
                            # 🚨 진입 직전 잠금 상태 재확인
                            if datetime.now() < self.last_lockout_end_time:
                                print(f"🚫 진입 시간 ({current_time.strftime('%H:%M:%S')})에 잠금 상태입니다. 진입 스킵.")
                                pending_signals = None
                                time.sleep(1) 
                                continue

                            print(f"\n💰 [{current_time.strftime('%H:%M:%S')}] 진입 실행!")
                            
                            if self.client and len(self.data_buffer) > 0:
                                entry_price = self.data_buffer[-1]['open'] 
                                
                                print(f"   초봉 open 진입가: {entry_price:.2f}")
                                # 변경: pending_signals 순회 중 최대 활성 포지션 도달하면 중단
                                for signal in pending_signals:
                                    if len(self.active_trades) >= self.max_active_trades:
                                        print(f"⚠️ 더 이상 진입하지 않습니다. 활성 거래 수가 최대({self.max_active_trades})입니다.")
                                        break
                                    entered = self.enter_trade(signal, entry_price)
                                    if not entered:
                                        break
                            else:
                                print("⚠️  API 미연결 또는 데이터 부족 - 진입 스킵")
                            
                            pending_signals = None
                    
                    # 거래 청산 확인
                    self.check_exits()
                    
                    # 매 분 정각 0~2초: 1분마다 통계 출력 (화면 덮어쓰기 없이 누적)
                    if 0 <= current_second <= 2:
                        if last_check_minute is not None and current_minute > last_check_minute:
                            
                            print("\n" + "-"*80)
                            print(f"🕐 {current_time.strftime('%Y-%m-%d %H:%M:%S')} - 실시간 상태")
                            if len(self.data_buffer) > 0:
                                print(f"💹 현재가: {self.data_buffer[-1]['close']:.2f}")
                            
                            self.print_statistics()
                            
                            time.sleep(1) 
                    
                except Exception as e:
                    print(f"❌ 업데이트 오류: {e}")
                    time.sleep(5) 
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 거래 중단")
            self.running = False
            self.print_statistics()

def main():
    """메인 실행 함수"""
    import sys
    
    # 1. 모델 경로 설정
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = 'model/rsi_price_action_v3_20251016_131305.pkl' 
    
    # 2. 임계값 오버라이드
    threshold_override = None
    if len(sys.argv) > 2:
        threshold_override = float(sys.argv[2])
    
    # 3. 역발상 모드 설정
    reverse_mode = False 
    if len(sys.argv) > 3:
        reverse_mode = sys.argv[3].lower() in ['true', '1', 'yes', 'reverse']
    
    try:
        system = StrategyTradingSystem(
            model_path, 
            threshold_override,
            reverse_mode=reverse_mode
        )
        
        if not system.fetch_initial_data('BTCUSDT'):
            print("API 연결 문제 또는 초기 데이터 로드 실패로 시스템을 종료합니다.")
            return
        
        system.run_realtime(symbol='BTCUSDT', check_interval=1)
        
    except Exception as e:
        print(f"❌ 시스템 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
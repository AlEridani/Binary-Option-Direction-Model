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

class BinaryOptionTradingSystem:
    def __init__(self, model_path):
        """실시간 거래 시스템 초기화"""
        print("="*80)
        print("🚀 바이너리 옵션 실시간 거래 시스템 (t+1 진입)")
        print("="*80)
        
        # 바이낸스 API 초기화
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        
        if not api_key or not api_secret:
            print("⚠️  API 키 없음 - 데모 모드로 실행")
            self.client = None
        else:
            self.client = Client(api_key, api_secret)
            print("✅ 바이낸스 API 연결 성공")
        
        # 모델 로드
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        
        print(f"\n📦 모델 로딩: {os.path.basename(model_path)}")
        self.model_package = joblib.load(model_path)
        self.long_model = self.model_package['long_model']
        self.short_model = self.model_package['short_model']
        self.feature_columns = self.model_package['feature_columns']
        self.config = self.model_package['config']
        
        print(f"✅ 모델 로드 완료")
        print(f"   룩백: {self.config['lookback']}분")
        print(f"   임계값: {self.config['threshold']}")
        print(f"   충돌모드: {self.config['conflict_mode']}")
        print(f"   옵션 기간: {self.config['option_duration']}분")
        
        # 데이터 버퍼
        self.lookback = self.config['lookback']
        self.option_duration = self.config['option_duration']
        buffer_size = self.lookback + self.option_duration + 100
        self.data_buffer = deque(maxlen=buffer_size)
        
        # 활성 거래
        self.active_trades = []
        
        # CSV 파일 설정
        self.csv_filename = f"trades_{datetime.now().strftime('%Y%m%d')}.csv"
        self._init_csv()
        
        # 통계
        self.stats = {
            'total': 0,
            'wins': 0,
            'losses': 0,
            'pending': 0,
            'long_trades': 0,
            'short_trades': 0
        }
        
        self.running = False
        
    def _init_csv(self):
        """CSV 파일 초기화"""
        if not os.path.exists(self.csv_filename):
            df = pd.DataFrame(columns=[
                'entry_time', 'exit_time', 'direction', 
                'entry_price', 'exit_price', 'result', 
                'probability', 'profit_pct'
            ])
            df.to_csv(self.csv_filename, index=False)
            print(f"📝 CSV 파일 생성: {self.csv_filename}")
        else:
            print(f"📝 기존 CSV 파일 사용: {self.csv_filename}")
    
    def fetch_initial_data(self, symbol='BTCUSDT'):
        """초기 데이터 로드"""
        print(f"\n📊 {symbol} 초기 데이터 로딩...")
        
        if self.client is None:
            print("⚠️  데모 모드 - 실제 데이터 없이 실행")
            return False
        
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval='1m',
                limit=self.lookback + 100
            )
            
            for kline in klines:
                candle = {
                    'timestamp': pd.to_datetime(kline[0], unit='ms'),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5])
                }
                self.data_buffer.append(candle)
            
            print(f"✅ {len(self.data_buffer)}개 캔들 로드 완료")
            return True
            
        except BinanceAPIException as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return False
    
    def create_features(self, df):
        """피처 생성 (학습 시와 동일)"""
        if len(df) < self.lookback:
            return None
        
        window = df.iloc[-self.lookback:]
        
        returns = (window['close'] / window['open'] - 1).values
        high_low_range = ((window['high'] - window['low']) / window['open']).values
        volume_changes = window['volume'].pct_change().fillna(0).values
        body = ((window['close'] - window['open']) / window['open']).values
        upper_shadow = ((window['high'] - window[['open', 'close']].max(axis=1)) / window['open']).values
        lower_shadow = ((window[['open', 'close']].min(axis=1) - window['low']) / window['open']).values
        
        recent_5 = returns[-5:]
        recent_10 = returns[-10:]
        recent_20 = returns[-20:]
        
        feature_dict = {}
        
        for j in range(15):
            feature_dict[f'ret_{j}'] = returns[-(j+1)]
            feature_dict[f'hl_range_{j}'] = high_low_range[-(j+1)]
            feature_dict[f'body_{j}'] = body[-(j+1)]
        
        for j in range(10):
            feature_dict[f'vol_chg_{j}'] = volume_changes[-(j+1)]
        
        feature_dict['ret_mean_5'] = recent_5.mean()
        feature_dict['ret_std_5'] = recent_5.std()
        feature_dict['ret_mean_10'] = recent_10.mean()
        feature_dict['ret_std_10'] = recent_10.std()
        feature_dict['ret_mean_all'] = returns.mean()
        feature_dict['ret_std_all'] = returns.std()
        
        consecutive_up = sum(1 for r in returns[::-1] if r > 0)
        consecutive_down = sum(1 for r in returns[::-1] if r < 0)
        feature_dict['consecutive_up'] = consecutive_up
        feature_dict['consecutive_down'] = consecutive_down
        feature_dict['up_ratio_5'] = (recent_5 > 0).sum() / 5
        feature_dict['up_ratio_10'] = (recent_10 > 0).sum() / 10
        
        current_price = window['close'].iloc[-1]
        high_max = window['high'].max()
        low_min = window['low'].min()
        
        feature_dict['price_position'] = (current_price - low_min) / (high_max - low_min) if high_max > low_min else 0.5
        feature_dict['dist_from_high'] = (high_max - current_price) / current_price
        feature_dict['dist_from_low'] = (current_price - low_min) / current_price
        feature_dict['volatility_5'] = recent_5.std()
        feature_dict['volatility_10'] = recent_10.std()
        feature_dict['volatility_ratio'] = recent_5.std() / (recent_10.std() + 1e-8)
        
        vol_mean_5 = window['volume'].iloc[-5:].mean()
        vol_mean_all = window['volume'].mean()
        feature_dict['volume_ratio_5'] = vol_mean_5 / (vol_mean_all + 1e-8)
        feature_dict['volume_trend'] = (window['volume'].iloc[-5:].mean() - window['volume'].iloc[-10:-5].mean()) / (window['volume'].iloc[-10:-5].mean() + 1e-8)
        feature_dict['momentum_5_20'] = recent_5.sum() - recent_20.sum()
        feature_dict['roc_5'] = (window['close'].iloc[-1] - window['close'].iloc[-6]) / window['close'].iloc[-6]
        feature_dict['body_mean_5'] = body[-5:].mean()
        feature_dict['upper_shadow_mean_5'] = upper_shadow[-5:].mean()
        feature_dict['lower_shadow_mean_5'] = lower_shadow[-5:].mean()
        feature_dict['body_to_range_ratio'] = abs(body[-1]) / (high_low_range[-1] + 1e-8)
        
        recent_highs = window['high'].iloc[-10:].values
        recent_lows = window['low'].iloc[-10:].values
        feature_dict['near_resistance'] = min([abs(current_price - h) / current_price for h in recent_highs])
        feature_dict['near_support'] = min([abs(current_price - l) / current_price for l in recent_lows])
        
        price_changes = window['close'].pct_change().iloc[-10:].values
        vol_changes_10 = window['volume'].pct_change().iloc[-10:].values
        corr = np.corrcoef(price_changes[1:], vol_changes_10[1:])[0, 1] if not np.isnan(price_changes[1:]).any() else 0
        feature_dict['price_volume_corr'] = corr if not np.isnan(corr) else 0
        
        feature_dict['current_ret'] = returns[-1]
        feature_dict['current_hl'] = high_low_range[-1]
        feature_dict['current_vol_chg'] = volume_changes[-1]
        
        return pd.Series(feature_dict)
    
    def check_entry_signal(self):
        """진입 신호 확인 (t 시점 종가로 판단)"""
        if len(self.data_buffer) < self.lookback + 10:
            return None
        
        df = pd.DataFrame(list(self.data_buffer))
        features = self.create_features(df)
        
        if features is None:
            return None
        
        try:
            X = features[self.feature_columns].values.reshape(1, -1)
            long_prob = self.long_model.predict(X)[0]
            short_prob = self.short_model.predict(X)[0]
            
            threshold = self.config['threshold']
            conflict_mode = self.config['conflict_mode']
            
            long_signal = long_prob > threshold
            short_signal = short_prob > threshold
            
            # 진입 로직
            signals = []
            
            if long_signal and not short_signal:
                signals.append({'direction': 'LONG', 'probability': long_prob})
            elif short_signal and not long_signal:
                signals.append({'direction': 'SHORT', 'probability': short_prob})
            elif long_signal and short_signal:
                if conflict_mode == 'higher':
                    if long_prob > short_prob:
                        signals.append({'direction': 'LONG', 'probability': long_prob})
                    else:
                        signals.append({'direction': 'SHORT', 'probability': short_prob})
                elif conflict_mode == 'both':
                    signals.append({'direction': 'LONG', 'probability': long_prob})
                    signals.append({'direction': 'SHORT', 'probability': short_prob})
            
            return signals if signals else None
            
        except Exception as e:
            print(f"❌ 신호 확인 오류: {e}")
            return None
    
    def wait_for_next_candle(self):
        """다음 봉 시작까지 대기"""
        now = datetime.now()
        next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
        wait_seconds = (next_minute - now).total_seconds()
        
        if wait_seconds > 0:
            print(f"⏳ 다음 봉 시작까지 {wait_seconds:.1f}초 대기...")
            time.sleep(wait_seconds + 0.5)  # 0.5초 여유
    
    def get_next_candle_open(self, symbol='BTCUSDT'):
        """다음 봉의 open 가격 가져오기"""
        if self.client is None:
            return None
        
        try:
            klines = self.client.get_klines(symbol=symbol, interval='1m', limit=1)
            return float(klines[0][1])  # open 가격
        except Exception as e:
            print(f"❌ Open 가격 조회 실패: {e}")
            return None
    
    def enter_trade(self, signal, entry_price):
        """거래 진입 (t+1 open 가격)"""
        entry_time = datetime.now()
        exit_time = entry_time + timedelta(minutes=self.option_duration)
        
        trade = {
            'direction': signal['direction'],
            'entry_time': entry_time,
            'exit_time': exit_time,
            'entry_price': entry_price,
            'exit_price': None,
            'probability': signal['probability'],
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
        
        print(f"\n📈 {'롱' if signal['direction'] == 'LONG' else '숏'} 진입 (t+1)")
        print(f"   진입 시간: {entry_time.strftime('%H:%M:%S')}")
        print(f"   진입가: {entry_price:.2f}")
        print(f"   확률: {signal['probability']:.2%}")
        print(f"   종료 예정: {exit_time.strftime('%H:%M:%S')}")
    
    def check_exits(self):
        """만료된 거래 확인 및 청산 (t+11 close 가격)"""
        if len(self.data_buffer) == 0:
            return
        
        current_time = datetime.now()
        completed_indices = []
        
        for idx, trade in enumerate(self.active_trades):
            if current_time >= trade['exit_time']:
                # t+11 close 가격 사용
                exit_price = self.data_buffer[-1]['close']
                trade['exit_price'] = exit_price
                trade['status'] = 'COMPLETED'
                
                # 결과 판정
                if trade['direction'] == 'LONG':
                    trade['result'] = 'WIN' if exit_price > trade['entry_price'] else 'LOSS'
                else:
                    trade['result'] = 'WIN' if exit_price < trade['entry_price'] else 'LOSS'
                
                # 수익률 계산
                profit_pct = ((exit_price - trade['entry_price']) / trade['entry_price']) * 100
                if trade['direction'] == 'SHORT':
                    profit_pct = -profit_pct
                
                trade['profit_pct'] = profit_pct
                
                # 통계 업데이트
                self.stats['pending'] -= 1
                if trade['result'] == 'WIN':
                    self.stats['wins'] += 1
                else:
                    self.stats['losses'] += 1
                
                # CSV 저장
                self._save_trade_to_csv(trade)
                
                # 결과 출력
                emoji = '✅' if trade['result'] == 'WIN' else '❌'
                print(f"\n{emoji} 거래 종료 - {trade['direction']}")
                print(f"   진입: {trade['entry_price']:.2f} → 종료: {exit_price:.2f}")
                print(f"   변화: {profit_pct:+.2f}%")
                print(f"   결과: {trade['result']}")
                
                completed_indices.append(idx)
        
        # 완료된 거래 제거
        for idx in reversed(completed_indices):
            self.active_trades.pop(idx)
    
    def _save_trade_to_csv(self, trade):
        """거래 내역 CSV 저장"""
        df = pd.DataFrame([{
            'entry_time': trade['entry_time'].strftime('%Y-%m-%d %H:%M:%S'),
            'exit_time': trade['exit_time'].strftime('%Y-%m-%d %H:%M:%S'),
            'direction': trade['direction'],
            'entry_price': trade['entry_price'],
            'exit_price': trade['exit_price'],
            'result': trade['result'],
            'probability': trade['probability'],
            'profit_pct': f"{trade['profit_pct']:+.2f}%"
        }])
        
        df.to_csv(self.csv_filename, mode='a', header=False, index=False)
    
    def print_statistics(self):
        """통계 출력"""
        print("\n" + "="*80)
        print("📊 실시간 거래 통계")
        print("="*80)
        
        if self.stats['total'] > 0:
            completed = self.stats['wins'] + self.stats['losses']
            win_rate = (self.stats['wins'] / completed * 100) if completed > 0 else 0
            
            expected_return = (win_rate/100 * 0.80) - ((100-win_rate)/100 * 1)
            
            print(f"총 거래: {self.stats['total']} (완료: {completed}, 대기: {self.stats['pending']})")
            
            if completed > 0:
                print(f"승리: {self.stats['wins']} | 패배: {self.stats['losses']}")
                print(f"승률: {win_rate:.1f}%")
                print(f"롱/숏: {self.stats['long_trades']}/{self.stats['short_trades']}")
                print(f"기대 수익률: {expected_return*100:+.1f}% per trade")
                print(f"손익분기: 55.56% | 차이: {win_rate - 55.56:+.2f}%p")
                
                if win_rate >= 55.56:
                    print(f"✅ 수익성 있음")
                else:
                    print(f"⚠️  손익분기 미달")
        else:
            print("거래 기록 없음")
    
    def run_realtime(self, symbol='BTCUSDT', check_interval=5):
        """실시간 거래 실행"""
        print(f"\n{'='*80}")
        print(f"🚀 실시간 거래 시작 (t+1 진입 방식)")
        print(f"{'='*80}")
        print(f"심볼: {symbol}")
        print(f"체크 간격: {check_interval}초")
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
                    
                    # 최신 데이터 업데이트
                    if self.client:
                        klines = self.client.get_klines(symbol=symbol, interval='1m', limit=2)
                        latest_candle = {
                            'timestamp': pd.to_datetime(klines[-1][0], unit='ms'),
                            'open': float(klines[-1][1]),
                            'high': float(klines[-1][2]),
                            'low': float(klines[-1][3]),
                            'close': float(klines[-1][4]),
                            'volume': float(klines[-1][5])
                        }
                        
                        if len(self.data_buffer) == 0 or self.data_buffer[-1]['timestamp'] < latest_candle['timestamp']:
                            self.data_buffer.append(latest_candle)
                    
                    # 매 분 정시에 신호 확인
                    if last_check_minute is None or current_minute > last_check_minute:
                        # t 시점 종가로 신호 판단
                        pending_signals = self.check_entry_signal()
                        
                        if pending_signals:
                            print(f"\n🔔 신호 발생! {len(pending_signals)}개")
                            # t+1까지 대기
                            self.wait_for_next_candle()
                            
                            # t+1 open 가격으로 진입
                            entry_price = self.get_next_candle_open(symbol)
                            if entry_price:
                                for signal in pending_signals:
                                    self.enter_trade(signal, entry_price)
                            
                            pending_signals = None
                        
                        last_check_minute = current_minute
                    
                    # 종료된 거래 확인
                    self.check_exits()
                    
                    # 30초마다 화면 업데이트
                    if current_time.second % 30 == 0:
                        os.system('cls' if os.name == 'nt' else 'clear')
                        print(f"🕐 {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        if len(self.data_buffer) > 0:
                            print(f"💹 현재가: {self.data_buffer[-1]['close']:.2f}")
                        self.print_statistics()
                    
                except Exception as e:
                    print(f"❌ 업데이트 오류: {e}")
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 거래 중단")
            self.running = False
            self.print_statistics()

def main():
    """메인 실행 함수"""
    import sys
    
    # 모델 파일 경로 (인자로 받거나 기본값 사용)
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = 'model/binary_option_FINAL_20251012_143910.pkl'
    
    try:
        system = BinaryOptionTradingSystem(model_path)
        
        if system.client:
            if not system.fetch_initial_data('BTCUSDT'):
                print("초기 데이터 로드 실패")
                return
        
        system.run_realtime(symbol='BTCUSDT', check_interval=5)
        
    except Exception as e:
        print(f"❌ 시스템 오류: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
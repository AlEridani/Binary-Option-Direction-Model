# real_trade.py - 실시간 거래 및 검증 모듈

import pandas as pd
import numpy as np
import json
import time
import os
from datetime import datetime, timedelta
import requests
import uuid
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class BinanceAPIClient:
    """바이낸스 API 클라이언트 (시뮬레이션)"""
    
    def __init__(self, api_key=None, api_secret=None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = "https://api.binance.com"
        
    def get_current_price(self, symbol="BTCUSDT"):
        """
        현재 가격 조회
        """
        try:
            url = f"{self.base_url}/api/v3/ticker/price"
            params = {"symbol": symbol}
            response = requests.get(url, params=params)
            data = response.json()
            return float(data['price'])
        except:
            # 시뮬레이션 모드
            return np.random.uniform(40000, 45000)
    
    def get_klines(self, symbol="BTCUSDT", interval="1m", limit=500):
        """
        캔들스틱 데이터 조회
        """
        try:
            url = f"{self.base_url}/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": limit
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            # DataFrame으로 변환
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # 타입 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except:
            # 시뮬레이션 데이터
            return self._generate_simulation_data(limit)
    
    def _generate_simulation_data(self, limit=500):
        """
        시뮬레이션용 데이터 생성
        """
        timestamps = pd.date_range(end=datetime.now(), periods=limit, freq='1min')
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


class RealTimeTrader:
    """실시간 거래 실행 클래스"""
    
    def __init__(self, config, model_trainer, api_client=None):
        self.config = config
        self.model_trainer = model_trainer
        self.api_client = api_client or BinanceAPIClient()
        
        # 거래 상태
        self.is_running = False
        self.current_position = None
        self.trade_history = deque(maxlen=self.config.EVALUATION_WINDOW)
        self.performance_metrics = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'current_streak': 0,
            'max_streak': 0,
            'total_profit': 0
        }
        
        # 필터
        self.trade_filters = {}
        self.load_filters()
        
    def load_filters(self):
        """
        거래 필터 로드
        """
        filter_path = os.path.join(self.config.FEATURE_LOG_DIR, 'trade_filters.json')
        if os.path.exists(filter_path):
            with open(filter_path, 'r') as f:
                self.trade_filters = json.load(f)
                print(f"거래 필터 로드: {len(self.trade_filters)} 개")
    
    def save_filters(self):
        """
        거래 필터 저장
        """
        filter_path = os.path.join(self.config.FEATURE_LOG_DIR, 'trade_filters.json')
        with open(filter_path, 'w') as f:
            json.dump(self.trade_filters, f, indent=2)
    
    def apply_filters(self, features):
        """
        거래 필터 적용
        """
        # 기본적으로 거래 허용
        should_trade = True
        filter_reasons = []
        
        # 변동성 필터
        if 'high_volatility' in self.trade_filters:
            filter_config = self.trade_filters['high_volatility']
            if 'atr_14' in features and features['atr_14'].iloc[-1] > filter_config.get('atr_14_threshold', float('inf')):
                should_trade = False
                filter_reasons.append("높은 변동성")
        
        # 거래량 필터
        if 'low_volume' in self.trade_filters:
            filter_config = self.trade_filters['low_volume']
            if 'volume_ratio' in features and features['volume_ratio'].iloc[-1] < filter_config.get('volume_ratio_threshold', 0):
                should_trade = False
                filter_reasons.append("낮은 거래량")
        
        # 시간대 필터
        if 'time_based' in self.trade_filters:
            filter_config = self.trade_filters['time_based']
            current_hour = datetime.now().hour
            if current_hour in filter_config.get('avoid_hours', []):
                should_trade = False
                filter_reasons.append(f"제외 시간대: {current_hour}시")
        
        return should_trade, filter_reasons
    
    def prepare_features(self, df):
        """
        실시간 피처 준비
        """
        from model_train import FeatureEngineer
        
        feature_engineer = FeatureEngineer()
        features = feature_engineer.create_feature_pool(df)
        
        # 마지막 행만 필요 (현재 시점)
        return features
    
    def make_prediction(self, features):
        """
        예측 수행
        """
        # 모델이 로드되어 있는지 확인
        if not self.model_trainer.models:
            print("모델이 로드되지 않았습니다.")
            return None, 0.5
        
        # 필터 적용
        should_trade, filter_reasons = self.apply_filters(features)
        
        if not should_trade:
            print(f"거래 필터에 의해 스킵: {', '.join(filter_reasons)}")
            return None, 0.5
        
        # 예측 수행
        try:
            # 마지막 행으로 예측
            X_current = features.iloc[[-1]]
            pred_proba = self.model_trainer.predict_proba(X_current)[0]
            prediction = 1 if pred_proba > 0.5 else 0
            
            return prediction, pred_proba
        except Exception as e:
            print(f"예측 오류: {e}")
            return None, 0.5
    
    def execute_trade(self, prediction, confidence, amount=100):
        """
        거래 실행
        """
        trade_id = str(uuid.uuid4())[:8]
        entry_time = datetime.now()
        entry_price = self.api_client.get_current_price()
        
        trade_info = {
            'trade_id': trade_id,
            'entry_time': entry_time.isoformat(),
            'entry_price': entry_price,
            'prediction': prediction,  # 1: UP, 0: DOWN
            'confidence': confidence,
            'amount': amount,
            'status': 'open'
        }
        
        # 포지션 설정
        self.current_position = trade_info
        
        # 로그 저장
        self.save_trade_log(trade_info)
        
        direction = "UP" if prediction == 1 else "DOWN"
        print(f"\n거래 실행: {trade_id}")
        print(f"- 방향: {direction}")
        print(f"- 신뢰도: {confidence:.2%}")
        print(f"- 진입가: {entry_price:.2f}")
        print(f"- 금액: ${amount}")
        
        return trade_id
    
    def check_trade_result(self, trade_id):
        """
        거래 결과 확인 (10분 후)
        """
        if self.current_position is None or self.current_position['trade_id'] != trade_id:
            return None
        
        # 진입 시간 확인
        entry_time = datetime.fromisoformat(self.current_position['entry_time'])
        current_time = datetime.now()
        
        # 10분 경과 확인
        if (current_time - entry_time).total_seconds() < 600:
            remaining = 600 - (current_time - entry_time).total_seconds()
            print(f"거래 진행 중... {remaining:.0f}초 남음")
            return None
        
        # 결과 확인
        entry_price = self.current_position['entry_price']
        exit_price = self.api_client.get_current_price()
        
        actual_direction = 1 if exit_price > entry_price else 0
        prediction = self.current_position['prediction']
        
        # 승패 판정
        is_win = (prediction == actual_direction)
        amount = self.current_position['amount']
        
        if is_win:
            profit = amount * self.config.WIN_RATE  # 80% 수익
            result = 1
        else:
            profit = -amount  # 전액 손실
            result = 0
        
        # 결과 업데이트
        self.current_position['exit_time'] = current_time.isoformat()
        self.current_position['exit_price'] = exit_price
        self.current_position['result'] = result
        self.current_position['profit_loss'] = profit
        self.current_position['status'] = 'closed'
        
        # 통계 업데이트
        self.update_performance(is_win, profit)
        
        # 거래 기록에 추가
        self.trade_history.append(result)
        
        # 로그 업데이트
        self.update_trade_log(trade_id, result, profit)
        
        print(f"\n거래 종료: {trade_id}")
        print(f"- 결과: {'승' if is_win else '패'}")
        print(f"- 진입가: {entry_price:.2f}")
        print(f"- 종료가: {exit_price:.2f}")
        print(f"- 손익: ${profit:.2f}")
        
        # 포지션 초기화
        self.current_position = None
        
        return result
    
    def update_performance(self, is_win, profit):
        """
        성능 통계 업데이트
        """
        self.performance_metrics['total_trades'] += 1
        
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
        
        # 승률 계산
        if self.performance_metrics['total_trades'] > 0:
            win_rate = self.performance_metrics['wins'] / self.performance_metrics['total_trades']
            self.performance_metrics['win_rate'] = win_rate
    
    def check_retrain_needed(self):
        """
        재학습 필요 여부 확인
        """
        if len(self.trade_history) >= self.config.EVALUATION_WINDOW:
            recent_win_rate = sum(self.trade_history) / len(self.trade_history)
            
            print(f"\n최근 {self.config.EVALUATION_WINDOW}건 거래 승률: {recent_win_rate:.2%}")
            
            if recent_win_rate < self.config.RETRAIN_THRESHOLD:
                print(f"승률이 {self.config.RETRAIN_THRESHOLD:.0%} 미만입니다. 재학습이 필요합니다.")
                return True
            
        return False
    
    def save_trade_log(self, trade_info):
        """
        거래 로그 저장
        """
        log_path = os.path.join(self.config.TRADE_LOG_DIR, 'trades.csv')
        
        # DataFrame 생성
        df_new = pd.DataFrame([trade_info])
        
        # 기존 파일이 있으면 추가
        if os.path.exists(log_path):
            df_existing = pd.read_csv(log_path)
            df = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df = df_new
        
        # 저장
        df.to_csv(log_path, index=False)
    
    def update_trade_log(self, trade_id, result, profit_loss):
        """
        거래 결과 업데이트
        """
        from data_merge import DataMerger
        merger = DataMerger(self.config)
        merger.update_trade_result(trade_id, result, profit_loss)
    
    def save_feature_log(self, features, trade_id):
        """
        거래 시점의 피처 저장
        """
        # 마지막 행 (현재 시점) 피처만 저장
        current_features = features.iloc[[-1]].copy()
        current_features['trade_id'] = trade_id
        current_features['timestamp'] = datetime.now()
        
        # 파일명
        today = datetime.now().strftime("%Y%m%d")
        log_path = os.path.join(self.config.FEATURE_LOG_DIR, f'features_{today}.csv')
        
        # 기존 파일이 있으면 추가
        if os.path.exists(log_path):
            df_existing = pd.read_csv(log_path)
            df = pd.concat([df_existing, current_features], ignore_index=True)
        else:
            df = current_features
        
        # 저장
        df.to_csv(log_path, index=False)
    
    def print_performance_summary(self):
        """
        성능 요약 출력
        """
        print("\n" + "="*50)
        print("실시간 거래 성능 요약")
        print("="*50)
        
        metrics = self.performance_metrics
        print(f"총 거래 수: {metrics['total_trades']}")
        print(f"승: {metrics['wins']} / 패: {metrics['losses']}")
        
        if metrics['total_trades'] > 0:
            win_rate = metrics.get('win_rate', 0)
            print(f"승률: {win_rate:.2%}")
        
        print(f"현재 연승: {metrics['current_streak']}")
        print(f"최대 연승: {metrics['max_streak']}")
        print(f"총 손익: ${metrics['total_profit']:.2f}")
        
        if metrics['total_trades'] > 0:
            avg_profit = metrics['total_profit'] / metrics['total_trades']
            print(f"평균 손익: ${avg_profit:.2f}")
        
        print("="*50)
    
    def run_live_trading(self, duration_hours=1, trade_interval_minutes=11):
        """
        실시간 거래 실행
        """
        print(f"\n실시간 거래 시작 (기간: {duration_hours}시간)")
        print("="*50)
        
        self.is_running = True
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=duration_hours)
        
        # 모델 로드
        if not self.model_trainer.models:
            print("모델 로딩 중...")
            success = self.model_trainer.load_model()
            if not success:
                print("모델을 찾을 수 없습니다. 먼저 학습을 진행하세요.")
                return
        
        last_trade_time = None
        
        try:
            while datetime.now() < end_time and self.is_running:
                current_time = datetime.now()
                
                # 현재 포지션이 없고, 거래 간격이 지났으면 새 거래
                if self.current_position is None:
                    if last_trade_time is None or \
                       (current_time - last_trade_time).total_seconds() >= trade_interval_minutes * 60:
                        
                        # 최신 데이터 가져오기
                        print(f"\n[{current_time.strftime('%H:%M:%S')}] 데이터 수집 중...")
                        df = self.api_client.get_klines(limit=500)
                        
                        # 피처 준비
                        features = self.prepare_features(df)
                        
                        # 예측
                        prediction, confidence = self.make_prediction(features)
                        
                        if prediction is not None:
                            # 거래 실행
                            trade_id = self.execute_trade(prediction, confidence)
                            
                            # 피처 로그 저장
                            self.save_feature_log(features, trade_id)
                            
                            last_trade_time = current_time
                
                # 진행 중인 거래 확인
                elif self.current_position is not None:
                    result = self.check_trade_result(self.current_position['trade_id'])
                    
                    if result is not None:
                        # 재학습 필요 확인
                        if self.check_retrain_needed():
                            print("\n재학습을 시작합니다...")
                            # 여기서 main_pipe의 재학습 트리거
                            self.trigger_retrain()
                        
                        # 성능 요약 출력
                        if self.performance_metrics['total_trades'] % 10 == 0:
                            self.print_performance_summary()
                
                # 대기
                time.sleep(10)
                
        except KeyboardInterrupt:
            print("\n거래 중단...")
        finally:
            self.is_running = False
            self.print_performance_summary()
    
    def trigger_retrain(self):
        """
        재학습 트리거
        """
        # 재학습 플래그 파일 생성
        flag_path = os.path.join(self.config.BASE_DIR, '.retrain_required')
        with open(flag_path, 'w') as f:
            f.write(datetime.now().isoformat())
        
        print("재학습 플래그 설정 완료")
    
    def backtest(self, historical_data, start_date=None, end_date=None):
        """
        백테스팅
        """
        print("\n백테스팅 시작...")
        
        # 데이터 필터링
        if start_date:
            historical_data = historical_data[historical_data['timestamp'] >= start_date]
        if end_date:
            historical_data = historical_data[historical_data['timestamp'] <= end_date]
        
        # 피처 생성
        from model_train import FeatureEngineer
        feature_engineer = FeatureEngineer()
        features = feature_engineer.create_feature_pool(historical_data)
        target = feature_engineer.create_target(historical_data, window=self.config.PREDICTION_WINDOW)
        
        # 유효 데이터만 사용
        valid_idx = target.notna()
        features = features[valid_idx]
        target = target[valid_idx]
        
        # 거래 시뮬레이션
        trades = []
        for i in range(len(features) - self.config.PREDICTION_WINDOW):
            # 현재 시점까지의 데이터로 예측
            X_current = features.iloc[[i]]
            
            # 예측
            pred_proba = self.model_trainer.predict_proba(X_current)[0]
            prediction = 1 if pred_proba > 0.5 else 0
            
            # 실제 결과
            actual = target.iloc[i]
            
            # 거래 기록
            trades.append({
                'timestamp': historical_data['timestamp'].iloc[i],
                'prediction': prediction,
                'actual': actual,
                'confidence': pred_proba,
                'correct': prediction == actual
            })
        
        # 결과 분석
        trades_df = pd.DataFrame(trades)
        
        total_trades = len(trades_df)
        correct_trades = trades_df['correct'].sum()
        win_rate = correct_trades / total_trades if total_trades > 0 else 0
        
        # 수익 계산
        wins = trades_df['correct'].sum()
        losses = (~trades_df['correct']).sum()
        profit = (wins * 100 * self.config.WIN_RATE) - (losses * 100)
        
        print(f"\n백테스팅 결과:")
        print(f"- 기간: {trades_df['timestamp'].min()} ~ {trades_df['timestamp'].max()}")
        print(f"- 총 거래: {total_trades}")
        print(f"- 승: {wins} / 패: {losses}")
        print(f"- 승률: {win_rate:.2%}")
        print(f"- 총 손익: ${profit:.2f}")
        print(f"- 평균 손익: ${profit/total_trades if total_trades > 0 else 0:.2f}")
        
        return trades_df


class TradingMonitor:
    """거래 모니터링 클래스"""
    
    def __init__(self, config):
        self.config = config
    
    def analyze_recent_trades(self, days=7):
        """
        최근 거래 분석
        """
        from data_merge import DataMerger
        merger = DataMerger(self.config)
        trades = merger.load_trade_logs()
        
        if trades.empty:
            print("거래 기록이 없습니다.")
            return None
        
        # 최근 N일 필터링
        cutoff = datetime.now() - timedelta(days=days)
        if 'entry_time' in trades.columns:
            trades['entry_time'] = pd.to_datetime(trades['entry_time'])
            recent_trades = trades[trades['entry_time'] >= cutoff]
        else:
            recent_trades = trades
        
        if recent_trades.empty:
            print(f"최근 {days}일간 거래 기록이 없습니다.")
            return None
        
        # 통계 계산
        stats = {
            'total_trades': len(recent_trades),
            'wins': (recent_trades['result'] == 1).sum() if 'result' in recent_trades.columns else 0,
            'losses': (recent_trades['result'] == 0).sum() if 'result' in recent_trades.columns else 0,
            'total_profit': recent_trades['profit_loss'].sum() if 'profit_loss' in recent_trades.columns else 0
        }
        
        if stats['total_trades'] > 0:
            stats['win_rate'] = stats['wins'] / stats['total_trades']
            stats['avg_profit'] = stats['total_profit'] / stats['total_trades']
        
        # 시간대별 분석
        if 'entry_time' in recent_trades.columns:
            recent_trades['hour'] = recent_trades['entry_time'].dt.hour
            hourly_stats = recent_trades.groupby('hour').agg({
                'result': ['count', 'mean']
            }).round(3)
            stats['hourly_performance'] = hourly_stats
        
        return stats
    
    def generate_report(self):
        """
        종합 리포트 생성
        """
        print("\n" + "="*60)
        print("거래 시스템 종합 리포트")
        print("="*60)
        
        # 7일 통계
        week_stats = self.analyze_recent_trades(7)
        if week_stats:
            print("\n[최근 7일 성과]")
            print(f"총 거래: {week_stats['total_trades']}")
            print(f"승/패: {week_stats['wins']}/{week_stats['losses']}")
            print(f"승률: {week_stats.get('win_rate', 0):.2%}")
            print(f"총 손익: ${week_stats['total_profit']:.2f}")
            print(f"평균 손익: ${week_stats.get('avg_profit', 0):.2f}")
        
        # 30일 통계
        month_stats = self.analyze_recent_trades(30)
        if month_stats:
            print("\n[최근 30일 성과]")
            print(f"총 거래: {month_stats['total_trades']}")
            print(f"승/패: {month_stats['wins']}/{month_stats['losses']}")
            print(f"승률: {month_stats.get('win_rate', 0):.2%}")
            print(f"총 손익: ${month_stats['total_profit']:.2f}")
            print(f"평균 손익: ${month_stats.get('avg_profit', 0):.2f}")
        
        print("\n" + "="*60)


# 사용 예시
if __name__ == "__main__":
    from config import Config
    from model_train import ModelTrainer
    
    # 설정 초기화
    Config.create_directories()
    
    # 모델 로드
    trainer = ModelTrainer(Config)
    trainer.load_model()
    
    # 실시간 거래 시작
    trader = RealTimeTrader(Config, trainer)
    
    # 백테스팅 (옵션)
    # trader.backtest(historical_data, start_date, end_date)
    
    # 실시간 거래 실행
    trader.run_live_trading(duration_hours=1, trade_interval_minutes=11)
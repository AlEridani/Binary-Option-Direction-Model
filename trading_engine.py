"""
trading_engine.py
30분봉 트레이딩 엔진
- 히스테리시스 (진입 0.6, 해제 0.58)
- TTL (6-9분)
- Δp 최소값 체크
- Refractory Period (30분)
- 50번 거래마다 승률 체크 → 재학습 트리거
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Tuple
import hashlib
import json

from config import Config
from timeframe_manager import TimeframeManager


class TradingState:
    """거래 상태 관리"""
    
    def __init__(self):
        # 히스테리시스 상태
        self.locked = False  # 진입 대기 상태
        self.lock_time = None  # 잠금 시각
        self.lock_probability = None  # 잠금 시 확률
        self.lock_regime = None  # 잠금 시 레짐
        self.lock_side = None  # 잠금 시 방향 (LONG/SHORT)
        
        # TTL
        self.ttl_start = None  # TTL 시작 시각
        self.ttl_seconds = None  # TTL 초
        
        # Refractory Period (재진입 금지)
        self.last_entry_time = {}  # {(symbol, side): timestamp}
        
        # 이전 확률 (Δp 계산용)
        self.prev_probability = None
    
    def reset(self):
        """상태 초기화"""
        self.locked = False
        self.lock_time = None
        self.lock_probability = None
        self.lock_regime = None
        self.lock_side = None
        self.ttl_start = None
        self.ttl_seconds = None
        self.prev_probability = None
    
    def is_in_refractory(self, symbol: str, side: str, current_time: datetime) -> bool:
        """Refractory Period 체크"""
        key = (symbol, side)
        if key not in self.last_entry_time:
            return False
        
        last_time = self.last_entry_time[key]
        elapsed = (current_time - last_time).total_seconds() / 60.0  # 분 단위
        
        return elapsed < Config.REFRACTORY_MINUTES
    
    def update_last_entry(self, symbol: str, side: str, current_time: datetime):
        """마지막 진입 시각 업데이트"""
        key = (symbol, side)
        self.last_entry_time[key] = current_time


class TradingEngine:
    """30분봉 트레이딩 엔진"""
    
    def __init__(self, model_trainer, symbol='BTCUSDT'):
        self.model = model_trainer
        self.symbol = symbol
        self.state = TradingState()
        self.tf_manager = TimeframeManager()
        
        # 설정
        self.cut_on = Config.CUT_ON
        self.cut_off = Config.CUT_OFF
        self.dp_min = Config.DP_MIN
        self.ttl_min = Config.TTL_MIN_SECONDS
        self.ttl_max = Config.TTL_MAX_SECONDS
        self.refractory_minutes = Config.REFRACTORY_MINUTES
    
    # ==========================================
    # 히스테리시스 로직
    # ==========================================
    
    def check_entry_signal(
        self, 
        current_probability: float,
        regime: int,
        current_time: datetime,
        features: pd.DataFrame
    ) -> Dict:
        """
        진입 신호 체크 (히스테리시스 + TTL + Δp + Refractory)
        
        Parameters:
        -----------
        current_probability : float
            현재 예측 확률 (캘리브레이션 적용 후)
        regime : int
            현재 레짐 (1: UP, -1: DOWN, 0: FLAT)
        current_time : datetime
            현재 시각
        features : DataFrame
            현재 피처 (필터 체크용)
        
        Returns:
        --------
        dict: {
            'should_enter': bool,
            'side': str (LONG/SHORT),
            'probability': float,
            'reason': str,
            'blocked_reason': str or None
        }
        """
        result = {
            'should_enter': False,
            'side': None,
            'probability': current_probability,
            'reason': None,
            'blocked_reason': None
        }
        
        # ======================================
        # 1. 히스테리시스 상태 체크
        # ======================================
        if not self.state.locked:
            # 잠금 안된 상태 → 컷온 체크
            if current_probability >= self.cut_on:
                # 상향 돌파 → 잠금
                self.state.locked = True
                self.state.lock_time = current_time
                self.state.lock_probability = current_probability
                self.state.lock_regime = regime
                self.state.lock_side = 'LONG'
                
                # TTL 랜덤 설정
                self.state.ttl_seconds = np.random.randint(self.ttl_min, self.ttl_max + 1)
                self.state.ttl_start = current_time
                
                result['reason'] = 'LOCKED_UP'
                print(f"  🔒 잠금 (상향): p={current_probability:.4f}, TTL={self.state.ttl_seconds}초")
                
            elif current_probability <= (1 - self.cut_on):
                # 하향 돌파 → 잠금 (SHORT)
                self.state.locked = True
                self.state.lock_time = current_time
                self.state.lock_probability = current_probability
                self.state.lock_regime = regime
                self.state.lock_side = 'SHORT'
                
                # TTL 랜덤 설정
                self.state.ttl_seconds = np.random.randint(self.ttl_min, self.ttl_max + 1)
                self.state.ttl_start = current_time
                
                result['reason'] = 'LOCKED_DOWN'
                print(f"  🔒 잠금 (하향): p={current_probability:.4f}, TTL={self.state.ttl_seconds}초")
            
            # 이전 확률 저장
            self.state.prev_probability = current_probability
            return result
        
        # ======================================
        # 2. 잠금된 상태 → 해제 또는 진입 체크
        # ======================================
        
        # 2-1. TTL 만료 체크
        elapsed = (current_time - self.state.ttl_start).total_seconds()
        if elapsed > self.state.ttl_seconds:
            result['blocked_reason'] = 'TTL_EXPIRED'
            print(f"  ⏱️ TTL 만료 ({elapsed:.1f}초 > {self.state.ttl_seconds}초) → 해제")
            self.state.reset()
            self.state.prev_probability = current_probability
            return result
        
        # 2-2. 컷오프 체크 (해제)
        if self.state.lock_side == 'LONG':
            if current_probability < self.cut_off:
                result['blocked_reason'] = 'CUTOFF_BREACHED'
                print(f"  ⬇️ 컷오프 하향 돌파 (p={current_probability:.4f} < {self.cut_off}) → 해제")
                self.state.reset()
                self.state.prev_probability = current_probability
                return result
        elif self.state.lock_side == 'SHORT':
            if current_probability > (1 - self.cut_off):
                result['blocked_reason'] = 'CUTOFF_BREACHED'
                print(f"  ⬆️ 컷오프 상향 돌파 (p={current_probability:.4f} > {1-self.cut_off:.4f}) → 해제")
                self.state.reset()
                self.state.prev_probability = current_probability
                return result
        
        # 2-3. Δp 체크 (모멘텀)
        if self.state.prev_probability is not None:
            dp = abs(current_probability - self.state.prev_probability)
            if dp < self.dp_min:
                result['blocked_reason'] = 'DP_TOO_SMALL'
                self.state.prev_probability = current_probability
                return result
        
        # 2-4. Refractory Period 체크
        if self.state.is_in_refractory(self.symbol, self.state.lock_side, current_time):
            result['blocked_reason'] = 'REFRACTORY_PERIOD'
            self.state.prev_probability = current_probability
            return result
        
        # 2-5. 필터 체크 (추가 가능)
        # TODO: adaptive_filters 적용
        
        # ======================================
        # 3. 모든 조건 통과 → 진입!
        # ======================================
        result['should_enter'] = True
        result['side'] = self.state.lock_side
        result['probability'] = current_probability
        result['reason'] = 'ENTRY_CONFIRMED'
        
        # 재진입 방지 업데이트
        self.state.update_last_entry(self.symbol, self.state.lock_side, current_time)
        
        # 상태 리셋
        lock_side = self.state.lock_side
        self.state.reset()
        self.state.prev_probability = current_probability
        
        print(f"  ✅ 진입 확정: {lock_side}, p={current_probability:.4f}, elapsed={elapsed:.1f}초")
        
        return result
    
    # ==========================================
    # 30분봉 단위 판단
    # ==========================================
    
    def decide_on_bar_close(
        self,
        df_30m: pd.DataFrame,
        bar_index: int
    ) -> Dict:
        """
        30분봉 close 시점에서 판단
        
        Parameters:
        -----------
        df_30m : DataFrame
            30분봉 데이터 (피처 포함)
        bar_index : int
            현재 바 인덱스
        
        Returns:
        --------
        dict: {
            'should_enter': bool,
            'side': str or None,
            'bar30_start': datetime,
            'bar30_end': datetime,
            'entry_ts': datetime,
            'label_ts': datetime,
            'probability': float,
            'regime': int,
            'reason': str,
            'blocked_reason': str or None
        }
        """
        # 현재 바 정보
        current_bar = df_30m.iloc[bar_index]
        bar30_start = pd.to_datetime(current_bar['bar30_start'], utc=True)
        bar30_end = pd.to_datetime(current_bar['bar30_end'], utc=True)
        
        # 진입/만기 시각
        entry_ts = bar30_end  # t+1 바의 open
        label_ts = bar30_end + pd.Timedelta(minutes=30)  # t+1 바의 close
        
        # 레짐
        regime = int(current_bar.get('regime', 0))
        
        # 피처 추출 (예측용)
        feature_cols = [c for c in df_30m.columns 
                        if c not in ['bar30_start', 'bar30_end', 'm1_index_entry', 'm1_index_label',
                                      'timestamp', 'regime', 'regime_final', 'is_weekend']]
        
        features = current_bar[feature_cols].to_frame().T
        
        # 예측
        try:
            probability = self.model.predict_proba_df(features, regime=regime)
            if isinstance(probability, np.ndarray):
                probability = float(probability[0])
            else:
                probability = float(probability)
        except Exception as e:
            print(f"  ⚠️ 예측 실패: {e}")
            probability = 0.5
        
        # 히스테리시스 체크
        signal = self.check_entry_signal(
            current_probability=probability,
            regime=regime,
            current_time=bar30_end,
            features=features
        )
        
        # 결과 구성
        result = {
            'should_enter': signal['should_enter'],
            'side': signal['side'],
            'bar30_start': bar30_start,
            'bar30_end': bar30_end,
            'entry_ts': entry_ts,
            'label_ts': label_ts,
            'probability': probability,
            'regime': regime,
            'reason': signal['reason'],
            'blocked_reason': signal['blocked_reason']
        }
        
        return result
    
    # ==========================================
    # 백테스트 루프
    # ==========================================
    
    def backtest(self, df_30m: pd.DataFrame) -> pd.DataFrame:
        """
        백테스트 실행
        
        Parameters:
        -----------
        df_30m : DataFrame
            30분봉 데이터 (피처 포함)
        
        Returns:
        --------
        DataFrame: 거래 로그
        """
        trades = []
        
        print(f"\n{'='*60}")
        print(f"백테스트 시작: {len(df_30m):,}개 바")
        print(f"{'='*60}\n")
        
        for i in range(len(df_30m) - 1):  # 마지막 바는 라벨 없음
            bar_start = pd.to_datetime(df_30m.iloc[i]['bar30_start'], utc=True)
            
            if i % 100 == 0:
                print(f"진행: {i:4d}/{len(df_30m)} ({i/len(df_30m)*100:.1f}%) - {bar_start}")
            
            decision = self.decide_on_bar_close(df_30m, i)
            
            if decision['should_enter']:
                # 진입가/만기가 계산
                entry_price = df_30m.iloc[i+1]['open']  # 다음 바 open
                label_price = df_30m.iloc[i+1]['close']  # 다음 바 close
                
                # 결과 판정
                side = decision['side']
                if side == 'LONG':
                    result = 1 if label_price > entry_price else 0
                elif side == 'SHORT':
                    result = 1 if label_price < entry_price else 0
                else:
                    result = 0
                
                # 거래 기록
                trade = {
                    'bar30_start': decision['bar30_start'],
                    'bar30_end': decision['bar30_end'],
                    'entry_ts': decision['entry_ts'],
                    'label_ts': decision['label_ts'],
                    'side': side,
                    'entry_price': entry_price,
                    'label_price': label_price,
                    'probability': decision['probability'],
                    'regime': decision['regime'],
                    'result': result,
                    'reason': decision['reason']
                }
                trades.append(trade)
                
                # 모델 성능 추적 업데이트
                status = self.model.update_performance(result)
                
                if status['need_retrain']:
                    print(f"\n{'='*60}")
                    print(f"🔄 재학습 트리거 발동!")
                    print(f"  거래 횟수: {status['trade_count']}번")
                    print(f"  최근 50번 승률: {status['win_rate']:.2%}")
                    print(f"  → 실제 시스템에서는 여기서 재학습 실행")
                    print(f"{'='*60}\n")
                    
                    # 실제 시스템에서는 여기서 재학습 호출
                    # self.retrain_model(...)
        
        trades_df = pd.DataFrame(trades)
        
        if len(trades_df) > 0:
            print(f"\n{'='*60}")
            print(f"백테스트 완료")
            print(f"{'='*60}")
            print(f"총 거래: {len(trades_df)}건")
            print(f"승률: {trades_df['result'].mean():.2%}")
            print(f"LONG: {(trades_df['side']=='LONG').sum()}건")
            print(f"SHORT: {(trades_df['side']=='SHORT').sum()}건")
            
            # 레짐별 승률
            print(f"\n레짐별 승률:")
            for regime in [-1, 0, 1]:
                regime_trades = trades_df[trades_df['regime'] == regime]
                if len(regime_trades) > 0:
                    regime_name = self.model._regime_to_name(regime)
                    wr = regime_trades['result'].mean()
                    print(f"  {regime_name:5s}: {wr:.2%} ({len(regime_trades)}건)")
        else:
            print(f"\n⚠️ 거래 없음")
        
        return trades_df
    
    # ==========================================
    # 실시간 거래 (1분마다 실행)
    # ==========================================
    
    def process_realtime(
        self, 
        df_1m_latest: pd.DataFrame
    ) -> Optional[Dict]:
        """
        실시간 처리 (1분마다 호출)
        
        Parameters:
        -----------
        df_1m_latest : DataFrame
            최신 1분봉 데이터 (충분한 히스토리 포함)
        
        Returns:
        --------
        dict or None: 진입 신호가 있으면 거래 정보 반환
        """
        # 1분봉 → 30분봉 변환
        df_30m = self.tf_manager.aggregate_1m_to_30m(df_1m_latest)
        
        if len(df_30m) < 2:
            return None
        
        # 피처 생성 필요 (실제로는 FeatureEngineer 사용)
        # 여기서는 간소화
        
        # 마지막 완성된 바에서 판단
        last_bar_idx = len(df_30m) - 2  # 마지막 완성된 바
        
        decision = self.decide_on_bar_close(df_30m, last_bar_idx)
        
        if decision['should_enter']:
            return decision
        
        return None


# ==========================================
# 테스트
# ==========================================
if __name__ == "__main__":
    from feature_engineer import FeatureEngineer
    from model_train import ModelTrainer
    
    Config.create_directories()
    Config.validate_config()
    
    # 샘플 데이터 생성
    print("샘플 1분봉 데이터 생성...")
    periods = 10000  # 약 7일
    ts = pd.date_range(
        end=datetime.now(timezone.utc), 
        periods=periods, 
        freq='1min', 
        tz='UTC'
    )
    
    np.random.seed(42)
    base = 42000 + np.random.randn(periods).cumsum() * 20
    
    rows = []
    for i, t in enumerate(ts):
        b = base[i]
        o = b + np.random.uniform(-20, 20)
        c = b + np.random.uniform(-20, 20)
        h = max(o, c) + np.random.uniform(0, 30)
        l = min(o, c) - np.random.uniform(0, 30)
        v = np.random.uniform(100, 1000)
        rows.append({
            'timestamp': t, 
            'open': o, 
            'high': h, 
            'low': l, 
            'close': c, 
            'volume': v
        })
    
    df_1m = pd.DataFrame(rows)
    
    # 피처 생성
    print("\n30분봉 피처 생성...")
    fe = FeatureEngineer()
    features = fe.create_feature_pool(df_1m, lookback_bars=100)
    
    # 라벨 생성
    tf_manager = TimeframeManager()
    df_30m = tf_manager.aggregate_1m_to_30m(df_1m)
    target = fe.create_target_30m(df_30m)
    
    # 유효 데이터
    valid = target.notna() & (target.index >= 100)
    X = features
    y = target[valid].reset_index(drop=True)
    
    min_len = min(len(X), len(y))
    X = X.iloc[:min_len]
    y = y.iloc[:min_len]
    
    # 모델 학습
    print("\n모델 학습...")
    trainer = ModelTrainer(Config)
    trainer.feature_selection_regime(X, y, regime_col='regime', top_k=30)
    trainer.train_ensemble_regime(X, y, regime_col='regime', test_size=0.2)
    
    # 30분봉에 피처 병합 (백테스트용)
    df_30m_with_features = df_30m.iloc[100:100+len(X)].reset_index(drop=True)
    for col in X.columns:
        if col not in df_30m_with_features.columns:
            df_30m_with_features[col] = X[col].values
    
    # 트레이딩 엔진 생성
    print("\n트레이딩 엔진 백테스트...")
    engine = TradingEngine(trainer, symbol='BTCUSDT')
    
    # 백테스트
    trades = engine.backtest(df_30m_with_features)
    
    # 결과 저장
    if len(trades) > 0:
        trades.to_csv('backtest_trades.csv', index=False)
        print(f"\n✓ 거래 로그 저장: backtest_trades.csv")
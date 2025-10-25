# timeframe_manager.py - 1분봉 → 30분봉 변환
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone


class TimeframeManager:
    """1분봉을 30분봉으로 집계하고 정확한 타임스탬프 관리"""
    
    @staticmethod
    def floor_to_30m(ts):
        """타임스탬프를 30분 단위로 내림"""
        ts = pd.to_datetime(ts, utc=True)
        minutes = ts.minute
        floored_minutes = (minutes // 30) * 30
        return ts.replace(minute=floored_minutes, second=0, microsecond=0)
    
    @staticmethod
    def aggregate_1m_to_30m(df_1m):
        """
        1분봉 → 30분봉 집계
        
        Parameters:
        -----------
        df_1m : DataFrame
            컬럼: timestamp, open, high, low, close, volume
        
        Returns:
        --------
        DataFrame
            컬럼: bar30_start, bar30_end, open, high, low, close, volume
        """
        df = df_1m.copy()
        
        # 타임스탬프 정규화
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values('timestamp')
        
        # 30분 바 시작 시간 계산
        df['bar30_start'] = df['timestamp'].apply(TimeframeManager.floor_to_30m)
        
        # 30분 단위로 집계
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        df_30m = df.groupby('bar30_start').agg(agg_dict).reset_index()
        
        # bar30_end 추가 (= bar30_start + 30분)
        df_30m['bar30_end'] = df_30m['bar30_start'] + pd.Timedelta(minutes=30)
        
        # 1분 인덱스 추가 (조인용)
        # entry 시점 = bar30_end의 1분 인덱스
        df_30m['m1_index_entry'] = (
            df_30m['bar30_end'].astype(np.int64) // 10**9 // 60
        ).astype(int)
        
        # label 시점 = bar30_end + 30분의 1분 인덱스
        df_30m['m1_index_label'] = (
            (df_30m['bar30_end'] + pd.Timedelta(minutes=30)).astype(np.int64) // 10**9 // 60
        ).astype(int)
        
        return df_30m
    
    @staticmethod
    def get_entry_price(df_1m, entry_ts):
        """
        진입가 계산: entry_ts의 1분봉 open
        
        Parameters:
        -----------
        df_1m : DataFrame
            1분봉 데이터
        entry_ts : datetime
            진입 시각 (= bar30_end)
        
        Returns:
        --------
        float: 진입가 (open)
        """
        df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], utc=True)
        entry_ts = pd.to_datetime(entry_ts, utc=True)
        
        mask = df_1m['timestamp'] == entry_ts
        if mask.sum() == 0:
            return None
        
        return float(df_1m.loc[mask, 'open'].iloc[0])
    
    @staticmethod
    def get_label_price(df_1m, label_ts):
        """
        만기가 계산: label_ts - 1분의 1분봉 close
        
        Parameters:
        -----------
        df_1m : DataFrame
            1분봉 데이터
        label_ts : datetime
            만기 시각 (= bar30_end + 30분)
        
        Returns:
        --------
        float: 만기가 (close)
        """
        df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], utc=True)
        label_ts = pd.to_datetime(label_ts, utc=True)
        
        # label_ts - 1분의 close
        actual_ts = label_ts - pd.Timedelta(minutes=1)
        
        mask = df_1m['timestamp'] == actual_ts
        if mask.sum() == 0:
            return None
        
        return float(df_1m.loc[mask, 'close'].iloc[0])
    
    @staticmethod
    def create_label(entry_price, label_price, side='LONG'):
        """
        라벨 생성
        
        Parameters:
        -----------
        entry_price : float
        label_price : float
        side : str
            'LONG' or 'SHORT'
        
        Returns:
        --------
        int: 1 (승), 0 (패), None (동가)
        """
        if entry_price is None or label_price is None:
            return None
        
        if side == 'LONG':
            if label_price > entry_price:
                return 1
            elif label_price < entry_price:
                return 0
            else:
                return None  # 동가
        elif side == 'SHORT':
            if label_price < entry_price:
                return 1
            elif label_price > entry_price:
                return 0
            else:
                return None
        else:
            raise ValueError(f"Unknown side: {side}")
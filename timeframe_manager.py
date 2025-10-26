"""
타임프레임 관리자 - 1분봉 ↔ 30분봉 집계 및 인덱스 매핑
버전: 1.3.1 (30분 옵션 최적화 + 미래 데이터 누수 방지)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple

import config


class TimeframeManager:
    """
    타임프레임 집계 및 인덱스 매핑 관리
    - 1분봉 → 30분봉 리샘플
    - 인덱스 정합성 유지
    - 타임프레임 간 변환
    - 미래 데이터 누수 방지
    """
    
    def __init__(self):
        """인자 없이 초기화 (Config 참조)"""
        self.bar_minutes = config.BAR_MINUTES
        self.bar_timedelta = timedelta(minutes=self.bar_minutes)
        
    def aggregate_1m_to_30m(self, df_1m: pd.DataFrame, realtime_safe: bool = False) -> pd.DataFrame:
        """
        1분봉을 30분봉으로 집계
        
        Args:
            df_1m: 1분봉 데이터 (timestamp, open, high, low, close, volume)
            realtime_safe: True면 마지막 30분봉 제외 (미완성 봉 방지)
        
        Returns:
            30분봉 데이터 (bar30_start, bar30_end, open, high, low, close, volume)
        """
        if df_1m.empty:
            return pd.DataFrame()
        
        # timestamp를 UTC로 변환 및 인덱스 설정
        df = df_1m.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
            if df['timestamp'].dt.tz is None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(config.TZ)
            df = df.set_index('timestamp')
        
        # 30분 단위로 리샘플
        df_30m = df.resample(f'{self.bar_minutes}T', label='left', closed='left').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        # 실시간 모드: 마지막 30분봉 제외 (아직 완성 안 됨)
        if realtime_safe and len(df_30m) > 0:
            # 마지막 봉의 1분봉 개수 확인
            last_bar_start = df_30m.index[-1]
            last_bar_end = last_bar_start + self.bar_timedelta
            
            # 원본 데이터에서 마지막 봉에 속한 1분봉 개수 확인
            mask = (df.index >= last_bar_start) & (df.index < last_bar_end)
            num_bars = mask.sum()
            
            # 30분(30개) 미만이면 미완성 봉으로 판단
            if num_bars < self.bar_minutes:
                df_30m = df_30m.iloc[:-1]
        
        if df_30m.empty:
            return pd.DataFrame()
        
        # bar30_start, bar30_end 컬럼 추가
        df_30m['bar30_start'] = df_30m.index
        df_30m['bar30_end'] = df_30m.index + self.bar_timedelta
        
        # 인덱스 리셋
        df_30m = df_30m.reset_index(drop=True)
        
        # 컬럼 순서 정리
        cols = ['bar30_start', 'bar30_end', 'open', 'high', 'low', 'close', 'volume']
        df_30m = df_30m[cols]
        
        return df_30m
    
    def get_current_30m_slot(self, dt: datetime) -> datetime:
        """
        주어진 시각이 속한 30분봉의 시작 시각 반환 (UTC 강제)
        
        Args:
            dt: 타임스탬프
        
        Returns:
            30분봉 시작 시각 (UTC)
        """
        # UTC 변환
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=config.TZ)
        elif dt.tzinfo != config.TZ:
            dt = dt.astimezone(config.TZ)
        
        # 분을 30으로 나눈 몫으로 30분 슬롯 계산
        slot_minute = (dt.minute // self.bar_minutes) * self.bar_minutes
        return dt.replace(minute=slot_minute, second=0, microsecond=0)
    
    def is_30m_complete(self, dt: datetime, buffer_seconds: int = 0) -> bool:
        """
        주어진 시각이 30분봉 완료 시점인지 확인
        
        Args:
            dt: 타임스탬프
            buffer_seconds: 완료 전 버퍼 (29분 시점 예측용, 예: 60초)
        
        Returns:
            30분봉 완료 여부
        """
        # UTC 변환
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=config.TZ)
        elif dt.tzinfo != config.TZ:
            dt = dt.astimezone(config.TZ)
        
        # 30분 슬롯 내 경과 시간 (초)
        slot_start = self.get_current_30m_slot(dt)
        elapsed_seconds = (dt - slot_start).total_seconds()
        
        # 버퍼 적용 (예: 29분 = 1740초)
        target_seconds = self.bar_minutes * 60 - buffer_seconds
        
        return elapsed_seconds >= target_seconds
    
    def map_1m_to_30m_index(self, df_1m: pd.DataFrame, df_30m: pd.DataFrame, m1_index: int) -> Optional[int]:
        """
        1분봉 인덱스를 30분봉 인덱스로 매핑 (타임스탬프 기반)
        
        Args:
            df_1m: 1분봉 데이터프레임
            df_30m: 30분봉 데이터프레임
            m1_index: 1분봉 인덱스
        
        Returns:
            30분봉 인덱스 (없으면 None)
        """
        if m1_index < 0 or m1_index >= len(df_1m):
            return None
        
        if df_30m.empty:
            return None
        
        # 1분봉 타임스탬프 추출
        if 'timestamp' in df_1m.columns:
            ts_1m = df_1m.iloc[m1_index]['timestamp']
        else:
            ts_1m = df_1m.index[m1_index]
        
        ts_1m = pd.to_datetime(ts_1m, utc=True)
        
        # 30분 슬롯 계산
        slot_start = self.get_current_30m_slot(ts_1m)
        
        # df_30m에서 해당 slot_start 찾기
        df_30m_copy = df_30m.copy()
        df_30m_copy['bar30_start'] = pd.to_datetime(df_30m_copy['bar30_start'], utc=True)
        matches = df_30m_copy[df_30m_copy['bar30_start'] == slot_start]
        
        if matches.empty:
            return None
        
        return matches.index[0]
    
    def map_30m_to_1m_range(self, df_1m: pd.DataFrame, df_30m: pd.DataFrame, m30_index: int) -> Optional[Tuple[int, int]]:
        """
        30분봉 인덱스에 해당하는 1분봉 인덱스 범위 반환 (타임스탬프 기반)
        
        Args:
            df_1m: 1분봉 데이터프레임
            df_30m: 30분봉 데이터프레임
            m30_index: 30분봉 인덱스
        
        Returns:
            (start_index, end_index) 튜플 (없으면 None)
        """
        if m30_index < 0 or m30_index >= len(df_30m):
            return None
        
        # 30분봉의 시작/종료 시각
        bar30_start = pd.to_datetime(df_30m.iloc[m30_index]['bar30_start'], utc=True)
        bar30_end = pd.to_datetime(df_30m.iloc[m30_index]['bar30_end'], utc=True)
        
        # 1분봉에서 해당 범위 찾기
        df_1m_copy = df_1m.copy()
        if 'timestamp' in df_1m_copy.columns:
            df_1m_copy['timestamp'] = pd.to_datetime(df_1m_copy['timestamp'], utc=True)
            mask = (df_1m_copy['timestamp'] >= bar30_start) & (df_1m_copy['timestamp'] < bar30_end)
        else:
            df_1m_copy.index = pd.to_datetime(df_1m_copy.index, utc=True)
            mask = (df_1m_copy.index >= bar30_start) & (df_1m_copy.index < bar30_end)
        
        indices = df_1m_copy[mask].index.tolist()
        
        if not indices:
            return None
        
        # DataFrame의 positional index로 변환
        start_idx = df_1m_copy.index.get_loc(indices[0])
        end_idx = df_1m_copy.index.get_loc(indices[-1])
        
        return (start_idx, end_idx)
    
    def get_bar30_boundaries(self, dt: datetime) -> Tuple[datetime, datetime]:
        """
        주어진 시각이 속한 30분봉의 시작/종료 시각 반환
        
        Args:
            dt: 타임스탬프
        
        Returns:
            (bar30_start, bar30_end) 튜플
        """
        bar30_start = self.get_current_30m_slot(dt)
        bar30_end = bar30_start + self.bar_timedelta
        return bar30_start, bar30_end
    
    def align_dataframes(self, df_1m: pd.DataFrame, df_30m: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        1분봉과 30분봉 데이터프레임의 시간 정렬
        
        Args:
            df_1m: 1분봉 데이터
            df_30m: 30분봉 데이터
        
        Returns:
            (정렬된 df_1m, 정렬된 df_30m)
        """
        if df_1m.empty or df_30m.empty:
            return df_1m, df_30m
        
        # timestamp 컬럼 확인 및 UTC 변환
        if 'timestamp' in df_1m.columns:
            df_1m = df_1m.copy()
            df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], utc=True)
            df_1m = df_1m.sort_values('timestamp').reset_index(drop=True)
        
        if 'bar30_start' in df_30m.columns:
            df_30m = df_30m.copy()
            df_30m['bar30_start'] = pd.to_datetime(df_30m['bar30_start'], utc=True)
            df_30m = df_30m.sort_values('bar30_start').reset_index(drop=True)
        
        return df_1m, df_30m
    
    def resample_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        범용 리샘플링 (4h, 1h, 15m 등)
        
        Args:
            df: 원본 데이터 (timestamp 인덱스 또는 컬럼)
            timeframe: '4h', '1h', '15m' 등
        
        Returns:
            리샘플된 데이터프레임 (timestamp 또는 bar_start/bar_end 컬럼 포함)
        """
        if df.empty:
            return pd.DataFrame()
        
        # timestamp를 인덱스로 설정 및 UTC 변환
        df_copy = df.copy()
        if 'timestamp' in df_copy.columns:
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], utc=True)
            df_copy = df_copy.set_index('timestamp')
        elif 'bar30_start' in df_copy.columns:
            df_copy['bar30_start'] = pd.to_datetime(df_copy['bar30_start'], utc=True)
            df_copy = df_copy.set_index('bar30_start')
        
        # 리샘플 규칙 변환
        resample_rule = timeframe.replace('h', 'H').replace('m', 'T')
        
        # 리샘플링
        df_resampled = df_copy.resample(resample_rule, label='left', closed='left').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        df_resampled = df_resampled.reset_index()
        
        # 일관성: bar_start/bar_end 형태로 통일
        col_name = f'bar_{timeframe}_start'
        df_resampled.rename(columns={df_resampled.columns[0]: col_name}, inplace=True)
        
        # bar_end 추가
        interval_minutes = self._parse_timeframe_minutes(timeframe)
        df_resampled[f'bar_{timeframe}_end'] = df_resampled[col_name] + timedelta(minutes=interval_minutes)
        
        return df_resampled
    
    def _parse_timeframe_minutes(self, timeframe: str) -> int:
        """타임프레임 문자열을 분 단위로 변환"""
        if 'h' in timeframe:
            hours = int(timeframe.replace('h', ''))
            return hours * 60
        elif 'm' in timeframe:
            return int(timeframe.replace('m', ''))
        else:
            return int(timeframe)
    
    def validate_aggregation(self, df_1m: pd.DataFrame, df_30m: pd.DataFrame) -> Tuple[bool, list]:
        """
        1분봉 → 30분봉 집계 결과 검증
        
        Args:
            df_1m: 원본 1분봉 데이터
            df_30m: 집계된 30분봉 데이터
        
        Returns:
            (검증 통과 여부, 오류 목록)
        """
        errors = []
        
        if df_1m.empty:
            errors.append("1분봉 데이터가 비어있음")
            return False, errors
        
        if df_30m.empty:
            errors.append("30분봉 데이터가 비어있음")
            return False, errors
        
        # 1. 30분봉 개수 체크 (정확히 일치해야 함)
        expected_bars = len(df_1m) // self.bar_minutes
        actual_bars = len(df_30m)
        
        # 완벽히 나누어떨어지지 않는 경우 허용 (마지막 미완성 봉)
        if actual_bars != expected_bars and actual_bars != expected_bars + 1:
            errors.append(f"30분봉 개수 불일치: 예상 {expected_bars}, 실제 {actual_bars}")
        
        # 2. 필수 컬럼 체크
        required_cols = ['bar30_start', 'bar30_end', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df_30m.columns]
        
        if missing_cols:
            errors.append(f"필수 컬럼 누락: {missing_cols}")
        
        # 3. OHLC 정합성 체크
        if 'high' in df_30m.columns and 'low' in df_30m.columns:
            invalid_hl = (df_30m['high'] < df_30m['low']).sum()
            if invalid_hl > 0:
                errors.append(f"High < Low 발견: {invalid_hl}건")
        
        if all(col in df_30m.columns for col in ['open', 'high', 'low', 'close']):
            invalid_ohlc = (
                (df_30m['high'] < df_30m['open']) |
                (df_30m['high'] < df_30m['close']) |
                (df_30m['low'] > df_30m['open']) |
                (df_30m['low'] > df_30m['close'])
            ).sum()
            
            if invalid_ohlc > 0:
                errors.append(f"OHLC 범위 위반: {invalid_ohlc}건")
        
        # 4. 시간 간격 체크
        if 'bar30_start' in df_30m.columns and len(df_30m) > 1:
            df_30m_copy = df_30m.copy()
            df_30m_copy['bar30_start'] = pd.to_datetime(df_30m_copy['bar30_start'])
            time_diffs = df_30m_copy['bar30_start'].diff().dt.total_seconds() / 60
            
            # 30분 간격 체크 (갭 있으면 명시적 에러)
            invalid_intervals = ((time_diffs < 29.5) | (time_diffs > 30.5)) & time_diffs.notna()
            if invalid_intervals.sum() > 0:
                errors.append(f"시간 간격 이상: {invalid_intervals.sum()}건 (주말/갭 확인 필요)")
        
        # 5. NaN 체크
        nan_counts = df_30m.isnull().sum()
        if nan_counts.sum() > 0:
            errors.append(f"NaN 발견: {nan_counts.to_dict()}")
        
        # 6. UTC 타임존 체크
        if 'bar30_start' in df_30m.columns:
            bar30_start_tz = pd.to_datetime(df_30m['bar30_start'].iloc[0]).tzinfo
            if bar30_start_tz is None:
                errors.append("bar30_start가 timezone-naive (UTC 변환 필요)")
        
        return len(errors) == 0, errors


# ============================================================
# 테스트 및 검증
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TimeframeManager 테스트 (v1.3.1)")
    print("=" * 60)
    
    # 테스트 데이터 생성
    start_time = pd.Timestamp('2025-01-01 00:00:00', tz='UTC')
    timestamps = [start_time + pd.Timedelta(minutes=i) for i in range(100)]
    
    df_1m_test = pd.DataFrame({
        'timestamp': timestamps,
        'open': np.random.uniform(50000, 51000, 100),
        'high': np.random.uniform(51000, 52000, 100),
        'low': np.random.uniform(49000, 50000, 100),
        'close': np.random.uniform(50000, 51000, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    })
    
    # OHLC 정합성 보장
    df_1m_test['high'] = df_1m_test[['open', 'close']].max(axis=1) + 100
    df_1m_test['low'] = df_1m_test[['open', 'close']].min(axis=1) - 100
    
    # TimeframeManager 초기화
    tf_manager = TimeframeManager()
    
    print("\n✅ TimeframeManager 초기화 성공")
    print(f"  - BAR_MINUTES: {tf_manager.bar_minutes}")
    print(f"  - BAR_TIMEDELTA: {tf_manager.bar_timedelta}")
    
    # 1분봉 → 30분봉 집계 (백테스트 모드)
    print("\n📊 1분봉 → 30분봉 집계 테스트 (백테스트)")
    df_30m_test = tf_manager.aggregate_1m_to_30m(df_1m_test, realtime_safe=False)
    
    print(f"  - 입력: {len(df_1m_test)}개 1분봉")
    print(f"  - 출력: {len(df_30m_test)}개 30분봉")
    print(f"  - 예상: {len(df_1m_test) // 30}개")
    
    if not df_30m_test.empty:
        print(f"\n  30분봉 샘플 (첫 3개):")
        print(df_30m_test.head(3)[['bar30_start', 'bar30_end', 'open', 'close']])
    
    # 실시간 모드 테스트
    print("\n📊 1분봉 → 30분봉 집계 테스트 (실시간 안전 모드)")
    df_30m_realtime = tf_manager.aggregate_1m_to_30m(df_1m_test, realtime_safe=True)
    print(f"  - 실시간 안전 모드: {len(df_30m_realtime)}개 30분봉 (마지막 미완성 봉 제외)")
    
    # 집계 검증
    print("\n🔍 집계 검증")
    valid, errors = tf_manager.validate_aggregation(df_1m_test, df_30m_test)
    
    if valid:
        print("  ✅ 검증 통과")
    else:
        print("  ❌ 검증 실패:")
        for error in errors:
            print(f"    - {error}")
    
    # 인덱스 매핑 테스트
    print("\n🔗 인덱스 매핑 테스트 (타임스탬프 기반)")
    test_indices = [0, 29, 30, 59, 60, 89]
    
    for m1_idx in test_indices:
        m30_idx = tf_manager.map_1m_to_30m_index(df_1m_test, df_30m_test, m1_idx)
        print(f"  - 1분봉 인덱스 {m1_idx} → 30분봉 인덱스 {m30_idx}")
    
    # 30분봉 완료 체크 (29분 시점 예측용)
    print("\n⏰ 30분봉 완료 체크 (29분 시점 예측)")
    test_times = [
        pd.Timestamp('2025-01-01 09:28:00', tz='UTC'),  # 29분 전
        pd.Timestamp('2025-01-01 09:29:00', tz='UTC'),  # 29분 정각
        pd.Timestamp('2025-01-01 09:30:00', tz='UTC'),  # 30분 완료
    ]
    
    for dt in test_times:
        is_complete_default = tf_manager.is_30m_complete(dt, buffer_seconds=0)
        is_complete_29min = tf_manager.is_30m_complete(dt, buffer_seconds=60)  # 29분 시점
        
        print(f"  - {dt.strftime('%H:%M')}: 완료={is_complete_default}, 29분예측={is_complete_29min}")
    
    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)
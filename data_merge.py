# data_merge.py - 데이터 병합 및 관리 모듈 (UTC 일원화 / 정규화 / Xy 정렬 안전화)

import os
import glob
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


# =========================
# 유틸
# =========================
def _to_utc(ts_series: pd.Series) -> pd.Series:
    """모든 타임스탬프를 tz-aware UTC로 통일"""
    if ts_series.dt.tz is None:
        # naive -> UTC로 간주
        return pd.to_datetime(ts_series, utc=True, errors='coerce')
    # tz-aware -> UTC로 변환
    return ts_series.dt.tz_convert('UTC')


def _normalize_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    가격 DF 정규화:
      - timestamp UTC tz-aware
      - 중복 제거
      - 정렬
    """
    if df.empty:
        return df

    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    # 수치형 안전 캐스팅(있을 경우)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _normalize_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.drop_duplicates().sort_values('timestamp')
    return df


def _normalize_trades_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if 'entry_time' in df.columns:
        df['entry_time'] = pd.to_datetime(df['entry_time'], utc=True, errors='coerce')
    if 'exit_time' in df.columns:
        df['exit_time'] = pd.to_datetime(df['exit_time'], utc=True, errors='coerce')
    # 금액/결과형 필드 숫자화
    for col in ['prediction', 'result', 'profit_loss', 'amount', 'confidence']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # trade_id 문자열 보장
    if 'trade_id' in df.columns:
        df['trade_id'] = df['trade_id'].astype(str)
    return df


# =========================
# Merger
# =========================
class DataMerger:
    """실시간 데이터 병합 및 관리"""

    def __init__(self, config):
        self.config = config
        self.merged_data: pd.DataFrame | None = None

    # ---------- Loaders ----------
    def load_price_data(self, start_date: pd.Timestamp | None = None,
                        end_date: pd.Timestamp | None = None) -> pd.DataFrame:
        price_dir = os.path.join(self.config.PRICE_DATA_DIR, 'raw')
        files = sorted(glob.glob(os.path.join(price_dir, '*.csv')))
        if not files:
            print("가격 데이터 파일이 없습니다.")
            return pd.DataFrame()

        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                if 'timestamp' not in df.columns:
                    continue
                df = _normalize_price_df(df)
                if start_date is not None:
                    start_date = pd.to_datetime(start_date, utc=True)
                    df = df[df['timestamp'] >= start_date]
                if end_date is not None:
                    end_date = pd.to_datetime(end_date, utc=True)
                    df = df[df['timestamp'] <= end_date]
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                print(f"파일 로드 실패: {f} - {e}")

        if not dfs:
            return pd.DataFrame()

        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df = _normalize_price_df(merged_df)
        return merged_df

    def load_trade_logs(self) -> pd.DataFrame:
        path = os.path.join(self.config.TRADE_LOG_DIR, 'trades.csv')
        if not os.path.exists(path):
            return pd.DataFrame()
        try:
            df = pd.read_csv(path)
            df = _normalize_trades_df(df)
            return df
        except Exception as e:
            print(f"거래 로그 로드 실패: {e}")
            return pd.DataFrame()

    def load_feature_logs(self) -> pd.DataFrame:
        feature_dir = self.config.FEATURE_LOG_DIR
        files = sorted(glob.glob(os.path.join(feature_dir, 'features_*.csv')))
        if not files:
            return pd.DataFrame()

        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f)
                if 'timestamp' in df.columns:
                    df = _normalize_feature_df(df)
                    if not df.empty:
                        dfs.append(df)
            except Exception as e:
                print(f"피처 로그 로드 실패: {f} - {e}")

        if not dfs:
            return pd.DataFrame()
        merged = pd.concat(dfs, ignore_index=True)
        merged = _normalize_feature_df(merged)
        return merged

    # ---------- Merge ----------
    def merge_all_data(self) -> pd.DataFrame | None:
        """
        가격 데이터를 기준으로:
          - 거래 로그: entry_time을 기준으로 가격 timestamp에 매핑(가장 가까운 1분 이내)
          - 피처 로그: timestamp 키로 left join
        """
        print("데이터 병합 시작...")

        price = self.load_price_data()
        trades = self.load_trade_logs()
        feats = self.load_feature_logs()

        if price.empty:
            print("가격 데이터가 없어 병합할 수 없습니다.")
            return None

        merged = price.copy()

        # --- 거래 로그 매핑(가장 가까운 1분 이내) ---
        if not trades.empty and 'entry_time' in trades.columns:
            # 기본적으로 1분봉 그리드와 entry_time을 asof 매칭
            left = merged[['timestamp']].rename(columns={'timestamp': 'ts'})
            right = trades[['trade_id', 'entry_time', 'prediction', 'result', 'profit_loss', 'amount', 'confidence']].copy()
            left = left.sort_values('ts')
            right = right.sort_values('entry_time')

            # merge_asof: ts 기준으로 가장 가까운 entry_time 매칭
            asof_join = pd.merge_asof(
                left, right, left_on='ts', right_on='entry_time',
                direction='nearest', tolerance=pd.Timedelta('1min')
            )

            # 결과 붙이기
            merged = merged.merge(asof_join.drop(columns=['ts']), left_on='timestamp', right_on='entry_time', how='left')
            # 컬럼 정리: entry_time 을 timestamp에 맞춰 그대로 보관
            # 필요시 suffix 처리 가능
        else:
            # 거래 로그가 없으면 pass
            pass

        # --- 피처 로그 조인 ---
        if not feats.empty:
            merged = pd.merge(merged, feats, on='timestamp', how='left', suffixes=('', '_feature'))

        # 최종 정규화 한 번 더
        merged = _normalize_price_df(merged)

        self.merged_data = merged

        # 통계 출력
        print("\n병합 완료:")
        print(f"- 전체 레코드 수: {len(merged)}")
        print(f"- 시작 시간(UTC): {merged['timestamp'].min()}")
        print(f"- 종료 시간(UTC): {merged['timestamp'].max()}")

        if 'trade_id' in merged.columns:
            trade_count = merged['trade_id'].notna().sum()
            print(f"- 매핑된 거래 기록 수: {int(trade_count)}")
            if 'result' in merged.columns:
                wr_df = merged[merged['result'].isin([0, 1])]
                if not wr_df.empty:
                    win_rate = wr_df['result'].mean()
                    print(f"- 매핑 구간 승률: {win_rate*100:.2f}%")

        return merged

    # ---------- Save ----------
    def save_merged_data(self, df: pd.DataFrame | None = None) -> bool:
        if df is None:
            df = self.merged_data
        if df is None or df.empty:
            print("저장할 데이터가 없습니다.")
            return False

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = self.config.RESULT_DIR
        os.makedirs(out_dir, exist_ok=True)
        fullpath = os.path.join(out_dir, f'merged_data_{ts}.pkl')
        latest = os.path.join(out_dir, 'training_data.pkl')

        df.to_pickle(fullpath)
        df.to_pickle(latest)
        print(f"병합 데이터 저장 완료: {fullpath}")
        return True

    # ---------- Train set ----------
    def get_training_data(self, lookback_days: int = 30):
        """
        학습용 데이터 준비:
          - 최신 병합 데이터가 있으면 사용, 없으면 즉시 병합
          - 최근 N일만 선택
          - FeatureEngineer로 X 생성, 타겟 y 생성
          - y.dropna 후 X/y 공통 인덱스 교집합으로 정렬(길이/정렬 문제 방지)
        """
        latest = os.path.join(self.config.RESULT_DIR, 'training_data.pkl')
        if os.path.exists(latest):
            df = pd.read_pickle(latest)
        else:
            df = self.merge_all_data()
            if df is None or df.empty:
                print("학습 데이터 준비 실패: 병합 결과 없음")
                return None, None

        # UTC 보장
        df = _normalize_price_df(df)

        # 최근 N일
        if 'timestamp' in df.columns and lookback_days is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            df = df[df['timestamp'] >= cutoff]

        if df.empty:
            print("최근 구간 데이터가 비어 있습니다.")
            return None, None

        # 피처/타겟 생성
        from model_train import FeatureEngineer  # 순환 import 방지 위해 내부 import
        fe = FeatureEngineer()

        X = fe.create_feature_pool(df)
        y = fe.create_target(df, window=self.config.PREDICTION_WINDOW)

        # y NaN 제거 후 공통 인덱스 정렬
        y = y.dropna()
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx].astype(int)

        # 방어적 클린업
        X = X.fillna(method='ffill').fillna(0).replace([np.inf, -np.inf], 0)

        return X, y

    # ---------- Mutators ----------
    def update_trade_result(self, trade_id: str, result: int, profit_loss: float) -> bool:
        """
        거래 결과 업데이트: trades.csv에 반영
        """
        path = os.path.join(self.config.TRADE_LOG_DIR, 'trades.csv')
        if not os.path.exists(path):
            return False

        try:
            df = pd.read_csv(path)
        except Exception:
            return False

        if 'trade_id' not in df.columns:
            return False

        mask = df['trade_id'].astype(str) == str(trade_id)
        if not mask.any():
            return False

        df.loc[mask, 'result'] = result
        df.loc[mask, 'profit_loss'] = profit_loss
        df.loc[mask, 'exit_time'] = datetime.now(timezone.utc).isoformat()

        df.to_csv(path, index=False)
        print(f"거래 {trade_id} 결과 업데이트 완료")
        return True

    def add_new_price_data(self, new_data: pd.DataFrame) -> bool:
        """
        새로운 가격 데이터 추가 (UTC 정규화 → 날짜별 raw 파일 append)
        """
        if new_data is None or new_data.empty:
            return False

        new_data = _normalize_price_df(new_data)
        if new_data.empty:
            return False

        # 오늘(UTC) 기준 파일명
        today_utc = datetime.now(timezone.utc).strftime("%Y%m%d")
        out_path = os.path.join(self.config.PRICE_DATA_DIR, 'raw', f'prices_{today_utc}.csv')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if os.path.exists(out_path):
            try:
                exist = pd.read_csv(out_path)
                exist = _normalize_price_df(exist)
                updated = pd.concat([exist, new_data], ignore_index=True)
                updated = _normalize_price_df(updated)
                updated.to_csv(out_path, index=False)
            except Exception as e:
                print(f"가격 데이터 업데이트 실패: {e}")
                return False
        else:
            new_data.to_csv(out_path, index=False)

        print(f"가격 데이터 추가 완료: {len(new_data)} 레코드 -> {os.path.basename(out_path)}")
        return True

    def cleanup_old_data(self, days_to_keep: int = 90):
        """
        오래된 raw 가격 파일 삭제, 거래로그는 아카이브
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

        # 가격 데이터 정리
        price_dir = os.path.join(self.config.PRICE_DATA_DIR, 'raw')
        for f in glob.glob(os.path.join(price_dir, '*.csv')):
            name = os.path.basename(f)
            if not name.startswith('prices_'):
                continue
            date_str = name.replace('prices_', '').replace('.csv', '')
            try:
                file_date = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
                if file_date < cutoff:
                    os.remove(f)
                    print(f"오래된 가격 파일 삭제: {name}")
            except Exception:
                continue

        # 거래 로그 아카이브
        path = os.path.join(self.config.TRADE_LOG_DIR, 'trades.csv')
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df = _normalize_trades_df(df)
                if 'entry_time' in df.columns:
                    old = df[df['entry_time'] < cutoff]
                    if not old.empty:
                        arch = os.path.join(self.config.TRADE_LOG_DIR, f"trades_archive_{cutoff.strftime('%Y%m%d')}.csv")
                        old.to_csv(arch, index=False)
                        df = df[df['entry_time'] >= cutoff]
                        df.to_csv(path, index=False)
                        print(f"거래 로그 아카이브 완료: {len(old)} 레코드 -> {os.path.basename(arch)}")
            except Exception as e:
                print(f"거래 로그 정리 실패: {e}")


# =========================
# Validator
# =========================
class DataValidator:
    """데이터 검증"""

    @staticmethod
    def validate_price_data(df: pd.DataFrame):
        issues = []
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            issues.append(f"누락된 컬럼: {missing}")

        if all(c in df.columns for c in ['open', 'high', 'low', 'close']):
            invalid_high = df[df['high'] < df[['open', 'close']].max(axis=1)]
            if not invalid_high.empty:
                issues.append(f"잘못된 고가: {len(invalid_high)} 레코드")
            invalid_low = df[df['low'] > df[['open', 'close']].min(axis=1)]
            if not invalid_low.empty:
                issues.append(f"잘못된 저가: {len(invalid_low)} 레코드")

        if 'timestamp' in df.columns:
            dup = df[df.duplicated(subset=['timestamp'], keep=False)]
            if not dup.empty:
                issues.append(f"중복 타임스탬프: {len(dup)} 레코드")

        nulls = df[required].isnull().sum()
        if nulls.any():
            issues.append(f"결측치: {nulls[nulls > 0].to_dict()}")

        return len(issues) == 0, issues

    @staticmethod
    def validate_trade_logs(df: pd.DataFrame):
        issues = []
        req = ['trade_id', 'entry_time', 'prediction']
        missing = [c for c in req if c not in df.columns]
        if missing:
            issues.append(f"누락된 컬럼: {missing}")

        if 'trade_id' in df.columns:
            dup = df[df.duplicated(subset=['trade_id'], keep=False)]
            if not dup.empty:
                issues.append(f"중복 trade_id: {len(dup)} 레코드")

        if 'prediction' in df.columns:
            invalid = df[~df['prediction'].isin([0, 1])]
            if not invalid.empty:
                issues.append(f"잘못된 prediction 값: {len(invalid)} 레코드")

        return len(issues) == 0, issues


# 단독 실행 테스트
if __name__ == "__main__":
    from config import Config

    merger = DataMerger(Config)
    merged = merger.merge_all_data()
    if merged is not None and not merged.empty:
        merger.save_merged_data(merged)
        X, y = merger.get_training_data(lookback_days=30)
        if X is not None:
            print("\n학습 데이터 준비 완료:")
            print(f"- X shape: {X.shape}")
            print(f"- y shape: {y.shape}")
            print(f"- 클래스 분포: {y.value_counts().to_dict()}")

"""
데이터 로더 - API 기반 입출력 및 캐싱 관리 (단일 심볼 / 날짜별 CSV)
버전: 2.0.0 (Binance API + prices_YYYYMMDD.csv + UTC 정규화)

핵심:
- 1분봉을 바이낸스 API에서 직접 수집 (spot/futures 선택)
- 날짜별 파일 캐싱: price_data/prices_YYYYMMDD.csv
- 부족한 날짜만 API로 보충하고, 병합 저장(원자적 저장)
- 기존 인터페이스 유지: save_price_data, load_price_data, prepare_training_data 등
- 피처 캐시(parquet), 시계열 분할, 데이터 품질 검증 포함
"""

from __future__ import annotations
import time
import requests
import pandas as pd
import numpy as np
import hashlib
from binance.client import Client
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Tuple, List, Optional, Dict
from sklearn.model_selection import TimeSeriesSplit

import config
from feature_engineer import FeatureEngineer


class DataLoader:
    """
    API 기반 데이터 입출력 및 캐싱 관리
    - 단일 심볼 (config.SYMBOL)
    - 날짜별 CSV: price_data/prices_YYYYMMDD.csv
    - SHA1 기반 피처 캐싱
    - 시계열 분할/검증 유틸
    """

    # -----------------------------
    # 초기화
    # -----------------------------
    def __init__(self):
        """Config 기준으로 경로/세션 초기화"""
        self.data_dir = config.PRICE_DATA_DIR
        self.api_client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
    
    def _get_filename(self, symbol: str, date: datetime) -> Path:
        date_str = date.strftime("%Y%m%d")
        return self.data_dir / f"{symbol.lower()}_{date_str}.csv"
    
    def _fetch_from_api(self, symbol: str, date: datetime) -> pd.DataFrame:
        """
        특정 날짜의 1분봉 데이터(API)
        """
        start_dt = datetime(date.year, date.month, date.day, 0, 0, 0, tzinfo=timezone.utc)
        end_dt = start_dt + timedelta(days=1) - timedelta(seconds=1)

        print(f"🌐 API 요청: {symbol} {date.date()} 데이터 가져오는 중...")
        klines = self.api_client.get_historical_klines(
            symbol=symbol,
            interval="1m",
            start_str=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            end_str=end_dt.strftime("%Y-%m-%d %H:%M:%S"),
        )

        if not klines:
            print(f"⚠️  {symbol} {date.date()} 데이터 없음")
            return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

        # 1) DataFrame으로 만든 뒤, 열 개수에 맞춰 컬럼명 부여 (바이낸스 표준: 12열)
        df = pd.DataFrame(klines)
        if df.shape[1] < 6:
            raise RuntimeError(f"예상보다 열이 적습니다: {df.shape[1]}열 (데이터 포맷 확인 필요)")
        # 표준 12열이면 이름 부여, 아니면 앞 12열까지 자르고 이름 부여
        std_cols = [
            "open_time_ms", "open", "high", "low", "close", "volume",
            "close_time_ms", "quote_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ]
        if df.shape[1] >= 12:
            df = df.iloc[:, :12]
            df.columns = std_cols
        else:
            # 6~11열인 비정형 응답 방어: 가능한 만큼만 이름 부여 후 사용 컬럼만 선택
            tmp_cols = std_cols[:df.shape[1]]
            df.columns = tmp_cols

        # 2) 필요한 6개 컬럼만 사용 (open_time → timestamp로 리네임)
        use_cols = ["open_time_ms", "open", "high", "low", "close", "volume"]
        missing = [c for c in use_cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"필수 컬럼 누락: {missing} (응답 포맷 변경 가능성)")

        df = df[use_cols].copy()
        df = df.rename(columns={"open_time_ms": "timestamp"})

        # 3) 타입 정리
        # timestamp → UTC datetime, 숫자 컬럼 → float
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        # 4) 품질 정리: 결측/중복/정렬
        df = df.dropna(subset=["timestamp","open","high","low","close","volume"])
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        # 5) 최종 표준 형식 반환
        return df[["timestamp","open","high","low","close","volume"]]
    
    def _load_or_download_day(self, symbol: str, date: datetime) -> pd.DataFrame:
        """
        특정 날짜의 CSV가 있으면 로드, 없으면 API로 받아 저장
        """
        file_path = self._get_filename(symbol, date)

        if file_path.exists():
            df = pd.read_csv(file_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df

        # 없으면 새로 다운로드
        df = self._fetch_from_api(symbol, date)
        if not df.empty:
            df.to_csv(file_path, index=False)
            print(f"✅ 저장 완료: {file_path.name} ({len(df)}개)")
        return df

    # -----------------------------
    # 퍼블릭: 저장 (단일 CSV 용도 - 호환 위해 남김)
    # -----------------------------
    def save_price_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        (레거시 호환) 임의 CSV로 저장 — 날짜별 파일이 아닌 단일 파일 저장이 필요할 때만 사용
        """
        filepath = self.data_dir / filename
        df_save = self._coerce_schema(df.copy())
        self._atomic_save_csv(df_save, filepath)
        print(f"✅ 데이터 저장 완료: {filepath}")

    # -----------------------------
    # 퍼블릭: 로드 (날짜별 파일 + API 보충)
    # -----------------------------
    def load_price_data(self, start_date: str = "2024-01-01", end_date: str = None, symbol: str = None) -> pd.DataFrame:
        """
        지정 구간(start_date ~ end_date) 데이터 통합 로드
        - end_date가 None이면 오늘(UTC)까지 자동 포함
        """
        symbol = symbol or config.DEFAULT_SYMBOL
        end_date = end_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        start_dt = pd.to_datetime(start_date, utc=True)
        end_dt = pd.to_datetime(end_date, utc=True)

        all_days = pd.date_range(start=start_dt, end=end_dt, freq="D")
        all_data = []

        for day in all_days:
            df_day = self._load_or_download_day(symbol, day)
            if not df_day.empty:
                all_data.append(df_day)

        if not all_data:
            print("❌ 로드된 데이터가 없습니다.")
            return pd.DataFrame()

        df = pd.concat(all_data, ignore_index=True)
        df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        print(f"✅ 통합 완료: {len(df):,}개 레코드 ({start_date} ~ {end_date})")
        return df.reset_index(drop=True)
    
    def fetch_from_api(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        바이낸스 API에서 1분봉 다운로드
        """
        start_ts = pd.to_datetime(start_date, utc=True)
        end_ts = pd.to_datetime(end_date, utc=True)
        print(f"⏳ {symbol} 데이터 수집 ({start_date} ~ {end_date})")

        klines = self.api_client.get_historical_klines(
            symbol=symbol,
            interval="1m",
            start_str=start_ts.strftime("%Y-%m-%d %H:%M:%S"),
            end_str=end_ts.strftime("%Y-%m-%d %H:%M:%S")
        )

        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].astype(float)
        df = df.sort_values("timestamp")
        return df
    # -----------------------------
    # 퍼블릭: 피처 캐시/학습 데이터 준비
    # -----------------------------
    def get_cache_key(self, df: pd.DataFrame, params: dict) -> str:
        """캐시 키(SHA1) 생성"""
        if 'timestamp' in df.columns and not df.empty:
            data_str = f"{df['timestamp'].min()}_{df['timestamp'].max()}_{len(df)}"
        else:
            data_str = f"{len(df)}_{float(df.get('close', pd.Series(dtype=float)).sum()):.2f}"
        params_str = json.dumps(params, sort_keys=True, default=str)
        combined = f"{data_str}_{params_str}"
        return hashlib.sha1(combined.encode()).hexdigest()

    def load_cached_features(self, cache_key: str) -> Optional[pd.DataFrame]:
        """캐시된 피처 로드"""
        cache_file = self.cache_dir / f"features_{cache_key}.parquet"
        if not cache_file.exists():
            return None
        try:
            df = pd.read_parquet(cache_file)
            print(f"✅ 캐시 로드: {cache_key[:8]}... ({len(df)}개)")
            return df
        except Exception as e:
            print(f"⚠️  캐시 로드 실패: {e}")
            return None

    def save_cached_features(self, cache_key: str, df: pd.DataFrame) -> None:
        """피처 캐싱"""
        cache_file = self.cache_dir / f"features_{cache_key}.parquet"
        try:
            df.to_parquet(cache_file, index=False)
            print(f"✅ 캐시 저장: {cache_key[:8]}... ({len(df)}개)")
        except Exception as e:
            print(f"⚠️  캐시 저장 실패: {e}")

    def prepare_training_data(self, df_1m: pd.DataFrame,
                              use_cache: bool = True,
                              lookback_30m_bars: int = 100) -> Tuple[pd.DataFrame, pd.Series]:
        """
        학습 데이터 준비 (X, y)
        - 최소 lookback_30m_bars * BAR_MINUTES 만큼의 1분봉 필요
        """
        min_1m_required = lookback_30m_bars * config.BAR_MINUTES
        if df_1m.empty or len(df_1m) < min_1m_required:
            print(f"⚠️  데이터 부족 (최소 {min_1m_required}개 1분봉 필요, 현재 {len(df_1m)}개)")
            return pd.DataFrame(), pd.Series(dtype=np.float64)

        cache_params = {
            'lookback_30m_bars': lookback_30m_bars,
            'bar_minutes': config.BAR_MINUTES,
            'feature_ver': config.FEATURE_VERSION,
            'data_ver': config.DATA_VERSION
        }
        cache_key = self.get_cache_key(df_1m, cache_params)

        if use_cache:
            df_features = self.load_cached_features(cache_key)
            if df_features is not None:
                return self._split_features_target(df_features)

        print(f"🔄 피처 생성 중... (lookback={lookback_30m_bars}개 30분봉)")
        df_features = self.feature_engineer.create_feature_pool(df_1m, lookback_30m_bars)
        if df_features.empty:
            print("❌ 피처 생성 실패")
            return pd.DataFrame(), pd.Series(dtype=np.float64)

        if use_cache:
            self.save_cached_features(cache_key, df_features)

        return self._split_features_target(df_features)

    # -----------------------------
    # 퍼블릭: 분할/검증/정리
    # -----------------------------
    def _split_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """피처와 타겟 분리"""
        if 'target' not in df.columns:
            print("⚠️  타겟 컬럼 없음")
            return df, pd.Series(dtype=np.float64)
        feature_cols = self.feature_engineer.get_feature_names(df)
        X = df[feature_cols].copy()
        y = df['target'].copy()
        return X, y

    def time_series_split(self, X: pd.DataFrame, y: pd.Series,
                          n_splits: int = 5,
                          test_size: Optional[int] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """시계열 교차 검증 분할 (겹치지 않음)"""
        if len(X) < config.MIN_TRAIN_SIZE * 2:
            print(f"⚠️  데이터 부족 (최소 {config.MIN_TRAIN_SIZE * 2}개 필요)")
            return []

        if test_size is None:
            test_size = len(X) // (n_splits + 1)

        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        splits = []
        for train_idx, test_idx in tscv.split(X):
            if len(train_idx) >= config.MIN_TRAIN_SIZE:
                splits.append((train_idx, test_idx))

        print(f"✅ 시계열 분할: {len(splits)}개 생성 (겹치지 않음)")
        for i, (train_idx, test_idx) in enumerate(splits, 1):
            print(f"  Split {i}: Train={len(train_idx):5d}, Test={len(test_idx):5d}")
        return splits

    def split_train_valid_test(self, X: pd.DataFrame, y: pd.Series) -> Tuple:
        """학습/검증/테스트 분할 (시계열 순서 유지)"""
        n = len(X)
        train_end = int(n * config.TRAIN_RATIO)
        valid_end = int(n * (config.TRAIN_RATIO + config.VALID_RATIO))

        X_train = X.iloc[:train_end].copy()
        X_valid = X.iloc[train_end:valid_end].copy()
        X_test = X.iloc[valid_end:].copy()

        y_train = y.iloc[:train_end].copy()
        y_valid = y.iloc[train_end:valid_end].copy()
        y_test = y.iloc[valid_end:].copy()

        print(f"✅ 데이터 분할 (시계열 순서 유지):")
        print(f"  Train: {len(X_train):5d} ({len(X_train)/n*100:.1f}%) [0:{train_end}]")
        print(f"  Valid: {len(X_valid):5d} ({len(X_valid)/n*100:.1f}%) [{train_end}:{valid_end}]")
        print(f"  Test:  {len(X_test):5d} ({len(X_test)/n*100:.1f}%) [{valid_end}:{n}]")
        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def validate_data_quality(self, X: pd.DataFrame, y: pd.Series) -> Tuple[bool, List[str]]:
        """데이터 품질 검증"""
        errors: List[str] = []

        if len(X) != len(y):
            errors.append(f"샘플 수 불일치: X={len(X)}, y={len(y)}")

        X_nan = X.isnull().sum().sum()
        y_nan = y.isnull().sum()
        if X_nan > 0:
            X_nan_ratio = X_nan / (len(X) * max(len(X.columns), 1))
            errors.append(f"X에 결측치 {X_nan}개 존재 ({X_nan_ratio:.2%})")
        if y_nan > 0:
            errors.append(f"y에 결측치 {y_nan}개 존재")

        if len(y) > 0:
            target_ratio = float(y.mean())
            if target_ratio < 0.40 or target_ratio > 0.60:
                errors.append(f"타겟 불균형 (40~60% 권장): UP={target_ratio:.2%}")

        object_cols = [col for col in X.columns if X[col].dtype == 'object']
        if object_cols:
            errors.append(f"피처가 object 타입: {object_cols[:5]}...")

        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            X_inf = np.isinf(X[numeric_cols]).sum().sum()
            if X_inf > 0:
                errors.append(f"X에 무한대 {int(X_inf)}개 존재")

        constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_cols:
            errors.append(f"상수 피처 존재: {constant_cols[:5]}...")

        if len(y) > 0:
            unique_targets = set(y.unique().tolist())
            if not unique_targets.issubset({0, 1}):
                errors.append(f"타겟 값 이상: {sorted(unique_targets)}")

        return len(errors) == 0, errors

    def clean_cache(self, older_than_days: int = 7) -> None:
        """오래된 피처 캐시 삭제"""
        if not self.cache_dir.exists():
            return
        cutoff_time = datetime.now().timestamp() - (older_than_days * 86400)
        deleted_count = 0
        for cache_file in self.cache_dir.glob("features_*.parquet"):
            if cache_file.stat().st_mtime < cutoff_time:
                cache_file.unlink(missing_ok=True)
                deleted_count += 1
        if deleted_count > 0:
            print(f"🗑️  캐시 정리: {deleted_count}개 파일 삭제 ({older_than_days}일 이상)")

    def get_data_info(self, df: pd.DataFrame) -> dict:
        """데이터 정보 요약"""
        info = {
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024 if not df.empty else 0.0,
            'dtypes': df.dtypes.value_counts().to_dict() if not df.empty else {},
            'null_counts': df.isnull().sum().to_dict() if not df.empty else {},
        }
        if 'timestamp' in df.columns and not df.empty:
            info['time_range'] = {
                'start': str(df['timestamp'].min()),
                'end': str(df['timestamp'].max()),
                'duration_days': float((df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400.0),
            }
        return info

    # -----------------------------
    # 내부 헬퍼
    # -----------------------------
    def _coerce_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """스키마/UTC 정규화 + 정렬/중복제거"""
        if df.empty:
            return df
        for col, dtype in config.PRICE_DATA_DTYPES.items():
            if col in df.columns:
                if dtype == 'datetime64[ns]':
                    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
                else:
                    try:
                        df[col] = df[col].astype(dtype)
                    except Exception:
                        # 일부 컬럼 타입 강제 실패시 무시
                        pass
        if 'timestamp' in df.columns:
            df = (df.dropna(subset=['timestamp'])
                    .sort_values('timestamp')
                    .drop_duplicates('timestamp', keep='last')
                    .reset_index(drop=True))
        return df

    def _atomic_save_csv(self, df: pd.DataFrame, path: Path) -> None:
        """원자적 CSV 저장"""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / (".tmp_" + path.name)
        df.to_csv(tmp, index=False)
        tmp.replace(path)
        print(f"💾 CSV 저장: {path.name} ({len(df)}개)")

    def _fetch_day(self, day_utc: pd.Timestamp) -> pd.DataFrame:
        """해당 UTC '하루'의 1분봉을 API로 가져오기"""
        day_start = pd.Timestamp(day_utc.floor("D"))
        start_ms = int(day_start.timestamp() * 1000)
        end_ms = int((day_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)).timestamp() * 1000)

        all_rows = []
        cur = start_ms
        step_ms = 1000 * 60 * 1000  # 1000분(≈16.6h)

        while cur <= end_ms:
            params = {
                "symbol": self.symbol,
                "interval": "1m",
                "limit": 1000,
                "startTime": cur,
                "endTime": min(cur + step_ms - 1, end_ms),
            }
            try:
                resp = self.session.get(self._endpoint, params=params, timeout=10)
                if resp.status_code != 200:
                    # 레이트리밋/간헐 오류 — 다음 구간 시도
                    time.sleep(0.5)
                    cur += step_ms
                    continue
                rows = resp.json()
                if not rows:
                    cur += step_ms
                    time.sleep(0.1)
                    continue
                all_rows.extend(rows)
                cur += step_ms
                time.sleep(0.2)
            except Exception:
                time.sleep(0.5)
                cur += step_ms

        if not all_rows:
            return pd.DataFrame()

        df = pd.DataFrame(
            [{
                "timestamp": pd.to_datetime(r[0], unit="ms", utc=True),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
            } for r in all_rows]
        )
        return self._coerce_schema(df)


# ============================================================
# 간이 테스트
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DataLoader 테스트 (v2.0.0)")
    print("=" * 60)

    loader = DataLoader()
    print("\n✅ 초기화")
    print(f"  - SYMBOL: {loader.symbol}")
    print(f"  - MARKET: {loader.market}")
    print(f"  - DATA_DIR: {loader.data_dir}")
    print(f"  - CACHE_DIR: {loader.cache_dir}")

    # 날짜 범위 (예시)
    start = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    print("\n📂 데이터 로드")
    df = loader.load_price_data(start, end)

    if not df.empty:
        info = loader.get_data_info(df)
        print("\n📊 데이터 정보")
        print(f"  - 행/열: {info['n_rows']:,} / {info['n_cols']}")
        if 'time_range' in info:
            tr = info['time_range']
            print(f"  - 기간: {tr['start']} ~ {tr['end']} (약 {tr['duration_days']:.2f}일)")

        print("\n🎯 학습 데이터 준비")
        X, y = loader.prepare_training_data(df, use_cache=True, lookback_30m_bars=100)
        if not X.empty:
            print(f"  - X: {X.shape}, y: {y.shape}")
            ok, errs = loader.validate_data_quality(X, y)
            print(f"  - 품질: {'OK' if ok else 'WARN'}")
            if not ok:
                for e in errs:
                    print("    -", e)

            print("\n✂️  시계열 분할")
            _ = loader.time_series_split(X, y, n_splits=3)

            print("\n🧪 Train/Valid/Test 분할")
            _ = loader.split_train_valid_test(X, y)
        else:
            print("  ❌ 학습 데이터 준비 실패")

    print("\n완료.")

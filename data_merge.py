"""
데이터 병합 - 가격/로그/거래/피처 통합 + 품질 검증
버전: 1.3.2 (30분 바이너리 옵션 최적화 + UTC 통일 + Feature Log 병합)
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import config
from data_loader import DataLoader
from log_manager import LogManager
from timeframe_manager import TimeframeManager


class DataMerger:
    """
    데이터 병합 및 품질 관리
    - 가격(1m→30m) + 거래로그 + 피처로그 병합
    - 타임프레임 정렬 (UTC 기준)
    - 데이터 품질 검증/정제
    - 재학습용 클린 데이터셋 생성
    - 30분 바이너리 옵션 구조 반영
    """

    def __init__(self):
        self.data_loader = DataLoader()
        self.log_manager = LogManager()
        self.tf_manager = TimeframeManager()
        # 기본 구성값 (config에 없을 때 대비)
        self.symbol = getattr(config, "SYMBOL", "BTCUSDT")
        self.feature_log_dir = Path(getattr(config, "FEATURE_LOG_DIR", "logs/feature"))

    # ===============================
    # 로더
    # ===============================
    def _load_trade_logs_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """날짜 범위(UTC)의 거래 로그 로드 및 UTC 정규화"""
        start_dt = pd.to_datetime(start_date, utc=True)
        end_dt = pd.to_datetime(end_date, utc=True)
        dfs: List[pd.DataFrame] = []
        cur = start_dt

        while cur <= end_dt:
            date_str = cur.strftime("%Y%m%d")
            try:
                df_day = self.log_manager.load_trade_log(date_str)
            except Exception:
                df_day = pd.DataFrame()

            if not df_day.empty:
                # 심볼 필터(있으면)
                if "symbol" in df_day.columns:
                    df_day = df_day[df_day["symbol"] == self.symbol] if self.symbol else df_day

                # 타임스탬프 UTC
                for col in ("bar30_start", "bar30_end", "entry_ts", "exit_ts"):
                    if col in df_day.columns:
                        df_day[col] = pd.to_datetime(df_day[col], utc=True, errors="coerce")

                dfs.append(df_day)

            cur += pd.Timedelta(days=1)

        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def _load_feature_logs_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        날짜 범위(UTC)의 피처 로그 로드.
        파일 규칙: {FEATURE_LOG_DIR}/{SYMBOL}_{YYYYMMDD}.csv
        필수 키: bar30_start, bar30_end (둘 중 하나라도 있으면 병합에 사용)
        주요 컬럼: p_up, margin, f:* (피처 스냅샷)
        """
        start_dt = pd.to_datetime(start_date, utc=True)
        end_dt = pd.to_datetime(end_date, utc=True)

        dfs: List[pd.DataFrame] = []
        cur = start_dt

        while cur <= end_dt:
            fname = f"{self.symbol}_{cur.strftime('%Y%m%d')}.csv"
            fpath = self.feature_log_dir / fname
            if fpath.exists():
                try:
                    df = pd.read_csv(fpath)
                except Exception:
                    df = pd.DataFrame()

                if not df.empty:
                    # UTC 변환
                    for col in ("bar30_start", "bar30_end", "pred_time", "ts"):
                        if col in df.columns:
                            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
                    # 수치형
                    for col in ("p_up", "margin", "regime"):
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors="coerce")

                    # 병합 키 최소 보장: bar30_start가 없고 bar30_end만 있는 경우, end-30분으로 추정
                    if "bar30_start" not in df.columns and "bar30_end" in df.columns:
                        df["bar30_start"] = df["bar30_end"] - pd.Timedelta(minutes=30)
                    if "bar30_end" not in df.columns and "bar30_start" in df.columns:
                        df["bar30_end"] = df["bar30_start"] + pd.Timedelta(minutes=30)

                    dfs.append(df)

            cur += pd.Timedelta(days=1)

        if not dfs:
            return pd.DataFrame()

        df_feat = pd.concat(dfs, ignore_index=True)

        # 중복 제거 (가장 최근 기록 우선)
        if "bar30_start" in df_feat.columns:
            df_feat = (df_feat.sort_values(["bar30_start"])
                              .drop_duplicates(subset=["bar30_start"], keep="last"))

        return df_feat

    # ===============================
    # 병합
    # ===============================
    def merge_all_data(self, start_date: str, end_date: str, filename: Optional[str] = None) -> pd.DataFrame:
        """
        전체 데이터 병합 (가격 1m→30m, 거래 로그, 피처 로그)
        Returns: 병합된 30분 프레임 DataFrame (UTC tz-aware)
        """
        print(f"\n📊 데이터 병합 시작: {start_date} ~ {end_date} (symbol={self.symbol})")

        # 1) 가격 로드 (1m)
        if filename:
            df_price_1m = self.data_loader.load_price_data(start_date, end_date, filename)
        else:
            # 사용 가능 파일 자동 탐색
            tried = []
            df_price_1m = pd.DataFrame()
            for fname in [
                f"{self.symbol.lower()}_1m.csv",
                "btcusdt_1m.csv",
                "test_merge_data.csv",
                "test_data_loader.csv",
                "test_merger.csv",
            ]:
                tried.append(fname)
                try:
                    df_price_1m = self.data_loader.load_price_data(start_date, end_date, fname)
                    if not df_price_1m.empty:
                        print(f"  ✅ 가격 파일 로드: {fname}")
                        break
                except Exception:
                    pass

        if df_price_1m.empty:
            print("⚠️  가격 데이터 없음")
            return pd.DataFrame()

        # 2) 1m → 30m 집계
        df_30m = self.tf_manager.aggregate_1m_to_30m(df_price_1m, realtime_safe=False)
        if df_30m.empty:
            print("⚠️  30분봉 집계 실패")
            return pd.DataFrame()

        # UTC/키 보장
        df_30m["bar30_start"] = pd.to_datetime(df_30m["bar30_start"], utc=True, errors="coerce")
        if "bar30_end" not in df_30m.columns:
            df_30m["bar30_end"] = df_30m["bar30_start"] + pd.Timedelta(minutes=30)

        # 3) 거래 로그 로드
        df_trades = self._load_trade_logs_range(start_date, end_date)
        has_trade = not df_trades.empty
        if has_trade:
            print(f"  ✅ 거래 로그 로드: {len(df_trades):,}행")
        else:
            print("  ⚠️ 거래 로그 없음")

        # 4) 가격+거래 병합
        df_merged = self._merge_price_and_trades(df_30m, df_trades) if has_trade else df_30m.copy()
        if not has_trade:
            # 기본 통계 컬럼
            df_merged["n_trades"] = 0
            df_merged["n_wins"] = 0
            df_merged["n_losses"] = 0
            df_merged["win_rate"] = 0.0

        # 5) 피처 로그 병합 (선택적)
        df_feat = self._load_feature_logs_range(start_date, end_date)
        if not df_feat.empty:
            keep_cols = ["bar30_start", "bar30_end"]
            # 붙일 컬럼(p_up, margin, f:*)
            attach_cols = []
            for col in df_feat.columns:
                if col in ("bar30_start", "bar30_end"):
                    continue
                if col.startswith("f:") or col in ("p_up", "margin", "pred_time", "regime"):
                    attach_cols.append(col)

            # 병합: 동일 키
            df_merged = df_merged.merge(
                df_feat[keep_cols + attach_cols],
                on=["bar30_start", "bar30_end"],
                how="left",
            )
            print(f"  ✅ 피처 로그 병합: cols +{len(attach_cols)}")
        else:
            print("  ⚠️ 피처 로그 없음 (logs/feature)")

        # 캐릭터 컬럼 기본형 변환 예시 (선택) : 없음

        # 6) 정렬/리셋
        df_merged = df_merged.sort_values("bar30_start").reset_index(drop=True)

        print(f"\n✅ 병합 완료: {len(df_merged):,}개 레코드")
        return df_merged

    def _merge_price_and_trades(self, df_price_30m: pd.DataFrame, df_trades: pd.DataFrame) -> pd.DataFrame:
        """
        가격(30m) + 거래 로그 병합
        - n_trades: ENTRY 수
        - n_wins / n_losses / win_rate: CLOSE & CLOSED 기준
        """
        if df_price_30m.empty:
            return df_trades.copy() if not df_trades.empty else pd.DataFrame()

        df_price = df_price_30m.copy()
        df_price["bar30_start"] = pd.to_datetime(df_price["bar30_start"], utc=True, errors="coerce")
        if "bar30_end" not in df_price.columns or df_price["bar30_end"].isna().any():
            df_price["bar30_end"] = df_price["bar30_start"] + pd.Timedelta(minutes=30)

        if df_trades.empty:
            return df_price

        df_tr = df_trades.copy()
        df_tr["bar30_start"] = pd.to_datetime(df_tr["bar30_start"], utc=True, errors="coerce")
        if "bar30_end" in df_tr.columns:
            df_tr["bar30_end"] = pd.to_datetime(df_tr["bar30_end"], utc=True, errors="coerce")

        # 심볼 필터
        if "symbol" in df_tr.columns and self.symbol:
            df_tr = df_tr[df_tr["symbol"] == self.symbol]

        # 중복 제거
        if "bar30_start" in df_price.columns:
            df_price = df_price.drop_duplicates(subset=["bar30_start"], keep="last")
        if "trade_id" in df_tr.columns:
            df_tr = df_tr.drop_duplicates(subset=["trade_id"], keep="last")

        # 1) ENTRY 수
        if "log_type" in df_tr.columns:
            df_entry = df_tr[df_tr["log_type"] == "ENTRY"].copy()
        else:
            # 폴백: 상태만 있는 경우, OPEN/ACTIVE를 ENTRY로 간주
            df_entry = df_tr[df_tr.get("status", pd.Series(dtype=str)).isin(["OPEN", "ACTIVE"])].copy()

        trade_counts = (
            df_entry.groupby("bar30_start")
            .size()
            .reset_index(name="n_trades")
        )

        # 2) CLOSE & CLOSED 승패 집계
        if "log_type" in df_tr.columns:
            df_closed = df_tr[(df_tr["log_type"] == "CLOSE") & (df_tr["status"] == "CLOSED")].copy()
        else:
            df_closed = df_tr[df_tr.get("status", pd.Series(dtype=str)) == "CLOSED"].copy()

        if not df_closed.empty:
            # 결과 집계
            result_counts = (
                df_closed.groupby(["bar30_start", "result"])
                .size()
                .unstack(fill_value=0)
            )
            result_counts["n_wins"] = result_counts.get("WIN", 0)
            result_counts["n_losses"] = result_counts.get("LOSS", 0)
            result_counts = (
                result_counts[["n_wins", "n_losses"]]
                .reset_index()
            )
            result_counts["win_rate"] = (
                result_counts["n_wins"]
                / (result_counts["n_wins"] + result_counts["n_losses"]).replace(0, np.nan)
            ).fillna(0.0)
        else:
            result_counts = pd.DataFrame(columns=["bar30_start", "n_wins", "n_losses", "win_rate"])

        # 병합
        df_merged = (
            df_price.merge(trade_counts, on="bar30_start", how="left")
                    .merge(result_counts, on="bar30_start", how="left")
        )

        # 결측/형
        for col in ("n_trades", "n_wins", "n_losses"):
            df_merged[col] = df_merged[col].fillna(0).astype(int)
        df_merged["win_rate"] = df_merged["win_rate"].fillna(0.0)

        return df_merged

    # ===============================
    # 검증 / 정제
    # ===============================
    def validate_merged_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if df.empty:
            return False, ["데이터가 비어있음"]

        # 1. 필수 컬럼
        required_cols = ["bar30_start", "bar30_end", "open", "high", "low", "close", "volume"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            errors.append(f"필수 컬럼 누락: {missing}")

        # 2. 타임스탬프 정렬/UTC
        if "bar30_start" in df.columns:
            s = pd.to_datetime(df["bar30_start"], utc=True, errors="coerce")
            if not s.is_monotonic_increasing:
                errors.append("타임스탬프가 정렬되지 않음")
            if s.isna().any():
                errors.append("타임스탬프 NaT 존재")

        # 3. OHLC 정합성
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            invalid_hl = (df["high"] < df["low"]).sum()
            if invalid_hl > 0:
                errors.append(f"High < Low: {invalid_hl}건")

            invalid_ohlc = (
                (df["high"] < df["open"]) |
                (df["high"] < df["close"]) |
                (df["low"] > df["open"])  |
                (df["low"] > df["close"])
            ).sum()
            if invalid_ohlc > 0:
                errors.append(f"OHLC 범위 위반: {invalid_ohlc}건")

        # 4. 중복
        if "bar30_start" in df.columns:
            dup = df["bar30_start"].duplicated().sum()
            if dup > 0:
                errors.append(f"중복 타임스탬프: {dup}건")

        # 5. NaN (critical)
        for c in ["open", "high", "low", "close", "volume"]:
            if c in df.columns:
                n = df[c].isna().sum()
                if n > 0:
                    errors.append(f"{c} 결측치: {n}건")

        # 6. 이상치
        if "close" in df.columns:
            inv = (df["close"] <= 0).sum()
            if inv > 0:
                errors.append(f"유효하지 않은 가격: {inv}건")

        # 7. 30분 간격
        if "bar30_start" in df.columns and len(df) > 1:
            s = pd.to_datetime(df["bar30_start"], utc=True, errors="coerce")
            diffs = s.diff().dt.total_seconds() / 60
            invalid = ((diffs < 29.5) | (diffs > 31.0)) & diffs.notna()
            if invalid.sum() > 0:
                errors.append(f"30분 간격 위반: {invalid.sum()}건 (갭 가능)")

        return len(errors) == 0, errors

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """정제: UTC, 중복 제거, 정렬, OHLC 보정, NaN/이상치 제거"""
        if df.empty:
            return df

        d = df.copy()
        # UTC
        for col in ("bar30_start", "bar30_end"):
            if col in d.columns:
                d[col] = pd.to_datetime(d[col], utc=True, errors="coerce")

        # 중복/정렬
        if "bar30_start" in d.columns:
            d = d.drop_duplicates(subset=["bar30_start"], keep="first").sort_values("bar30_start")

        # OHLC 보정
        if all(c in d.columns for c in ["open", "high", "low", "close"]):
            d["high"] = d[["open", "high", "close"]].max(axis=1)
            d["low"] = d[["open", "low", "close"]].min(axis=1)

        # 결측치 제거 (핵심)
        critical = [c for c in ["open", "high", "low", "close", "volume"] if c in d.columns]
        if critical:
            d = d.dropna(subset=critical)

        # 이상치 제거
        if "close" in d.columns:
            d = d[d["close"] > 0]
        if "volume" in d.columns:
            d = d[d["volume"] >= 0]

        return d.reset_index(drop=True)

    def align_timeframes(self, df_price_30m: pd.DataFrame, df_trades: pd.DataFrame) -> pd.DataFrame:
        """타임프레임 정렬 (30m 기준, UTC) 후 병합"""
        if df_price_30m.empty:
            return pd.DataFrame()
        if df_trades.empty:
            return df_price_30m

        p_aligned, t_aligned = self.tf_manager.align_dataframes(df_price_30m, df_trades)
        return self._merge_price_and_trades(p_aligned, t_aligned)

    # ===============================
    # 재학습 데이터셋
    # ===============================
    def create_retrain_dataset(
        self,
        start_date: str,
        end_date: str,
        lookback_30m_bars: int = 100
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        재학습용 데이터셋 생성
        - DataLoader.prepare_training_data 호출 (캐시 미사용)
        """
        print(f"\n🔄 재학습 데이터셋 생성: {start_date} ~ {end_date}")
        df_1m = self.data_loader.load_price_data(start_date, end_date)
        if df_1m.empty:
            print("❌ 1분봉 데이터 없음")
            return pd.DataFrame(), pd.Series(dtype=float)

        print(f"🔧 피처 생성 중... (lookback={lookback_30m_bars})")
        X, y = self.data_loader.prepare_training_data(
            df_1m,
            use_cache=False,
            lookback_30m_bars=lookback_30m_bars
        )

        if X.empty or y.empty:
            print("❌ 피처 생성 실패")
            return pd.DataFrame(), pd.Series(dtype=float)

        print("🧪 데이터 품질 검증...")
        valid, errors = self.data_loader.validate_data_quality(X, y)
        if not valid:
            print("⚠️  품질 이슈:")
            for e in errors:
                print(f"  - {e}")

        print(f"✅ 재학습 세트: X={X.shape}, y={y.shape}, UP={y.mean():.2%}")
        return X, y

    # ===============================
    # 리포트
    # ===============================
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict:
        if df.empty:
            return {
                "total_records": 0,
                "missing_ratio": 0.0,
                "duplicate_ratio": 0.0,
                "outlier_ratio": 0.0,
                "quality_score": 0.0,
                "time_range": None,
            }

        total = len(df)
        missing = df.isnull().sum().sum()
        total_vals = df.shape[0] * df.shape[1]
        missing_ratio = (missing / total_vals) if total_vals else 0.0

        duplicate_ratio = 0.0
        if "bar30_start" in df.columns:
            duplicate_ratio = df["bar30_start"].duplicated().sum() / total

        outlier_ratio = 0.0
        if "close" in df.columns:
            outlier_ratio = (df["close"] <= 0).sum() / total

        time_range = None
        if "bar30_start" in df.columns:
            s = pd.to_datetime(df["bar30_start"], utc=True, errors="coerce")
            if not s.empty:
                time_range = {
                    "start": str(s.min()),
                    "end": str(s.max()),
                    "duration_hours": (s.max() - s.min()).total_seconds() / 3600,
                }

        quality_score = 100 * (1 - missing_ratio - duplicate_ratio - outlier_ratio)
        quality_score = max(0, min(100, quality_score))

        return {
            "total_records": total,
            "missing_ratio": missing_ratio,
            "duplicate_ratio": duplicate_ratio,
            "outlier_ratio": outlier_ratio,
            "quality_score": quality_score,
            "time_range": time_range,
        }

    def analyze_trade_performance(self, df_trades: pd.DataFrame) -> Dict:
        """거래 성과 분석 (CLOSED 기준)"""
        if df_trades.empty:
            return {
                "total_trades": 0,
                "closed_trades": 0,
                "win_rate": 0.0,
                "avg_confidence": 0.0,
                "regime_distribution": {},
            }

        df_closed = df_trades[df_trades.get("status", pd.Series(dtype=str)) == "CLOSED"].copy()
        if df_closed.empty:
            return {
                "total_trades": len(df_trades),
                "closed_trades": 0,
                "win_rate": 0.0,
                "avg_confidence": 0.0,
                "regime_distribution": {},
            }

        win_rate = (df_closed.get("result", pd.Series(dtype=str)) == "WIN").mean()

        # 평균 신뢰도 (p_at_entry 또는 p_up)
        if "p_at_entry" in df_closed.columns:
            avg_confidence = pd.to_numeric(df_closed["p_at_entry"], errors="coerce").mean()
        elif "p_up" in df_closed.columns:
            avg_confidence = pd.to_numeric(df_closed["p_up"], errors="coerce").mean()
        else:
            avg_confidence = 0.0

        # 레짐 분포
        regime_dist: Dict[str, int] = {}
        if "regime" in df_closed.columns:
            for reg, cnt in df_closed["regime"].value_counts().items():
                name = {1: "UP", 0: "FLAT", -1: "DOWN"}.get(reg, f"REGIME_{reg}")
                regime_dist[name] = int(cnt)

        return {
            "total_trades": int(len(df_trades)),
            "closed_trades": int(len(df_closed)),
            "win_rate": float(win_rate),
            "avg_confidence": float(avg_confidence) if not np.isnan(avg_confidence) else 0.0,
            "regime_distribution": regime_dist,
        }


# ============================================================
# 선택: 간단 실행 테스트 (로컬 더미 데이터에서만)
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DataMerger 테스트 (v1.3.2)")
    print("=" * 60)

    merger = DataMerger()
    print("✅ 초기화 완료")

    # 더미 파일이 있는 경우만 동작하도록 시도
    try:
        df_merged = merger.merge_all_data("2025-01-01", "2025-01-01")
        if not df_merged.empty:
            print(f"병합 레코드: {len(df_merged)}")
            ok, errs = merger.validate_merged_data(df_merged)
            print("검증:", "OK" if ok else f"WARN -> {errs}")
            rep = merger.get_data_quality_report(df_merged)
            print("품질 점수:", f"{rep['quality_score']:.1f}/100")
        else:
            print("병합 결과가 비어있음 (더미 파일 미존재 가능)")
    except Exception as e:
        print("테스트 중 예외:", e)

"""
로그 관리자 - 거래/피처 로그 표준화 및 무결성 검증 (UTC ISO, 원자적 쓰기, trade_id 전역 업데이트)
버전: 1.3.1 (real_trade.py A-방식 호환)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import shutil

import config


# ----------------------------
# 시간/파일 유틸
# ----------------------------
def _to_utc(ts: datetime) -> pd.Timestamp | None:
    """datetime을 UTC Timestamp로 변환"""
    if ts is None:
        return None
    t = pd.Timestamp(ts)
    if t.tz is not None:
        return t.tz_convert("UTC")
    return t.tz_localize("UTC")


def _iso(dt: datetime | pd.Timestamp | str | None) -> str:
    """UTC ISO8601(Z) 문자열"""
    if dt is None or (isinstance(dt, float) and np.isnan(dt)):
        return ""
    t = pd.to_datetime(dt, utc=True, errors="coerce")
    return "" if pd.isna(t) else t.isoformat().replace("+00:00", "Z")


def _read_csv_safe(fpath: Path) -> pd.DataFrame:
    if not fpath.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(fpath)
    except Exception as e:
        print(f"⚠️  CSV 로드 실패: {fpath.name} ({e})")
        return pd.DataFrame()


def _atomic_write_csv(df: pd.DataFrame, fpath: Path):
    fpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = fpath.parent / (".tmp_" + fpath.name)
    df.to_csv(tmp, index=False)
    shutil.move(str(tmp), str(fpath))


# ----------------------------
# LogManager
# ----------------------------
class LogManager:
    """
    - 거래 로그(통합/엔트리/청산), 피처 로그
    - 스키마 검증, UTC ISO 기록, 원자적 쓰기
    - trade_id 기반 결과 업데이트(최근 N일 검색)
    - real_trade.py A-방식과 키(bar30_start/bar30_end) 1:1 매칭
    """

    def __init__(self, search_days: int = 14):
        # 디렉토리
        self.trade_log_dir: Path = config.TRADE_LOG_DIR
        self.entry_dir: Path = config.TRADE_ENTRY_DIR
        self.close_dir: Path = config.TRADE_CLOSE_DIR
        self.meta_dir: Path = getattr(config, "TRADE_META_DIR", self.trade_log_dir / "meta")
        self.feature_log_dir: Path = config.FEATURE_LOG_DIR

        # 스키마(있으면 사용, 없으면 유니온)
        self.trade_columns: List[str] = getattr(config, "TRADE_LOG_COLUMNS", [])
        self.feature_columns: List[str] = getattr(config, "FEATURE_LOG_COLUMNS", [])
        self.trade_dtypes: Dict[str, str] = getattr(config, "TRADE_LOG_DTYPES", {})

        self.search_days = int(search_days)

    # =================================================
    # 공통: 유니온 append (trade/feature 겸용)
    # =================================================
    def _append_union(self, fpath: Path, df_new: pd.DataFrame, expected_columns: Optional[List[str]] = None):
        """
        기존 파일과 컬럼 유니온 후 append → 원자적 overwrite.
        expected_columns가 주어지면 해당 순서/세트로 정렬(부족한 컬럼은 생성).
        """
        df_old = _read_csv_safe(fpath)

        if expected_columns:
            # expected 기준으로 정렬/보강
            def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
                miss = [c for c in expected_columns if c not in df.columns]
                if miss:
                    for c in miss:
                        df[c] = np.nan
                return df[expected_columns]

            if df_old.empty:
                out = _ensure_cols(df_new.copy())
            else:
                out = pd.concat([_ensure_cols(df_old), _ensure_cols(df_new)], ignore_index=True)
        else:
            # 파일/신규의 유니온
            if df_old.empty:
                out = df_new
            else:
                all_cols = list(dict.fromkeys(list(df_old.columns) + list(df_new.columns)))
                out = pd.concat([df_old.reindex(columns=all_cols),
                                 df_new.reindex(columns=all_cols)], ignore_index=True)

        _atomic_write_csv(out, fpath)

    # =================================================
    # 엔트리 로그 (간소/상세)
    # =================================================
    def log_trade_entry_simple(
        self,
        trade_id: str,
        direction: str,
        entry_price: float,
        entry_ts: datetime,
        p_at_entry: Optional[float] = None,
        regime: int = 0,
        **kwargs,
    ) -> None:
        """
        단순 엔트리 기록.
        - real_trade.py에서 사용하는 키를 그대로 수용:
          symbol, kind(=REAL|PAPER), payout, tenor_min, bar30_start, bar30_end 등
        - p_at_entry 또는 p_raw_at_entry 둘 다 허용
        """
        date_str = _to_utc(entry_ts).strftime("%Y%m%d")

        # 확률 필드 정규화
        p_raw = kwargs.pop("p_raw_at_entry", None)
        if p_at_entry is None and p_raw is not None:
            p_at_entry = p_raw
        if p_at_entry is None:
            p_at_entry = np.nan

        # kind → mode(백워드 호환)
        kind = kwargs.pop("kind", None)
        mode = kwargs.pop("mode", None)
        if mode is None and kind is not None:
            mode = "LIVE" if kind == "REAL" else "PAPER"

        # 선택적 시간 필드 ISO화
        for k in ["bar30_start", "bar30_end", "cross_time", "label_ts"]:
            if k in kwargs:
                kwargs[k] = _iso(kwargs[k])

        base = {
            "trade_id": trade_id,
            "symbol": kwargs.pop("symbol", getattr(config, "SYMBOL", "BTCUSDT")),
            "side": "UP" if direction == "UP" else "DOWN",
            "entry_ts": _iso(entry_ts),
            "entry_price": float(entry_price) if entry_price is not None else np.nan,
            "p_raw_at_entry": float(p_at_entry) if p_at_entry is not None else np.nan,
            "p_cal_at_entry": float(kwargs.pop("p_cal_at_entry", p_at_entry if p_at_entry is not None else np.nan)),
            "cal_method": kwargs.pop("cal_method", "identity"),
            "cal_ver": kwargs.pop("cal_ver", ""),
            "regime": int(regime) if regime is not None else 0,
            "status": "OPEN",
            "mode": mode if mode is not None else "LIVE",
            "payout": kwargs.pop("payout", getattr(config, "PAYOUT_30M_PLUS", 0.85)),
            "tenor_min": kwargs.pop("tenor_min", getattr(config, "OPTION_TENOR_MIN", 30)),
            "model_ver": getattr(config, "MODEL_VERSION", ""),
            "feature_ver": getattr(config, "FEATURE_VERSION", ""),
            "filter_ver": getattr(config, "FILTER_VERSION", ""),
            "cutoff_ver": getattr(config, "CUTOFF_VERSION", ""),
            "data_ver": getattr(config, "DATA_VERSION", ""),
        }

        row = {**{c: None for c in self.trade_columns}, **base, **kwargs}
        df_row = pd.DataFrame([row])

        # 파일 경로
        f_union = self.trade_log_dir / f"{date_str}_trades.csv"
        f_entry = self.entry_dir / f"{date_str}_entries.csv"

        self._append_union(f_union, df_row, expected_columns=self.trade_columns or None)
        self._append_union(f_entry, df_row, expected_columns=self.trade_columns or None)

    def log_trade_entry(
        self,
        trade_id: str,
        bar30_start: datetime,
        bar30_end: datetime,
        entry_ts: datetime,
        m1_index_entry: Optional[int],
        entry_price: float,
        side: str,
        regime: int,
        regime_score: float,
        adx: float,
        di_plus: float,
        di_minus: float,
        p_raw_at_entry: float,
        p_cal_at_entry: float,
        cal_method: str,
        cal_ver: str,
        cut_on: float,
        cut_off: float,
        filters_applied: str,
        **kwargs,
    ) -> None:
        """상세 엔트리(확장 필드 포함)"""
        date_str = _to_utc(entry_ts).strftime("%Y%m%d")

        row = {**{c: None for c in self.trade_columns}, **{
            "trade_id": trade_id,
            "symbol": kwargs.get("symbol", getattr(config, "SYMBOL", "BTCUSDT")),
            "bar30_start": _iso(bar30_start),
            "bar30_end": _iso(bar30_end),
            "entry_ts": _iso(entry_ts),
            "m1_index_entry": m1_index_entry,
            "entry_price": entry_price,
            "side": "UP" if side == "UP" else "DOWN",
            "regime": regime,
            "regime_score": regime_score,
            "adx": adx, "di_plus": di_plus, "di_minus": di_minus,
            "p_raw_at_entry": p_raw_at_entry,
            "p_cal_at_entry": p_cal_at_entry,
            "cal_method": cal_method,
            "cal_ver": cal_ver,
            "cut_on": cut_on, "cut_off": cut_off,
            "filters_applied": filters_applied,
            "status": "OPEN",
            "mode": kwargs.get("mode", "LIVE"),
            "payout": kwargs.get("payout", getattr(config, "PAYOUT_30M_PLUS", 0.85)),
            "tenor_min": kwargs.get("tenor_min", getattr(config, "OPTION_TENOR_MIN", 30)),
            "model_ver": getattr(config, "MODEL_VERSION", ""),
            "feature_ver": getattr(config, "FEATURE_VERSION", ""),
            "filter_ver": getattr(config, "FILTER_VERSION", ""),
            "cutoff_ver": getattr(config, "CUTOFF_VERSION", ""),
            "data_ver": getattr(config, "DATA_VERSION", ""),
        }}
        if "cross_time" in kwargs:
            row["cross_time"] = _iso(kwargs["cross_time"])

        df_row = pd.DataFrame([row])

        f_union = self.trade_log_dir / f"{date_str}_trades.csv"
        f_entry = self.entry_dir / f"{date_str}_entries.csv"
        self._append_union(f_union, df_row, expected_columns=self.trade_columns or None)
        self._append_union(f_entry, df_row, expected_columns=self.trade_columns or None)

    # =================================================
    # 클로즈 로그 (단순)
    # =================================================
    def log_trade_close_simple(
        self,
        trade_id: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        entry_ts: datetime,
        close_ts: datetime,
        p_at_entry: float,
        regime: int,
        bar30_start: datetime,
        bar30_end: datetime,
        kind: str = "PAPER",
        symbol: str = None,
        payout: float = None,
        tenor_min: int = None,
        result: str = "WIN",
        pnl: float = 0.0,
        **kwargs,
    ) -> None:
        """단순 청산 기록 + 통합 로그 업데이트(가능한 경우)"""
        date_str = _to_utc(close_ts).strftime("%Y%m%d")
        symbol = symbol or getattr(config, "SYMBOL", "BTCUSDT")
        payout = payout if payout is not None else getattr(config, "PAYOUT_30M_PLUS", 0.85)
        tenor_min = tenor_min if tenor_min is not None else getattr(config, "OPTION_TENOR_MIN", 30)

        mode = "LIVE" if (kind == "REAL") else "PAPER"

        base = {
            "trade_id": trade_id,
            "symbol": symbol,
            "side": "UP" if direction == "UP" else "DOWN",
            "bar30_start": _iso(bar30_start),
            "bar30_end": _iso(bar30_end),
            "entry_ts": _iso(entry_ts),
            "label_ts": _iso(close_ts),
            "entry_price": float(entry_price) if entry_price is not None else np.nan,
            "label_price": float(exit_price) if exit_price is not None else np.nan,
            "p_raw_at_entry": float(p_at_entry) if p_at_entry is not None else np.nan,
            "p_cal_at_entry": float(kwargs.pop("p_cal_at_entry", p_at_entry if p_at_entry is not None else np.nan)),
            "regime": int(regime) if regime is not None else 0,
            "status": "CLOSED",
            "result": result,
            "payout": float(payout),
            "tenor_min": int(tenor_min),
            "mode": mode,
            "pnl": float(pnl),
            "model_ver": getattr(config, "MODEL_VERSION", ""),
            "feature_ver": getattr(config, "FEATURE_VERSION", ""),
            "filter_ver": getattr(config, "FILTER_VERSION", ""),
            "cutoff_ver": getattr(config, "CUTOFF_VERSION", ""),
            "data_ver": getattr(config, "DATA_VERSION", ""),
        }
        row = {**{c: None for c in self.trade_columns}, **base, **kwargs}
        df_row = pd.DataFrame([row])

        # 청산 파일
        f_close = self.close_dir / f"{date_str}_closes.csv"
        self._append_union(f_close, df_row, expected_columns=self.trade_columns or None)

        # 통합 로그에도 반영(가능하면 해당 trade_id 업데이트)
        # trade_id가 None일 수도 있어 실패해도 조용히 진행
        if trade_id:
            self.update_trade_result(trade_id, result=result, label_price=exit_price, label_ts=close_ts, pnl=pnl, payout=payout)

    # =================================================
    # 결과 업데이트 (trade_id 전역 탐색)
    # =================================================
    def update_trade_result(
        self, trade_id: str, result: str, label_price: float, label_ts: Optional[datetime] = None, **kwargs
    ) -> bool:
        """
        trade_id를 최근 N일 통합 로그에서 탐색 → 해당 행 업데이트.
        업데이트된 행은 close 로그에도 적재.
        """
        if label_ts is None:
            label_ts = datetime.utcnow()

        files = self._recent_trade_files(self.search_days)
        updated = False
        updated_row: Optional[pd.DataFrame] = None

        for f in files:
            df = _read_csv_safe(f)
            if df.empty or "trade_id" not in df.columns:
                continue

            mask = df["trade_id"].astype(str) == str(trade_id)
            if mask.any():
                df.loc[mask, "result"] = result
                df.loc[mask, "label_price"] = label_price
                df.loc[mask, "label_ts"] = _iso(label_ts)
                df.loc[mask, "status"] = "CLOSED"
                # 추가 필드 업데이트
                for k, v in kwargs.items():
                    if k in df.columns:
                        df.loc[mask, k] = v

                _atomic_write_csv(df, f)

                updated_row = df.loc[mask].copy()
                tsu = _to_utc(label_ts)
                close_file = self.close_dir / f"{tsu.strftime('%Y%m%d')}_closes.csv"
                self._append_union(close_file, updated_row, expected_columns=self.trade_columns or None)
                updated = True
                break

        if not updated:
            print(f"⚠️  trade_id를 찾지 못함: {trade_id}")
        return updated

    # =================================================
    # 피처 로그
    # =================================================
    def log_feature(
        self,
        bar30_start: datetime,
        bar30_end: datetime,
        open_price: float,
        high: float,
        low: float,
        close: float,
        volume: float,
        features: Dict,
        target: int,
    ) -> None:
        """
        바 앵커 기준의 피처 스냅샷 기록
        """
        date_str = _to_utc(bar30_start).strftime("%Y%m%d")
        base = {
            "bar30_start": _iso(bar30_start),
            "bar30_end": _iso(bar30_end),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "target": target,
            "feature_ver": getattr(config, "FEATURE_VERSION", ""),
            "data_ver": getattr(config, "DATA_VERSION", ""),
        }

        row = {**{c: None for c in self.feature_columns}, **base, **features}
        df_row = pd.DataFrame([row])

        f_feat = self.feature_log_dir / f"{date_str}_features.csv"
        self._append_union(f_feat, df_row, expected_columns=self.feature_columns or None)

    # =================================================
    # 로더/요약
    # =================================================
    def load_trade_log(self, date_str: str) -> pd.DataFrame:
        f = self.trade_log_dir / f"{date_str}_trades.csv"
        df = _read_csv_safe(f)
        return self._coerce_dtypes(df)

    def load_recent_trades(self, n: int = 50) -> pd.DataFrame:
        files = sorted(self.trade_log_dir.glob("*_trades.csv"), reverse=True)
        dfs: List[pd.DataFrame] = []
        total = 0
        for f in files:
            df = _read_csv_safe(f)
            if df.empty:
                continue
            dfs.append(df)
            total += len(df)
            if total >= n:
                break
        if not dfs:
            return pd.DataFrame(columns=self.trade_columns)
        out = pd.concat(dfs, ignore_index=True)
        out = self._coerce_dtypes(out)
        return out.tail(n).reset_index(drop=True)

    def get_trade_summary(self, date_str: str) -> Dict:
        df = self.load_trade_log(date_str)
        if df.empty:
            return {
                'date': date_str, 'total_trades': 0, 'open_trades': 0,
                'closed_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0
            }

        total = len(df)
        open_cnt = (df.get("status") == "OPEN").sum() if "status" in df else 0
        closed = df[df.get("status") == "CLOSED"] if "status" in df else pd.DataFrame()
        if len(closed) > 0:
            wins = (closed.get("result") == "WIN").sum()
            win_rate = wins / len(closed)
        else:
            win_rate = 0.0

        pnl = 0.0
        if "payout" in df.columns and len(closed) > 0:
            wins = closed[closed["result"] == "WIN"]
            losses = closed[closed["result"] == "LOSS"]
            pnl = wins["payout"].sum() - len(losses)

        return {
            'date': date_str,
            'total_trades': int(total),
            'open_trades': int(open_cnt),
            'closed_trades': int(len(closed)),
            'win_rate': float(win_rate),
            'total_pnl': float(pnl)
        }

    # =================================================
    # 검증
    # =================================================
    def validate_trade_log(self, df: pd.DataFrame) -> Dict:
        res = {'valid': True, 'errors': [], 'warnings': []}
        if df.empty:
            res['warnings'].append("로그가 비어있음")
            return res

        # 필수 컬럼
        needed = self.trade_columns or [
            "trade_id", "symbol", "side", "entry_ts", "entry_price",
            "status", "mode", "bar30_start", "bar30_end"
        ]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            res['errors'].append(f"필수 컬럼 누락: {missing}")
            res['valid'] = False

        if "trade_id" in df.columns:
            dup = df["trade_id"].astype(str).duplicated().sum()
            if dup > 0:
                res['warnings'].append(f"중복 trade_id: {int(dup)}개")

        # 핵심 결측
        for c in ["trade_id", "entry_ts", "entry_price", "side", "status"]:
            if c in df.columns and df[c].isnull().sum() > 0:
                res['errors'].append(f"{c} 결측치 {int(df[c].isnull().sum())}개")
                res['valid'] = False

        # 가격 유효성
        if "entry_price" in df.columns:
            badp = (pd.to_numeric(df["entry_price"], errors="coerce") <= 0).sum()
            if badp > 0:
                res['errors'].append(f"유효하지 않은 가격 {int(badp)}개")
                res['valid'] = False

        # side/result 값
        if "side" in df.columns:
            invalid_side = (~df["side"].isin(["UP", "DOWN", None])).sum()
            if invalid_side > 0:
                res['errors'].append(f"유효하지 않은 side {int(invalid_side)}개")
                res['valid'] = False

        if "result" in df.columns and "status" in df.columns:
            closed = df[df["status"] == "CLOSED"]
            if not closed.empty:
                invalid_result = (~closed["result"].isin(["WIN", "LOSS", "CANCELLED"])).sum()
                if invalid_result > 0:
                    res['warnings'].append(f"유효하지 않은 result {int(invalid_result)}개")

        # 시간 무결성
        if "entry_ts" in df.columns:
            et = pd.to_datetime(df["entry_ts"], utc=True, errors="coerce")
            if et.isna().any():
                res['errors'].append("entry_ts 파싱 실패 레코드 존재")
                res['valid'] = False
            else:
                if not et.sort_values().equals(et):
                    res['warnings'].append("entry_ts가 정렬되지 않음")
                dups = et.duplicated().sum()
                if dups > 0:
                    res['warnings'].append(f"entry_ts 중복 {int(dups)}건")

        return res

    # =================================================
    # 내부 유틸
    # =================================================
    def _coerce_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        # 시간 파싱
        for c in ["bar30_start", "bar30_end", "entry_ts", "label_ts", "cross_time"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
        # 숫자 변환(가능한 범위)
        num_cols = [
            "entry_price", "label_price", "payout", "tenor_min", "pnl",
            "regime_score", "adx", "di_plus", "di_minus",
            "p_raw_at_entry", "p_cal_at_entry", "cut_on", "cut_off",
            "ttl_used_sec", "refractory_window"
        ]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def _recent_trade_files(self, days: int) -> List[Path]:
        files = sorted(self.trade_log_dir.glob("*_trades.csv"), reverse=True)
        if days <= 0:
            return files

        now_utc = pd.Timestamp.now(tz="UTC")
        cutoff = now_utc - pd.Timedelta(days=days)

        out: List[Path] = []
        for f in files:
            try:
                date_str = f.name.split("_")[0]
                dt = pd.to_datetime(date_str, format="%Y%m%d").tz_localize("UTC")
                if dt >= cutoff.normalize():
                    out.append(f)
            except Exception:
                out.append(f)
        return out


# ----------------------------
# Quick self-test
# ----------------------------
if __name__ == "__main__":
    print("=" * 60, "\nLogManager quick test\n", "=" * 60)
    lm = LogManager()

    now = pd.Timestamp("2025-01-15 10:30:00Z")

    # 엔트리(simple)
    lm.log_trade_entry_simple(
        trade_id="T001",
        direction="UP",
        entry_price=50000.0,
        entry_ts=now.to_pydatetime(),
        p_at_entry=0.65,
        regime=1,
        bar30_start=now,
        bar30_end=now + pd.Timedelta(minutes=30),
        adx=30.0,
        regime_score=0.8,
        p_cal_at_entry=0.65,
        cal_method="identity",
        cal_ver="cal_a",
        kind="PAPER",
        symbol="BTCUSDT",
        payout=0.85,
        tenor_min=30,
    )

    # 엔트리(detail)
    lm.log_trade_entry(
        trade_id="T002",
        bar30_start=now,
        bar30_end=now + pd.Timedelta(minutes=30),
        entry_ts=now,
        m1_index_entry=1000,
        entry_price=50100.0,
        side="DOWN",
        regime=-1,
        regime_score=-0.6,
        adx=35.0,
        di_plus=20.0,
        di_minus=30.0,
        p_raw_at_entry=0.70,
        p_cal_at_entry=0.70,
        cal_method="identity",
        cal_ver="cal_a",
        cut_on=0.60,
        cut_off=0.55,
        filters_applied='{"adx_filter": true}',
        symbol="BTCUSDT",
        kind="REAL",
        payout=0.85,
        tenor_min=30,
    )

    # 클로즈(simple)
    lm.log_trade_close_simple(
        trade_id="T001",
        direction="UP",
        entry_price=50000.0,
        exit_price=50200.0,
        entry_ts=now.to_pydatetime(),
        close_ts=(now + pd.Timedelta(minutes=30)).to_pydatetime(),
        p_at_entry=0.65,
        regime=1,
        bar30_start=now.to_pydatetime(),
        bar30_end=(now + pd.Timedelta(minutes=30)).to_pydatetime(),
        kind="PAPER",
        symbol="BTCUSDT",
        payout=0.85,
        tenor_min=30,
        result="WIN",
        pnl=0.85,
    )

    # 전역 업데이트 (이미 클로즈했어도 안전)
    lm.update_trade_result("T001", "WIN", 50200.0, label_ts=now + pd.Timedelta(minutes=30), payout=0.85)

    # 검증/요약
    date_str = now.strftime("%Y%m%d")
    df = lm.load_trade_log(date_str)
    print("loaded rows:", len(df))
    print(lm.validate_trade_log(df))
    print(lm.get_trade_summary(date_str))

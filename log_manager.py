"""
로그 관리자 - 거래/피처 로그 표준화 및 무결성 검증 (UTC ISO, 원자적 쓰기, trade_id 전역 업데이트)
버전: 1.3.1
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


def _to_utc(ts: datetime) -> pd.Timestamp:
    return pd.Timestamp(ts, tz="UTC")


def _iso(dt: datetime | pd.Timestamp | str | None) -> str:
    if dt is None or (isinstance(dt, float) and np.isnan(dt)):
        return ""
    t = pd.to_datetime(dt, utc=True, errors="coerce")
    return "" if pd.isna(t) else t.isoformat().replace("+00:00", "Z")


def _atomic_write_csv(df: pd.DataFrame, fpath: Path):
    fpath.parent.mkdir(parents=True, exist_ok=True)
    tmp = fpath.parent / (".tmp_" + fpath.name)
    df.to_csv(tmp, index=False)
    shutil.move(str(tmp), str(fpath))


def _read_csv_safe(fpath: Path) -> pd.DataFrame:
    if not fpath.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(fpath)
    except Exception as e:
        print(f"⚠️  CSV 로드 실패: {fpath.name} ({e})")
        return pd.DataFrame()


class LogManager:
    """
    - 거래 로그(통합/엔트리/청산), 피처 로그
    - 스키마 검증, UTC ISO 기록, 원자적 쓰기
    - trade_id 기반 결과 업데이트(최근 N일 검색)
    """

    def __init__(self, search_days: int = 14):
        self.trade_log_dir = config.TRADE_LOG_DIR
        self.entry_dir = config.TRADE_ENTRY_DIR
        self.close_dir = config.TRADE_CLOSE_DIR
        self.meta_dir = config.TRADE_META_DIR
        self.feature_log_dir = config.FEATURE_LOG_DIR

        self.trade_columns = config.TRADE_LOG_COLUMNS
        self.feature_columns = config.FEATURE_LOG_COLUMNS
        self.trade_dtypes = config.TRADE_LOG_DTYPES
        self.search_days = int(search_days)

    # -------------------------------------------------
    # 엔트리(간소/상세)
    # -------------------------------------------------
    def log_trade_entry_simple(
        self, trade_id: str, direction: str, entry_price: float, entry_ts: datetime,
        p_raw_at_entry: float, regime: int, **kwargs
    ) -> None:
        date_str = _to_utc(entry_ts).strftime("%Y%m%d")

        base = {
            "trade_id": trade_id,
            "entry_ts": _iso(entry_ts),
            "entry_price": entry_price,
            "side": direction,
            "p_raw_at_entry": p_raw_at_entry,
            "p_cal_at_entry": kwargs.pop("p_cal_at_entry", p_raw_at_entry),
            "cal_method": kwargs.pop("cal_method", "identity"),
            "cal_ver": kwargs.pop("cal_ver", ""),
            "regime": regime,
            "status": "OPEN",
            "mode": kwargs.pop("mode", "LIVE"),
            "model_ver": config.MODEL_VERSION,
            "feature_ver": config.FEATURE_VERSION,
            "filter_ver": config.FILTER_VERSION,
            "cutoff_ver": config.CUTOFF_VERSION,
            "data_ver": config.DATA_VERSION,
        }
        # 선택적 시간 필드 ISO화
        for k in ["bar30_start","bar30_end","cross_time","label_ts"]:
            if k in kwargs: kwargs[k] = _iso(kwargs[k])

        row = {**{c: None for c in self.trade_columns}, **base, **kwargs}
        df_row = pd.DataFrame([row])

        self._append_union(self.trade_log_dir / f"{date_str}_trades.csv", df_row)
        self._append_union(self.entry_dir / f"{date_str}_entries.csv", df_row)

    def log_trade_entry(
        self, trade_id: str, bar30_start: datetime, bar30_end: datetime, entry_ts: datetime,
        m1_index_entry: Optional[int], entry_price: float, side: str,
        regime: int, regime_score: float, adx: float, di_plus: float, di_minus: float,
        p_raw_at_entry: float, p_cal_at_entry: float, cal_method: str, cal_ver: str,
        cut_on: float, cut_off: float, filters_applied: str, **kwargs
    ) -> None:
        date_str = _to_utc(entry_ts).strftime("%Y%m%d")

        row = {**{c: None for c in self.trade_columns}, **{
            "trade_id": trade_id,
            "bar30_start": _iso(bar30_start),
            "bar30_end": _iso(bar30_end),
            "entry_ts": _iso(entry_ts),
            "m1_index_entry": m1_index_entry,
            "entry_price": entry_price,
            "side": side,
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
            "model_ver": config.MODEL_VERSION,
            "feature_ver": config.FEATURE_VERSION,
            "filter_ver": config.FILTER_VERSION,
            "cutoff_ver": config.CUTOFF_VERSION,
            "data_ver": config.DATA_VERSION,
        }}

        if "cross_time" in kwargs: row["cross_time"] = _iso(kwargs["cross_time"])
        df_row = pd.DataFrame([row])

        self._append_union(self.trade_log_dir / f"{date_str}_trades.csv", df_row)
        self._append_union(self.entry_dir / f"{date_str}_entries.csv", df_row)

    # -------------------------------------------------
    # 결과 업데이트 (trade_id 전역 탐색)
    # -------------------------------------------------
    def update_trade_result(
        self, trade_id: str, result: str, label_price: float, label_ts: Optional[datetime] = None, **kwargs
    ) -> bool:
        """
        trade_id를 최근 N일 통합 로그에서 탐색 → 해당 행 업데이트.
        """
        if label_ts is None:
            label_ts = datetime.utcnow()

        # 검색 파일 목록(최근 N일)
        files = self._recent_trade_files(self.search_days)

        updated = False
        updated_row: Optional[pd.DataFrame] = None

        for f in files:
            df = _read_csv_safe(f)
            if df.empty or "trade_id" not in df.columns:
                continue

            mask = df["trade_id"].astype(str) == str(trade_id)
            if mask.any():
                # 업데이트
                df.loc[mask, "result"] = result
                df.loc[mask, "label_price"] = label_price
                df.loc[mask, "label_ts"] = _iso(label_ts)
                df.loc[mask, "status"] = "CLOSED"
                for k, v in kwargs.items():
                    if k in df.columns:
                        df.loc[mask, k] = v
                _atomic_write_csv(df, f)

                # 청산 로그에도 적재
                updated_row = df.loc[mask].copy()
                close_file = self.close_dir / f"{pd.Timestamp(label_ts, tz='UTC').strftime('%Y%m%d')}_closes.csv"
                self._append_union(close_file, updated_row)
                updated = True
                break

        if not updated:
            print(f"⚠️  trade_id를 찾지 못함: {trade_id}")
        return updated

    # -------------------------------------------------
    # 피처 로그
    # -------------------------------------------------
    def log_feature(
        self, bar30_start: datetime, bar30_end: datetime,
        open_price: float, high: float, low: float, close: float, volume: float,
        features: Dict, target: int
    ) -> None:
        date_str = _to_utc(bar30_start).strftime("%Y%m%d")
        base = {
            "bar30_start": _iso(bar30_start),
            "bar30_end": _iso(bar30_end),
            "open": open_price, "high": high, "low": low, "close": close, "volume": volume,
            "target": target,
            "feature_ver": config.FEATURE_VERSION, "data_ver": config.DATA_VERSION
        }
        row = {**{c: None for c in self.feature_columns}, **base, **features}
        df_row = pd.DataFrame([row])
        self._append_union(self.feature_log_dir / f"{date_str}_features.csv", df_row)

    # -------------------------------------------------
    # 로더/요약
    # -------------------------------------------------
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
            if df.empty: continue
            dfs.append(df)
            total += len(df)
            if total >= n: break
        if not dfs:
            return pd.DataFrame(columns=self.trade_columns)
        out = pd.concat(dfs, ignore_index=True)
        out = self._coerce_dtypes(out)
        return out.tail(n).reset_index(drop=True)

    def get_trade_summary(self, date_str: str) -> Dict:
        df = self.load_trade_log(date_str)
        if df.empty:
            return {'date': date_str, 'total_trades': 0, 'open_trades': 0, 'closed_trades': 0, 'win_rate': 0.0, 'total_pnl': 0.0}

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

        return {'date': date_str, 'total_trades': total, 'open_trades': int(open_cnt), 'closed_trades': int(len(closed)), 'win_rate': float(win_rate), 'total_pnl': float(pnl)}

    # -------------------------------------------------
    # 검증
    # -------------------------------------------------
    def validate_trade_log(self, df: pd.DataFrame) -> Dict:
        res = {'valid': True, 'errors': [], 'warnings': []}
        if df.empty:
            res['warnings'].append("로그가 비어있음")
            return res

        missing = [c for c in self.trade_columns if c not in df.columns]
        if missing:
            res['errors'].append(f"필수 컬럼 누락: {missing}")
            res['valid'] = False

        if "trade_id" in df.columns:
            dup = df["trade_id"].duplicated().sum()
            if dup > 0:
                res['warnings'].append(f"중복 trade_id: {int(dup)}개")

        # 핵심 결측
        for c in ["trade_id","entry_ts","entry_price","side","status"]:
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
            invalid_side = (~df["side"].isin(["UP","DOWN",None])).sum()
            if invalid_side > 0:
                res['errors'].append(f"유효하지 않은 side {int(invalid_side)}개")
                res['valid'] = False

        if "result" in df.columns and "status" in df.columns:
            closed = df[df["status"] == "CLOSED"]
            invalid_result = (~closed["result"].isin(["WIN","LOSS","CANCELLED"])).sum()
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

    # -------------------------------------------------
    # 내부 유틸
    # -------------------------------------------------
    def _append_union(self, fpath: Path, df_new: pd.DataFrame):
        """
        기존 파일과 컬럼 유니온 후 append → 원자적 overwrite.
        """
        df_old = _read_csv_safe(fpath)
        if df_old.empty:
            out = df_new[self.trade_columns] if set(self.trade_columns).issuperset(df_new.columns) else df_new
        else:
            all_cols = list(dict.fromkeys(list(df_old.columns) + list(df_new.columns)))
            out = pd.concat([df_old.reindex(columns=all_cols), df_new.reindex(columns=all_cols)], ignore_index=True)
        _atomic_write_csv(out, fpath)

    def _coerce_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        # 시간 파싱
        for c in ["bar30_start","bar30_end","entry_ts","label_ts","cross_time"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
        # 숫자 변환(가능한 범위)
        for c in ["entry_price","label_price","payout","regime_score","adx","di_plus","di_minus",
                  "p_raw_at_entry","p_cal_at_entry","cut_on","cut_off","ttl_used_sec","refractory_window"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def _recent_trade_files(self, days: int) -> List[Path]:
        files = sorted(self.trade_log_dir.glob("*_trades.csv"), reverse=True)
        if days <= 0:
            return files
        cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(days=days)
        out = []
        for f in files:
            # 파일명에서 날짜 추출
            try:
                date_str = f.name.split("_")[0]
                dt = pd.to_datetime(date_str, format="%Y%m%d", utc=True)
                if dt >= cutoff.normalize():
                    out.append(f)
            except Exception:
                out.append(f)  # 포맷 불명은 일단 포함
        return out


if __name__ == "__main__":
    print("="*60, "\nLogManager quick test\n", "="*60)
    lm = LogManager()

    now = pd.Timestamp("2025-01-15 10:30:00Z")
    # 엔트리
    lm.log_trade_entry_simple(
        trade_id="T001", direction="UP", entry_price=50000.0, entry_ts=now.to_pydatetime(),
        p_raw_at_entry=0.65, regime=1,
        bar30_start=now, bar30_end=now + pd.Timedelta(minutes=30),
        adx=30.0, regime_score=0.8, p_cal_at_entry=0.65, cal_method="identity", cal_ver="cal_a"
    )
    # 상세
    lm.log_trade_entry(
        trade_id="T002", bar30_start=now, bar30_end=now+pd.Timedelta(minutes=30),
        entry_ts=now, m1_index_entry=1000, entry_price=50100.0, side="DOWN",
        regime=-1, regime_score=-0.6, adx=35.0, di_plus=20.0, di_minus=30.0,
        p_raw_at_entry=0.70, p_cal_at_entry=0.70, cal_method="identity", cal_ver="cal_a",
        cut_on=0.60, cut_off=0.55, filters_applied='{"adx_filter": true}'
    )
    # 업데이트
    lm.update_trade_result("T001", "WIN", 50200.0, label_ts=now+pd.Timedelta(minutes=30), payout=0.85)

    # 검증
    date_str = now.strftime("%Y%m%d")
    df = lm.load_trade_log(date_str)
    print("loaded:", len(df))
    print(lm.validate_trade_log(df))

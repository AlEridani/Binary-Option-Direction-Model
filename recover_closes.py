"""
recover_closes.py
엔트리/피처 로그를 사용해 누락된 CLOSE를 복구하는 유틸리티

사용 예:
  python recover_closes.py --day 2025-10-31 --source exchange
  python recover_closes.py --from 2025-10-29 --to 2025-10-31 --source local --local-csv data/btcusdt_1m.csv

동작:
- TRADE_LOG_DIR의 *_trades.csv를 스캔하여 OPEN 또는 label 미기록 건을 수집
- bar30_end(없으면 entry_ts + tenor) 기준으로 만기시각 결정
- 실제 시세(거래소 Binance / 로컬 OHLCV)에서 만기 종가(label_price) 취득
- result/pnl/payout/label_ts를 산출하여 LogManager.update_trade_result()로 기록
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

import config
from log_manager import LogManager

# ===== 가격소스: 거래소(Binance) =====
def fetch_close_from_exchange(symbol: str, bar30_end_utc: pd.Timestamp) -> float | None:
    """
    bar30_end_utc가 끝나는 30m 캔들의 종가를 거래소에서 조회.
    - Binance klines는 [open_time, close_time) 반개구간 느낌이라,
      정확히 bar30_end를 'close_time'으로 갖는 캔들을 가져오려면
      bar30_start = bar30_end - 30m 로 쿼리 범위를 맞춰야 함.
    """
    try:
        from binance.client import Client
    except Exception as e:
        print(f"Binance 라이브러리 로드 실패: {e}")
        return None

    try:
        client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
    except Exception as e:
        print(f"Binance 클라이언트 생성 실패: {e}")
        return None

    try:
        end = pd.to_datetime(bar30_end_utc, utc=True)
        start = end - pd.Timedelta(minutes=getattr(config, "OPTION_TENOR_MIN", 30))

        # Binance API는 밀리초 타임스탬프 사용
        klines = client.get_klines(
            symbol=getattr(config, "SYMBOL", "BTCUSDT"),
            interval=getattr(config, "INTERVAL_30m", "30m"),
            startTime=int(start.timestamp() * 1000),
            endTime=int(end.timestamp() * 1000),
            limit=2,
        )
        if not klines:
            return None

        # 가장 마지막 바가 우리가 원하는 바(끝이 bar30_end)여야 함
        # 포맷: [open_time, open, high, low, close, ..., close_time, ...]
        rows = pd.DataFrame(klines, columns=[
            'open_time','open','high','low','close','volume',
            'close_time','quote_volume','trades','taker_buy_base','taker_buy_quote','ignore'
        ])
        rows['open_time']  = pd.to_datetime(rows['open_time'], unit='ms', utc=True)
        rows['close_time'] = pd.to_datetime(rows['close_time'], unit='ms', utc=True)
        rows['close'] = rows['close'].astype(float)

        # close_time이 bar30_end와 같은 행 찾기 (정확 일치)
        hit = rows[rows['close_time'] == end]
        if hit.empty:
            # 일부 환경에서 close_time이 약간 앞서거나 뒤로 밀릴 수 있음 → 근접치 허용
            tol = pd.Timedelta(seconds=5)
            hit = rows[(rows['close_time'] >= end - tol) & (rows['close_time'] <= end + tol)]
        if hit.empty:
            return None
        return float(hit['close'].iloc[-1])
    except Exception as e:
        print(f"거래소 가격 조회 실패: {e}")
        return None

# ===== 가격소스: 로컬 1분/30분 OHLCV =====
def fetch_close_from_local(bar30_end_utc: pd.Timestamp,
                           local_df: pd.DataFrame) -> float | None:
    """
    로컬 OHLCV에서 bar30_end 시점의 30m 종가를 구함.
    - local_df가 30m면 bar30_end 시점과 동일한 timestamp의 close 사용
    - local_df가 1m면 bar30_end 시각의 바로 '직전 1분' 종가 사용(=해당 30m 바의 마지막 1분 close)
    """
    if local_df is None or local_df.empty:
        return None

    df = local_df.copy()
    # timestamp 컬럼 정규화
    if 'timestamp' in df.columns:
        ts = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        df = df.assign(timestamp=ts).dropna(subset=['timestamp'])
        df.set_index('timestamp', inplace=True)
    elif isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
    else:
        return None

    need_cols = {'close'}
    if not need_cols.issubset(df.columns):
        return None

    # 30분 데이터인지 1분 데이터인지 자동 감지
    if len(df.index) >= 3:
        deltas = (df.index[1:] - df.index[:-1]).astype('timedelta64[m]')
        median_step = np.median(deltas)
    else:
        median_step = 1.0

    end = pd.to_datetime(bar30_end_utc, utc=True)
    if abs(median_step - 30.0) <= 1.0:
        # 30m 데이터
        # 보통 'bar30_start'가 timestamp로 저장되어 있고, 종가는 다음 bar 시작 직전에 확정됨.
        # 여기서는 end 시각과 동일한 인덱스를 기대(종종 end==다음 bar start).
        if end in df.index:
            return float(df.loc[end, 'close'])
        # 근접치 허용
        near = df.index.get_indexer([end], method='nearest', tolerance='2min')
        if near.size and near[0] != -1:
            return float(df.iloc[near[0]]['close'])
        return None
    else:
        # 1m 데이터: bar30_end의 바로 직전 1분 close 사용
        prev = end - pd.Timedelta(minutes=1)
        if prev in df.index:
            return float(df.loc[prev, 'close'])
        near = df.index.get_indexer([prev], method='nearest', tolerance='90s')
        if near.size and near[0] != -1:
            return float(df.iloc[near[0]]['close'])
        return None


def compute_result(entry_price: float, exit_price: float, side: str) -> tuple[str, float]:
    """
    side=UP: exit > entry → WIN
    side=DOWN: exit < entry → WIN
    그렇지 않으면 LOSS. pnl은 payout or -1.0
    """
    payout = float(getattr(config, "PAYOUT_30M_PLUS", 0.85))
    if pd.isna(entry_price) or pd.isna(exit_price):
        return "LOSS", -1.0  # 안전 디폴트(데이터 부족시 LOSS 처리)
    if str(side).upper() == "UP":
        ok = exit_price > entry_price
    else:
        ok = exit_price < entry_price
    return ("WIN", payout) if ok else ("LOSS", -1.0)


def load_local_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # timestamp/o/h/l/c/v 유추
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ["time","timestamp","date","datetime"]:
            rename_map[c] = "timestamp"
        elif lc == "open":
            rename_map[c] = "open"
        elif lc == "high":
            rename_map[c] = "high"
        elif lc == "low":
            rename_map[c] = "low"
        elif lc == "close":
            rename_map[c] = "close"
        elif lc in ("volume","vol"):
            rename_map[c] = "volume"
    df = df.rename(columns=rename_map)
    return df


def iter_open_or_unlabeled_trades(day_from: str|None, day_to: str|None):
    """
    TRADE_LOG_DIR의 *_trades.csv에서
    - status != CLOSED 이거나
    - label_price/label_ts/result 미기록(결측)인 행
    만 yield.
    day_from/day_to를 지정하면 해당 기간(entry_ts 기준)만 필터링.
    """
    log_dir = Path(config.TRADE_LOG_DIR)
    files = sorted(log_dir.glob("*_trades.csv"))
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if df.empty:
            continue

        # 시간 파싱
        for c in ("entry_ts","bar30_start","bar30_end","label_ts"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")

        # 기간 필터
        if day_from:
            df = df[df["entry_ts"] >= pd.Timestamp(day_from, tz="UTC")]
        if day_to:
            df = df[df["entry_ts"] < (pd.Timestamp(day_to, tz="UTC") + pd.Timedelta(days=1))]

        # 미닫힘 선택
        need_cols = set(df.columns)
        unlabeled_mask = (
            (("status" in need_cols) & (df["status"] != "CLOSED")) |
            (("label_price" in need_cols) & (df["label_price"].isna())) |
            (("result" in need_cols) & (df["result"].isna()))
        )
        cand = df[unlabeled_mask].copy()
        if cand.empty:
            continue

        # 중복 trade_id는 최신만 남기거나 첫건만 사용 — 여기서는 첫건 유지
        yield f.name, cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", help="복구 대상 UTC 날짜(단일). 예: 2025-10-31")
    ap.add_argument("--from", dest="date_from", help="복구 시작 UTC 날짜(포함)")
    ap.add_argument("--to", dest="date_to", help="복구 종료 UTC 날짜(포함)")
    ap.add_argument("--source", choices=["exchange","local"], default="exchange")
    ap.add_argument("--local-csv", dest="local_csv", help="--source local일 때 사용할 OHLCV CSV 경로(1분 또는 30분)")
    args = ap.parse_args()

    # 기간 해석
    day_from = day_to = None
    if args.day:
        day_from = args.day
        day_to = args.day
    else:
        day_from = args.date_from
        day_to   = args.date_to

    local_df = None
    if args.source == "local":
        if not args.local_csv:
            print("❌ --source local 사용 시 --local-csv 경로가 필요합니다.")
            return
        local_df = load_local_csv(args.local_csv)

    lm = LogManager()

    total = 0
    fixed = 0

    for fname, df_open in iter_open_or_unlabeled_trades(day_from, day_to):
        if df_open.empty:
            continue

        # 필요한 컬럼 정리
        need = ["trade_id","entry_ts","bar30_start","bar30_end","entry_price","side"]
        for c in need:
            if c not in df_open.columns:
                df_open[c] = np.nan

        for _, row in df_open.iterrows():
            total += 1
            trade_id = str(row.get("trade_id", ""))
            entry_ts = pd.to_datetime(row.get("entry_ts"), utc=True, errors="coerce")
            side = str(row.get("side","")).upper() or "UP"
            entry_price = float(row.get("entry_price", np.nan))
            tenor = int(getattr(config, "OPTION_TENOR_MIN", 30))

            bar30_start = pd.to_datetime(row.get("bar30_start"), utc=True, errors="coerce")
            bar30_end   = pd.to_datetime(row.get("bar30_end"),   utc=True, errors="coerce")
            if pd.isna(bar30_end):
                # 엔트리 구조가 "다음 바 시작에 진입"이라면 close는 (entry_ts + tenor)에서 종가로 판정
                bar30_end = entry_ts + pd.Timedelta(minutes=tenor)

            # 실제 만기 종가(label_price) 취득
            if args.source == "exchange":
                exit_price = fetch_close_from_exchange(getattr(config, "SYMBOL", "BTCUSDT"), bar30_end)
            else:
                exit_price = fetch_close_from_local(bar30_end, local_df)

            if exit_price is None:
                print(f"[SKIP] {trade_id} — 만기 가격 조회 실패 ({bar30_end})")
                continue

            result, pnl = compute_result(entry_price, exit_price, side)

            ok = False
            try:
                ok = lm.update_trade_result(
                    trade_id=trade_id,
                    result=result,
                    label_price=float(exit_price),
                    label_ts=bar30_end.to_pydatetime(),
                    payout=float(getattr(config, "PAYOUT_30M_PLUS", 0.85)),
                    pnl=float(pnl)
                )
            except Exception as e:
                print(f"[ERR] update_trade_result 실패: {trade_id} ({e})")

            if ok:
                fixed += 1
                print(f"[FIXED] {trade_id} → {result}  label@{bar30_end:%Y-%m-%d %H:%M}  exit={exit_price:.2f}")
            else:
                print(f"[MISS] trade_id 미발견: {trade_id}")

    print(f"\n완료: 대상 {total}건, 갱신 {fixed}건")


if __name__ == "__main__":
    main()

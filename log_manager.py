# log_manager.py
# 거래/피처 로그 관리 (30분봉 시스템)
# - 날짜별 CSV 저장 (logs/trades/, logs/trades/entries/, logs/trades/closes/, logs/features/)
# - 진입/청산 분리 기록 + 통합 파일 업데이트
# - 원자적 쓰기, 스키마/타입 보정, 자동 무결성 검증
# - real_trade 호환 어댑터 추가

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from config import Config


class LogManager:
    """로그 관리 (거래/피처) — 진입/청산 분리 + 통합 파일 업데이트"""

    def __init__(self):
        self.config = Config
        self.trade_log_dir = self.config.TRADE_LOG_DIR
        self.feature_log_dir = self.config.FEATURE_LOG_DIR

        # 서브 디렉토리(진입/청산/메타)
        self.trade_entries_dir = self.trade_log_dir / "entries"
        self.trade_closes_dir = self.trade_log_dir / "closes"
        self.trade_meta_dir = self.trade_log_dir / "meta"

        self._ensure_dirs()
        self.current_versions = self._load_current_versions()

    # ---------------------------
    # 내부 유틸
    # ---------------------------
    def _ensure_dirs(self):
        for d in [
            self.trade_log_dir,
            self.feature_log_dir,
            self.trade_entries_dir,
            self.trade_closes_dir,
            self.trade_meta_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def _load_current_versions(self) -> Dict[str, str]:
        version_file = self.config.MODEL_DIR / 'current_versions.json'
        if version_file.exists():
            with open(version_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        now_ver = self.config.get_version_string()
        return {
            'model_ver': now_ver,
            'feature_ver': now_ver,
            'filter_ver': now_ver,
            'cutoff_ver': now_ver,
            'data_ver': now_ver
        }

    def update_versions(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.current_versions:
                self.current_versions[key] = value
        version_file = self.config.MODEL_DIR / 'current_versions.json'
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump(self.current_versions, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _to_iso(dt: Optional[datetime]) -> Optional[str]:
        if dt is None:
            return None
        if isinstance(dt, str):
            return dt
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _to_utc_datetime(x) -> Optional[datetime]:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return None
        try:
            dt = pd.to_datetime(x, utc=True, errors='coerce')
            if pd.isna(dt):
                return None
            return dt.to_pydatetime()
        except Exception:
            return None

    @staticmethod
    def _minute_index(dt: datetime) -> int:
        """1분 인덱스(정수): epoch_seconds // 60"""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() // 60)

    def _atomic_write(self, df: pd.DataFrame, path: Path, mode: str, header: bool):
        tmp = Path(str(path) + ".tmp")
        if mode == 'a' and path.exists():
            old = pd.read_csv(path)
            merged = pd.concat([old, df], ignore_index=True)
            merged.to_csv(tmp, index=False)
        else:
            df.to_csv(tmp, index=False)
        os.replace(tmp, path)

    def _atomic_overwrite(self, df: pd.DataFrame, path: Path):
        tmp = Path(str(path) + ".tmp")
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)

    def _cast_trade_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """숫자/불리언/타임스탬프 기본 캐스팅"""
        if df is None or df.empty:
            return df

        # timestamps
        for col in ['bar30_start', 'bar30_end', 'entry_ts', 'label_ts', 'cross_time']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')

        num_cols = [
            'm1_index_entry', 'm1_index_label',
            'entry_price', 'label_price',
            'payout', 'result',
            'regime', 'is_weekend', 'regime_score',
            'adx', 'di_plus', 'di_minus',
            'p_at_entry', 'dp_at_entry',
            'cut_on', 'cut_off',
            'ttl_used_sec', 'refractory_window'
        ]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        if 'ttl_valid' in df.columns:
            df['ttl_valid'] = df['ttl_valid'].astype('boolean')

        return df

    def _order_trade_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Config.TRADE_LOG_COLUMNS 순으로 정렬"""
        if df is None or df.empty:
            return df
        base = list(getattr(self.config, 'TRADE_LOG_COLUMNS', []))
        extras = [c for c in ['side', 'status'] if c in df.columns and c not in base]
        ordered = [c for c in base if c in df.columns] + extras + [c for c in df.columns if c not in base + extras]
        return df[ordered]

    def _is_weekend(self, dt: datetime) -> int:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.weekday() >= 5)

    # ---------------------------
    # 경로
    # ---------------------------
    def _trade_path(self, date_str: str) -> Path:
        return self.config.get_log_path('trade', date_str)

    def _trade_entry_path(self, date_str: str) -> Path:
        return self.trade_entries_dir / f"{date_str}_entries.csv"

    def _trade_close_path(self, date_str: str) -> Path:
        return self.trade_closes_dir / f"{date_str}_closes.csv"

    def _feature_path(self, date_str: str) -> Path:
        return self.config.get_log_path('feature', date_str)

    # ---------------------------
    # Trade: 진입 (ENTRY) - 상세 버전
    # ---------------------------
    def log_trade_entry(
        self,
        trade_id: str,
        bar30_start: datetime,
        bar30_end: datetime,
        entry_ts: datetime,
        m1_index_entry: Optional[int],
        side: str,
        entry_price: float,
        regime: int,
        regime_score: float,
        adx: float,
        di_plus: float,
        di_minus: float,
        p_at_entry: float,
        dp_at_entry: float,
        cut_on: float,
        cut_off: float,
        cross_time: Optional[datetime],
        ttl_used_sec: float,
        ttl_valid: bool,
        refractory_window: int,
        filters_applied: str,
        reason_code: str,
        mode: str = 'LIVE',
        payout: float = None,
        is_weekend: Optional[int] = None
    ):
        """진입(OPEN) 로그 기록 — 청산 정보 없이 기록"""
        date_str = pd.to_datetime(bar30_start, utc=True).strftime("%Y%m%d")
        trade_path = self._trade_path(date_str)
        entry_path = self._trade_entry_path(date_str)

        is_weekend_calc = self._is_weekend(pd.to_datetime(bar30_start, utc=True).to_pydatetime())
        label_ts = pd.to_datetime(bar30_end, utc=True) + pd.Timedelta(minutes=self.config.BAR_MINUTES)

        if m1_index_entry is None:
            m1_index_entry = self._minute_index(pd.to_datetime(entry_ts, utc=True).to_pydatetime())

        row = {
            'trade_id': trade_id,
            'bar30_start': self._to_iso(bar30_start),
            'bar30_end': self._to_iso(bar30_end),
            'entry_ts': self._to_iso(entry_ts),
            'label_ts': self._to_iso(label_ts),
            'm1_index_entry': m1_index_entry,
            'm1_index_label': self._minute_index(label_ts.to_pydatetime()),
            'entry_price': entry_price,
            'label_price': np.nan,
            'payout': payout if payout is not None else self.config.PAYOUT_RATIO,
            'result': np.nan,
            'side': str(side).upper(),
            'regime': regime,
            'is_weekend': is_weekend_calc,
            'regime_score': regime_score,
            'adx': adx,
            'di_plus': di_plus,
            'di_minus': di_minus,
            'p_at_entry': p_at_entry,
            'dp_at_entry': dp_at_entry,
            'cut_on': cut_on,
            'cut_off': cut_off,
            'cross_time': self._to_iso(cross_time) if cross_time else None,
            'ttl_used_sec': ttl_used_sec,
            'ttl_valid': bool(ttl_valid),
            'refractory_window': refractory_window,
            'filters_applied': filters_applied,
            'reason_code': reason_code,
            'blocked_reason': None,
            'model_ver': self.current_versions['model_ver'],
            'feature_ver': self.current_versions['feature_ver'],
            'filter_ver': self.current_versions['filter_ver'],
            'cutoff_ver': self.current_versions['cutoff_ver'],
            'data_ver': self.current_versions['data_ver'],
            'mode': (mode or 'LIVE').upper(),
            'status': 'OPEN'
        }
        df = pd.DataFrame([row])
        df = self._cast_trade_types(df)
        df = self._order_trade_columns(df)

        self._atomic_write(df, trade_path, mode='a', header=not trade_path.exists())
        self._atomic_write(df, entry_path, mode='a', header=not entry_path.exists())

        print(f"✓ 진입 로그 저장: {trade_path.name} (trade_id={trade_id})")

    # ---------------------------
    # Trade: real_trade 호환 어댑터 (간소화)
    # ---------------------------
    def log_trade_entry_simple(
        self,
        trade_id: str,
        direction: int,
        entry_price: float,
        entry_time: datetime,
        expiry_time: datetime,
        p_up: float,
        regime: Optional[int],
        bar30_start: datetime,
        bar30_end: datetime,
        stake_recommended: Optional[float] = None,
        model_version: Optional[str] = None,
        features_dict: Optional[Dict] = None
    ):
        """
        real_trade.py 호환용 간소화 인터페이스
        
        Parameters:
        -----------
        direction : int
            0 (DOWN) or 1 (UP)
        features_dict : Dict, optional
            추가 피처 {'regime_score': float, 'adx_14': float, ...}
        """
        # direction → side 변환
        side = 'LONG' if direction == 1 else 'SHORT'
        
        # features_dict에서 추출
        if features_dict:
            regime_score = float(features_dict.get('regime_score', 0.0))
            adx = float(features_dict.get('adx_14', 0.0))
            di_plus = float(features_dict.get('di_plus_14', 0.0))
            di_minus = float(features_dict.get('di_minus_14', 0.0))
            dp_at_entry = float(features_dict.get('dp', 0.0))
        else:
            regime_score = 0.0
            adx = 0.0
            di_plus = 0.0
            di_minus = 0.0
            dp_at_entry = 0.0
        
        # TTL 계산
        if entry_time > bar30_end:
            ttl_used_sec = (entry_time - bar30_end).total_seconds()
        else:
            ttl_used_sec = 0.0
        
        # 상세 로깅 호출
        self.log_trade_entry(
            trade_id=trade_id,
            bar30_start=bar30_start,
            bar30_end=bar30_end,
            entry_ts=entry_time,
            m1_index_entry=None,
            side=side,
            entry_price=entry_price,
            regime=regime if regime is not None else 0,
            regime_score=regime_score,
            adx=adx,
            di_plus=di_plus,
            di_minus=di_minus,
            p_at_entry=p_up,
            dp_at_entry=dp_at_entry,
            cut_on=self.config.CUT_ON,
            cut_off=self.config.CUT_OFF,
            cross_time=None,
            ttl_used_sec=ttl_used_sec,
            ttl_valid=True,
            refractory_window=self.config.REFRACTORY_WINDOW_MINUTES,
            filters_applied="auto",
            reason_code="SIGNAL_ENTRY",
            mode='LIVE',
            payout=self.config.PAYOUT_RATIO
        )
        
        # 추가 메타 저장
        if stake_recommended is not None or model_version is not None:
            self._save_trade_meta(trade_id, {
                'stake_recommended': stake_recommended,
                'model_version': model_version
            })

    def _save_trade_meta(self, trade_id: str, meta: Dict):
        """거래 메타 정보 별도 저장 (JSONL)"""
        date_str = datetime.now(timezone.utc).strftime('%Y%m%d')
        path = self.trade_meta_dir / f"{date_str}_meta.jsonl"
        
        record = {
            'trade_id': trade_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **meta
        }
        
        with open(path, 'a', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, default=str)
            f.write('\n')

    def load_trade_meta(self, date_str: Optional[str] = None) -> pd.DataFrame:
        """거래 메타 정보 로드"""
        if date_str is None:
            date_str = datetime.now(timezone.utc).strftime('%Y%m%d')
        
        path = self.trade_meta_dir / f"{date_str}_meta.jsonl"
        
        if not path.exists():
            return pd.DataFrame()
        
        records = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    records.append(json.loads(line.strip()))
                except:
                    continue
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
        
        return df

    # ---------------------------
    # Trade: 청산/업데이트 (CLOSE)
    # ---------------------------
    def update_trade_result(
        self,
        trade_id: str,
        result: int,
        label_price: float,
        label_ts: Optional[datetime] = None,
        blocked_reason: Optional[str] = None,
        date_hint: Optional[str] = None,
        search_days: int = 3
    ) -> bool:
        """청산 결과 업데이트"""
        candidate_dates: List[str] = []
        if date_hint:
            candidate_dates = [date_hint]
        else:
            now = datetime.now(timezone.utc)
            for i in range(search_days):
                d = (now - pd.Timedelta(days=i)).strftime("%Y%m%d")
                candidate_dates.append(d)

        target_path = None
        target_date = None
        row_idx = None
        df = None

        for d in candidate_dates:
            p = self._trade_path(d)
            if not p.exists():
                continue
            temp = pd.read_csv(p)
            if 'trade_id' in temp.columns:
                mask = temp['trade_id'] == trade_id
                if mask.any():
                    target_path = p
                    target_date = d
                    row_idx = temp.index[mask][0]
                    df = temp
                    break

        if target_path is None or df is None:
            print(f"⚠️ 업데이트 실패: 최근 {search_days}일 내 trade_id={trade_id}를 찾지 못했습니다.")
            return False

        df = self._cast_trade_types(df)

        df.loc[row_idx, 'result'] = int(result)
        df.loc[row_idx, 'label_price'] = float(label_price)
        if label_ts is not None:
            df.loc[row_idx, 'label_ts'] = self._to_iso(label_ts)
        if blocked_reason is not None:
            df.loc[row_idx, 'blocked_reason'] = blocked_reason
        
        if 'status' not in df.columns:
            df['status'] = np.where(df['result'].notna(), 'CLOSED', 'OPEN')
        df.loc[row_idx, 'status'] = 'CLOSED'

        df = self._order_trade_columns(df)
        self._atomic_overwrite(df, target_path)

        close_date = target_date or pd.to_datetime(df.loc[row_idx, 'bar30_start'], utc=True).strftime("%Y%m%d")
        close_path = self._trade_close_path(close_date)
        snapshot = df.loc[[row_idx]].copy()
        self._atomic_write(snapshot, close_path, mode='a', header=not close_path.exists())

        print(f"✓ 청산 업데이트 완료: {target_path.name} (trade_id={trade_id})")
        return True

    # ---------------------------
    # Trade: 로딩
    # ---------------------------
    def load_trade_log(self, date_str: Optional[str] = None) -> pd.DataFrame:
        if date_str is None:
            date_str = datetime.now(self.config.LOG_TIMEZONE).strftime("%Y%m%d")
        p = self._trade_path(date_str)
        if not p.exists():
            return pd.DataFrame()
        df = pd.read_csv(p)
        return self._cast_trade_types(df)

    def load_all_trades(self, start_date: str, end_date: str) -> pd.DataFrame:
        """기간 [start_date, end_date] (YYYYMMDD)"""
        all_trades = []
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        current = start
        while current <= end:
            d = current.strftime("%Y%m%d")
            df = self.load_trade_log(d)
            if not df.empty:
                all_trades.append(df)
            current = current + pd.Timedelta(days=1)
        if not all_trades:
            return pd.DataFrame()
        out = pd.concat(all_trades, ignore_index=True)
        return self._cast_trade_types(out)

    def load_recent_trades(self, n: int = 50) -> pd.DataFrame:
        """최근 N개 거래 로드"""
        all_trades = []
        now = datetime.now(timezone.utc)
        
        for i in range(7):  # 최근 7일
            d = (now - pd.Timedelta(days=i)).strftime("%Y%m%d")
            df = self.load_trade_log(d)
            if not df.empty:
                all_trades.append(df)
        
        if not all_trades:
            return pd.DataFrame()
        
        combined = pd.concat(all_trades, ignore_index=True)
        combined = self._cast_trade_types(combined)
        
        # 최신순 정렬
        if 'entry_ts' in combined.columns:
            combined = combined.sort_values('entry_ts', ascending=False)
        
        return combined.head(n).reset_index(drop=True)

    # ---------------------------
    # Feature: 기록/로딩
    # ---------------------------
    def log_feature(
        self,
        bar30_start: datetime,
        bar30_end: datetime,
        pred_ts: datetime,
        entry_ts: datetime,
        label_ts: datetime,
        m1_index_entry: int,
        m1_index_label: int,
        cut_on: float,
        cut_off: float,
        p_prev: Optional[float],
        p_now: float,
        p_cal: float,
        dp: float,
        dmin: float,
        regime: int,
        vol_ratio: float,
        spread_bps: float,
        vwap_gap_bps: float,
        filters_passed: str,
        signal_id: str
    ):
        date_str = pd.to_datetime(bar30_start, utc=True).strftime("%Y%m%d")
        path = self._feature_path(date_str)
        row = {
            'bar30_start': self._to_iso(bar30_start),
            'bar30_end': self._to_iso(bar30_end),
            'pred_ts': self._to_iso(pred_ts),
            'entry_ts': self._to_iso(entry_ts),
            'label_ts': self._to_iso(label_ts),
            'm1_index_entry': int(m1_index_entry),
            'm1_index_label': int(m1_index_label),
            'cut_on': float(cut_on),
            'cut_off': float(cut_off),
            'p_prev': (None if p_prev is None else float(p_prev)),
            'p_now': float(p_now),
            'p_cal': float(p_cal),
            'dp': float(dp),
            'dmin': float(dmin),
            'regime': int(regime),
            'vol_ratio': float(vol_ratio),
            'spread_bps': float(spread_bps),
            'vwap_gap_bps': float(vwap_gap_bps),
            'filters_passed': filters_passed,
            'signal_id': signal_id,
            'model_ver': self.current_versions['model_ver'],
            'feature_ver': self.current_versions['feature_ver'],
            'filter_ver': self.current_versions['filter_ver'],
            'cutoff_ver': self.current_versions['cutoff_ver'],
            'data_ver': self.current_versions['data_ver']
        }
        df = pd.DataFrame([row])
        self._atomic_write(df, path, mode='a', header=not path.exists())

    def load_feature_log(self, date_str: Optional[str] = None) -> pd.DataFrame:
        if date_str is None:
            date_str = datetime.now(self.config.LOG_TIMEZONE).strftime("%Y%m%d")
        path = self._feature_path(date_str)
        if not path.exists():
            return pd.DataFrame()
        df = pd.read_csv(path)
        for col in ['bar30_start', 'bar30_end', 'pred_ts', 'entry_ts', 'label_ts']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
        return df

    # ---------------------------
    # 무결성 검증
    # ---------------------------
    def validate_trade_log(self, df: pd.DataFrame) -> Dict:
        errors = []
        if df is None or df.empty:
            return {'valid': True, 'errors': []}

        required_cols = list(getattr(self.config, 'TRADE_LOG_COLUMNS', []))
        if 'side' not in required_cols:
            required_cols = required_cols + ['side']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            errors.append(f"필수 컬럼 누락: {missing_cols}")

        df = self._cast_trade_types(df)

        if 'entry_ts' in df.columns and 'bar30_end' in df.columns:
            tol = pd.Timedelta(seconds=300)  # 5분
            mismatch = ((df['entry_ts'] - df['bar30_end']).abs() > tol).sum()
            if mismatch > 0:
                errors.append(f"entry_ts != bar30_end (±{int(tol.total_seconds())}초 초과): {mismatch}건")

        if 'label_ts' in df.columns and 'bar30_end' in df.columns:
            expected_label = df['bar30_end'] + pd.Timedelta(minutes=self.config.BAR_MINUTES)
            mask_closed = (df.get('status', 'CLOSED') == 'CLOSED') if 'status' in df.columns else df['result'].notna()
            comp_mask = mask_closed & df['label_ts'].notna()
            mismatch = (df.loc[comp_mask, 'label_ts'] != expected_label[comp_mask]).sum()
            if mismatch > 0:
                errors.append(f"label_ts != bar30_end + {self.config.BAR_MINUTES}분: {mismatch}건")

        nan_cols = []
        for c in df.columns:
            if df[c].isna().any():
                if c in ['result', 'label_price', 'blocked_reason', 'cross_time'] and \
                   (('status' in df.columns and (df['status'] == 'OPEN').any())):
                    continue
                nan_cols.append(c)
        if nan_cols:
            errors.append(f"NaN 존재: {sorted(set(nan_cols))}")

        if 'trade_id' in df.columns:
            dup_count = df['trade_id'].duplicated().sum()
            if dup_count > 0:
                errors.append(f"중복 trade_id: {dup_count}건")

        if 'ttl_valid' in df.columns and df['ttl_valid'].notna().any():
            invalid_ttl = (~df['ttl_valid'].fillna(True)).sum()
            if invalid_ttl > 0:
                errors.append(f"ttl_valid=False: {invalid_ttl}건 (경고)")

        return {'valid': len(errors) == 0, 'errors': errors}

    def validate_feature_log(self, df: pd.DataFrame) -> Dict:
        errors = []
        if df is None or df.empty:
            return {'valid': True, 'errors': []}

        required_cols = list(getattr(self.config, 'FEATURE_LOG_COLUMNS', []))
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            errors.append(f"필수 컬럼 누락: {missing_cols}")

        if df.isna().any().any():
            nan_cols = df.columns[df.isna().any()].tolist()
            errors.append(f"NaN 존재: {nan_cols}")

        if 'signal_id' in df.columns:
            dup_count = df['signal_id'].duplicated().sum()
            if dup_count > 0:
                errors.append(f"중복 signal_id: {dup_count}건")

        return {'valid': len(errors) == 0, 'errors': errors}

    # ---------------------------
    # 통계
    # ---------------------------
    def get_trade_summary(self, date_str: Optional[str] = None) -> Dict:
        df = self.load_trade_log(date_str)
        if df.empty:
            return {'total_trades': 0}

        total = len(df)
        win_rate = float(df['result'].mean()) if 'result' in df.columns else float('nan')
        long_trades = int((df['side'] == 'LONG').sum()) if 'side' in df.columns else 0
        short_trades = int((df['side'] == 'SHORT').sum()) if 'side' in df.columns else 0
        avg_prob = float(df['p_at_entry'].mean()) if 'p_at_entry' in df.columns else float('nan')
        regime_dist = df['regime'].value_counts().to_dict() if 'regime' in df.columns else {}

        blocked = {}
        if 'blocked_reason' in df.columns:
            bser = df['blocked_reason'].dropna()
            blocked = bser.value_counts().to_dict()

        return {
            'total_trades': total,
            'win_rate': win_rate,
            'long_trades': long_trades,
            

            'short_trades': short_trades,
            'avg_probability': avg_prob,
            'regime_distribution': regime_dist,
            'blocked_reasons': blocked
        }

    def print_daily_summary(self, date_str: Optional[str] = None):
        if date_str is None:
            date_str = datetime.now(self.config.LOG_TIMEZONE).strftime("%Y%m%d")
        s = self.get_trade_summary(date_str)

        print(f"\n{'='*60}")
        print(f"거래 요약: {date_str}")
        print(f"{'='*60}")

        if s['total_trades'] == 0:
            print("거래 없음")
            return

        print(f"총 거래: {s['total_trades']}건")
        wr = s.get('win_rate', float('nan'))
        print(f"승률: {wr:.2%}" if pd.notna(wr) else "승률: N/A")
        print(f"LONG: {s.get('long_trades', 0)}건")
        print(f"SHORT: {s.get('short_trades', 0)}건")
        ap = s.get('avg_probability', float('nan'))
        print(f"평균 확률: {ap:.4f}" if pd.notna(ap) else "평균 확률: N/A")

        if s.get('regime_distribution'):
            print(f"\n레짐 분포:")
            for regime, count in s['regime_distribution'].items():
                print(f"  {regime}: {count}건")

        if s.get('blocked_reasons'):
            print(f"\n차단 사유:")
            for reason, count in s['blocked_reasons'].items():
                if reason and str(reason) != 'nan':
                    print(f"  {reason}: {count}건")


# =========================
# 단독 테스트
# =========================
if __name__ == "__main__":
    Config.create_directories()
    lm = LogManager()

    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    bar30_start = now.replace(minute=(now.minute // 30) * 30)
    bar30_end = bar30_start + pd.Timedelta(minutes=30)

    print("\n" + "="*60)
    print("LogManager 테스트")
    print("="*60)

    # [테스트 1] 상세 로깅
    print("\n[테스트 1] 상세 로깅 (기존 방식)")
    lm.log_trade_entry(
        trade_id='TEST_001',
        bar30_start=bar30_start,
        bar30_end=bar30_end,
        entry_ts=bar30_end,
        m1_index_entry=None,
        side='LONG',
        entry_price=42000.0,
        regime=1,
        regime_score=0.7,
        adx=25.0,
        di_plus=30.0,
        di_minus=20.0,
        p_at_entry=0.65,
        dp_at_entry=0.02,
        cut_on=0.6,
        cut_off=0.58,
        cross_time=bar30_end,
        ttl_used_sec=300.0,
        ttl_valid=True,
        refractory_window=30,
        filters_applied='filter1,filter2',
        reason_code='ENTRY_CONFIRMED',
        mode='LIVE',
        payout=0.85
    )

    # [테스트 2] 간소화 로깅 (real_trade 호환)
    print("\n[테스트 2] 간소화 로깅 (real_trade 호환)")
    lm.log_trade_entry_simple(
        trade_id='TEST_002',
        direction=1,  # UP
        entry_price=42100.0,
        entry_time=bar30_end + pd.Timedelta(minutes=1),
        expiry_time=bar30_end + pd.Timedelta(minutes=30),
        p_up=0.67,
        regime=1,
        bar30_start=bar30_start,
        bar30_end=bar30_end,
        stake_recommended=25.0,
        model_version='v1.2.3',
        features_dict={
            'regime_score': 0.8,
            'adx_14': 28.0,
            'di_plus_14': 32.0,
            'di_minus_14': 18.0,
            'dp': 0.03
        }
    )

    print("\n[테스트 3] 간소화 로깅 - DOWN")
    lm.log_trade_entry_simple(
        trade_id='TEST_003',
        direction=0,  # DOWN
        entry_price=42050.0,
        entry_time=bar30_end + pd.Timedelta(minutes=2),
        expiry_time=bar30_end + pd.Timedelta(minutes=30),
        p_up=0.35,  # DOWN이므로 0.5 미만
        regime=-1,
        bar30_start=bar30_start,
        bar30_end=bar30_end,
        stake_recommended=15.0,
        model_version='v1.2.3',
        features_dict={
            'regime_score': -0.6,
            'adx_14': 22.0,
            'di_plus_14': 18.0,
            'di_minus_14': 28.0,
            'dp': 0.02
        }
    )

    # [테스트 4] 청산 업데이트
    print("\n[테스트 4] 청산 업데이트")
    lm.update_trade_result(
        trade_id='TEST_001',
        result=1,
        label_price=42100.0,
        label_ts=bar30_end + pd.Timedelta(minutes=Config.BAR_MINUTES)
    )

    lm.update_trade_result(
        trade_id='TEST_002',
        result=1,
        label_price=42150.0,
        label_ts=bar30_end + pd.Timedelta(minutes=Config.BAR_MINUTES)
    )

    lm.update_trade_result(
        trade_id='TEST_003',
        result=0,
        label_price=42080.0,
        label_ts=bar30_end + pd.Timedelta(minutes=Config.BAR_MINUTES)
    )

    # [테스트 5] 로드 및 검증
    print("\n[테스트 5] 로드 및 검증")
    d = bar30_start.strftime("%Y%m%d")
    df = lm.load_trade_log(d)
    print(f"\n거래 로그 ({len(df)}건):")
    if not df.empty:
        display_cols = ['trade_id', 'side', 'entry_price', 'label_price', 'result', 'p_at_entry', 'status']
        available_cols = [c for c in display_cols if c in df.columns]
        print(df[available_cols])

    val = lm.validate_trade_log(df)
    print(f"\n검증: {'✓ OK' if val['valid'] else '✗ FAIL'}")
    if not val['valid']:
        for err in val['errors']:
            print(f"  - {err}")

    # [테스트 6] 메타 로드
    print("\n[테스트 6] 메타 로드")
    meta_df = lm.load_trade_meta(d)
    if not meta_df.empty:
        print("\n거래 메타 정보:")
        print(meta_df[['trade_id', 'stake_recommended', 'model_version']])
    else:
        print("메타 없음")

    # [테스트 7] 최근 거래 로드
    print("\n[테스트 7] 최근 거래 로드")
    recent = lm.load_recent_trades(n=10)
    print(f"최근 {len(recent)}건 거래 로드됨")
    if not recent.empty:
        print(recent[['trade_id', 'side', 'result']].head())

    # [테스트 8] 일일 요약
    print("\n[테스트 8] 일일 요약")
    lm.print_daily_summary(d)

    print("\n" + "="*60)
    print("✓ 테스트 완료")
    print("="*60)
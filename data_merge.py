# data_merge.py - 데이터 병합 및 관리 모듈 (UTC & ts_min 안전, 30분봉 LogManager 호환)
# - 가격: PRICE_DATA_DIR/raw/prices_YYYYMMDD.csv (또는 prices.csv) 모아서 사용
# - 거래: logs/trades/YYYYMMDD.csv (LogManager가 쓰는 일자별 통합 파일)
# - 피처: logs/features/features_YYYYMMDD.csv
# - ts_min 통일 생성 (가격: timestamp, 거래: bar30_end, 피처: entry_ts 기본)
# - 레거시 trades.csv도 자동 호환

import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple, List


class DataMerger:
    """실시간 데이터 병합 및 관리 (30분봉 로그 스키마 호환)"""

    def __init__(self, config):
        self.config = config
        self.merged_data = None

    # ========================
    # 기본 유틸
    # ========================
    @staticmethod
    def _to_utc_series(s: pd.Series) -> pd.Series:
        """문자열/naive datetime을 UTC-aware Timestamp로 변환"""
        if s is None:
            return pd.Series([], dtype='datetime64[ns, UTC]')
        return pd.to_datetime(s, errors='coerce', utc=True)

    @staticmethod
    def _dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        return df.loc[:, ~df.columns.duplicated()].copy()

    @classmethod
    def _ensure_ts_min(cls, df: pd.DataFrame, time_col: str) -> pd.DataFrame:
        """
        df[time_col]을 UTC로 변환하고 분단위로 내림한 'ts_min' 컬럼 생성(항상 재계산)
        """
        df = df.copy()
        if 'ts_min' in df.columns:
            df.drop(columns=['ts_min'], inplace=True)
        if time_col not in df.columns:
            df['ts_min'] = pd.NaT
            return df
        ts = cls._to_utc_series(df[time_col])
        df[time_col] = ts
        df['ts_min'] = ts.dt.floor('T')
        return df

    # ========================
    # 가격 로딩
    # ========================
    def _price_files(self) -> List[str]:
        raw_dir = os.path.join(self.config.PRICE_DATA_DIR, 'raw')
        files = sorted(glob.glob(os.path.join(raw_dir, 'prices_*.csv')))
        # 백업 플랜: 단일 prices.csv가 있을 수도 있음
        alt = os.path.join(self.config.PRICE_DATA_DIR, 'prices.csv')
        if os.path.exists(alt):
            files.append(alt)
        return files

    def load_price_data(self,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        files = self._price_files()
        if not files:
            return pd.DataFrame()

        dfs = []
        for fp in files:
            try:
                df = pd.read_csv(fp)
            except Exception:
                continue
            df = self._dedup_columns(df)
            # 항상 ts_min 재계산
            if 'ts_min' in df.columns:
                df.drop(columns=['ts_min'], inplace=True, errors='ignore')
            if 'timestamp' in df.columns:
                df['timestamp'] = self._to_utc_series(df['timestamp'])
                if start_date is not None:
                    s = pd.to_datetime(start_date, utc=True)
                    df = df[df['timestamp'] >= s]
                if end_date is not None:
                    e = pd.to_datetime(end_date, utc=True)
                    df = df[df['timestamp'] <= e]
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        price = pd.concat(dfs, ignore_index=True)
        price = self._dedup_columns(price)
        if 'timestamp' in price.columns:
            price = price.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='last')
        return price

    # ========================
    # 거래 로딩 (LogManager 포맷 우선)
    # ========================
    def _trade_files_by_days(self, days: int = 7) -> List[str]:
        """
        logs/trades/YYYYMMDD.csv 최근 N일 파일
        """
        out = []
        base = self.config.TRADE_LOG_DIR
        now = datetime.now(timezone.utc)
        for i in range(days):
            d = (now - timedelta(days=i)).strftime("%Y%m%d")
            fp = base / f"{d}.csv"
            if fp.exists():
                out.append(str(fp))
        return sorted(out)

    def load_trade_logs(self, days: int = 7, include_open: bool = False, join_meta: bool = True) -> pd.DataFrame:
        """
        LogManager가 생성한 날짜별 트레이드 로그를 로드해 표준 스키마로 어댑트.
        (기존 load_trade_logs 대체용)
        """
        from datetime import datetime, timezone, timedelta
        import glob
        base_dir = self.config.TRADE_LOG_DIR
        dfs = []

        for i in range(days):
            d = (datetime.now(timezone.utc) - timedelta(days=i)).strftime("%Y%m%d")
            try:
                daily_path = self.config.get_log_path('trade', d)
            except Exception:
                daily_path = Path(base_dir) / f"{d}.csv"

            if os.path.exists(daily_path):
                df = pd.read_csv(daily_path)
                if df.empty:
                    continue

                # 시간/숫자 변환
                for col in ['entry_ts', 'label_ts', 'bar30_start', 'bar30_end', 'cross_time']:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')

                for col in ['result','entry_price','label_price','p_at_entry','dp_at_entry','regime']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                df['entry_time'] = df.get('entry_ts', pd.NaT)
                df['exit_time']  = df.get('label_ts', pd.NaT)
                df['p_up']       = df.get('p_at_entry', np.nan)

                if 'side' in df.columns:
                    side = df['side'].astype(str).str.upper()
                    df['direction'] = np.where(side == 'LONG', 1, np.where(side == 'SHORT', 0, np.nan))
                elif 'direction' in df.columns:
                    df['direction'] = pd.to_numeric(df['direction'], errors='coerce')
                else:
                    df['direction'] = np.nan

                if not include_open:
                    if 'status' in df.columns:
                        df = df[df['status'] == 'CLOSED']
                    else:
                        df = df[df['result'].notna()]

                df['regime'] = df.get('regime', 0).fillna(0)

                keep = [
                    'trade_id','entry_time','exit_time','direction','p_up','result',
                    'entry_price','label_price','regime','side','p_at_entry','dp_at_entry',
                    'bar30_start','bar30_end','status','model_ver','feature_ver','filter_ver','cutoff_ver'
                ]
                df = df[[c for c in keep if c in df.columns]]

                # 메타 병합
                if join_meta:
                    meta_path = Path(base_dir) / 'meta' / f"{d}_meta.jsonl"
                    if meta_path.exists():
                        records = []
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    records.append(json.loads(line.strip()))
                                except:
                                    pass
                        if records:
                            m = pd.DataFrame(records)
                            if 'timestamp' in m.columns:
                                m['timestamp'] = pd.to_datetime(m['timestamp'], utc=True, errors='coerce')
                            if 'trade_id' in m.columns:
                                meta_cols = [c for c in ['trade_id','stake_recommended','model_version'] if c in m.columns]
                                m = m[meta_cols].drop_duplicates('trade_id', keep='last')
                                df = df.merge(m, on='trade_id', how='left')

                dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        out = pd.concat(dfs, ignore_index=True)
        for col in ['entry_time','exit_time','bar30_start','bar30_end']:
            if col in out.columns:
                out[col] = pd.to_datetime(out[col], utc=True, errors='coerce')
        if 'entry_time' in out.columns:
            out = out.sort_values('entry_time')
        if 'p_up' not in out.columns and 'p_at_entry' in out.columns:
            out['p_up'] = out['p_at_entry']

        return out.reset_index(drop=True)


    # ========================
    # 피처 로딩 (LogManager 포맷)
    # ========================
    def _feature_files_by_days(self, days: int = 7) -> List[str]:
        """
        logs/features/features_YYYYMMDD.csv 최근 N일 파일
        """
        out = []
        now = datetime.now(timezone.utc)
        for i in range(days):
            d = (now - timedelta(days=i)).strftime("%Y%m%d")
            fp = self.config.get_log_path('feature', d)
            if Path(fp).exists():
                out.append(str(fp))
        return sorted(out)

    def load_feature_logs(self, days: int = 7) -> pd.DataFrame:
        files = self._feature_files_by_days(days=days)
        if not files:
            return pd.DataFrame()

        dfs = []
        for fp in files:
            try:
                df = pd.read_csv(fp)
            except Exception:
                continue
            dfs.append(df)
        if not dfs:
            return pd.DataFrame()

        feats = pd.concat(dfs, ignore_index=True)
        feats = self._dedup_columns(feats)

        # 타임스탬프 캐스팅
        for c in ['bar30_start', 'bar30_end', 'pred_ts', 'entry_ts', 'label_ts']:
            if c in feats.columns:
                feats[c] = self._to_utc_series(feats[c])

        # 피처 정렬/중복제거
        # ts_min = entry_ts(있으면) → 없으면 bar30_end
        if 'entry_ts' in feats.columns and feats['entry_ts'].notna().any():
            feats = self._ensure_ts_min(feats, 'entry_ts')
        elif 'bar30_end' in feats.columns:
            feats = self._ensure_ts_min(feats, 'bar30_end')
        else:
            feats['ts_min'] = pd.NaT

        feats = feats.sort_values('ts_min').drop_duplicates(subset=['ts_min'], keep='last')
        return feats

    # ========================
    # 병합
    # ========================
    def merge_all_data(self, price_days: int = 7, trade_days: int = 7, feature_days: int = 7) -> pd.DataFrame:
        """
        가격/거래/피처를 ts_min 기준으로 병합
        - 가격: 정확 매칭
        - 거래: backward asof(최대 5분 허용)
        - 피처: 정확 매칭
        """
        print("데이터 병합 시작...")

        price = self.load_price_data()  # 가격은 전체 파일에서 자동 필터
        trades = self.load_trade_logs(days=trade_days)
        feats = self.load_feature_logs(days=feature_days)

        frames = []
        if not price.empty and 'timestamp' in price.columns:
            price = self._ensure_ts_min(price, 'timestamp')
            frames.append(price[['ts_min']].dropna())
        if not trades.empty:
            frames.append(trades[['ts_min']].dropna())
        if not feats.empty:
            frames.append(feats[['ts_min']].dropna())

        if not frames:
            print("병합할 데이터가 없습니다.")
            return pd.DataFrame()

        base = pd.concat(frames, ignore_index=True).drop_duplicates().sort_values('ts_min')
        merged = base.copy()

        # 가격: 정확 조인
        if not price.empty:
            right_p = price.drop(columns=['timestamp'], errors='ignore').drop_duplicates('ts_min', keep='last')
            merged = merged.merge(right_p, on='ts_min', how='left')

        # 거래: 가장 가까운 이전 시점 asof (5분 허용)
        if not trades.empty:
            right_t = trades.drop_duplicates('ts_min', keep='last')
            merged = pd.merge_asof(
                merged.sort_values('ts_min'),
                right_t.sort_values('ts_min'),
                on='ts_min',
                direction='backward',
                tolerance=pd.Timedelta('5min'),
                suffixes=('', '_trade')
            )

        # 피처: 정확 조인
        if not feats.empty:
            right_f = feats.drop_duplicates('ts_min', keep='last')
            merged = merged.merge(right_f, on='ts_min', how='left', suffixes=('', '_feature'))

        # 대표 timestamp
        if 'timestamp' in merged.columns and pd.api.types.is_datetime64_any_dtype(merged['timestamp']):
            ts = merged['timestamp']
        else:
            ts = merged['ts_min']
        merged['timestamp'] = ts

        merged = self._dedup_columns(merged).sort_values('ts_min').reset_index(drop=True)
        self.merged_data = merged

        # 리포트
        print("\n" + "="*60)
        print("병합 완료:")
        print("="*60)
        print(f"- 전체 레코드 수: {len(merged):,}")
        print(f"- 시작 시간: {merged['timestamp'].min()}")
        print(f"- 종료 시간: {merged['timestamp'].max()}")

        if 'trade_id' in merged.columns:
            tc = merged['trade_id'].notna().sum()
            print(f"- 거래 기록 수: {tc:,}")
            trades_with_price = merged[merged['trade_id'].notna() & merged.get('close').notna()]
            print(f"- 가격 매칭된 거래: {len(trades_with_price):,}건")
            missing = tc - len(trades_with_price)
            if missing > 0:
                print(f"  ⚠️ 가격 누락: {missing}건 (학습 제외됨)")

            if 'result' in merged.columns:
                wr = merged['result'].dropna()
                if not wr.empty:
                    wins = (wr == 1).sum()
                    total = len(wr)
                    print(f"- 승률: {wr.mean()*100:.2f}% ({wins}/{total})")

            if 'regime' in merged.columns:
                print("\n[레짐 분포]")
                regime_data = merged[merged['trade_id'].notna()]['regime']
                labels = {1: "UP 🟢", -1: "DOWN 🔴", 0: "FLAT ⚪"}
                total_with_regime = regime_data.notna().sum()
                if total_with_regime > 0:
                    for rv, cnt in regime_data.value_counts().sort_index().items():
                        name = labels.get(int(rv), f"REGIME-{int(rv)}")
                        pct = (cnt / total_with_regime) * 100
                        print(f"  {name:10s}: {cnt:4d}건 ({pct:5.1f}%)")
        print("="*60 + "\n")

        return merged

    # ========================
    # 학습 데이터 준비
    # ========================
    def build_balanced_training(self, df: pd.DataFrame, min_per_class: int = 2000, recent_days: int = 30) -> pd.DataFrame:
        df = df.dropna(subset=['target', 'timestamp']).copy()
        df['timestamp'] = self._to_utc_series(df['timestamp'])
        if df['timestamp'].isna().all():
            return df.sample(frac=1.0, random_state=42).reset_index(drop=True)

        recent_cut = df['timestamp'].max() - pd.Timedelta(days=recent_days)
        recent = df[df['timestamp'] >= recent_cut]
        up = recent[recent['target'] == 1]
        dn = recent[recent['target'] == 0]

        if len(up) < min_per_class:
            need = min_per_class - len(up)
            pool = df[(df['target'] == 1) & (df['timestamp'] < recent_cut)]
            take = min(need, len(pool))
            if take > 0:
                up = pd.concat([up, pool.sample(take, replace=(len(pool) < need), random_state=42)])

        if len(dn) < min_per_class:
            need = min_per_class - len(dn)
            pool = df[(df['target'] == 0) & (df['timestamp'] < recent_cut)]
            take = min(need, len(pool))
            if take > 0:
                dn = pd.concat([dn, pool.sample(take, replace=(len(pool) < need), random_state=42)])

        balanced = pd.concat([up, dn]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        return balanced

    @staticmethod
    def dedupe_by_hash(df: pd.DataFrame, feature_cols: List[str], round_n: int = 4) -> pd.DataFrame:
        if df.empty:
            return df
        f = df[feature_cols].round(round_n)
        keys = f.apply(lambda r: hash(tuple(r.values)), axis=1)
        return df.loc[~keys.duplicated()].copy()

    def save_merged_data(self, df: Optional[pd.DataFrame] = None) -> bool:
        if df is None:
            df = self.merged_data
        if df is None or df.empty:
            print("저장할 데이터가 없습니다.")
            return False
        os.makedirs(self.config.RESULT_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.config.RESULT_DIR, f'merged_data_{ts}.pkl')
        df.to_pickle(path)
        latest = os.path.join(self.config.RESULT_DIR, 'training_data.pkl')
        df.to_pickle(latest)
        print(f"병합 데이터 저장 완료: {path}")
        return True

    def get_training_data(self,
                          lookback_days: int = 30,
                          apply_balance: bool = True,
                          apply_dedupe: bool = True,
                          dedupe_round: int = 4,
                          min_per_class: int = 2000) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        학습용 데이터 준비
        - 최근 N일 필터
        - FeatureEngineer로 feature/target 생성
        - (옵션) 클래스 밸런싱 + 디듀프
        - ★ regime 컬럼 보존/통계
        """
        latest_path = os.path.join(self.config.RESULT_DIR, 'training_data.pkl')
        if os.path.exists(latest_path):
            df = pd.read_pickle(latest_path)
        else:
            df = self.merge_all_data()
            if df is None or df.empty:
                return None, None

        if 'timestamp' in df.columns:
            cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
            ts = self._to_utc_series(df['timestamp'])
            df = df.assign(timestamp=ts)
            df = df[df['timestamp'] >= cutoff].copy()

        # FeatureEngineer 위치 별도 모듈이라면 아래 임포트 라인 확인
        from feature_engineer import FeatureEngineer
        fe = FeatureEngineer()

        X = fe.create_feature_pool(df)
        y = fe.create_target(df, window=self.config.PREDICTION_WINDOW)

        valid = y.notna()
        X = X[valid].copy()
        y = y[valid].copy()
        if X.empty or y.empty:
            return None, None

        if 'timestamp' in df.columns:
            try:
                ts_series = df.loc[X.index, 'timestamp']
            except Exception:
                ts_series = df['timestamp'].iloc[-len(X):].reset_index(drop=True)
                X = X.reset_index(drop=True)
                y = y.reset_index(drop=True)
        else:
            ts_series = pd.Series([pd.NaT]*len(X), dtype='datetime64[ns, UTC]')

        tmp = X.copy()
        tmp['target'] = y.values
        tmp['timestamp'] = ts_series.values

        feat_cols = list(X.columns)
        if apply_balance:
            tmp = self.build_balanced_training(tmp, min_per_class=min_per_class, recent_days=30)
        if apply_dedupe and len(tmp) > 0:
            tmp = self.dedupe_by_hash(tmp, feat_cols, round_n=dedupe_round)

        X_final = tmp[feat_cols].copy()
        y_final = tmp['target'].copy()

        # 레짐 통계
        if 'regime' in X_final.columns:
            have = int(X_final['regime'].notna().sum())
            none = int(X_final['regime'].isna().sum())
            print("\n[학습 데이터 레짐 정보]")
            print(f"  레짐 정보 있음: {have:,}건")
            print(f"  레짐 정보 없음: {none:,}건")

        return X_final, y_final

    # ==============
    # 레거시 지원 (선택)
    # ==============
    def add_new_price_data(self, new_data: pd.DataFrame) -> bool:
        """실시간 가격 추가 (분단위 디듀프, 최신값 우선) — 기존 파이프와 호환용"""
        today = datetime.now().strftime("%Y%m%d")
        fp = os.path.join(self.config.PRICE_DATA_DIR, 'raw', f'prices_{today}.csv')

        new = new_data.copy()
        new['timestamp'] = pd.to_datetime(new['timestamp'], errors='coerce', utc=True)
        new['ts_min'] = new['timestamp'].dt.floor('T')

        if os.path.exists(fp):
            existing = pd.read_csv(fp)
            if 'timestamp' in existing.columns:
                existing['timestamp'] = pd.to_datetime(existing['timestamp'], errors='coerce', utc=True)
                existing['ts_min'] = existing['timestamp'].dt.floor('T')
            merged = pd.concat([existing, new], ignore_index=True)
        else:
            merged = new

        merged = merged.sort_values('timestamp').drop_duplicates(subset=['ts_min'], keep='last')
        merged.drop(columns=['ts_min'], inplace=True, errors='ignore')
        merged.to_csv(fp, index=False, encoding='utf-8-sig')
        print(f"가격 데이터 추가 완료: {len(new_data)} 레코드")
        return True

    def cleanup_old_data(self, days_to_keep: int = 90):
        """오래된 가격 raw 파일 정리 & 레거시 거래 로그 아카이브(옵션)"""
        cutoff = datetime.now() - timedelta(days=days_to_keep)

        price_files = glob.glob(os.path.join(self.config.PRICE_DATA_DIR, 'raw', '*.csv'))
        for file in price_files:
            filename = os.path.basename(file)
            if filename.startswith('prices_'):
                date_str = filename.replace('prices_', '').replace('.csv', '')
                try:
                    file_date = datetime.strptime(date_str, "%Y%m%d")
                    if file_date < cutoff:
                        os.remove(file)
                        print(f"오래된 파일 삭제: {filename}")
                except Exception:
                    continue


class DataValidator:
    """데이터 검증"""

    @staticmethod
    def validate_price_data(df: pd.DataFrame):
        issues = []
        required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing = [c for c in required if c not in df.columns]
        if missing:
            issues.append(f"누락된 컬럼: {missing}")

        if all(c in df.columns for c in ['open','high','low','close']):
            invalid_high = df[df['high'] < df[['open','close']].max(axis=1)]
            if not invalid_high.empty:
                issues.append(f"잘못된 고가: {len(invalid_high)} 레코드")
            invalid_low = df[df['low'] > df[['open','close']].min(axis=1)]
            if not invalid_low.empty:
                issues.append(f"잘못된 저가: {len(invalid_low)} 레코드")

        key = 'ts_min' if 'ts_min' in df.columns else 'timestamp'
        if key in df.columns:
            d0 = df.dropna(subset=[key])
            dups = d0[d0.duplicated(subset=[key], keep=False)]
            if not dups.empty:
                issues.append(f"중복 {key}: {len(dups)} 레코드")

        if set(required).issubset(df.columns):
            nulls = df[required].isnull().sum()
            if nulls.any():
                issues.append(f"결측치: {nulls[nulls > 0].to_dict()}")

        return len(issues) == 0, issues

    @staticmethod
    def validate_trade_logs(df: pd.DataFrame):
        """
        거래 로그 검증 (30분봉 LogManager 스키마)
          - 필수: trade_id, bar30_end(or entry_ts), side, result(선택)
          - regime ∈ {-1,0,1}
        """
        issues = []
        required_any_time = [('bar30_end', 'entry_ts')]  # 둘 중 하나는 있어야 함
        required = ['trade_id', 'side']

        missing = [c for c in required if c not in df.columns]
        if missing:
            issues.append(f"누락된 컬럼: {missing}")

        ok_time = ('bar30_end' in df.columns) or ('entry_ts' in df.columns)
        if not ok_time:
            issues.append("누락된 시간 컬럼: bar30_end 또는 entry_ts 필요")

        if 'trade_id' in df.columns:
            dups = df[df.duplicated(subset=['trade_id'], keep=False)]
            if not dups.empty:
                issues.append(f"중복 trade_id: {len(dups)} 레코드")

        if 'side' in df.columns:
            bad = df[~df['side'].isin(['LONG', 'SHORT'])]
            if not bad.empty:
                issues.append(f"잘못된 side 값: {len(bad)} 레코드")

        if 'regime' in df.columns:
            regime_data = pd.to_numeric(df['regime'], errors='coerce').dropna()
            bad_regime = regime_data[~regime_data.isin([-1, 0, 1])]
            if not bad_regime.empty:
                issues.append(f"잘못된 regime 값: {len(bad_regime)} 레코드")

        return len(issues) == 0, issues


# =========================
# 단독 테스트
# =========================
if __name__ == "__main__":
    from config import Config

    dm = DataMerger(Config)

    merged = dm.merge_all_data()
    if not merged.empty:
        merged = merged.dropna(subset=['open','high','low','close','volume'], how='any')
        dm.save_merged_data(merged)

        X, y = dm.get_training_data(lookback_days=30)
        if X is not None:
            print("\n학습 데이터 준비 완료:")
            print(f"- 피처 shape: {X.shape}")
            print(f"- 타겟 shape: {y.shape}")
            print(f"- 클래스 분포: {y.value_counts().to_dict()}")

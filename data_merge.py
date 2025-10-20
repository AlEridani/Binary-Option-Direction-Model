# data_merge.py - 데이터 병합 및 관리 모듈 (UTC & ts_min 안전, 거래/피처만 있어도 병합 가능, regime 지원)

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta, timezone
import glob
from pathlib import Path


class DataMerger:
    """실시간 데이터 병합 및 관리"""

    def __init__(self, config):
        self.config = config
        self.merged_data = None

    # ------------------------
    # 내부 헬퍼: 안전한 시간/컬럼 처리
    # ------------------------
    @staticmethod
    def _to_utc_series(s):
        """문자열/naive datetime을 UTC-aware Timestamp로 변환"""
        if s is None:
            return pd.Series([], dtype='datetime64[ns, UTC]')
        return pd.to_datetime(s, errors='coerce', utc=True)

    @staticmethod
    def _dedup_columns(df):
        """중복 컬럼명이 있으면 첫 번째만 유지"""
        if df is None or df.empty:
            return df
        return df.loc[:, ~df.columns.duplicated()].copy()

    @classmethod
    def _ensure_ts_min(cls, df, time_col):
        """
        df[time_col]을 UTC로 변환하고 분단위로 내림한 'ts_min' 컬럼을 '한 번만' 만든다.
        - 기존에 ts_min이 있으면 제거 후 재계산 (중복 라벨 방지)
        - time_col이 없거나 전부 NaT면 ts_min은 NaT
        """
        df = df.copy()
        if 'ts_min' in df.columns:
            df = df.drop(columns=['ts_min'])
        if time_col not in df.columns:
            df['ts_min'] = pd.NaT
            return df

        ts = cls._to_utc_series(df[time_col])
        df[time_col] = ts
        df['ts_min'] = ts.dt.floor('T')
        return df

    # ------------------------
    # 데이터 로더
    # ------------------------
    def load_price_data(self, start_date=None, end_date=None):
        """
        가격 데이터 로드 (UTC 변환, timestamp 중복 제거/정렬)
        - 원천에 ts_min이 있어도 무시 (재계산)
        """
        price_dir = os.path.join(self.config.PRICE_DATA_DIR, 'raw')
        files = glob.glob(os.path.join(price_dir, '*.csv'))

        if not files:
            return pd.DataFrame()

        dfs = []
        for file in sorted(files):
            df = pd.read_csv(file)
            df = self._dedup_columns(df)
            if 'ts_min' in df.columns:
                df = df.drop(columns=['ts_min'])

            if 'timestamp' in df.columns:
                df['timestamp'] = self._to_utc_series(df['timestamp'])

                if start_date:
                    start_ts = self._to_utc_series(pd.Series([start_date]))[0]
                    df = df[df['timestamp'] >= start_ts]
                if end_date:
                    end_ts = self._to_utc_series(pd.Series([end_date]))[0]
                    df = df[df['timestamp'] <= end_ts]

            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df = self._dedup_columns(merged_df)

        if 'timestamp' in merged_df.columns:
            merged_df = merged_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')

        return merged_df

    def load_trade_logs(self):
        """
        거래 로그 로드 (UTC 변환, regime 컬럼 처리)
        - regime 컬럼이 없으면 추가 (레거시 데이터 대응)
        - ts_min은 병합 시 통일 생성
        """
        path = os.path.join(self.config.TRADE_LOG_DIR, 'trades.csv')
        if not os.path.exists(path):
            return pd.DataFrame()

        df = pd.read_csv(path)
        df = self._dedup_columns(df)

        # ★ regime 컬럼 처리 (없으면 추가)
        if 'regime' not in df.columns:
            df['regime'] = pd.NA
            print("ℹ️  레거시 데이터: regime 컬럼 추가 (NaN)")
        else:
            # regime 컬럼이 있어도 숫자형으로 변환
            df['regime'] = pd.to_numeric(df['regime'], errors='coerce')

        # ★ p_up 컬럼도 숫자형으로 변환 (있으면)
        if 'p_up' in df.columns:
            df['p_up'] = pd.to_numeric(df['p_up'], errors='coerce')

        if 'entry_time' in df.columns:
            df['entry_time'] = self._to_utc_series(df['entry_time'])
        if 'exit_time' in df.columns:
            df['exit_time'] = self._to_utc_series(df['exit_time'])

        if 'ts_min' in df.columns:
            df = df.drop(columns=['ts_min'])
        
        return df

    def load_feature_logs(self):
        """피처 로그 로드 (UTC 변환), ts_min은 병합 시 통일 생성"""
        feature_dir = self.config.FEATURE_LOG_DIR
        files = glob.glob(os.path.join(feature_dir, 'features_*.csv'))
        if not files:
            return pd.DataFrame()

        dfs = []
        for file in sorted(files):
            df = pd.read_csv(file)
            df = self._dedup_columns(df)
            if 'timestamp' in df.columns:
                df['timestamp'] = self._to_utc_series(df['timestamp'])
            if 'ts_min' in df.columns:
                df = df.drop(columns=['ts_min'])
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        merged_df = pd.concat(dfs, ignore_index=True)
        merged_df = self._dedup_columns(merged_df)
        if 'timestamp' in merged_df.columns:
            merged_df = merged_df.drop_duplicates().sort_values('timestamp')
        return merged_df

    # ------------------------
    # 병합
    # ------------------------
    def merge_all_data(self):
        """
        모든 데이터 병합 (ts_min 기준)
        - 가격/거래/피처 각각에서 ts_min 생성(1회) 후 병합
        - 가격 데이터가 없어도 거래/피처만으로 병합 가능
        - 최종 timestamp는 (가능하면) 가격의 timestamp, 없으면 ts_min
        - ★ regime 컬럼 포함 통계 출력
        """
        print("데이터 병합 시작...")

        price = self.load_price_data()
        trades = self.load_trade_logs()
        feats = self.load_feature_logs()

        frames = []
        if not price.empty and 'timestamp' in price.columns:
            price = self._ensure_ts_min(price, 'timestamp')
            frames.append(price[['ts_min']].dropna())
        if not trades.empty and 'entry_time' in trades.columns:
            trades = self._ensure_ts_min(trades, 'entry_time')
            frames.append(trades[['ts_min']].dropna())
        if not feats.empty and 'timestamp' in feats.columns:
            feats = self._ensure_ts_min(feats, 'timestamp')
            frames.append(feats[['ts_min']].dropna())

        if not frames:
            print("병합할 데이터가 없습니다.")
            return None

        base = pd.concat(frames).drop_duplicates().sort_values('ts_min')

        merged = base.copy()

        if not price.empty:
            right = price.drop(columns=['timestamp'], errors='ignore')
            right = right.drop_duplicates('ts_min', keep='last')
            merged = merged.merge(right, on='ts_min', how='left')

        if not trades.empty:
            keep_cols = [c for c in trades.columns if c not in ['entry_time', 'exit_time', 'ts_min']]
            right = trades[['ts_min'] + keep_cols].drop_duplicates('ts_min', keep='last')
            merged = merged.merge(right, on='ts_min', how='left', suffixes=('', '_trade'))

        if not feats.empty:
            right = feats.drop(columns=['timestamp', 'ts_min'], errors='ignore')
            right = pd.concat([feats[['ts_min']], right], axis=1).drop_duplicates('ts_min', keep='last')           
            merged = merged.merge(right, on='ts_min', how='left', suffixes=('', '_feature'))

        # 대표 timestamp 생성
        if 'timestamp' in merged.columns and pd.api.types.is_datetime64_any_dtype(merged['timestamp']):
            ts = merged['timestamp']
        else:
            ts = merged['ts_min']
        merged['timestamp'] = ts

        merged = self._dedup_columns(merged).sort_values('ts_min')
        self.merged_data = merged

        # ★ 통계 출력 (regime 포함)
        print("\n" + "="*60)
        print("병합 완료:")
        print("="*60)
        print(f"- 전체 레코드 수: {len(merged):,}")
        print(f"- 시작 시간: {merged['timestamp'].min()}")
        print(f"- 종료 시간: {merged['timestamp'].max()}")

        if 'trade_id' in merged.columns:
            tc = merged['trade_id'].notna().sum()
            print(f"- 거래 기록 수: {tc:,}")
            
            if 'result' in merged.columns:
                wr = merged['result'].dropna()
                if not wr.empty:
                    wins = (wr == 1).sum()
                    total = len(wr)
                    print(f"- 승률: {wr.mean()*100:.2f}% ({wins}/{total})")
            
            # ★ 레짐 분포 통계
            if 'regime' in merged.columns:
                print("\n[레짐 분포]")
                regime_data = merged[merged['trade_id'].notna()]['regime']
                
                regime_labels = {
                    0.0: "UP 트렌드 🟢",
                    1.0: "DOWN 트렌드 🔴",
                    2.0: "FLAT 횡보 ⚪"
                }
                
                total_with_regime = regime_data.notna().sum()
                total_without_regime = regime_data.isna().sum()
                
                if total_with_regime > 0:
                    regime_counts = regime_data.value_counts().sort_index()
                    for regime_val, count in regime_counts.items():
                        regime_name = regime_labels.get(regime_val, f"REGIME-{int(regime_val)}")
                        pct = (count / total_with_regime) * 100
                        print(f"  {regime_name:20s}: {count:4d}건 ({pct:5.1f}%)")
                
                if total_without_regime > 0:
                    print(f"  {'레거시 (N/A)':20s}: {total_without_regime:4d}건")
                
                # ★ 레짐별 승률 (result가 있으면)
                if 'result' in merged.columns:
                    print("\n[레짐별 승률]")
                    regime_result = merged[merged['regime'].notna() & merged['result'].notna()]
                    
                    if len(regime_result) > 0:
                        for regime_val in sorted(regime_result['regime'].unique()):
                            regime_mask = regime_result['regime'] == regime_val
                            regime_subset = regime_result[regime_mask]
                            
                            wins = (regime_subset['result'] == 1).sum()
                            total = len(regime_subset)
                            win_rate = wins / total if total > 0 else 0
                            
                            regime_name = regime_labels.get(regime_val, f"REGIME-{int(regime_val)}")
                            
                            # 승률에 따른 이모지
                            if win_rate >= 0.60:
                                emoji = "🔥"
                            elif win_rate >= 0.55:
                                emoji = "✅"
                            elif win_rate >= 0.50:
                                emoji = "⚠️"
                            else:
                                emoji = "❌"
                            
                            print(f"  {regime_name:20s}: {win_rate*100:5.1f}% ({wins:3d}/{total:3d}) {emoji}")

        print("="*60 + "\n")

        return merged

    # ------------------------
    # 학습 데이터 준비
    # ------------------------
    def build_balanced_training(self, df, min_per_class=2000, recent_days=30):
        """
        최근 구간을 중심으로 하되, 부족 클래스는 과거에서 보충해
        최소 표본수를 맞추는 균형 데이터셋 구성.
        df: feature + target + timestamp 포함 DataFrame
        """
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
    def dedupe_by_hash(df, feature_cols, round_n=4):
        """
        피처값을 반올림해 해시 키로 유사 샘플 제거.
        연속진입 등으로 비슷한 샘플이 도배될 때 과학습/편향 완화.
        """
        if df.empty:
            return df
        f = df[feature_cols].round(round_n)
        keys = f.apply(lambda r: hash(tuple(r.values)), axis=1)
        return df.loc[~keys.duplicated()].copy()

    def save_merged_data(self, df=None):
        """병합된 데이터 저장 (피클 2개: 타임스탬프/최신)"""
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

    def get_training_data(self, lookback_days=30, apply_balance=True, apply_dedupe=True,
                          dedupe_round=4, min_per_class=2000):
        """
        학습용 데이터 준비
        - 최신 병합 데이터 없으면 새로 병합
        - 최근 N일 필터
        - FeatureEngineer로 feature/target 생성
        - (옵션) 클래스 밸런싱 + 디듀프 적용
        - ★ regime 컬럼 보존
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

        from model_train import FeatureEngineer
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
        
        # ★ regime 컬럼 통계 출력
        if 'regime' in X_final.columns:
            regime_with_data = X_final['regime'].notna().sum()
            regime_without_data = X_final['regime'].isna().sum()
            print(f"\n[학습 데이터 레짐 정보]")
            print(f"  레짐 정보 있음: {regime_with_data:,}건")
            print(f"  레짐 정보 없음: {regime_without_data:,}건 (레거시)")
        
        return X_final, y_final

    def update_trade_result(self, trade_id, result, profit_loss):
        """거래 결과 업데이트"""
        path = os.path.join(self.config.TRADE_LOG_DIR, 'trades.csv')
        if not os.path.exists(path):
            return False

        df = pd.read_csv(path)
        df = self._dedup_columns(df)
        if 'trade_id' not in df.columns:
            return False

        mask = df['trade_id'] == trade_id
        if mask.any():
            df.loc[mask, 'result'] = result
            df.loc[mask, 'profit_loss'] = profit_loss
            df.loc[mask, 'exit_time'] = datetime.now(timezone.utc).isoformat()
            df.to_csv(path, index=False, encoding='utf-8-sig')
            print(f"거래 {trade_id} 결과 업데이트 완료")
            return True
        return False

    def add_new_price_data(self, new_data):
        """실시간 저장 (분단위 디듀프, 최신값 우선)"""
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
        merged = merged.drop(columns=['ts_min'], errors='ignore')
        merged.to_csv(fp, index=False, encoding='utf-8-sig')
        print(f"가격 데이터 추가 완료: {len(new_data)} 레코드")
        return True

    def cleanup_old_data(self, days_to_keep=90):
        """오래된 데이터 정리 & 거래 로그 아카이브"""
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

        trade_log = os.path.join(self.config.TRADE_LOG_DIR, 'trades.csv')
        if os.path.exists(trade_log):
            df = pd.read_csv(trade_log)
            df = self._dedup_columns(df)
            if 'entry_time' in df.columns:
                df['entry_time'] = self._to_utc_series(df['entry_time'])
                cutoff_utc = cutoff.replace(tzinfo=timezone.utc)
                old = df[df['entry_time'] < cutoff_utc]
                if not old.empty:
                    archive = os.path.join(self.config.TRADE_LOG_DIR,
                                           f'trades_archive_{cutoff.strftime("%Y%m%d")}.csv')
                    old.to_csv(archive, index=False, encoding='utf-8-sig')
                    df = df[df['entry_time'] >= cutoff_utc]
                    df.to_csv(trade_log, index=False, encoding='utf-8-sig')
                    print(f"거래 로그 아카이브 완료: {len(old)} 레코드")


class DataValidator:
    """데이터 검증 클래스"""

    @staticmethod
    def validate_price_data(df):
        """가격 데이터 검증"""
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

        nulls = df[required].isnull().sum()
        if nulls.any():
            issues.append(f"결측치: {nulls[nulls > 0].to_dict()}")

        return len(issues) == 0, issues

    @staticmethod
    def validate_trade_logs(df):
        """거래 로그 검증 (현재 스키마: direction 사용, regime 선택적)"""
        issues = []
        required = ['trade_id', 'entry_time', 'direction']
        missing = [c for c in required if c not in df.columns]
        if missing:
            issues.append(f"누락된 컬럼: {missing}")

        if 'trade_id' in df.columns:
            dups = df[df.duplicated(subset=['trade_id'], keep=False)]
            if not dups.empty:
                issues.append(f"중복 trade_id: {len(dups)} 레코드")

        if 'direction' in df.columns:
            bad = df[~df['direction'].isin([0, 1])]
            if not bad.empty:
                issues.append(f"잘못된 direction 값: {len(bad)} 레코드")
        
        # ★ regime 검증 (선택적)
        if 'regime' in df.columns:
            regime_data = df['regime'].dropna()
            if len(regime_data) > 0:
                bad_regime = regime_data[~regime_data.isin([0, 1, 2])]
                if not bad_regime.empty:
                    issues.append(f"잘못된 regime 값: {len(bad_regime)} 레코드")

        return len(issues) == 0, issues


# 사용 예시
if __name__ == "__main__":
    from config import Config

    merger = DataMerger(Config)

    merged = merger.merge_all_data()
    
    if merged is not None:
        merged = merged.dropna(subset=['open','high','low','close','volume'])
        merger.save_merged_data()

        X, y = merger.get_training_data(lookback_days=30)
        if X is not None:
            print("\n학습 데이터 준비 완료:")
            print(f"- 피처 shape: {X.shape}")
            print(f"- 타겟 shape: {y.shape}")
            print(f"- 클래스 분포: {y.value_counts().to_dict()}")
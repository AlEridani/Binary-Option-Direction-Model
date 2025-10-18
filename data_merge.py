# data_merge.py - 병합/관리 (timestamp=close_time 가정)
import os
import glob
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


class DataMerger:
    def __init__(self, config):
        self.config = config
        self.merged_data = None

    def load_price_data(self, start_date=None, end_date=None) -> pd.DataFrame:
        price_dir = os.path.join(self.config.PRICE_DATA_DIR, "raw")
        files = sorted(glob.glob(os.path.join(price_dir, "*.csv")))
        if not files:
            print("가격 데이터 파일이 없습니다.")
            return pd.DataFrame()

        dfs = []
        for fp in files:
            df = pd.read_csv(fp)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
                if start_date:
                    df = df[df["timestamp"] >= pd.to_datetime(start_date, utc=True)]
                if end_date:
                    df = df[df["timestamp"] <= pd.to_datetime(end_date, utc=True)]
            dfs.append(df)

        merged = pd.concat(dfs, ignore_index=True)
        merged = merged.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        return merged

    def load_trade_logs(self) -> pd.DataFrame:
        fp = os.path.join(self.config.TRADE_LOG_DIR, "trades.csv")
        if not os.path.exists(fp):
            return pd.DataFrame()
        cols = [
            "trade_id","entry_time","entry_price","prediction","confidence",
            "amount","status","result","profit_loss","exit_time","hold_secs"
        ]
        df = pd.read_csv(fp, names=cols, header=0)
        df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
        df["exit_time"]  = pd.to_datetime(df["exit_time"],  utc=True, errors="coerce")
        return df

    def load_feature_logs(self) -> pd.DataFrame:
        fl_dir = self.config.FEATURE_LOG_DIR
        files = sorted(glob.glob(os.path.join(fl_dir, "features_*.csv")))
        if not files:
            return pd.DataFrame()
        dfs = []
        for fp in files:
            df = pd.read_csv(fp)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            dfs.append(df)
        merged = pd.concat(dfs, ignore_index=True)
        return merged.drop_duplicates().sort_values("timestamp")

    def merge_all_data(self) -> pd.DataFrame:
        print("데이터 병합 시작...")
        price = self.load_price_data()
        if price.empty:
            print("가격 데이터 없음")
            return None

        trades = self.load_trade_logs()
        feats  = self.load_feature_logs()

        merged = price.copy()
        # 필요시 feature timestamp로 조인(분석/라벨 생성 용도)
        if not feats.empty:
            merged = pd.merge(merged, feats, on="timestamp", how="left", suffixes=("", "_feat"))

        self.merged_data = merged
        print(f"병합 완료: rows={len(merged)}, 기간={merged['timestamp'].min()} ~ {merged['timestamp'].max()}")
        return merged

    def save_merged_data(self, df=None):
        if df is None:
            df = self.merged_data
        if df is None or df.empty:
            print("저장할 데이터가 없습니다.")
            return False
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.config.RESULT_DIR, f"merged_data_{ts}.pkl")
        df.to_pickle(path)
        latest = os.path.join(self.config.RESULT_DIR, "training_data.pkl")
        df.to_pickle(latest)
        print(f"병합 데이터 저장: {path}")
        return True

    def get_training_data(self, lookback_days=30):
        latest = os.path.join(self.config.RESULT_DIR, "training_data.pkl")
        if os.path.exists(latest):
            df = pd.read_pickle(latest)
        else:
            df = self.merge_all_data()
            if df is None:
                return None, None

        cutoff = pd.Timestamp.utcnow().tz_localize("UTC") - pd.Timedelta(days=lookback_days)
        if "timestamp" in df.columns:
            df = df[df["timestamp"] >= cutoff]

        from model_train import FeatureEngineer
        fe = FeatureEngineer()
        X = fe.create_feature_pool(df)
        y = fe.create_target(df, window=self.config.PREDICTION_WINDOW)
        valid = y.notna()
        return X[valid], y[valid]

    def add_new_price_data(self, new_df: pd.DataFrame):
        today = pd.Timestamp.utcnow().tz_localize("UTC").strftime("%Y%m%d")
        fp = os.path.join(self.config.PRICE_DATA_DIR, "raw", f"prices_{today}.csv")
        if os.path.exists(fp):
            ex = pd.read_csv(fp)
            upd = pd.concat([ex, new_df], ignore_index=True)
            upd = upd.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
        else:
            upd = new_df
        upd.to_csv(fp, index=False)
        print(f"가격 데이터 추가: {len(new_df)} rows")


class DataValidator:
    @staticmethod
    def validate_price_data(df: pd.DataFrame):
        issues = []
        req = ["timestamp","open","high","low","close","volume"]
        miss = [c for c in req if c not in df.columns]
        if miss:
            issues.append(f"누락 컬럼: {miss}")
        if "timestamp" in df.columns:
            dup = df[df.duplicated(subset=["timestamp"], keep=False)]
            if not dup.empty:
                issues.append(f"중복 timestamp: {len(dup)}")
        return (len(issues) == 0), issues

# data_merge.py - 데이터 병합 및 관리 모듈

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import glob
from pathlib import Path

class DataMerger:
    """실시간 데이터 병합 및 관리"""
    
    def __init__(self, config):
        self.config = config
        self.merged_data = None
        
    def load_price_data(self, start_date=None, end_date=None):
        """
        가격 데이터 로드
        """
        price_dir = os.path.join(self.config.PRICE_DATA_DIR, 'raw')
        files = glob.glob(os.path.join(price_dir, '*.csv'))
        
        if not files:
            print("가격 데이터 파일이 없습니다.")
            return pd.DataFrame()
        
        dfs = []
        for file in sorted(files):
            df = pd.read_csv(file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # 날짜 필터링
                if start_date:
                    df = df[df['timestamp'] >= start_date]
                if end_date:
                    df = df[df['timestamp'] <= end_date]
            
            dfs.append(df)
        
        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_df = merged_df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
            return merged_df
        
        return pd.DataFrame()
    
    def load_trade_logs(self):
        """
        거래 로그 로드
        """
        trade_log_path = os.path.join(self.config.TRADE_LOG_DIR, 'trades.csv')
        
        if os.path.exists(trade_log_path):
            df = pd.read_csv(trade_log_path)
            if 'entry_time' in df.columns:
                df['entry_time'] = pd.to_datetime(df['entry_time'])
            if 'exit_time' in df.columns:
                df['exit_time'] = pd.to_datetime(df['exit_time'])
            return df
        
        return pd.DataFrame()
    
    def load_feature_logs(self):
        """
        피처 로그 로드
        """
        feature_log_dir = self.config.FEATURE_LOG_DIR
        feature_files = glob.glob(os.path.join(feature_log_dir, 'features_*.csv'))
        
        if not feature_files:
            return pd.DataFrame()
        
        dfs = []
        for file in sorted(feature_files):
            df = pd.read_csv(file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            dfs.append(df)
        
        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            return merged_df.drop_duplicates().sort_values('timestamp')
        
        return pd.DataFrame()
    
    def merge_all_data(self):
        """
        모든 데이터 병합
        """
        print("데이터 병합 시작...")
        
        # 각 데이터 로드
        price_data = self.load_price_data()
        trade_logs = self.load_trade_logs()
        feature_logs = self.load_feature_logs()
        
        if price_data.empty:
            print("가격 데이터가 없어 병합할 수 없습니다.")
            return None
        
        # 기본 데이터프레임은 가격 데이터
        merged = price_data.copy()
        
        # 거래 로그 병합
        if not trade_logs.empty:
            # 거래 결과를 가격 데이터에 매핑
            for idx, trade in trade_logs.iterrows():
                entry_time = trade['entry_time']
                
                # 해당 시간의 인덱스 찾기
                time_mask = merged['timestamp'] == entry_time
                if time_mask.any():
                    merged.loc[time_mask, 'trade_id'] = trade.get('trade_id', idx)
                    merged.loc[time_mask, 'prediction'] = trade.get('prediction', np.nan)
                    merged.loc[time_mask, 'actual_result'] = trade.get('result', np.nan)
                    merged.loc[time_mask, 'profit_loss'] = trade.get('profit_loss', np.nan)
        
        # 피처 로그 병합
        if not feature_logs.empty:
            # timestamp 기준으로 조인
            merged = pd.merge(merged, feature_logs, on='timestamp', how='left', suffixes=('', '_feature'))
        
        # 결과 저장
        self.merged_data = merged
        
        # 통계 출력
        print(f"\n병합 완료:")
        print(f"- 전체 레코드 수: {len(merged)}")
        print(f"- 시작 시간: {merged['timestamp'].min()}")
        print(f"- 종료 시간: {merged['timestamp'].max()}")
        
        if 'trade_id' in merged.columns:
            trade_count = merged['trade_id'].notna().sum()
            print(f"- 거래 기록 수: {trade_count}")
            
            if 'actual_result' in merged.columns:
                win_rate = merged[merged['actual_result'].notna()]['actual_result'].mean()
                print(f"- 승률: {win_rate*100:.2f}%")
        
        return merged
    
    def save_merged_data(self, df=None):
        """
        병합된 데이터 저장
        """
        if df is None:
            df = self.merged_data
        
        if df is None or df.empty:
            print("저장할 데이터가 없습니다.")
            return False
        
        # 결과 디렉토리에 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.config.RESULT_DIR, f'merged_data_{timestamp}.pkl')
        df.to_pickle(filepath)
        
        # 최신 파일 링크 생성
        latest_path = os.path.join(self.config.RESULT_DIR, 'training_data.pkl')
        df.to_pickle(latest_path)
        
        print(f"병합 데이터 저장 완료: {filepath}")
        return True
    
    def get_training_data(self, lookback_days=30):
        """
        학습용 데이터 준비
        """
        # 최신 병합 데이터 로드
        latest_path = os.path.join(self.config.RESULT_DIR, 'training_data.pkl')
        
        if os.path.exists(latest_path):
            df = pd.read_pickle(latest_path)
        else:
            # 새로 병합
            df = self.merge_all_data()
            if df is None:
                return None, None
        
        # 최근 N일 데이터만 사용
        if 'timestamp' in df.columns:
            cutoff_date = datetime.now() - timedelta(days=lookback_days)
            df = df[df['timestamp'] >= cutoff_date]
        
        from model_train import FeatureEngineer
        
        # 피처 및 타겟 생성
        feature_engineer = FeatureEngineer()
        features = feature_engineer.create_feature_pool(df)
        target = feature_engineer.create_target(df, window=self.config.PREDICTION_WINDOW)
        
        # 유효한 데이터만 반환
        valid_idx = target.notna()
        return features[valid_idx], target[valid_idx]
    
    def update_trade_result(self, trade_id, result, profit_loss):
        """
        거래 결과 업데이트
        """
        trade_log_path = os.path.join(self.config.TRADE_LOG_DIR, 'trades.csv')
        
        if os.path.exists(trade_log_path):
            df = pd.read_csv(trade_log_path)
            
            # trade_id로 해당 행 찾기
            mask = df['trade_id'] == trade_id
            if mask.any():
                df.loc[mask, 'result'] = result
                df.loc[mask, 'profit_loss'] = profit_loss
                df.loc[mask, 'exit_time'] = datetime.now().isoformat()
                
                # 저장
                df.to_csv(trade_log_path, index=False)
                print(f"거래 {trade_id} 결과 업데이트 완료")
                return True
        
        return False
    
    def add_new_price_data(self, new_data):
        """
        새로운 가격 데이터 추가
        """
        # 오늘 날짜 파일명
        today = datetime.now().strftime("%Y%m%d")
        filepath = os.path.join(self.config.PRICE_DATA_DIR, 'raw', f'prices_{today}.csv')
        
        # 기존 파일이 있으면 추가, 없으면 생성
        if os.path.exists(filepath):
            existing = pd.read_csv(filepath)
            updated = pd.concat([existing, new_data], ignore_index=True)
            updated = updated.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
        else:
            updated = new_data
        
        # 저장
        updated.to_csv(filepath, index=False)
        print(f"가격 데이터 추가 완료: {len(new_data)} 레코드")
        
        return True
    
    def cleanup_old_data(self, days_to_keep=90):
        """
        오래된 데이터 정리
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # 가격 데이터 정리
        price_files = glob.glob(os.path.join(self.config.PRICE_DATA_DIR, 'raw', '*.csv'))
        for file in price_files:
            # 파일명에서 날짜 추출
            filename = os.path.basename(file)
            if filename.startswith('prices_'):
                date_str = filename.replace('prices_', '').replace('.csv', '')
                try:
                    file_date = datetime.strptime(date_str, "%Y%m%d")
                    if file_date < cutoff_date:
                        os.remove(file)
                        print(f"오래된 파일 삭제: {filename}")
                except:
                    continue
        
        # 거래 로그 정리 (아카이브)
        trade_log_path = os.path.join(self.config.TRADE_LOG_DIR, 'trades.csv')
        if os.path.exists(trade_log_path):
            df = pd.read_csv(trade_log_path)
            if 'entry_time' in df.columns:
                df['entry_time'] = pd.to_datetime(df['entry_time'])
                
                # 오래된 데이터 아카이브
                old_trades = df[df['entry_time'] < cutoff_date]
                if not old_trades.empty:
                    archive_path = os.path.join(self.config.TRADE_LOG_DIR, 
                                               f'trades_archive_{cutoff_date.strftime("%Y%m%d")}.csv')
                    old_trades.to_csv(archive_path, index=False)
                    
                    # 현재 파일에서는 제거
                    df = df[df['entry_time'] >= cutoff_date]
                    df.to_csv(trade_log_path, index=False)
                    print(f"거래 로그 아카이브 완료: {len(old_trades)} 레코드")


class DataValidator:
    """데이터 검증 클래스"""
    
    @staticmethod
    def validate_price_data(df):
        """
        가격 데이터 검증
        """
        issues = []
        
        # 필수 컬럼 확인
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"누락된 컬럼: {missing_columns}")
        
        # OHLC 논리 검증
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_high = df[df['high'] < df[['open', 'close']].max(axis=1)]
            if not invalid_high.empty:
                issues.append(f"잘못된 고가: {len(invalid_high)} 레코드")
            
            invalid_low = df[df['low'] > df[['open', 'close']].min(axis=1)]
            if not invalid_low.empty:
                issues.append(f"잘못된 저가: {len(invalid_low)} 레코드")
        
        # 중복 타임스탬프 확인
        if 'timestamp' in df.columns:
            duplicates = df[df.duplicated(subset=['timestamp'], keep=False)]
            if not duplicates.empty:
                issues.append(f"중복 타임스탬프: {len(duplicates)} 레코드")
        
        # 결측치 확인
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            issues.append(f"결측치: {null_counts[null_counts > 0].to_dict()}")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_trade_logs(df):
        """
        거래 로그 검증
        """
        issues = []
        
        # 필수 컬럼 확인
        required_columns = ['trade_id', 'entry_time', 'prediction']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"누락된 컬럼: {missing_columns}")
        
        # 중복 trade_id 확인
        if 'trade_id' in df.columns:
            duplicates = df[df.duplicated(subset=['trade_id'], keep=False)]
            if not duplicates.empty:
                issues.append(f"중복 trade_id: {len(duplicates)} 레코드")
        
        # prediction 값 검증 (0 또는 1)
        if 'prediction' in df.columns:
            invalid_pred = df[~df['prediction'].isin([0, 1])]
            if not invalid_pred.empty:
                issues.append(f"잘못된 prediction 값: {len(invalid_pred)} 레코드")
        
        return len(issues) == 0, issues


# 사용 예시
if __name__ == "__main__":
    from config import Config
    
    # 데이터 병합기 초기화
    merger = DataMerger(Config)
    
    # 모든 데이터 병합
    merged_data = merger.merge_all_data()
    
    if merged_data is not None:
        # 병합 데이터 저장
        merger.save_merged_data()
        
        # 학습용 데이터 준비
        X, y = merger.get_training_data(lookback_days=30)
        if X is not None:
            print(f"\n학습 데이터 준비 완료:")
            print(f"- 피처 shape: {X.shape}")
            print(f"- 타겟 shape: {y.shape}")
            print(f"- 클래스 분포: {y.value_counts().to_dict()}")
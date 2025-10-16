import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import joblib
from datetime import datetime
import os
import glob
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 RSI + 프라이스 액션 ML 모델 (v2: 장기 추세 필터 & 모델 단순화)")
print("="*80)

# ============================================================
# 설정
# ============================================================
CONFIG = {
    'lookback': 30,         # 30분 룩백
    'option_duration': 10,  # 10분 옵션
    'threshold': 0.65,      # 진입 임계값
    'data_folder': '1m',    # CSV 파일들이 있는 폴더
    'train_months': [1, 2, 3, 4, 5, 6, 7, 8, 9], # 학습: 1~9월
    'test_months': [10],    # 검증: 10월 (있으면)
}

# ============================================================
# 1. CSV 파일 로드 (기존과 동일)
# ============================================================
print("\n📂 CSV 파일 검색 중...")
csv_folder = CONFIG['data_folder']
csv_files = sorted(glob.glob(f"{csv_folder}/*.csv"))
if not csv_files:
    print(f"❌ '{csv_folder}' 폴더에서 CSV 파일을 찾을 수 없습니다!")
    exit(1)
print(f"✅ {len(csv_files)}개 CSV 파일 발견.")

print(f"\n📊 데이터 로딩 중...")
dfs = []
for csv_file in csv_files:
    try:
        df_temp = pd.read_csv(csv_file, header=None)
        df_temp.columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ]
        df_temp['timestamp'] = pd.to_datetime(df_temp['open_time'], unit='us') # ### 참고: 바이낸스 데이터는 보통 ms 단위 ###
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df_temp = df_temp[required_cols].copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
        dfs.append(df_temp)
    except Exception as e:
        print(f"   ❌ {os.path.basename(csv_file)}: 오류 - {e}")

if not dfs:
    print("❌ 로드된 데이터가 없습니다!")
    exit(1)

df = pd.concat(dfs, ignore_index=True)
df = df.sort_values('timestamp').reset_index(drop=True)
df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)

print(f"\n✅ 총 {len(df):,}개 1분봉 데이터 로드")
print(f"   기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

# ============================================================
# 2. 15분봉 집계 (기존과 동일)
# ============================================================
print("\n📈 15분봉 집계 중...")
df.set_index('timestamp', inplace=True)
df_15m = df.resample('15T').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
}).dropna()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df_15m['rsi_14'] = calculate_rsi(df_15m['close'], period=14)
df_15m['rsi_7'] = calculate_rsi(df_15m['close'], period=7)
df_15m['rsi_21'] = calculate_rsi(df_15m['close'], period=21)
df_15m['is_bullish_15m'] = (df_15m['close'] > df_15m['open']).astype(int)
df_15m['is_bearish_15m'] = (df_15m['close'] < df_15m['open']).astype(int)
df_15m['ema_20_15m'] = df_15m['close'].ewm(span=20, adjust=False).mean()
df_15m['ema_50_15m'] = df_15m['close'].ewm(span=50, adjust=False).mean()

df = df.reset_index()
df_15m_reindex = df_15m.reindex(pd.to_datetime(df['timestamp']), method='ffill').reset_index()

for col in ['rsi_14', 'rsi_7', 'rsi_21', 'is_bullish_15m', 'is_bearish_15m', 'ema_20_15m', 'ema_50_15m']:
    if col in df_15m_reindex.columns:
        df[col] = df_15m_reindex[col].values
print("✅ 15분봉 피처를 1분봉에 병합")

# ============================================================
# 2.5. 장기 추세 필터 (1시간봉) 생성 ### 추가된 부분 ###
# ============================================================
print("\n⏳ 장기 추세 필터 (1시간봉) 생성 중...")
df.set_index('timestamp', inplace=True)

df_1h = df.resample('1H').agg({
    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'
}).dropna()

# 1시간봉 기준 50 EMA 계산
df_1h['ema_50_1h'] = df_1h['close'].ewm(span=50, adjust=False).mean()

# 1분봉으로 다시 머지 (forward fill)
df_1h_reindex = df_1h.reindex(df.index, method='ffill')

df['ema_50_1h'] = df_1h_reindex['ema_50_1h']
# 현재 1분봉 종가가 1시간봉 EMA 위에 있으면 상승추세(1), 아니면 하락추세(0)
df['is_uptrend_1h'] = (df['close'] > df['ema_50_1h']).astype(int)

df.reset_index(inplace=True)
print("✅ 1시간봉 EMA 필터 병합 완료")

# ============================================================
# 3. 1분봉 지표 계산 (기존과 동일)
# ============================================================
print("\n📊 1분봉 지표 계산 중...")
df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
df['body'] = df['close'] - df['open']
df['body_abs'] = abs(df['body'])
df['range'] = df['high'] - df['low']
df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
df['body_ratio'] = df['body_abs'] / (df['range'] + 1e-8)
df['upper_shadow_ratio'] = df['upper_shadow'] / (df['range'] + 1e-8)
df['lower_shadow_ratio'] = df['lower_shadow'] / (df['range'] + 1e-8)
df['is_bullish'] = (df['close'] > df['open']).astype(int)
df['is_bearish'] = (df['close'] < df['open']).astype(int)
df['is_doji'] = (df['body_ratio'] < 0.1).astype(int)
df['is_hammer'] = ((df['lower_shadow_ratio'] > 0.6) & (df['upper_shadow_ratio'] < 0.15) & (df['body_ratio'] < 0.3)).astype(int)
df['is_shooting_star'] = ((df['upper_shadow_ratio'] > 0.6) & (df['lower_shadow_ratio'] < 0.15) & (df['body_ratio'] < 0.3)).astype(int)
df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
df['volume_ratio'] = df['volume'] / (df['volume_ma_10'] + 1e-8)
df['volume_surge'] = (df['volume'] > df['volume_ma_10'] * 1.5).astype(int)
df['recent_high_20'] = df['high'].rolling(window=20).max()
df['recent_low_20'] = df['low'].rolling(window=20).min()
df['distance_to_high'] = (df['recent_high_20'] - df['close']) / df['close']
df['distance_to_low'] = (df['close'] - df['recent_low_20']) / df['close']
df['trend_ema'] = (df['ema_20'] > df['ema_50']).astype(int)
df['trend_15m'] = (df['ema_20_15m'] > df['ema_50_15m']).astype(int)
print("✅ 1분봉 지표 계산 완료")

# ============================================================
# 4. RSI 상태 피처 생성 (기존과 동일)
# ============================================================
print("\n🎯 RSI 상태 피처 생성 중...")
df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
df['rsi_extreme_oversold'] = (df['rsi_14'] < 20).astype(int)
df['rsi_extreme_overbought'] = (df['rsi_14'] > 80).astype(int)
df['rsi_neutral'] = ((df['rsi_14'] >= 40) & (df['rsi_14'] <= 60)).astype(int)
df['rsi_low_with_hammer'] = (df['rsi_oversold'] & df['is_hammer']).astype(int)
df['rsi_low_with_bullish'] = (df['rsi_oversold'] & df['is_bullish']).astype(int)
df['rsi_high_with_shooting'] = (df['rsi_overbought'] & df['is_shooting_star']).astype(int)
df['rsi_high_with_bearish'] = (df['rsi_overbought'] & df['is_bearish']).astype(int)
df['rsi_low_uptrend'] = (df['rsi_oversold'] & df['trend_15m']).astype(int)
df['rsi_high_downtrend'] = (df['rsi_overbought'] & (df['trend_15m'] == 0)).astype(int)
df['rsi_increasing'] = (df['rsi_14'].diff() > 0).astype(int)
df['rsi_decreasing'] = (df['rsi_14'].diff() < 0).astype(int)
df['after_15m_bullish'] = df['is_bullish_15m'].shift(1).fillna(0).astype(int)
df['after_15m_bearish'] = df['is_bearish_15m'].shift(1).fillna(0).astype(int)
df['rsi_low_after_15m_bull'] = (df['rsi_oversold'] & df['after_15m_bullish']).astype(int)
df['rsi_high_after_15m_bear'] = (df['rsi_overbought'] & df['after_15m_bearish']).astype(int)
print("✅ RSI 상태 피처 생성 완료")

# ============================================================
# 5. 타겟 생성 (기존과 동일)
# ============================================================
print("\n🎯 타겟 생성 중...")
df['future_price'] = df['close'].shift(-CONFIG['option_duration'])
df['target_long'] = (df['future_price'] > df['close']).astype(int)
df['target_short'] = (df['future_price'] < df['close']).astype(int)
print("✅ 타겟 생성 완료")

# ============================================================
# 6. 룩백 윈도우 피처 생성
# ============================================================
print("\n🔄 룩백 윈도우 피처 생성 중...")

def create_lookback_features(df, lookback=30):
    features_list = []
    for i in range(lookback, len(df)):
        window = df.iloc[i-lookback:i]
        feature_dict = {'index': i, 'timestamp': df.loc[i, 'timestamp']}
        
        # ... (기존 피처들) ...
        feature_dict['current_rsi_14'] = df.loc[i, 'rsi_14']
        feature_dict['rsi_oversold'] = df.loc[i, 'rsi_oversold']
        feature_dict['rsi_overbought'] = df.loc[i, 'rsi_overbought']
        feature_dict['rsi_low_with_hammer'] = df.loc[i, 'rsi_low_with_hammer']
        feature_dict['rsi_high_with_shooting'] = df.loc[i, 'rsi_high_with_shooting']
        feature_dict['is_hammer'] = df.loc[i, 'is_hammer']
        feature_dict['is_shooting_star'] = df.loc[i, 'is_shooting_star']
        feature_dict['volume_surge'] = df.loc[i, 'volume_surge']
        feature_dict['trend_ema'] = df.loc[i, 'trend_ema']
        feature_dict['trend_15m'] = df.loc[i, 'trend_15m']
        
        ### 추가됨: 장기 추세 피처를 모델이 학습하도록 추가 ###
        feature_dict['is_uptrend_1h'] = df.loc[i, 'is_uptrend_1h']
        
        feature_dict['distance_to_high'] = df.loc[i, 'distance_to_high']
        feature_dict['distance_to_low'] = df.loc[i, 'distance_to_low']
        
        returns = (window['close'] / window['open'] - 1).values
        for j in range(min(15, lookback)):
            feature_dict[f'ret_{j}'] = returns[-(j+1)]
            
        feature_dict['target_long'] = df.loc[i, 'target_long']
        feature_dict['target_short'] = df.loc[i, 'target_short']
        
        features_list.append(feature_dict)
        if (i - lookback) % 20000 == 0:
            print(f"   진행: {i - lookback:,} / {len(df) - lookback:,}")
    return pd.DataFrame(features_list)

features_df = create_lookback_features(df, CONFIG['lookback'])
print(f"✅ {len(features_df):,}개 샘플 생성")
features_df = features_df.dropna()
print(f"   NaN 제거 후: {len(features_df):,}개")

# ============================================================
# 7. 학습/검증 데이터 분리 (기존과 동일)
# ============================================================
print("\n✂️  데이터 분리 중...")
features_df['month'] = pd.to_datetime(features_df['timestamp']).dt.month
train_data = features_df[features_df['month'].isin(CONFIG['train_months'])].copy()
test_data = features_df[features_df['month'].isin(CONFIG['test_months'])].copy()
print(f"✅ 학습 데이터: {len(train_data):,}개 (월: {CONFIG['train_months']})")
print(f"✅ 검증 데이터: {len(test_data):,}개 (월: {CONFIG['test_months']})")

# ============================================================
# 8. 피처 컬럼 정의 (기존과 동일)
# ============================================================
exclude_cols = ['index', 'timestamp', 'month', 'target_long', 'target_short']
feature_columns = [col for col in features_df.columns if col not in exclude_cols]
print(f"\n📋 사용 피처: {len(feature_columns)}개")

# ============================================================
# 9. 모델 학습 - LONG (기존과 동일)
# ============================================================
print("\n" + "="*80)
print("📈 LONG 모델 학습")
print("="*80)
X_train_long = train_data[feature_columns]
y_train_long = train_data['target_long']

model_long = LGBMClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=3, num_leaves=7,
    min_child_samples=200, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1
)
print("학습 중...")
model_long.fit(X_train_long, y_train_long)
print("✅ 학습 완료")

# ============================================================
# 10. 모델 학습 - SHORT ### 변경됨: 모델 파라미터 단순화 ###
# ============================================================
print("\n" + "="*80)
print("📉 SHORT 모델 학습")
print("="*80)
X_train_short = train_data[feature_columns]
y_train_short = train_data['target_short']

# LONG 모델과 동일한 파라미터로 단순화하여 과적합 방지
model_short = LGBMClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=3, num_leaves=7,
    min_child_samples=200, subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1
)
print("학습 중...")
model_short.fit(X_train_short, y_train_short)
print("✅ 학습 완료")

# ============================================================
# 11. 검증 ### 변경됨: 장기 추세 필터 적용 ###
# ============================================================
if len(test_data) > 0:
    print("\n" + "="*80)
    print("🔍 검증 데이터 평가 (장기 추세 필터 적용)")
    print("="*80)
    
    X_test = test_data[feature_columns]
    y_test_long = test_data['target_long']
    y_test_short = test_data['target_short']
    
    # --- LONG 평가 ---
    test_long_prob = model_long.predict_proba(X_test)[:, 1]
    test_long_pred = (test_long_prob > CONFIG['threshold']).astype(int)
    
    # --- SHORT 평가 ---
    test_short_prob = model_short.predict_proba(X_test)[:, 1]
    test_short_pred = (test_short_prob > CONFIG['threshold']).astype(int)
    
    # --- 장기 추세 필터 적용 ---
    # is_uptrend_1h 피처를 test_data에서 가져옴
    trend_filter = test_data['is_uptrend_1h'].values

    # Long 신호는 1시간봉 상승 추세(1)일 때만 유효
    filtered_long_pred = test_long_pred & (trend_filter == 1)
    
    # Short 신호는 1시간봉 하락 추세(0)일 때만 유효
    filtered_short_pred = test_short_pred & (trend_filter == 0)
    
    # 필터링된 예측 기반으로 진입점 결정
    long_entries = test_data[filtered_long_pred == 1]
    short_entries = test_data[filtered_short_pred == 1]
    
    # 승률 계산
    long_wins = (long_entries['target_long'] == 1).sum()
    long_total = len(long_entries)
    long_winrate = (long_wins / long_total * 100) if long_total > 0 else 0
    
    short_wins = (short_entries['target_short'] == 1).sum()
    short_total = len(short_entries)
    short_winrate = (short_wins / short_total * 100) if short_total > 0 else 0
    
    total_trades = long_total + short_total
    total_wins = long_wins + short_wins
    total_winrate = (total_wins / total_trades) * 100 if total_trades > 0 else 0
    
    print(f"\nLONG (상승 추세 필터 적용):")
    print(f"  원래 신호: {test_long_pred.sum()}개")
    print(f"  필터 후 거래: {long_total}개")
    print(f"  승률: {long_winrate:.2f}%")
    
    print(f"\nSHORT (하락 추세 필터 적용):")
    print(f"  원래 신호: {test_short_pred.sum()}개")
    print(f"  필터 후 거래: {short_total}개")
    print(f"  승률: {short_winrate:.2f}%")
    
    print(f"\n통합:")
    print(f"  총 거래: {total_trades}개")
    print(f"  총 승률: {total_winrate:.2f}%")

# ============================================================
# 12. 모델 저장 (기존과 동일)
# ============================================================
print("\n" + "="*80)
print("💾 모델 저장")
print("="*80)
os.makedirs('model', exist_ok=True)
model_package = {
    'long_model': model_long, 'short_model': model_short,
    'feature_columns': feature_columns, 'config': CONFIG,
    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}
filename = f"model/rsi_price_action_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
joblib.dump(model_package, filename)
print(f"✅ 모델 저장: {filename}")
print("\n✅ 전체 파이프라인 완료!")
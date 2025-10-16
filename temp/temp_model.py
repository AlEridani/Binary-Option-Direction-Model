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
print("🚀 RSI + 프라이스 액션 ML 모델 (다수 CSV 처리)")
print("="*80)

# ============================================================
# 설정
# ============================================================
CONFIG = {
    'lookback': 30,           # 30분 룩백
    'option_duration': 10,    # 10분 옵션
    'threshold': 0.55,        # 진입 임계값 (0.70 → 0.55로 낮춤)
    'data_folder': '1m',      # CSV 파일들이 있는 폴더
    'train_months': [1, 2, 3, 4, 5, 6, 7, 8, 9],  # 학습: 1~9월
    'test_months': [10],      # 검증: 10월 (있으면)
}

# ============================================================
# 1. CSV 파일 로드
# ============================================================
print("\n📂 CSV 파일 검색 중...")

csv_folder = CONFIG['data_folder']
csv_files = sorted(glob.glob(f"{csv_folder}/*.csv"))

if len(csv_files) == 0:
    print(f"❌ '{csv_folder}' 폴더에서 CSV 파일을 찾을 수 없습니다!")
    print(f"   폴더 구조 확인: {csv_folder}/")
    exit(1)

print(f"✅ {len(csv_files)}개 CSV 파일 발견:")
for f in csv_files:
    print(f"   - {os.path.basename(f)}")

# 모든 CSV 파일 병합
print(f"\n📊 데이터 로딩 중...")
dfs = []

for csv_file in csv_files:
    try:
        # 바이낸스 원시 데이터 형식 (컬럼명 없음)
        # 형식: open_time, open, high, low, close, volume, close_time, quote_volume, trades, taker_buy_volume, taker_buy_quote_volume, ignore
        df_temp = pd.read_csv(csv_file, header=None)
        
        # 컬럼명 수동 할당
        df_temp.columns = [
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_volume',
            'taker_buy_quote_volume', 'ignore'
        ]
        
        # timestamp를 open_time에서 변환 (나노초 단위)
        df_temp['timestamp'] = pd.to_datetime(df_temp['open_time'], unit='ns')
        
        # 필요한 컬럼만 선택
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df_temp = df_temp[required_cols].copy()
        
        # 데이터 타입 변환
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
        
        dfs.append(df_temp)
        print(f"   ✅ {os.path.basename(csv_file)}: {len(df_temp):,}개 행")
        
    except Exception as e:
        print(f"   ❌ {os.path.basename(csv_file)}: 오류 - {e}")

if len(dfs) == 0:
    print("❌ 로드된 데이터가 없습니다!")
    exit(1)

# 데이터 병합
df = pd.concat(dfs, ignore_index=True)
print(f"\n✅ 총 {len(df):,}개 1분봉 데이터 로드")

# 타임스탬프 정규화
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

# 중복 제거
original_len = len(df)
df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
if len(df) < original_len:
    print(f"⚠️  중복 제거: {original_len - len(df):,}개 행 삭제")

print(f"   기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
print(f"   최종: {len(df):,}개")

# ============================================================
# 2. 15분봉 집계 (RSI 계산용)
# ============================================================
print("\n📈 15분봉 집계 중...")

df.set_index('timestamp', inplace=True)
df_15m = df.resample('15T').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

print(f"✅ {len(df_15m):,}개 15분봉 생성")

# 15분봉 RSI 계산
def calculate_rsi(series, period=14):
    """RSI 계산"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df_15m['rsi_14'] = calculate_rsi(df_15m['close'], period=14)
df_15m['rsi_7'] = calculate_rsi(df_15m['close'], period=7)
df_15m['rsi_21'] = calculate_rsi(df_15m['close'], period=21)

# 15분봉 캔들 패턴
df_15m['body_15m'] = df_15m['close'] - df_15m['open']
df_15m['range_15m'] = df_15m['high'] - df_15m['low']
df_15m['is_bullish_15m'] = (df_15m['close'] > df_15m['open']).astype(int)
df_15m['is_bearish_15m'] = (df_15m['close'] < df_15m['open']).astype(int)

# 15분봉 이동평균
df_15m['ema_20_15m'] = df_15m['close'].ewm(span=20, adjust=False).mean()
df_15m['ema_50_15m'] = df_15m['close'].ewm(span=50, adjust=False).mean()

print("✅ 15분봉 지표 계산 완료")

# 1분봉으로 다시 머지 (forward fill)
df = df.reset_index()
df_15m_reindex = df_15m.reindex(df['timestamp'], method='ffill')
df_15m_reindex = df_15m_reindex.reset_index()

# 15분봉 피처를 1분봉에 추가
for col in ['rsi_14', 'rsi_7', 'rsi_21', 'is_bullish_15m', 'is_bearish_15m', 
            'ema_20_15m', 'ema_50_15m']:
    if col in df_15m_reindex.columns:
        df[col] = df_15m_reindex[col].values

print("✅ 15분봉 피처를 1분봉에 병합")

# ============================================================
# 3. 1분봉 지표 계산
# ============================================================
print("\n📊 1분봉 지표 계산 중...")

# 이동평균선
df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

# 캔들 정보
df['body'] = df['close'] - df['open']
df['body_abs'] = abs(df['body'])
df['range'] = df['high'] - df['low']
df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']

# 비율
df['body_ratio'] = df['body_abs'] / (df['range'] + 1e-8)
df['upper_shadow_ratio'] = df['upper_shadow'] / (df['range'] + 1e-8)
df['lower_shadow_ratio'] = df['lower_shadow'] / (df['range'] + 1e-8)

# 캔들 패턴
df['is_bullish'] = (df['close'] > df['open']).astype(int)
df['is_bearish'] = (df['close'] < df['open']).astype(int)
df['is_doji'] = (df['body_ratio'] < 0.1).astype(int)

# 해머 패턴
df['is_hammer'] = (
    (df['lower_shadow_ratio'] > 0.6) & 
    (df['upper_shadow_ratio'] < 0.15) & 
    (df['body_ratio'] < 0.3)
).astype(int)

# 슈팅스타
df['is_shooting_star'] = (
    (df['upper_shadow_ratio'] > 0.6) & 
    (df['lower_shadow_ratio'] < 0.15) & 
    (df['body_ratio'] < 0.3)
).astype(int)

# 거래량
df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
df['volume_ratio'] = df['volume'] / (df['volume_ma_10'] + 1e-8)
df['volume_surge'] = (df['volume'] > df['volume_ma_10'] * 1.5).astype(int)

# 지지/저항
df['recent_high_20'] = df['high'].rolling(window=20).max()
df['recent_low_20'] = df['low'].rolling(window=20).min()
df['distance_to_high'] = (df['recent_high_20'] - df['close']) / df['close']
df['distance_to_low'] = (df['close'] - df['recent_low_20']) / df['close']

# 추세
df['trend_ema'] = (df['ema_20'] > df['ema_50']).astype(int)
df['trend_15m'] = (df['ema_20_15m'] > df['ema_50_15m']).astype(int)

print("✅ 1분봉 지표 계산 완료")

# ============================================================
# 4. RSI 상태 피처 (ML이 학습할 핵심)
# ============================================================
print("\n🎯 RSI 상태 피처 생성 중...")

# RSI 레벨
df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
df['rsi_extreme_oversold'] = (df['rsi_14'] < 20).astype(int)
df['rsi_extreme_overbought'] = (df['rsi_14'] > 80).astype(int)
df['rsi_neutral'] = ((df['rsi_14'] >= 40) & (df['rsi_14'] <= 60)).astype(int)

# RSI + 캔들 조합 (ML이 학습할 패턴)
df['rsi_low_with_hammer'] = (df['rsi_oversold'] & df['is_hammer']).astype(int)
df['rsi_low_with_bullish'] = (df['rsi_oversold'] & df['is_bullish']).astype(int)
df['rsi_high_with_shooting'] = (df['rsi_overbought'] & df['is_shooting_star']).astype(int)
df['rsi_high_with_bearish'] = (df['rsi_overbought'] & df['is_bearish']).astype(int)

# RSI + 추세 조합
df['rsi_low_uptrend'] = (df['rsi_oversold'] & df['trend_15m']).astype(int)
df['rsi_high_downtrend'] = (df['rsi_overbought'] & (df['trend_15m'] == 0)).astype(int)

# RSI 변화
df['rsi_increasing'] = (df['rsi_14'].diff() > 0).astype(int)
df['rsi_decreasing'] = (df['rsi_14'].diff() < 0).astype(int)

# 15분봉 양봉/음봉 직후
df['after_15m_bullish'] = df['is_bullish_15m'].shift(1).fillna(0).astype(int)
df['after_15m_bearish'] = df['is_bearish_15m'].shift(1).fillna(0).astype(int)

# RSI + 15분봉 양봉 조합 (당신이 원하는 패턴!)
df['rsi_low_after_15m_bull'] = (df['rsi_oversold'] & df['after_15m_bullish']).astype(int)
df['rsi_high_after_15m_bear'] = (df['rsi_overbought'] & df['after_15m_bearish']).astype(int)

print("✅ RSI 상태 피처 생성 완료")

# ============================================================
# 5. 타겟 생성
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
    """과거 N분 데이터로 피처 생성"""
    
    features_list = []
    
    for i in range(lookback, len(df)):
        window = df.iloc[i-lookback:i]
        
        feature_dict = {
            'index': i,
            'timestamp': df.loc[i, 'timestamp'],
        }
        
        # 현재 시점 RSI 상태 (가장 중요!)
        feature_dict['current_rsi_14'] = df.loc[i, 'rsi_14']
        feature_dict['current_rsi_7'] = df.loc[i, 'rsi_7']
        feature_dict['current_rsi_21'] = df.loc[i, 'rsi_21']
        feature_dict['rsi_oversold'] = df.loc[i, 'rsi_oversold']
        feature_dict['rsi_overbought'] = df.loc[i, 'rsi_overbought']
        feature_dict['rsi_extreme_oversold'] = df.loc[i, 'rsi_extreme_oversold']
        feature_dict['rsi_extreme_overbought'] = df.loc[i, 'rsi_extreme_overbought']
        
        # RSI + 캔들 조합 패턴
        feature_dict['rsi_low_with_hammer'] = df.loc[i, 'rsi_low_with_hammer']
        feature_dict['rsi_low_with_bullish'] = df.loc[i, 'rsi_low_with_bullish']
        feature_dict['rsi_high_with_shooting'] = df.loc[i, 'rsi_high_with_shooting']
        feature_dict['rsi_high_with_bearish'] = df.loc[i, 'rsi_high_with_bearish']
        
        # RSI + 추세
        feature_dict['rsi_low_uptrend'] = df.loc[i, 'rsi_low_uptrend']
        feature_dict['rsi_high_downtrend'] = df.loc[i, 'rsi_high_downtrend']
        
        # RSI + 15분봉 양봉 (핵심!)
        feature_dict['rsi_low_after_15m_bull'] = df.loc[i, 'rsi_low_after_15m_bull']
        feature_dict['rsi_high_after_15m_bear'] = df.loc[i, 'rsi_high_after_15m_bear']
        feature_dict['after_15m_bullish'] = df.loc[i, 'after_15m_bullish']
        
        # 현재 캔들 패턴
        feature_dict['is_hammer'] = df.loc[i, 'is_hammer']
        feature_dict['is_shooting_star'] = df.loc[i, 'is_shooting_star']
        feature_dict['is_doji'] = df.loc[i, 'is_doji']
        feature_dict['volume_surge'] = df.loc[i, 'volume_surge']
        
        # 추세
        feature_dict['trend_ema'] = df.loc[i, 'trend_ema']
        feature_dict['trend_15m'] = df.loc[i, 'trend_15m']
        
        # 지지/저항 근처
        feature_dict['distance_to_high'] = df.loc[i, 'distance_to_high']
        feature_dict['distance_to_low'] = df.loc[i, 'distance_to_low']
        
        # 과거 15개 캔들의 수익률, 몸통, 레인지
        returns = (window['close'] / window['open'] - 1).values
        bodies = window['body'].values / window['open'].values
        ranges = window['range'].values / window['open'].values
        
        for j in range(min(15, lookback)):
            feature_dict[f'ret_{j}'] = returns[-(j+1)]
            feature_dict[f'body_{j}'] = bodies[-(j+1)]
            feature_dict[f'range_{j}'] = ranges[-(j+1)]
        
        # 통계
        feature_dict['ret_mean_5'] = returns[-5:].mean()
        feature_dict['ret_std_5'] = returns[-5:].std()
        feature_dict['ret_mean_10'] = returns[-10:].mean()
        feature_dict['ret_std_10'] = returns[-10:].std()
        
        # 거래량
        vol_changes = window['volume'].pct_change().fillna(0).values
        for j in range(min(10, lookback)):
            feature_dict[f'vol_chg_{j}'] = vol_changes[-(j+1)]
        
        # 최근 패턴 빈도
        feature_dict['hammer_count_5'] = window['is_hammer'].iloc[-5:].sum()
        feature_dict['bullish_count_5'] = window['is_bullish'].iloc[-5:].sum()
        
        # 타겟
        feature_dict['target_long'] = df.loc[i, 'target_long']
        feature_dict['target_short'] = df.loc[i, 'target_short']
        
        features_list.append(feature_dict)
        
        if (i - lookback) % 10000 == 0:
            print(f"   진행: {i - lookback:,} / {len(df) - lookback:,}")
    
    return pd.DataFrame(features_list)

features_df = create_lookback_features(df, CONFIG['lookback'])
print(f"✅ {len(features_df):,}개 샘플 생성")

# NaN 제거
features_df = features_df.dropna()
print(f"   NaN 제거 후: {len(features_df):,}개")

# ============================================================
# 7. 학습/검증 데이터 분리
# ============================================================
print("\n✂️  데이터 분리 중...")

features_df['month'] = pd.to_datetime(features_df['timestamp']).dt.month

train_data = features_df[features_df['month'].isin(CONFIG['train_months'])].copy()
test_data = features_df[features_df['month'].isin(CONFIG['test_months'])].copy()

print(f"✅ 학습 데이터: {len(train_data):,}개 (월: {CONFIG['train_months']})")
print(f"✅ 검증 데이터: {len(test_data):,}개 (월: {CONFIG['test_months']})")

if len(test_data) == 0:
    print("⚠️  10월 데이터가 없습니다. 학습만 진행합니다.")

# ============================================================
# 8. 피처 컬럼 정의
# ============================================================
exclude_cols = ['index', 'timestamp', 'month', 'target_long', 'target_short']
feature_columns = [col for col in features_df.columns if col not in exclude_cols]

print(f"\n📋 사용 피처: {len(feature_columns)}개")
print("   주요 RSI 피처:")
rsi_features = [col for col in feature_columns if 'rsi' in col.lower()]
for col in rsi_features[:15]:
    print(f"   - {col}")
if len(rsi_features) > 15:
    print(f"   ... 외 {len(rsi_features) - 15}개")

# ============================================================
# 9. 모델 학습 - LONG
# ============================================================
print("\n" + "="*80)
print("📈 LONG 모델 학습")
print("="*80)

X_train_long = train_data[feature_columns]
y_train_long = train_data['target_long']

model_long = LGBMClassifier(
    n_estimators=200,        # 500 → 200 (과적합 방지)
    learning_rate=0.1,       # 0.03 → 0.1 (빠른 학습)
    max_depth=3,             # 6 → 3 (트리 깊이 제한, 핵심!)
    num_leaves=7,            # 31 → 7 (복잡도 감소, 핵심!)
    min_child_samples=200,   # 50 → 200 (더 많은 샘플 필요)
    subsample=0.7,           # 0.8 → 0.7 (배깅 강화)
    colsample_bytree=0.7,    # 0.8 → 0.7 (피처 샘플링 강화)
    reg_alpha=1.0,           # 0.5 → 1.0 (L1 정규화 강화)
    reg_lambda=1.0,          # 0.5 → 1.0 (L2 정규화 강화)
    random_state=42,
    verbose=-1
)

print("학습 중...")
model_long.fit(X_train_long, y_train_long)

# 학습 평가
train_long_prob = model_long.predict_proba(X_train_long)[:, 1]
train_long_pred = (train_long_prob > CONFIG['threshold']).astype(int)
train_long_signals = train_long_pred.sum()
train_long_wins = ((train_long_pred == 1) & (y_train_long == 1)).sum()
train_long_winrate = (train_long_wins / train_long_signals * 100) if train_long_signals > 0 else 0

print(f"✅ 학습 완료")
print(f"   신호: {train_long_signals:,}개 ({train_long_signals/len(train_data)*100:.1f}%)")
print(f"   승률: {train_long_winrate:.2f}%")

# 확률 분포 분석
print(f"\n📊 확률 분포 분석:")
print(f"   50% 이상: {(train_long_prob > 0.50).sum():,}개 ({(train_long_prob > 0.50).sum()/len(train_data)*100:.1f}%)")
print(f"   55% 이상: {(train_long_prob > 0.55).sum():,}개 ({(train_long_prob > 0.55).sum()/len(train_data)*100:.1f}%)")
print(f"   60% 이상: {(train_long_prob > 0.60).sum():,}개 ({(train_long_prob > 0.60).sum()/len(train_data)*100:.1f}%)")
print(f"   65% 이상: {(train_long_prob > 0.65).sum():,}개 ({(train_long_prob > 0.65).sum()/len(train_data)*100:.1f}%)")
print(f"   70% 이상: {(train_long_prob > 0.70).sum():,}개 ({(train_long_prob > 0.70).sum()/len(train_data)*100:.1f}%)")

# 임계값별 승률
for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
    pred = (train_long_prob > thresh).astype(int)
    if pred.sum() > 0:
        wins = ((pred == 1) & (y_train_long == 1)).sum()
        wr = (wins / pred.sum() * 100)
        print(f"   임계값 {thresh:.2f}: 신호 {pred.sum():,}개, 승률 {wr:.2f}%")

# 피처 중요도
feature_importance_long = pd.DataFrame({
    'feature': feature_columns,
    'importance': model_long.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n📊 LONG 모델 주요 피처 TOP 15:")
for idx, row in feature_importance_long.head(15).iterrows():
    print(f"   {row['feature']}: {row['importance']:.1f}")

# RSI 피처의 중요도 확인
rsi_importance = feature_importance_long[feature_importance_long['feature'].str.contains('rsi', case=False)]
print(f"\n🎯 RSI 관련 피처 중요도:")
for idx, row in rsi_importance.head(10).iterrows():
    print(f"   {row['feature']}: {row['importance']:.1f}")

# ============================================================
# 10. 모델 학습 - SHORT
# ============================================================
print("\n" + "="*80)
print("📉 SHORT 모델 학습")
print("="*80)

X_train_short = train_data[feature_columns]
y_train_short = train_data['target_short']

model_short = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=31,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=0.5,
    random_state=42,
    verbose=-1
)

print("학습 중...")
model_short.fit(X_train_short, y_train_short)

# 학습 평가
train_short_prob = model_short.predict_proba(X_train_short)[:, 1]
train_short_pred = (train_short_prob > CONFIG['threshold']).astype(int)
train_short_signals = train_short_pred.sum()
train_short_wins = ((train_short_pred == 1) & (y_train_short == 1)).sum()
train_short_winrate = (train_short_wins / train_short_signals * 100) if train_short_signals > 0 else 0

print(f"✅ 학습 완료")
print(f"   신호: {train_short_signals:,}개 ({train_short_signals/len(train_data)*100:.1f}%)")
print(f"   승률: {train_short_winrate:.2f}%")

# 확률 분포 분석
print(f"\n📊 확률 분포 분석:")
print(f"   50% 이상: {(train_short_prob > 0.50).sum():,}개 ({(train_short_prob > 0.50).sum()/len(train_data)*100:.1f}%)")
print(f"   55% 이상: {(train_short_prob > 0.55).sum():,}개 ({(train_short_prob > 0.55).sum()/len(train_data)*100:.1f}%)")
print(f"   60% 이상: {(train_short_prob > 0.60).sum():,}개 ({(train_short_prob > 0.60).sum()/len(train_data)*100:.1f}%)")
print(f"   65% 이상: {(train_short_prob > 0.65).sum():,}개 ({(train_short_prob > 0.65).sum()/len(train_data)*100:.1f}%)")
print(f"   70% 이상: {(train_short_prob > 0.70).sum():,}개 ({(train_short_prob > 0.70).sum()/len(train_data)*100:.1f}%)")

# 임계값별 승률
for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
    pred = (train_short_prob > thresh).astype(int)
    if pred.sum() > 0:
        wins = ((pred == 1) & (y_train_short == 1)).sum()
        wr = (wins / pred.sum() * 100)
        print(f"   임계값 {thresh:.2f}: 신호 {pred.sum():,}개, 승률 {wr:.2f}%")

# 피처 중요도
feature_importance_short = pd.DataFrame({
    'feature': feature_columns,
    'importance': model_short.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n📊 SHORT 모델 주요 피처 TOP 15:")
for idx, row in feature_importance_short.head(15).iterrows():
    print(f"   {row['feature']}: {row['importance']:.1f}")

# ============================================================
# 11. 검증 (10월 데이터 있으면)
# ============================================================
if len(test_data) > 0:
    print("\n" + "="*80)
    print("🔍 검증 데이터 평가")
    print("="*80)
    
    X_test = test_data[feature_columns]
    y_test_long = test_data['target_long']
    y_test_short = test_data['target_short']
    
    # LONG 평가
    test_long_prob = model_long.predict_proba(X_test)[:, 1]
    test_long_pred = (test_long_prob > CONFIG['threshold']).astype(int)
    
    long_entries = test_data[test_long_pred == 1]
    long_wins = (long_entries['target_long'] == 1).sum()
    long_total = len(long_entries)
    long_winrate = (long_wins / long_total * 100) if long_total > 0 else 0
    
    # SHORT 평가
    test_short_prob = model_short.predict_proba(X_test)[:, 1]
    test_short_pred = (test_short_prob > CONFIG['threshold']).astype(int)
    
    short_entries = test_data[test_short_pred == 1]
    short_wins = (short_entries['target_short'] == 1).sum()
    short_total = len(short_entries)
    short_winrate = (short_wins / short_total * 100) if short_total > 0 else 0
    
    # 통합
    total_trades = long_total + short_total
    total_wins = long_wins + short_wins
    total_winrate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    
    print(f"\nLONG:")
    print(f"  거래: {long_total}개")
    print(f"  승률: {long_winrate:.2f}%")
    
    print(f"\nSHORT:")
    print(f"  거래: {short_total}개")
    print(f"  승률: {short_winrate:.2f}%")
    
    print(f"\n통합:")
    print(f"  총 거래: {total_trades}개")
    print(f"  총 승률: {total_winrate:.2f}%")
    print(f"  손익분기: 55.56%")
    
    if total_winrate >= 55.56:
        print(f"  ✅ 수익 가능!")
    else:
        print(f"  ❌ 손익분기 미달")

# ============================================================
# 12. 모델 저장
# ============================================================
print("\n" + "="*80)
print("💾 모델 저장")
print("="*80)

os.makedirs('model', exist_ok=True)

model_package = {
    'long_model': model_long,
    'short_model': model_short,
    'feature_columns': feature_columns,
    'config': CONFIG,
    'feature_importance_long': feature_importance_long,
    'feature_importance_short': feature_importance_short,
    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

filename = f"model/rsi_price_action_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
joblib.dump(model_package, filename)

print(f"✅ 모델 저장: {filename}")

# ============================================================
# 13. 학습 요약
# ============================================================
print("\n" + "="*80)
print("📊 학습 요약")
print("="*80)

print(f"\n데이터:")
print(f"  학습 샘플: {len(train_data):,}개")
print(f"  검증 샘플: {len(test_data):,}개")
print(f"  피처 개수: {len(feature_columns)}개")

print(f"\n학습 성과:")
print(f"  LONG 신호 (임계값 {CONFIG['threshold']}): {train_long_signals:,}개 ({train_long_signals/len(train_data)*100:.1f}%)")
print(f"  LONG 승률: {train_long_winrate:.2f}%")
print(f"  SHORT 신호 (임계값 {CONFIG['threshold']}): {train_short_signals:,}개 ({train_short_signals/len(train_data)*100:.1f}%)")
print(f"  SHORT 승률: {train_short_winrate:.2f}%")

# 통합 통계
total_train_signals = train_long_signals + train_short_signals
total_train_wins = train_long_wins + train_short_wins
total_train_winrate = (total_train_wins / total_train_signals * 100) if total_train_signals > 0 else 0

print(f"\n통합 통계:")
print(f"  총 신호: {total_train_signals:,}개 ({total_train_signals/len(train_data)*100:.1f}%)")
print(f"  총 승률: {total_train_winrate:.2f}%")
print(f"  손익분기: 55.56%")
if total_train_winrate >= 55.56:
    print(f"  ✅ 학습 데이터 기준 수익 가능")
else:
    print(f"  ❌ 학습 데이터 기준 손익분기 미달")

if len(test_data) > 0:
    print(f"\n검증 성과:")
    print(f"  총 거래: {total_trades}개")
    print(f"  총 승률: {total_winrate:.2f}%")
    print(f"  손익분기 여부: {'✅ 통과' if total_winrate >= 55.56 else '❌ 미달'}")

print(f"\n핵심 RSI 피처 (TOP 5):")
rsi_top = rsi_importance.head(5)
for idx, row in rsi_top.iterrows():
    print(f"  {row['feature']}: {row['importance']:.1f}")

print(f"\n🎉 학습 완료!")
print(f"\n📝 다음 단계:")
print(f"1. 피처 중요도 확인 - RSI 관련 피처가 상위권인지 체크")
print(f"2. 실시간 거래 시스템에 모델 적용")
print(f"3. 소액으로 실전 테스트")
print(f"4. 데이터 수집 시스템 병행 실행")

# ============================================================
# 14. 패턴 분석 (옵션)
# ============================================================
print("\n" + "="*80)
print("🔍 학습된 패턴 분석")
print("="*80)

# RSI 과매도 + 15분봉 양봉 패턴의 효과
if 'rsi_low_after_15m_bull' in train_data.columns:
    pattern_cases = train_data[train_data['rsi_low_after_15m_bull'] == 1]
    if len(pattern_cases) > 0:
        pattern_wins = (pattern_cases['target_long'] == 1).sum()
        pattern_winrate = (pattern_wins / len(pattern_cases) * 100)
        print(f"\n'RSI 과매도 + 15분 양봉' 패턴:")
        print(f"  발생 횟수: {len(pattern_cases):,}개")
        print(f"  다음 10분 상승: {pattern_wins:,}개")
        print(f"  승률: {pattern_winrate:.2f}%")
        
        if pattern_winrate > 55:
            print(f"  ✅ 이 패턴은 유효합니다!")
        else:
            print(f"  ⚠️  이 패턴만으로는 부족합니다. 추가 필터 필요")

# RSI 과매수 + 15분봉 음봉 패턴
if 'rsi_high_after_15m_bear' in train_data.columns:
    pattern_cases = train_data[train_data['rsi_high_after_15m_bear'] == 1]
    if len(pattern_cases) > 0:
        pattern_wins = (pattern_cases['target_short'] == 1).sum()
        pattern_winrate = (pattern_wins / len(pattern_cases) * 100)
        print(f"\n'RSI 과매수 + 15분 음봉' 패턴:")
        print(f"  발생 횟수: {len(pattern_cases):,}개")
        print(f"  다음 10분 하락: {pattern_wins:,}개")
        print(f"  승률: {pattern_winrate:.2f}%")
        
        if pattern_winrate > 55:
            print(f"  ✅ 이 패턴은 유효합니다!")
        else:
            print(f"  ⚠️  이 패턴만으로는 부족합니다. 추가 필터 필요")

# 해머 패턴 효과
hammer_cases = train_data[train_data['is_hammer'] == 1]
if len(hammer_cases) > 0:
    hammer_wins = (hammer_cases['target_long'] == 1).sum()
    hammer_winrate = (hammer_wins / len(hammer_cases) * 100)
    print(f"\n'해머 캔들' 패턴:")
    print(f"  발생 횟수: {len(hammer_cases):,}개")
    print(f"  다음 10분 상승: {hammer_wins:,}개")
    print(f"  승률: {hammer_winrate:.2f}%")

print("\n" + "="*80)
print("✅ 전체 파이프라인 완료!")
print("="*80)
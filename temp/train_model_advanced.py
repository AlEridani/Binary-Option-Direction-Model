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
print("🚀 RSI + 프라이스 액션 ML 모델 (v4: 1시간 EMA 필터 & 최소 검증)")
print("="*80)

# ============================================================
# 설정
# ============================================================
CONFIG = {
    'lookback': 30,            # 30분 룩백
    'option_duration': 10,     # 10분 옵션
    'threshold': 0.65,         # 예측 임계값
    'data_folder': '1m',       # 학습 CSV 폴더
    'test_folder': 'test1m',   # 최소 검증용 CSV 폴더
    'roc_threshold': 0.0003,   # ROC 타겟 임계값
}

# ============================================================
# 1. CSV 파일 로드
# ============================================================
print("\n📂 CSV 파일 검색 중...")
csv_files = sorted(glob.glob(f"{CONFIG['data_folder']}/*.csv"))
if not csv_files:
    print(f"❌ '{CONFIG['data_folder']}' 폴더에서 CSV 파일을 찾을 수 없습니다!")
    exit(1)
print(f"✅ {len(csv_files)}개 CSV 파일 발견.")

dfs = []
for csv_file in csv_files:
    try:
        df_temp = pd.read_csv(csv_file, header=None)
        df_temp.columns = [
            'open_time','open','high','low','close','volume',
            'close_time','quote_volume','trades','taker_buy_volume',
            'taker_buy_quote_volume','ignore'
        ]
        df_temp['timestamp'] = pd.to_datetime(df_temp['open_time'], unit='us')
        df_temp = df_temp[['timestamp','open','high','low','close','volume']]
        for col in ['open','high','low','close','volume']:
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
        dfs.append(df_temp)
    except Exception as e:
        print(f"   ❌ {os.path.basename(csv_file)}: 오류 - {e}")

if not dfs:
    print("❌ 로드된 데이터가 없습니다!")
    exit(1)

df = pd.concat(dfs, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
print(f"✅ 총 {len(df):,}개 1분봉 데이터 로드")
print(f"   기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

# ============================================================
# 2. 15분봉, 1시간봉 집계 및 피처 생성
# ============================================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta>0,0)).rolling(period).mean()
    loss = (-delta.where(delta<0,0)).rolling(period).mean()
    rs = gain/loss
    return 100 - (100 / (1 + rs))

df.set_index('timestamp', inplace=True)

# 15분봉
df_15m = df.resample('15T').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()
df_15m['rsi_14'] = calculate_rsi(df_15m['close'],14)
df_15m['is_bullish_15m'] = (df_15m['close']>df_15m['open']).astype(int)
df_15m['is_bearish_15m'] = (df_15m['close']<df_15m['open']).astype(int)
df_15m['ema_20_15m'] = df_15m['close'].ewm(span=20,adjust=False).mean()
df_15m['ema_50_15m'] = df_15m['close'].ewm(span=50,adjust=False).mean()
df_15m_reindex = df_15m.reindex(df.index, method='ffill').reset_index()
for col in ['rsi_14','is_bullish_15m','is_bearish_15m','ema_20_15m','ema_50_15m']:
    df[col] = df_15m_reindex[col].values

# 1시간봉
df_1h = df.resample('1H').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna()
df_1h['ema_50_1h'] = df_1h['close'].ewm(span=50,adjust=False).mean()
df_1h_reindex = df_1h.reindex(df.index, method='ffill')
df['ema_50_1h'] = df_1h_reindex['ema_50_1h']
df['is_uptrend_1h'] = (df['close']>df['ema_50_1h']).astype(int)

# ============================================================
# 3. 1분봉 지표
# ============================================================
df['ema_20'] = df['close'].ewm(span=20,adjust=False).mean()
df['ema_50'] = df['close'].ewm(span=50,adjust=False).mean()
df['body'] = df['close']-df['open']
df['body_abs'] = abs(df['body'])
df['range'] = df['high']-df['low']
df['upper_shadow'] = df['high'] - df[['open','close']].max(axis=1)
df['lower_shadow'] = df[['open','close']].min(axis=1) - df['low']
df['body_ratio'] = df['body_abs'] / (df['range']+1e-8)
df['upper_shadow_ratio'] = df['upper_shadow'] / (df['range']+1e-8)
df['lower_shadow_ratio'] = df['lower_shadow'] / (df['range']+1e-8)
df['is_bullish'] = (df['close']>df['open']).astype(int)
df['is_bearish'] = (df['close']<df['open']).astype(int)
df['is_hammer'] = ((df['lower_shadow_ratio']>0.6)&(df['upper_shadow_ratio']<0.15)&(df['body_ratio']<0.3)).astype(int)
df['is_shooting_star'] = ((df['upper_shadow_ratio']>0.6)&(df['lower_shadow_ratio']<0.15)&(df['body_ratio']<0.3)).astype(int)
df['volume_ma_10'] = df['volume'].rolling(10).mean()
df['volume_ratio'] = df['volume']/(df['volume_ma_10']+1e-8)
df['volume_surge'] = (df['volume']>df['volume_ma_10']*1.5).astype(int)
df['recent_high_20'] = df['high'].rolling(20).max()
df['recent_low_20'] = df['low'].rolling(20).min()
df['distance_to_high'] = (df['recent_high_20']-df['close'])/df['close']
df['distance_to_low'] = (df['close']-df['recent_low_20'])/df['close']
df['trend_ema'] = (df['ema_20']>df['ema_50']).astype(int)
df['trend_15m'] = (df['ema_20_15m']>df['ema_50_15m']).astype(int)

# ============================================================
# 4. RSI 상태
# ============================================================
df['rsi_oversold'] = (df['rsi_14']<30).astype(int)
df['rsi_overbought'] = (df['rsi_14']>70).astype(int)
df['rsi_neutral'] = ((df['rsi_14']>=40)&(df['rsi_14']<=60)).astype(int)

# ============================================================
# 5. 타겟 (1시간 EMA 필터 적용)
# ============================================================
df['future_price'] = df['close'].shift(-CONFIG['option_duration'])
df['future_return'] = df['future_price']/df['close']-1
df['target_long'] = ((df['future_return']>CONFIG['roc_threshold']) & (df['is_uptrend_1h']==1)).astype(int)
df['target_short'] = ((df['future_return']<-CONFIG['roc_threshold']) & (df['is_uptrend_1h']==0)).astype(int)

df = df.reset_index()

# ============================================================
# 6. 룩백 윈도우 피처
# ============================================================
def create_lookback_features(df, lookback=30):
    features_list = []
    for i in range(lookback,len(df)):
        window = df.iloc[i-lookback:i]
        f = {'index':i,'timestamp':df.loc[i,'timestamp']}
        f['body_ratio']=df.loc[i,'body_ratio']
        f['upper_shadow_ratio']=df.loc[i,'upper_shadow_ratio']
        f['lower_shadow_ratio']=df.loc[i,'lower_shadow_ratio']
        f['is_hammer']=df.loc[i,'is_hammer']
        f['is_shooting_star']=df.loc[i,'is_shooting_star']
        f['ema_20']=df.loc[i,'ema_20']
        f['ema_50']=df.loc[i,'ema_50']
        f['ema_50_1h']=df.loc[i,'ema_50_1h']
        f['trend_ema']=df.loc[i,'trend_ema']
        f['trend_15m']=df.loc[i,'trend_15m']
        f['rsi_oversold']=df.loc[i,'rsi_oversold']
        f['rsi_overbought']=df.loc[i,'rsi_overbought']
        f['rsi_neutral']=df.loc[i,'rsi_neutral']
        f['volume_ratio']=df.loc[i,'volume_ratio']
        f['volume_surge']=df.loc[i,'volume_surge']
        f['distance_to_high']=df.loc[i,'distance_to_high']
        f['distance_to_low']=df.loc[i,'distance_to_low']
        f['target_long']=df.loc[i,'target_long']
        f['target_short']=df.loc[i,'target_short']
        features_list.append(f)
    return pd.DataFrame(features_list)

features_df = create_lookback_features(df, CONFIG['lookback'])
features_df = features_df.dropna().reset_index(drop=True)

# ============================================================
# 7. 학습/검증 분리 (검증 최소화)
# ============================================================
train_df = features_df[features_df['timestamp']<'2025-09-01']  # 최대한 학습
test_df = features_df[features_df['timestamp']>='2025-09-01']  # 최소 검증

X_train = train_df.drop(['index','timestamp','target_long','target_short'],axis=1)
y_train_long = train_df['target_long']
y_train_short = train_df['target_short']

X_test = test_df.drop(['index','timestamp','target_long','target_short'],axis=1)
y_test_long = test_df['target_long']
y_test_short = test_df['target_short']

# ============================================================
# 8. 모델 학습
# ============================================================
print("\n========================================")
print("📈 LONG 모델 학습")
print("========================================")
model_long = LGBMClassifier(n_estimators=500, max_depth=6, class_weight='balanced')
model_long.fit(X_train, y_train_long)

print("\n========================================")
print("📉 SHORT 모델 학습")
print("========================================")
model_short = LGBMClassifier(n_estimators=500, max_depth=6, class_weight='balanced')
model_short.fit(X_train, y_train_short)

# ============================================================
# 9. 검증/결과
# ============================================================
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

def evaluate(model,X,y,label):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1]
    print(f"\nConfusion matrix ({label}):\n",confusion_matrix(y,y_pred))
    print(f"\nClassification report ({label}):\n",classification_report(y,y_pred))
    print(f"ROC AUC ({label}): {roc_auc_score(y,y_prob):.4f}")

evaluate(model_long,X_test,y_test_long,'LONG')
evaluate(model_short,X_test,y_test_short,'SHORT')

# ============================================================
# 10. 모델 저장
# ============================================================
model_folder = 'model'
os.makedirs(model_folder,exist_ok=True)
ts = datetime.now().strftime('%Y%m%d_%H%M%S')
joblib.dump(model_long, f"{model_folder}/rsi_price_action_long_{ts}.pkl")
joblib.dump(model_short, f"{model_folder}/rsi_price_action_short_{ts}.pkl")
print(f"\n✅ 모델 저장 완료: {model_folder}/rsi_price_action_*_{ts}.pkl")

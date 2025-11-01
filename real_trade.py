# real_trade.py
# 30분 바이너리 옵션 실시간 거래 및 백테스트 모듈 (A 방식)
#
# 핵심 규칙
# - 29분 50~59초에 예측(분%30==29 & 초 50~59)
# - 바로 "다음 바" 정각에 진입 (entry_at = floor_30m(now) + 30m)
# - 엔트리/클로즈 로그와 피쳐 로그는 같은 앵커 키(bar30_start/bar30_end)로 기록
# - config.decide() 사용, TTL/조기청산 없음, feature_engineer의 target 정의에 따름

from __future__ import annotations

import time
import csv
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta

from binance.client import Client

import config
from timeframe_manager import TimeframeManager
from feature_engineer import FeatureEngineer
from model_train import ModelTrainer
from log_manager import LogManager


# ==============================
# 주문 실행 어댑터
# ==============================
class OrderExecutor:
    """
    단일 실행 경로:
      - mode="REAL": 실주문만
      - mode="PAPER": 페이퍼만
      - mode="AUTO": 실주문 시도 -> 실패하면 페이퍼로 폴백
    모든 경우 동일한 레코드 스키마를 반환하여 상위 로직은 동일하게 처리.
    """
    def __init__(self, client, mode: str = "AUTO"):
        self.client = client  # binance Client 또는 None
        self.mode = mode.upper()

    def place_option_order(self, symbol: str, direction: str, qty: float, ttl_min: int):
        """
        옵션 ‘가상’ 표준 인터페이스.
        반환 dict 필수 키:
          kind: "REAL"|"PAPER"
          ok: bool
          order_id: str|None
          reason: str|None
          placed_at: datetime (UTC)
          expire_at: datetime (UTC)
        """
        now = datetime.now(timezone.utc)
        expire_at = now + timedelta(minutes=ttl_min)

        # 내부 헬퍼: 페이퍼 주문
        def _paper(reason=None):
            return {
                "kind": "PAPER",
                "ok": True,
                "order_id": None,
                "reason": reason,
                "placed_at": now,
                "expire_at": expire_at,
            }

        # REAL 전용 시도
        def _try_real():
            if self.client is None:
                raise RuntimeError("Client is None")
            # ⚠️ 여기는 바이낸스 옵션 주문 API 연동 자리
            # 실제 엔드포인트/파라미터는 추후 연결 (현재는 더미로 실패 시뮬레이션)
            # ex) self.client.create_options_order(...)
            raise NotImplementedError("Binance Options API wiring is pending")

        # 모드별 분기
        if self.mode == "PAPER":
            return _paper(reason="PAPER_MODE")
        if self.mode == "REAL":
            try:
                _try_real()
                # 성공 시 실제 order_id 등 세팅
                return {
                    "kind": "REAL",
                    "ok": True,
                    "order_id": "REAL_ORDER_ID_PLACEHOLDER",
                    "reason": None,
                    "placed_at": now,
                    "expire_at": expire_at,
                }
            except Exception as e:
                return {
                    "kind": "REAL",
                    "ok": False,
                    "order_id": None,
                    "reason": f"REAL_FAIL: {e}",
                    "placed_at": now,
                    "expire_at": expire_at,
                }

        # AUTO: REAL 시도 후 실패 → PAPER
        try:
            _try_real()
            return {
                "kind": "REAL",
                "ok": True,
                "order_id": "REAL_ORDER_ID_PLACEHOLDER",
                "reason": None,
                "placed_at": now,
                "expire_at": expire_at,
            }
        except Exception as e:
            return _paper(reason=f"AUTO_FALLBACK: {e}")


# ==============================
# 실시간 / 백테스트 매니저
# ==============================
class RealTradeManager:
    """30분 바이너리 옵션 실거래 관리 (A 방식: 다음 바 정각 진입)"""

    def __init__(self, trainer: ModelTrainer | None = None, *args, **kwargs):
        self.client = Client(config.BINANCE_API_KEY, config.BINANCE_API_SECRET)
        self.tfm = TimeframeManager()
        self.fe = FeatureEngineer()
        self.trainer = trainer or ModelTrainer()

        # 포지션 상태
        self.in_position = False
        self.entry_time = None
        self.entry_price = None
        self.entry_dir = None
        self.entry_regime = None
        self.entry_kind = None
        self.entry_oid = None
        self.entry_bar30_start = None
        self.entry_bar30_end = None
        self.entry_p_up = None

        # ✅ 만기(청산) 시각 고정 — 엔트리 시 bar30_end를 저장
        self.exit_at = None

        self.latest_df = None

        # 예측 타이밍 제어
        self._pred_window = (50, 59)   # 29분의 50~59초 구간에서만 예측
        self._last_pred_anchor: datetime | None = None  # 직전 예측이 향하던 '다음 바 시작' 시간

        # 안전 종료 핸들러
        self._stop = False
        import signal

        def _sigint_handler(signum, frame):
            self._stop = True
            print("\n⏹️  SIGINT 수신 — 안전 종료 준비 중...")

        try:
            signal.signal(signal.SIGINT, _sigint_handler)
        except Exception:
            pass

        # 번들 로드
        self.model_loaded = False
        self._load_latest_bundle()

        # 실행기
        exec_mode = getattr(config, "EXECUTION_MODE", "AUTO")
        self.executor = OrderExecutor(self.client, mode=exec_mode)

        # 로그 매니저
        self.logger = LogManager()

    # --------------- 유틸리티 ---------------

    def _safe_sleep(self, seconds: float, chunk: float = 0.5):
        """Ctrl+C에 잘 반응하도록 쪼개서 슬립"""
        remain = max(0.0, float(seconds))
        while remain > 0 and not self._stop:
            t = min(remain, chunk)
            try:
                time.sleep(t)
            except KeyboardInterrupt:
                self._stop = True
                print("\n⏹️  사용자 종료(대기 중)")
                break
            remain -= t

    def _wait_until(self, target_dt: datetime):
        """target_dt(UTC)까지 안전 대기"""
        while not self._stop:
            now = datetime.now(timezone.utc)
            if now >= target_dt:
                break
            remaining = (target_dt - now).total_seconds()
            self._safe_sleep(min(1.0, remaining))

    @staticmethod
    def _floor_30m(dt: datetime) -> datetime:
        """dt(UTC)를 30분 단위로 내림(정각/30분)"""
        return dt.replace(minute=(dt.minute // 30) * 30, second=0, microsecond=0, tzinfo=timezone.utc)

    def _compute_bar_anchors(self, now_utc: datetime) -> tuple[datetime, datetime, datetime]:
        """
        A 방식: 지금 시각(now_utc)을 기준으로
        - entry_at = floor_30m(now) + 30m (다음 바 시작 정각)
        - bar30_start = entry_at
        - bar30_end   = entry_at + 30m
        """
        base = self._floor_30m(now_utc)
        entry_at = base + timedelta(minutes=30)
        bar30_start = entry_at
        bar30_end = entry_at + timedelta(minutes=30)
        return bar30_start, bar30_end, entry_at

    def _log_feature_snapshot(
        self,
        symbol: str,
        bar30_start: datetime,
        bar30_end: datetime,
        regime: int,
        p_up: float,
        margin: float,
        extra: dict | None = None,
    ):
        """
        피쳐 스냅샷을 '바 앵커' 기준으로 저장
        파일명: feature_log/{SYMBOL}_features_YYYYMMDD.csv
        """
        day_str = bar30_start.strftime("%Y%m%d")
        path = getattr(config, "FEATURE_LOG_DIR", "feature_log")
        Path(path).mkdir(parents=True, exist_ok=True)
        fpath = Path(path) / f"{symbol}_features_{day_str}.csv"

        row = {
            "symbol": symbol,
            "bar30_start": bar30_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bar30_end": bar30_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "regime": int(regime),
            "p_up": float(p_up),
            "margin": float(margin),
            "pred_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        if extra:
            row.update({k: extra[k] for k in extra})

        header = ["symbol", "bar30_start", "bar30_end", "regime", "p_up", "margin", "pred_at", "decision"]
        if "decision" not in row:
            row["decision"] = "UP" if row["p_up"] >= 0.5 else "DOWN"

        write_header = not fpath.exists()
        with fpath.open("a", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=header, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow(row)

    # --------------- 데이터/모델 준비 ---------------

    def fetch_latest_klines(self, limit: int = None) -> pd.DataFrame:
        """최신 30분봉 데이터 수집 (UTC)"""
        limit = limit or getattr(config, "LIMIT_30m", 200)
        try:
            klines = self.client.get_klines(
                symbol=config.SYMBOL,
                interval=getattr(config, "INTERVAL_30m", "30m"),
                limit=limit,
            )
            df = pd.DataFrame(
                klines,
                columns=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "trades",
                    "taker_buy_base",
                    "taker_buy_quote",
                    "ignore",
                ],
            )
            # UTC 시간 변환
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
            # 숫자형 변환
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            df.set_index("open_time", inplace=True)
            df.sort_index(inplace=True)
            return df
        except Exception as e:
            print(f"❌ 데이터 수집 실패: {e}")
            return pd.DataFrame()

    def _load_latest_bundle(self):
        base = config.MODEL_DIR
        latest = base / "latest"
        print(f"[DEBUG] try bundle: {latest}")
        if self.trainer.load_bundle("latest"):
            self.model_loaded = True
            print("✅ 번들 로드: latest")
            return

        # fallback: 가장 최근 bundle_* 찾기
        bundles = sorted(base.glob("bundle_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if bundles:
            name = bundles[0].name
            print(f"[DEBUG] fallback bundle: {name}")
            if self.trainer.load_bundle(name):
                self.model_loaded = True
                print(f"✅ 번들 로드: {name}")
                return

        print("❌ 번들 로드 실패 (models/latest 혹은 bundle_* 확인)")
        self.model_loaded = False

    # --------------- 타이밍 제어 ---------------

    def should_predict_now(self) -> bool:
        """
        30분 바 닫히기 직전(분%30==29)의 지정된 초 구간(기본 50~59초)에,
        '다음 바 시작(anchor_start)'을 기준으로 한 번만 예측.
        """
        now = datetime.now(timezone.utc)
        # 29분대가 아니면 skip
        if (now.minute % 30) != 29:
            return False
        # 초 범위 체크
        lo, hi = self._pred_window
        if not (lo <= now.second <= hi):
            return False

        # 이번 예측이 향하는 '다음 바 시작' 타임스탬프(앵커)를 계산
        base = self._floor_30m(now)
        next_anchor = base + timedelta(minutes=30)  # 다음 바 시작 정각

        # 직전 앵커와 동일하면 중복 예측 방지
        if self._last_pred_anchor is not None and self._last_pred_anchor == next_anchor:
            return False

        self._last_pred_anchor = next_anchor
        return True

    # --------------- 예측/판단/실행 ---------------

    def predict_and_decide(self, df: pd.DataFrame) -> dict:
        """
        - 모델 로드 여부 확인
        - 30분봉 피처 생성
        - 메타 컬럼 제거(regime/target/timestamp 등)
        - 학습 시 피처 목록/순서에 정렬(가능하면 레짐별)
        - 결측/타입 보정 후 마지막 1행으로 예측
        - config.decide 로 진입여부/방향 판정
        """
        try:
            if not self.model_loaded:
                print("⚠️  모델 미로딩 (self.model_loaded=False)")
                return {
                    "should_enter": False,
                    "direction": None,
                    "pred_proba": 0.0,
                    "regime": 0,
                    "margin": 0.0,
                }

            # 1) 피처 생성
            df_in = df.copy()
            df_feat = self.fe.create_feature_pool(df_in)
            if df_feat is None or df_feat.empty:
                print("❌ 피처 생성 실패 (빈 DataFrame)")
                return {
                    "should_enter": False,
                    "direction": None,
                    "pred_proba": 0.0,
                    "regime": 0,
                    "margin": 0.0,
                }

            # 2) 필수 라벨
            if "regime" not in df_feat.columns:
                print("⚠️  'regime' 없음 → 0(FLAT)로 대체")
                df_feat["regime"] = 0

            # 3) 최소 샘플 수
            need = int(getattr(config, "PREDICT_BARS_30m", 30))
            if len(df_feat) < need:
                print(f"⏳ 샘플 부족: {len(df_feat)} < {need}")
                return {
                    "should_enter": False,
                    "direction": None,
                    "pred_proba": 0.0,
                    "regime": int(df_feat["regime"].iloc[-1]),
                    "margin": 0.0,
                }

            # 4) 최근 구간/레짐
            recent = df_feat.iloc[-need:].copy()
            regime = int(recent["regime"].iloc[-1])

            # 5) 학습-예측 피처 정합
            meta_cols = {
                "regime",
                "target",
                "timestamp",
                "bar30_start",
                "open_time",
                "close_time",
                "data_ver",
                "feature_ver",
            }
            candidate = [c for c in df_feat.columns if c not in meta_cols]

            train_cols = None
            model_for_regime = None
            if hasattr(self.trainer, "get_feature_names_for_regime"):
                try:
                    cols = self.trainer.get_feature_names_for_regime(regime)
                    if cols:
                        train_cols = list(cols)
                except Exception:
                    pass

            if train_cols is None and hasattr(self.trainer, "get_model_for_regime"):
                try:
                    model_for_regime = self.trainer.get_model_for_regime(regime)
                    if model_for_regime is not None:
                        if hasattr(model_for_regime, "feature_name_") and model_for_regime.feature_name_:
                            train_cols = list(model_for_regime.feature_name_)
                        elif hasattr(model_for_regime, "feature_names_in_"):
                            train_cols = list(model_for_regime.feature_names_in_)
                except Exception:
                    pass

            if train_cols is None:
                for attr in ("feature_names_", "features_", "feat_cols_", "feature_names_in_"):
                    if hasattr(self.trainer, attr):
                        cols = getattr(self.trainer, attr)
                        if cols:
                            train_cols = list(cols)
                            break

            if train_cols:
                use_cols = [c for c in train_cols if c in candidate]
                dropped = [c for c in train_cols if c not in use_cols]
                if dropped:
                    print(
                        f"[WARN] 학습 피처 중 계산 불가 {len(dropped)}개 제외: "
                        f"{dropped[:8]}{' ...' if len(dropped)>8 else ''}"
                    )
            else:
                use_cols = candidate
                print("[WARN] 학습 피처 목록을 찾지 못함 → 메타 제외 후보 전체 사용")

            if not use_cols:
                print("❌ 사용할 피처가 없습니다(use_cols 비어있음)")
                return {
                    "should_enter": False,
                    "direction": None,
                    "pred_proba": 0.0,
                    "regime": regime,
                    "margin": 0.0,
                }

            # 6) 결측/타입 보정
            before_na = recent[use_cols].isna().mean().mean()
            recent[use_cols] = recent[use_cols].ffill().bfill()
            x_last = recent[use_cols].iloc[[-1]].copy()
            for c in x_last.columns:
                if not pd.api.types.is_numeric_dtype(x_last[c]):
                    x_last[c] = pd.to_numeric(x_last[c], errors="coerce")
            if x_last.isna().any(axis=1).item():
                x_last = x_last.fillna(0.0)
            x_last = x_last.reindex(columns=use_cols, fill_value=0.0)

            # 7) 예측
            try:
                proba = self.trainer.predict_with_regime(x_last, regime)
                if isinstance(proba, (list, tuple, np.ndarray)):
                    proba = np.asarray(proba)
                    if proba.ndim == 2 and proba.shape[1] >= 2:
                        pred_proba = float(proba[0, 1])
                    else:
                        pred_proba = float(proba.ravel()[0])
                else:
                    pred_proba = float(proba)
            except Exception as pe:
                print(f"❌ 예측 단계 실패: {pe}")
                pred_proba = 0.5

            regime_name = {1: "UP", -1: "DOWN", 0: "FLAT"}.get(regime, "UNKNOWN")
            print(
                f"[PRED] regime={regime_name}({regime})  used_feats={len(use_cols)}  "
                f"NaN(mean_before)={before_na:.3f}  p_up={pred_proba:.4f}"
            )

            if getattr(config, "FORCE_ENTRY", False):
                direction = "UP" if pred_proba >= 0.5 else "DOWN"
                print(f"⚙️  FORCE_ENTRY=True → 무조건 진입: {direction}")
                return {
                    "should_enter": True,
                    "direction": direction,
                    "pred_proba": float(pred_proba),
                    "regime": int(regime),
                    "margin": 0.0,
                }

            # 8) 진입 판단
            margin = float(getattr(config, "dynamic_margin", lambda **k: 0.0)(ece50=0.0, entropy=0.0))
            try:
                direction = config.decide(pred_proba, margin)
            except Exception:
                direction = "UP" if pred_proba >= 0.5 else "DOWN"
            should_enter = direction is not None

            return {
                "should_enter": bool(should_enter),
                "direction": direction if should_enter else None,
                "pred_proba": float(pred_proba),
                "regime": int(regime),
                "margin": float(margin),
            }

        except KeyboardInterrupt:
            print("\n⏹️  사용자 종료(Ctrl+C)")
            self._stop = True
            return {
                "should_enter": False,
                "direction": None,
                "pred_proba": 0.0,
                "regime": 0,
                "margin": 0.0,
            }

        except Exception as e:
            print(f"❌ 예측 실패(상위): {e}")
            return {
                "should_enter": False,
                "direction": None,
                "pred_proba": 0.0,
                "regime": 0,
                "margin": 0.0,
            }

    def execute_entry(self, direction: str, pred_proba: float, regime: int,
                      bar30_start: datetime, bar30_end: datetime):
        now = datetime.now(timezone.utc)
        regime_name = {1: "UP", -1: "DOWN", 0: "FLAT"}.get(regime, "UNKNOWN")

        # 엔트리 가격 스냅샷
        last_close = float("nan")
        try:
            if self.latest_df is not None and not self.latest_df.empty:
                last_close = float(self.latest_df["close"].iloc[-1])
        except Exception:
            pass

        # 실행기
        qty = getattr(config, "POSITION_SIZE", 1)
        ttl = getattr(config, "OPTION_TENOR_MIN", 30)
        res = self.executor.place_option_order(config.SYMBOL, direction, qty, ttl)

        kind = res["kind"]
        ok = res["ok"]
        order_id = res["order_id"]
        expire_at = res["expire_at"]
        reason = res["reason"]

        print("\n" + "=" * 60)
        print(f"🚀 엔트리 [{kind}] ok={ok} order_id={order_id} reason={reason}")
        print(f"UTC: {now:%Y-%m-%d %H:%M:%S} dir={direction} p_up={pred_proba:.4f} regime={regime_name}")
        print(f"entry_close={last_close} tenor={ttl}min  expire_at={expire_at:%H:%M:%S} UTC")
        print("=" * 60 + "\n")

        if ok:
            # 상태 저장 (CLOSE 시 필요)
            self.in_position = True
            self.entry_time = now
            self.entry_price = last_close
            self.entry_dir = direction
            self.entry_regime = regime
            self.entry_kind = kind
            self.entry_oid = order_id
            self.entry_bar30_start = bar30_start
            self.entry_bar30_end = bar30_end
            self.entry_p_up = float(pred_proba)

            # ✅ 이번 포지션의 만기(청산) 시각을 바 앵커로 고정
            self.exit_at = bar30_end
            print(f"[ENTRY] will exit at {self.exit_at:%Y-%m-%d %H:%M:%S} UTC")

            # ENTRY 로그
            try:
                self.logger.log_trade_entry_simple(
                    trade_id=order_id or f"{kind}_{now.strftime('%Y%m%dT%H%M%S')}",
                    direction=direction,                 # 'UP'/'DOWN'
                    entry_price=float(last_close),
                    entry_ts=now,
                    p_raw_at_entry=float(pred_proba),    # 표준 컬럼(캘리브 전)
                    p_cal_at_entry=float(pred_proba),    # 캘리브 없으면 동일
                    cal_method="identity",
                    cal_ver=getattr(config, "CALIB_VERSION", ""),
                    regime=int(regime),
                    bar30_start=bar30_start,
                    bar30_end=bar30_end,
                    # 가능 시 아래 2개도 로그로 남기면 복구 정확도↑ (모르는 필드는 무시됨)
                    exit_at=bar30_end,
                    exit_anchor=bar30_end,
                    kind=kind,
                    symbol=getattr(config, "SYMBOL", "BTCUSDT"),
                    payout=getattr(config, "PAYOUT_30M_PLUS", 0.85),
                    tenor_min=ttl,
                )
            except Exception as e:
                print(f"[WARN] entry log failed: {e}")
        else:
            print("⛔ 진입 실패 (REAL 전용 & 실패, 폴백 없음)")

    def check_exit(self):
        """
        만기(=exit_at) 시각이 지났는지로 청산 판단.
        in_position/entry_time 플래그가 사라져도 exit_at이 남아있으면 청산 시도.
        """
        now = datetime.now(timezone.utc)

        # 🔒 안전장치 1: in_position/entry_time이 없어도 exit_at이 있으면 그걸 신뢰
        if self.exit_at is None:
            # 과거 버전 호환: entry_bar30_end가 있다면 그걸 만기로 사용
            try:
                if self.entry_bar30_end is not None:
                    self.exit_at = self.entry_bar30_end
            except Exception:
                pass

        # 안전장치 2: 아무 정보도 없으면 리턴
        if self.exit_at is None:
            return

        # 아직 만기 전이면 10초마다 대기 로그
        if now < self.exit_at:
            remain = (self.exit_at - now).total_seconds()
            if int(remain) % 10 == 0:
                print(f"[EXIT-WAIT] {int(remain)}s until expiry ({self.exit_at:%H:%M:%S} UTC)")
            return

        # 만기 스냅샷용 종가
        exit_price = float("nan")
        try:
            df = self.fetch_latest_klines(limit=max(2, getattr(config, "LIMIT_30m", 200)))
            if not df.empty:
                exit_price = float(df["close"].iloc[-1])
        except Exception:
            pass

        # 가상 판정
        correct = None
        if self.entry_dir == "UP":
            correct = exit_price > self.entry_price
        elif self.entry_dir == "DOWN":
            correct = exit_price < self.entry_price

        payout = getattr(config, "PAYOUT_30M_PLUS", 0.85)
        pnl = payout if correct else -1.0
        res = "WIN" if correct else "LOSS"

        print("\n" + "=" * 60)
        print(f"⏰ 만기 채점 kind={getattr(self,'entry_kind','PAPER')} oid={getattr(self,'entry_oid',None)}")
        print(f"in={self.entry_price} out={exit_price} dir={self.entry_dir} → {res} pnl={pnl:+.2f}")
        print("=" * 60 + "\n")

        # CLOSE 로그 (trade_id 동일 키로 결과 업데이트)
        try:
            trade_id = (
                getattr(self, "entry_oid", None)
                or f"{getattr(self,'entry_kind','PAPER')}_{self.entry_time.strftime('%Y%m%dT%H%M%S')}"
            )
            self.logger.update_trade_result(
                trade_id=trade_id,
                result=res,
                label_price=float(exit_price),
                label_ts=now,
                payout=float(payout),
                pnl=float(pnl),
            )
        except Exception as e:
            print(f"[WARN] close log failed: {e}")

        # 상태 초기화
        self.in_position = False
        self.entry_time = None
        self.entry_price = None
        self.entry_dir = None
        self.entry_regime = None
        self.entry_kind = None
        self.entry_oid = None
        self.entry_bar30_start = None
        self.entry_bar30_end = None
        self.entry_p_up = None
        self.exit_at = None  # ✅ 만기 시각 초기화

    # ---------- 보호장치: 로그의 OPEN들을 스캔해 강제 청산 ----------

    def _iter_due_open_trades(self, lookback_days: int = 2):
        """
        최근 N일 로그에서 OPEN 상태이며 bar30_end(혹은 exit_at)가 now 이전인 레코드들을 yield.
        LogManager에 의존하지 않고 CSV를 직접 스캔하는 보수적 구현.
        파일명 규칙은 {SYMBOL}_trades_YYYYMMDD.csv 로 가정 (config에서 다르면 수정)
        """
        log_dir = getattr(config, "TRADE_LOG_DIR", "trade_log")
        symbol = getattr(config, "SYMBOL", "BTCUSDT")
        now = datetime.now(timezone.utc)

        # 오늘 + lookback_days 일수
        days = [now.date()]
        for i in range(1, lookback_days + 1):
            days.append((now - timedelta(days=i)).date())

        for d in days:
            day_str = d.strftime("%Y%m%d")
            path = Path(log_dir) / f"{symbol}_trades_{day_str}.csv"
            if not path.exists():
                continue

            try:
                with path.open("r", newline="", encoding="utf-8") as fp:
                    r = csv.DictReader(fp)
                    for row in r:
                        status = (row.get("status") or row.get("trade_status") or "").upper()
                        if status and status != "OPEN":
                            continue

                        # 만기 판단 기준: exit_at 우선, 없으면 bar30_end
                        exit_str = row.get("exit_at") or row.get("bar30_end") or ""
                        if not exit_str:
                            continue
                        try:
                            exit_at = pd.to_datetime(exit_str, utc=True)
                            if exit_at.tzinfo is None:
                                exit_at = exit_at.tz_localize("UTC")
                        except Exception:
                            continue

                        if now >= exit_at:
                            yield row, exit_at
            except Exception as e:
                print(f"[GUARD] read error {path}: {e}")

    def _force_close_by_row(self, row, exit_at):
        """
        OPEN 행(row)을 받아, 현재 30분봉의 종가로 WIN/LOSS 계산 후 같은 trade_id로 CLOSE 업데이트.
        """
        trade_id = (
            row.get("trade_id")
            or row.get("order_id")
            or row.get("id")
            or (
                (row.get("kind", "PAPER") + "_" + (row.get("bar30_start", "") or "")
                 .replace(":", "")
                 .replace("-", "")
                 .replace("Z", "")
                 .replace("T", ""))
            )
        )

        # 엔트리 정보
        try:
            entry_price = float(row.get("entry_price") or row.get("in") or "nan")
        except Exception:
            entry_price = float("nan")

        direction = (row.get("direction") or row.get("side") or "").upper()
        payout = float(row.get("payout") or getattr(config, "PAYOUT_30M_PLUS", 0.85))

        # 만기 종가 스냅샷(가장 최근 30m 종가)
        exit_price = float("nan")
        try:
            df = self.fetch_latest_klines(limit=max(2, getattr(config, "LIMIT_30m", 200)))
            if not df.empty:
                exit_price = float(df["close"].iloc[-1])
        except Exception:
            pass

        # 판정
        correct = None
        if direction == "UP":
            correct = exit_price > entry_price
        elif direction == "DOWN":
            correct = exit_price < entry_price

        pnl = payout if correct else -1.0
        res = "WIN" if correct else "LOSS"

        try:
            self.logger.update_trade_result(
                trade_id=trade_id,
                result=res,
                label_price=float(exit_price),
                label_ts=datetime.now(timezone.utc),
                payout=float(payout),
                pnl=float(pnl),
            )
            print(f"[FORCE-CLOSE] {trade_id} → {res} (in={entry_price}, out={exit_price})")
        except Exception as e:
            print(f"[WARN] force close failed for {trade_id}: {e}")

    # --------------- 메인 루프 ---------------

    def run_live(self):
        """실시간 거래 루프 (A 방식)"""
        print("🔄 실시간 거래 시작 (30분 주기)")
        print("⏰ 29분 50~59초에 예측 → 다음 바 정각 진입")
        print("📊 양방향 진입 (UP/DOWN)\n")

        while not self._stop:
            try:
                # 청산 체크
                self.check_exit()

                # 🛡️ 보호장치: 로그에 남아있지만 메모리 플래그가 사라진 OPEN들을 강제 청산
                try:
                    for row, exit_at in self._iter_due_open_trades(lookback_days=2):
                        self._force_close_by_row(row, exit_at)
                except Exception as _e:
                    print(f"[GUARD] scan open trades err: {_e}")

                # 예측 타이밍 체크
                if self.should_predict_now() and not self.in_position:
                    print(f"🔍 예측 시작: {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")

                    # 데이터 수집 & 캐시
                    df = self.fetch_latest_klines()
                    self.latest_df = df

                    if df.empty:
                        print("⚠️ 데이터 없음, 스킵")
                    else:
                        # 예측 및 진입 판단
                        result = self.predict_and_decide(df)

                        regime_name = {1: "UP", -1: "DOWN", 0: "FLAT"}.get(result["regime"], "UNKNOWN")
                        print(
                            f"레짐: {regime_name} ({result['regime']}), "
                            f"예측 확률: {result['pred_proba']:.4f}, "
                            f"마진: {result['margin']:.4f}"
                        )

                        if result["should_enter"]:
                            # === A 방식: 다음 바 앵커/정각 계산 ===
                            now_utc = datetime.now(timezone.utc)
                            bar30_start, bar30_end, entry_at = self._compute_bar_anchors(now_utc)

                            # 피쳐 스냅샷: 바 앵커 기준 기록
                            try:
                                self._log_feature_snapshot(
                                    symbol=getattr(config, "SYMBOL", "BTCUSDT"),
                                    bar30_start=bar30_start,
                                    bar30_end=bar30_end,
                                    regime=int(result["regime"]),
                                    p_up=float(result["pred_proba"]),
                                    margin=float(result["margin"]),
                                    extra={
                                        "decision": ("UP" if result["pred_proba"] >= 0.5 else "DOWN"),
                                        "pred_at": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                    },
                                )
                            except Exception:
                                pass

                            # 다음 정각까지 대기
                            wait_sec = (entry_at - now_utc).total_seconds()
                            if wait_sec < 0:
                                entry_at += timedelta(minutes=30)
                                wait_sec = (entry_at - now_utc).total_seconds()

                            print(f"⏳ {wait_sec:.1f}초 후 진입 ({entry_at.strftime('%H:%M:%S')} UTC)")
                            self._wait_until(entry_at)
                            if self._stop:
                                break

                            # 진입 실행
                            self.execute_entry(
                                result["direction"],
                                result["pred_proba"],
                                result["regime"],
                                bar30_start=bar30_start,
                                bar30_end=bar30_end,
                            )
                        else:
                            print("⛔ 진입 조건 미충족, 스킵\n")

                self._safe_sleep(0.5)

            except KeyboardInterrupt:
                print("\n⏹️  사용자 종료(Ctrl+C)")
                self._stop = True
            except Exception as e:
                print(f"❌ 에러 발생: {e}")
                self._safe_sleep(3.0)

        print("🛑 라이브 루프 종료")

    # --------------- 간단 백테스트 ---------------

    def backtest(self, df_1m, start_date=None, end_date=None):
        btm = BacktestManager()  # ModelTrainer 기반
        return btm.run_backtest(df_1m, start_date=start_date, end_date=end_date)


# ==============================
# 간단 백테스터
# ==============================
class BacktestManager:
    """
    30분 바이너리 옵션용 간단 백테스터.
    - 1분봉 입력 → 30분봉 집계 → 피처 생성 → 레짐별 모델 예측 → 성능 집계
    - 모델은 model_train.py에서 저장한 'latest' 번들을 로드
    """
    def __init__(self):
        self.fe = FeatureEngineer()
        self.tfm = TimeframeManager()
        self.trainer = ModelTrainer()
        ok = self.trainer.load_bundle("latest")
        if not ok:
            print("⚠️  최신 번들을 찾지 못했습니다. 먼저 학습을 수행하세요.")

    def _prepare_features(self, df_in: pd.DataFrame) -> pd.DataFrame:
        """입력이 1분봉이든 30분봉이든 받아서 df_feat 생성"""
        if df_in is None or df_in.empty:
            raise RuntimeError("입력 데이터가 비어 있습니다.")

        d = df_in.copy()
        if "timestamp" not in d.columns:
            if isinstance(d.index, pd.DatetimeIndex):
                d = d.reset_index().rename(columns={"index": "timestamp"})

        candid_cols = {"open", "high", "low", "close", "volume"}
        if not candid_cols.issubset(set(d.columns)):
            raise RuntimeError("OHLCV 컬럼이 부족합니다. 필요: open/high/low/close/volume")

        df_30 = None
        if "bar30_start" in d.columns:
            df_30 = d.rename(columns={"bar30_start": "timestamp"})[
                ["timestamp", "open", "high", "low", "close", "volume"]
            ].copy()

        if df_30 is None and "timestamp" in d.columns:
            try:
                ts = pd.to_datetime(d["timestamp"], utc=True, errors="coerce").dropna()
                ts = ts.sort_values().unique()
                if len(ts) > 2:
                    deltas = pd.Series(ts[1:] - ts[:-1]).astype("timedelta64[m]")
                    share_30m = (abs(deltas - 30) < 1e-6).mean()
                else:
                    share_30m = 0.0
            except Exception:
                share_30m = 0.0
            if share_30m > 0.8:
                df_30 = d[["timestamp", "open", "high", "low", "close", "volume"]].copy()

        if df_30 is None:
            df_30 = self.tfm.aggregate_1m_to_30m(d, realtime_safe=False)
            if "bar30_start" not in df_30.columns:
                raise RuntimeError("30분 집계 결과에 bar30_start가 없습니다.")
            df_30 = df_30.rename(columns={"bar30_start": "timestamp"})[
                ["timestamp", "open", "high", "low", "close", "volume"]
            ].copy()

        df_feat = self.fe.create_feature_pool(df_30)

        if "timestamp" not in df_feat.columns:
            if isinstance(df_feat.index, pd.DatetimeIndex):
                df_feat = df_feat.copy()
                ts_idx = df_feat.index
                if ts_idx.tz is None:
                    ts_idx = ts_idx.tz_localize("UTC")
                else:
                    ts_idx = ts_idx.tz_convert("UTC")
                df_feat["timestamp"] = ts_idx
            elif "bar30_start" in df_feat.columns:
                df_feat = df_feat.rename(columns={"bar30_start": "timestamp"})
            else:
                if "timestamp" in df_30.columns and len(df_30) == len(df_feat):
                    df_feat = df_feat.copy()
                    df_feat["timestamp"] = pd.to_datetime(df_30["timestamp"], utc=True)
                else:
                    raise RuntimeError("피처에서 'timestamp'를 복구할 수 없습니다.")

        for col in ("regime", "target"):
            if col not in df_feat.columns:
                raise RuntimeError("피처에 'regime' 또는 'target' 컬럼이 없습니다. FeatureEngineer 라벨링을 확인하세요.")

        return df_feat

    def run_backtest(self, df_1m: pd.DataFrame, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        """간단 백테스트 실행."""
        if df_1m is None or df_1m.empty:
            raise ValueError("빈 데이터로는 백테스트를 수행할 수 없습니다.")

        if start_date:
            df_1m = df_1m[df_1m["timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
        if end_date:
            df_1m = df_1m[df_1m["timestamp"] <= pd.Timestamp(end_date, tz="UTC")]

        df_feat = self._prepare_features(df_1m)

        feature_cols = self.fe.get_feature_names(df_feat)
        X_all = df_feat[feature_cols].copy()
        regimes = df_feat["regime"].astype(int).to_numpy()
        y_true = df_feat["target"].astype(int).to_numpy()

        p_up_list = []
        for i in range(len(df_feat)):
            regime = int(regimes[i])
            x_row = X_all.iloc[[i]]
            try:
                proba = self.trainer.predict_with_regime(x_row, regime)
                p_up = float(proba[0, 1])
            except Exception:
                p_up = 0.5
            p_up_list.append(p_up)

        p_up = np.array(p_up_list)
        decision = (p_up >= 0.5).astype(int)

        win = (decision == y_true).astype(int)
        payout = getattr(config, "PAYOUT_30M_PLUS", 0.85)
        pnl = np.where(win == 1, payout, -1.0)
        cum_pnl = np.cumsum(pnl)

        ts_series = (
            pd.to_datetime(df_feat["timestamp"], utc=True, errors="coerce")
            if "timestamp" in df_feat.columns
            else pd.Series([pd.NaT] * len(df_feat))
        )

        result = pd.DataFrame(
            {
                "timestamp": ts_series.to_numpy(),
                "regime": regimes,
                "p_up": p_up,
                "decision": decision,  # 1=UP, 0=DOWN
                "target": y_true,  # 실제 방향
                "win": win,
                "pnl": pnl,
                "cum_pnl": cum_pnl,
            }
        )
        return result


# ==============================
# 진입점
# ==============================
def main():
    import sys

    try:
        if len(sys.argv) < 2:
            print("사용법: python real_trade.py [live|backtest]")
            return
        mode = sys.argv[1]
        if mode == "live":
            rtm = RealTradeManager()
            rtm.run_live()
        elif mode == "backtest":
            from data_loader import DataLoader

            dl = DataLoader()
            df = dl.load_price_data()
            if df.empty:
                print("❌ 데이터 로드 실패")
                return
            btm = BacktestManager()
            df_result = btm.run_backtest(df)
            if not df_result.empty:
                output_path = "backtest_result_30m.csv"
                df_result.to_csv(output_path, index=False)
                print(f"✅ 결과 저장: {output_path}")
        else:
            print(f"❌ 알 수 없는 모드: {mode}")
            print("사용법: python real_trade.py [live|backtest]")
    except KeyboardInterrupt:
        print("\n⏹️  사용자 종료(Ctrl+C)")
    finally:
        print("🛑 종료 정리 완료")


if __name__ == "__main__":
    main()

"""
PSS SCALP - "Will it reach +1R before -1R?"  XGBoost vs CatBoost (walk-forward)

Usage:
    pip install xgboost catboost scikit-learn pandas numpy matplotlib
    python pss_ml_compare.py  PSS_SCALP_XAUUSDm___2026.csv

Label   : y = 1 if the trade touched +1R before its -1R stop  (min_to_1R is not NaN)
Features: ONLY things known at entry (no MFE/MAE/duration/result columns -> no leakage)
Test    : expanding-window walk-forward, chronological, with purging of overlapping trades
Decision: execute the trade when P(+1R before -1R) >= threshold
Outputs : printed report, oof_predictions.csv, equity_compare.png
"""
import sys, warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
SEED = 42
import os
if len(sys.argv) > 1:
    CSV = sys.argv[1]
elif os.path.exists("PSS_SCALP_XAUUSDm 2026 new.csv"):
    CSV = "PSS_SCALP_XAUUSDm 2026 new.csv"
elif os.path.exists("PSS_SCALP_XAUUSDm 2023-2025.csv"):
    CSV = "PSS_SCALP_XAUUSDm 2023-2025.csv"
else:
    CSV = "PSS_SCALP_XAUUSDm___2026.csv"

# ----------------------------------------------------------------------------
# 1. Data + label + entry-time features
# ----------------------------------------------------------------------------
df = pd.read_csv(CSV)
df = df[df.exit_reason != "OPEN"].copy()                 # unfinished trades have no label
df["open_dt"] = pd.to_datetime(df.open_time, format="%Y.%m.%d %H:%M:%S")
df["close_dt"] = pd.to_datetime(df.close_time, format="%Y.%m.%d %H:%M:%S")
df = df.sort_values("open_dt").reset_index(drop=True)

df["y"] = df.min_to_1R.notna().astype(int)               # reached +1R before -1R stop
df["dow"] = df.open_dt.dt.dayofweek                      # 0=Mon
df["spread_to_sl"] = df.spread_pips / df.sl_pips
df["zone_to_sl"] = df.zone_size_pips / df.sl_pips
df["engulf_to_sl"] = df.engulf_depth / df.sl_pips
df["log_body"] = np.log1p(df.body_size_ratio)

NUM = ["sl_pips", "zone_size_pips", "zone_age", "zone_touches", "log_body",
       "engulf_depth", "spread_pips", "spread_to_sl", "zone_to_sl", "engulf_to_sl",
       "open_hour"]
CAT = ["signal_type", "dow"]                             # signal_type: 5 / 42 / 53
FEATS = NUM + CAT
X_all = df[FEATS].copy()

print(f"Trades: {len(df)}   base rate P(+1R before -1R) = {df.y.mean():.3f}")
print(df.groupby("signal").y.agg(["count", "mean"]).round(3).to_string(), "\n")

# ----------------------------------------------------------------------------
# 2. Models (fixed, conservative hyper-parameters - NOT tuned on the test folds)
# ----------------------------------------------------------------------------
def make_models():
    models = {}
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = lambda: XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0,
            eval_metric="logloss", random_state=SEED, n_jobs=1, verbosity=0)
    except ImportError:
        print("!! xgboost not installed - skipping")
    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = lambda: CatBoostClassifier(
            iterations=400, depth=4, learning_rate=0.03, l2_leaf_reg=5.0,
            random_seed=SEED, verbose=0, thread_count=1,
            cat_features=CAT)
    except ImportError:
        print("!! catboost not installed - skipping")
    # reference models, always available
    models["sklearn-HGB (reference)"] = lambda: HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.03, max_iter=200, min_samples_leaf=15,
        l2_regularization=5.0, random_state=SEED)
    models["Logistic (baseline)"] = lambda: make_pipeline(
        StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
    return models

def prep(name, X):
    """XGBoost / HGB / Logistic want numerics (one-hot signal+dow for logistic); CatBoost takes raw cats."""
    X = X.copy()
    if name == "CatBoost":
        for c in CAT:
            X[c] = X[c].astype(str)
        return X
    if name.startswith("Logistic"):
        return pd.get_dummies(X, columns=CAT, dtype=float).reindex(
            columns=_LOGIT_COLS, fill_value=0.0)
    return X.astype(float)

_LOGIT_COLS = pd.get_dummies(X_all, columns=CAT, dtype=float).columns

# ----------------------------------------------------------------------------
# 3. Walk-forward (expanding window, purged)
# ----------------------------------------------------------------------------
N = len(df)
START = int(N * 0.50)                 # first 50% = initial training set
N_FOLDS = 5
edges = np.linspace(START, N, N_FOLDS + 1).astype(int)

models = make_models()
oof = {m: np.full(N, np.nan) for m in models}
train_rate = np.full(N, np.nan)

for k in range(N_FOLDS):
    te = np.arange(edges[k], edges[k + 1])
    test_start = df.open_dt.iloc[te[0]]
    # PURGE: only train on trades that had already CLOSED before the test fold began
    tr = np.where(df.close_dt < test_start)[0]
    tr = tr[tr < edges[k]]
    train_rate[te] = df.y.iloc[tr].mean()
    print(f"fold {k+1}: train={len(tr):3d}  test={len(te):3d}  "
          f"{df.open_dt.iloc[te[0]].date()} -> {df.open_dt.iloc[te[-1]].date()}")
    for name, ctor in models.items():
        m = ctor()
        m.fit(prep(name, X_all.iloc[tr]), df.y.iloc[tr])
        oof[name][te] = m.predict_proba(prep(name, X_all.iloc[te]))[:, 1]

mask = ~np.isnan(train_rate)
T = df[mask].copy()
y = T.y.values

# ----------------------------------------------------------------------------
# 4. Statistical quality of the probabilities
# ----------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
def boot_auc(p, y, B=2000):
    idx = [rng.integers(0, len(y), len(y)) for _ in range(B)]
    a = [roc_auc_score(y[i], p[i]) for i in idx if 0 < y[i].sum() < len(i)]
    return np.percentile(a, [2.5, 97.5])

print("\n" + "=" * 78)
print(f"OUT-OF-SAMPLE  ({mask.sum()} trades, base rate {y.mean():.3f})")
print("=" * 78)
rows = []
for name in models:
    p = oof[name][mask]
    lo, hi = boot_auc(p, y)
    rows.append(dict(model=name, AUC=roc_auc_score(y, p), AUC_lo=lo, AUC_hi=hi,
                     logloss=log_loss(y, p), brier=brier_score_loss(y, p)))
base_p = train_rate[mask]
rows.append(dict(model="(always base-rate)", AUC=0.5, AUC_lo=np.nan, AUC_hi=np.nan,
                 logloss=log_loss(y, base_p), brier=brier_score_loss(y, base_p)))
print(pd.DataFrame(rows).round(4).to_string(index=False))
print("AUC ~0.5 = no skill.  If the CI includes 0.5, the model is not proven better than a coin flip.")

# paired bootstrap XGB vs CatBoost
if "XGBoost" in oof and "CatBoost" in oof:
    px, pc = oof["XGBoost"][mask], oof["CatBoost"][mask]
    d = []
    for _ in range(2000):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            d.append(roc_auc_score(y[i], px[i]) - roc_auc_score(y[i], pc[i]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"AUC(XGB) - AUC(CatBoost) = {np.mean(d):+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")

# ----------------------------------------------------------------------------
# 5. Trading decision: execute only if P >= threshold
#    Two ways to score a skipped/executed trade:
#      (a) REAL    : the EA's real outcome (SL -1R / TP +2R) = result_R
#      (b) 1R-1R   : hypothetical 1R target, 1R stop  -> +1 if y else -1
# ----------------------------------------------------------------------------
def pf(r):
    w, l = r[r > 0].sum(), -r[r < 0].sum()
    return w / l if l > 0 else np.inf

def score(sel, R):
    r = R[sel]
    return dict(kept=int(sel.sum()), pct=100 * sel.mean(), win=100 * (r > 0).mean() if len(r) else np.nan,
                sumR=r.sum(), expR=r.mean() if len(r) else np.nan, PF=pf(r) if len(r) else np.nan)

R_real = T.result_R.values
R_11 = np.where(y == 1, 1.0, -1.0)
print("\nTake-all baseline on same period:")
print("  REAL 2R/1R :", {k: round(v, 2) for k, v in score(np.ones(len(T), bool), R_real).items()})
print("  1R/1R      :", {k: round(v, 2) for k, v in score(np.ones(len(T), bool), R_11).items()})

print("\nPre-declared rules (no peeking at test results):")
for name in models:
    p = oof[name][mask]
    for label, thr in [("P >= train base rate", base_p), ("P >= 0.50", 0.50), ("P >= 0.55", 0.55)]:
        sel = p >= thr
        a, b = score(sel, R_real), score(sel, R_11)
        print(f"  {name:24s} {label:22s} kept {a['kept']:3d} ({a['pct']:4.0f}%) | "
              f"REAL win {a['win']:4.1f}% sumR {a['sumR']:+6.1f} exp {a['expR']:+.2f} PF {a['PF']:.2f} | "
              f"1R/1R sumR {b['sumR']:+6.1f} exp {b['expR']:+.2f}")

print("\nThreshold sweep (REAL outcomes) - for inspection only, picking the best row here is optimistic:")
grid = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
tab = []
for name in models:
    p = oof[name][mask]
    for thr in grid:
        s = score(p >= thr, R_real)
        if s["kept"] >= 10:
            tab.append(dict(model=name, thr=thr, kept=s["kept"], win=s["win"], sumR=s["sumR"], expR=s["expR"], PF=s["PF"]))
print(pd.DataFrame(tab).round(2).to_string(index=False))

# ----------------------------------------------------------------------------
# 6. Per-signal view of the best-AUC GBM at P >= 0.50 (does it help each signal?)
# ----------------------------------------------------------------------------
print("\nPer signal (REAL R), execute if P >= 0.50 vs take-all:")
for name in models:
    p = oof[name][mask]
    for sig, g in T.groupby("signal"):
        ix = np.where(T.signal.values == sig)[0]
        a = score(np.ones(len(ix), bool), R_real[ix])
        b = score(p[ix] >= 0.50, R_real[ix])
        print(f"  {name:24s} {sig:18s} all: n={a['kept']:3d} sumR {a['sumR']:+6.1f} | filtered: n={b['kept']:3d} sumR {b['sumR']:+6.1f}")

# ----------------------------------------------------------------------------
# 7. Feature importance (what the trees actually used)
# ----------------------------------------------------------------------------
print("\nFeature importance (model fitted on all data except the last fold's test block):")
tr_all = np.where(df.close_dt < df.open_dt.iloc[edges[-2]])[0]
for name in ("XGBoost", "CatBoost"):
    if name not in models:
        continue
    m = models[name]()
    m.fit(prep(name, X_all.iloc[tr_all]), df.y.iloc[tr_all])
    imp = (m.get_feature_importance() if name == "CatBoost" else m.feature_importances_)
    s = pd.Series(imp, index=FEATS).sort_values(ascending=False)
    print(f"  {name}: " + ", ".join(f"{k} {v:.2f}" for k, v in (s / s.sum()).head(6).items()))

# ----------------------------------------------------------------------------
# 8. Save outputs
# ----------------------------------------------------------------------------
out = T[["pos_id", "signal", "open_time", "close_time", "y", "result_R"]].copy()
for name in models:
    out[f"p_{name}"] = oof[name][mask]
out.to_csv("oof_predictions.csv", index=False)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(T))
    ax.plot(x, np.cumsum(R_real), label="Take all", color="grey", lw=2)
    for name in models:
        p = oof[name][mask]
        ax.plot(x, np.cumsum(np.where(p >= 0.50, R_real, 0.0)), label=f"{name} (P>=0.50)")
    ax.set_title("Out-of-sample cumulative R (real 2R/1R outcomes), walk-forward")
    ax.set_xlabel("trade # (chronological, OOS only)"); ax.set_ylabel("cumulative R")
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig("equity_compare.png", dpi=130)
    print("\nSaved oof_predictions.csv, equity_compare.png")
except Exception as e:
    print("plot skipped:", e)

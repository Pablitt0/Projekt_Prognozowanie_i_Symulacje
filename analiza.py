# -*- coding: utf-8 -*-
# ============================================================
# PROJEKT: Prognoza zużycia energii elektrycznej w Polsce
# Zadania 2–5 (EDA → Modele → Prognozy X → Prognoza warunkowa)
# ============================================================
# Dane:
#   Polska         – Zuzycie_energii_polska.xlsx      (2004–2024, n=21)
#   Województwa    – Zuzycie_energii_wojewodztwa.xlsx (panel 16×21=336)
#
# Modele:
#   Polska : ln(ZUZYCIE) = β₀ + β₁ln(PKB_pc) + β₂ln(CENA) + β₃HDD
#   FE/woj : ln(ZUZYCIE) = Xβ + Σδₖ·dₖ  (15 efektów stałych)
#
# Metodologia: dane 2004–2023, prognoza ex-ante 2024
# ============================================================

import os, sys, pickle, warnings
warnings.filterwarnings("ignore")

_terminal = len(sys.argv) > 0 and sys.argv[0].endswith(".py")
if _terminal:
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except: pass

import numpy as np
import pandas as pd
import matplotlib
if _terminal: matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from scipy import stats
from scipy.stats import f as f_dist

try:
    from pmdarima import auto_arima
    HAS_PMDARIMA = True
except ImportError:
    HAS_PMDARIMA = False

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

# ── Styl wykresów ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "font.family": "DejaVu Sans",
    "axes.labelsize": 10, "axes.titlesize": 11,
})
BLUE   = "#1a5c96"; RED    = "#c0392b"; GREEN  = "#27ae60"
GRAY   = "#7f8c8d"; ORANGE = "#e67e22"; PURPLE = "#8e44ad"
PALETTE = list(plt.cm.tab20.colors[:16])

TRAIN_END  = 2023          # modele na danych 2004-2023
TEST_YRS   = [2022, 2023]  # lata referencyjne in-sample (wizualizacje)
FC_YR      = 2024          # horyzont prognozy ex-ante

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def save(name):
    plt.savefig(os.path.join(SCRIPT_DIR, name), bbox_inches="tight")
    plt.close()
    print(f"  Zapisano: {name}")

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.1:   return "."
    return ""

def ok(cond): return "OK ✓" if cond else "!"

def measures(y_act, y_hat):
    """Miary jakości – klucze bez %, używane w Z3."""
    ya = np.asarray(y_act, float)
    yh = np.asarray(y_hat, float)
    e  = ya - yh
    mse = (e**2).mean()
    u_num = np.sqrt((e**2).sum())
    u_den = np.sqrt((ya**2).sum()) + np.sqrt((yh**2).sum())
    # Theil U decomposition: UM + UV + UC = 1
    mean_bias = (yh.mean() - ya.mean())**2
    var_diff  = (yh.std(ddof=0) - ya.std(ddof=0))**2
    r = float(np.corrcoef(ya, yh)[0, 1]) if len(ya) > 1 else 0.0
    cov_part  = 2 * (1 - r) * ya.std(ddof=0) * yh.std(ddof=0)
    denom = mean_bias + var_diff + cov_part
    UM = mean_bias / denom if denom else np.nan
    UV = var_diff  / denom if denom else np.nan
    UC = cov_part  / denom if denom else np.nan
    return dict(
        ME=e.mean(), MPE=(e/ya*100).mean(),
        MAE=np.abs(e).mean(), MAPE=(np.abs(e)/ya*100).mean(),
        RMSE=np.sqrt(mse),
        RMSPE=np.sqrt(((e/ya)**2).mean())*100,
        TheilU=u_num/u_den if u_den else np.nan,
        UM=UM, UV=UV, UC=UC,
    )

def miary(y_act, y_hat):
    """Miary jakości – klucze z %, używane w Z5."""
    m = measures(y_act, y_hat)
    return {"ME": m["ME"], "MPE%": m["MPE"], "MAE": m["MAE"],
            "MAPE%": m["MAPE"], "RMSE": m["RMSE"],
            "RMSPE%": m["RMSPE"], "TheilU": m["TheilU"],
            "UM": m["UM"], "UV": m["UV"], "UC": m["UC"]}

def rmspe_fn(y_act, y_hat):
    ya, yh = np.asarray(y_act, float), np.asarray(y_hat, float)
    return np.sqrt(((ya - yh)**2 / ya**2).mean()) * 100

def mape_fn(y_act, y_hat):
    ya, yh = np.asarray(y_act, float), np.asarray(y_hat, float)
    return (np.abs(ya - yh) / ya * 100).mean()

def best_method(res_dict):
    return min(res_dict, key=lambda k: res_dict[k]["rmspe"])

def _X1(t):
    t = np.atleast_1d(np.asarray(t, float)).ravel()
    return np.column_stack([np.ones(len(t)), t])

def _X2(t):
    t = np.atleast_1d(np.asarray(t, float)).ravel()
    return np.column_stack([np.ones(len(t)), t, t**2])

def forecast_all(y_train, t_train, y_test, t_test, t_fc):
    """5 metod prognozy → dict metoda → {pred_test, pred_fc, fitted, rmspe, mape}.
    rmspe/mape liczone na wartościach dopasowanych do pełnej próby uczącej (ex-post 2004–TRAIN_END).
    pred_test – prognoza dla t_test (wizualizacja TEST_YRS).
    pred_fc   – prognoza ex-ante dla t_fc.
    fitted    – wartości dopasowane do próby uczącej (do obliczeń miar jakości).
    """
    y_tr = np.asarray(y_train, float)
    t_tr = np.asarray(t_train, float)
    t_te = np.atleast_1d(np.asarray(t_test, float))
    t_f  = np.atleast_1d(np.asarray(t_fc,  float))
    n_ahead = len(t_te) + len(t_f)
    res = {}

    # 1. OLS liniowy
    m = sm.OLS(y_tr, _X1(t_tr)).fit()
    res["OLS_lin"] = {
        "pred_test": m.predict(_X1(t_te)),
        "pred_fc":   m.predict(_X1(t_f)),
        "fitted":    np.asarray(m.fittedvalues).ravel(),
    }

    # 2. OLS kwadratowy
    m2 = sm.OLS(y_tr, _X2(t_tr)).fit()
    res["OLS_kw"] = {
        "pred_test": m2.predict(_X2(t_te)),
        "pred_fc":   m2.predict(_X2(t_f)),
        "fitted":    np.asarray(m2.fittedvalues).ravel(),
    }

    # 3. ARIMA (auto)
    def _ar1_fallback_fc():
        try:
            from statsmodels.tsa.arima.model import ARIMA
            fit = ARIMA(y_tr, order=(1,0,0)).fit()
            return (np.asarray(fit.forecast(n_ahead)).ravel(),
                    np.asarray(fit.fittedvalues).ravel())
        except Exception:
            from statsmodels.tsa.ar_model import AutoReg
            fit = AutoReg(y_tr, lags=1).fit()
            fc_v = np.asarray(fit.predict(len(y_tr), len(y_tr)+n_ahead-1)).ravel()
            fv   = np.asarray(fit.fittedvalues).ravel()
            pad  = np.full(len(y_tr)-len(fv), float("nan"))
            return fc_v, np.concatenate([pad, fv])

    if HAS_PMDARIMA:
        try:
            am = auto_arima(y_tr, seasonal=False, suppress_warnings=True,
                            error_action="ignore", stepwise=True, max_p=3, max_q=2)
            fc       = np.asarray(am.predict(n_periods=n_ahead)).ravel()
            try:
                fitted_a = np.asarray(am.predict_in_sample()).ravel()
            except Exception:
                fitted_a = y_tr.copy()
        except Exception:
            fc, fitted_a = _ar1_fallback_fc()
    else:
        fc, fitted_a = _ar1_fallback_fc()
    res["ARIMA"] = {"pred_test": fc[:len(t_te)], "pred_fc": fc[len(t_te):], "fitted": fitted_a}

    # 4. Holt
    hm_obj = None
    try:
        hm_obj = SimpleExpSmoothing(y_tr, initialization_method="estimated").fit(
            optimized=True, use_brute=False)
        hf = np.asarray(hm_obj.forecast(n_ahead)).ravel()
        fitted_h = np.asarray(hm_obj.fittedvalues).ravel()
    except Exception:
        try:
            hm_obj = SimpleExpSmoothing(y_tr).fit()
            hf = np.asarray(hm_obj.forecast(n_ahead)).ravel()
            fitted_h = np.asarray(hm_obj.fittedvalues).ravel()
        except Exception:
            fc_fb, fitted_fb = _ar1_fallback_fc()
            hf = fc_fb; fitted_h = fitted_fb
    res["Holt"] = {"pred_test": hf[:len(t_te)], "pred_fc": hf[len(t_te):], "fitted": fitted_h}

    # 5. Pawłowski (ważona regresja liniowa)
    w = np.arange(1, len(y_tr)+1, dtype=float); w /= w.sum()
    mp = sm.WLS(y_tr, _X1(t_tr), weights=w).fit()
    res["Pawl"] = {
        "pred_test": mp.predict(_X1(t_te)),
        "pred_fc":   mp.predict(_X1(t_f)),
        "fitted":    np.asarray(mp.fittedvalues).ravel(),
    }

    for k, v in res.items():
        v["pred_test"] = np.asarray(v["pred_test"]).ravel()
        v["pred_fc"]   = np.asarray(v["pred_fc"]).ravel()
        v["fitted"]    = np.asarray(v["fitted"]).ravel()
        # RMSPE/MAPE na pełnej próbie uczącej (in-sample ex-post 2004–TRAIN_END)
        fit_v = v["fitted"]
        valid  = ~np.isnan(fit_v)
        v["rmspe"] = rmspe_fn(y_tr[valid], fit_v[valid]) if valid.any() else float("inf")
        v["mape"]  = mape_fn(y_tr[valid],  fit_v[valid]) if valid.any() else float("inf")
        v["ms"]    = measures(y_tr[valid], fit_v[valid]) if valid.any() else {}
    return res

# ══════════════════════════════════════════════════════════════
# WCZYTANIE I PRZYGOTOWANIE DANYCH (raz dla wszystkich zadań)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("PROJEKT – Prognoza zużycia energii elektrycznej w Polsce")
print("=" * 60)

# ── Polska ────────────────────────────────────────────────────
df_p = pd.read_excel(os.path.join(SCRIPT_DIR, "Zuzycie_energii_polska.xlsx"))
df_p = df_p.sort_values("rok").reset_index(drop=True)
df_p["pkb_per_capita"] = df_p["pkb_mln_zl"] * 1e6 / df_p["ludnosc"]
df_p["ln_zuzycie"]     = np.log(df_p["zuzycie_energii_GWh"])
df_p["ln_pkb_pc"]      = np.log(df_p["pkb_per_capita"])
df_p["ln_cena"]        = np.log(df_p["cena_energii_zl_kWh"])

dp_tr  = df_p[df_p["rok"] <= TRAIN_END].copy()    # 2004-2023, n=20
dp_te  = df_p[df_p["rok"].isin(TEST_YRS)].copy()  # 2022-2023, in-sample (Z4 + wizualizacje)
t_tr_p = dp_tr["rok"].values
t_te_p = dp_te["rok"].values
t_fc   = np.array([FC_YR])

# ── Województwa ───────────────────────────────────────────────
df_w = pd.read_excel(os.path.join(SCRIPT_DIR, "Zuzycie_energii_wojewodztwa.xlsx"))
df_w = df_w.sort_values(["wojewodztwo", "rok"]).reset_index(drop=True)
mask0 = df_w["dochod_os"] <= 0
if mask0.any():
    df_w.loc[mask0, "dochod_os"] = np.nan
    df_w["dochod_os"] = (df_w.groupby("wojewodztwo")["dochod_os"]
                            .transform(lambda s: s.interpolate(
                                method="linear", limit=3, limit_direction="both")))
df_w["ln_zuzycie"]        = np.log(df_w["zuzycie_energii_GWh"])
df_w["ln_dochod_os"]      = np.log(df_w["dochod_os"])
df_w["ln_cena"]           = np.log(df_w["cena_energii_zl_kWh"])
df_w["ln_dochod_os_lag1"] = df_w.groupby("wojewodztwo")["ln_dochod_os"].shift(1)

PROV = sorted(df_w["wojewodztwo"].unique())
dw_tr = df_w[(df_w["rok"] > 2004) & (df_w["rok"] <= TRAIN_END)].copy()  # 2005-2023
dw_te = df_w[df_w["rok"].isin(TEST_YRS)].copy()  # 2022-2023, in-sample

X_P  = ["ln_pkb_pc", "ln_cena", "hdd"]
X_FE = ["ln_dochod_os_lag1", "ln_cena", "urbanizacja_pct", "liczba_os", "pow_os", "hdd"]

print(f"Polska : {len(df_p)} obs ({int(df_p.rok.min())}–{int(df_p.rok.max())}), "
      f"model={len(dp_tr)} (2004–{TRAIN_END}), FC={FC_YR}")
print(f"Woj.   : {len(df_w)} obs | {len(PROV)} woj × {df_w.rok.nunique()} lat, "
      f"model={len(dw_tr)} (2005–{TRAIN_END})")


# ╔══════════════════════════════════════════════════════════════╗
# ║              ZADANIE 2 – ANALIZA OPISOWA                    ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("ZADANIE 2 – ANALIZA OPISOWA DANYCH")
print("=" * 60)

VAR_INFO = [
    ("ZUZYCIE",     "Zużycie energii elektrycznej",        "GWh",       "GUS / URE",       "Y – zm. objaśniana"),
    ("PKB_pc",      "PKB per capita",                      "PLN/os",    "GUS BDL",         "X – model Polska"),
    ("CENA",        "Cena energii elektrycznej",           "PLN/kWh",   "URE / GUS",       "X – oba modele"),
    ("HDD",         "Stopniodni grzewcze",                 "°C·dzień", "IMGW / Eurostat", "X – oba modele"),
    ("CDD",         "Stopniodni chłodnicze",               "°C·dzień", "IMGW / Eurostat", "X – usunięte (p>0.5)"),
    ("DOCHOD_OS",   "Dochód rozp. na osobę",               "PLN/os",    "GUS BDL",         "X – model FE woj."),
    ("URBANIZACJA", "Stopień urbanizacji",                 "%",         "GUS BDL",         "X – model FE woj."),
    ("LICZBA_OS",   "Śr. liczba osób w gosp. dom.",        "osoby",     "GUS",             "X – model FE woj."),
    ("POW_OS",      "Pow. użytkowa mieszkania na os.",     "m²/os",     "GUS BDL",         "X – model FE woj."),
    ("LUDNOSC",     "Ludność ogółem",                      "os.",       "GUS BDL",         "kontrolna"),
]

print(f"\n  {'Symbol':<14} {'Nazwa':<40} {'Jedn.':<12} {'Źródło':<18} Rola")
print("  " + "─" * 96)
for row in VAR_INFO:
    print(f"  {row[0]:<14} {row[1]:<40} {row[2]:<12} {row[3]:<18} {row[4]}")

# PNG 08 – Opis zmiennych (tabela)
fig, ax = plt.subplots(figsize=(18, 5))
ax.axis("off")
tbl = ax.table(cellText=[list(r) for r in VAR_INFO],
               colLabels=["Symbol", "Nazwa zmiennej", "Jednostka", "Źródło", "Rola w modelu"],
               cellLoc="left", loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
tbl.auto_set_column_width([0,1,2,3,4])
for j in range(5):
    tbl[0,j].set_facecolor("#1a5c96"); tbl[0,j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(VAR_INFO)+1):
    bg = "#f0f5ff" if i % 2 == 0 else "white"
    for j in range(5): tbl[i,j].set_facecolor(bg)
ax.set_title("Opis zmiennych – Prognoza zużycia energii elektrycznej w Polsce",
             fontsize=12, fontweight="bold", pad=12)
save("z2_08_opis_zmiennych.png")

# PNG 09 – Statystyki opisowe
cols_p = [c for c in ["zuzycie_energii_GWh","pkb_per_capita","cena_energii_zl_kWh","hdd","cdd","ludnosc"] if c in df_p.columns]
cols_w = [c for c in ["zuzycie_energii_GWh","dochod_os","cena_energii_zl_kWh","urbanizacja_pct","liczba_os","pow_os","hdd"] if c in df_w.columns]
desc_p = df_p[cols_p].describe().round(2)
desc_w = df_w[cols_w].describe().round(2)
print("\nStatystyki Polska:\n", desc_p.to_string())
print("\nStatystyki Województwa:\n", desc_w.to_string())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 4))
for ax, desc, title in [(ax1, desc_p, "Polska"), (ax2, desc_w, "Województwa (panel)")]:
    ax.axis("off")
    t = ax.table(cellText=desc.round(2).astype(str).values,
                 rowLabels=desc.index,
                 colLabels=[c.replace("_","\n") for c in desc.columns],
                 cellLoc="center", loc="center", bbox=[0,0,1,1])
    t.auto_set_font_size(False); t.set_fontsize(8)
    for j in range(len(desc.columns)):
        t[0,j].set_facecolor("#1a5c96"); t[0,j].set_text_props(color="white", fontweight="bold")
    ax.set_title(f"Statystyki opisowe – {title}", fontsize=11, fontweight="bold", pad=6)
plt.tight_layout()
save("z2_09_statystyki.png")

# PNG 01 – Szereg czasowy Polska
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("Zmienne modelu – Polska (2004–2024)", fontsize=13, fontweight="bold")
for ax, (col, label, color) in zip(axes.flat, [
    ("zuzycie_energii_GWh", "Zużycie energii elektrycznej [GWh]", BLUE),
    ("pkb_per_capita",       "PKB per capita [PLN/os.]",           GREEN),
    ("cena_energii_zl_kWh", "Cena energii elektrycznej [PLN/kWh]", RED),
    ("hdd",                  "Stopniodni grzewcze (HDD)",           GRAY),
]):
    ax.plot(df_p["rok"], df_p[col], "o-", color=color, lw=2.2, ms=6)
    ax.fill_between(df_p["rok"], df_p[col], alpha=0.08, color=color)
    ax.set_title(label, fontweight="bold"); ax.set_xlabel("Rok")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
plt.tight_layout()
save("z2_01_polska_szereg.png")

# PNG 02 – Szeregi województwa 4×4
fig, axes = plt.subplots(4, 4, figsize=(22, 16), sharex=True)
fig.suptitle("Zużycie energii elektrycznej per województwo (2004–2024)", fontsize=14, fontweight="bold")
for i, (ax, prov) in enumerate(zip(axes.flat, PROV)):
    dp = df_w[df_w["wojewodztwo"] == prov]
    ax.plot(dp["rok"], dp["zuzycie_energii_GWh"], "o-", color=PALETTE[i], lw=2, ms=4)
    ax.set_title(prov, fontsize=9, fontweight="bold")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1e3:.0f}k"))
    if i % 4 == 0: ax.set_ylabel("GWh")
    if i >= 12:    ax.set_xlabel("Rok")
plt.tight_layout()
save("z2_02_woj_szeregi.png")

# PNG 03 – Porównanie województw (2024)
df_2024 = df_w[df_w["rok"]==2024].sort_values("zuzycie_energii_GWh", ascending=True)
fig, ax = plt.subplots(figsize=(12, 7))
bars = ax.barh(df_2024["wojewodztwo"], df_2024["zuzycie_energii_GWh"], color=BLUE, alpha=0.85, edgecolor="white")
for bar, val in zip(bars, df_2024["zuzycie_energii_GWh"]):
    ax.text(val+30, bar.get_y()+bar.get_height()/2, f"{val:,.0f}", va="center", fontsize=9)
ax.set_xlabel("Zużycie energii elektrycznej [GWh]")
ax.set_title("Zużycie energii elektrycznej per województwo (2024)", fontsize=12, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
plt.tight_layout()
save("z2_03_woj_porownanie.png")

# PNG 04 – Korelacja Polska
corr_cols_p = [c for c in [
    "zuzycie_energii_GWh",
    "pkb_per_capita",
    "cena_energii_zl_kWh",
    "HDD",
    "dochod_os",
    "urbanizacja_pct",
    "liczba_os",
    "pow_os"
] if c in df_p.columns]

labels_map = {
    "zuzycie_energii_GWh":  "Zużycie\n[GWh]",
    "PKB_pc":               "PKB\npc",
    "cena_energii_zl_kWh":  "Cena\n[PLN/kWh]",
    "HDD":                  "HDD",
    "dochod_os":            "Dochód\nos.",
    "urbanizacja_pct":      "Urbanizacja\n[%]",
    "liczba_os":            "Liczba\nos.",
    "pow_os":               "Pow.\nos."
}

labels_cor_p = [labels_map[c] for c in corr_cols_p]

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df_p[corr_cols_p].corr(), annot=True, fmt=".2f", cmap="RdBu_r",
            vmin=-1, vmax=1, ax=ax, linewidths=0.5, annot_kws={"size":10})
ax.set_xticklabels(labels_cor_p, rotation=30, ha="right")
ax.set_yticklabels(labels_cor_p, rotation=0)
ax.set_title("Macierz korelacji Pearsona – Polska (2004–2024)", fontsize=12, fontweight="bold")
plt.tight_layout()
save("z2_04_korelacja_polska.png")

# PNG 05 – Scatter Polska
scatter_p = [(c,l,col) for c,l,col in [
    ("pkb_per_capita","PKB per capita [PLN/os.]",GREEN),
    ("cena_energii_zl_kWh","Cena energii [PLN/kWh]",RED),
    ("hdd","Stopniodni grzewcze (HDD)",BLUE),
    ("cdd","Stopniodni chłodnicze (CDD)",GRAY),
] if c in df_p.columns]
fig, axes = plt.subplots(1, len(scatter_p), figsize=(5*len(scatter_p), 5))
fig.suptitle("Zależność zużycia energii od zmiennych objaśniających – Polska", fontsize=12, fontweight="bold")
if len(scatter_p) == 1: axes = [axes]
for ax, (col, xlabel, color) in zip(axes, scatter_p):
    tmp = df_p[["zuzycie_energii_GWh", col]].dropna()
    ax.scatter(tmp[col], tmp["zuzycie_energii_GWh"], color=color, s=70, alpha=0.75, edgecolors="none")
    xr = np.linspace(tmp[col].min(), tmp[col].max(), 100)
    ax.plot(xr, np.poly1d(np.polyfit(tmp[col], tmp["zuzycie_energii_GWh"], 1))(xr), "--k", lw=1.5)
    r, pv = stats.pearsonr(tmp[col], tmp["zuzycie_energii_GWh"])
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel("Zużycie energii [GWh]", fontsize=9)
    ax.set_title(f"r = {r:.3f}  (p = {pv:.3f})", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
plt.tight_layout()
save("z2_05_scatter_polska.png")


# PNG 07 – Scatter Województwa
scatter_w = [(c,l,col) for c,l,col in [
    ("dochod_os","Dochód na osobę [PLN]","#2ecc71"),
    ("cena_energii_zl_kWh","Cena energii [PLN/kWh]",RED),
    ("urbanizacja_pct","Urbanizacja [%]","#9b59b6"),
    ("liczba_os","Liczba osób w gosp.",BLUE),
    ("pow_os","Pow. mieszk. na os. [m²]",GRAY),
    ("hdd","HDD",ORANGE),
] if c in df_w.columns]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Zależność zużycia energii od zmiennych objaśniających – panel województw", fontsize=12, fontweight="bold")
for ax, (col, xlabel, color) in zip(axes.flat, scatter_w):
    tmp = df_w[["zuzycie_energii_GWh", col]].dropna()
    ax.scatter(tmp[col], tmp["zuzycie_energii_GWh"], color=color, s=20, alpha=0.35, edgecolors="none")
    xr = np.linspace(tmp[col].min(), tmp[col].max(), 100)
    ax.plot(xr, np.poly1d(np.polyfit(tmp[col], tmp["zuzycie_energii_GWh"], 1))(xr), "--k", lw=1.8)
    r, pv = stats.pearsonr(tmp[col], tmp["zuzycie_energii_GWh"])
    ax.set_xlabel(xlabel, fontsize=9); ax.set_ylabel("Zużycie energii [GWh]", fontsize=9)
    ax.set_title(f"r = {r:.3f}  (p = {pv:.3f})", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
plt.tight_layout()
save("z2_07_scatter_woj.png")

# Podsumowanie Z2
print(f"\n  Zużycie 2004: {df_p.loc[df_p.rok==2004,'zuzycie_energii_GWh'].values[0]:,.0f} GWh")
print(f"  Zużycie 2024: {df_p.loc[df_p.rok==2024,'zuzycie_energii_GWh'].values[0]:,.0f} GWh")
zmiana = (df_p.loc[df_p.rok==2024,'zuzycie_energii_GWh'].values[0] /
          df_p.loc[df_p.rok==2004,'zuzycie_energii_GWh'].values[0] - 1) * 100
print(f"  Zmiana 2004→2024: {zmiana:+.1f}%")
print("  Z2: pliki z2_01..z2_09_*.png gotowe")


# ╔══════════════════════════════════════════════════════════════╗
# ║           ZADANIE 3 – MODELE EKONOMETRYCZNE                 ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("ZADANIE 3 – MODELE EKONOMETRYCZNE")
print("=" * 60)

# ── Model Polska – OLS ────────────────────────────────────────
y_tr_p = dp_tr["ln_zuzycie"].values
X_tr_p = sm.add_constant(dp_tr[X_P].values)
model_p = sm.OLS(y_tr_p, X_tr_p).fit()
print("\n[Model Polska]")
print(model_p.summary())

vif_p = pd.DataFrame({"Zmienna": ["const"]+X_P,
    "VIF": [variance_inflation_factor(X_tr_p,i) for i in range(X_tr_p.shape[1])]})
print("\nVIF:\n", vif_p.to_string(index=False))

dw_p  = durbin_watson(model_p.resid)
bg_lm_p, bg_p,  _, _ = acorr_breusch_godfrey(model_p, nlags=2)
sw_s_p, sw_p_p        = stats.shapiro(model_p.resid)
bp_lm_p, bp_p_p, _, _ = het_breuschpagan(model_p.resid, X_tr_p)
print(f"DW={dw_p:.4f}  BG p={bg_p:.4f}  SW p={sw_p_p:.4f}  BP p={bp_p_p:.4f}")

# ── Dodatkowe testy – Model Polska ───────────────────────────

# Ljung-Box (lag=2 max przy n=21)
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_p = acorr_ljungbox(model_p.resid, lags=[1, 2], return_df=True)

# Jarque-Bera – pomijamy dla n=21, moc za niska

# RESET – forma funkcyjna
from statsmodels.stats.diagnostic import linear_reset
reset_p = linear_reset(model_p, power=2, use_f=True)

# White – pomijamy dla n=21, za mało obserwacji

# ADF – stacjonarność zmiennej zależnej
from statsmodels.tsa.stattools import adfuller
adf_p = adfuller(dp_tr["ln_zuzycie"], maxlag=2, autolag=None)

# Condition number – multikolinearność
cond_p = np.linalg.cond(X_tr_p)

print(f"\n  Ljung-Box lag=1 p={lb_p['lb_pvalue'].iloc[0]:.4f}  lag=2 p={lb_p['lb_pvalue'].iloc[1]:.4f}")
print(f"  RESET p={reset_p.pvalue:.4f}  {'forma OK ✓' if reset_p.pvalue>0.05 else 'błędna specyfikacja!'}")
print(f"  ADF p={adf_p[1]:.4f}  {'stacjonarny ✓' if adf_p[1]<0.05 else 'niestacjonarny – wynik orientacyjny (n=21)'}")
print(f"  Condition number={cond_p:.1f}  {'OK ✓' if cond_p<1000 else 'wysoka multikolinearność!'}")

X_te_p = pd.DataFrame({"const":np.ones(len(dp_te)),
    "ln_pkb_pc":dp_te["ln_pkb_pc"].values,
    "ln_cena":dp_te["ln_cena"].values, "hdd":dp_te["hdd"].values})
y_hat_te_p = np.exp(np.asarray(model_p.predict(X_te_p)).ravel())
y_act_te_p = dp_te["zuzycie_energii_GWh"].values
ms_p_te = measures(y_act_te_p, y_hat_te_p)
ms_p_tr = measures(np.exp(y_tr_p), np.exp(model_p.fittedvalues))
print(f"RMSPE% in-sample (2004–{TRAIN_END})={ms_p_tr['RMSPE']:.2f}%")

# PNG 01 – Koeficjenty + miary
coef_rows_p = [[lbl, f"{model_p.params[i]:.4f}", f"{model_p.bse[i]:.4f}",
                f"{model_p.tvalues[i]:.3f}", f"{model_p.pvalues[i]:.4f}", sig_stars(model_p.pvalues[i])]
               for i, lbl in enumerate(["Stała","ln(PKB_pc)","ln(CENA)","HDD"])]
fit_rows_p = [["R²",f"{model_p.rsquared:.4f}"],["R² skoryg.",f"{model_p.rsquared_adj:.4f}"],
              ["AIC",f"{model_p.aic:.2f}"],["BIC",f"{model_p.bic:.2f}"],
              ["F-stat",f"{model_p.fvalue:.2f}"],["p(F)",f"{model_p.f_pvalue:.6f}"],
              ["N",f"{len(dp_tr)}"],
              [f"RMSPE% in-sample (2004–{TRAIN_END})",f"{ms_p_tr['RMSPE']:.2f}%"]]
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(18,5))
fig.suptitle("Model Polska:  ln(ZUZYCIE) = β₀ + β₁·ln(PKB_pc) + β₂·ln(CENA) + β₃·HDD", fontsize=12, fontweight="bold")
ax1.axis("off")
t1=ax1.table(cellText=coef_rows_p,colLabels=["Zmienna","β","SE","t","p-value","Ist."],cellLoc="center",loc="center",bbox=[0,0,1,1])
t1.auto_set_font_size(False);t1.set_fontsize(11)
for j in range(6): t1[0,j].set_facecolor("#1a5c96");t1[0,j].set_text_props(color="white",fontweight="bold")
for i,row in enumerate(coef_rows_p,1):
    bg="#f0f5ff" if i%2==0 else "white"
    for j in range(6): t1[i,j].set_facecolor(bg)
    if row[5]=="***": t1[i,5].set_facecolor("#c8e6c9")
    elif row[5] in ("**","*"): t1[i,5].set_facecolor("#fff9c4")
ax1.set_title("Współczynniki OLS",fontsize=11,fontweight="bold",pad=6)
ax2.axis("off")
t2=ax2.table(cellText=fit_rows_p,colLabels=["Miara","Wartość"],cellLoc="center",loc="center",bbox=[0.15,0,0.7,1])
t2.auto_set_font_size(False);t2.set_fontsize(11)
for j in range(2): t2[0,j].set_facecolor("#1a5c96");t2[0,j].set_text_props(color="white",fontweight="bold")
for i in range(1,len(fit_rows_p)+1):
    bg="#f0f5ff" if i%2==0 else "white"; t2[i,0].set_facecolor(bg);t2[i,1].set_facecolor(bg)
t2[len(fit_rows_p),1].set_facecolor("#c8e6c9" if ms_p_tr['RMSPE']<=10 else "#ffcdd2")
ax2.set_title("Miary dopasowania",fontsize=11,fontweight="bold",pad=6)
plt.tight_layout()
save("z3_01_model_polska_koef.png")

# PNG 02 – Diagnostyka reszt Polska
fig, axes = plt.subplots(2,3,figsize=(18,10))
fig.suptitle("Model Polska – diagnostyka reszt",fontsize=13,fontweight="bold")
resid_p=model_p.resid; fitted_p=model_p.fittedvalues; yrs_tr=dp_tr["rok"].values
ax=axes[0,0]; ax.plot(yrs_tr,resid_p,"o-",color=BLUE,lw=1.8,ms=6); ax.axhline(0,color=RED,lw=1.5,ls="--")
ax.set_title("Reszty w czasie"); ax.set_xlabel("Rok"); ax.set_ylabel("eₜ"); ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
ax=axes[0,1]; ax.scatter(fitted_p,resid_p,color=BLUE,s=70,alpha=0.8); ax.axhline(0,color=RED,lw=1.5,ls="--")
ax.set_title("Reszty vs dopasowane"); ax.set_xlabel("Ŷ"); ax.set_ylabel("eₜ")
ax=axes[0,2]; ax.hist(resid_p,bins=8,color=BLUE,alpha=0.7,edgecolor="white",density=True)
xr=np.linspace(resid_p.min(),resid_p.max(),100); ax.plot(xr,stats.norm.pdf(xr,resid_p.mean(),resid_p.std()),"r-",lw=2)
ax.set_title("Histogram reszt"); ax.set_xlabel("eₜ")
ax=axes[1,0]
(osm,osr),(slope,intercept,_)=stats.probplot(resid_p,dist="norm")
ax.scatter(osm,osr,color=BLUE,s=60,alpha=0.85); ax.plot(osm,slope*np.array(osm)+intercept,"r-",lw=2)
ax.set_title(f"QQ-plot  SW p={sw_p_p:.3f}")
ax=axes[1,1]; max_lag=min(10,len(resid_p)//2)
acf_v=[pd.Series(resid_p).autocorr(lag=l) for l in range(1,max_lag)]
ax.bar(range(1,max_lag),acf_v,color=BLUE,alpha=0.7)
ci95=1.96/np.sqrt(len(resid_p)); ax.axhline(ci95,color=RED,ls="--",lw=1.5); ax.axhline(-ci95,color=RED,ls="--",lw=1.5)
ax.axhline(0,color="black",lw=0.8); ax.set_title(f"ACF reszt  DW={dw_p:.3f}")
ax=axes[1,2]; ax.axis("off")
test_rows_p=[
    ["Durbin-Watson",            f"{dw_p:.4f}",   ok(1.5<dw_p<2.5)],
    ["Breusch-Godfrey p",        f"{bg_p:.4f}",   ok(bg_p>0.05)],
    ["Ljung-Box lag=1 p",        f"{lb_p['lb_pvalue'].iloc[0]:.4f}", ok(lb_p['lb_pvalue'].iloc[0]>0.05)],
    ["Shapiro-Wilk p",           f"{sw_p_p:.4f}", ok(sw_p_p>0.05)],
    ["Breusch-Pagan p",          f"{bp_p_p:.4f}", ok(bp_p_p>0.05)],
    ["RESET p",                  f"{reset_p.pvalue:.4f}", ok(reset_p.pvalue>0.05)],
    ["ADF p (orientacyjny n=21)",f"{adf_p[1]:.4f}", ok(adf_p[1]<0.05)],
    ["Condition number",         f"{cond_p:.1f}", ok(cond_p<1000)],
]
t3=ax.table(cellText=test_rows_p,colLabels=["Test","Wartość","Wniosek"],cellLoc="center",loc="center",bbox=[0,0.1,1,0.75])
t3.auto_set_font_size(False);t3.set_fontsize(10)
for j in range(3): t3[0,j].set_facecolor("#1a5c96");t3[0,j].set_text_props(color="white",fontweight="bold")
for i,row in enumerate(test_rows_p,1):
    bg="#c8e6c9" if row[2].startswith("OK") else "#ffcdd2"
    for j in range(3): t3[i,j].set_facecolor(bg)
ax.set_title("Testy diagnostyczne",fontsize=11,fontweight="bold",pad=6)
plt.tight_layout(); save("z3_02_model_polska_diag.png")

# PNG 03 – Dopasowanie + krocząca ex-post Polska
X_full_p=pd.DataFrame({"const":np.ones(len(df_p)),"ln_pkb_pc":df_p["ln_pkb_pc"].values,
                        "ln_cena":df_p["ln_cena"].values,"hdd":df_p["hdd"].values})
y_hat_full_p=np.exp(np.asarray(model_p.predict(X_full_p)).ravel())
# PNG 03 – Dopasowanie modelu Polska
fig,ax=plt.subplots(figsize=(14,6))
ax.plot(df_p[df_p["rok"]<=TRAIN_END]["rok"],df_p[df_p["rok"]<=TRAIN_END]["zuzycie_energii_GWh"],
        "o-",color=BLUE,lw=2.2,ms=7,zorder=5,label=f"Rzeczywiste (2004–{TRAIN_END})")
ax.plot(dp_tr["rok"].values,y_hat_full_p[:len(dp_tr)],
        "s--",color=GREEN,lw=1.8,ms=5,alpha=0.85,
        label=f"Dopasowane in-sample  RMSPE={ms_p_tr['RMSPE']:.1f}%")
ax.fill_between(dp_tr["rok"].values,
                dp_tr["zuzycie_energii_GWh"].values,
                y_hat_full_p[:len(dp_tr)],
                alpha=0.10,color=GREEN)
ax.axvline(TRAIN_END+0.5,color=GRAY,ls=":",lw=1.8,alpha=0.7)
ax.set_title(f"Model Polska – dopasowanie in-sample (2004–{TRAIN_END})",fontsize=11,fontweight="bold")
ax.set_xlabel("Rok"); ax.set_ylabel("Zużycie energii [GWh]")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
ax.legend(loc="upper left"); ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
plt.tight_layout(); save("z3_03_model_polska_fit.png")

# ── Model FE – Województwa ────────────────────────────────────
print("\n[Model FE – Województwa]")
dummies_tr = pd.get_dummies(dw_tr["wojewodztwo"], drop_first=True).astype(float)
dum_cols   = list(dummies_tr.columns)
ref_prov   = [p for p in PROV if p not in dum_cols][0]
assert dummies_tr.values.min()==0 and dummies_tr.values.max()==1, "Dummies nie są 0/1!"
print(f"Ref: {ref_prov} | Dummies: {len(dum_cols)} (0/1 ✓)")

y_tr_w = dw_tr["ln_zuzycie"].values
X_vars_tr = dw_tr[X_FE].values
X_tr_w = np.column_stack([np.ones(len(dw_tr)), X_vars_tr, dummies_tr.values])
model_fe = sm.OLS(y_tr_w, X_tr_w).fit()
print(model_fe.summary())

n_main = 1+len(X_FE)
vif_fe = pd.DataFrame({"Zmienna": ["const"]+X_FE,
    "VIF": [variance_inflation_factor(X_tr_w,i) for i in range(n_main)]})
print("\nVIF (główne):\n", vif_fe.to_string(index=False))

dw_fe  = durbin_watson(model_fe.resid)
bg_lm_fe, bg_fe,  _, _ = acorr_breusch_godfrey(model_fe, nlags=2)
sw_s_fe, sw_p_fe        = stats.shapiro(model_fe.resid)
bp_lm_fe, bp_p_fe, _, _ = het_breuschpagan(model_fe.resid, X_tr_w)
print(f"DW={dw_fe:.4f}  BG p={bg_fe:.4f}  SW p={sw_p_fe:.4f}  BP p={bp_p_fe:.4f}")

# ── Dodatkowe testy – Model FE ────────────────────────────────

# Ljung-Box (lag=4 OK przy n=288)
lb_fe = acorr_ljungbox(model_fe.resid, lags=[1, 2, 4], return_df=True)

# Jarque-Bera (wiarygodny przy n=288)
from statsmodels.stats.stattools import jarque_bera
jb_stat_fe, jb_p_fe, jb_skew_fe, jb_kurt_fe = jarque_bera(model_fe.resid)

# White – tylko na głównych zmiennych, bez dummy
from statsmodels.stats.diagnostic import het_white
white_fe_stat, white_fe_p, _, _ = het_white(model_fe.resid, X_tr_w[:, :n_main])

# RESET – orientacyjny przy panelu z dummy
reset_fe = linear_reset(model_fe, power=2, use_f=True)

# Condition number – cała macierz X
cond_fe = np.linalg.cond(X_tr_w)
# Condition number – tylko główne zmienne (bez dummy)
cond_fe_main = np.linalg.cond(X_tr_w[:, :n_main])

print(f"\n  Ljung-Box lag=1 p={lb_fe['lb_pvalue'].iloc[0]:.4f}  lag=2 p={lb_fe['lb_pvalue'].iloc[1]:.4f}  lag=4 p={lb_fe['lb_pvalue'].iloc[2]:.4f}")
print(f"  Jarque-Bera p={jb_p_fe:.4f}  skewness={jb_skew_fe:.3f}  kurtosis={jb_kurt_fe:.3f}")
print(f"  White (bez dummy) p={white_fe_p:.4f}  {'homoskedastyczność ✓' if white_fe_p>0.05 else 'heteroskedastyczność!'}")
print(f"  RESET p={reset_fe.pvalue:.4f}  {'forma OK ✓' if reset_fe.pvalue>0.05 else 'błędna specyfikacja (orientacyjny przy panelu)'}")
print(f"  Condition number (pełny)={cond_fe:.1f}  (główne zmienne)={cond_fe_main:.1f}")

model_pool = sm.OLS(y_tr_w, np.column_stack([np.ones(len(dw_tr)), X_vars_tr])).fit()
q    = len(dum_cols)
f_fe = ((model_pool.ssr-model_fe.ssr)/q)/(model_fe.ssr/model_fe.df_resid)
p_fe = 1-f_dist.cdf(f_fe, q, model_fe.df_resid)
print(f"F-test FE: F({q},{int(model_fe.df_resid)})={f_fe:.3f}  p={p_fe:.6f}  → {'FE istotne ✓' if p_fe<0.05 else 'FE nieistotne'}")

dummies_te = pd.get_dummies(dw_te["wojewodztwo"]).astype(float)
for c in dum_cols:
    if c not in dummies_te.columns: dummies_te[c] = 0.0
dummies_te = dummies_te[dum_cols]
X_te_w = np.column_stack([np.ones(len(dw_te)), dw_te[X_FE].values, dummies_te.values])
y_hat_te_w  = np.exp(np.asarray(model_fe.predict(X_te_w)).ravel())
y_act_te_w  = dw_te["zuzycie_energii_GWh"].values
ms_fe_te = measures(y_act_te_w, y_hat_te_w)
ms_fe_tr = measures(np.exp(y_tr_w), np.exp(model_fe.fittedvalues))

prov_rmspe = {}
for prov in PROV:
    mask = dw_te["wojewodztwo"].values==prov
    if mask.sum()==0: continue
    ya=y_act_te_w[mask]; yh=y_hat_te_w[mask]
    prov_rmspe[prov] = np.sqrt(((ya-yh)**2/ya**2).mean())*100
good_fe = sum(1 for v in prov_rmspe.values() if v<=10)
print(f"RMSPE% in-sample (2005–{TRAIN_END})={ms_fe_tr['RMSPE']:.2f}%")

# PNG 04 – Koeficjenty FE
coef_rows_fe = [[lbl, f"{model_fe.params[i]:.4f}", f"{model_fe.bse[i]:.4f}",
                 f"{model_fe.tvalues[i]:.3f}", f"{model_fe.pvalues[i]:.4f}", sig_stars(model_fe.pvalues[i])]
                for i,lbl in enumerate(["Stała","ln(Dochód_os lag1)","ln(CENA)","Urbanizacja [%]","Liczba os.","Pow./os.","HDD"])]
fit_rows_fe = [["R²",f"{model_fe.rsquared:.4f}"],["R² skoryg.",f"{model_fe.rsquared_adj:.4f}"],
               ["AIC",f"{model_fe.aic:.2f}"],["BIC",f"{model_fe.bic:.2f}"],
               ["F-stat",f"{model_fe.fvalue:.2f}"],["p(F)",f"{model_fe.f_pvalue:.6f}"],
               ["N",f"{len(dw_tr)}"],["F-test dummies (p)",f"{p_fe:.6f}"],
               [f"RMSPE% in-sample (2005–{TRAIN_END})",f"{ms_fe_tr['RMSPE']:.2f}%"],
               [f"Woj. RMSPE≤10% (in-sample)",f"{good_fe}/{len(prov_rmspe)}"]]
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(18,5))
fig.suptitle("Model FE – Województwa:  ln(ZUZYCIE) = Xβ + Σδₖ·dₖ  (15 efektów stałych)",fontsize=12,fontweight="bold")
ax1.axis("off")
t1=ax1.table(cellText=coef_rows_fe,colLabels=["Zmienna","β","SE","t","p-value","Ist."],cellLoc="center",loc="center",bbox=[0,0,1,1])
t1.auto_set_font_size(False);t1.set_fontsize(10)
for j in range(6): t1[0,j].set_facecolor("#1a5c96");t1[0,j].set_text_props(color="white",fontweight="bold")
for i,row in enumerate(coef_rows_fe,1):
    bg="#f0f5ff" if i%2==0 else "white"
    for j in range(6): t1[i,j].set_facecolor(bg)
    if row[5]=="***": t1[i,5].set_facecolor("#c8e6c9")
    elif row[5] in ("**","*"): t1[i,5].set_facecolor("#fff9c4")
ax1.set_title("Współczynniki (zmienne główne)",fontsize=11,fontweight="bold",pad=6)
ax2.axis("off")
t2=ax2.table(cellText=fit_rows_fe,colLabels=["Miara","Wartość"],cellLoc="center",loc="center",bbox=[0.1,0,0.8,1])
t2.auto_set_font_size(False);t2.set_fontsize(10)
for j in range(2): t2[0,j].set_facecolor("#1a5c96");t2[0,j].set_text_props(color="white",fontweight="bold")
for i in range(1,len(fit_rows_fe)+1):
    bg="#f0f5ff" if i%2==0 else "white"; t2[i,0].set_facecolor(bg);t2[i,1].set_facecolor(bg)
ax2.set_title("Miary dopasowania",fontsize=11,fontweight="bold",pad=6)
plt.tight_layout(); save("z3_04_model_fe_koef.png")

# PNG 05 – Diagnostyka FE
fig,axes=plt.subplots(2,3,figsize=(18,10))
fig.suptitle("Model FE – diagnostyka reszt",fontsize=13,fontweight="bold")
resid_fe=model_fe.resid; fitted_fe=model_fe.fittedvalues
ax=axes[0,0]; ax.scatter(range(len(resid_fe)),resid_fe,color=BLUE,s=8,alpha=0.35); ax.axhline(0,color=RED,lw=1.5,ls="--"); ax.set_title("Reszty (kolejność obs.)")
ax=axes[0,1]; ax.scatter(fitted_fe,resid_fe,color=BLUE,s=8,alpha=0.35); ax.axhline(0,color=RED,lw=1.5,ls="--"); ax.set_title("Reszty vs dopasowane")
ax=axes[0,2]; ax.hist(resid_fe,bins=20,color=BLUE,alpha=0.7,edgecolor="white",density=True)
xr=np.linspace(resid_fe.min(),resid_fe.max(),100); ax.plot(xr,stats.norm.pdf(xr,resid_fe.mean(),resid_fe.std()),"r-",lw=2); ax.set_title("Histogram reszt")
ax=axes[1,0]
(osm,osr),(slope,intercept,_)=stats.probplot(resid_fe,dist="norm")
ax.scatter(osm,osr,color=BLUE,s=18,alpha=0.6); ax.plot(osm,slope*np.array(osm)+intercept,"r-",lw=2); ax.set_title(f"QQ-plot  SW p={sw_p_fe:.3f}")
ax=axes[1,1]
box_data=[resid_fe[dw_tr["wojewodztwo"].values==p] for p in PROV[:8]]
ax.boxplot(box_data,patch_artist=True,boxprops=dict(facecolor=BLUE+"88"),medianprops=dict(color=RED,lw=2))
ax.set_xticklabels([p[:6] for p in PROV[:8]],rotation=35,ha="right",fontsize=8)
ax.axhline(0,color=RED,ls="--",lw=1.5); ax.set_title("Reszty per województwo (8)")
ax=axes[1,2]; ax.axis("off")
test_rows_fe=[
    ["Durbin-Watson (orientacyjny)", f"{dw_fe:.4f}",  ok(1.5<dw_fe<2.5)],
    ["Breusch-Godfrey p",           f"{bg_fe:.4f}",  ok(bg_fe>0.05)],
    ["Ljung-Box lag=4 p",           f"{lb_fe['lb_pvalue'].iloc[2]:.4f}", ok(lb_fe['lb_pvalue'].iloc[2]>0.05)],
    ["Shapiro-Wilk p (n=288)",      f"{sw_p_fe:.4f}", ok(sw_p_fe>0.05)],
    ["Jarque-Bera p",               f"{jb_p_fe:.4f}", ok(jb_p_fe>0.05)],
    ["Breusch-Pagan p",             f"{bp_p_fe:.4f}", ok(bp_p_fe>0.05)],
    ["White p (bez dummy)",         f"{white_fe_p:.4f}", ok(white_fe_p>0.05)],
    ["RESET p (orientacyjny)",      f"{reset_fe.pvalue:.4f}", ok(reset_fe.pvalue>0.05)],
    ["F-test dummies p",            f"{p_fe:.6f}", "FE istotne ✓" if p_fe<0.05 else "FE nieistotne"],
    ["Condition number (główne)",   f"{cond_fe_main:.1f}", ok(cond_fe_main<1000)],
]
t4=ax.table(cellText=test_rows_fe,colLabels=["Test","Wartość","Wniosek"],cellLoc="center",loc="center",bbox=[0,0.08,1,0.85])
t4.auto_set_font_size(False);t4.set_fontsize(9)
for j in range(3): t4[0,j].set_facecolor("#1a5c96");t4[0,j].set_text_props(color="white",fontweight="bold")
for i,row in enumerate(test_rows_fe,1):
    bg="#c8e6c9" if ("OK" in row[2] or "istotne ✓" in row[2]) else "#ffcdd2"
    for j in range(3): t4[i,j].set_facecolor(bg)
ax.set_title("Testy diagnostyczne",fontsize=11,fontweight="bold",pad=6)
plt.tight_layout(); save("z3_05_model_fe_diag.png")

# PNG 06 – Dopasowanie in-sample FE (8 województw)
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
fig.suptitle(f"Model FE – dopasowanie in-sample per województwo (2005–{TRAIN_END})", fontsize=12, fontweight="bold")
for i, (ax, prov) in enumerate(zip(axes.flat, PROV[:8])):
    idx_p = dw_tr["wojewodztwo"].values == prov
    if not idx_p.any():
        ax.set_title(prov, fontsize=9); continue
    t_p     = dw_tr[idx_p]["rok"].values
    y_act_p = np.exp(y_tr_w[idx_p])
    y_fit_p = np.exp(model_fe.fittedvalues[idx_p])
    ax.plot(t_p, y_act_p, "o-", color=BLUE, lw=2, ms=5, label="Rzeczywiste")
    ax.plot(t_p, y_fit_p, "s--", color=RED, lw=2, ms=5, label="Dopasowane")
    ax.fill_between(t_p, y_act_p, y_fit_p, alpha=0.10, color=RED)
    rmspe_r = prov_rmspe.get(prov, float("nan"))
    ax.set_title(f"{prov}  RMSPE={rmspe_r:.1f}%", fontsize=9)
    ax.set_xlabel("Rok", fontsize=8)
    if i % 4 == 0: ax.set_ylabel("GWh")
    if i == 0: ax.legend(fontsize=7, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
plt.tight_layout(); save("z3_06_model_fe_fit.png")

# PNG 07 – Porównanie modeli
comp_data=[
    ["Model Polska (OLS)",f"{model_p.rsquared:.4f}",f"{model_p.rsquared_adj:.4f}",
     f"{model_p.aic:.1f}",f"{model_p.bic:.1f}",f"{ms_p_tr['RMSPE']:.2f}%",
     f"{len(dp_tr)} obs","OK ✓" if ms_p_tr['RMSPE']<=10 else "Wysoki"],
    ["Model FE Woj. (panel)",f"{model_fe.rsquared:.4f}",f"{model_fe.rsquared_adj:.4f}",
     f"{model_fe.aic:.1f}",f"{model_fe.bic:.1f}",f"{ms_fe_tr['RMSPE']:.2f}%",
     f"{good_fe}/16 woj.","OK ✓" if ms_fe_tr['RMSPE']<=10 else "Wysoki"],
]
comp_cols=["Model","R²","R²adj","AIC","BIC",f"RMSPE% in-sample\n(2004–{TRAIN_END})","Woj./obs.","Status"]
fig,ax=plt.subplots(figsize=(16,3)); ax.axis("off")
t=ax.table(cellText=comp_data,colLabels=comp_cols,cellLoc="center",loc="center",bbox=[0,0,1,1])
t.auto_set_font_size(False);t.set_fontsize(11)
for j in range(len(comp_cols)): t[0,j].set_facecolor("#1a5c96");t[0,j].set_text_props(color="white",fontweight="bold")
for i,row in enumerate(comp_data,1):
    ok_r=row[-1].startswith("OK")
    for j in range(len(comp_cols)): t[i,j].set_facecolor("#dff0d8" if ok_r else "#fff3cd")
    t[i,5].set_facecolor("#c8e6c9" if ok_r else "#ffcdd2")
ax.set_title("Porównanie modeli – miary jakości dopasowania i prognozy",fontsize=12,fontweight="bold",pad=10)
plt.tight_layout(); save("z3_07_rmspe_porownanie.png")

print("  Z3: pliki z3_01..z3_07_*.png gotowe")


# ╔══════════════════════════════════════════════════════════════╗
# ║        ZADANIE 4 – PROGNOZY ZMIENNYCH OBJAŚNIAJĄCYCH        ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("ZADANIE 4 – PROGNOZY ZMIENNYCH OBJAŚNIAJĄCYCH")
print("=" * 60)

METHODS = ["OLS_lin","OLS_kw","ARIMA","Holt","Pawl"]
COLORS_M = {"OLS_lin":BLUE,"OLS_kw":GREEN,"ARIMA":PURPLE,"Holt":GRAY,"Pawl":"#16a085"}

# ── Polska (3 zmienne) ────────────────────────────────────────
VARS_P = {
    "pkb_per_capita":      ("PKB per capita [PLN/os.]", GREEN),
    "cena_energii_zl_kWh": ("Cena energii [PLN/kWh]",   RED),
    "hdd":                 ("Stopniodni grzewcze (HDD)", BLUE),
}
res_p = {}
for col,(label,_) in VARS_P.items():
    res = forecast_all(dp_tr[col].values, t_tr_p, dp_te[col].values, t_te_p, t_fc)
    best = best_method(res)
    res_p[col] = res
    print(f"\n  {col}: najlepsza={best}  RMSPE={res[best]['rmspe']:.2f}%  FC{FC_YR}={res[best]['pred_fc'][0]:,.3f}")
    for mth,v in res.items():
        print(f"    {mth:<10} RMSPE={v['rmspe']:6.2f}%  MAPE={v['mape']:6.2f}%{'  ◄' if mth==best else ''}")

# PNG 01 – RMSPE% tabela Polska
rows_rp=[]
for mth in METHODS:
    row=[mth]
    for col in VARS_P:
        v=res_p[col][mth]["rmspe"]; best_c=best_method(res_p[col])
        row.append(f"{v:.2f}%{'*' if mth==best_c else ''}")
    rows_rp.append(row)
col_labels_rp=["Metoda"]+[VARS_P[c][0].split(" [")[0].replace("Stopniodni grzewcze ","HDD") for c in VARS_P]
fig,ax=plt.subplots(figsize=(14,5)); ax.axis("off")
t=ax.table(cellText=rows_rp,colLabels=col_labels_rp,cellLoc="center",loc="center",bbox=[0,0,1,1])
t.auto_set_font_size(False);t.set_fontsize(10)
for j in range(len(col_labels_rp)): t[0,j].set_facecolor("#1a5c96");t[0,j].set_text_props(color="white",fontweight="bold")
for i,row in enumerate(rows_rp,1):
    bg="#f0f5ff" if i%2==0 else "white"
    for j in range(len(col_labels_rp)): t[i,j].set_facecolor(bg)
    for jj,col in enumerate(VARS_P,1):
        bc=best_method(res_p[col]); val=res_p[col][METHODS[i-1]]["rmspe"]
        if METHODS[i-1]==bc: t[i,jj].set_facecolor("#c8e6c9");t[i,jj].set_text_props(fontweight="bold")
        if val>10: t[i,jj].set_facecolor("#ffcdd2")
ax.set_title(f"RMSPE% ex-post prognoz zmiennych X – Model Polska (2004–{TRAIN_END})  (* = najlepsza metoda)",fontsize=11,fontweight="bold",pad=10)
plt.tight_layout(); save("z4_01_polska_rmspe.png")

# PNG 02-04 – wykresy per zmienna Polska (dopasowanie 2004–TRAIN_END + ex-ante FC_YR)
for (col,(label,_)),fname in zip(VARS_P.items(),["z4_02_polska_pkb.png","z4_03_polska_cena.png","z4_04_polska_hdd.png"]):
    best=best_method(res_p[col])
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(18,6),gridspec_kw={"width_ratios":[3,1]})
    fig.suptitle(f"Prognoza {label} – Polska  [najlepsza: {best}]",fontsize=12,fontweight="bold")
    ax1.plot(t_tr_p, dp_tr[col].values,"o-",color=BLUE,lw=2.2,ms=7,zorder=6,label=f"Rzeczywiste (2004–{TRAIN_END})")
    for mth in METHODS:
        v=res_p[col][mth]; clr=COLORS_M[mth]; is_best=(mth==best)
        fit_v=v["fitted"]; valid=~np.isnan(fit_v)
        # krzywa dopasowania na próbie uczącej
        ax1.plot(t_tr_p[valid],fit_v[valid],
                 "-" if is_best else "--",color=clr,
                 lw=2.2 if is_best else 1.0,alpha=1.0 if is_best else 0.45,
                 label=f"{mth} RMSPE={v['rmspe']:.1f}%")
        # punkt ex-ante
        ax1.plot(t_fc, v["pred_fc"],
                 "^" if is_best else "v",color=clr,
                 ms=10 if is_best else 6,alpha=1.0 if is_best else 0.55,zorder=5)
    ax1.axvline(TRAIN_END+0.5,color=GRAY,ls=":",lw=1.5)
    ax1.set_xlabel("Rok"); ax1.set_ylabel(label)
    ax1.legend(fontsize=8,loc="upper left",ncol=2); ax1.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.2f}" if x<10 else f"{x:,.0f}"))
    tbl_d=[[mth,f"{res_p[col][mth]['rmspe']:.2f}%","✓" if res_p[col][mth]['rmspe']<=10 else "✗"] for mth in METHODS]
    ax2.axis("off")
    t2=ax2.table(cellText=tbl_d,colLabels=["Metoda",f"RMSPE% (2004–{TRAIN_END})","≤10%?"],
                 cellLoc="center",loc="center",bbox=[0,0,1,1])
    t2.auto_set_font_size(False);t2.set_fontsize(10)
    for j in range(3): t2[0,j].set_facecolor("#1a5c96");t2[0,j].set_text_props(color="white",fontweight="bold")
    for i,row in enumerate(tbl_d,1):
        ok2=row[2]=="✓"
        bg="#c8e6c9" if METHODS[i-1]==best else ("#dff0d8" if ok2 else "#ffcdd2")
        for j in range(3): t2[i,j].set_facecolor(bg)
    ax2.set_title(f"FC {FC_YR} = {res_p[col][best]['pred_fc'][0]:,.2f}",fontsize=10,pad=6)
    plt.tight_layout(); save(fname)

# ── Województwa (6 zmiennych, per province) ───────────────────
VARS_W = {
    "dochod_os":           ("Dochód na osobę [PLN]",    GREEN),
    "cena_energii_zl_kWh": ("Cena energii [PLN/kWh]",   RED),
    "urbanizacja_pct":     ("Urbanizacja [%]",           PURPLE),
    "liczba_os":           ("Liczba os. w gosp.",        BLUE),
    "pow_os":              ("Pow. mieszk. na os. [m²]", ORANGE),
    "hdd":                 ("HDD",                       GRAY),
}
res_w = {}
for col,(label,_) in VARS_W.items():
    res_w[col]={}; rmspe_list=[]
    for prov in PROV:
        dp=df_w[df_w["wojewodztwo"]==prov].sort_values("rok")
        y_tr2=dp[dp["rok"]<=TRAIN_END][col].values
        t_tr2=dp[dp["rok"]<=TRAIN_END]["rok"].values
        y_te2=dp[dp["rok"].isin(TEST_YRS)][col].values  # tylko do wizualizacji
        t_te2=dp[dp["rok"].isin(TEST_YRS)]["rok"].values
        if len(y_tr2)<5 or np.any(np.isnan(y_tr2)):
            res_w[col][prov]=None; continue
        r=forecast_all(y_tr2,t_tr2,y_te2,t_te2,t_fc); res_w[col][prov]=r
        rmspe_list.append(r[best_method(r)]["rmspe"])
    med=np.median(rmspe_list) if rmspe_list else float("nan")
    good=sum(1 for v in rmspe_list if v<=10)
    print(f"  {col}: medRMSPE={med:.2f}%  woj≤10%={good}/16")

# PNG 05 – RMSPE% tabela Województwa
rmspe_rows=[]
for mth in METHODS:
    row=[mth]
    for col in VARS_W:
        vals=[res_w[col][p][mth]["rmspe"] for p in PROV if res_w[col].get(p) is not None]
        row.append(f"{np.mean(vals):.1f}%" if vals else "—")
    rmspe_rows.append(row)
best_row2=["Najlepsza*"]
for col in VARS_W:
    vals=[res_w[col][p][best_method(res_w[col][p])]["rmspe"] for p in PROV if res_w[col].get(p) is not None]
    best_row2.append(f"{np.mean(vals):.1f}%*" if vals else "—")
rmspe_rows.append(best_row2)
col_lw=["Metoda"]+[VARS_W[c][0].split(" [")[0][:16] for c in VARS_W]
fig,ax=plt.subplots(figsize=(18,5)); ax.axis("off")
t=ax.table(cellText=rmspe_rows,colLabels=col_lw,cellLoc="center",loc="center",bbox=[0,0,1,1])
t.auto_set_font_size(False);t.set_fontsize(9)
for j in range(len(col_lw)): t[0,j].set_facecolor("#1a5c96");t[0,j].set_text_props(color="white",fontweight="bold")
for i in range(1,len(rmspe_rows)+1):
    bg="#f0f5ff" if i%2==0 else "white"
    for j in range(len(col_lw)): t[i,j].set_facecolor(bg)
for j in range(len(col_lw)): t[len(rmspe_rows),j].set_facecolor("#c8e6c9");t[len(rmspe_rows),j].set_text_props(fontweight="bold")
ax.set_title(f"Średnia RMSPE% ex-post (2004–{TRAIN_END}) prognoz zmiennych X – Model FE Województwa",fontsize=10,fontweight="bold",pad=10)
plt.tight_layout(); save("z4_05_woj_rmspe.png")

# PNG 06 – Miary jakości prognoz zmiennych X – Polska (najlepsza metoda, in-sample 2004–TRAIN_END)
m_keys_z4 = ["ME","MPE","MAE","MAPE","RMSE","RMSPE","TheilU","UM","UV","UC"]
m_labs_z4  = ["ME","MPE%","MAE","MAPE%","RMSE","RMSPE%","TheilU","UM","UV","UC"]
best_labels_p = {col: best_method(res_p[col]) for col in VARS_P}
col_hdrs_z4 = ["Miara"] + [f"{VARS_P[c][0].split(' [')[0]}\n(met. {best_labels_p[c]})" for c in VARS_P]
miary_rows_z4 = []
for mk, ml in zip(m_keys_z4, m_labs_z4):
    row = [ml]
    for col in VARS_P:
        best = best_labels_p[col]
        ms = res_p[col][best].get("ms", {})
        val = ms.get(mk, float("nan"))
        if mk in ("MPE","MAPE","RMSPE"):
            row.append(f"{val:.2f}%")
        elif mk in ("TheilU","UM","UV","UC"):
            row.append(f"{val:.4f}")
        else:
            row.append(f"{val:.2f}")
    miary_rows_z4.append(row)
fig, ax = plt.subplots(figsize=(16, 6)); ax.axis("off")
t = ax.table(cellText=miary_rows_z4, colLabels=col_hdrs_z4, cellLoc="center", loc="center", bbox=[0,0,1,1])
t.auto_set_font_size(False); t.set_fontsize(10)
for j in range(len(col_hdrs_z4)):
    t[0,j].set_facecolor("#1a5c96"); t[0,j].set_text_props(color="white", fontweight="bold")
rmspe_row_idx = m_keys_z4.index("RMSPE")
for i in range(1, len(m_keys_z4)+1):
    bg = "#f0f5ff" if i % 2 == 0 else "white"
    for j in range(len(col_hdrs_z4)): t[i,j].set_facecolor(bg)
for j in range(1, len(col_hdrs_z4)):
    t[rmspe_row_idx+1, j].set_facecolor("#c8e6c9"); t[rmspe_row_idx+1, j].set_text_props(fontweight="bold")
ax.set_title(f"Miary jakości prognoz zmiennych X – Model Polska (najlepsza metoda, in-sample 2004–{TRAIN_END})  |  TheilU: UM=obciążoność, UV=wahania, UC=kierunek",
             fontsize=10, fontweight="bold", pad=10)
plt.tight_layout(); save("z4_06_polska_miary_x.png")

# ── Analiza trafności Z4: prognozy X na FC_YR vs wartości rzeczywiste ──
print(f"\n[Analiza trafności Z4 – prognoza X na {FC_YR} vs rzeczywiste]")
df_fc_act = df_p[df_p["rok"] == FC_YR]  # rzeczywiste wartości X dla roku FC_YR
trafnosc_p_rows = []
for col, (label, _) in VARS_P.items():
    best = best_method(res_p[col])
    y_fc  = float(res_p[col][best]["pred_fc"][0])
    if len(df_fc_act) > 0 and col in df_fc_act.columns:
        y_real = float(df_fc_act[col].values[0])
        blad   = y_fc - y_real
        blad_w = (y_fc - y_real) / y_real * 100
        print(f"  {col:<28} metoda={best:<10} prognoza={y_fc:>12,.3f}  rzecz.={y_real:>12,.3f}  błąd={blad:>+12,.3f}  błąd%={blad_w:>+7.2f}%")
        trafnosc_p_rows.append([label.split(" [")[0], best, f"{y_fc:,.3f}", f"{y_real:,.3f}", f"{blad:+,.3f}", f"{blad_w:+.2f}%"])
    else:
        print(f"  {col:<28} metoda={best:<10} prognoza={y_fc:>12,.3f}  brak danych rzeczywistych")
        trafnosc_p_rows.append([label.split(" [")[0], best, f"{y_fc:,.3f}", "—", "—", "—"])

df_fc_act_w = df_w[df_w["rok"] == FC_YR]
trafnosc_w_rows = []
for col, (label, _) in VARS_W.items():
    fc_vals = []; real_vals = []
    for prov in PROV:
        r = res_w[col].get(prov)
        if r is None: continue
        best = best_method(r)
        fc_vals.append(float(r[best]["pred_fc"][0]))
        row_r = df_fc_act_w[df_fc_act_w["wojewodztwo"] == prov]
        if len(row_r) > 0 and col in row_r.columns:
            real_vals.append(float(row_r[col].values[0]))
    if fc_vals and real_vals and len(fc_vals) == len(real_vals):
        bledy_w = [(f-r)/r*100 for f,r in zip(fc_vals, real_vals)]
        med_blad = float(np.median(bledy_w))
        mean_blad = float(np.mean(bledy_w))
        print(f"  {col:<28} mediana_błąd%={med_blad:>+7.2f}%  średnia_błąd%={mean_blad:>+7.2f}%")
        trafnosc_w_rows.append([label.split(" [")[0], f"{med_blad:+.2f}%", f"{mean_blad:+.2f}%", f"{len(fc_vals)}"])
    else:
        trafnosc_w_rows.append([label.split(" [")[0], "—", "—", f"{len(fc_vals)}"])

# PNG 09 – Tabela trafności X Polska
if trafnosc_p_rows:
    fig, ax = plt.subplots(figsize=(16, 3)); ax.axis("off")
    t = ax.table(cellText=trafnosc_p_rows,
                 colLabels=["Zmienna", "Metoda", f"Prognoza {FC_YR}", f"Rzeczywiste {FC_YR}", "Błąd abs.", "Błąd wzgl."],
                 cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False); t.set_fontsize(10)
    for j in range(6): t[0,j].set_facecolor("#1a5c96"); t[0,j].set_text_props(color="white", fontweight="bold")
    for i, row in enumerate(trafnosc_p_rows, 1):
        ok_r = row[5] != "—" and abs(float(row[5].replace("%","").replace("+",""))) <= 5
        bg = "#e8f5e9" if ok_r else "#fff8e1"
        for j in range(6): t[i,j].set_facecolor(bg)
        if row[5] != "—": t[i,5].set_facecolor("#c8e6c9" if ok_r else "#ffcdd2"); t[i,5].set_text_props(fontweight="bold")
    ax.set_title(f"Analiza trafności prognoz zmiennych X – Model Polska (prognoza {FC_YR} vs rzeczywiste {FC_YR})",
                 fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout(); save("z4_09_trafnosc_x_polska.png")

print(f"  Z4: pliki z4_01..z4_09_*.png gotowe")


# ╔══════════════════════════════════════════════════════════════╗
# ║          ZADANIE 5 – PROGNOZA WARUNKOWA                     ║
# ╚══════════════════════════════════════════════════════════════╝
print("\n" + "=" * 60)
print("ZADANIE 5 – PROGNOZA WARUNKOWA ZMIENNEJ OBJAŚNIANEJ")
print("=" * 60)

# ── Model Polska ──────────────────────────────────────────────
best_pkb  = best_method(res_p["pkb_per_capita"])
best_cena = best_method(res_p["cena_energii_zl_kWh"])
best_hdd  = best_method(res_p["hdd"])
print(f"  PKB_pc→{best_pkb}  CENA→{best_cena}  HDD→{best_hdd}")

def build_X_polska(pkb_v, cena_v, hdd_v):
    n=len(pkb_v)
    return pd.DataFrame({"const":np.ones(n),"ln_pkb_pc":np.log(np.asarray(pkb_v,float)),
                          "ln_cena":np.log(np.asarray(cena_v,float)),"hdd":np.asarray(hdd_v,float)})

X_cond_te=build_X_polska(res_p["pkb_per_capita"][best_pkb]["pred_test"],
                          res_p["cena_energii_zl_kWh"][best_cena]["pred_test"],
                          res_p["hdd"][best_hdd]["pred_test"])
X_cond_fc=build_X_polska(res_p["pkb_per_capita"][best_pkb]["pred_fc"],
                          res_p["cena_energii_zl_kWh"][best_cena]["pred_fc"],
                          res_p["hdd"][best_hdd]["pred_fc"])

pred_te_obj=model_p.get_prediction(X_cond_te); pred_fc_obj=model_p.get_prediction(X_cond_fc)
ln_hat_te=np.asarray(pred_te_obj.predicted_mean).ravel()
ln_ci_te=pred_te_obj.conf_int(alpha=0.05)
ln_hat_fc=np.asarray(pred_fc_obj.predicted_mean).ravel()
ln_ci_fc=pred_fc_obj.conf_int(alpha=0.05)

y_hat_te5_p=np.exp(ln_hat_te); y_ci_te_lo=np.exp(ln_ci_te[:,0]); y_ci_te_hi=np.exp(ln_ci_te[:,1])
y_hat_fc_p=np.exp(ln_hat_fc);  y_ci_fc_lo=np.exp(ln_ci_fc[:,0]); y_ci_fc_hi=np.exp(ln_ci_fc[:,1])
ms5_p=miary(y_act_te_p, y_hat_te5_p)

X_bench=pd.DataFrame({"const":np.ones(len(dp_te)),"ln_pkb_pc":dp_te["ln_pkb_pc"].values,
                        "ln_cena":dp_te["ln_cena"].values,"hdd":dp_te["hdd"].values})
ms5_bench=miary(y_act_te_p, np.exp(np.asarray(model_p.predict(X_bench)).ravel()))

print(f"  Ex-post (rzecz. X, {TEST_YRS[0]}–{TEST_YRS[-1]}): RMSPE%={ms5_bench['RMSPE%']:.2f}%")
print(f"  Ex-post (prog. X, {TEST_YRS[0]}–{TEST_YRS[-1]}): RMSPE%={ms5_p['RMSPE%']:.2f}%")
print(f"  Ex-ante {FC_YR}: {y_hat_fc_p[0]:,.0f} GWh  95%CI [{y_ci_fc_lo[0]:,.0f}–{y_ci_fc_hi[0]:,.0f}]")

# PNG 01 – Prognoza Polska (dopasowanie in-sample + ex-ante)
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df_p[df_p["rok"]<=TRAIN_END]["rok"],
        df_p[df_p["rok"]<=TRAIN_END]["zuzycie_energii_GWh"],
        "o-", color=BLUE, lw=2.2, ms=7, label=f"Dane historyczne (2004–{TRAIN_END})")
ax.plot(dp_tr["rok"].values, np.exp(model_p.fittedvalues),
        "s--", color=GREEN, lw=1.8, ms=5, alpha=0.85,
        label=f"Dopasowanie in-sample  RMSPE={ms_p_tr['RMSPE']:.1f}%")
ax.errorbar(FC_YR, y_hat_fc_p[0],
            yerr=[[y_hat_fc_p[0]-y_ci_fc_lo[0]], [y_ci_fc_hi[0]-y_hat_fc_p[0]]],
            fmt="^", color=ORANGE, ms=13, lw=2.5,
            label=f"Ex-ante {FC_YR} = {y_hat_fc_p[0]:,.0f} GWh  95%CI [{y_ci_fc_lo[0]:,.0f}–{y_ci_fc_hi[0]:,.0f}]")
ax.axvline(TRAIN_END+0.5, color=GRAY, ls=":", lw=1.8, alpha=0.7)
ax.set_title(f"Model Polska – dopasowanie in-sample (2004–{TRAIN_END}) i prognoza ex-ante {FC_YR}",
             fontsize=11, fontweight="bold")
ax.set_xlabel("Rok"); ax.set_ylabel("Zużycie energii elektrycznej [GWh]")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend(loc="upper left", fontsize=9); ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
plt.tight_layout(); save("z5_01_polska_prognoza.png")

# ── Model FE – Województwa ────────────────────────────────────
COL_MAP={"cena_energii_zl_kWh":"cena_energii_zl_kWh","urbanizacja_pct":"urbanizacja_pct",
          "liczba_os":"liczba_os","pow_os":"pow_os","hdd":"hdd"}
prov_results={}
XtXinv=np.linalg.pinv(X_tr_w.T @ X_tr_w)
mse_fe=model_fe.mse_resid

for prov in PROV:
    dp=df_w[df_w["wojewodztwo"]==prov].sort_values("rok").reset_index(drop=True)
    dp_te_w=dp[dp["rok"].isin(TEST_YRS)]
    if len(dp_te_w)==0: continue

    actual_dochod_2022=dp[dp["rok"]==2022]["dochod_os"].values[0]
    r_d=res_w["dochod_os"].get(prov)
    if r_d is None:
        pred_d23=pred_d24=actual_dochod_2022
    else:
        bd=best_method(r_d)
        pred_d23=float(r_d[bd]["pred_test"][0]); pred_d24=float(r_d[bd]["pred_test"][1])
    lag23=np.log(actual_dochod_2022); lag24=np.log(max(pred_d23,1.0)); lag25=np.log(max(pred_d24,1.0))

    def _get_fc(col_raw, idx_te, is_fc=False):
        r=res_w[col_raw].get(prov)
        if r is None:
            actual=dp_te_w[COL_MAP.get(col_raw,col_raw)].values
            return float(actual[idx_te]) if not is_fc else float(actual[-1])
        best=best_method(r)
        if is_fc: return float(np.asarray(r[best]["pred_fc"]).ravel()[0])
        return float(np.asarray(r[best]["pred_test"]).ravel()[idx_te])

    rows_X=[]
    for yr_idx,(lag_v,) in enumerate([(lag23,),(lag24,),(lag25,)]):
        is_fc=(yr_idx==2); idx_te=yr_idx if yr_idx<2 else 1
        rows_X.append({"const":1.0,"ln_dochod_os_lag1":lag_v,
            "ln_cena":np.log(max(_get_fc("cena_energii_zl_kWh",idx_te,is_fc),1e-6)),
            "urbanizacja_pct":_get_fc("urbanizacja_pct",idx_te,is_fc),
            "liczba_os":_get_fc("liczba_os",idx_te,is_fc),
            "pow_os":_get_fc("pow_os",idx_te,is_fc),
            "hdd":_get_fc("hdd",idx_te,is_fc)})
    dum_vec={c:(1.0 if prov==c else 0.0) for c in dum_cols}
    main_c=["const","ln_dochod_os_lag1","ln_cena","urbanizacja_pct","liczba_os","pow_os","hdd"]
    X_pred=np.array([[r[c] for c in main_c]+[dum_vec[c] for c in dum_cols] for r in rows_X])
    ln_hat=model_fe.predict(X_pred); y_hat=np.exp(np.asarray(ln_hat).ravel())
    y_hat_te5=y_hat[:2]; y_hat_fc5=y_hat[2]
    y_act_te5=dp_te_w["zuzycie_energii_GWh"].values
    se2=mse_fe*(1+float(np.squeeze(X_pred[2:3]@XtXinv@X_pred[2:3].T))); se=np.sqrt(max(se2,0))
    ci_lo5=np.exp(ln_hat[2]-1.96*se); ci_hi5=np.exp(ln_hat[2]+1.96*se)
    ms5 = miary(y_act_te5, y_hat_te5)
    prov_results[prov]={"y_hat_te":y_hat_te5,"y_act_te":y_act_te5,"y_hat_fc":y_hat_fc5,
                         "ci_lo":ci_lo5,"ci_hi":ci_hi5,"ms":ms5}
    print(f"  {prov:<25} RMSPE%={ms5['RMSPE%']:6.2f}%  FC{FC_YR}={y_hat_fc5:,.0f}")

good5=sum(1 for v in prov_results.values() if v["ms"]["RMSPE%"]<=10)
all_rmspe5=[v["ms"]["RMSPE%"] for v in prov_results.values()]
fc_2025_polska=float(y_hat_fc_p[0])
fc_2025_woj_sum=sum(v["y_hat_fc"] for v in prov_results.values())
print(f"  Woj≤10%: {good5}/16  med={np.median(all_rmspe5):.2f}%  agregat RMSPE in-sample={ms_fe_tr['RMSPE']:.2f}%")

# PNG 03 – Siatka 4×4 prognoza per woj.
fig,axes=plt.subplots(4,4,figsize=(24,18))
fig.suptitle("Model FE – prognoza warunkowa zużycia energii per województwo",fontsize=14,fontweight="bold")
for i,(ax,prov) in enumerate(zip(axes.flat,PROV)):
    dp=df_w[df_w["wojewodztwo"]==prov].sort_values("rok")
    ax.plot(dp["rok"],dp["zuzycie_energii_GWh"],"o-",color=BLUE,lw=1.8,ms=4,label="Rzecz.")
    r=prov_results.get(prov)
    if r is not None:
        ax.plot(TEST_YRS,r["y_hat_te"],"s--",color=RED,lw=2,ms=7)
        ax.errorbar(FC_YR,r["y_hat_fc"],yerr=[[r["y_hat_fc"]-r["ci_lo"]],[r["ci_hi"]-r["y_hat_fc"]]],
                    fmt="^",color=ORANGE,ms=9,lw=2,label=f"{FC_YR}={r['y_hat_fc']:,.0f}")
        rms=r["ms"]["RMSPE%"]; mark="✓" if rms<=10 else "✗"
        ax.set_title(f"{prov}  {mark} {rms:.1f}%",fontsize=8,fontweight="bold")
    else: ax.set_title(prov,fontsize=8)
    ax.axvline(TRAIN_END+0.5,color=GRAY,ls=":",lw=1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x/1e3:.1f}k"))
    ax.set_xlabel("Rok",fontsize=7)
    if i%4==0: ax.set_ylabel("GWh",fontsize=8)
    if i==0: ax.legend(fontsize=6,loc="upper left")
plt.tight_layout(); save("z5_03_woj_prognoza.png")

# PNG 04 – Tabela miar per woj.
tbl5=[]
for prov in PROV:
    r=prov_results.get(prov)
    if r is None: continue
    ms=r["ms"]; ok5w="✓" if ms["RMSPE%"]<=10 else "✗"
    tbl5.append([prov,ok5w,f"{ms['ME']:.0f}",f"{ms['MPE%']:.1f}%",f"{ms['MAE']:.0f}",
                 f"{ms['MAPE%']:.1f}%",f"{ms['RMSE']:.0f}",f"{ms['RMSPE%']:.1f}%",
                 f"{ms.get('UM',float('nan')):.3f}",f"{ms.get('UV',float('nan')):.3f}",f"{ms.get('UC',float('nan')):.3f}",
                 f"{r['y_hat_fc']:,.0f}"])
tbl5_cols=["Województwo","OK?","ME","MPE%","MAE","MAPE%","RMSE","RMSPE%","UM","UV","UC",f"FC {FC_YR} [GWh]"]
fig,ax=plt.subplots(figsize=(26,9)); ax.axis("off")
t=ax.table(cellText=tbl5,colLabels=tbl5_cols,cellLoc="center",loc="center",bbox=[0,0,1,1])
t.auto_set_font_size(False);t.set_fontsize(9)
for j in range(len(tbl5_cols)): t[0,j].set_facecolor("#1a5c96");t[0,j].set_text_props(color="white",fontweight="bold")
for i,row in enumerate(tbl5,1):
    ok5r=row[1]=="✓"; bg="#e8f5e9" if ok5r else "#fff8e1"
    for j in range(len(tbl5_cols)): t[i,j].set_facecolor(bg)
    ri=tbl5_cols.index("RMSPE%"); t[i,ri].set_facecolor("#c8e6c9" if ok5r else "#ffcdd2"); t[i,ri].set_text_props(fontweight="bold")
ax.set_title(f"Model FE – miary jakości prognozy warunkowej per województwo (in-sample 2004–{TRAIN_END})",fontsize=11,fontweight="bold",pad=10)
plt.tight_layout(); save("z5_04_woj_miary.png")

# ── Analiza trafności ex-ante: prognoza Y na FC_YR vs rzeczywiste ──────────
y_real_p_fc  = df_p.loc[df_p["rok"]==FC_YR, "zuzycie_energii_GWh"]
y_real_w_fc  = df_w[df_w["rok"]==FC_YR].groupby("rok")["zuzycie_energii_GWh"].sum()
has_real_fc  = len(y_real_p_fc) > 0

print(f"\n[Analiza trafności ex-ante – prognoza Y na {FC_YR} vs rzeczywiste]")
if has_real_fc:
    y_real_p = float(y_real_p_fc.values[0])
    blad_p   = fc_2025_polska  - y_real_p
    blad_p_w = blad_p / y_real_p * 100
    y_real_w = float(y_real_w_fc.values[0]) if len(y_real_w_fc) > 0 else float("nan")
    blad_w_m = fc_2025_woj_sum - y_real_w
    blad_w_w = blad_w_m / y_real_w * 100 if y_real_w > 0 else float("nan")
    print(f"  Model Polska  : prognoza={fc_2025_polska:,.0f}  rzeczywiste={y_real_p:,.0f}  "
          f"błąd={blad_p:+,.0f} GWh  błąd%={blad_p_w:+.2f}%")
    print(f"  Model FE (woj): prognoza={fc_2025_woj_sum:,.0f}  rzeczywiste={y_real_w:,.0f}  "
          f"błąd={blad_w_m:+,.0f} GWh  błąd%={blad_w_w:+.2f}%")
else:
    y_real_p = y_real_w = float("nan")
    blad_p = blad_p_w = blad_w_m = blad_w_w = float("nan")
    print(f"  Brak danych rzeczywistych za {FC_YR} – analiza trafności niemożliwa.")

# PNG 05 – Wykres danych 2004–TRAIN_END + prognoza FC_YR vs rzeczywiste FC_YR
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df_p[df_p["rok"]<=TRAIN_END]["rok"],
        df_p[df_p["rok"]<=TRAIN_END]["zuzycie_energii_GWh"],
        "o-", color=BLUE, lw=2.2, ms=7, label=f"Dane historyczne (2004–{TRAIN_END})")
ax.errorbar(FC_YR, fc_2025_polska,
            yerr=[[fc_2025_polska-y_ci_fc_lo[0]], [y_ci_fc_hi[0]-fc_2025_polska]],
            fmt="^", color=GREEN, ms=13, lw=2.5,
            label=f"Prognoza OLS {FC_YR} = {fc_2025_polska:,.0f} GWh  95%CI [{y_ci_fc_lo[0]:,.0f}–{y_ci_fc_hi[0]:,.0f}]")
ax.errorbar(FC_YR+0.15, fc_2025_woj_sum, fmt="s", color=RED, ms=11, lw=2,
            label=f"Prognoza FE {FC_YR} = {fc_2025_woj_sum:,.0f} GWh")
if has_real_fc and not np.isnan(y_real_p):
    ax.plot(FC_YR, y_real_p, "D", color=ORANGE, ms=14, zorder=10,
            label=f"Rzeczywiste {FC_YR} = {y_real_p:,.0f} GWh")
ax.axvline(TRAIN_END+0.5, color=GRAY, ls=":", lw=1.8, alpha=0.7)
ax.set_xlabel("Rok"); ax.set_ylabel("Zużycie energii elektrycznej [GWh]")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{x:,.0f}"))
ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
ax.legend(fontsize=9, loc="upper left")
ax.set_title(f"Prognoza zużycia energii – dane 2004–{TRAIN_END} oraz ex-ante {FC_YR}",
             fontsize=12, fontweight="bold")
plt.tight_layout(); save("z5_05_prognoza_exante.png")

# PNG 06 – Tabela trafności ex-ante
tbl_traf = []
if has_real_fc and not np.isnan(y_real_p):
    tbl_traf.append(["Model Polska (OLS)", f"{fc_2025_polska:,.0f}", f"{y_real_p:,.0f}",
                     f"{blad_p:+,.0f}", f"{blad_p_w:+.2f}%",
                     "✓" if abs(blad_p_w)<=5 else "✗"])
    tbl_traf.append(["Model FE (województwa)", f"{fc_2025_woj_sum:,.0f}", f"{y_real_w:,.0f}",
                     f"{blad_w_m:+,.0f}", f"{blad_w_w:+.2f}%",
                     "✓" if abs(blad_w_w)<=5 else "✗"])
    fig, ax = plt.subplots(figsize=(16, 2.5)); ax.axis("off")
    t = ax.table(cellText=tbl_traf,
                 colLabels=["Model", f"Prognoza {FC_YR} [GWh]", f"Rzeczywiste {FC_YR} [GWh]",
                            "Błąd abs. [GWh]", "Błąd wzgl. [%]", "≤5%?"],
                 cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False); t.set_fontsize(11)
    for j in range(6): t[0,j].set_facecolor("#1a5c96"); t[0,j].set_text_props(color="white", fontweight="bold")
    for i, row in enumerate(tbl_traf, 1):
        ok_r = row[5]=="✓"
        for j in range(6): t[i,j].set_facecolor("#e8f5e9" if ok_r else "#fff8e1")
        t[i,4].set_facecolor("#c8e6c9" if ok_r else "#ffcdd2"); t[i,4].set_text_props(fontweight="bold")
    ax.set_title(f"Analiza trafności prognoz ex-ante {FC_YR} – zużycie energii elektrycznej [GWh]",
                 fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout(); save("z5_06_trafnosc_exante.png")
else:
    print(f"  PNG 06 pominięty – brak danych rzeczywistych za {FC_YR}")

# ══════════════════════════════════════════════════════════════
# PODSUMOWANIE KOŃCOWE
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PODSUMOWANIE PROJEKTU")
print("=" * 60)
print(f"\n  Model Polska  : R²={model_p.rsquared:.4f}  RMSPE% in-sample={ms_p_tr['RMSPE']:.2f}%  "
      f"FC {FC_YR}={fc_2025_polska:,.0f} GWh  95%CI=[{y_ci_fc_lo[0]:,.0f}–{y_ci_fc_hi[0]:,.0f}]")
print(f"  Model FE (woj): R²={model_fe.rsquared:.4f}  RMSPE% in-sample={ms_fe_tr['RMSPE']:.2f}%  "
      f"Woj≤10%={good5}/16  FC {FC_YR}={fc_2025_woj_sum:,.0f} GWh")
if has_real_fc and not np.isnan(y_real_p):
    print(f"\n  Trafność OLS  : błąd={blad_p:+,.0f} GWh  ({blad_p_w:+.2f}%)")
    print(f"  Trafność FE   : błąd={blad_w_m:+,.0f} GWh  ({blad_w_w:+.2f}%)")
print(f"\n  Pliki PNG: z2_01..09, z3_01..07, z4_01..06+09, z5_01+03..06")
sys.stdout.flush()

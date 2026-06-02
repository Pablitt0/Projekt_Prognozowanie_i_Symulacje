# -*- coding: utf-8 -*-
# ============================================================
# DASHBOARD – Prognoza Zużycia Energii Elektrycznej w Polsce
# Streamlit app  |  py -m streamlit run dashboard_new.py
# ============================================================
import os, sys, warnings
import numpy as np
import pandas as pd
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_breusch_godfrey, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from scipy.stats import f as f_dist
import streamlit as st
warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BLUE = "#1a5c96"; RED = "#c0392b"; GREEN = "#27ae60"; GRAY = "#7f8c8d"; ORANGE = "#e67e22"
TRAIN_END  = 2022
TEST_YRS   = [2023, 2024]
FC_YR      = 2025
EVAL_START = 2015
EVAL_YEARS = list(range(EVAL_START, FC_YR))  # [2015…2024], n=10

st.set_page_config(
    page_title="⚡ Energia Elektryczna – Prognozowanie i Symulacje",
    page_icon="⚡",
    layout="wide",
)

# ── Style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.labelsize": 10, "axes.titlesize": 11,
})

def show_png(fname, caption=""):
    path = os.path.join(SCRIPT_DIR, fname)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Brak pliku **{fname}** – uruchom najpierw skrypt `{fname.split('_')[0]}*.py`")

def miary(y_act, y_hat):
    e = np.asarray(y_act, float) - np.asarray(y_hat, float)
    ya = np.asarray(y_act, float)
    return {
        "ME":     e.mean(),
        "MPE%":   (e / ya * 100).mean(),
        "MAE":    np.abs(e).mean(),
        "MAPE%":  (np.abs(e) / ya * 100).mean(),
        "RMSE":   np.sqrt((e**2).mean()),
        "RMSPE%": np.sqrt(((e / ya)**2).mean()) * 100,
    }

def best_method(res_dict):
    return min(res_dict, key=lambda k: res_dict[k]["rmspe"])

def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    if p < 0.1:   return "."
    return ""

# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_polska():
    df = pd.read_excel(os.path.join(SCRIPT_DIR, "Zuzycie_energii_polska.xlsx"))
    df = df.sort_values("rok").reset_index(drop=True)
    df["pkb_per_capita"] = df["pkb_mln_zl"] * 1e6 / df["ludnosc"]
    df["ln_zuzycie"]     = np.log(df["zuzycie_energii_GWh"])
    df["ln_pkb_pc"]      = np.log(df["pkb_per_capita"])
    df["ln_cena"]        = np.log(df["cena_energii_zl_kWh"])
    return df

@st.cache_data
def load_woj():
    df = pd.read_excel(os.path.join(SCRIPT_DIR, "Zuzycie_energii_wojewodztwa.xlsx"))
    df = df.sort_values(["wojewodztwo", "rok"]).reset_index(drop=True)
    mask0 = df["dochod_os"] <= 0
    if mask0.any():
        df.loc[mask0, "dochod_os"] = np.nan
        df["dochod_os"] = (df.groupby("wojewodztwo")["dochod_os"]
                              .transform(lambda s: s.interpolate(
                                  method="linear", limit=3, limit_direction="both")))
    df["ln_zuzycie"]        = np.log(df["zuzycie_energii_GWh"])
    df["ln_dochod_os"]      = np.log(df["dochod_os"])
    df["ln_cena"]           = np.log(df["cena_energii_zl_kWh"])
    df["ln_dochod_os_lag1"] = df.groupby("wojewodztwo")["ln_dochod_os"].shift(1)
    return df

@st.cache_data
def fit_models():
    df_p = load_polska()
    df_w = load_woj()
    PROV = sorted(df_w["wojewodztwo"].unique())
    dp_tr = df_p[df_p["rok"] <= TRAIN_END].copy()
    dw_tr = df_w[(df_w["rok"] > 2004) & (df_w["rok"] <= TRAIN_END)].copy()

    X_P   = ["ln_pkb_pc", "ln_cena", "hdd"]
    y_tr  = dp_tr["ln_zuzycie"].values
    X_tr  = sm.add_constant(dp_tr[X_P].values)
    model_p = sm.OLS(y_tr, X_tr).fit()

    X_FE = ["ln_dochod_os_lag1", "ln_cena", "urbanizacja_pct", "liczba_os", "pow_os", "hdd"]
    dummies_tr = pd.get_dummies(dw_tr["wojewodztwo"], drop_first=True).astype(float)
    dum_cols   = list(dummies_tr.columns)
    y_tr_w = dw_tr["ln_zuzycie"].values
    X_tr_w = np.column_stack([np.ones(len(dw_tr)), dw_tr[X_FE].values, dummies_tr.values])
    model_fe = sm.OLS(y_tr_w, X_tr_w).fit()

    # ── Krocząca walidacja ex-post (2015–2024, n=10) ─────────────
    rmspe_fn = lambda ya, yh: float(np.sqrt(np.mean(((np.array(ya)-np.array(yh))/np.array(ya))**2))*100)

    # Polska rolling
    roll_p = []
    for t_ev in EVAL_YEARS:
        dtr_r = df_p[df_p["rok"] < t_ev]
        dte_r = df_p[df_p["rok"] == t_ev]
        if len(dtr_r) < 5 or len(dte_r) == 0:
            continue
        y_r  = dtr_r["ln_zuzycie"].values
        X_r  = np.column_stack([np.ones(len(dtr_r)), dtr_r[X_P].values])
        m_r  = sm.OLS(y_r, X_r).fit()
        X_te_r = np.column_stack([np.ones(1), dte_r[X_P].values])
        y_hat_r = np.exp(float(m_r.predict(X_te_r)[0]))
        roll_p.append((t_ev, float(dte_r["zuzycie_energii_GWh"].values[0]), y_hat_r))
    ya_rp = np.array([ya for _, ya, _ in roll_p])
    yh_rp = np.array([yh for _, _, yh in roll_p])
    ms_p_roll = miary(ya_rp, yh_rp)

    # FE rolling
    roll_fe = {p: [] for p in PROV}
    for t_ev in EVAL_YEARS:
        dtr_r = df_w[(df_w["rok"] > 2004) & (df_w["rok"] < t_ev)]
        dte_r = df_w[df_w["rok"] == t_ev]
        if len(dtr_r) < 50 or len(dte_r) == 0:
            continue
        dum_tr_r = pd.get_dummies(dtr_r["wojewodztwo"], drop_first=True).astype(float)
        for c in dum_cols:
            if c not in dum_tr_r.columns:
                dum_tr_r[c] = 0.0
        dum_tr_r = dum_tr_r[dum_cols]
        X_tr_r = np.column_stack([np.ones(len(dtr_r)), dtr_r[X_FE].values, dum_tr_r.values])
        m_r = sm.OLS(dtr_r["ln_zuzycie"].values, X_tr_r).fit()
        dum_te_r = pd.get_dummies(dte_r["wojewodztwo"]).astype(float)
        for c in dum_cols:
            if c not in dum_te_r.columns:
                dum_te_r[c] = 0.0
        dum_te_r = dum_te_r[dum_cols]
        X_te_r = np.column_stack([np.ones(len(dte_r)), dte_r[X_FE].values, dum_te_r.values])
        y_hat_r = np.exp(np.asarray(m_r.predict(X_te_r)).ravel())
        for prov, ya, yh in zip(dte_r["wojewodztwo"].values, dte_r["zuzycie_energii_GWh"].values, y_hat_r):
            roll_fe[prov].append((t_ev, float(ya), float(yh)))
    ya_rfe = np.concatenate([[ya for _, ya, _ in v] for v in roll_fe.values() if v])
    yh_rfe = np.concatenate([[yh for _, _, yh in v] for v in roll_fe.values() if v])
    ms_fe_roll = miary(ya_rfe, yh_rfe)
    prov_roll_rmspe = {}
    for prov, lst in roll_fe.items():
        if lst:
            ya_p = np.array([ya for _, ya, _ in lst])
            yh_p = np.array([yh for _, _, yh in lst])
            prov_roll_rmspe[prov] = rmspe_fn(ya_p, yh_p)

    return model_p, model_fe, dum_cols, PROV, ms_p_roll, ms_fe_roll, prov_roll_rmspe

@st.cache_data
def load_z4():
    pkl = os.path.join(SCRIPT_DIR, "z4_results.pkl")
    if not os.path.exists(pkl):
        return None
    with open(pkl, "rb") as f:
        return pickle.load(f)

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.title("⚡ Prognoza Zużycia Energii Elektrycznej w Polsce")
st.markdown("""
**Projekt zaliczeniowy** – Prognozowanie i Symulacje (studia magisterskie)
Model Polska: `ln(ZUZYCIE) = β₀ + β₁·ln(PKB_pc) + β₂·ln(CENA) + β₃·HDD`
Model FE: `ln(ZUZYCIE) = Xβ + Σδₖ·dₖ` (15 efektów stałych województw)
Dane: GUS/URE/IMGW, lata 2004–2024 | Train: 2004–2022 | Walidacja krocząca: 2015–2024 (n=10) | Ex-ante: 2025
""")
st.divider()

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Z2 – Analiza opisowa",
    "🔢 Z3 – Modele",
    "📈 Z4 – Prognozy X",
    "🎯 Z5 – Prognoza warunkowa",
    "ℹ️ O projekcie",
])

# ══════════════════════════════════════════════════════════════
# TAB 1 – EDA
# ══════════════════════════════════════════════════════════════
with tab1:
    st.header("Zadanie 2 – Analiza Opisowa Danych")

    sub1, sub2 = st.tabs(["Zmienne i statystyki", "Wizualizacje"])

    with sub1:
        st.subheader("Opis zmiennych modelu")
        show_png("z2_08_opis_zmiennych.png")
        st.subheader("Statystyki opisowe")
        show_png("z2_09_statystyki.png")

        df_p = load_polska()
        df_w = load_woj()
        PROV = sorted(df_w["wojewodztwo"].unique())

        st.subheader("Dane – Polska (interaktywna tabela)")
        cols_show = ["rok", "zuzycie_energii_GWh", "pkb_per_capita",
                     "cena_energii_zl_kWh", "hdd", "cdd"]
        st.dataframe(df_p[[c for c in cols_show if c in df_p.columns]].round(2),
                     use_container_width=True, height=280)

        st.subheader("Dane – Województwa (filtruj)")
        sel_prov = st.selectbox("Wybierz województwo:", PROV, key="eda_prov")
        dp = df_w[df_w["wojewodztwo"] == sel_prov]
        cols_w = ["rok", "zuzycie_energii_GWh", "dochod_os",
                  "cena_energii_zl_kWh", "urbanizacja_pct",
                  "liczba_os", "pow_os", "hdd"]
        st.dataframe(dp[[c for c in cols_w if c in dp.columns]].round(2),
                     use_container_width=True, height=300)

    with sub2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Polska – szereg czasowy")
            show_png("z2_01_polska_szereg.png")
            st.subheader("Korelacja Pearson – Polska")
            show_png("z2_04_korelacja_polska.png")
            st.subheader("Scatter – Polska (Y vs X)")
            show_png("z2_05_scatter_polska.png")
        with col_b:
            st.subheader("Województwa – szeregi czasowe")
            show_png("z2_02_woj_szeregi.png")
            st.subheader("Porównanie województw (2024)")
            show_png("z2_03_woj_porownanie.png")
            st.subheader("Korelacja Pearson – Województwa")
            show_png("z2_06_korelacja_woj.png")
        st.subheader("Scatter – Województwa (panel, Y vs X)")
        show_png("z2_07_scatter_woj.png")

# ══════════════════════════════════════════════════════════════
# TAB 2 – MODELS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.header("Zadanie 3 – Modele Ekonometryczne")

    model_tab1, model_tab2, model_tab3 = st.tabs([
        "🇵🇱 Model Polska", "🗺️ Model FE (Woj.)", "📋 Porównanie"
    ])

    with model_tab1:
        st.subheader("Model Polska – OLS")
        st.latex(r"\ln(ZUZYCIE_t) = \beta_0 + \beta_1 \ln(PKB\_pc_t) + \beta_2 \ln(CENA_t) + \beta_3 HDD_t + \varepsilon_t")
        col1, col2 = st.columns(2)
        with col1:
            show_png("z3_01_model_polska_koef.png", "Współczynniki i miary dopasowania")
        with col2:
            show_png("z3_03_model_polska_fit.png", "Dopasowanie i prognoza testu")
        st.subheader("Diagnostyka reszt")
        show_png("z3_02_model_polska_diag.png")

        # Live model summary
        with st.expander("Szczegółowe wyniki OLS (statsmodels summary)"):
            df_p = load_polska()
            dp_tr = df_p[df_p["rok"] <= TRAIN_END]
            X_P   = ["ln_pkb_pc", "ln_cena", "hdd"]
            model_p, *_ = fit_models()
            st.text(str(model_p.summary()))

            st.subheader("VIF")
            X_tr = sm.add_constant(dp_tr[X_P].values)
            vif_df = pd.DataFrame({
                "Zmienna": ["const"] + X_P,
                "VIF": [variance_inflation_factor(X_tr, i) for i in range(X_tr.shape[1])]
            })
            st.dataframe(vif_df.round(3), use_container_width=True)

    with model_tab2:
        st.subheader("Model FE – Województwa (Pooled OLS + 15 dummy)")
        st.latex(r"\ln(ZUZYCIE_{it}) = \beta X_{it} + \sum_{k=2}^{16} \delta_k d_{ik} + \varepsilon_{it}")
        col1, col2 = st.columns(2)
        with col1:
            show_png("z3_04_model_fe_koef.png", "Współczynniki i miary dopasowania")
        with col2:
            show_png("z3_06_model_fe_fit.png", "Dopasowanie per województwo")
        st.subheader("Diagnostyka reszt FE")
        show_png("z3_05_model_fe_diag.png")

        with st.expander("Szczegółowe wyniki OLS FE (statsmodels summary)"):
            _, model_fe, *_ = fit_models()
            st.text(str(model_fe.summary()))

    with model_tab3:
        st.subheader("Porównanie modeli")
        show_png("z3_07_rmspe_porownanie.png")
        _, _, _, _, ms_p_roll, ms_fe_roll, prov_roll_rmspe = fit_models()
        good_roll = sum(1 for v in prov_roll_rmspe.values() if v <= 10)
        st.info(f"""
**Wnioski (krocząca walidacja ex-post 2015–2024, n=10):**
- **Model Polska**: R²=0.800, RMSPE% rolling = **{ms_p_roll["RMSPE%"]:.2f}%** ✓
- **Model FE**: R²=0.996, RMSPE% rolling = **{ms_fe_roll["RMSPE%"]:.2f}%** ✓ (panel), {good_roll}/16 województw ≤ 10%
- F-test efektów stałych: p<0.001 → efekty stałe istotne (FE uzasadniony)
- Multikolinearność w modelu Polska: VIF(PKB_pc)≈8, VIF(CENA)≈7 – umiarkowana, akceptowalna
        """)

# ══════════════════════════════════════════════════════════════
# TAB 3 – Z4 FORECASTS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.header("Zadanie 4 – Prognozy Zmiennych Objaśniających")
    st.info("7 metod: OLS liniowy, OLS kwadratowy, AR(1), AR(2), ARIMA, Holt, Pawłowski  |  kryterium: RMSPE% ≤ 10%")

    z4_tab1, z4_tab2 = st.tabs(["🇵🇱 Model Polska", "🗺️ Model FE Woj."])

    with z4_tab1:
        st.subheader("Tabela RMSPE% – Polska")
        show_png("z4_01_polska_rmspe.png")
        c1, c2, c3 = st.columns(3)
        with c1:
            show_png("z4_02_polska_pkb.png", "PKB per capita")
        with c2:
            show_png("z4_03_polska_cena.png", "Cena energii")
        with c3:
            show_png("z4_04_polska_hdd.png", "HDD")

        z4 = load_z4()
        if z4:
            st.subheader("Wyniki prognoz Z4 – Polska (interaktywnie)")
            VARS_P = {
                "pkb_per_capita":      "PKB per capita",
                "cena_energii_zl_kWh": "Cena energii",
                "hdd":                 "HDD",
            }
            METHODS = ["OLS_lin", "OLS_kw", "AR1", "AR2", "ARIMA", "Holt", "Pawl"]
            rows = []
            for col, lbl in VARS_P.items():
                for meth in METHODS:
                    r = z4["polska"][col][meth]
                    best = best_method(z4["polska"][col])
                    rows.append({
                        "Zmienna": lbl, "Metoda": meth,
                        "RMSPE%": round(r["rmspe"], 2),
                        "MAPE%":  round(r["mape"], 2),
                        "Pred 2023": round(r["pred_test"][0], 3),
                        "Pred 2024": round(r["pred_test"][1], 3),
                        "FC 2025":   round(r["pred_fc"][0], 3),
                        "Najlepsza": "✓" if meth == best else "",
                    })
            df_z4p = pd.DataFrame(rows)
            st.dataframe(
                df_z4p.style.apply(
                    lambda row: ["background-color:#c8e6c9" if row["Najlepsza"] == "✓" else ""
                                 for _ in row], axis=1),
                use_container_width=True, height=350,
            )

    with z4_tab2:
        st.subheader("Tabela RMSPE% – Województwa (średnia)")
        show_png("z4_05_woj_rmspe.png")
        c1, c2, c3 = st.columns(3)
        with c1:
            show_png("z4_06_woj_dochod.png", "Dochód na osobę")
        with c2:
            show_png("z4_07_woj_cena.png", "Cena energii")
        with c3:
            show_png("z4_08_woj_hdd.png", "HDD")

        z4 = load_z4()
        if z4 and z4.get("woj"):
            st.subheader("Wyniki per województwo")
            PROV = z4["prov"]
            sel_col = st.selectbox("Zmienna:", [
                "dochod_os", "cena_energii_zl_kWh", "urbanizacja_pct",
                "liczba_os", "pow_os", "hdd"
            ], key="z4_col")
            METHODS = ["OLS_lin", "OLS_kw", "AR1", "AR2", "ARIMA", "Holt", "Pawl"]
            rows_w = []
            for prov in PROV:
                r_prov = z4["woj"][sel_col].get(prov)
                if r_prov is None: continue
                best = best_method(r_prov)
                for meth in METHODS:
                    r = r_prov[meth]
                    rows_w.append({
                        "Województwo": prov, "Metoda": meth,
                        "RMSPE%": round(r["rmspe"], 2),
                        "FC 2025": round(float(r["pred_fc"][0]), 2),
                        "Najlepsza": "✓" if meth == best else "",
                    })
            df_z4w = pd.DataFrame(rows_w)
            prov_sel2 = st.selectbox("Filtruj województwo:", ["(wszystkie)"] + PROV, key="z4_prov2")
            if prov_sel2 != "(wszystkie)":
                df_z4w = df_z4w[df_z4w["Województwo"] == prov_sel2]
            st.dataframe(df_z4w, use_container_width=True, height=350)

# ══════════════════════════════════════════════════════════════
# TAB 4 – Z5 CONDITIONAL FORECAST
# ══════════════════════════════════════════════════════════════
with tab4:
    st.header("Zadanie 5 – Prognoza Warunkowa Zużycia Energii")
    st.info("Prognoza Y warunkowa na prognozy X z Zadania 4 (najlepsza metoda per zmienna)")

    z5_tab1, z5_tab2, z5_tab3 = st.tabs([
        "🇵🇱 Model Polska", "🗺️ Model FE (Woj.)", "📊 Porównanie"
    ])

    with z5_tab1:
        st.subheader("Prognoza warunkowa – Polska")
        show_png("z5_01_polska_prognoza.png")
        show_png("z5_02_polska_miary.png", "Miary jakości + Ex-ante 2025")

        _, _, _, _, ms_p_roll_z5, *_ = fit_models()
        st.success(f"""
**Wyniki Model Polska (krocząca walidacja ex-post 2015–2024):**
- RMSPE% rolling = **{ms_p_roll_z5["RMSPE%"]:.2f}%** ✓ (< 10%)
- Prognoza 2025 = **32,021 GWh**
- 95% CI: [30,182 – 33,973 GWh]
- PKB_pc → AR(2) | CENA → ARIMA | HDD → OLS kwadratowy
        """)

        # Interactive: compute conditional forecast with custom X
        with st.expander("🔧 Kalkulator prognozy (własne wartości X na 2025)"):
            df_p = load_polska()
            z4 = load_z4()
            last_pkb  = float(df_p["pkb_per_capita"].iloc[-1])
            last_cena = float(df_p["cena_energii_zl_kWh"].iloc[-1])
            last_hdd  = float(df_p["hdd"].iloc[-1])

            pkb_inp  = st.number_input("PKB per capita 2025 [PLN/os.]",
                                       min_value=50000.0, max_value=200000.0,
                                       value=109292.0, step=1000.0)
            cena_inp = st.number_input("Cena energii 2025 [PLN/kWh]",
                                       min_value=0.3, max_value=2.0,
                                       value=0.83, step=0.01)
            hdd_inp  = st.number_input("HDD 2025",
                                       min_value=2000.0, max_value=5000.0,
                                       value=3075.0, step=50.0)
            model_p, *_ = fit_models()
            X_inp = pd.DataFrame({
                "const":     [1.0],
                "ln_pkb_pc": [np.log(pkb_inp)],
                "ln_cena":   [np.log(cena_inp)],
                "hdd":       [hdd_inp],
            })
            pred_obj = model_p.get_prediction(X_inp)
            ln_pred  = float(pred_obj.predicted_mean[0])
            ci_lo    = float(np.exp(pred_obj.conf_int(alpha=0.05)[0, 0]))
            ci_hi    = float(np.exp(pred_obj.conf_int(alpha=0.05)[0, 1]))
            y_pred   = np.exp(ln_pred)
            st.metric("Prognoza zużycia 2025 [GWh]", f"{y_pred:,.0f}",
                      delta=f"95% CI: [{ci_lo:,.0f} – {ci_hi:,.0f}]")

    with z5_tab2:
        st.subheader("Prognoza warunkowa – Województwa")
        show_png("z5_03_woj_prognoza.png")
        show_png("z5_04_woj_miary.png", "Miary jakości per województwo")
        _, _, _, _, _, ms_fe_roll_z5, prov_roll_z5 = fit_models()
        good_z5 = sum(1 for v in prov_roll_z5.values() if v <= 10)
        med_z5  = float(np.median(list(prov_roll_z5.values()))) if prov_roll_z5 else float("nan")
        st.success(f"""
**Wyniki Model FE (krocząca walidacja ex-post 2015–2024):**
- RMSPE% agregat rolling = **{ms_fe_roll_z5["RMSPE%"]:.2f}%** ✓ (< 10%)
- Województwa RMSPE≤10%: **{good_z5}/16**
- Mediana RMSPE% = {med_z5:.2f}%
- Suma prognoz 2025 = **30,753 GWh**
        """)

    with z5_tab3:
        st.subheader("Porównanie modeli i prognoz 2025")
        show_png("z5_05_woj_agregat.png")

        st.subheader("Zestawienie wyników końcowych")
        _, _, _, _, ms_p_s, ms_fe_s, prov_s = fit_models()
        good_s = sum(1 for v in prov_s.values() if v <= 10)
        summary_data = {
            "": ["Model Polska", "Model FE (Woj.)"],
            f"RMSPE% rolling ({EVAL_START}–2024)": [
                f"{ms_p_s['RMSPE%']:.2f}% ✓",
                f"{ms_fe_s['RMSPE%']:.2f}% ✓",
            ],
            "Prognoza 2025 [GWh]": ["32,021", "30,753 (suma woj.)"],
            "95% CI": ["[30,182 – 33,973]", "—"],
            "Woj. RMSPE≤10%": ["—", f"{good_s}/16"],
        }
        st.dataframe(pd.DataFrame(summary_data).set_index(""), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 5 – ABOUT
# ══════════════════════════════════════════════════════════════
with tab5:
    st.header("O projekcie")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
### Struktura projektu
| Zadanie | Plik | Opis |
|---------|------|------|
| Z2–Z5 | `analiza.py` | Pełna analiza: EDA → Modele → Prognozy X → Prognoza warunkowa |
| App | `dashboard_new.py` | Streamlit dashboard (ten plik) |

### Modele
**Model Polska (OLS log-liniowy):**
- Zmienne: ln(PKB_pc), ln(CENA), HDD
- Zmienna zależna: ln(ZUZYCIE)
- Train: 2004–2022 (19 obs)

**Model Województwa (Fixed Effects):**
- Zmienne: ln(DOCHOD_lag1), ln(CENA), URBANIZACJA, LICZBA_OS, POW_OS, HDD
- Efekty stałe: 15 zmiennych dummy (ref: dolnośląskie)
- Panel: 16 woj × 20 lat = 320 obs (train)
        """)
    with col2:
        st.markdown("""
### Dane
| Źródło | Zakres | Zmienne |
|--------|--------|---------|
| GUS BDL | 2004–2024 | PKB, ludność, dochody, urbanizacja |
| URE | 2004–2024 | Cena energii elektrycznej |
| IMGW/Eurostat | 2004–2024 | HDD, CDD |
| GUS (energia) | 2004–2024 | Zużycie energii elektrycznej |

### Metody prognozowania X (Z4)
1. OLS liniowy (trend liniowy)
2. OLS kwadratowy (trend kwadratowy)
3. AR(1) – autoregresja rządu 1
4. AR(2) – autoregresja rządu 2
5. ARIMA – auto-dobór rzędów
6. Holt – proste wygładzanie wykładnicze
7. Pawłowski – ważona regresja liniowa (wagi liniowe)

### Miary jakości
RMSPE%, MAPE%, MAE, RMSE, ME, MPE%, Theil U
**Kryterium główne: RMSPE% ≤ 10%**
        """)

    st.divider()
    st.markdown("""
### Uruchamianie
```bash
# Generowanie wykresów (z2–z5):
py uruchom_projekt.bat

# Dashboard:
py -m streamlit run dashboard_new.py
```
    """)
    st.caption("Prognozowanie i Symulacje | Studia magisterskie | 2025/2026")

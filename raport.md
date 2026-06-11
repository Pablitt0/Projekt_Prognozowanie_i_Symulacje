---
title: "Prognoza zużycia energii elektrycznej w Polsce"
output:
  word_document:
    toc: true
    toc_depth: 2
---

**Projekt zaliczeniowy – Prognozowanie i Symulacje**
**Autorzy:** Patryk Siuda, Paweł Cyrocki, Damian Skinder
**Studiowany kierunek:** Analityka Gospodarcza, st. II (tryb online75)

---

## 1. Cel i zakres projektu

Celem projektu jest zbudowanie krótkoterminowej prognozy warunkowej zużycia energii elektrycznej w Polsce na rok 2024, przy wykorzystaniu modeli ekonometrycznych estymowanych na danych historycznych rocznych z lat 2004–2023. Dane zebrano w podziale na województwa. Pochodzą one z różnych źródeł: GUS, GUS BDL, Eurostat.

W naszej hipotezie badawczej zakładamy, że zużycie energii elektrycznej w Polsce można skutecznie prognozować (RMSPE ≤ 10%) za pomocą wybranych modeli ekonometrycznych.

Przebieg realizacji projektu:
1.	Organizacja zespołu projektowego, przydział ról i ustalenie harmonogramu prac.
2.	Wybór tematu, zbieranie danych i wstępna analiza wraz z postawieniem hipotez badawczych.
3.	Budowa modeli ekonometrycznych, porównanie wyników
4.	Rozbudowa bazy danych, poszerzenie analiz oraz usprawnienie modeli i wizualizacji.
5.	Wyznaczenie prognoz zużycia energii i wprowadzanie finalnych poprawek.
6.	Opracowanie wniosków końcowych z raportem i przygotowanie prezentacji.

**Jakość prognoz wyznaczono za pomocą poniższych wskaźników:**
•	Miary jakości (ME, MPE, MAE, MAPE, RMSE, RMSPE, U-Theil (i jego składowe: UM, UV, UC)) obliczane są w porównaniu do danych rzeczywistych z lat 2004–2023.
•	RMSPE stanowi główne kryterium wyboru najlepszej metody prognozowania zmiennych – prognozę uznajemy za trafną dla poziomu RMSPE ≤ 10%.
•	Trafność prognoz ex-ante na rok 2024 oceniana jest przez porównanie z wartościami rzeczywistymi (błąd absolutny i względny)

---

## 2. Dane

### 2.1 Źródła i opis zmiennych

| Symbol | Nazwa | Jedn. | Źródło | Rola |
|--------|-------|-------|--------|------|
| zuzycie_energii_GWh | Zużycie energii elektrycznej | GWh | GUS / URE | Y – zm. objaśniana |
| PKB_pc (lub pkb_per_capita) | PKB per capita | PLN/os | GUS BDL | X – model Polska |
| cena-energii_zl_kWh | Cena energii elektrycznej | PLN/kWh | URE / GUS | X – oba modele |
| HDD | Stopniodni grzewcze | °C·dzień | IMGW / Eurostat | X – oba modele |
| dochod_os | Dochód rozp. na osobę | PLN/os | GUS BDL | X – model FE woj. |
| urbanizacja_pct | Stopień urbanizacji | % | GUS BDL | X – model FE woj. |
| liczba_os | Śr. liczba osób w gosp. dom. | osoby | GUS | X – model FE woj. |
| pow_os | Pow. użytkowa mieszk. na os. | m²/os | GUS BDL | X – model FE woj. |

> **Uwaga:** Zmienna CDD (stopniodni chłodnicze) została usunięta z modeli po uzyskaniu nieistotnego statystycznie współczynnika (p > 0,5), co wskazuje na jej marginalną rolę w polskich warunkach klimatycznych.

### 2.2 Statystyki opisowe – Polska (2004–2024, n=21)

| Zmienna | Średnia | Odch. std. | Min | Max |
|---------|---------|-----------|-----|-----|
| Zużycie energii [GWh] | 28 229 | 2 020 | 22 804 | 32 661 |
| PKB per capita [PLN/os] | 50 155 | 20 682 | 24 554 | 97 453 |
| Cena energii [PLN/kWh] | 0,63 | 0,15 | 0,41 | 1,01 |
| HDD [°C·dzień] | 3 283 | 269 | 2 738 | 3 923 |

**Główne obserwacje z EDA:**
- Zużycie energii wzrosło o **+43,2%** w latach 2004–2024 (z 22 804 do 32 661 GWh)
- PKB per capita wzrósł ponad trzykrotnie (24 554 → 97 453 PLN/os), wskazując na silny wzrost gospodarczy
- Cena energii wykazuje skok w 2022–2023 (kryzys energetyczny), osiągając 1,01 PLN/kWh
- HDD charakteryzuje się umiarkowaną zmiennością (CV ≈ 8%), co odzwierciedla naturalną zmienność klimatyczną

### 2.3 Dane panelowe – województwa (16 × 21 = 336 obserwacji)

Dane obejmują 16 województw w latach 2004–2024. Zmienne objaśniające dla modelu FE różnią się od modelu Polska, gdyż dostępność danych jest odmienna na poziomie regionalnym:
- Zamiast PKB per capita zastosowano **dochód rozporządzalny na osobę** (dane GUS BDL)
- Dodano zmienne strukturalne: urbanizację, przeciętną liczbę osób w gospodarstwie domowym oraz powierzchnię mieszkania na osobę

---

## 3. Modele ekonometryczne (Zadanie 3)

### 3.1 Model Polska – log-liniowy OLS

**Specyfikacja:**
$$\ln(\text{ZUZYCIE}_t) = \beta_0 + \beta_1 \ln(\text{PKB\_pc}_t) + \beta_2 \ln(\text{CENA}_t) + \beta_3 \cdot \text{HDD}_t + \varepsilon_t$$

**Wyniki estymacji (train 2004–2023, n=20):**

| Parametr | Estymacja | Błąd std. | t | p-value | Ist. |
|----------|-----------|-----------|---|---------|------|
| stała (β₀) | 9,3854 | 1,018 | 9,218 | 0,000 | *** |
| ln(PKB_pc) (β₁) | 0,0803 | 0,080 | 1,003 | 0,331 | – |
| ln(CENA) (β₂) | 0,1437 | 0,126 | 1,138 | 0,272 | – |
| HDD (β₃) | 2,00×10⁻⁵ | 4,91×10⁻⁵ | 0,408 | 0,689 | – |

**Miary dopasowania (in-sample 2004–2023):**

| Miara | Wartość |
|-------|---------|
| R² | 0,6980 |
| R² skorygowany | 0,6416 |
| AIC | −68,27 |
| BIC | −64,28 |
| F-statystyka | 12,32 (p = 0,000198) |
| N | 20 |
| **RMSPE% in-sample (2004–2023)** | **3,71% ✓** |

**VIF:**

| Zmienna | VIF |
|---------|-----|
| ln(PKB_pc) | 10,00 |
| ln(CENA) | 8,20 |
| HDD | 1,69 |

**Interpretacja:**
- Model wykazuje istotność globalną (F-test p < 0,001), mimo nieistotnych indywidualnych t-testów
- Przyczyną jest **współliniowość**: VIF(ln_PKB_pc) ≈ 10, VIF(ln_CENA) ≈ 8 – typowy problem krótkich szeregów makroekonomicznych
- PKB i CENA wykazują silne trendy wzrostowe w latach 2004–2023, co utrudnia separację efektów
- RMSPE% = 3,71% mieści się w kryterium ≤ 10% ✓

**Diagnostyka reszt:**

| Test | Wartość | Wniosek |
|------|---------|---------|
| Durbin-Watson | 0,6510 | ! Silna autokorelacja (DW < 1,5) |
| Breusch-Godfrey p | 0,1696 | OK ✓ Brak istotnej autokorelacji |
| Ljung-Box lag=1 p | 0,1021 | OK ✓ |
| Shapiro-Wilk p | 0,0002 | ! Odrzucenie normalności (n=20, wrażliwy test) |
| Breusch-Pagan p | 0,8508 | OK ✓ Brak heteroskedastyczności |
| RESET p | 0,0000 | ! Błędna specyfikacja formy funkcyjnej |
| ADF p (orientacyjny) | 0,0597 | ! Niestacjonarność (orientacyjny przy n=21) |
| Condition number | 379 472 | ! Wysoka multikolinearność |

> **Uwaga diagnostyczna:** Naruszenie normalności reszt przy n=20 jest charakterystyczne dla makroekonomicznych szeregów czasowych z obserwacjami anomalnymi (kryzys 2008–2009, pandemia 2020–2021, kryzys energetyczny 2022–2023). DW sugeruje autokorelację, jednak test BG nie potwierdza jej na poziomie istotności 5%. RESET wskazuje na potencjalne problemy ze specyfikacją, co jest oczekiwane przy silnej multikoliniowości. Model nadal pozostaje użyteczny prognostycznie (RMSPE% = 3,71% ≤ 10%).

---

### 3.2 Model FE (efekty stałe) – dane panelowe

**Specyfikacja:**
$$\ln(\text{ZUZYCIE}_{it}) = \alpha + \mathbf{x}_{it}^T \boldsymbol{\beta} + \sum_{k=2}^{16} \delta_k \cdot d_{ik} + \varepsilon_{it}$$

gdzie $d_{ik}$ to zmienne zero-jedynkowe dla województw (kategoria referencyjna: dolnośląskie).

**Okres estymacji:** 2005–2023 (n = 19 lat × 16 województw = 304 obserwacje; rok 2004 wykluczony ze względu na zmienną opóźnioną)

**Zmienne objaśniające:**
- `ln_dochod_os_lag1` – logarytm dochodu na osobę z opóźnieniem o 1 rok
- `ln_cena` – logarytm ceny energii elektrycznej
- `urbanizacja_pct` – stopień urbanizacji [%]
- `liczba_os` – przeciętna liczba osób w gosp. domowym
- `pow_os` – powierzchnia użytkowa mieszkania na osobę [m²]
- `hdd` – stopniodni grzewcze

**Wyniki estymacji (zmienne główne):**

| Parametr | Estymacja | Błąd std. | t | p-value | Ist. |
|----------|-----------|-----------|---|---------|------|
| stała | 5,8039 | 0,421 | 13,781 | 0,000 | *** |
| ln(Dochód_os lag1) | 0,2305 | 0,038 | 6,110 | 0,000 | *** |
| ln(CENA) | −0,0441 | 0,031 | −1,424 | 0,156 | – |
| Urbanizacja [%] | −0,0026 | 0,004 | −0,716 | 0,474 | – |
| Liczba os. | −0,0104 | 0,034 | −0,309 | 0,758 | – |
| Pow./os. [m²] | −0,0114 | 0,004 | −2,853 | 0,005 | ** |
| HDD | 3,98×10⁻⁵ | 1,11×10⁻⁵ | 3,573 | 0,000 | *** |

**Miary dopasowania (in-sample 2005–2023):**

| Miara | Wartość |
|-------|---------|
| R² | 0,9953 |
| R² skorygowany | 0,9950 |
| AIC | −1 084 |
| BIC | −1 002 |
| F-statystyka | 2 874 (p ≈ 0) |
| N | 304 |
| F-test dummies (p) | 0,000000 – efekty stałe wysoce istotne ✓ |
| **RMSPE% in-sample (2005–2023)** | **3,87% ✓** |
| Woj. RMSPE ≤ 10% (in-sample) | 14/16 |

**Test łącznej istotności efektów stałych:** F(15, 282) = 2 425,9, p ≈ 0 → **efekty stałe wysoce istotne**

**VIF (zmienne główne, bez dummy):**

| Zmienna | VIF |
|---------|-----|
| ln(Dochód_os lag1) | 31,72 |
| ln(CENA) | 6,95 |
| Urbanizacja [%] | 242,73 |
| Liczba os. | 13,18 |
| Pow./os. [m²] | 25,10 |
| HDD | 2,05 |

> **Uwaga:** VIF dla `urbanizacja_pct` jest bardzo wysoki (242,73), co wskazuje na silną współliniowość ze zmiennymi strukturalnymi. Usunięcie tej zmiennej pogarszało jednak jakość prognoz, dlatego pozostawiono ją w modelu z adnotacją o ograniczonej interpretowalności parametru.

**Diagnostyka reszt FE:**

| Test | Wartość | Wniosek |
|------|---------|---------|
| Durbin-Watson (orientacyjny) | 0,7883 | ! Silna autokorelacja (panel) |
| Breusch-Godfrey p | 0,0000 | ! Autokorelacja |
| Ljung-Box lag=4 p | 0,0000 | ! Autokorelacja |
| Shapiro-Wilk p | 0,0000 | ! Odrzucenie normalności |
| Jarque-Bera p | 0,0000 | ! Odrzucenie normalności |
| Breusch-Pagan p | 0,0139 | ! Heteroskedastyczność |
| White p (bez dummy) | 0,0000 | ! Heteroskedastyczność |
| RESET p (orientacyjny) | 0,0549 | OK ✓ Forma funkcyjna akceptowalna |
| F-test dummies p | 0,000000 | FE istotne ✓ |
| Condition number (główne) | 383 784 | ! Wysoka multikolinearność |

> W modelu panelowym autokorelacja i odrzucenie normalności są typowe i oczekiwane przy n=304. Heteroskedastyczność wynika z dużego zróżnicowania województw pod względem zużycia energii. Nie wpływają istotnie na użyteczność prognostyczną modelu – RMSPE% = 3,87% ≤ 10%.

**Porównanie modeli – miary jakości in-sample:**

| Model | R² | R²adj | AIC | BIC | RMSPE% in-sample | Woj./obs. | Status |
|-------|----|-------|-----|-----|-----------------|-----------|--------|
| Model Polska (OLS) | 0,6980 | 0,6416 | −68,3 | −64,3 | 3,71% | 20 obs | OK ✓ |
| Model FE Woj. (panel) | 0,9953 | 0,9950 | −1 084 | −1 002 | 3,87% | 14/16 woj. | OK ✓ |

---

## 4. Prognozy zmiennych objaśniających (Zadanie 4)

Zmienne objaśniające na rok 2024 prognozowano **5 metodami**, wybierając najlepszą na podstawie RMSPE% obliczonego na **wartościach dopasowanych do pełnej próby uczącej 2004–2023**.

### 4.1 Zastosowane metody prognozowania

| Symbol | Metoda | Opis |
|--------|--------|------|
| OLS_lin | Regresja liniowa OLS | Trend liniowy w czasie |
| OLS_kw | Regresja kwadratowa OLS | Trend paraboliczny w czasie |
| ARIMA | Auto-ARIMA | Auto-dobór p,d,q (pmdarima) |
| Holt | Wygładzanie wykładnicze Holta | SimpleExpSmoothing |
| Pawl | Metoda Pawłowskiego | WLS z wagami rosnącymi liniowo |

### 4.2 Wyniki – Model Polska (3 zmienne)

RMSPE% in-sample 2004–2023 dla każdej metody (* = najlepsza):

| Metoda | PKB per capita | Cena energii | HDD |
|--------|---------------|-------------|-----|
| OLS_lin | 9,62% | 8,38% | 5,76% |
| OLS_kw | **7,09%*** | 8,08% | **5,76%*** |
| ARIMA | 28,17% | 21,72% | 23,19% |
| Holt | 7,14% | **6,80%*** | 6,95% |
| Pawl | 17,51% | 8,58% | 5,79% |

| Zmienna | Najlepsza metoda | RMSPE% | FC 2024 |
|---------|-----------------|--------|---------|
| PKB per capita [PLN/os] | OLS_kw | 7,09% ✓ | 89 289 |
| Cena energii [PLN/kWh] | Holt | 6,80% ✓ | 0,910 |
| HDD [°C·dzień] | OLS_kw | 5,76% ✓ | 3 031 |

**Pełne miary jakości prognoz zmiennych X – najlepsza metoda, in-sample 2004–2023:**

| Miara | PKB per capita (OLS_kw) | Cena energii (Holt) | HDD (OLS_kw) |
|-------|------------------------|---------------------|--------------|
| ME | −0,00 | 0,03 | 0,00 |
| MPE% | −0,34% | 3,74% | −0,34% |
| MAE | 2 865,63 | 0,04 | 152,21 |
| MAPE% | 6,24% | 5,53% | 4,56% |
| RMSE | 3 309,68 | 0,05 | 198,13 |
| RMSPE% | **7,09%** | **6,80%** | **5,76%** |
| TheilU | 0,0325 | 0,0392 | 0,0299 |
| UM | 0,0000 | 0,2729 | 0,0000 |
| UV | 0,0090 | 0,0773 | 0,2855 |
| UC | 0,9910 | 0,6497 | 0,7145 |

> **Interpretacja:** Wszystkie trzy prognozy spełniają kryterium RMSPE% ≤ 10%. TheilU bliski 0 świadczy o dobrej jakości. Składowe: **UM** = udział obciążenia (bias) – PKB i HDD ≈ 0 (brak systematycznego błędu), CENA = 0,27 (umiarkowane obciążenie); **UV** = niedopasowanie wahań; **UC** = zgodność kierunku – PKB i CENA dominuje UC (≥ 0,65), co oznacza dobrą odpowiedź na kierunek zmian.

### 4.3 Analiza trafności prognoz X (prognoza 2024 vs rzeczywiste 2024)

| Zmienna | Metoda | Prognoza 2024 | Rzeczywiste 2024 | Błąd abs. | Błąd wzgl. |
|---------|--------|--------------|-----------------|-----------|-----------|
| PKB per capita [PLN/os] | OLS_kw | 89 289 | 97 453 | −8 164 | −8,38% |
| Cena energii [PLN/kWh] | Holt | 0,910 | 1,010 | −0,100 | −9,90% |
| HDD [°C·dzień] | OLS_kw | 3 031 | 2 738 | +294 | +10,74% |

> Wszystkie trzy zmienne prognozowane są z błędem poniżej 11%. Prognozy PKB i Ceny są niedoszacowane – model nie uchwycił w pełni dalszego wzrostu PKB oraz utrzymania wysokich cen energii w 2024 roku. HDD jest przeszacowane (przewidziano zimniejszy rok niż był w rzeczywistości).

### 4.4 Wyniki – Model FE województwa (6 zmiennych)

Mediana RMSPE% po 16 województwach na próbie uczącej 2004–2023:

| Zmienna | Mediana RMSPE% | Woj. ≤ 10% |
|---------|---------------|-----------|
| Dochód na osobę [PLN] | 6,78% ✓ | 16/16 |
| Cena energii [PLN/kWh] | 6,92% ✓ | 16/16 |
| Urbanizacja [%] | 0,14% ✓ | 16/16 |
| Liczba osób w gosp. | 1,75% ✓ | 16/16 |
| Pow. mieszk. na os. [m²] | 0,87% ✓ | 16/16 |
| HDD [°C·dzień] | 5,90% ✓ | 16/16 |

> Wszystkie 6 zmiennych i wszystkie 16 województw spełniają kryterium RMSPE% ≤ 10%. Zmienne strukturalne (urbanizacja, liczba osób, pow. mieszk.) prognozują się z minimalnym błędem dzięki ich wolnozmiennej naturze.

---

## 5. Prognoza warunkowa (Zadanie 5)

### 5.1 Metodologia

Prognozę warunkową zużycia energii na rok **2024** wyznaczono w dwóch krokach:

1. **Prognoza zmiennych X** (Zadanie 4) → wartości oczekiwane na 2024
2. **Podstawienie prognozowanych X** do modelu ekonometrycznego (Zadanie 3):
   - Model Polska: `ŷ₂₀₂₄ = exp(β̂₀ + β̂₁·ln(x̂₁) + β̂₂·ln(x̂₂) + β̂₃·x̂₃)`
   - Model FE: dla każdego województwa z osobna, następnie agregacja do sumy krajowej

Przedziały ufności obliczono metodą analityczną:
$$\text{SE}_{\hat{y}} = \sqrt{\hat{\sigma}^2 \cdot \left(1 + \mathbf{x}_0^T (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{x}_0\right)}$$

### 5.2 Wyniki – Model Polska

**Prognoza ex-ante 2024:** **31 193 GWh**, 95% CI: [29 836 – 32 612 GWh]

**Miary jakości dopasowania modelu Polska (in-sample 2004–2023):**

- RMSPE% in-sample = **3,71%** ✓ (kryterium ≤ 10% spełnione)
- MAPE% in-sample = 2,50%
- R² = 0,698

> Szczegółowe miary jakości prognoz zmiennych objaśniających (ME, MPE%, MAE, MAPE%, RMSE, RMSPE%, TheilU, UM, UV, UC) zawarte są w sekcji 4.2.

### 5.3 Wyniki – Model FE (agregacja 16 województw)

**Prognoza ex-ante 2024 (suma): 30 248 GWh**

| Województwo | RMSPE% | Status | FC 2024 [GWh] |
|-------------|--------|--------|---------------|
| Dolnośląskie | 6,9% | ✓ | 2 277 |
| Kujawsko-pomorskie | 8,0% | ✓ | 1 600 |
| Lubelskie | 16,2% | ✗ | 1 460 |
| Lubuskie | 5,3% | ✓ | 794 |
| Mazowieckie | 2,3% | ✓ | 4 922 |
| Małopolskie | 5,7% | ✓ | 2 844 |
| Opolskie | 6,8% | ✓ | 826 |
| Podkarpackie | 13,1% | ✗ | 1 244 |
| Podlaskie | 7,9% | ✓ | 937 |
| Pomorskie | 3,5% | ✓ | 1 772 |
| Warmińsko-mazurskie | 8,1% | ✓ | 1 039 |
| Wielkopolskie | 1,4% | ✓ | 2 724 |
| Zachodniopomorskie | 4,0% | ✓ | 1 234 |
| Łódzkie | 5,7% | ✓ | 2 010 |
| Śląskie | 9,0% | ✓ | 3 783 |
| Świętokrzyskie | 7,6% | ✓ | 782 |
| **SUMA** | **med. 6,80%** | **14/16 ✓** | **30 248** |

> Lubelskie (16,2%) i Podkarpackie (13,1%) przekraczają kryterium 10% – oba województwa charakteryzują się dużą zmiennością struktury zużycia energii w badanym okresie.

### 5.4 Analiza trafności ex-ante – prognoza Y 2024 vs rzeczywiste 2024

| Model | Prognoza 2024 [GWh] | Rzeczywiste 2024 [GWh] | Błąd abs. [GWh] | Błąd wzgl. [%] | ≤5%? |
|-------|--------------------|-----------------------|-----------------|----------------|------|
| Model Polska (OLS) | 31 193 | 32 661 | −1 468 | −4,49% | ✓ |
| Model FE (województwa) | 30 248 | 32 661 | −2 413 | −7,39% | ✗ |

> Oba modele niedoszacowują zużycia energii w 2024 roku, co wynika z wyższego wzrostu PKB i utrzymania wysokich cen energii niż zakładały prognozy. Model Polska spełnia kryterium ≤ 5% błędu względnego (−4,49%), model FE nie spełnia (−7,39%).

### 5.5 Porównanie obu modeli

| Miara | Model Polska | Model FE Woj. |
|-------|-------------|--------------|
| R² (train 2004–2023) | 0,6980 | 0,9953 |
| RMSPE% in-sample | 3,71% ✓ | 3,87% ✓ |
| FC 2024 [GWh] | **31 193** | **30 248** |
| Różnica FC | — | −945 GWh (−3,0%) |
| Błąd wzgl. vs rzecz. 2024 | **−4,49% ✓** | −7,39% ✗ |
| Woj. ≤ 10% | — | 14/16 |

Oba modele spełniają kryterium RMSPE% ≤ 10% na próbie uczącej. Model Polska lepiej trafił prognozę 2024 (błąd −4,49%), natomiast model FE dostarcza dodatkowej informacji na poziomie regionalnym (16 województw).

---

## 6. Wnioski

### 6.1 Prognoza na 2024 rok

Zużycie energii elektrycznej w Polsce w roku 2024 prognozowane było na:
- **31 193 GWh** (Model Polska OLS, 95% CI: 29 836–32 612 GWh)
- **30 248 GWh** (Model FE – agregacja województw)

Wartość rzeczywista wyniosła **32 661 GWh**. Oba modele niedoszacowały zużycia energii: Model Polska o 4,49% (błąd mieści się w kryterium ≤ 5%), Model FE o 7,39%.

### 6.2 Kluczowe czynniki wzrostu

1. **Wzrost PKB per capita** – główny czynnik długoterminowego wzrostu; elastyczność dochodowa β₁ = 0,08 (model Polska)
2. **Ceny energii** – wyższe ceny mogą hamować konsumpcję (β₂ = 0,14); sygnał osłabiony multikoliniowością
3. **Warunki klimatyczne (HDD)** – silniejsze zimy zwiększają zużycie energii grzewczej
4. **Dochód rozporządzalny (model FE)** – elastyczność dochodowa β = 0,23, istotna statystycznie (p < 0,001)

### 6.3 Ograniczenia modeli

1. **Krótki szereg czasowy** (n=20) ogranicza precyzję estymacji i moc testów diagnostycznych
2. **Kryzys cen energii 2022–2023** – bezprecedensowy wzrost cen jest trudny do uchwycenia przez modele trendu; błąd prognozy ceny wynosi −9,90%
3. **Multikoliniowość** – PKB i CENA wykazują silne trendy (VIF ≈ 8–10), co utrudnia izolację ich wpływów, choć nie wpływa negatywnie na RMSPE%
4. **Autokorelacja reszt** – charakterystyczna dla rocznych szeregów makroekonomicznych; nie korygowana, gdyż nie jest wymagana kursowo
5. **Efekty strukturalne** – transformacja energetyczna i odnawialne źródła energii mogą zmieniać zależności strukturalne w kolejnych latach

### 6.4 Ocena modeli według kryterium RMSPE% ≤ 10%

| Model | RMSPE% in-sample (2004–2023) | Błąd ex-ante 2024 | Status |
|-------|-----------------------------|--------------------|--------|
| Model Polska (OLS log-liniowy) | **3,71%** | −4,49% | ✓ Spełnia kryterium |
| Model FE Województwa | **3,87%** | −7,39% | ✓ In-sample / ✗ Trafność 2024 |

---

## 7. Pliki projektu

| Plik | Opis |
|------|------|
| `analiza.py` | Główny skrypt analityczny (Zadania 2–5) |
| `_md2docx.py` | Konwerter raport.md do raport.docx |
| `uruchom_projekt.bat` | Skrypt uruchomieniowy (CLI) |
| `Zuzycie_energii_polska.xlsx` | Dane dla modelu Polska (2004–2024) |
| `Zuzycie_energii_wojewodztwa.xlsx` | Dane panelowe dla 16 województw (2004–2024) |
| `z2_01..z2_09_*.png` | Wykresy EDA (analiza opisowa) |
| `z3_01..z3_07_*.png` | Wykresy modeli ekonometrycznych |
| `z4_01..z4_05_*.png` | Tabele RMSPE prognoz zmiennych X |
| `z4_09_*.png` | Analiza trafności prognoz X (2024 vs rzecz.) |
| `z5_01..z5_06_*.png` | Wykresy prognoz warunkowych Y i analiza trafności |

### Uruchomienie

```bash
py analiza.py
py _md2docx.py
```

---

## 8. Metodologia – szczegóły techniczne

### 8.1 Transformacja logarytmiczna

Obie zmienne objaśniające (PKB_pc, CENA) oraz zmienna objaśniana (ZUZYCIE) poddano transformacji logarytmicznej. Uzasadnienie:
- Relacja ekonomiczna między PKB a zużyciem energii ma charakter elastyczności stałej
- Logarytmowanie stabilizuje wariancję przy rosnących wartościach szeregów
- Współczynniki β₁, β₂ interpretowane są bezpośrednio jako elastyczności

Prognoza na oryginalnej skali: `Y_hat = exp(y_hat)`

### 8.2 Efekty stałe (Fixed Effects)

Model FE zastosowano zamiast modelu RE (Random Effects) ze względu na:
- Wysoce istotny F-test łącznej istotności efektów stałych: F(15, 282) = 2 425,9, p ≈ 0
- Korelację nieobserwowalnych efektów indywidualnych z zmiennymi objaśniającymi

### 8.3 Zmienna opóźniona

Zmienną `ln_dochod_os` zastosowano z opóźnieniem o jeden rok (`lag1`), co:
- Eliminuje potencjalną endogeniczność
- Modeluje realny mechanizm transmisji: zmiana dochodów wpływa na decyzje energetyczne z opóźnieniem (zakup urządzeń, zmiana standardu mieszkaniowego)
- Powoduje wykluczenie roku 2004 ze zbioru uczącego FE (efektywny train: 2005–2023, n=304)

### 8.4 Selekcja zmiennych i metod

- **CDD** usunięte: p > 0,5 – chłodzenie elektryczne ma marginalną rolę w Polsce
- **Ludność** nieuwzględniona jako regressor – silna korelacja z PKB, nie poprawia jakości prognoz
- **AR1 i AR2** usunięte z metod prognozowania – zawierają się w Auto-ARIMA, która automatycznie dobiera optymalny rząd procesu

### 8.5 Miary jakości prognoz

| Symbol | Nazwa | Formuła |
|--------|-------|---------|
| ME | Średni błąd | mean(e_t) |
| MPE% | Średni błąd procentowy | mean(e_t / y_t) × 100 |
| MAE | Średni błąd bezwzględny | mean(abs(e_t)) |
| MAPE% | Średni abs. błąd procentowy | mean(abs(e_t / y_t)) × 100 |
| RMSE | Pierwiastek MSE | sqrt(mean(e_t²)) |
| RMSPE% | Pierwiastek MSE procentowy | sqrt(mean((e_t / y_t)²)) × 100 |
| TheilU | Statystyka Theila U | sqrt(sum e²) / (sqrt(sum y²) + sqrt(sum ŷ²)) |
| UM | Obciążoność (bias) | (mean_hat − mean_act)² / MSE |
| UV | Niedopasowanie wahań | (sd_hat − sd_act)² / MSE |
| UC | Niedopasowanie kierunku | 2(1−r)·sd_act·sd_hat / MSE |

gdzie e_t = y_t − ŷ_t, oraz UM + UV + UC = 1.

---

*Projekt zrealizowany w języku Python 3.12. Główne biblioteki: NumPy, Pandas, Statsmodels, Matplotlib, Seaborn, pmdarima.*

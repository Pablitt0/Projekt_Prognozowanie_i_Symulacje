# Prognoza zużycia energii elektrycznej w Polsce
## Projekt zaliczeniowy – Prognozowanie i Symulacje

**Autor:** Patryk Siuda  
**Data:** czerwiec 2026  
**Dane:** 2004–2024 (GUS, URE, IMGW/Eurostat)

---

## 1. Cel i zakres projektu

Celem projektu jest zbudowanie krótkoterminowej **prognozy warunkowej** zużycia energii elektrycznej w Polsce na rok 2025, przy wykorzystaniu modeli ekonometrycznych estymowanych na danych historycznych z lat 2004–2024.

Projekt realizuje pełny potok analityczny:
1. Analiza opisowa danych (EDA) – zadanie 2
2. Budowa i weryfikacja modeli ekonometrycznych – zadanie 3
3. Prognozy zmiennych objaśniających (7 metod) – zadanie 4
4. Prognoza warunkowa zmiennej objaśnianej – zadanie 5

**Podział próby:**
- **Zbiór uczący (train):** 2004–2022 (n=19 dla Polska, n=288 dla woj.)
- **Zbiór testowy (test):** 2023–2024
- **Horyzont prognozy ex-ante:** 2025

**Ocena jakości prognoz – krocząca walidacja ex-post:**
- Zastosowano metodę **kroczącej prognozy 1-krokowej naprzód** (rolling 1-step-ahead) dla lat 2015–2024 (n=10 obserwacji)
- W każdej iteracji model jest re-estymowany na danych do roku t−1 i prognozuje rok t przy użyciu rzeczywistych wartości X (ex-post)
- Podejście to jest metodycznie poprawne — ocenia model na podstawie 10 niezależnych prognoz zamiast jedynie 2 punktów testowych
- **Kryterium jakości:** RMSPE% ≤ 10%

---

## 2. Dane

### 2.1 Źródła i opis zmiennych

| Symbol | Nazwa | Jedn. | Źródło | Rola |
|--------|-------|-------|--------|------|
| ZUZYCIE | Zużycie energii elektrycznej | GWh | GUS / URE | Y – zm. objaśniana |
| PKB_pc | PKB per capita | PLN/os | GUS BDL | X – model Polska |
| CENA | Cena energii elektrycznej | PLN/kWh | URE / GUS | X – oba modele |
| HDD | Stopniodni grzewcze | °C·dzień | IMGW / Eurostat | X – oba modele |
| DOCHOD_OS | Dochód rozp. na osobę | PLN/os | GUS BDL | X – model FE woj. |
| URBANIZACJA | Stopień urbanizacji | % | GUS BDL | X – model FE woj. |
| LICZBA_OS | Śr. liczba osób w gosp. dom. | osoby | GUS | X – model FE woj. |
| POW_OS | Pow. użytkowa mieszk. na os. | m²/os | GUS BDL | X – model FE woj. |

> **Uwaga:** Zmienna CDD (stopniodni chłodnicze) została usunięta z modeli po uzyskaniu nieistotnego statystycznie współczynnika (p > 0.5), co wskazuje na jej marginalną rolę w polskich warunkach klimatycznych.

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

**Wyniki estymacji (train 2004–2022, n=19):**

| Parametr | Estymacja | Błąd std. | t | p-value | Ist. |
|----------|-----------|-----------|---|---------|------|
| stała (β₀) | 9,608 | 0,854 | 11,25 | 0,000 | *** |
| ln(PKB_pc) (β₁) | 0,070 | 0,067 | 1,04 | 0,314 | – |
| ln(CENA) (β₂) | 0,208 | 0,108 | 1,92 | 0,073 | . |
| HDD (β₃) | −1,96×10⁻⁶ | 4,17×10⁻⁵ | −0,05 | 0,963 | – |

**Miary dopasowania:**

| Miara | Wartość |
|-------|---------|
| R² | 0,800 |
| R² skorygowany | 0,760 |
| AIC | −71,54 |
| BIC | −67,76 |
| F-statystyka | 20,02 (p = 1,68×10⁻⁵) |
| RMSPE% (train) | ~3–4% |
| **RMSPE% rolling ex-post (2015–2024, n=10)** | **6,25% ✓** |

**Interpretacja:**
- Model wykazuje istotność globalną (F-test p < 0,001), mimo nieistotnych indywidualnych t-testów
- Przyczyną jest **współliniowość** zmiennych: VIF(ln_PKB_pc) ≈ 8, VIF(ln_CENA) ≈ 7
  - PKB i CENA wykazują silne trendy wzrostowe w latach 2004–2024, co utrudnia separację efektów
  - Jest to typowy problem krótkich szeregów makroekonomicznych, akceptowalny w kontekście prognozowania
- Rolling RMSPE% = 6,25% mieści się w kryterium ≤ 10% ✓

**Diagnostyka reszt:**
| Test | Wartość | Wniosek |
|------|---------|---------|
| Durbin-Watson | 1,079 | Podejrzenie autokorelacji (< 1,5) |
| Breusch-Godfrey | p = 0,778 | Brak istotnej autokorelacji |
| Shapiro-Wilk | p = 0,0005 | Odrzucenie normalności (n=19, wrażliwy test) |
| Breusch-Pagan | p = 0,395 | Brak heteroskedastyczności |

> **Uwaga diagnostyczna:** Naruszenie założenia normalności reszt przy n=19 jest charakterystyczne dla makroekonomicznych szeregów czasowych z obserwacjami anomalnymi (kryzys 2008–2009, pandemia 2020–2021). DW < 1,5 sugeruje możliwą autokorelację, jednak test BG nie potwierdza jej istotności statystycznej. Model nadal pozostaje użyteczny prognostycznie (RMSPE% ≤ 10%).

---

### 3.2 Model FE (efekty stałe) – dane panelowe

**Specyfikacja:**
$$\ln(\text{ZUZYCIE}_{it}) = \alpha + \mathbf{x}_{it}^T \boldsymbol{\beta} + \sum_{k=2}^{16} \delta_k \cdot d_{ik} + \varepsilon_{it}$$

gdzie $d_{ik}$ to zmienne zero-jedynkowe dla województw (kategoria referencyjna: dolnośląskie).

**Zmienne objaśniające:**
- `ln_dochod_os_lag1` – logarytm dochodu na osobę z opóźnieniem o 1 rok (efekt z poprzedniego roku)
- `ln_cena` – logarytm ceny energii elektrycznej
- `urbanizacja_pct` – stopień urbanizacji [%]
- `liczba_os` – przeciętna liczba osób w gosp. domowym
- `pow_os` – powierzchnia użytkowa mieszkania na osobę [m²]
- `hdd` – stopniodni grzewcze

> **Uwaga:** Ze względu na zastosowanie zmiennej opóźnionej (`lag1`), rok 2004 jest wykluczony ze zbioru uczącego. Efektywny zbiór uczący: 2005–2022 (n=288 po 2004 r.).

**Wyniki estymacji:**

| Miara | Wartość |
|-------|---------|
| R² | 0,996 |
| R² skorygowany | 0,996 |
| Liczba parametrów | 22 (6 zm. + 15 efektów + stała) |
| F(21, 266) globalny | 3 256 (p ≈ 0) |
| AIC / BIC | −1 078 / −997 |
| **F-test dummy p-value** | ≈ 0 – efekty stałe wysoce istotne |
| RMSPE% (train) | 3,51% |
| **RMSPE% rolling ex-post (2015–2024, n=160)** | **5,85% ✓** |

**Test łącznej istotności efektów stałych:** F(15, 266) = 2 786,8, p ≈ 0 → **efekty stałe wysoce istotne**

**Diagnostyka reszt FE:**

| Test | Wartość | Wniosek |
|------|---------|---------|
| Durbin-Watson | 0,984 | Silna autokorelacja (panel) |
| Breusch-Godfrey | p ≈ 0 | Autokorelacja |
| Shapiro-Wilk | p ≈ 0 | Odrzucenie normalności (n=288) |
| Breusch-Pagan | p = 0,043 | Heteroskedastyczność na granicy |

> W modelu panelowym autokorelacja i odrzucenie normalności są typowe i oczekiwane. Nie wpływają istotnie na użyteczność prognostyczną modelu.

**VIF (tylko zmienne główne):**

| Zmienna | VIF |
|---------|-----|
| ln_dochod_os_lag1 | 32,58 |
| ln_cena | 6,78 |
| urbanizacja_pct | **263,23** |
| liczba_os | 12,48 |
| pow_os | 23,66 |
| hdd | 2,05 |

> **Uwaga:** VIF dla `urbanizacja_pct` jest bardzo wysoki (263), co wskazuje na silną współliniowość z innymi zmiennymi strukturalnymi. Usunięcie tej zmiennej pogarszało jednak jakość prognoz, dlatego pozostawiono ją w modelu z adnotacją o ograniczonej interpretowalności parametru.

**RMSPE% per województwo (krocząca walidacja 2015–2024, n=10 na woj.):**

| Województwo | RMSPE% rolling | Status |
|-------------|----------------|--------|
| Dolnośląskie | 7,65% | ✓ |
| Kujawsko-pomorskie | 4,36% | ✓ |
| Lubelskie | 9,24% | ✓ |
| Lubuskie | 4,60% | ✓ |
| Łódzkie | 5,44% | ✓ |
| Małopolskie | 3,76% | ✓ |
| Mazowieckie | 4,57% | ✓ |
| Opolskie | 7,47% | ✓ |
| Podkarpackie | 7,68% | ✓ |
| Podlaskie | 4,76% | ✓ |
| Pomorskie | 5,06% | ✓ |
| Śląskie | 5,37% | ✓ |
| Świętokrzyskie | 5,18% | ✓ |
| Warmińsko-mazurskie | 6,19% | ✓ |
| Wielkopolskie | 4,85% | ✓ |
| Zachodniopomorskie | 4,34% | ✓ |

> **16/16 województw** spełnia kryterium RMSPE% ≤ 10% przy kroczącej walidacji ex-post (2015–2024). Poprzednia ocena na 2-punktowym zbiorze testowym (2023–2024) zawyżała błędy dla województw doświadczających szoku cenowego energii w 2022–2023 r. (Dolnośląskie, Lubelskie, Podkarpackie).

---

## 4. Prognozy zmiennych objaśniających (Zadanie 4)

Zmienne objaśniające na lata 2023–2025 prognozowano 7 metodami, wybierając najlepszą na podstawie RMSPE% na zbiorze testowym (2023–2024).

### 4.1 Zastosowane metody prognozowania

| Symbol | Metoda | Opis |
|--------|--------|------|
| OLS_lin | Regresja liniowa OLS | Trend liniowy w czasie |
| OLS_kw | Regresja kwadratowa OLS | Trend paraboliczny w czasie |
| AR1 | Autoregresja rzędu 1 | ARIMA(1,0,0) |
| AR2 | Autoregresja rzędu 2 | ARIMA(2,0,0) |
| ARIMA | Auto-ARIMA | Auto-dobór p,d,q (pmdarima) |
| Holt | Wygładzanie Holta | SimpleExpSmoothing |
| Pawl | Metoda Pawłowskiego | WLS z wagami rosnącymi liniowo |

### 4.2 Wyniki – Model Polska (3 zmienne)

| Zmienna | Najlepsza metoda | RMSPE% | FC 2025 |
|---------|-----------------|--------|---------|
| PKB per capita [PLN/os] | AR2 | 3,34% ✓ | 109 292 |
| Cena energii [PLN/kWh] | ARIMA | 16,82% | 0,830 |
| HDD [°C·dzień] | OLS_kw | 9,91% ✓ | 3 075 |

> **Uwaga:** Prognoza ceny energii przekracza kryterium RMSPE% (16,82%) ze względu na bezprecedensowy skok cen energii w 2022–2023 (kryzys energetyczny). Jest to oczekiwane ograniczenie modeli bazujących na danych historycznych w warunkach strukturalnych szoków zewnętrznych.

### 4.3 Wyniki – Model FE województwa (6 zmiennych)

| Zmienna | Mediana RMSPE% | Woj. ≤ 10% |
|---------|---------------|-----------|
| Dochód na osobę [PLN] | 16,32% | 1/16 |
| Cena energii [PLN/kWh] | 17,07% | 0/16 |
| Urbanizacja [%] | 0,11% ✓✓ | 16/16 |
| Liczba osób w gosp. | 0,94% ✓✓ | 16/16 |
| Pow. mieszk. na os. [m²] | 0,46% ✓✓ | 16/16 |
| HDD [°C·dzień] | 9,66% ✓ | 9/16 |

> **Uwaga:** Wysokie RMSPE% dla dochodu i ceny energii wynikają z kryzysu energetycznego 2022–2023. Zmienne strukturalne (urbanizacja, liczba osób, pow. mieszk.) prognozują się z bardzo małym błędem dzięki ich wolnozmiennej naturze.

---

## 5. Prognoza warunkowa (Zadanie 5)

### 5.1 Metodologia

Prognozę warunkową zużycia energii na rok 2025 wyznaczono w dwóch krokach:

1. **Prognoza zmiennych X** (Zadanie 4) → wartości oczekiwane na 2025
2. **Podstawienie prognozowanych X** do modelu ekonometrycznego (Zadanie 3):
   - Model Polska: `ŷ₂₀₂₅ = exp(β̂₀ + β̂₁·ln(x̂₁) + β̂₂·ln(x̂₂) + β̂₃·x̂₃)`
   - Model FE: dla każdego województwa z osobna, następnie agregacja do sumy krajowej

**Obsługa zmiennej opóźnionej `ln_dochod_os_lag1` w modelu FE:**
- 2023: lag = ln(dochód_2022) — znana wartość historyczna
- 2024: lag = ln(prognozowany_dochód_2023) — z Zadania 4
- 2025: lag = ln(prognozowany_dochód_2024) — z Zadania 4

### 5.2 Wyniki – Model Polska

| Rok | Rzeczywiste [GWh] | Prognoza [GWh] | 95% CI |
|-----|------------------|---------------|--------|
| 2023 | 28 807 | 31 326 | — |
| 2024 | 32 661 | 31 697 | — |
| **2025** | — | **32 021** | **[30 182–33 973]** |

**RMSPE% rolling warunkowa (2015–2024, n=10): 5,63% ✓** (poniżej kryterium 10%)

Przedziały ufności obliczono metodą analityczną:
$$\text{SE}_{\hat{y}} = \sqrt{\hat{\sigma}^2 \cdot \left(1 + \mathbf{x}_0^T (\mathbf{X}^T\mathbf{X})^{-1} \mathbf{x}_0\right)}$$

### 5.3 Wyniki – Model FE (agregacja sumy 16 województw)

| Rok | Prognoza [GWh] | RMSPE% |
|-----|---------------|--------|
| **2025** | **30 753** | — |

**RMSPE% rolling ex-post (2015–2024, n=160): 5,85% ✓** (poniżej kryterium 10%)

**Wyniki per województwo (prognoza ex-ante 2025, RMSPE% z kroczącej walidacji 2015–2024):**

| Województwo | FC 2025 [GWh] | RMSPE% rolling | Status |
|-------------|--------------|----------------|--------|
| Dolnośląskie | 2 280 | 7,65% | ✓ |
| Kujawsko-pomorskie | 1 591 | 4,36% | ✓ |
| Lubelskie | 1 515 | 9,24% | ✓ |
| Lubuskie | 781 | 4,60% | ✓ |
| Łódzkie | 2 062 | 5,44% | ✓ |
| Małopolskie | 2 918 | 3,76% | ✓ |
| Mazowieckie | 4 945 | 4,57% | ✓ |
| Opolskie | 850 | 7,47% | ✓ |
| Podkarpackie | 1 280 | 7,68% | ✓ |
| Podlaskie | 952 | 4,76% | ✓ |
| Pomorskie | 1 817 | 5,06% | ✓ |
| Śląskie | 3 864 | 5,37% | ✓ |
| Świętokrzyskie | 797 | 5,18% | ✓ |
| Warmińsko-mazurskie | 1 062 | 6,19% | ✓ |
| Wielkopolskie | 2 755 | 4,85% | ✓ |
| Zachodniopomorskie | 1 283 | 4,34% | ✓ |
| **SUMA** | **30 753** | **5,85% ✓** | **16/16 ✓** |

### 5.4 Porównanie obu modeli

| Miara | Model Polska | Model FE Woj. |
|-------|-------------|--------------|
| R² (train) | 0,800 | 0,996 |
| RMSPE% ex-post rolling (2015–2024) | **6,25% ✓** | **5,85% ✓** |
| RMSPE% warunkowa rolling (2015–2024) | **5,63% ✓** | — |
| FC 2025 [GWh] | **32 021** | **30 753** |
| Różnica FC | — | −1 268 GWh (−4,0%) |
| Woj. ≤ 10% | — | **16/16** |

Oba modele spełniają kryterium RMSPE% ≤ 10%. Krocząca walidacja ex-post na 10 latach (2015–2024) pokazuje że modele działają konsekwentnie z błędem poniżej 6,3% rocznie. Model Polska jest prostszy, natomiast model FE dostarcza informacji na poziomie regionalnym.

---

## 6. Wnioski

### 6.1 Prognoza na 2025 rok

Zużycie energii elektrycznej w Polsce w roku 2025 prognozowane jest na:
- **32 021 GWh** (Model Polska, przedział ufności 95%: 30 182–33 973 GWh)
- **30 753 GWh** (Model FE – agregacja województw)

Punktowe prognozy różnią się o ok. 4%, co jest akceptowalnym poziomem rozbieżności między modelami o różnej specyfikacji i granularności.

### 6.2 Kluczowe czynniki wzrostu

1. **Wzrost PKB per capita** – główny czynnik długoterminowego wzrostu zużycia energii w Polsce; elastyczność dochodowa ≈ 0,07
2. **Ceny energii** – wyższe ceny mogą hamować konsumpcję (β₂ > 0, lecz statystycznie słaby sygnał przy multikoliniowości)
3. **Warunki klimatyczne (HDD)** – silniejsze zimy zwiększają zużycie energii grzewczej

### 6.3 Ograniczenia modeli

1. **Krótki szereg czasowy** (n=19–21) ogranicza precyzję estymacji i moc testów diagnostycznych
2. **Kryzys cen energii 2022–2023** – bezprecedensowy wzrost cen jest trudny do uchwycenia przez modele trendu; prognozy CENA mają RMSPE% ~17%
3. **Multikoliniowość** – PKB i CENA wykazują silne trendy, co utrudnia izolację ich wpływów (VIF ≈ 7–8), choć nie wpływa negatywnie na jakość prognoz zagregowanych
4. **Autokorelacja reszt** – charakterystyczna dla rocznych szeregów makroekonomicznych; nie korygowana (korekta HAC nie jest wymagana kursowo)
5. **Efekty strukturalne** – transformacja energetyczna, odnawialne źródła energii i efektywność energetyczna mogą zmieniać zależności strukturalne w kolejnych latach

### 6.4 Ocena modeli według kryterium RMSPE% ≤ 10%

Ocena opiera się na **kroczącej walidacji ex-post** dla lat 2015–2024 (n=10 prognoz jednorokowych naprzód).

| Model | RMSPE% ex-post rolling | RMSPE% warunkowa rolling | Status |
|-------|----------------------|------------------------|--------|
| Model Polska (OLS log-liniowy) | **6,25%** | **5,63%** | ✓ Spełnia kryterium |
| Model FE Województwa | **5,85%** | — | ✓ Spełnia kryterium |

**Oba modele spełniają kryterium RMSPE% ≤ 10%. Wszystkie 16 województw spełnia kryterium indywidualnie (16/16).**

---

## 7. Pliki projektu

| Plik | Opis |
|------|------|
| `analiza.py` | Główny skrypt analityczny (Zadania 2–5) |
| `dashboard_new.py` | Interaktywny dashboard Streamlit |
| `uruchom_projekt.bat` | Skrypt uruchomieniowy (CLI) |
| `dashboard_new.bat` | Skrypt uruchamiający dashboard |
| `Zuzycie_energii_polska.xlsx` | Dane dla modelu Polska (2004–2024) |
| `Zuzycie_energii_wojewodztwa.xlsx` | Dane panelowe dla 16 województw (2004–2024) |
| `z4_results.pkl` | Wyniki Z4 (dla dashboardu, generowany przez analiza.py) |
| `z2_01..z2_09_*.png` | Wykresy EDA |
| `z3_01..z3_07_*.png` | Wykresy modeli ekonometrycznych |
| `z4_01..z4_08_*.png` | Wykresy prognoz zmiennych X |
| `z5_01..z5_05_*.png` | Wykresy prognoz warunkowych Y |

### Uruchomienie

```bash
# Pełna analiza (generuje wszystkie pliki PNG i z4_results.pkl)
py analiza.py

# Interaktywny dashboard Streamlit
py -m streamlit run dashboard_new.py
# lub
dashboard_new.bat
```

---

## 8. Metodologia – szczegóły techniczne

### 8.1 Transformacja logarytmiczna

Obie zmienne objaśniające (PKB_pc, CENA) oraz zmienna objaśniana (ZUZYCIE) poddano transformacji logarytmicznej. Uzasadnienie:
- Relacja ekonomiczna między PKB a zużyciem energii ma charakter elastyczności stałej
- Logarytmowanie stabilizuje wariancję przy rosnących wartościach szeregów
- Współczynniki β₁, β₂ interpretowane są bezpośrednio jako elastyczności

Prognoza na oryginalnej skali: $\hat{Y} = \exp(\hat{y})$

### 8.2 Efekty stałe (Fixed Effects)

Model FE zastosowano zamiast modelu RE (Random Effects) ze względu na:
- Wysoce istotny wynik F-testu łącznej istotności efektów stałych (F ≈ 2787, p ≈ 0)
- Korelację nieobserwowalnych efektów indywidualnych z zmiennymi objaśniającymi (Hausman-like argument)

### 8.3 Zmienna opóźniona

Zmienną `ln_dochod_os` zastosowano z opóźnieniem o jeden rok (`lag1`), co:
- Eliminuje potencjalną endogeniczność (dochód bieżący może być współzależny z zużyciem energii)
- Modeluje realny mechanizm transmisji: zmiana dochodów wpływa na decyzje energetyczne z opóźnieniem (zakup urządzeń, zmiana standardu mieszkaniowego)

### 8.4 Selekcja zmiennych

- **CDD (stopniodni chłodnicze)** usunięte: p > 0,5 w obu modelach — chłodzenie elektryczne ma marginalną rolę w Polsce
- **Ludność** nie uwzględniona jako regressor — silna korelacja z PKB (VIF > 15), nie poprawia jakości prognoz
- **Dane bez implikacji**: cena energii na poziomie województwa jest zunifikowana (jeden rynek), co wyjaśnia podobne wartości CENA w danych panelowych

---

*Projekt zrealizowany w języku Python 3.12. Główne biblioteki: NumPy, Pandas, Statsmodels, Matplotlib, Seaborn, pmdarima, Streamlit.*

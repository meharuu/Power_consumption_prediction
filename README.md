# Building Load Forecasting — Dacon Electricity Consumption Challenge

**Hourly electricity demand forecasting for 100 South Korean buildings**, June–August 2024.
Final score: **7.50% SMAPE**.

A competition entry built around the observation that a hotel, a school, and a telecom data
centre have almost nothing in common except the units on the y-axis — so the work went into
per-building analysis and feature engineering rather than into a bigger model.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-9ACD32)
![XGBoost](https://img.shields.io/badge/XGBoost-EC3C3C)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)

---

## The problem

Predict hourly power consumption (kWh) for 100 buildings across a summer, given weather
observations and building metadata. Scored by **SMAPE** — symmetric mean absolute percentage
error:

```python
def smape(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) /
                   ((np.abs(y_true) + np.abs(y_pred)) / 2)) * 100
```

The metric choice shapes everything. Because SMAPE is relative, an error of 5 kWh on a small
school costs far more than the same 5 kWh on a department store. Optimising raw RMSE would
quietly trade away accuracy on exactly the buildings the metric cares about most, so
low-consumption buildings needed as much attention as the large ones.

## Why per-building analysis came first

The 100 buildings span **ten types**, and their load shapes are qualitatively different:

| Korean | Type | Load behaviour |
|---|---|---|
| 호텔 | Hotel | Occupancy-driven, weak weekday effect |
| 학교 | School | Sharp weekday/term structure, near-flat on holidays |
| 병원 | Hospital | High baseload, weather-insensitive |
| 백화점 | Department Store | Opening-hours step function |
| IDC(전화국) | IDC (Telecom Centre) | Almost constant — cooling-dominated |
| 아파트 | Apartment | Evening-peaked, residential rhythm |
| 상용 | Commercial | Standard office profile |
| 연구소 | Research Institute | Mixed |
| 공공 | Public Facility | Mixed |
| 건물기타 | Other | Mixed |

A single global model has to reconcile a data centre that never varies with a school that
drops to nothing every weekend. So the first step was `bldgs.ipynb` — individual profiling of
representative buildings (1, 7, 8, 10, 26, 77, 87, 100), plotting daily load curves to find
where the anomalies actually lived before writing any features.

That analysis is what produced the rest: the holiday flags, the period bands, and the
building-type-segmented models all came out of looking at individual load curves rather than
from a generic feature checklist.

## Feature engineering

This is where the score came from. Roughly 30 engineered features across five families.

### Cyclical time encoding

Hour, day, day-of-week, week, and month are each encoded as a sine/cosine pair:

```python
full_df['Hour_sin'] = np.sin(2 * np.pi * full_df['Hour'] / 24)
full_df['Hour_cos'] = np.cos(2 * np.pi * full_df['Hour'] / 24)
```

Raw integers tell a tree that hour 23 and hour 0 are 23 apart when they're actually adjacent.
The sin/cos pair puts them next to each other on a circle, so the model can learn a smooth
daily rhythm instead of splitting awkwardly at midnight. The same logic applies across
weekday boundaries and month ends.

### Korean holiday calendar

Public holidays were encoded explicitly and then **merged with weekends into a single
`is_holiday` flag**:

```python
full_df["is_holiday"] = (
    full_df["Datetime"].dt.normalize().isin(korean_holidays) | (is_weekend == 1)
)
```

Merging them was a deliberate call. For most of these buildings a public holiday and a
Saturday produce the same load shape — the building is closed — so treating them as one
concept gives the model more examples of a single pattern rather than two sparse ones.

### Interaction terms

Weather doesn't act on consumption independently of time or building state, so the
interactions are explicit:

```
Solar_Temp        Solar_Humidity     Solar_Wind
Temp_Humidity     Temp_Hour_sin      Temp_Day_sin
Humidity_Hour_sin Holiday_Temp       Holiday_Solar    Holiday_Hour
Temp2             Humidity2
```

`Temp_Hour_sin` matters because 30 °C at 3pm and 30 °C at 3am mean very different things for
air conditioning load. `Holiday_Temp` matters because a hot day only drives cooling demand in
a building someone is occupying. `Temp2` and `Humidity2` capture the non-linear comfort
response — cooling load rises faster than linearly as temperature climbs.

Gradient-boosted trees can in principle discover interactions themselves, but only by
spending depth on them. Handing them over directly is cheaper and works better on a dataset
of this size.

### Lag and rolling features

```python
bld['lag_1h']  = bld['Power Consumption (kWh)'].shift(1)
bld['MA_3h']   = bld['Power Consumption (kWh)'].rolling(3).mean()
bld['MA_24h']  = bld['Power Consumption (kWh)'].rolling(24).mean()
# plus rolling mean / std / median over 24h, 72h, and 168h windows
```

The three window lengths are chosen to match real periodicities: 24h for the daily cycle,
168h for the weekly one, 72h as a medium-term trend. Rolling **median** alongside mean is
deliberate — the median is robust to the consumption spikes that the mean would smear across
a whole window.

### Regime and building features

Time-of-day flags (`is_Daytime`, `is_Early_Morning`, `is_Night`, `is_Evening`,
`is_Late_Night`, `is_peak_hour`) give trees clean split points, and load-level bands
(`is_low_period`, `is_medium_period`, `is_high_period`) let the model condition on the
operating regime. Building metadata — total floor area, air-conditioned area, solar capacity,
ESS storage capacity — provides the cross-building scale that lets one model serve buildings
of very different sizes.

## Outlier handling

Anomalies were **clipped per building, not dropped**, using the interquartile range:

```python
Q1 = building_data['Power Consumption (kWh)'].quantile(0.25)
Q3 = building_data['Power Consumption (kWh)'].quantile(0.75)
# clip to [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
```

Two decisions in there worth stating:

**Per building, not globally.** A reading that's an extreme outlier for a school is an
ordinary Tuesday for a department store. Global thresholds would delete the top of every
small building's distribution while missing genuine faults in the large ones.

**Clipped, not removed.** Dropping rows from a time series punches holes in it, which
invalidates every lag and rolling feature computed downstream. Clipping keeps the index
continuous and the sequence intact while pulling the extreme value back to a plausible
magnitude. A centred three-point smoothing pass handles isolated spikes similarly.

## Models

LightGBM, XGBoost, CatBoost, and Random Forest were each trained and compared, with
**`TimeSeriesSplit`** for cross-validation rather than a random `KFold` — random folds let
the model train on the future and validate on the past, which leaks badly in a forecasting
problem and produces validation scores that don't survive the leaderboard.

Hyperparameters were tuned with `GridSearchCV`. Gradient-boosted trees dominate here for the
usual reasons: heavy categorical and flag features, non-linear weather interactions, and a
few thousand rows per building — territory where boosting reliably beats both linear models
and anything deep.

## The notebooks

Each notebook is a different modelling configuration, kept separate so approaches could be
compared rather than overwritten:

| Notebook | Purpose |
|---|---|
| `bldgs.ipynb` | Per-building EDA and anomaly profiling — the analysis the rest is built on |
| `new.ipynb` | Global model, full feature set, IQR outlier clipping, rolling windows |
| `new2.ipynb` | Global model variant with `GridSearchCV` tuning |
| `hotel.ipynb` | Hotel-segmented model |
| `school.ipynb` | School-segmented model |
| `try.ipynb` | School-type experiments |

## Data

Not included in this repository — download from the
[Dacon competition page](https://dacon.io):

| File | Contents |
|---|---|
| `train.csv` | Hourly consumption per building, Jun–Aug 2024 |
| `test.csv` | Forecast period |
| `building_info.csv` | Type, floor area, A/C area, solar and ESS capacity |

Column headers are Korean and are renamed on load; `building_type_map` in each notebook
handles the type translation.

## Running it

```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost matplotlib seaborn
```

Place the competition CSVs in the repository root and run `bldgs.ipynb` first for the EDA,
then any modelling notebook.

## Current state

An exploratory competition repository rather than a packaged project. Known rough edges:

- **The winning configuration isn't identified.** Six notebooks, one 7.50% SMAPE — the README
  should say which produced it, and ideally that notebook should be named accordingly.
- **Notebook names carry no information.** `new.ipynb`, `new2.ipynb`, and `try.ipynb` should
  describe their approach.
- **Notebook outputs are committed**, making the repository ~56 MB for ~11k lines of actual
  code. Stripping outputs (`nbstripout`) would cut it by well over 90%.
- **`smape()` is redefined a dozen times** within single notebooks, along with duplicated
  feature-engineering blocks. A shared `utils.py` would remove most of the duplication.
- **No `requirements.txt`**, and no pinned versions.
- **Large commented-out blocks** of superseded feature code remain throughout.
- **`fillna(method="bfill")`** is deprecated in pandas 2.x; use `.bfill()`.

## Possible next steps

- [ ] Identify and rename the notebook that produced 7.50% SMAPE
- [ ] Extract shared feature engineering and the metric into `utils.py`
- [ ] Strip notebook outputs and add `requirements.txt`
- [ ] Train one model per building type and compare against the global model on the same folds
- [ ] Blend the global and segmented predictions
- [ ] Weight training samples by inverse consumption to align the loss with SMAPE
- [ ] Add a per-building SMAPE breakdown to see which buildings carry the error

## License

MIT

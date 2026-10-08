# Dataset Audit & Schema Report
## UCI Autism Spectrum Disorder Screening Adult Dataset (ID: 426)

### 1. Dataset Provenance & Overview
- **Source**: UCI Machine Learning Repository (Dataset ID: 426) / Dr. Fadi Fayez Thabtah
- **Primary Source File**: [`data/autism+screening+adult/Autism-Adult-Data.arff`](file:///d:/or%20project/data/autism+screening+adult/Autism-Adult-Data.arff) & [`data/autism_screening_data.csv`](file:///d:/or%20project/data/autism_screening_data.csv)
- **Total Records**: 704
- **Total Original Columns**: 21
- **Domain**: Adult Autism Spectrum Disorder (ASD) 10-Item Behavioral Screening (AQ-10)

---

### 2. Attribute Schema & Descriptions

| # | Column Name | ARFF Data Type | Description & Values | Missing Values (`?`) | Pipeline Role |
|---|---|---|---|---|---|
| 1 | `A1_Score` | Binary `{0,1}` | AQ-10 Item 1: Notices small sounds when others do not | 0 (0.0%) | Model Feature (AQ-10) |
| 2 | `A2_Score` | Binary `{0,1}` | AQ-10 Item 2: Concentrates more on whole picture than small details | 0 (0.0%) | Model Feature (AQ-10) |
| 3 | `A3_Score` | Binary `{0,1}` | AQ-10 Item 3: Easy to do more than one thing at once | 0 (0.0%) | Model Feature (AQ-10) |
| 4 | `A4_Score` | Binary `{0,1}` | AQ-10 Item 4: If interrupted, can switch back quickly | 0 (0.0%) | Model Feature (AQ-10) |
| 5 | `A5_Score` | Binary `{0,1}` | AQ-10 Item 5: Finds it easy to 'read between lines' | 0 (0.0%) | Model Feature (AQ-10) |
| 6 | `A6_Score` | Binary `{0,1}` | AQ-10 Item 6: Knows how to tell if someone is bored | 0 (0.0%) | Model Feature (AQ-10) |
| 7 | `A7_Score` | Binary `{0,1}` | AQ-10 Item 7: Finds it difficult to work out characters' intentions | 0 (0.0%) | Model Feature (AQ-10) |
| 8 | `A8_Score` | Binary `{0,1}` | AQ-10 Item 8: Likes collecting information about categories | 0 (0.0%) | Model Feature (AQ-10) |
| 9 | `A9_Score` | Binary `{0,1}` | AQ-10 Item 9: Easy to work out what someone is thinking or feeling | 0 (0.0%) | Model Feature (AQ-10) |
| 10 | `A10_Score` | Binary `{0,1}` | AQ-10 Item 10: Difficult to work out people's intentions | 0 (0.0%) | Model Feature (AQ-10) |
| 11 | `age` | Numeric | Age in years (range: 17 to 64; outliers cleaned) | 2 (0.28%) | Model Feature (Numeric) |
| 12 | `gender` | Nominal `{'f', 'm'}` | Biological sex / gender identifier | 0 (0.0%) | Model Feature (Categorical) |
| 13 | `ethnicity` | Nominal | Ethnic background (11 unique categories) | 95 (13.49%) | Model Feature (Categorical) |
| 14 | `jundice` | Nominal `{'no', 'yes'}` | Born with neonatal jaundice condition | 0 (0.0%) | Model Feature (Categorical) |
| 15 | `austim` | Nominal `{'no', 'yes'}` | Immediate family member with pervasive developmental disorder | 0 (0.0%) | Model Feature (Categorical) |
| 16 | `contry_of_res` | Nominal | Country of residence (67 unique countries) | 0 (0.0%) | Model Feature (Categorical) |
| 17 | `used_app_before` | Nominal `{'no', 'yes'}` | Prior usage of screening mobile application | 0 (0.0%) | Model Feature (Categorical) |
| 18 | `result` | Numeric | Sum total of $A_1 + \dots + A_{10}$ ($[0, 10]$) | 0 (0.0%) | **Excluded (Collinear Leakage)** |
| 19 | `age_desc` | Nominal | Constant string (`'18 and more'`) across all rows | 0 (0.0%) | **Excluded (Zero Variance)** |
| 20 | `relation` | Nominal | Person completing the screening (`Self`, `Parent`, etc.) | 95 (13.49%) | Model Feature (Categorical) |
| 21 | `Class/ASD` | Nominal `{'NO', 'YES'}` | Screening outcome: ASD Traits Observed vs Non-ASD | 0 (0.0%) | **Target Variable** |

---

### 3. Target Distribution & Analysis
- **Target Name**: `Class/ASD`
- **Mapping**: `{'NO': 0, 'YES': 1}`
- **Class Breakdown**:
  - `0 (Non-ASD)`: **515 samples (73.15%)**
  - `1 (ASD Traits Observed)`: **189 samples (26.85%)**
- **Imbalance Ratio**: Approximately $2.72 : 1$. Handled during model training via `class_weight='balanced'`.

---

### 4. Leakage Analysis & Feature Selection Rationale

1. **Exclusion of `result`**:
   - `result` represents the exact algebraic summation $\sum_{i=1}^{10} A_i$.
   - Since all 10 individual questionnaire item scores are provided as distinct features to the Random Forest model, retaining `result` introduces redundant collinearity and potential target leakage if clinical threshold rules were correlated with `result`.
2. **Exclusion of `age_desc`**:
   - All 704 records contain identical value `'18 and more'`. It exhibits zero variance and contains zero discriminative information.
3. **Final Feature Set**:
   - **18 Features**: 10 AQ-10 item scores + `age` (numeric) + 7 categorical demographic/history features.

---

### 5. Real Test Subsets Generated
Extracted strictly from the genuine 704-record UCI dataset without fabrication:
- [`data/test_real_25.csv`](file:///d:/or%20project/data/test_real_25.csv): 25 records with `Class/ASD`
- [`data/test_real_50.csv`](file:///d:/or%20project/data/test_real_50.csv): 50 records with `Class/ASD`
- [`data/test_real_100.csv`](file:///d:/or%20project/data/test_real_100.csv): 100 records with `Class/ASD`
- [`data/test_real_25_no_target.csv`](file:///d:/or%20project/data/test_real_25_no_target.csv): 25 records without `Class/ASD` column for blind inference testing.

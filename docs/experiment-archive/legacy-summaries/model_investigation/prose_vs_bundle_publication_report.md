# Publication-Ready Comparison of Prose vs Bundle Representations in HQ and Lambda Feature Families

## Executive summary

This report compares **prose** and **bundle** variants within the **HQ** and **Lambda** feature families using repeated screening runs from the uploaded metrics file. The aim is to answer a narrow question: **what does bundle add over prose, and how does that differ by model architecture?**

Across both HQ and Lambda, **bundle consistently outperforms prose**. However, the **size** of the gain varies sharply by architecture:

* **Logistic regression / ridge:** small but consistent gains
* **XGBoost:** smallest gains overall
* **MLP:** largest gains, especially for classification

The absolute scores matter here. The best-performing models are still the **linear models**:

* **Classification:** logistic regression remains best in absolute terms for both prose and bundle
* **Regression:** ridge remains best in absolute terms for both prose and bundle, with one exception on **Lambda bundle**, where **MLP** slightly edges ridge on **MAE** only

This pattern suggests that bundle likely adds **some genuine incremental signal**, but that the extra representation is not transformative for the strongest models. The larger MLP gains are consistent with a second explanation as well: **bundle may also introduce overlapping representations that nonlinear models can exploit more aggressively.**

## Data and comparison setup

### Representation definitions

* **Prose:** tokenized prose description of the founder/profile data
* **Bundle:** prose **plus** tokenized structured-data representation

For a given HQ or Lambda family, the move from prose to bundle therefore adds a second representation of structured information. This can improve performance by supplying extra signal, but it can also introduce **redundancy** because some underlying facts may already appear in both views.

### Evaluation basis

All values below are the **mean across 4 repeated runs**. Deltas are computed as:

> \*\*Delta = Bundle mean − Prose mean\*\*

For **error metrics** such as **Brier, RMSE, and MAE**, a **negative** delta is an improvement.

\---

## Classification results

### HQ family: absolute values and prose-to-bundle deltas

|Model|ROC-AUC (Prose)|ROC-AUC (Bundle)|Δ ROC-AUC|PR-AUC (Prose)|PR-AUC (Bundle)|Δ PR-AUC|F0.5 (Prose)|F0.5 (Bundle)|Δ F0.5|Brier (Prose)|Brier (Bundle)|Δ Brier|
|-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|Logistic regression|0.9341|0.9387|0.0046|0.792|0.8037|0.0117|0.7773|0.7873|0.01|0.0656|0.0631|-0.0026|
|XGBoost|0.9328|0.9355|0.0027|0.78|0.7863|0.0062|0.7648|0.7706|0.0059|0.1184|0.1149|-0.0035|
|MLP|0.8654|0.8855|0.0201|0.6782|0.707|0.0288|0.6628|0.6873|0.0245|0.0736|0.0692|-0.0044|

**Interpretation.**  
Within HQ classification, the ranking by absolute performance does **not** change when moving from prose to bundle: **logistic regression remains best**, followed by **XGBoost**, then **MLP**. Bundle improves all three models, but the gain is much larger for **MLP** (`Δ ROC-AUC = +0.0201`, `Δ PR-AUC = +0.0288`) than for **logistic regression** (`+0.0046`, `+0.0117`) or **XGBoost** (`+0.0027`, `+0.0062`).

### Lambda family: absolute values and prose-to-bundle deltas

|Model|ROC-AUC (Prose)|ROC-AUC (Bundle)|Δ ROC-AUC|PR-AUC (Prose)|PR-AUC (Bundle)|Δ PR-AUC|F0.5 (Prose)|F0.5 (Bundle)|Δ F0.5|Brier (Prose)|Brier (Bundle)|Δ Brier|
|-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|Logistic regression|0.9349|0.9394|0.0045|0.7922|0.8028|0.0106|0.7745|0.7829|0.0084|0.0659|0.0636|-0.0023|
|XGBoost|0.9275|0.9298|0.0023|0.7769|0.7808|0.0039|0.7596|0.7639|0.0042|0.1272|0.1232|-0.004|
|MLP|0.8657|0.8994|0.0336|0.7096|0.7488|0.0392|0.6859|0.7274|0.0415|0.0746|0.0692|-0.0054|

**Interpretation.**  
The Lambda pattern is even clearer. **Logistic regression** remains the strongest classifier in absolute terms under both prose and bundle, while **XGBoost** again shows only small bundle gains. **MLP** receives the largest lift from bundle (`Δ ROC-AUC = +0.0336`, `Δ PR-AUC = +0.0392`, `Δ F0.5 = +0.0415`), but even after that lift it still does **not** surpass logistic regression in absolute ROC-AUC or PR-AUC.

### Cross-architecture classification comparison

Three points stand out:

1. **Bundle helps every classifier** in both HQ and Lambda.
2. **The best absolute classifier is still logistic regression** in every case.
3. **MLP benefits the most from bundle**, but from a lower baseline.

This combination is important. If bundle contained large amounts of wholly new signal, one might expect more substantial absolute reordering among architectures. Instead, the top architecture remains unchanged, and the largest gains accrue to the most flexible model. That pattern is compatible with **some real incremental signal**, but also with **redundant or partially overlapping representations** that MLP can exploit more easily than linear models.

\---

## Regression results

### HQ family: absolute values and prose-to-bundle deltas

|Model|R² (Prose)|R² (Bundle)|Δ R²|Pearson (Prose)|Pearson (Bundle)|Δ Pearson|RMSE (Prose)|RMSE (Bundle)|Δ RMSE|MAE (Prose)|MAE (Bundle)|Δ MAE|
|-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|Ridge|0.4059|0.4187|0.0128|0.6409|0.6503|0.0094|0.2691|0.2659|-0.0032|0.1849|0.1815|-0.0035|
|XGBoost|0.3502|0.3578|0.0077|0.5969|0.6028|0.006|0.2826|0.2808|-0.0018|0.1985|0.1965|-0.002|
|MLP|0.3404|0.3687|0.0283|0.5867|0.6096|0.0229|0.2826|0.2766|-0.0061|0.1909|0.1859|-0.005|

**Interpretation.**  
For HQ regression, **ridge** is best in absolute terms under both prose and bundle. Bundle improves all three models, but again the largest gains appear in **MLP** (`Δ R² = +0.0283`, `Δ Pearson = +0.0229`). Ridge improves more modestly (`Δ R² = +0.0128`), while **XGBoost** improves the least (`Δ R² = +0.0077`).

### Lambda family: absolute values and prose-to-bundle deltas

|Model|R² (Prose)|R² (Bundle)|Δ R²|Pearson (Prose)|Pearson (Bundle)|Δ Pearson|RMSE (Prose)|RMSE (Bundle)|Δ RMSE|MAE (Prose)|MAE (Bundle)|Δ MAE|
|-|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|-:|
|Ridge|0.4084|0.4211|0.0127|0.64|0.6499|0.0098|0.2683|0.2651|-0.0031|0.1807|0.1775|-0.0032|
|XGBoost|0.3581|0.3653|0.0072|0.6019|0.6076|0.0057|0.2804|0.2787|-0.0017|0.1947|0.193|-0.0017|
|MLP|0.3787|0.4033|0.0246|0.6167|0.6364|0.0197|0.2748|0.2693|-0.0056|0.1808|0.1745|-0.0063|

**Interpretation.**  
The Lambda regression pattern closely mirrors HQ. **Ridge** remains best on **R²**, **Pearson**, and **RMSE** under both prose and bundle, although **MLP** on **Lambda bundle** attains the lowest **MAE** (`0.1745` vs `0.1775` for ridge). As in classification, **XGBoost** shows the smallest prose-to-bundle gains.

### Cross-architecture regression comparison

Again, the largest bundle gains appear for **MLP**, but the strongest absolute model remains **ridge** on most metrics. This suggests that the extra representation in bundle is not changing which model family is best; rather, it offers a modest lift within each family and a larger lift for the most flexible model.

\---

## Stability and repeatability

The results should be interpreted alongside repeat-to-repeat variability. In the underlying file, the strongest linear and XGBoost classification results are very stable across repeats, with **ROC-AUC standard deviations typically around 0.0004 to 0.0006** for logistic regression and similarly small values for XGBoost. **MLP** is notably less stable, with materially larger variation across repeats.

This matters because it strengthens a conservative reading of the bundle effect:

* For **logistic regression and ridge**, the bundle gains are **small but repeatable**
* For **MLP**, the gains are larger but should be treated more cautiously because the model is inherently less stable on this dataset

\---

## Substantive interpretation

The comparison of **absolute values** and **deltas** leads to a more nuanced conclusion than deltas alone.

### What the deltas show

Bundle helps every architecture in both HQ and Lambda.

### What the absolute values show

The best absolute performers remain the **linear models**, not the nonlinear ones. Bundle improves them, but only modestly.

### Combined interpretation

Taken together, the most plausible interpretation is:

> Bundle contains some additional usable information beyond prose alone, but a meaningful portion of its benefit may reflect a second encoding of overlapping founder information rather than wholly independent new signal.

That interpretation is strongest because:

* the gains are **consistent** across architectures
* the gains are **modest** for the best-performing linear models
* the gains are **largest** for MLP, which is the architecture most capable of exploiting duplicated or entangled representations

\---

## Practical implications

### If the goal is prediction

Bundle is defensible. It improves every model family and does not degrade stability for the strongest models.

### If the goal is interpretation

Caution is warranted. Because prose and tokenized structured inputs are derived from overlapping facts, bundle may blur attribution and make it harder to determine whether the improvement is due to:

* genuinely new semantic signal, or
* restating structured information in a second form

\---

## Recommended conclusion for a paper or thesis

> Relative to prose alone, bundle consistently improves performance for both HQ and Lambda feature families across all model architectures. However, the absolute ranking of model families remains largely unchanged: logistic regression is still the strongest classifier and ridge is still the strongest regressor on most metrics. This indicates that bundle contributes some additional predictive signal, but that the gain is modest for the best-performing models and substantially larger for MLP, suggesting that bundle may also introduce overlapping representations that nonlinear models can exploit more strongly.

## Suggested next analysis

To distinguish **complementary signal** from **redundancy**, the next step should be a compact ablation study comparing:

1. HQ/Lambda alone
2. Prose alone
3. Structured tokenization alone
4. Prose + HQ/Lambda
5. Prose + structured tokenization
6. Full bundle

That would show whether the structured-token component contributes unique information after prose and engineered features are already present.


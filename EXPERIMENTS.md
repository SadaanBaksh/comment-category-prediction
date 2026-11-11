# Experiment Log

A record of each approach I tried, what the numbers showed, and what I took away from it. Unless noted otherwise, scores are **Macro-F1 on a 20% stratified holdout** of the 198k training rows. The random seed varied between notebooks (`random_state=0` or `42`), so differences of about ±0.003 are within noise.

---

## 01: Baseline LightGBM · [`01_baseline_lgbm.ipynb`](notebooks/experiments/01_baseline_lgbm.ipynb)

**Setup**
- Word TF-IDF (1–2-grams, 20k features) inside a `ColumnTransformer`, together with scaled numeric features and one-hot identity columns.
- Thread-level aggregates per `post_id`: volume, vote mean/std/sum, engagement, negativity ratio.
- Four small keyword lexicons (violent, moral, group, personal attack) as binary flags.
- LightGBM with 600 trees, 64 leaves and `colsample_bytree=0.8`.

**Results**

| Decision rule | Class 3 F1 | Macro-F1 |
|---|---|---|
| Argmax | 0.58 (recall 0.48) | 0.80 |
| "Class 3 if p > 0.2" | 0.63 (recall 0.66) | 0.8125 |
| Per-class weights tuned with Powell | n/a | **0.8157** |

**Takeaway:** Class 0 was essentially solved (0.96), and Class 3 was the bottleneck. Moving the decision boundary helped straight away, so post-processing became part of every pipeline that followed.

---

## 02: Hierarchical "fuzzy" violence gate · [`02_hierarchical_violence_gate.ipynb`](notebooks/experiments/02_hierarchical_violence_gate.ipynb)

**Hypothesis:** a model that only has to separate violent from non-violent comments should do better on Class 3 than a 4-way model.

**Setup**
- **Model A:** binary LightGBM, Class 3 vs. everything else.
- **Model B:** 3-class LightGBM for Classes 0/1/2, trained with `class_weight="balanced"`.
- Two ways of combining them:
  - **Soft:** rescale p(violent) above a threshold, then share the remaining probability among B's classes.
  - **Hard:** if p(violent) > t, predict 3; otherwise use B's prediction.

**Results**

| Component | Score |
|---|---|
| Model A, Class 3 F1 at threshold 0.2 | 0.61 |
| Model B, 3-class Macro-F1 | 0.87 |
| Combined, soft (best t = 0.05) | 0.692 |
| Combined, hard (t = 0.1) | 0.738 |

**Takeaway:** Even a dedicated binary model topped out at about 0.61–0.63 F1 on Class 3. The main notebook found the same ceiling with Logistic Regression and SVMs. The bottleneck was the **feature representation**, not the model architecture, and splitting into two models added a second place for errors to occur. This result is why I shifted effort to feature engineering.

---

## 03: Stacking Logistic Regression probabilities · [`03_stacking_lr_probabilities.ipynb`](notebooks/experiments/03_stacking_lr_probabilities.ipynb)

**Setup:** 5-fold out-of-fold Logistic Regression on character TF-IDF produced four probability columns. These were added as features to a LightGBM trained on the metadata and lexicon features, with no raw TF-IDF.

**Results:** level-1 LR reached 0.42–0.43 Macro-F1 per fold. The stacked LightGBM reached **0.7605** with a threshold, below the baseline.

**Takeaway:** The level-1 vectoriser was configured as `analyzer='char_wb', ngram_range=(1, 2)`, which means single characters and character pairs. Those carry almost no meaning, so the stacker received weak inputs. The lesson: **check each stage of a stack on its own before combining them.** A weak level-1 model makes the whole stack worse.

---

## 04: Hybrid TF-IDF + "maxed-out" LightGBM · [`04_hybrid_tfidf_lgbm.ipynb`](notebooks/experiments/04_hybrid_tfidf_lgbm.ipynb)

This notebook became the basis of the final solution.

**What changed from the baseline**
- **Elongation squashing:** a regex collapses runs of 3 or more repeated characters ("kiiilll" → "kil").
- **Hybrid vectorisation:** word 1–3-grams (50k, sublinear TF) + `char_wb` 3–5-grams (50k), fitted on train and test text together.
- **χ² SelectPercentile (25%):** 100k → 25k text features.
- **Expanded lexicons:** roughly 2–3× more terms, adding weapons and threat phrases ("blow up", "lock her up"), political groups, and insult vocabulary.
- **New count features:** violence density, `count_you`, `count_they`, and a `targeted_violence_flag` (lethal word AND targeting pronoun).
- **LightGBM:** 3,000 trees, 96 leaves, depth 15, **`colsample_bytree=0.1`**.

**Result:** **0.8277** Macro-F1 after weight optimisation, up from 0.8157.

**⚠️ A bug I caught:** later in this notebook, the threshold optimiser reported **0.99997**. The cause was that `model_final` had been retrained on the *full* training set, which includes `X_valid`, and then scored on `X_valid`. That number is meaningless, and I didn't use it. Since then I've followed one rule: **post-processing parameters are always fitted on predictions from a model that never saw those rows.**

---

## 05: SVD compression + pseudo-labelling · [`05_svd_and_pseudo_labeling.ipynb`](notebooks/experiments/05_svd_and_pseudo_labeling.ipynb)

### 5a. Truncated SVD

**Hypothesis:** compressing the sparse TF-IDF into dense "semantic" components would speed up training and might generalise better.

| Setup | Variance explained |
|---|---|
| 150 components on the full TF-IDF | 11.8% |
| 200 components after χ² filtering | 28.7% |
| 1,000 components on the full TF-IDF | 34.2% |

LightGBM on 200 SVD components + metadata: 0.74 with argmax and **0.7836** after weight tuning, clearly below the sparse version.

**Takeaway:** Toxic-text signal is concentrated in rare, specific n-grams. SVD keeps the directions with the most variance, which are mostly common words, so the discriminative tail gets discarded. For this kind of problem, **sparse features + χ² selection is the better way to reduce dimensions**.

### 5b. Pseudo-labelling

**Setup:** predict on the test set, keep rows above a per-class confidence threshold (0.95 / 0.90 / 0.90 / 0.85), add them to the training data, and retrain.

**Result:** public leaderboard **0.83486 → 0.83349**.

**Takeaway:** Confirmation bias. The confident rows are the ones the model already classifies well, so they add nothing near the hard boundaries, and any confident mistakes get reinforced. I reverted this change.

---

## Final solution · [`final_solution.ipynb`](notebooks/final_solution.ipynb)

This notebook brings everything together and adds the documented EDA and the required three-model comparison:

| Model | Macro-F1 |
|---|---|
| Logistic Regression | 0.7362 |
| SGDClassifier (linear SVM) | 0.7444 |
| LightGBM (500 trees, early-stopped) | 0.8127 |
| LightGBM + optimised class weights | **0.8315** |

Hyperparameters came from a `RandomizedSearchCV` run (10 iterations, 3-fold, scored on `f1_macro`). It's commented out in the notebook because the search takes a long time to run. The final model was retrained on all 198k rows with 3,000 trees (124.75 min).

**Note on class weights:** the class weights `{0: 1.0, 1: 2.0, 2: 1.2, 3: 5.5}` are passed to the linear models' `class_weight` parameter. For LightGBM, the minority-class boost comes entirely from the post-hoc probability multipliers. The submission used the multipliers `[1.248, 2.273, 1.235, 4.461]`, picked by trial and error, rather than the values the optimiser found in this run.

---

## Ideas for next time

- **Transformer embeddings** (e.g. DistilBERT or a fine-tuned RoBERTa) for Class 3, where the meaning of a sentence matters more than its keywords.
- **K-fold out-of-fold threshold tuning** instead of one holdout, so the weights don't overfit a single split.
- **Fixing the stacking experiment** with proper word/char n-gram ranges. A strong linear model plus LightGBM is usually a good ensemble.
- **Focal loss or a custom LightGBM objective** aimed directly at minority-class recall.

# BMI 500 — Homework 3: Model-based Bias Removal Using Synthetic Blood Pressure Data

**Student:** Suyang Xi  
**Email:** sxi5@emory.edu  
**Course:** BMI 500, Lecture 7  
**Selected question:** HW3 (Parts A and B only)

## Research hypothesis

I expected sigmoid models to fit age-related systolic blood pressure (SBP) better and Gaussian models to fit diastolic blood pressure (DBP) better than quadratic models. I also expected that increasing the male fraction in training data would inflate accuracy on similarly imbalanced test sets while reducing female recall. Adding realistic synthetic female samples should reduce that recall gap.

The original, handwritten prediction is preserved in [hypothesis.pdf](hypothesis.pdf).

## Part A — Fitting blood pressure trends over age

I fitted quadratic and sigmoid models to age-related mean SBP, and quadratic and Gaussian models to mean DBP, separately for males and females. Fits were compared using mean squared error (MSE), $R^2$, and Bayesian information criterion (BIC), with lower BIC preferred.

**Important data limitation:** The embedded 20 sex-by-age observations are **approximate visual readings of Figure 3** from Mousavi et al., not exact, author-supplied sex-specific numerical values. The youngest and oldest age-bin midpoints (5 and 95 years) are assumptions. As a result, the fitted coefficients and statistics are illustrative and sensitive to digitization error; they should not be treated as clinical estimates.

| Sex | Outcome | Lower-BIC model | Winning BIC | Observation |
| --- | --- | --- | ---: | --- |
| Female | SBP | Quadratic | 3.3583 | Simpler model preferred |
| Male | SBP | Sigmoid | 11.8374 | Nonlinear curve fits better |
| Female | DBP | Quadratic | 0.6981 | Simpler model preferred |
| Male | DBP | Quadratic | 4.1054 | Nearly tied with Gaussian (BIC difference 0.162) |

The quadratic age-squared coefficient represents curvature in mmHg/year², the linear age coefficient uses mmHg/year, and the intercept uses mmHg. For the sigmoid, the baseline and rise are in mmHg, the midpoint is in years, and steepness is in year⁻¹. The Gaussian DBP baseline and height are in mmHg, while its peak age and width are in years. Parameters may be unstable when the data do not clearly exhibit the entire expected curve, and neither family should be extrapolated beyond observed ages without caution.

Model-fit figures are displayed within the [Part A notebook](partA_model_fitting_ALL_IN_ONE.ipynb).

**Part A conclusion:** The handwritten preference for nonlinear models was only partly supported: sigmoid won for male SBP, but BIC favored quadratics for female SBP and both DBP series. These conclusions depend on the approximate figure readings.

## Part B — Training imbalance and bias mitigation

I generated the homework's synthetic bivariate Gaussian SBP/DBP data for four male/female ratios (50/50, 75/25, 90/10, and 97/3), each with 60,000 observations. Each training/test split was stratified (70/30). I trained logistic regression classifiers to predict sex (male = 1; female = 0) and evaluated them on their own test sets and on an independent, balanced 10,000-person test set.

### B(i–iii): Why accuracy can be misleading

| Training male share | Own-test accuracy | Balanced-test accuracy | Balanced-test male recall | Balanced-test female recall | Balanced-test AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50% | 58.17% | 58.52% | 59.42% | 57.62% | 0.6162 |
| 75% | 75.11% | 50.31% | 99.74% | 0.88% | 0.6159 |
| 90% | 90.00% | 50.00% | 100.00% | 0.00% | 0.6161 |
| 97% | 97.00% | 50.00% | 100.00% | 0.00% | 0.6161 |

At 90/10 and 97/3, the default-threshold classifier predicts everyone as male. The high accuracy on each imbalanced test set therefore hides zero female recall. The nearly unchanged ROC AUC (~0.616 on the shared balanced test) indicates that the underlying ranking ability has not improved meaningfully. In a healthcare application, subgroup performance must be examined rather than relying on overall accuracy.

ROC and classification-metric figures are displayed within the [Part B notebook](partB_bias_analysis_ALL_IN_ONE.ipynb).

### B(iv): Penalizing recall disparity

For the original 90/10 training split, I searched over female class weights and classification thresholds. I selected candidates using a separate validation subset and the lecture-inspired objective

$$
C = (1 - \text{balanced accuracy}) + \lambda\lvert R_M - R_F\rvert,
$$

where $R_M$ and $R_F$ are male and female recall. The independent balanced test set was **not** used to select hyperparameters.

At $\lambda=0$, the selected model gave **58.37%** balanced-test accuracy and a **2.98-percentage-point** recall gap. For $\lambda\geq0.10$ in the tested grid, the selected configuration reached **58.52%** accuracy and a **1.00-percentage-point** gap. The finite grid gives an approximate rather than exact optimum. Equal recalls alone do not establish clinically meaningful fairness.

### B(v): Synthetic female augmentation

I held the original 37,800 male training samples fixed and augmented the 4,200 female training samples with 33,600 synthetic female-labeled examples, resulting in balanced training data. The alternatives were (a) sampling from the homework's specified female bivariate Gaussian and (b) uniform sampling across SBP 70–210 mmHg and DBP 35–125 mmHg.

All three methods below were evaluated on the **same untouched independent balanced test set**.

| Training method | Balanced-test accuracy | Male recall | Female recall | Recall gap | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original 90/10 | 50.00% | 100.00% | 0.00% | 100.00% | 0.6161 |
| Gaussian augmentation | **58.46%** | 59.42% | 57.50% | **1.92%** | 0.6160 |
| Uniform augmentation | 42.52% | 54.64% | 30.40% | 24.24% | 0.3938 |

Gaussian augmentation better matches the **assumed simulation distribution**, whereas uniform augmentation assigns a female label to many unrealistic combinations and can distort the classifier. This is not proof that real female blood pressure follows a bivariate Gaussian. In practice, synthetic-data quality and real-population validation are essential; sex is not clinically determined by SBP/DBP alone.

The fairness and augmentation plots are also displayed within the Part B notebook. To add standalone PNGs for the README, rerun the notebook and commit its `figures/` outputs.

## Discussion — Hypothesis revisited

**Partly supported.** The hypothesized loss of female recall under male-heavy training was observed: female recall fell to zero while accuracy on the original test distribution rose. Gaussian-based augmentation substantially reduced the recall gap, in line with the prediction about realistic synthetic data. The model-shape hypothesis was not consistently supported by BIC across sex and outcome. These are controlled simulations and approximate trend fits, not evidence of clinical fairness or deployable sex prediction.

## Reproducibility

Requires Python 3.10+ (the saved Part B run used Python 3.13.5). Download or clone the repository, then install:

```bash
python -m pip install -r requirements.txt
jupyter notebook
```

Run the following notebooks in order, using **Kernel > Restart & Run All** for each:

1. [`partA_model_fitting_ALL_IN_ONE.ipynb`](partA_model_fitting_ALL_IN_ONE.ipynb) — embedded graph-estimated age table, model fits and BIC summaries.
2. [`partB_bias_analysis_ALL_IN_ONE.ipynb`](partB_bias_analysis_ALL_IN_ONE.ipynb) — data generation, classification, fairness sweep and augmentation.

No external CSV is required: Part A embeds its approximate readings, and Part B generates data with fixed random seeds. Part B saves PNG plots under `figures/` and CSV metrics under `outputs/` when run. Part A saves PNGs if its `SAVE_RESULTS` flag is set to `True`; figures are also displayed inline.

## AI use statement

**Statement:** ChatGPT was used to assist with extracting approximate blood-pressure values from the referenced paper's figure, developing and checking Python notebook code, and editing parts of the written explanations.

**Full public AI chat link(s):** **[TODO: add the public share URL for this chat and every other AI chat used for the homework before submission.]** The course requires complete, accessible chat transcripts; check the URL(s) in a signed-out browser before submitting.

## References

- Mousavi et al., *Learning from Two Decades of Blood Pressure Data: Demography-Specific Patterns Across 75 Million Patient Encounters*, [arXiv:2402.01598](https://arxiv.org/pdf/2402.01598).
- Emory University BMI 500, Lecture 7, HW3 assignment sheet (October 2026).

## Submission checklist

- [x] Handwritten hypothesis committed (`hypothesis.pdf`)
- [x] Part A and Part B reproducible notebooks
- [x] Results, discussion and dependencies in repository
- [ ] Commit standalone `.png` figure files for the README (required by the homework)
- [ ] Add **all** required public AI chat link(s) above
- [ ] Create **one-page `slide.pdf` yourself** and commit it (the assignment prohibits AI-generated slides)
- [ ] Check repository readability and run both notebooks on your own laptop
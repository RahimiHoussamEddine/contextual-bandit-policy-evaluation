# Evaluation design and estimator definitions

## Logged decisions and target policy

Each CSV row is one item shown at one of three interface positions. The action is `item_id`, the
outcome is binary `click`, and `propensity_score` is the logging-policy probability of the displayed
item at that position. The target policy is the dataset's independent uniform Random policy,
`pi(a | x, position) = 1/80`; this matches all Random-policy propensities in the downloaded sample.
The behavior policy is BTS, and its logged row propensity is used in every importance ratio.

This setup treats each item-position as a contextual bandit decision. It does not estimate the
joint probability of a complete three-item slate and does not account for interference between
items shown together.

## Chronological split

The split is fixed in `src/bandit_ope/splits.py`:

| Role | UTC dates | Use |
| --- | --- | --- |
| Train | 2019-11-24, 2019-11-26 | Fit reward models on BTS rows |
| Validation | 2019-11-27 | Choose reward model and clipping cap |
| Test | 2019-11-28, 2019-11-29, 2019-11-30 | Final OPE and Random holdout comparison |

The byte sample contains no rows from 2019-11-25. Rows are never randomly shuffled across dates.
The training windows are selected from past records only; item affinity fields are treated as
pre-decision features supplied by the source.

## Reward-model selection

The first baseline is a smoothed empirical click rate by item and position:

`q_hat(a, p) = (clicks(a, p) + 100 * global_CTR) / (rows(a, p) + 100)`.

The contextual candidates are L2 logistic regressions with `C` in `{0.1, 1, 10}`. Features are item,
position, four categorical user-feature columns, and `log1p` of the affinity between the user and
the selected item. Unknown categories at evaluation time are ignored by the encoder. The selected
model is the one with lowest validation log loss; Brier score breaks ties. No test metric is used
for model selection.

For DM, the chosen model predicts the reward for each of the 80 catalog items at a test context and
position. Their arithmetic mean is the model estimate under the uniform target policy.

## Off-policy estimators

For logged row `i`, define `w_i = pi(a_i | x_i, p_i) / mu(a_i | x_i, p_i)`, with observed reward
`r_i`, logged action `a_i`, and selected reward model `q_hat`.

- **IPS:** `mean(w_i * r_i)`.
- **SNIPS:** `sum(w_i * r_i) / sum(w_i)`.
- **DM:** `mean_a~pi(q_hat(x_i, a, p_i))`, then averaged across test contexts.
- **DR:** `mean(q_pi(x_i, p_i) + w_i * (r_i - q_hat(x_i, a_i, p_i)))`.

Validation clipping candidates are no clipping and maximum weights of 5, 10, 20, 50, and 100. The
selected candidate minimizes absolute error to the independent Random validation click rate; exact
ties prefer no clipping. Clipping can reduce variance while adding bias, so the UI also lets readers
inspect a different cap without relabeling it as the selected estimate.

## Benchmark and uncertainty

The sampled Random logs are an independent on-policy reference for the target policy. Reference and
BTS rows are selected from corresponding byte regions and shared calendar dates, but the ranges are
not exactly time-matched. The comparison is therefore a useful offline diagnostic, not ground truth
for the full campaign.

The report includes naive row-IID Wald intervals for IPS, SNIPS, DM, DR, and the Random reference.
They omit within-user/session dependence and uncertainty from deterministic byte-range selection;
they should not be interpreted as campaign-wide confidence intervals. Three per-day estimates are
shown as a complementary stability check.

## Sample integrity and support

The downloader pins a dataset revision, checks ETag and HTTP `Content-Range` for every request,
parses only complete CSV records, retains source row IDs, and hashes the local sample files. It
validates timestamps, arm and position ranges, binary rewards, positive propensities, source-row
ordering, duplicate IDs, minimum row count, and date coverage. In the observed BTS test sample,
225/240 position-item cells are present; sparse empirical coverage is surfaced in the dashboard.

The deterministic sample is not a probability sample. Reported metrics describe only these sampled
rows and do not support an unbiased extrapolation to all logged impressions.

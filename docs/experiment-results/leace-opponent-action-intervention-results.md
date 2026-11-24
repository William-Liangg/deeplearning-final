# Opponent Style Erasure: Results

## Question

After watching an opponent play 20 hands, does the poker model use information about that opponent's style to predict the opponent's next action?

## Where we started

Our first experiment showed that opponent style was easy to read from the trained model's internal state:

- trained model: **96%**;
- simple action counts: **90%**;
- untrained model: **89%**;
- random labels: **25%**;
- chance: **25%**.

This showed that the model contained information about style. It did not show that the model used that information.

A classifier can sometimes read information that has no effect on the model's prediction. To test use, we needed to remove the readable information inside the model and see whether its prediction changed.

## What we tried first

We first removed the three directions used by a style classifier.

This produced a small increase in opponent-action prediction loss: **0.00150**. However, a new classifier could still recognize style with **96.35%** accuracy, compared with **96.19%** before the intervention.

The intervention had not erased style. The small prediction change therefore could not tell us whether style mattered.

## Options we considered

One option was iterative nullspace projection, or INLP. This repeatedly trains classifiers and removes their directions until a new classifier can no longer recover the concept.

We decided against it. Later research showed that repeated projection can make unnecessary and partly random changes to a representation. That would make a prediction change difficult to interpret: it might come from removing style, or simply from damaging the activation.

We instead chose LEACE. LEACE finds one closed-form affine transformation designed to remove linear information while minimizing least-squares distortion. With four style classes, its change has rank at most three.

## Checking that LEACE worked

Before putting LEACE inside the transformer, we tested whether it actually erased held-out style information.

- Before LEACE: **96.19%**
- After LEACE: **25.00%**
- Chance: **25.00%**
- Shuffled-label LEACE: **96.05%**
- Ten random rank-three controls: **96.05%–96.18%**

A new classifier was trained after every transformation. The classifier could not recover style after true-label LEACE, but it could still recover style after the controls.

LEACE changed only a rank-three part of the 256-dimensional state. Even so, the change was not negligible: it removed **8.95%** of centered activation energy. That created an important problem for the causal test. Ordinary random rank-three controls changed much less energy, so they were not enough to distinguish style removal from general disruption.

## What we landed on

We intervened at `blocks.4.hook_resid_post`, at the internal state immediately before the first opponent action after hand 20.

We changed only that one state in each held-out session. The model then continued through its remaining transformer block and output layer normally.

We compared true-label LEACE with:

- untouched execution;
- shuffled-label LEACE;
- ten ordinary random rank-three interventions;
- ten random rank-three interventions scaled to match LEACE's activation change;
- a shuffled-label displacement scaled to match LEACE's activation change.

The scaling used only the 7,000 fitting sessions. The 3,000 test sessions were not used to choose transformations or their sizes.

The magnitude-matched controls were an imperfect but important safeguard. Scaling a projection past its natural size can overshoot, so these controls were not clean erasers. Their purpose was narrower: test whether an equally large rank-three change would produce the same prediction damage.

## Results

| Condition | Style accuracy | Opponent-action loss | Change from untouched |
|---|---:|---:|---:|
| Untouched | 96.19% | 1.31640 | — |
| True-label LEACE | 25.00% | 1.42964 | **+0.11324** |
| Shuffled-label LEACE | 96.05% | 1.31617 | -0.00023 |
| Magnitude-matched shuffled displacement | 96.18% | 1.37577 | +0.05937 |
| Ordinary random controls | 96.05%–96.18% | 1.30564–1.32040 | -0.01076–+0.00400 |
| Magnitude-matched random controls | 96.02%–96.18% | 1.29270–1.33511 | -0.02370–+0.01871 |

### The key causal comparison

The fairest random controls are the magnitude-matched ones because they changed the activation by approximately the same amount as true-label LEACE.

| Condition | Opponent-action loss | Correct-action probability | Four-action accuracy |
|---|---:|---:|---:|
| No ablation | 1.31640 | 33.50% | 57.43% |
| True-label LEACE ablation | **1.42964** | **28.35%** | **54.47%** |
| Magnitude-matched random ablations | 1.29270–1.33511 | 32.62%–34.19% | 57.23%–57.77% |

The roughly three-percentage-point accuracy drop is modest: style is one useful input rather than the model's main prediction signal. The probability and loss measurements show a larger confidence effect. Compared with no ablation, correct-action probability fell by 5.15 percentage points, about 15% relative, while cross-entropy increased by 8.6%.

The likely true-label LEACE loss increase over no ablation is between **0.10205 and 0.12574**. True-label LEACE also had higher loss than every magnitude-matched random ablation. Even its smallest difference from one of those controls was **0.09453**, with a 95% interval of **0.08264 to 0.10632**.

Compared with the magnitude-matched shuffled displacement, true-label LEACE's additional loss was **0.05387**, with a likely range of **0.02879 to 0.07672**.

## What this means

The model did not merely contain readable information about opponent style. Information removed by LEACE contributed to predicting the opponent's next action.

The magnitude-matched shuffled intervention also hurt prediction. This confirms that a large internal change is disruptive by itself. But true-label LEACE caused substantially more harm than that control and every magnitude-matched random control.

Together, the experiments support a causal claim at this specific decision point: the model uses linearly represented opponent-style information when predicting the opponent's next action.

## What this does not mean

This does not prove that every trace of style was removed. A nonlinear classifier might still recover style, and style may remain elsewhere in the network.

It does not show that style lives in three individual neurons. LEACE removed a three-dimensional subspace spread across the activation.

It also does not show that the model uses style to choose an adaptive poker strategy. The model was tested on next-action prediction in sessions generated with a fixed hero bot.

Finally, this result comes from one checkpoint, one layer, one point in each session, and one model seed. Repeating it across independently trained models would establish how reliable the effect is.

## Detailed measurements

The complete measurements are in [`opponent_style_probing/measurements/leace_opponent_action_seed0.json`](../../opponent_style_probing/measurements/leace_opponent_action_seed0.json).

The initial readability experiment is recorded in [`docs/experiment-results/hand20-style-probe-results.md`](hand20-style-probe-results.md). The failed classifier-direction intervention and the LEACE feasibility check are summarized above rather than maintained as separate runnable studies.

The erasure method and causal-control concern are discussed in:

- Belrose et al., [LEACE: Perfect Linear Concept Erasure in Closed Form](https://arxiv.org/abs/2306.03819)
- Dobrzeniecka, Fokkens, and Sommerauer, [Improving Causal Interventions in Amnesic Probing with Mean Projection or LEACE](https://aclanthology.org/2025.findings-acl.674/)

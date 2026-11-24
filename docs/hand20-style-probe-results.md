# Opponent Style Probe: Results

## Question

After watching an opponent play 20 hands, does the poker model contain information about that opponent's style?

## What we did

We tested the model on 10,000 poker sessions.

We used 7,000 sessions to teach a simple classifier how to recognize four opponent styles. We tested it on the other 3,000 sessions.

We compared four sources of information:

- the trained model's internal state;
- the untrained model's internal state;
- simple counts of the opponent's actions;
- randomly mixed-up style labels.

## Results

- Trained model: **96%**
- Simple action counts: **90%**
- Untrained model: **89%**
- Random labels: **25%**
- Chance: **25%**

The trained model scored about 6 percentage points higher than simple action counts. The likely difference is between about 5 and 7 percentage points.

## What this means

After 20 hands, the trained model's internal state contains clear information about the opponent's style.

It contains more useful information than a simple count of folds, checks, calls, and raises.

## What this does not mean

This experiment does not show that the model uses this information when choosing an action.

It also does not prove that the model has a general or human-like idea of opponent style.

Those questions require another experiment.

## Detailed measurements

The complete measurements are in [`opponent_style_probing/measurements/hand20_seed0.json`](../opponent_style_probing/measurements/hand20_seed0.json).

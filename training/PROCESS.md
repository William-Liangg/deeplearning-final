# How the training was carried out

This document walks through the training work step by step, in plain language. For the
exact commands and file layout, see [`README.md`](README.md).

## The goal

We wanted to know whether a model that only learns to **predict poker moves** figures
out, on its own, **what kind of opponent it is playing against**. It is never told the
opponent's style.

To test that, we did three things:

1. Trained the model.
2. Compared it against simple prediction rules.
3. Prepared it for probing, the next stage, which looks inside the model to see whether it
   stores the opponent's style.

---

## Step 1: Creating the data

We simulated poker games with RLCard, a library of card-game environments. The game is
**heads-up limit Texas hold'em**: two players, with fixed bet sizes.

- **A session** is 50 hands between the same two players: a fixed, middle-of-the-road
  "hero" bot and one opponent bot.
- **Each opponent has one of four styles:**
  - **tight:** plays few hands and mostly bets with strong cards;
  - **loose:** plays many hands;
  - **bluffer:** often bets with weak cards;
  - **random:** picks a random legal move.
- **No two opponents play exactly alike.** Each opponent's settings were varied slightly.
- **Each session is written as a sequence of symbols (tokens):** cards, moves and markers,
  68 kinds in total. One hand looks like this:

  ```
  [HAND] O:call H:raise O:call [FLOP] 8c 8s Td H:check O:check [TURN] 6s H:check O:check [RIVER] 7c H:raise O:call [SHOW] 2s 3c O:lost
  ```

  `H:` is the hero and `O:` is the opponent.

We generated **100,000 sessions**: 90,000 to train on and 10,000 held back for testing.
The random seeds are fixed, so anyone can regenerate exactly the same data. Every session
fits in the model's 1,024-token window (the longest is 953 tokens), so nothing is cut off.

Each opponent's style was saved separately. **The model never sees it.** It exists only to
check the results and for probing.

## Step 2: Setting up the model

- **A small transformer**, the same kind of model as GPT but much smaller: 6 layers and
  about **5 million parameters**.
- **Built with TransformerLens 3.9,** a library that makes it easy to look inside a model.
  That matters for probing. Version 4.0 removed the model type we needed, so the version
  is pinned.
- **Started from random weights,** with no pretraining. Anything it learns about opponents
  therefore comes from our poker data alone.
- **Trained on Brown's Oscar computing cluster,** in its own Python environment.

## Step 3: What the model was trained to do

The model reads a session one token at a time and, at each point, predicts what comes
next.

- **It is graded only on predicting moves** (fold, check, call or raise), by either player.
- **It is not graded on predicting cards,** which are random, **or results** (won or
  lost), which are outcomes rather than decisions. It still sees all of these as input.
- **Every training batch is made of whole sessions.** Two opponents are never mixed in one
  sequence, because that would create a fake history.

## Step 4: Checks before training

Before the full training, we confirmed four things:

1. **The model can learn:** it memorizes a tiny sample of sessions as well as is
   mathematically possible.
2. **Saving works:** a saved model reloads and gives exactly the same answers.
3. **We can see inside it:** the internal activity of every layer is accessible, which
   probing needs.
4. **We can change what's inside:** editing an internal value changes the model's later
   predictions, which patching needs.

All four passed.

## Step 5: Training

- **Settings:**
  - a fixed learning rate of 0.0003;
  - batches of 32 sessions, processed 8 at a time;
  - 3 full passes over the training data (8,439 update steps);
  - a safeguard against overly large updates (gradient clipping).
- **Three separate runs.** We trained the model three times from different random
  starting points (seeds 0, 1 and 2) to make sure the results aren't luck.
  - Seed 0 ran first.
  - Once it finished successfully, seeds 1 and 2 started automatically and ran side by
    side.
  - Each run took 33–52 minutes on one GPU.
- **Snapshots during training.** Each run saved the model:
  - at the start, untrained;
  - at 0.1%, 0.3%, 1%, 3%, 10%, 30%, 60% and 100% of training. They are spaced closely at
    the start because learning is fastest early on, and this lets probing look at *when*
    the model starts recognizing opponents;
  - in full at the end of each pass, so an interrupted run could resume.
- **Learning finished early.** Most of it happened in the first pass, roughly the first
  10% of steps. The extra passes changed little.

## Step 6: Comparing against simple rules

A score means little on its own. So we compared the model against simple ways of
predicting the opponent's next move:

- **Always guess the most common move.**
- **Count this opponent's moves so far in the session,** and guess the one he makes most.
- **The same counting, but only among legal moves.** For example, you can't check when
  facing a bet. This is the strongest and fairest simple rule, because the model can also
  see what's legal.
- **The same trained model, shown only the current hand,** as if its memory were wiped
  before each hand. If the full model does better, it must be using earlier hands.
- **The untrained model,** as a sanity check.

## Step 7: Measuring the ceiling

Opponents decide partly based on their hidden cards, so some moves can never be
predicted. To find the best score that is realistically possible, we trained an **oracle
model**:

- It is a second copy, trained the same way, but **told the opponent's style** at the
  start of every session.
- It was trained for one pass, since the main runs stopped improving after the first.

We also tried a counting rule that is told the style. The style labels were used only for
these comparisons, never by the main model.

## Step 8: How we scored

The main score is **how often the opponent's next move is predicted correctly, choosing
among the 4 opponent moves**, measured on the 10,000 held-back sessions.

We use this rather than the model's raw score for a reason: at the start of each hand, who
acts first is random. The raw score partly measures guessing that coin flip correctly,
which says nothing about understanding the opponent.

---

## Results

| Predictor | Correct |
|---|---|
| Always the most common move | 31.8% |
| Count this opponent's moves | 41.6% |
| Model shown only the current hand | 54.4% |
| Count this opponent's moves, legal moves only | 60.3% |
| **Trained model (average of 3 runs)** | **61.3%** |
| Oracle model (told the style) | 61.8% |

The untrained model scores 28.3%.

How the score changes as the session goes on:

| Hand in the session | 1 | 2–5 | 6–10 | 11–20 | 21–30 | 31–40 | 41–50 |
|---|---|---|---|---|---|---|---|
| Trained model | 53.7% | 58.8% | 60.6% | 61.7% | 61.8% | 61.8% | 62.0% |
| Oracle model | 61.6% | 61.6% | 61.6% | 62.0% | 61.8% | 61.8% | 61.9% |
| Counting, legal moves only | 51.6% | 57.3% | 59.4% | 60.6% | 60.9% | 61.1% | 61.3% |
| Model shown only the current hand (seed 0 only) | 53.6% | 54.4% | 54.4% | 54.4% | 54.2% | 54.3% | 54.3% |

**What this shows**

- **The three runs agree closely** (61.4%, 61.3% and 61.2%), so the result is reliable.
- **The model learns about the opponent as it watches.** It starts at 53.7% on the first
  hand and climbs to 62%. The same model limited to the current hand stays flat at about
  54%.
- **Within 10 to 20 hands, it predicts as well as the oracle model,** which was told the
  style from the start. The only gap is in the first few hands, before it has seen the
  opponent play.

**What it doesn't show yet**

- **The ceiling is low** (about 62%), because of the hidden cards. Even the best simple
  rule is only about 1.5 points below the oracle.
- **Counting gets close too** (61.3% by the end). From its predictions alone, we can't tell
  whether the model has learned *what kind of player* the opponent is, or is just keeping
  something like a running tally of his moves.

Answering that is the job of **probing**: looking inside the model to see whether the
opponent's style is stored in its internal activity.

---

## Changes from the original plan

- **Trained our own small model** instead of fine-tuning an existing poker language model
  (TexasHoldEm-Llama-3.2-1B), so that no outside training could explain what the model
  learns.
- **Four opponent styles** (tight, loose, bluffer, random) instead of the proposal's
  game-theory-optimal, tight and aggressive players.
- **Ran on the Oscar cluster** instead of Colab.
- **Seeds 1 and 2 started automatically** once seed 0 succeeded, instead of after a manual
  review, to save time.
- **Added comparisons beyond the spec:** the legal-moves counting rule, the oracle model
  and the style-told counting rule.
- **Main score is the 4-move accuracy** described in Step 8, with the raw score still
  recorded.
- **The "current hand only" comparison** restarts each hand at the beginning of the
  model's window. A second version kept each hand at its original position, but that
  shows the model a situation it never saw in training, so it was used only as a
  diagnostic. This choice is pending team confirmation.
- **A known quirk:** in RLCard, the order of play after the flop differs slightly from
  standard poker. It is the same for every opponent style, so it shouldn't affect the
  comparison, but it is worth mentioning in the report.

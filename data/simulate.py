"""Generate poker sessions between a fixed hero bot and a styled opponent bot.

Each session = many hands against the SAME opponent, saved as one token sequence.
The opponent's style is saved as a label but never appears in the tokens.

Usage:
    python data/simulate.py --sessions 1000 --hands 50 --out data/sessions_small.jsonl
"""

import argparse
import json
import random

import rlcard

RANKS = "23456789TJQKA"

# Each style is three knobs:
#   play_rate:  fraction of hands it's willing to continue with (higher = looser)
#   aggression: when continuing, how often it raises instead of calling
#   bluff_rate: with a weak hand, how often it raises anyway
STYLES = {
    "tight":   {"play_rate": 0.25, "aggression": 0.5, "bluff_rate": 0.02},
    "loose":   {"play_rate": 0.75, "aggression": 0.2, "bluff_rate": 0.05},
    "bluffer": {"play_rate": 0.45, "aggression": 0.6, "bluff_rate": 0.40},
    "random":  None,  # picks a legal action uniformly
}

# Std of the noise added to each knob per session. 0 = every bluffer is identical.
JITTER = 0.08

HERO = {"play_rate": 0.5, "aggression": 0.4, "bluff_rate": 0.1}


def card_token(card):
    """RLCard 'CT' (suit, rank) -> 'Tc' (rank, suit)."""
    return card[1] + card[0].lower()


def hand_strength(hand, board):
    """Rough strength in [0, 1]. Good enough to make styles behave differently."""
    ranks = [RANKS.index(c[1]) for c in hand]
    if not board:
        score = sum(ranks) / 24
        if ranks[0] == ranks[1]:
            score += 0.4
        return min(score, 1.0)

    board_ranks = [RANKS.index(c[1]) for c in board]
    matches = sum(board_ranks.count(r) for r in ranks)
    if ranks[0] == ranks[1]:
        matches += 1
    if matches >= 2:
        return 0.95
    if matches == 1:
        return 0.65 + 0.02 * max(ranks)
    return max(ranks) / 24


def bot_action(style, state):
    legal = state["raw_legal_actions"]
    if style is None:
        return random.choice(legal)

    obs = state["raw_obs"]
    strength = hand_strength(obs["hand"], obs["public_cards"])
    passive = "check" if "check" in legal else "fold"

    if strength > 1 - style["play_rate"]:
        if "raise" in legal and random.random() < style["aggression"]:
            return "raise"
        return "check" if "check" in legal else "call"

    if "raise" in legal and random.random() < style["bluff_rate"]:
        return "raise"
    return passive


def play_hand(env, opp_style):
    """Play one hand. Player 0 = hero, player 1 = opponent. Returns tokens."""
    tokens = ["[HAND]"]
    env.reset()
    shown_board = 0
    streets = {3: "[FLOP]", 4: "[TURN]", 5: "[RIVER]"}

    while not env.is_over():
        pid = env.get_player_id()
        state = env.get_state(pid)

        board = state["raw_obs"]["public_cards"]
        if len(board) > shown_board:
            tokens.append(streets[len(board)])
            tokens += [card_token(c) for c in board[shown_board:]]
            shown_board = len(board)

        if pid == 0:
            action = bot_action(HERO, state)
            tokens.append(f"H:{action}")
        else:
            action = bot_action(opp_style, state)
            tokens.append(f"O:{action}")
        env.step(action, raw_action=True)

    # Showdown: opponent's cards are revealed only if nobody folded
    if tokens[-1] not in ("H:fold", "O:fold"):
        opp_hand = [c.get_index() for c in env.game.players[1].hand]
        tokens.append("[SHOW]")
        tokens += [card_token(c) for c in opp_hand]

    tokens.append("O:won" if env.get_payoffs()[1] > 0 else "O:lost")
    return tokens


def sample_knobs(style_name):
    """Jitter the archetype's knobs so no two opponents play exactly alike."""
    base = STYLES[style_name]
    if base is None:
        return None
    return {k: round(min(max(v + random.gauss(0, JITTER), 0.0), 1.0), 3) for k, v in base.items()}


def make_session(env, n_hands):
    style_name = random.choice(list(STYLES))
    knobs = sample_knobs(style_name)
    tokens = []
    for _ in range(n_hands):
        tokens += play_hand(env, knobs)
    return {"style": style_name, "knobs": knobs, "tokens": tokens}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sessions", type=int, default=1000)
    parser.add_argument("--hands", type=int, default=50)
    parser.add_argument("--out", default="data/sessions_small.jsonl")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    env = rlcard.make("limit-holdem", config={"seed": args.seed})

    with open(args.out, "w") as f:
        for _ in range(args.sessions):
            f.write(json.dumps(make_session(env, args.hands)) + "\n")
    print(f"Wrote {args.sessions} sessions to {args.out}")


if __name__ == "__main__":
    main()
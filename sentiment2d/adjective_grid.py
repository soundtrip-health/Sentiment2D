"""Build a grid of representative adjectives over the Sentiment2D (valence, arousal) space.

Each candidate adjective is rated once with Sentiment2D (inside a carrier sentence such as
"I feel {}.") and the ratings are cached to CSV. The cached ratings are then sorted into a grid
over [-1, 1] x [-1, 1] and up to `k` adjectives are chosen for each grid cell.

Two grid layouts are supported:
  * "bins": round(2 / spacing) bins per axis, labelled by their centres
    (spacing 0.2 -> 10 x 10 = 100 cells with centres at -0.9, -0.7, ..., 0.9).
  * "points" (default): grid points from -1 to 1 inclusive (spacing 0.2 -> 11 x 11 = 121 points); each
    word goes to its nearest grid point, so cells on the edges are half-width.

Candidate words come from a hand-curated list of affect adjectives (data/affect_adjectives.txt)
and, optionally, from all WordNet adjectives above a word-frequency floor (--wordnet). Every
rating row records its source, so the grid can be built from either pool (--sources).

Example:
    python -m sentiment2d.adjective_grid --out adjective_grid.json
    python -m sentiment2d.adjective_grid --wordnet 3.0 --rerate --sources curated wordnet
    python -m sentiment2d.adjective_grid --layout bins --rank frequency --fill -k 5

Requires `wordfreq` (pip install wordfreq), used to rank words by familiarity, and `nltk`
(with the wordnet corpus) if --wordnet is used.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_WORDLIST = Path(__file__).parent / "data" / "affect_adjectives.txt"
DEFAULT_TEMPLATE = "I feel {}."


def load_wordlist(paths):
    """Read word files (whitespace-separated entries, '#' comments) into a de-duplicated list.
    Use hyphens for multi-word entries (e.g. laid-back)."""
    words = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if not line.strip().startswith("#"):
                words.extend(w.lower() for w in line.split())
    return list(dict.fromkeys(words))


def wordnet_adjectives(min_zipf=3.0):
    """Return all single-word, lower-case WordNet adjectives with zipf frequency >= min_zipf."""
    import nltk
    from nltk.corpus import wordnet as wn
    from wordfreq import zipf_frequency

    nltk.download("wordnet", quiet=True)
    lemmas = {lem.name() for pos in ("a", "s") for ss in wn.all_synsets(pos) for lem in ss.lemmas()}
    words = [w for w in lemmas if w.isalpha() and w.islower() and zipf_frequency(w, "en") >= min_zipf]
    return sorted(words)


def rate_words(words, template=DEFAULT_TEMPLATE, batch_size=64, model=None):
    """Return a DataFrame of word, valence, arousal, zipf (word frequency), template for each word.

    Valence and arousal are computed exactly as in Sentiment2D, but with the texts passed to the
    zero-shot pipeline in GPU batches.
    """
    from wordfreq import zipf_frequency

    if model is None:
        from sentiment2d import Sentiment2D

        model = Sentiment2D()
    happy, angry, gloomy, calm = model.classes
    texts = [template.format(w) for w in words]
    valence, arousal = [], []
    for out in model.classifier(texts, model.classes, batch_size=batch_size):
        s = dict(zip(out["labels"], out["scores"]))
        valence.append(s[happy] + s[calm] - s[angry] - s[gloomy])
        arousal.append(s[happy] + s[angry] - s[calm] - s[gloomy])
    return pd.DataFrame(
        {
            "word": words,
            "valence": valence,
            "arousal": arousal,
            "zipf": [zipf_frequency(w, "en") for w in words],
            "template": template,
        }
    )


def grid_axis(spacing, layout="points"):
    """Return the grid-cell coordinates along one axis."""
    n = int(round(2 / spacing))
    if not np.isclose(n * spacing, 2):
        raise ValueError(f"spacing {spacing} does not evenly divide the range [-1, 1]")
    if layout == "bins":
        return -1 + spacing * (np.arange(n) + 0.5)
    if layout == "points":
        return -1 + spacing * np.arange(n + 1)
    raise ValueError(f"unknown layout {layout!r}")


def cell_index(x, spacing, layout="points"):
    """Map scores in [-1, 1] to integer cell indices along one axis."""
    n = int(round(2 / spacing))
    x = np.asarray(x, dtype=float)
    if layout == "bins":
        return np.clip(np.floor((x + 1) / spacing), 0, n - 1).astype(int)
    return np.clip(np.round((x + 1) / spacing), 0, n).astype(int)


def build_grid(
    ratings, spacing=0.2, k=10, layout="points", rank="distance", min_zipf=0.0, fill=False, sources=None
):
    """Sort rated words into grid cells and pick up to k words per cell.

    :param ratings: DataFrame with word, valence, arousal, zipf columns (see rate_words).
    :param spacing: grid spacing along each axis.
    :param k: maximum number of words to return per cell.
    :param layout: "bins" (cells labelled by centre) or "points" (nearest grid point, inclusive).
    :param rank: "frequency" puts the most common words first; "distance" puts the words
                 closest to the cell coordinate first.
    :param min_zipf: drop words whose zipf frequency is below this value.
    :param fill: if True, cells with fewer than k words get a separate "fill" list holding the
                 words from outside the cell that are nearest to the cell coordinate.
    :param sources: if given, only use ratings whose "source" column is in this list.
    :return: list of dicts, one per cell, with valence, arousal, words, n_in_cell (and fill).
             The "words" list only ever holds words rated inside the cell.
    """
    keep = ratings.zipf >= min_zipf
    if sources is not None:
        keep &= ratings.source.isin(sources)
    df = ratings[keep].copy()
    axis = grid_axis(spacing, layout)
    df["vi"] = cell_index(df.valence, spacing, layout)
    df["ai"] = cell_index(df.arousal, spacing, layout)
    df["dist"] = np.hypot(df.valence - axis[df.vi], df.arousal - axis[df.ai])
    sort_cols, ascending = (["zipf", "dist"], [False, True]) if rank == "frequency" else (["dist"], [True])
    groups = {key: g.sort_values(sort_cols, ascending=ascending) for key, g in df.groupby(["vi", "ai"])}

    cells = []
    for ai in reversed(range(len(axis))):  # high arousal first, like reading a plot top-down
        for vi in range(len(axis)):
            g = groups.get((vi, ai))
            cells.append(
                {
                    "valence": round(float(axis[vi]), 6),
                    "arousal": round(float(axis[ai]), 6),
                    "words": [] if g is None else g.word.head(k).tolist(),
                    "n_in_cell": 0 if g is None else len(g),
                }
            )
            if fill and cells[-1]["n_in_cell"] < k:
                outside = df[(df.vi != vi) | (df.ai != ai)]
                d = np.hypot(outside.valence - axis[vi], outside.arousal - axis[ai])
                n_fill = k - cells[-1]["n_in_cell"]
                cells[-1]["fill"] = outside.word.iloc[np.argsort(d.to_numpy(), kind="stable")[:n_fill]].tolist()
    return cells


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--words", nargs="+", default=[str(DEFAULT_WORDLIST)], help="curated word list file(s)")
    p.add_argument("--wordnet", type=float, metavar="MIN_ZIPF", help="also rate WordNet adjectives this frequent")
    p.add_argument("--sources", nargs="+", default=["curated"], help="rating sources to build the grid from")
    p.add_argument("--ratings", default="adjective_ratings.csv", help="ratings cache (CSV); reused if present")
    p.add_argument("--rerate", action="store_true", help="ignore an existing ratings cache")
    p.add_argument("--template", default=DEFAULT_TEMPLATE, help="carrier sentence; {} is replaced by the word")
    p.add_argument("--spacing", type=float, default=0.2)
    p.add_argument("-k", type=int, default=10, help="words per grid cell")
    p.add_argument("--layout", choices=["points", "bins"], default="points")
    p.add_argument("--rank", choices=["distance", "frequency"], default="distance")
    p.add_argument("--min-zipf", type=float, default=0.0, help="minimum word frequency (zipf scale)")
    p.add_argument("--fill", action="store_true", help="top up short cells with the nearest outside words")
    p.add_argument("--out", default="adjective_grid.json")
    args = p.parse_args(argv)

    ratings_path = Path(args.ratings)
    if ratings_path.exists() and not args.rerate:
        ratings = pd.read_csv(ratings_path, keep_default_na=False)
        print(f"Loaded {len(ratings)} ratings from {ratings_path} (use --rerate to re-rate)")
        cached = set(ratings.template)
        if cached != {args.template}:
            raise SystemExit(f"Cached ratings used template(s) {cached}, not {args.template!r}; use --rerate")
    else:
        curated = load_wordlist(args.words)
        parts = [rate_words(curated, template=args.template).assign(source="curated")]
        if args.wordnet is not None:
            extra = [w for w in wordnet_adjectives(args.wordnet) if w not in set(curated)]
            parts.append(rate_words(extra, template=args.template).assign(source="wordnet"))
        ratings = pd.concat(parts, ignore_index=True)
        print(f"Rated {len(ratings)} words with template {args.template!r}")
        ratings.to_csv(ratings_path, index=False, float_format="%.4f")
        print(f"Saved ratings to {ratings_path}")

    cells = build_grid(
        ratings, args.spacing, args.k, args.layout, args.rank, args.min_zipf, args.fill, args.sources
    )
    meta = {
        k: getattr(args, k) for k in ("spacing", "k", "layout", "rank", "min_zipf", "fill", "sources", "template")
    }
    Path(args.out).write_text(json.dumps({**meta, "cells": cells}, indent=1))

    short = [c for c in cells if c["n_in_cell"] < args.k]
    print(f"Wrote {len(cells)} cells to {args.out}; {len(short)} have fewer than {args.k} words")
    for c in short:
        print(f"  ({c['valence']:+.1f}, {c['arousal']:+.1f}): {c['n_in_cell']} words")


if __name__ == "__main__":
    main()

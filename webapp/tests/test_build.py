"""The one-command launcher skips steps whose outputs are current and redoes stale ones."""
from __future__ import annotations

import json

import pandas as pd

from recsys.build import data_fingerprint, step_embeddings
from recsys.synth.generate import write


def _world_dir(small_world, tmp_path):
    visible, truth, meta = small_world
    write(tmp_path, visible, truth, meta)
    return tmp_path


def test_embeddings_built_once_then_skipped(small_world, tmp_path):
    d = _world_dir(small_world, tmp_path)
    step_embeddings(d, ["tfidf"])
    first = json.loads((d / "embeddings" / "tfidf.json").read_text())
    step_embeddings(d, ["tfidf"])
    assert json.loads((d / "embeddings" / "tfidf.json").read_text()) == first     # not rebuilt
    assert first["texts_example"][0].endswith(".")                                   # dish text recorded


def test_changed_dishes_make_embeddings_and_fingerprint_stale(small_world, tmp_path):
    d = _world_dir(small_world, tmp_path)
    step_embeddings(d, ["tfidf"])
    before = data_fingerprint(d)
    items = pd.read_parquet(d / "items.parquet")
    items.loc[0, "name"] = "A brand new dish"
    items.to_parquet(d / "items.parquet", index=False)
    assert data_fingerprint(d)["items"] != before["items"]
    step_embeddings(d, ["tfidf"])
    after = data_fingerprint(d)
    assert after["spaces"]["tfidf"] != before["spaces"]["tfidf"]


def test_fingerprint_is_stable(small_world, tmp_path):
    d = _world_dir(small_world, tmp_path)
    assert data_fingerprint(d) == data_fingerprint(d)

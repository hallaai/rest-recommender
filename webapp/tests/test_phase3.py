"""Phase-3 retrievers and ranking: content embeddings, two-tower cold-start, features, ranker."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="module")
def ds(small_world, tmp_path_factory):
    from recsys.data import Dataset
    from recsys.synth.generate import write
    from recsys.embed import build
    d = tmp_path_factory.mktemp("p3")
    visible, truth, meta = small_world
    write(d, visible, truth, meta)
    build("tfidf", d)                 # cheap embeddings for the content model
    return Dataset(d), d


def test_content_retriever_uses_embeddings(ds):
    from recsys.data import VAL_END_DAY, TEST_END_DAY
    from recsys.eval.base import FitContext
    from recsys.eval.content import Content
    data, d = ds
    rec = Content("tfidf")
    rec.fit(FitContext(data, VAL_END_DAY))
    reqs = data.requests(VAL_END_DAY, TEST_END_DAY, sample=100, seed=0)
    had_signal = False
    for r in reqs:
        cand = data.available_mains(r.lat, r.lon, r.hour)
        if len(cand) < 5:
            continue
        s = rec.recommend(r, cand)
        if data.history(r.user, r.day).n:
            assert np.any(s != 0)              # a user with history gets a taste vector
            had_signal = True
        else:
            assert np.all(s == 0)              # no history -> no content signal
    assert had_signal


def test_feature_builder_shape_and_names(ds):
    from recsys.data import VAL_END_DAY, TEST_END_DAY
    from recsys.eval.features import FEATURE_NAMES, FeatureBuilder
    data, d = ds
    fb = FeatureBuilder(data)
    fb.set_popularity(VAL_END_DAY)
    r = data.requests(VAL_END_DAY, TEST_END_DAY, sample=1, seed=0)[0]
    cand = data.available_mains(r.lat, r.lon, r.hour)
    X = fb.row(r, cand, {"buy_again": np.ones(len(cand))})
    assert X.shape == (len(cand), len(FEATURE_NAMES))
    assert np.isfinite(X).all()
    # distance feature is non-negative; is_reorder is 0/1
    dist = X[:, FEATURE_NAMES.index("distance_km")]
    assert (dist >= 0).all()
    assert set(np.unique(X[:, FEATURE_NAMES.index("is_reorder")])) <= {0.0, 1.0}


def test_two_tower_gives_cold_users_signal(ds):
    from recsys.data import VAL_END_DAY, TEST_END_DAY
    from recsys.eval.base import FitContext
    from recsys.eval.two_tower import TwoTower
    data, d = ds
    rec = TwoTower(epochs=4, batch=1024, text_backend="tfidf")
    rec.fit(FitContext(data, VAL_END_DAY))
    # a cold user (no history) still gets a non-degenerate vector from profile features
    r = data.requests(VAL_END_DAY, TEST_END_DAY, sample=1, seed=1)[0]
    v_known = rec.user_vector(r, known=True)
    v_cold = rec.user_vector(r, known=False)
    assert np.isfinite(v_known).all() and np.isfinite(v_cold).all()
    assert np.linalg.norm(v_cold) > 0.5            # normalised, non-zero
    assert not np.allclose(v_known, v_cold)        # the id contributes something for a known user


def test_ranker_trains_and_scores(ds):
    from recsys.data import TRAIN_END_DAY, VAL_END_DAY
    from recsys.eval.base import FitContext
    from recsys.eval.baselines import BuyAgain, Popularity
    from recsys.eval.features import FeatureBuilder
    from recsys.eval.ranker import LGBMRanker, build_training_data
    data, d = ds
    retr = {"buy_again": BuyAgain(), "popularity": Popularity()}
    for rec in retr.values():
        rec.fit(FitContext(data, TRAIN_END_DAY))
    fb = FeatureBuilder(data)
    fb.set_popularity(TRAIN_END_DAY)
    reqs = data.requests(TRAIN_END_DAY, VAL_END_DAY, sample=300, seed=0)
    cands = [data.available_mains(r.lat, r.lon, r.hour) for r in reqs]
    keep = [i for i, c in enumerate(cands) if len(c) >= 5 and reqs[i].target in set(c.tolist())]
    reqs, cands = [reqs[i] for i in keep], [cands[i] for i in keep]
    train = build_training_data(data, retr, fb, reqs, cands)
    rk = LGBMRanker(loss="listwise", n_estimators=40)
    rk.fit(train)
    s = rk.score(train.X[:50])
    assert len(s) == 50 and np.isfinite(s).all()
    assert rk.feature_importance and sum(rk.feature_importance.values()) > 0

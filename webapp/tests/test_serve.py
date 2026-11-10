"""The live recommendation service: availability, business rules, cold-start, reasons."""
from __future__ import annotations

import numpy as np
import pytest

from recsys.serve import MAX_PER_RESTAURANT, TOP_K, RecommendService


@pytest.fixture(scope="module")
def service(small_world, tmp_path_factory):
    from recsys.data import Dataset
    from recsys.synth.generate import write
    d = tmp_path_factory.mktemp("serve")
    visible, truth, meta = small_world
    write(d, visible, truth, meta)
    svc = RecommendService(Dataset(d))
    svc.fit()
    return svc


def _active_user(svc):
    orders = np.bincount(svc.data.o_user, minlength=svc.data.n_users)
    return int(orders.argmax())


def test_fits_all_baselines(service):
    assert set(service.algorithms) == set(service.fit_seconds)
    assert len(service.algorithms) == 7


def test_recommendations_respect_availability_and_rules(service):
    u = _active_user(service)
    row = service.data.users_df.iloc[u]
    req = service.make_request(u, float(row["home_lat"]), float(row["home_lon"]), 21, 4, False, False)
    avail = set(service.data.available_mains(req.lat, req.lon, req.hour).tolist())
    for algo in service.algorithms:
        out = service.recommend(algo, req)
        ids = [it["item_id"] for it in out["items"]]
        assert len(ids) <= TOP_K
        assert all(i in avail for i in ids)                       # only available dishes
        assert all(service.data.item_cat[i] == 0 for i in ids)    # only mains
        rests = [it["restaurant_id"] for it in out["items"]]
        for r in set(rests):
            assert rests.count(r) <= MAX_PER_RESTAURANT           # variety rule


def test_buy_again_only_returns_reorders(service):
    u = _active_user(service)
    row = service.data.users_df.iloc[u]
    req = service.make_request(u, float(row["home_lat"]), float(row["home_lon"]), 20, 4, False, False)
    out = service.recommend("buy_again", req)
    assert out["items"] and all(it["is_reorder"] for it in out["items"])
    assert all("ordered" in it["reason"] for it in out["items"])


def test_cold_start_user_has_no_signal_for_personalised(service):
    # an unknown user id (map point): personalised models must report no signal
    req = service.make_request(-1, service.data.rest_lat[0], service.data.rest_lon[0], 14, 2, False, False)
    for algo in ("buy_again", "item_item", "ease", "als"):
        out = service.recommend(algo, req)
        assert out["no_signal"] and not out["items"], algo
    # popularity still works
    assert service.recommend("geo_time_pop", req)["items"]


def test_latency_is_reported(service):
    u = _active_user(service)
    row = service.data.users_df.iloc[u]
    req = service.make_request(u, float(row["home_lat"]), float(row["home_lon"]), 13, 1, False, False)
    out = service.recommend("popularity", req)
    assert out["latency_ms"] >= 0 and out["n_candidates"] > 0

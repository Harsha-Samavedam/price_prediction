import numpy as np
from dataclasses import asdict

from claim_reaction.news_graph import build_news_graph


class FakeEmbedder:
    def encode(self, texts, role):
        vocab = {"law": [1.0, 0.0], "sports": [0.0, 1.0], "policy": [0.8, 0.2]}
        return [vocab.get(text, [1.0, 0.0]) for text in texts]


def base_records():
    return [
        {"url": "u1", "title": "law", "provider_seen_at": "2026-01-01T23:59:00-05:00", "market_slug": "m1"},
        {"url": "u2", "title": "policy", "provider_seen_at": "2026-01-02T02:00:00Z", "market_slug": "m2"},
        {"url": "u3", "title": "sports", "provider_seen_at": "2026-01-02T12:00:00Z", "market_slug": "m1"},
    ]


def test_global_url_dedup_and_retrieval_associations_are_preserved():
    rows = base_records() + [{"url": "u1", "title": "duplicate", "provider_seen_at": "2026-01-02T01:00:00Z", "market_slug": "m2"}]
    graph = build_news_graph(rows, [{"market_slug": "m1", "question": "law"},
                                    {"market_slug": "m2", "question": "policy"}], FakeEmbedder(), .8, 1)
    all_members = [member for claim in graph.claims for member in claim.members]
    assert {member.url for member in all_members} == {"u1", "u2", "u3"}
    u1 = next(member for member in all_members if member.url == "u1")
    assert u1.market_ids == ("m2",)
    assert u1.title == "duplicate"
    assert u1.provider_seen_at == "2026-01-02T01:00:00Z"
    assert len(dict(graph.article_provenance)["u1"]) == 2


def test_clusters_are_utc_day_local_and_similarity_edges_cover_every_market():
    graph = build_news_graph(base_records(), [{"market_id": "m1", "question": "law"},
                                               {"market_id": "m2", "question": "policy"},
                                               {"market_id": "m3", "question": "sports"}], FakeEmbedder(), .7, 1)
    # 23:59 at -05:00 is next UTC day, same as u2; u3 is also same day.
    assert len(graph.claims) == 3  # one time-stamped online claim snapshot per unique article
    assert len(graph.market_edges) == len(graph.claims) * 3
    assert any(candidate.market_id == "m1" and candidate.retrieved_member
               for candidate in graph.link_candidates)


def test_reproducible_components_and_candidates_without_outcome_fields():
    rows = base_records()
    markets = [{"market_id": f"m{i}", "question": q, "outcome": i, "yes_price": i / 10}
               for i, q in enumerate(("law", "policy", "sports"))]
    first = build_news_graph(rows, markets, FakeEmbedder(), .75, 2)
    second = build_news_graph(rows, markets, FakeEmbedder(), .75, 2)
    assert first == second
    assert all(edge.market_id in {"m0", "m1", "m2"} for edge in first.market_edges)
    assert all(candidate.cosine_similarity >= -1 for candidate in first.link_candidates)


def test_later_headline_and_duplicate_associations_do_not_change_earlier_event():
    records = [
        {"url": "first", "title": "law", "provider_seen_at": "2026-01-01T10:00:00Z", "market_slug": "m1"},
        {"url": "first", "title": "future duplicate title", "provider_seen_at": "2026-01-01T12:00:00Z", "market_slug": "m2"},
        {"url": "later", "title": "policy", "provider_seen_at": "2026-01-01T13:00:00Z", "market_slug": "m2"},
    ]
    markets = [{"market_id": "m1", "question": "law"}, {"market_id": "m2", "question": "policy"}]
    graph = build_news_graph(records, markets, FakeEmbedder(), .7, 1)
    first = next(claim for claim in graph.claims if claim.event_time == "2026-01-01T10:00:00Z")
    first_candidates = [c for c in graph.link_candidates if c.claim_id == first.claim_id]
    assert [m.url for m in first.members] == ["first"]
    assert first.members[0].market_ids == ("m1",)
    assert first.is_first_report
    assert {candidate.market_id for candidate in first_candidates if candidate.retrieved_member} == {"m1"}
    # Full duplicate provenance is retained separately from the as-of event snapshot.
    provenance = dict(graph.article_provenance)["first"]
    assert len(provenance) == 2


def test_cluster_snapshots_identify_one_first_report_without_backdating():
    rows = [
        {"url": "b", "title": "policy", "provider_seen_at": "2026-01-01T10:00:00Z"},
        {"url": "a", "title": "law", "provider_seen_at": "2026-01-01T10:00:00Z"},
        {"url": "c", "title": "policy", "provider_seen_at": "2026-01-01T11:00:00Z"},
    ]
    graph = build_news_graph(rows, [{"market_id": "m", "question": "law"}], FakeEmbedder(), .7, 1)
    claims = sorted(graph.claims, key=lambda claim: (claim.event_time, claim.claim_id))
    same_time = [claim for claim in claims if claim.event_time == "2026-01-01T10:00:00Z"]
    assert len(same_time) == 2
    assert sum(claim.is_first_report for claim in same_time) == 1
    assert same_time[0].cluster_id == same_time[1].cluster_id
    later = next(claim for claim in claims if claim.event_time == "2026-01-01T11:00:00Z")
    assert later.cluster_id == same_time[0].cluster_id
    assert not later.is_first_report
    assert later.earliest_seen_at == later.event_time
    assert len(later.members) == 3


def test_each_snapshot_records_the_article_that_triggered_it():
    rows = [
        {"url": "first", "title": "law", "provider_seen_at": "2026-01-01T10:00:00Z"},
        {"url": "later", "title": "policy", "provider_seen_at": "2026-01-01T11:00:00Z"},
    ]
    graph = build_news_graph(rows, [], FakeEmbedder(), .7, 1)
    first = next(claim for claim in graph.claims if claim.event_article_url == "first")
    later = next(claim for claim in graph.claims if claim.event_article_url == "later")

    assert first.is_first_report
    assert first.event_title == "law"
    assert later.event_title == "policy"
    assert later.event_article_url == "later"
    assert later.representative_title == "law"
    assert asdict(later)["event_title"] == "policy"
    assert asdict(later)["event_article_url"] == "later"


def test_duplicate_ties_are_order_independent_and_zero_centroid_is_handled():
    class OpposingEmbedder:
        def encode(self, texts, role):
            return [[1.0, 0.0] if text in ("a", "aa") else [-1.0, 0.0] for text in texts]

    records = [
        {"url": "b", "title": "b", "provider_seen_at": "2026-01-01T10:01:00Z"},
        {"url": "dup", "title": "z", "provider_seen_at": "2026-01-01T09:00:00Z", "market_slug": "m2"},
        {"url": "dup", "title": "aa", "provider_seen_at": "2026-01-01T09:00:00Z", "market_slug": "m1"},
    ]
    markets = [{"market_id": "m1", "question": "a"}, {"market_id": "m2", "question": "b"}]
    first = build_news_graph(records, markets, OpposingEmbedder(), -1.0, 1)
    second = build_news_graph(list(reversed(records)), markets, OpposingEmbedder(), -1.0, 1)
    assert first == second
    assert all(np.isfinite(edge.cosine_similarity) for edge in first.market_edges)


def test_bridge_headline_does_not_merge_incompatible_cluster_members():
    class BridgeEmbedder:
        def encode(self, texts, role):
            vectors = {
                "A": [1.0, 0.0],
                "B": [0.8660254, 0.5],
                "C": [0.5, 0.8660254],
            }
            return [vectors.get(text, [1.0, 0.0]) for text in texts]

    records = [
        {"url": "a", "title": "A", "provider_seen_at": "2026-01-01T10:00:00Z"},
        {"url": "b", "title": "B", "provider_seen_at": "2026-01-01T11:00:00Z"},
        {"url": "c", "title": "C", "provider_seen_at": "2026-01-01T12:00:00Z"},
    ]
    graph = build_news_graph(records, [], BridgeEmbedder(), .8, 1)

    assert np.dot([1.0, 0.0], [0.8660254, 0.5]) >= .8
    assert np.dot([0.8660254, 0.5], [0.5, 0.8660254]) >= .8
    assert np.dot([1.0, 0.0], [0.5, 0.8660254]) < .8
    assert [article.url for article in graph.claims[1].members] == ["a", "b"]
    assert [article.url for article in graph.claims[2].members] == ["c"]

"""Regresiones de curación ante fusiones encadenadas y metadatos dañados."""
import curation


def test_repeats_counts_transitive_merges():
    cur = {"retired": {
        "e:a": {"merged_into": "e:root"},
        "e:b": {"merged_into": "e:a"},
        "e:c": {"merged_into": "e:root"},
    }}
    assert curation.repeats(cur, "e:root") == 4


def test_repeats_terminates_on_cycle():
    cur = {"retired": {
        "e:a": {"merged_into": "e:b"},
        "e:b": {"merged_into": "e:a"},
    }}
    assert curation.repeats(cur, "e:a") == 2


def test_repeats_handles_broken_metadata():
    cur = {"retired": {"e:a": None, "e:b": {"merged_into": "e:root"}}}
    assert curation.repeats(cur, "e:root") == 2

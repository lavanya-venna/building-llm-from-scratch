"""Tests for hindi/data/reconstruct_documents.py.

IndicCorpV2's raw export is one sentence/paragraph-fragment per line, with
blank-text lines marking document boundaries -- these tests pin down how
those lines get regrouped into real multi-sentence documents.
"""
from hindi.data.reconstruct_documents import regroup_paragraph_documents


def test_regroup_joins_consecutive_nonblank_lines_into_one_document():
    records = [
        {"text": "पहला वाक्य।"},
        {"text": "दूसरा वाक्य।"},
        {"text": ""},
        {"text": "तीसरा वाक्य।"},
    ]
    docs = regroup_paragraph_documents(records, source_name="indiccorp_v2")
    assert len(docs) == 2
    assert docs[0]["text"] == "पहला वाक्य।\nदूसरा वाक्य।"
    assert docs[1]["text"] == "तीसरा वाक्य।"


def test_regroup_assigns_sequential_doc_ids_and_source():
    records = [{"text": "क"}, {"text": ""}, {"text": "ख"}]
    docs = regroup_paragraph_documents(records, source_name="src")
    assert docs[0]["doc_id"] == "src-0"
    assert docs[1]["doc_id"] == "src-1"
    assert all(d["source"] == "src" for d in docs)


def test_regroup_ignores_leading_and_trailing_blank_lines():
    records = [{"text": ""}, {"text": "क"}, {"text": "ख"}, {"text": ""}]
    docs = regroup_paragraph_documents(records, source_name="src")
    assert len(docs) == 1
    assert docs[0]["text"] == "क\nख"


def test_regroup_handles_all_blank_input():
    records = [{"text": ""}, {"text": ""}]
    docs = regroup_paragraph_documents(records, source_name="src")
    assert docs == []

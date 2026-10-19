from aml_graphrag.application.indexing import ENC, chunk, clean, merge


def test_chunks_overlap_and_cover_the_text():
    text = " ".join(f"word{i}" for i in range(2000))
    pieces = chunk(text, size=600, overlap=100)
    assert all(len(ENC.encode(p)) <= 600 for p in pieces)
    assert pieces[0].split()[0] == "word0" and pieces[-1].split()[-1] == "word1999"


def test_clean_drops_the_link_lists():
    text = "Lead.\n\n== History ==\nMore.\n\n== See also ==\n- Something\n\n== References =="
    assert clean(text) == "Lead.\n\n== History ==\nMore."


def test_merge_joins_names_and_adds_up_repeated_relationships():
    a = {"entities": [{"name": "Liberty Reserve", "type": "organization", "description": "A currency."},
                      {"name": "Costa Rica", "type": "geo", "description": "A country."}],
         "relationships": [{"source": "Liberty Reserve", "target": "Costa Rica", "description": "Based in.", "strength": 8}]}
    b = {"entities": [{"name": "liberty  reserve", "type": "organization", "description": "Shut down in 2013."}],
         "relationships": [{"source": "Costa Rica", "target": "Liberty Reserve", "description": "Seized it.", "strength": 6},
                           {"source": "Arthur Budovsky", "target": "Liberty Reserve", "description": "Founded.", "strength": 9}]}
    nodes, edges = merge([("c1", a), ("c2", b)])
    assert nodes["LIBERTY RESERVE"]["descriptions"] == ["A currency.", "Shut down in 2013."]
    assert nodes["LIBERTY RESERVE"]["chunks"] == ["c1", "c2"]
    assert edges[("COSTA RICA", "LIBERTY RESERVE")]["weight"] == 14
    assert nodes["ARTHUR BUDOVSKY"]["type"] == "unknown"  # only seen as a relationship endpoint

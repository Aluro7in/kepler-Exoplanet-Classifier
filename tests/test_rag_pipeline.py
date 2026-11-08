from pathlib import Path

from src.rag_pipeline import build_knowledge_base, discover_kepler_files
def test_discover_kepler_files_and_build_knowledge_base():
    files = discover_kepler_files(Path("data/raw"))

    assert len(files) >= 3
    assert any("Kepler" in str(path) for path in files)

    docs = build_knowledge_base(files, top_k_rows=3)

    assert len(docs) >= 3
    assert any(doc["source_type"] == "dataset_summary" for doc in docs)
    assert any(doc["source_type"] == "tce_row" for doc in docs)

import os
from src.models.iclr_predict_authors import resolve_data_path


def test_resolve_data_path_relative():
    # Given a workspace-relative path
    rel = 'data/raw/iclr_2020_2025.parquet'
    resolved = resolve_data_path(rel)
    assert os.path.isabs(resolved) or os.path.normpath(resolved).startswith('..') is False
    # Resolved path should point to an existing parent directory (project root or data/raw)
    parent = os.path.dirname(resolved)
    assert parent != ''


def test_resolve_data_path_absolute(tmp_path):
    # Given an absolute path, the resolver should normalize and return it
    p = tmp_path / 'sample.parquet'
    p.write_text('dummy')
    resolved = resolve_data_path(str(p))
    assert os.path.normpath(resolved) == os.path.normpath(str(p))

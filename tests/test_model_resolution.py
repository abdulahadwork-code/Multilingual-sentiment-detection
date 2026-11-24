import pytest
from xlmr_sentiment.inference import resolve_model_dir
def test_local_dir_with_config_is_used(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    assert resolve_model_dir(tmp_path) == tmp_path
@pytest.mark.parametrize("bad", ["artifacts\\missing\\best", "no_slash_name", "a/b/c"])
def test_non_repo_strings_raise_without_network(bad):
    with pytest.raises(FileNotFoundError):
        resolve_model_dir(bad)
def test_missing_absolute_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_model_dir(tmp_path / "missing")

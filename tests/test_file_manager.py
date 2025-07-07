import datetime as dt
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ifera.file_manager import (
    FileManager,
    FileOperations,
    FileManagerContext,
    RuleType,
    get_literal_prefix,
    import_function,
    match_pattern,
    partial_substitute_pattern,
    pattern_to_regex,
    pattern_to_regex_custom,
    substitute_pattern,
)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


def test_pattern_to_regex_custom():
    regex, names = pattern_to_regex_custom("path/{name}/{date:\\d{8}}.txt")
    assert names == ["name", "date"]
    assert regex == "^path/(.+?)/(\\d{8)\\}\\.txt$"


def test_get_literal_prefix():
    assert get_literal_prefix("foo/{bar}/baz") == "foo/"
    assert get_literal_prefix("foo/bar") == "foo/bar"


def test_pattern_to_regex():
    regex, names = pattern_to_regex("/foo/{bar}/baz-{qux}.txt")
    assert regex == r"^/foo/(.+?)/baz\-(.+?)\.txt$"
    assert names == ["bar", "qux"]


def test_match_pattern():
    pattern = "file:/data/{source}/{symbol}.csv"
    file = "file:/data/raw/XYZ.csv"
    wildcards, matched = match_pattern(pattern, file)
    assert matched is True
    assert wildcards == {"source": "raw", "symbol": "XYZ"}

    wildcards, matched = match_pattern(pattern, "s3:/data/raw/XYZ.csv")
    assert matched is False
    assert wildcards == {}

    # Pattern without wildcards should still match exactly
    pattern_no_wc = "file:/data/raw/XYZ.csv"
    wildcards, matched = match_pattern(pattern_no_wc, "file:/data/raw/XYZ.csv")
    assert matched is True
    assert wildcards == {}

    wildcards, matched = match_pattern(pattern_no_wc, "file:/data/raw/ABC.csv")
    assert matched is False
    assert wildcards == {}


def test_substitute_pattern():
    assert (
        substitute_pattern("file:{symbol}.txt", {"symbol": "AAPL"}) == "file:AAPL.txt"
    )
    with pytest.raises(ValueError):
        substitute_pattern("file:{symbol}.txt", {"other": "x"})


def test_partial_substitute_pattern():
    result = partial_substitute_pattern("s3:{bucket}/{name}", {"bucket": "b"})
    assert result == "s3:b/{name}"


def test_expand_dependency_wildcards(monkeypatch):
    dep_entry = {
        "pattern": "s3:tensor/data/{symbol}-{code}.pt",
        "wildcard_expansion": "s3:raw/data/{symbol}-{code:[A-Z]{2}}.zip",
    }
    objects = [
        "raw/data/CL-AA.zip",
        "raw/data/CL-BB.zip",
        "raw/data/CL-cc.zip",
    ]
    mock = MagicMock(return_value=objects)
    monkeypatch.setattr("ifera.file_manager.list_s3_objects", mock)
    fm = FileManager(config_file="../tests/test_dependencies.yml")
    ctx = FileManagerContext()
    result = fm._expand_dependency_wildcards(dep_entry, {"symbol": "CL"}, ctx)
    mock.assert_called_once_with("raw/data/CL-")
    assert result == []


def test_expand_dependency_wildcards_function():
    dep_entry = {
        "pattern": "s3:tensor/data/{symbol}-{code}.pt",
        "expansion_function": "tests.helper_module.expand_codes",
    }
    fm = FileManager(config_file="../tests/test_dependencies.yml")
    ctx = FileManagerContext()
    result = fm._expand_dependency_wildcards(dep_entry, {"symbol": "CL"}, ctx)
    assert result == [
        "s3:tensor/data/CL-AA.pt",
        "s3:tensor/data/CL-BB.pt",
    ]


# ---------------------------------------------------------------------------
# FileOperations tests
# ---------------------------------------------------------------------------


def test_file_operations_exists_cache(monkeypatch):
    fop = FileOperations()
    file = "file:/tmp/foo.txt"
    mock_exists = MagicMock(return_value=True)
    monkeypatch.setattr("ifera.file_manager.os.path.exists", mock_exists)
    assert fop.exists(file) is True
    assert mock_exists.call_count == 1
    assert fop.exists(file) is True
    assert mock_exists.call_count == 1
    fop.exists_cache.clear()
    fop.mtime_cache[file] = None
    assert fop.exists(file) is False


def test_file_operations_get_mtime(monkeypatch):
    fop = FileOperations()
    file = "file:/tmp/foo.txt"
    ts = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    monkeypatch.setattr("ifera.file_manager.os.path.exists", lambda p: True)
    mock_getmtime = MagicMock(return_value=ts.timestamp())
    monkeypatch.setattr("ifera.file_manager.os.path.getmtime", mock_getmtime)
    assert fop.get_mtime(file) == ts
    assert fop.get_mtime(file) == ts
    assert mock_getmtime.call_count == 1
    with pytest.raises(ValueError):
        fop.get_mtime("ftp:/invalid")


def test_import_function():
    func = import_function("math.sqrt")
    assert func(9) == 3
    with pytest.raises(ImportError):
        import_function("math.nope")


# ---------------------------------------------------------------------------
# FileManager graph tests
# ---------------------------------------------------------------------------


def test_file_manager_get_dependencies(file_manager_instance):
    fm = file_manager_instance
    deps = fm.get_dependencies("file:/tmp/processed/ABC.txt")
    assert deps == ["file:/tmp/raw/ABC.txt"]


def test_file_manager_get_node_params(file_manager_instance):
    fm = file_manager_instance
    params = fm.get_node_params(RuleType.DEPENDENCY, "file:/tmp/processed/ABC.txt")
    assert params == {"symbol": "ABC", "extra": "value"}


def test_dependencies_max_last_modified(monkeypatch, file_manager_instance):
    times = {"file:/tmp/raw/ABC.txt": dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)}

    class DummyFOP:
        def __init__(self) -> None:
            pass

        def get_mtime(self, file: str):
            return times.get(file)

    monkeypatch.setattr("ifera.file_manager.FileOperations", DummyFOP)
    fm = file_manager_instance
    result = fm.dependencies_max_last_modified("file:/tmp/processed/ABC.txt")
    assert result == times["file:/tmp/raw/ABC.txt"]


def test_build_subgraph(file_manager_instance):
    fm = file_manager_instance
    fm.build_subgraph("file:/tmp/processed/ABC.txt", RuleType.DEPENDENCY)
    graph = fm.dependency_graph
    assert "file:/tmp/raw/ABC.txt" in graph
    node = graph.nodes["file:/tmp/processed/ABC.txt"]
    assert node["wildcards"] == {"symbol": "ABC"}
    assert graph.has_edge("file:/tmp/processed/ABC.txt", "file:/tmp/raw/ABC.txt")
    assert "refresh_function" in node


def test_is_up_to_date_true(monkeypatch, file_manager_instance):
    now = dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc)
    times = {
        "file:/tmp/raw/ABC.txt": now - dt.timedelta(days=1),
        "file:/tmp/processed/ABC.txt": now,
    }

    class DummyFOP:
        def __init__(self, t):
            self.t = t

        def get_mtime(self, file: str):
            return self.t.get(file)

        def exists(self, _file: str) -> bool:
            return True

    monkeypatch.setattr("ifera.file_manager.FileOperations", lambda: DummyFOP(times))
    fm = file_manager_instance
    assert fm.is_up_to_date("file:/tmp/processed/ABC.txt") is True


def test_is_up_to_date_missing(monkeypatch, file_manager_instance):
    now = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    times = {"file:/tmp/raw/ABC.txt": now}

    class DummyFOP:
        def __init__(self, t):
            self.t = t

        def get_mtime(self, file: str):
            return self.t.get(file)

    monkeypatch.setattr("ifera.file_manager.FileOperations", lambda: DummyFOP(times))
    fm = file_manager_instance
    assert fm.is_up_to_date("file:/tmp/processed/ABC.txt") is False


def test_refresh_file_calls_process(monkeypatch, file_manager_instance):
    now = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    times = {
        "file:/tmp/raw/ABC.txt": now,
        "file:/tmp/processed/ABC.txt": now - dt.timedelta(days=1),
    }

    class DummyFOP:
        def __init__(self, t):
            self.t = t

        def get_mtime(self, file: str):
            return self.t.get(file)

        def remove_from_cache(self, _file: str):
            pass

        def remove(self, file: str, scheme):
            pass

        def exists(self, _file: str) -> bool:
            return True

        def exists(self, _file: str) -> bool:
            return True

        def exists(self, _file: str) -> bool:
            return True

        def exists(self, _file: str) -> bool:
            return True

        def exists(self, _file: str) -> bool:
            return True

    process_mock = MagicMock()
    monkeypatch.setattr("tests.helper_module.process", process_mock)
    monkeypatch.setattr("ifera.file_manager.FileOperations", lambda: DummyFOP(times))
    monkeypatch.setattr("ifera.file_manager.os.path.exists", lambda p: False)

    fm = file_manager_instance
    fm.refresh_file("file:/tmp/processed/ABC.txt")
    process_mock.assert_called_once_with(symbol="ABC", extra="value")


def test_refresh_stale_file_refresh_branch(monkeypatch, file_manager_refresh_instance):
    keys = ["raw/CL-AA.txt", "raw/CL-BB.txt"]
    monkeypatch.setattr("ifera.file_manager.list_s3_objects", lambda prefix: keys)
    now = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    times = {
        "file:/tmp/intermediate/CL-AA.txt": now,
        "file:/tmp/intermediate/CL-BB.txt": now,
        "file:/tmp/output/CL.txt": None,
    }

    class DummyFOP:
        def __init__(self, t):
            self.t = t

        def get_mtime(self, file: str):
            return self.t.get(file)

        def remove_from_cache(self, _file: str):
            pass

        def remove(self, file: str, scheme):
            pass

        def exists(self, _file: str) -> bool:
            return True

    combine_mock = MagicMock()
    monkeypatch.setattr("tests.helper_module.combine", combine_mock)
    monkeypatch.setattr("ifera.file_manager.FileOperations", lambda: DummyFOP(times))
    monkeypatch.setattr("ifera.file_manager.os.path.exists", lambda p: False)

    fm = file_manager_refresh_instance
    fm.refresh_file("file:/tmp/output/CL.txt")
    combine_mock.assert_called_once_with(symbol="CL", codes=["AA", "BB"])


def test_refresh_stale_file_refresh_branch_expansion_function(
    monkeypatch, file_manager_expand_function_instance
):
    now = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    times = {
        "file:/tmp/intermediate/CL-AA.txt": now,
        "file:/tmp/intermediate/CL-BB.txt": now,
        "file:/tmp/output/CL.txt": None,
    }

    class DummyFOP:
        def __init__(self, t):
            self.t = t

        def get_mtime(self, file: str):
            return self.t.get(file)

        def remove_from_cache(self, _file: str):
            pass

        def remove(self, file: str, scheme):
            pass

        def exists(self, _file: str) -> bool:
            return True

    combine_mock = MagicMock()
    import_function.cache_clear()
    monkeypatch.setattr("tests.helper_module.combine", combine_mock)
    monkeypatch.setattr("ifera.file_manager.FileOperations", lambda: DummyFOP(times))
    monkeypatch.setattr("ifera.file_manager.os.path.exists", lambda p: False)

    fm = file_manager_expand_function_instance
    fm.refresh_file("file:/tmp/output/CL.txt")
    combine_mock.assert_called_once_with(symbol="CL", codes=["AA", "BB"])


def test_expansion_function_with_dependencies(
    monkeypatch, file_manager_func_depends_instance
):
    now = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    times = {
        "file:/tmp/meta/CL.txt": None,
        "file:/tmp/intermediate/CL-AA.txt": now,
        "file:/tmp/intermediate/CL-BB.txt": now,
        "file:/tmp/output/CL.txt": None,
    }

    class DummyFOP:
        def __init__(self, t):
            self.t = t

        def get_mtime(self, file: str):
            return self.t.get(file)

        def remove_from_cache(self, _file: str):
            pass

        def remove(self, file: str, scheme):
            pass

        def exists(self, _file: str) -> bool:
            return True

    fetch_mock = MagicMock()
    combine_mock = MagicMock()
    import_function.cache_clear()
    monkeypatch.setattr("tests.helper_module.fetch", fetch_mock)
    monkeypatch.setattr("tests.helper_module.combine", combine_mock)
    monkeypatch.setattr("ifera.file_manager.FileOperations", lambda: DummyFOP(times))
    monkeypatch.setattr("ifera.file_manager.os.path.exists", lambda p: False)

    fm = file_manager_func_depends_instance
    fm.refresh_file("file:/tmp/output/CL.txt")

    assert fetch_mock.call_count >= 1
    fetch_mock.assert_called_with(symbol="CL")
    combine_mock.assert_called_once_with(symbol="CL", codes=["AA", "BB"])


def test_refresh_stale_file_list_args_error(monkeypatch, file_manager_refresh_instance):
    keys = ["raw/CL-AA.txt"]
    monkeypatch.setattr("ifera.file_manager.list_s3_objects", lambda prefix: keys)
    now = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    times = {
        "file:/tmp/intermediate/CL-AA.txt": now,
        "file:/tmp/output/CL.txt": None,
    }

    class DummyFOP:
        def __init__(self, t):
            self.t = t

        def get_mtime(self, file: str):
            return self.t.get(file)

        def remove_from_cache(self, _file: str):
            pass

        def remove(self, file: str, scheme):
            pass

        def exists(self, _file: str) -> bool:
            return True

    fm = file_manager_refresh_instance
    fm.build_subgraph("file:/tmp/output/CL.txt", RuleType.DEPENDENCY)
    fm.build_subgraph("file:/tmp/output/CL.txt", RuleType.REFRESH)
    for dep in fm.refresh_graph.successors("file:/tmp/output/CL.txt"):
        fm.refresh_graph.nodes[dep]["wildcards"] = {}

    monkeypatch.setattr("ifera.file_manager.FileOperations", lambda: DummyFOP(times))
    monkeypatch.setattr("ifera.file_manager.os.path.exists", lambda p: False)

    dummy_fop = DummyFOP(times)
    ctx = FileManagerContext(cache={}, fop=dummy_fop, temp_files=[])  # type: ignore
    with pytest.raises(RuntimeError):
        fm._refresh_stale_file("file:/tmp/output/CL.txt", False, ctx)


def test_parse_refresh_rule():
    fm = FileManager(config_file="../tests/test_dependencies.yml")
    func, args, spec, deps = fm._parse_refresh_rule("tests.helper_module.process", "f")
    assert func == "tests.helper_module.process"
    assert args == {}
    assert spec == {}
    assert deps == []

    func_dict = {
        "name": "tests.helper_module.combine",
        "additional_args": {"a": 1},
        "list_args": {"codes": "code"},
    }
    func, args, spec, deps = fm._parse_refresh_rule(func_dict, "f")
    assert func == "tests.helper_module.combine"
    assert args == {"a": 1}
    assert spec == {"codes": "code"}
    assert deps == []


def test_build_list_args(monkeypatch, file_manager_refresh_instance):
    keys = ["raw/CL-AA.txt", "raw/CL-BB.txt"]
    monkeypatch.setattr("ifera.file_manager.list_s3_objects", lambda prefix: keys)
    fm = file_manager_refresh_instance
    file = "file:/tmp/output/CL.txt"
    fm.build_subgraph(file, RuleType.DEPENDENCY)
    fm.build_subgraph(file, RuleType.REFRESH)
    spec = {"codes": "code"}
    result = fm._build_list_args(file, RuleType.REFRESH, spec)
    assert result == {"codes": ["AA", "BB"]}


def test_where_clause_selects_rule(file_manager_where_instance):
    fm = file_manager_where_instance
    deps = fm.get_dependencies("file:/tmp/foo/1m/AAA.txt")
    assert deps == ["file:/tmp/raw/foo/AAA.txt"]
    deps = fm.get_dependencies("file:/tmp/bar/3m/AAA.txt")
    assert deps == ["file:/tmp/raw/bar/AAA.txt"]


def test_where_clause_no_match(file_manager_where_instance):
    fm = file_manager_where_instance
    fm.build_subgraph("file:/tmp/foo/2m/AAA.txt", RuleType.DEPENDENCY)
    graph = fm.dependency_graph
    assert "file:/tmp/foo/2m/AAA.txt" not in graph

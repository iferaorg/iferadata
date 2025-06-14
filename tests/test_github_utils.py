import datetime as dt
from unittest.mock import MagicMock

import pytest
from github.GithubException import GithubException

from ifera import github_utils


# ---------------------------------------------------------------------------
# parse_github_url tests
# ---------------------------------------------------------------------------


def test_parse_github_url_valid():
    url = "github:owner/repo/path/to/file.txt"
    owner, repo, path = github_utils.parse_github_url(url)
    assert owner == "owner"
    assert repo == "repo"
    assert path == "path/to/file.txt"


def test_parse_github_url_invalid_scheme():
    with pytest.raises(ValueError):
        github_utils.parse_github_url("https://owner/repo/file.txt")


def test_parse_github_url_invalid_format():
    with pytest.raises(ValueError):
        github_utils.parse_github_url("github://owner/repo")


# ---------------------------------------------------------------------------
# check_github_file_exists tests
# ---------------------------------------------------------------------------


def test_check_github_file_exists_true(mock_github):
    url = "github:owner/repo/file.txt"
    repo = MagicMock()
    mock_github.github_client.get_repo.return_value = repo
    repo.get_contents.return_value = MagicMock()

    assert github_utils.check_github_file_exists(url) is True
    mock_github.github_client.get_repo.assert_called_once_with("owner/repo")
    repo.get_contents.assert_called_once_with("file.txt")


def test_check_github_file_exists_false(mock_github):
    url = "github:owner/repo/missing.txt"
    repo = MagicMock()
    repo.get_contents.side_effect = GithubException(404, {})
    mock_github.github_client.get_repo.return_value = repo

    assert github_utils.check_github_file_exists(url) is False
    mock_github.github_client.get_repo.assert_called_once_with("owner/repo")
    repo.get_contents.assert_called_once_with("missing.txt")


def test_check_github_file_exists_error(mock_github):
    url = "github:owner/repo/error.txt"
    repo = MagicMock()
    repo.get_contents.side_effect = GithubException(500, {})
    mock_github.github_client.get_repo.return_value = repo

    with pytest.raises(GithubException):
        github_utils.check_github_file_exists(url)


# ---------------------------------------------------------------------------
# get_github_last_modified tests
# ---------------------------------------------------------------------------


def test_get_github_last_modified(mock_github):
    url = "github:owner/repo/file.txt"
    repo = MagicMock()
    commit = MagicMock()
    commit.commit.committer.date = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
    repo.get_commits.return_value = [commit]
    mock_github.github_client.get_repo.return_value = repo

    result = github_utils.get_github_last_modified(url)
    assert result == commit.commit.committer.date
    mock_github.github_client.get_repo.assert_called_once_with("owner/repo")
    repo.get_commits.assert_called_once_with(path="file.txt")


def test_get_github_last_modified_not_found(mock_github):
    url = "github:owner/repo/missing.txt"
    repo = MagicMock()
    repo.get_commits.side_effect = GithubException(404, {})
    mock_github.github_client.get_repo.return_value = repo

    assert github_utils.get_github_last_modified(url) is None
    repo.get_commits.assert_called_once_with(path="missing.txt")


def test_get_github_last_modified_no_commits(mock_github):
    url = "github:owner/repo/empty.txt"
    repo = MagicMock()
    repo.get_commits.return_value = []
    mock_github.github_client.get_repo.return_value = repo

    assert github_utils.get_github_last_modified(url) is None
    repo.get_commits.assert_called_once_with(path="empty.txt")

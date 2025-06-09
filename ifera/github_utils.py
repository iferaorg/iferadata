from github import Github
from github.GithubException import GithubException
from urllib.parse import urlparse
from typing import Tuple
import datetime as dt

from .settings import settings
from .decorators import singleton, ThreadSafeCache

@singleton
class GitHubClientSingleton:
    def __init__(self) -> None:
        token = settings.GITHUB_TOKEN
        if token:
            self._github_client = Github(token)
        else:
            # Note: unauthenticated has stricter rate limits
            self._github_client = Github()

    @property
    def github_client(self) -> Github:
        """Return the GitHub client instance."""
        return self._github_client


def parse_github_url(url: str) -> Tuple[str, str, str]:
    """Parse a GitHub URL into owner, repo, and path components.

    Format: github://owner/repo/path/to/file.ext
    """
    parts = urlparse(url)

    if parts.scheme != "github":
        raise ValueError(f"Not a GitHub URL: {url}")

    # Split the path, removing leading slash
    path_parts = parts.path.split("/", 2)

    if len(path_parts) < 3:
        raise ValueError(
            f"Invalid GitHub URL format: {url}. Expected github://owner/repo/path/to/file"
        )

    owner = path_parts[0]
    repo = path_parts[1]
    file_path = path_parts[2]

    return owner, repo, file_path


@ThreadSafeCache()
def check_github_file_exists(url: str) -> bool:
    """Check if a file exists in the GitHub repository."""
    github_client = GitHubClientSingleton().github_client

    owner, repo, file_path = parse_github_url(url)
    try:
        github_client.get_repo(f"{owner}/{repo}").get_contents(file_path)
        return True
    except GithubException as e:
        if e.status == 404:
            return False
        raise


@ThreadSafeCache()
def get_github_last_modified(url: str) -> dt.datetime | None:
    """Get the last modified date of a file in the GitHub repository."""
    github_client = GitHubClientSingleton().github_client

    owner, repo, file_path = parse_github_url(url)
    try:
        commits = github_client.get_repo(f"{owner}/{repo}").get_commits(path=file_path)
        if not commits:
            return None
        # Get the last commit for the file
        last_commit = commits[0]
        return last_commit.commit.committer.date
    except GithubException as e:
        if e.status == 404:
            return None
        raise

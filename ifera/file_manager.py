import os
import re
import datetime
from typing import Dict, List, Optional, Set, Tuple, Callable, Any
import yaml
import networkx as nx
import boto3
from urllib.parse import urlparse
import importlib
from functools import lru_cache
from github import Github
from github.GithubException import GithubException
import requests

# Initialize S3 client
s3_client = boto3.client("s3")

# Initialize GitHub client (using singleton pattern to avoid multiple instances)
_github_client = None


def get_github_client() -> Github:
    """Get or create a GitHub client instance."""
    global _github_client
    if _github_client is None:
        # Use environment variable for token if available, otherwise unauthenticated
        token = os.environ.get("GITHUB_TOKEN")
        if token:
            _github_client = Github(token)
        else:
            # Note: unauthenticated has stricter rate limits
            _github_client = Github()
    return _github_client


# Helper Functions


def pattern_to_regex(pattern_path: str) -> Tuple[str, List[str]]:
    """Convert a pattern path to a regex and extract wildcard names."""
    wildcard_names = []

    def replacer(match):
        wildcard_names.append(match.group(1))
        return "(.+?)"

    regex_path = re.sub(r"\{(\w+)\}", replacer, pattern_path)
    return "^" + regex_path + "$", wildcard_names


def match_pattern(pattern: str, file: str) -> Optional[Dict[str, str]]:
    """Match a file against a pattern and extract wildcard values."""
    pattern_parts = urlparse(pattern)
    file_parts = urlparse(file)

    # Quick scheme check
    if pattern_parts.scheme != file_parts.scheme:
        return None

    # Check bucket/domain matches for S3
    if pattern_parts.scheme == "s3" and pattern_parts.netloc != file_parts.netloc:
        return None

    # For local paths, ensure netloc is empty
    if pattern_parts.scheme == "local" and pattern_parts.netloc != "":
        return None

    # For GitHub paths, check repository match
    if pattern_parts.scheme == "github" and pattern_parts.netloc != file_parts.netloc:
        return None

    regex, wildcard_names = pattern_to_regex(pattern_parts.path)
    match = re.match(regex, file_parts.path)

    if match:
        return dict(zip(wildcard_names, match.groups()))
    return None


def substitute_pattern(pattern: str, wildcards: Dict[str, str]) -> str:
    """Substitute wildcard values into a pattern."""

    def replacer(match):
        wildcard = match.group(1)
        if wildcard not in wildcards:
            raise ValueError(f"Wildcard '{wildcard}' not found in provided values")
        return wildcards[wildcard]

    return re.sub(r"\{(\w+)\}", replacer, pattern)


def parse_github_url(url: str) -> Tuple[str, str, str]:
    """Parse a GitHub URL into owner, repo, and path components.

    Format: github://owner/repo/path/to/file.ext
    """
    parts = urlparse(url)
    if parts.scheme != "github":
        raise ValueError(f"Not a GitHub URL: {url}")

    # Split the path, removing leading slash
    path_parts = parts.path.lstrip("/").split("/", 2)
    if len(path_parts) < 3:
        raise ValueError(
            f"Invalid GitHub URL format: {url}. Expected github://owner/repo/path/to/file"
        )

    owner = path_parts[0]
    repo = path_parts[1]
    file_path = path_parts[2]

    return owner, repo, file_path


class FileOperations:
    """Abstract file operations for different storage systems."""

    @staticmethod
    def exists(file: str) -> bool:
        """Check if a file (local, S3, or GitHub) exists."""
        parts = urlparse(file)
        if parts.scheme == "local":
            return os.path.exists(parts.path.lstrip("/"))
        elif parts.scheme == "s3":
            try:
                s3_client.head_object(Bucket=parts.netloc, Key=parts.path.lstrip("/"))
                return True
            except s3_client.exceptions.ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    return False
                raise
        elif parts.scheme == "github":
            try:
                owner, repo_name, path = parse_github_url(file)
                g = get_github_client()
                repo = g.get_repo(f"{owner}/{repo_name}")

                # Try to get the file contents to check existence
                try:
                    repo.get_contents(path)
                    return True
                except GithubException as e:
                    if e.status == 404:  # File not found
                        return False
                    raise
            except Exception as e:
                print(f"Error checking GitHub file existence: {e}")
                return False
        raise ValueError(f"Unsupported scheme: {parts.scheme}")

    @staticmethod
    def get_mtime(file: str) -> datetime.datetime:
        """Get the modification time of a file (local, S3, or GitHub) in UTC."""
        parts = urlparse(file)
        if parts.scheme == "local":
            path = parts.path.lstrip("/")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Local file not found: {path}")
            return datetime.datetime.fromtimestamp(
                os.path.getmtime(path), tz=datetime.timezone.utc
            )
        elif parts.scheme == "s3":
            try:
                response = s3_client.head_object(
                    Bucket=parts.netloc, Key=parts.path.lstrip("/")
                )
                return response["LastModified"]
            except s3_client.exceptions.ClientError as e:
                if e.response["Error"]["Code"] == "404":
                    raise FileNotFoundError(
                        f"S3 object not found: {parts.netloc}/{parts.path}"
                    )
                raise
        elif parts.scheme == "github":
            owner, repo_name, path = parse_github_url(file)

            try:
                g = get_github_client()
                repo = g.get_repo(f"{owner}/{repo_name}")

                # Get commit information for the file
                commits = list(repo.get_commits(path=path))
                if not commits:
                    raise FileNotFoundError(
                        f"No commit history for file: {owner}/{repo_name}/{path}"
                    )

                # Get the date of the most recent commit
                latest_commit = commits[0]
                return latest_commit.commit.committer.date

            except GithubException as e:
                if e.status == 404:
                    raise FileNotFoundError(
                        f"GitHub file not found: {owner}/{repo_name}/{path}"
                    )
                raise
            except Exception as e:
                raise RuntimeError(f"Error getting GitHub file timestamp: {e}")
        raise ValueError(f"Unsupported scheme: {parts.scheme}")


@lru_cache(maxsize=100)
def import_function(func_str: str) -> Callable:
    """Import a function from a string (e.g., 'module.function')."""
    try:
        module_name, func_name = func_str.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, func_name)
    except (ValueError, ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import function '{func_str}': {e}")


class FileManager:
    def __init__(self, config_file: str = "dependencies.yaml"):
        """Initialize with a list of dependency rules."""
        self.graph = nx.DiGraph()
        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
                if not isinstance(config, dict) or "rules" not in config:
                    raise ValueError(f"Invalid config format in {config_file}")
                self.rules = config["rules"]
        except (yaml.YAMLError, IOError) as e:
            raise ValueError(f"Error loading config from {config_file}: {e}")

    def build_subgraph(self, file: str) -> None:
        """Build the dependency subgraph starting from a file."""
        self._build_subgraph(file, set())

    def _build_subgraph(self, file: str, visited: Set[str]) -> None:
        """Recursively build the graph for a file and its dependencies."""
        if file in visited:
            return
        visited.add(file)

        for rule in self.rules:
            wildcards = match_pattern(rule["dependent"], file)
            if wildcards:
                self.graph.add_node(
                    file, refresh_function=rule["refresh_function"], wildcards=wildcards
                )
                for dep_pattern in rule["depends_on"]:
                    try:
                        dep_file = substitute_pattern(dep_pattern, wildcards)
                        self.graph.add_edge(file, dep_file)
                        self._build_subgraph(dep_file, visited)
                    except ValueError as e:
                        # Log the error but continue with other dependencies
                        print(
                            f"Warning: {e} when processing dependency {dep_pattern} for {file}"
                        )

    def is_up_to_date(self, file: str) -> bool:
        """Check if a file is up-to-date, building its subgraph if needed."""
        if file not in self.graph:
            self.build_subgraph(file)
        return self._is_up_to_date(file, {})

    def _is_up_to_date(self, file: str, cache: Dict[str, bool]) -> bool:
        """Recursively check if a file and its dependencies are up-to-date."""
        if file in cache:
            return cache[file]

        # Check if the file exists
        try:
            if not FileOperations.exists(file):
                cache[file] = False
                return False

            mtime = FileOperations.get_mtime(file)

            # Check all dependencies
            for dep in self.graph.successors(file):
                if not self._is_up_to_date(dep, cache):
                    cache[file] = False
                    return False

                try:
                    dep_mtime = FileOperations.get_mtime(dep)
                    if mtime <= dep_mtime:
                        cache[file] = False
                        return False
                except FileNotFoundError:
                    # Dependency doesn't exist, so file is outdated
                    cache[file] = False
                    return False

            cache[file] = True
            return True

        except (ValueError, FileNotFoundError) as e:
            print(f"Error checking if {file} is up to date: {e}")
            cache[file] = False
            return False

    def refresh_file(self, file: str) -> None:
        """Refresh a file if stale or missing, building its subgraph if needed."""
        if file not in self.graph:
            self.build_subgraph(file)
        self._refresh_file(file)

    def _refresh_file(self, file: str) -> None:
        """Recursively refresh dependencies, then the file if needed."""
        # First refresh all dependencies
        for dep in self.graph.successors(file):
            self._refresh_file(dep)

        # Then check if this file needs refreshing
        if not self.is_up_to_date(file):
            try:
                node_data = self.graph.nodes[file]
                refresh_func_str = node_data.get("refresh_function")
                if not refresh_func_str:
                    raise ValueError(f"No refresh function defined for file {file}")

                refresh_func = import_function(refresh_func_str)
                wildcards = node_data.get("wildcards", {})

                # Execute the refresh function with the extracted wildcards
                refresh_func(**wildcards)

            except Exception as e:
                raise RuntimeError(f"Failed to refresh file {file}: {e}")

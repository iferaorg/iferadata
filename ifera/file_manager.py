import os
import re
import datetime
from typing import Dict, List, Optional, Set, Tuple, Callable
import yaml
import networkx as nx
import boto3
from urllib.parse import urlparse
import importlib
from functools import lru_cache
from github import Github
from github.GithubException import GithubException
from .config import BaseInstrumentConfig
from .enums import Scheme, Source
from .url_utils import make_instrument_url
from .settings import settings
from .decorators import singleton
from .s3_utils import check_s3_file_exists, get_s3_last_modified

# Initialize GitHub client (using singleton pattern to avoid multiple instances)
_github_client = None


def get_github_client() -> Github:
    """Get or create a GitHub client instance."""
    global _github_client
    if _github_client is None:
        # Use token from settings if available, otherwise unauthenticated
        token = settings.GITHUB_TOKEN
        if token:
            _github_client = Github(token)
        else:
            # Note: unauthenticated has stricter rate limits
            _github_client = Github()
    return _github_client


# Helper Functions


def pattern_to_regex(pattern_path: str) -> Tuple[str, List[str]]:
    """
    Convert a pattern path to a regex and extract wildcard names.

    Args:
        pattern_path (str): A pattern string with {wildcard} placeholders, e.g.,
                           "file:data/{source}/{type}/{interval}/{symbol}.{ext}"

    Returns:
        Tuple[str, List[str]]: A tuple containing the regex pattern and a list of wildcard names.
    """
    # Split the pattern_path into literal parts and {wildcard} parts
    split_list = re.split(r"(\{\w+\})", pattern_path)
    regex_parts = []
    wildcard_names = []

    # Process each part from the split
    for part in split_list:
        if re.match(r"\{\w+\}", part):
            # This is a {wildcard} part, e.g., "{source}"
            name = part[1:-1]  # Extract the name inside braces, e.g., "source"
            wildcard_names.append(name)
            regex_parts.append("(.+?)")  # Non-greedy capturing group
        else:
            # This is a literal part, escape any special regex characters
            regex_parts.append(re.escape(part))

    # Combine all parts into a single regex string
    regex_string = "".join(regex_parts)
    # Add ^ and $ to match the entire string
    return "^" + regex_string + "$", wildcard_names


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
    if pattern_parts.scheme == "file" and pattern_parts.netloc != "":
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
    path_parts = parts.path.split("/", 2)
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

    def __init__(self):
        self.exists_cache = {}
        self.mtime_cache = {}

    def exists(self, file: str) -> bool:
        """Check if a file (file, S3, or GitHub) exists."""
        if file in self.exists_cache:
            return self.exists_cache[file]

        if file in self.mtime_cache:
            self.exists_cache[file] = self.mtime_cache[file] is not None
            return self.exists_cache[file]

        parts = urlparse(file)
        try:
            scheme = Scheme(parts.scheme)
        except ValueError:
            raise ValueError(f"Unsupported scheme: {parts.scheme}")

        if scheme == Scheme.FILE:
            result = os.path.exists(parts.path)
        elif scheme == Scheme.S3:
            result = check_s3_file_exists(settings.S3_BUCKET, parts.path)
        elif scheme == Scheme.GITHUB:
            try:
                owner, repo_name, path = parse_github_url(file)
                g = get_github_client()
                repo = g.get_repo(f"{owner}/{repo_name}")

                # Try to get the file contents to check existence
                try:
                    repo.get_contents(path)
                    result = True
                except GithubException as e:
                    if e.status == 404:  # File not found
                        result = False
                    else:
                        raise
            except Exception as e:
                print(f"Error checking GitHub file existence: {e}")
                return False
        else:
            raise ValueError(f"Unsupported scheme: {parts.scheme}")

        self.exists_cache[file] = result
        return result

    def get_mtime(self, file: str) -> datetime.datetime | None:
        """
        Get the modification time of a file (file, S3, or GitHub) in UTC.
        Returns None if the file does not exist.
        """
        if file in self.mtime_cache:
            return self.mtime_cache[file]

        parts = urlparse(file)
        try:
            scheme = Scheme(parts.scheme)
        except ValueError:
            raise ValueError(f"Unsupported scheme: {parts.scheme}")

        if scheme == Scheme.FILE:
            path = parts.path
            if not os.path.exists(path):
                result = None
            else:
                result = datetime.datetime.fromtimestamp(
                    os.path.getmtime(path), tz=datetime.timezone.utc
                )
        elif scheme == Scheme.S3:
            result = get_s3_last_modified(settings.S3_BUCKET, parts.path)
        elif scheme == Scheme.GITHUB:
            owner, repo_name, path = parse_github_url(file)

            try:
                g = get_github_client()
                repo = g.get_repo(f"{owner}/{repo_name}")

                # Get commit information for the file
                commits = list(repo.get_commits(path=path))
                if not commits:
                    result = None
                else:
                    result = commits[0].commit.committer.date
            except GithubException as e:
                if e.status == 404:
                    result = None
                else:
                    raise
            except Exception as e:
                raise RuntimeError(f"Error getting GitHub file timestamp: {e}")
        else:
            raise ValueError(f"Unsupported scheme: {parts.scheme}")

        self.mtime_cache[file] = result
        return result


@lru_cache(maxsize=100)
def import_function(func_str: str) -> Callable:
    """Import a function from a string (e.g., 'module.function')."""
    try:
        module_name, func_name = func_str.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, func_name)
    except (ValueError, ImportError, AttributeError) as e:
        raise ImportError(f"Failed to import function '{func_str}': {e}")


@singleton
class FileManager:
    def __init__(self, config_file: str = "dependencies.yml"):
        """Initialize with a list of dependency rules."""

        self.graph = nx.DiGraph()
        try:
            # Load the configuration file from the package directory
            package_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(package_dir, config_file)
            with open(file_path, "r") as f:
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

                # Only match the first rule that applies
                break

    def is_up_to_date(self, file: str) -> bool:
        """Check if a file is up-to-date, building its subgraph if needed."""
        if file not in self.graph:
            self.build_subgraph(file)

        if file not in self.graph:
            raise ValueError(f"File {file} not found in dependency graph")

        fop = FileOperations()
        return self._is_up_to_date(file, {}, fop)

    def _is_up_to_date(
        self, file: str, cache: Dict[str, bool], fop: FileOperations
    ) -> bool:
        """Recursively check if a file and its dependencies are up-to-date."""
        if file in cache:
            return cache[file]

        result = True

        # Check if the file exists
        try:
            mtime = fop.get_mtime(file)

            if mtime is None:
                result = False
            else:
                # Check all dependencies
                for dep in self.graph.successors(file):
                    if not self._is_up_to_date(dep, cache, fop):
                        result = False
                        break

                    dep_mtime = fop.get_mtime(dep)

                    # Check if the file is newer than its dependencies
                    if dep_mtime is None or mtime <= dep_mtime:
                        result = False
                        break
        except (ValueError, FileNotFoundError) as e:
            print(f"Error checking if {file} is up to date: {e}")
            result = False

        cache[file] = result
        return result

    def refresh_file(self, file: str, reset: bool = False) -> None:
        """Refresh a file if stale or missing, building its subgraph if needed."""
        if file not in self.graph:
            self.build_subgraph(file)

        if file not in self.graph:
            raise ValueError(f"File {file} not found in dependency graph")

        fop = FileOperations()
        self._refresh_file(file, reset, {}, fop)

    def _refresh_file(
        self, file: str, reset: bool, cache: Dict[str, bool], fop: FileOperations
    ) -> None:
        """Recursively refresh dependencies, then the file if needed."""
        has_successors = False

        # First refresh all dependencies
        for dep in self.graph.successors(file):
            self._refresh_file(dep, reset, cache, fop)
            has_successors = True

        # Then check if this file needs refreshing
        if has_successors and (reset or not self._is_up_to_date(file, cache, fop)):
            try:
                node_data = self.graph.nodes[file]
                refresh_func_str = node_data.get("refresh_function")
                if not refresh_func_str:
                    raise ValueError(f"No refresh function defined for file {file}")

                refresh_func = import_function(refresh_func_str)
                wildcards = node_data.get("wildcards", {})

                # Execute the refresh function with the extracted wildcards
                refresh_func(**wildcards, reset=reset)

            except Exception as e:
                raise RuntimeError(f"Failed to refresh file {file}: {e}")

    def get_dependencies(self, file: str) -> List[str]:
        """Get all dependencies for a file."""
        if file not in self.graph:
            self.build_subgraph(file)
        return list(self.graph.successors(file))


def refresh_file(
    scheme: Scheme,
    source: Source,
    instrument: BaseInstrumentConfig,
    reset: bool = False,
) -> None:
    """
    Refresh a file from S3 if it is not up to date.
    The file is expected to be in a specific directory structure.
    The S3 file is expected to exist and up-to-date.
    """
    fm = FileManager()
    url = make_instrument_url(
        scheme,
        source,
        instrument,
    )
    fm.refresh_file(url, reset=reset)

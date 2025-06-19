import datetime
import importlib
import os
import re
from enum import Enum
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import networkx as nx
import yaml

from .decorators import singleton
from .enums import Scheme
from .github_utils import check_github_file_exists, get_github_last_modified
from .s3_utils import check_s3_file_exists, get_s3_last_modified, list_s3_objects
from .settings import settings

# Helper Functions


def pattern_to_regex_custom(pattern_path: str) -> Tuple[str, List[str]]:
    """
    Convert a pattern with possible regex constraints inside wildcards to a regex.
    Supports placeholders of the form {name} or {name:regex}.
    Returns a tuple of (regex_pattern, list_of_wildcard_names).
    """
    regex_parts = []
    wildcard_names = []
    last_index = 0
    # This regex matches both {wildcard} and {wildcard:regex}
    placeholder_regex = re.compile(r"\{([^}:]+)(?::([^}]+))?\}")

    for match in placeholder_regex.finditer(pattern_path):
        literal_text = pattern_path[last_index : match.start()]
        regex_parts.append(re.escape(literal_text))
        name = match.group(1)
        wildcard_names.append(name)
        custom_regex = match.group(2) if match.group(2) is not None else ".+?"
        regex_parts.append(f"({custom_regex})")
        last_index = match.end()

    regex_parts.append(re.escape(pattern_path[last_index:]))
    regex_string = "^" + "".join(regex_parts) + "$"

    return regex_string, wildcard_names


def get_literal_prefix(pattern: str) -> str:
    """
    Extract the literal prefix of a pattern up to the first placeholder.
    """
    index = pattern.find("{")

    if index == -1:
        return pattern
    else:
        return pattern[:index]


def expand_dependency_wildcards(
    dependency_entry: dict, known_wildcards: dict
) -> List[str]:
    """
    Expand a dependency that requires wildcard expansion.

    The dependency_entry should have keys:
      - "pattern": a dependency pattern (which may include missing wildcards)
      - "wildcard_expansion": a pattern (typically an S3 URL) used to discover
         the missing wildcard values. This pattern can specify constraints
         (e.g. {contract_code:[A-Z][0-9][0-9]}).

    Returns a list of fully substituted dependency file paths.
    """
    try:
        dep_pattern = dependency_entry["pattern"]
        expansion_pattern = dependency_entry["wildcard_expansion"]
    except KeyError as e:
        print(f"Error: Missing key in dependency expansion rule: {e}")
        return []

    # Substitute known wildcards into the expansion_pattern.
    try:
        substituted_expansion = partial_substitute_pattern(
            expansion_pattern, known_wildcards
        )
    except ValueError as e:
        print(f"Error in substitution for expansion pattern: {e}")
        return []

    # Parse the substituted URL
    parsed_url = urlparse(substituted_expansion)

    if parsed_url.scheme != "s3":
        print("Warning: Wildcard expansion currently only implemented for S3 schemes.")
        return []

    # Remove leading slash from path (if any) to form an S3 key
    s3_path_pattern = parsed_url.path.lstrip("/")

    # Build a regex from the s3_path_pattern with constraints
    regex_str, missing_wildcards = pattern_to_regex_custom(s3_path_pattern)

    # Use the literal prefix (up to the first placeholder) to list S3 objects
    prefix = get_literal_prefix(s3_path_pattern)
    object_keys = list_s3_objects(prefix)

    matching_deps = []
    pattern_re = re.compile(regex_str)

    for key in object_keys:
        m = pattern_re.match(key)
        if m:
            captured_values = m.groups()
            # Merge known wildcards with the newly extracted ones from the expansion.
            new_wildcards = dict(known_wildcards)

            for name, value in zip(missing_wildcards, captured_values):
                new_wildcards[name] = value
            try:
                dep_file = substitute_pattern(dep_pattern, new_wildcards)
                matching_deps.append(dep_file)
            except ValueError as e:
                print(f"Error substituting in dependency pattern: {e}")
                continue

    if not matching_deps:
        print(
            "Warning: No matching files found for wildcard expansion with pattern "
            f"{substituted_expansion}"
        )

    return matching_deps


def pattern_to_regex(pattern_path: str) -> Tuple[str, List[str]]:
    """
    Convert a pattern path to a regex and extract wildcard names.
    """
    # Split the pattern_path into literal parts and {wildcard} parts
    split_list = re.split(r"(\{\w+\})", pattern_path)
    regex_parts = []
    wildcard_names = []

    # Process each part from the split
    for part in split_list:
        if re.match(r"\{\w+\}", part):
            name = part[1:-1]
            wildcard_names.append(name)
            regex_parts.append("(.+?)")
        else:
            regex_parts.append(re.escape(part))

    regex_string = "".join(regex_parts)

    return "^" + regex_string + "$", wildcard_names


def match_pattern(pattern: str, file: str) -> Optional[Dict[str, str]]:
    """Match a file against a pattern and extract wildcard values."""
    pattern_parts = urlparse(pattern)
    file_parts = urlparse(file)

    if pattern_parts.scheme != file_parts.scheme or pattern_parts.netloc != "":
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


def partial_substitute_pattern(pattern: str, wildcards: Dict[str, str]) -> str:
    """
    Partially substitute wildcard values into a pattern.
    If a wildcard is not provided, leave it unchanged.
    """

    def replacer(match):
        # Get the value for the wildcard; if not available, use the original placeholder.
        value = wildcards.get(match.group(1))
        if value is None:
            return match.group(0)
        return value

    return re.sub(r"\{(\w+)\}", replacer, pattern)


def _where_matches(where_clause, wildcards: dict) -> bool:
    """Return True if provided wildcards satisfy the where clause."""
    if not where_clause:
        return True

    conditions: dict = {}
    if isinstance(where_clause, dict):
        conditions = where_clause
    elif isinstance(where_clause, list):
        for item in where_clause:
            if isinstance(item, dict):
                conditions.update(item)

    for key, allowed in conditions.items():
        value = wildcards.get(key)
        if value is None:
            return False
        if isinstance(allowed, list):
            if value not in allowed:
                return False
        else:
            if value != allowed:
                return False
    return True


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
        scheme = Scheme(parts.scheme)

        if scheme == Scheme.FILE:
            path = parts.path
            if not os.path.isabs(path):
                path = os.path.join(settings.DATA_FOLDER, path)
            result = os.path.exists(path)
        elif scheme == Scheme.S3:
            result = check_s3_file_exists(parts.path)
        elif scheme == Scheme.GITHUB:
            result = check_github_file_exists(file)
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
            if not os.path.isabs(path):
                path = os.path.join(settings.DATA_FOLDER, path)
            if not os.path.exists(path):
                result = None
            else:
                result = datetime.datetime.fromtimestamp(
                    os.path.getmtime(path), tz=datetime.timezone.utc
                )
        elif scheme == Scheme.S3:
            result = get_s3_last_modified(parts.path)
        elif scheme == Scheme.GITHUB:
            result = get_github_last_modified(file)
        else:
            raise ValueError(f"Unsupported scheme: {parts.scheme}")

        self.mtime_cache[file] = result
        return result

    def remove_from_cache(self, file: str) -> None:
        """Remove a file from the cache."""
        if file in self.exists_cache:
            del self.exists_cache[file]
        if file in self.mtime_cache:
            del self.mtime_cache[file]

    def remove(self, file: str, scheme_filter: Scheme) -> None:
        """Remove a file from the filesystem or S3."""
        parts = urlparse(file)
        try:
            scheme = Scheme(parts.scheme)
        except ValueError:
            raise ValueError(f"Unsupported scheme: {parts.scheme}")

        if scheme == Scheme.FILE and scheme_filter == Scheme.FILE:
            path = parts.path
            if not os.path.isabs(path):
                path = os.path.join(settings.DATA_FOLDER, path)
            if os.path.exists(path):
                os.remove(path)
        elif scheme == Scheme.S3 and scheme_filter == Scheme.S3:
            raise NotImplementedError("S3 deletion not implemented")
        else:
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


class RuleType(Enum):
    DEPENDENCY = "dependency"
    REFRESH = "refresh"


@singleton
class FileManager:

    def __init__(self, config_file: str = "dependencies.yml"):
        """Initialize with a list of dependency rules."""
        self.dependency_graph = nx.DiGraph()
        self.refresh_graph = nx.DiGraph()
        try:
            package_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(package_dir, config_file)
            with open(file_path, "r") as f:
                config = yaml.safe_load(f)
                if (
                    not isinstance(config, dict)
                    or "dependency_rules" not in config
                    or "refresh_rules" not in config
                ):
                    raise ValueError(f"Invalid config format in {config_file}")
                self.dependency_rules = config["dependency_rules"]
                self.refresh_rules = config["refresh_rules"]
        except (yaml.YAMLError, IOError) as e:
            raise ValueError(f"Error loading config from {config_file}: {e}")

    def get_graph(self, rule_type: RuleType) -> nx.DiGraph:
        """Get the dependency graph or refresh graph based on the rule type."""
        if rule_type == RuleType.DEPENDENCY:
            return self.dependency_graph
        elif rule_type == RuleType.REFRESH:
            return self.refresh_graph
        else:
            raise ValueError(f"Invalid rule type: {rule_type}")

    def get_rules(self, rule_type: RuleType) -> List[dict]:
        """Get the rules based on the rule type."""
        if rule_type == RuleType.DEPENDENCY:
            return self.dependency_rules
        elif rule_type == RuleType.REFRESH:
            return self.refresh_rules
        else:
            raise ValueError(f"Invalid rule type: {rule_type}")

    def add_dependencies(
        self,
        parent_node: str,
        dependencies: List[str | dict],
        wildcards,
        rule_type: RuleType,
    ) -> None:
        graph = self.get_graph(rule_type)

        for dep in dependencies:
            if isinstance(dep, str):
                try:
                    dep_file = substitute_pattern(dep, wildcards)
                    self.build_subgraph(dep_file, rule_type)
                    graph.add_edge(parent_node, dep_file)
                except ValueError as e:
                    print(
                        f"Warning: {e} when processing dependency {dep} for {parent_node}"
                    )
            elif isinstance(dep, dict):
                # Process dependency with wildcard expansion.
                expanded_files = expand_dependency_wildcards(dep, wildcards)
                if not expanded_files:
                    print(
                        "Warning: Skipping dependency "
                        f"{dep.get('pattern', '')} for {parent_node} as no "
                        "expansion matches were found."
                    )
                for dep_file in expanded_files:
                    self.build_subgraph(dep_file, rule_type)
                    graph.add_edge(parent_node, dep_file)
            else:
                print(
                    f"Warning: Unexpected dependency type {dep} for file {parent_node}"
                )

    def build_subgraph(self, file: str, rule_type: RuleType) -> None:
        """Recursively build the graph for a file and its dependencies."""
        graph = self.get_graph(rule_type)

        if file in graph:
            return

        graph.add_node(file)
        rules = self.get_rules(rule_type)

        for rule in rules:
            wildcards = match_pattern(rule["dependent"], file)

            if wildcards and _where_matches(rule.get("where"), wildcards):
                graph.add_node(file, wildcards=wildcards)

                if rule.get("refresh_function"):
                    graph.nodes[file]["refresh_function"] = rule["refresh_function"]

                if rule.get("depends_on"):
                    self.add_dependencies(
                        file,
                        rule["depends_on"],
                        wildcards,
                        rule_type,
                    )

                break

    def is_up_to_date(self, file: str) -> bool:
        """Check if a file is up-to-date, building its subgraph if needed."""
        self.build_subgraph(file, RuleType.DEPENDENCY)

        if file not in self.dependency_graph:
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
                for dep in self.dependency_graph.successors(file):
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
        self.build_subgraph(file, RuleType.DEPENDENCY)

        if file not in self.dependency_graph:
            raise ValueError(f"File {file} not found in dependency graph")

        fop = FileOperations()
        temp_files = []
        try:
            self._refresh_file(file, reset, {}, fop, temp_files)
        finally:
            # Remove temporary files
            for temp_file in temp_files:
                if temp_file != file:
                    fop.remove(temp_file, Scheme.FILE)

    def _refresh_file(
        self,
        file: str,
        reset: bool,
        cache: Dict[str, bool],
        fop: FileOperations,
        temp_files: List[str],
    ) -> None:
        """Recursively refresh dependencies, then the file if needed."""
        has_successors = False

        # First refresh all dependencies
        for dep in self.dependency_graph.successors(file):
            self._refresh_file(dep, reset, cache, fop, temp_files)
            has_successors = True

        # Then check if this file needs refreshing
        if has_successors and (reset or not self._is_up_to_date(file, cache, fop)):
            self.build_subgraph(file, RuleType.REFRESH)
            self._refresh_stale_file(file, reset, cache, fop, temp_files)

    def _select_refresh_node(
        self,
        file: str,
        reset: bool,
        cache: Dict[str, bool],
        fop: FileOperations,
        temp_files: List[str],
    ) -> tuple[str | dict | None, RuleType, dict]:
        """Return the refresh rule node and its type for the given file."""
        node_data = self.dependency_graph.nodes[file]
        refresh_func_node = node_data.get("refresh_function")

        if refresh_func_node and all(
            self._is_up_to_date(dep, cache, fop)
            for dep in self.dependency_graph.successors(file)
        ):
            return refresh_func_node, RuleType.DEPENDENCY, node_data

        for dep in self.refresh_graph.successors(file):
            self.build_subgraph(dep, RuleType.DEPENDENCY)
            if reset or not self._is_up_to_date(dep, cache, fop):
                self._refresh_stale_file(dep, reset, cache, fop, temp_files)

        node_data = self.refresh_graph.nodes[file]
        return node_data.get("refresh_function"), RuleType.REFRESH, node_data

    @staticmethod
    def _parse_refresh_rule(func_node: str | dict, file: str) -> tuple[str, dict, dict]:
        """Extract function information from a refresh rule."""
        func_str: str | None = None
        additional_arguments: dict = {}
        list_args_spec: dict = {}

        if isinstance(func_node, str):
            func_str = func_node
        elif isinstance(func_node, dict):
            func_str = func_node["name"]
            additional_arguments = dict(func_node.get("additional_args", {}))
            list_args_spec = func_node.get("list_args", {})
        else:
            raise ValueError(f"Invalid process function for file {file}")

        if not func_str:
            raise ValueError(f"No process function defined for file {file}")

        return func_str, additional_arguments, list_args_spec

    def _build_list_args(
        self, file: str, rule_type: RuleType, list_args_spec: dict
    ) -> dict:
        """Construct list-type arguments based on rule specifications."""
        additional_arguments: dict[str, list[str]] = {}
        if not list_args_spec:
            return additional_arguments

        for arg_name, wildcard_key in list_args_spec.items():
            values: list[str] = []
            graph = self.get_graph(rule_type)
            for dep in graph.successors(file):
                dep_wc = graph.nodes[dep].get("wildcards", {})
                val = dep_wc.get(wildcard_key)
                if val is not None and val not in values:
                    values.append(val)

            if not values:
                raise RuntimeError(
                    f"Could not build list argument '{arg_name}' for '{file}'; "
                    f"none of the direct dependencies expose wildcard '{wildcard_key}'."
                )

            additional_arguments[arg_name] = values

        return additional_arguments

    @staticmethod
    def _ensure_temp_file(file: str, temp_files: list[str]) -> None:
        """Add file to temporary list if it needs to be created."""
        parts = urlparse(file)
        scheme = Scheme(parts.scheme)
        if (
            scheme == Scheme.FILE
            and not os.path.exists(
                os.path.join(settings.DATA_FOLDER, parts.path)
                if not os.path.isabs(parts.path)
                else parts.path
            )
            and file not in temp_files
        ):
            temp_files.append(file)

    def _refresh_stale_file(
        self,
        file: str,
        reset: bool,
        cache: Dict[str, bool],
        fop: FileOperations,
        temp_files: List[str],
    ) -> None:
        func_node, rule_type, node_data = self._select_refresh_node(
            file, reset, cache, fop, temp_files
        )

        if not func_node:
            return

        func_str, additional_arguments, list_args_spec = self._parse_refresh_rule(
            func_node, file
        )
        additional_arguments.update(
            self._build_list_args(file, rule_type, list_args_spec)
        )

        self._ensure_temp_file(file, temp_files)

        process_func = import_function(func_str)
        wildcards = node_data.get("wildcards", {})
        process_func(**wildcards, **additional_arguments)

        cache[file] = True
        fop.remove_from_cache(file)

    def get_dependencies(self, file: str) -> List[str]:
        """Get all dependencies for a file."""
        self.build_subgraph(file, RuleType.DEPENDENCY)
        return list(self.dependency_graph.successors(file))

    def get_node_params(self, rule_type: RuleType, file: str) -> dict:
        """Get parameters for a node in the graph."""
        graph = self.get_graph(rule_type)
        self.build_subgraph(file, rule_type)

        if file not in graph:
            raise ValueError(f"File {file} not found in {rule_type.value} graph")

        node = graph.nodes[file]
        params = node.get("wildcards", {})
        refresh_func = node.get("refresh_function")

        if refresh_func and isinstance(refresh_func, dict):
            additional_args = refresh_func.get("additional_args", {})
            params.update(additional_args)

        return params

    def dependencies_max_last_modified(
        self,
        file: str,
        rule_type: RuleType = RuleType.DEPENDENCY,
        scheme_filter: Optional[Scheme] = None,
    ) -> datetime.datetime | None:
        """Get the maximum last modified time of all dependencies."""
        self.build_subgraph(file, rule_type)
        graph = self.get_graph(rule_type)

        if file not in graph:
            raise ValueError(f"File {file} not found in {rule_type.value} graph")

        max_mtime = None
        fop = FileOperations()

        for dep in graph.successors(file):
            if scheme_filter:
                parts = urlparse(dep)
                scheme = Scheme(parts.scheme)

                if scheme != scheme_filter:
                    continue

            mtime = fop.get_mtime(dep)
            if mtime is not None:
                if max_mtime is None or mtime > max_mtime:
                    max_mtime = mtime

        return max_mtime

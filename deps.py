#!/usr/bin/env python3
"""
Conan Dependency Graph Solver

A Python library to solve dependency conflicts in Conan packages by finding
optimal compatible versions through remote repository analysis.
"""

import re
import json
import subprocess
import logging
from typing import Dict, List, Set, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import ast
import os


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConanRef(NamedTuple):
    """Represents a Conan package reference"""
    name: str
    version: str
    user: str
    channel: str
    
    @classmethod
    def parse(cls, ref_str: str) -> 'ConanRef':
        """Parse a conan reference string with various formats"""
        ref_str = ref_str.strip()
        
        # More comprehensive version pattern that handles all your cases
        # Allows: digits, letters, dots, dashes, underscores, plus signs
        version_pattern = r'[\w\.\-\+]+'
        # Package name can contain underscores and dashes
        package_pattern = r'[\w\-\_]+'
        # Channel pattern - more permissive for beta versions with dots
        channel_pattern = r'[\w\.\-\+]*'
        # User pattern
        user_pattern = r'[\w\.\-\+\_]+'
        
        patterns = [
            # Standard format: package/version@user/channel
            rf'^({package_pattern})/({version_pattern})@({user_pattern})/({channel_pattern})$',
            # Missing channel with trailing slash: package/version@user/
            rf'^({package_pattern})/({version_pattern})@({user_pattern})/$',
            # Missing channel entirely: package/version@user
            rf'^({package_pattern})/({version_pattern})@({user_pattern})$',
            # Just package/version (no user/channel)
            rf'^({package_pattern})/({version_pattern})$'
        ]
        
        for i, pattern in enumerate(patterns):
            match = re.match(pattern, ref_str)
            if match:
                groups = match.groups()
                
                if i == 0:  # Standard format
                    return cls(groups[0], groups[1], groups[2], groups[3])
                elif i == 1:  # Missing channel with trailing slash
                    return cls(groups[0], groups[1], groups[2], "")
                elif i == 2:  # Missing channel entirely
                    return cls(groups[0], groups[1], groups[2], "")
                elif i == 3:  # Just package/version
                    return cls(groups[0], groups[1], "", "")
        
        raise ValueError(f"Invalid conan reference format: {ref_str}")
    
    def __str__(self) -> str:
        if self.user and self.channel:
            return f"{self.name}/{self.version}@{self.user}/{self.channel}"
        elif self.user:
            return f"{self.name}/{self.version}@{self.user}"
        else:
            return f"{self.name}/{self.version}"
    
    def matches_pattern(self, pattern: str) -> bool:
        """Check if this reference matches a search pattern like 'lib_b*'"""
        return self.name.startswith(pattern.rstrip('*'))


@dataclass
class Requirement:
    """Represents a single requirement with its options"""
    ref: ConanRef
    force: bool = False
    override: bool = False
    visible: bool = True
    
    def __str__(self) -> str:
        return str(self.ref)


@dataclass
class PackageInfo:
    """Information about a Conan package"""
    ref: ConanRef
    requirements: List[Requirement] = field(default_factory=list)
    settings: Dict[str, str] = field(default_factory=dict)
    options: Dict[str, str] = field(default_factory=dict)
    available_versions: List[str] = field(default_factory=list)
    
    @property
    def dependencies(self) -> List[ConanRef]:
        """Backward compatibility - return just the ConanRef objects"""
        return [req.ref for req in self.requirements]


class ConanfileParser:
    """Parser for conanfile.py files"""
    
    @staticmethod
    def parse_conanfile(filepath: str) -> PackageInfo:
        """Parse a conanfile.py and extract dependency information"""
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Parse the AST to extract information
        tree = ast.parse(content)
        
        package_info = PackageInfo(ref=ConanRef("", "", "", ""))
        requirements = []
        
        class ConanfileVisitor(ast.NodeVisitor):
            def visit_Assign(self, node):
                # Handle old-style requires = [...] format
                if isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'requires':
                    if isinstance(node.value, (ast.Str, ast.Constant)):
                        # Single requirement
                        req_str = node.value.s if hasattr(node.value, 's') else node.value.value
                        try:
                            req = Requirement(ref=ConanRef.parse(req_str))
                            requirements.append(req)
                        except ValueError as e:
                            logger.warning(f"Failed to parse requirement '{req_str}': {e}")
                    elif isinstance(node.value, (ast.List, ast.Tuple)):
                        # Multiple requirements
                        for elt in node.value.elts:
                            if isinstance(elt, (ast.Str, ast.Constant)):
                                req_str = elt.s if hasattr(elt, 's') else elt.value
                                try:
                                    req = Requirement(ref=ConanRef.parse(req_str))
                                    requirements.append(req)
                                except ValueError as e:
                                    logger.warning(f"Failed to parse requirement '{req_str}': {e}")
                
                # Extract package name and version if available
                if isinstance(node.targets[0], ast.Name):
                    if node.targets[0].id == 'name' and isinstance(node.value, (ast.Str, ast.Constant)):
                        package_info.ref = package_info.ref._replace(
                            name=node.value.s if hasattr(node.value, 's') else node.value.value
                        )
                    elif node.targets[0].id == 'version' and isinstance(node.value, (ast.Str, ast.Constant)):
                        package_info.ref = package_info.ref._replace(
                            version=node.value.s if hasattr(node.value, 's') else node.value.value
                        )
            
            def visit_FunctionDef(self, node):
                # Look for requirements() method
                if node.name == 'requirements':
                    for stmt in node.body:
                        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                            call = stmt.value
                            # Check for self.requires() calls
                            if (isinstance(call.func, ast.Attribute) and 
                                isinstance(call.func.value, ast.Name) and 
                                call.func.value.id == 'self' and 
                                call.func.attr == 'requires'):
                                
                                req = ConanfileParser._parse_requires_call(call)
                                if req:
                                    requirements.append(req)
                
                # Continue visiting nested nodes
                self.generic_visit(node)
        
        visitor = ConanfileVisitor()
        visitor.visit(tree)
        
        package_info.requirements = requirements
        return package_info
    
    @staticmethod
    def _parse_requires_call(call_node: ast.Call) -> Optional[Requirement]:
        """Parse a self.requires() call and extract requirement information"""
        if not call_node.args:
            return None
        
        # Get the package reference (first argument)
        first_arg = call_node.args[0]
        if isinstance(first_arg, (ast.Str, ast.Constant)):
            ref_str = first_arg.s if hasattr(first_arg, 's') else first_arg.value
        else:
            return None
        
        try:
            conan_ref = ConanRef.parse(ref_str)
        except ValueError:
            logger.warning(f"Failed to parse conan reference: {ref_str}")
            return None
        
        # Parse keyword arguments
        force = False
        override = False
        visible = True
        
        for keyword in call_node.keywords:
            if keyword.arg == 'force':
                if isinstance(keyword.value, ast.Constant):
                    force = bool(keyword.value.value)
                elif isinstance(keyword.value, ast.NameConstant):  # Python < 3.8
                    force = bool(keyword.value.value)
            elif keyword.arg == 'override':
                if isinstance(keyword.value, ast.Constant):
                    override = bool(keyword.value.value)
                elif isinstance(keyword.value, ast.NameConstant):
                    override = bool(keyword.value.value)
            elif keyword.arg == 'visible':
                if isinstance(keyword.value, ast.Constant):
                    visible = bool(keyword.value.value)
                elif isinstance(keyword.value, ast.NameConstant):
                    visible = bool(keyword.value.value)
        
        return Requirement(
            ref=conan_ref,
            force=force,
            override=override,
            visible=visible
        )


class ConanRemoteClient:
    """Client for interacting with Conan remotes"""
    
    def __init__(self, remotes: List[str] = None):
        """
        Initialize with multiple remotes
        Args:
            remotes: List of remote names, defaults to ["conancenter"]
        """
        self.remotes = remotes or ["conancenter"]
        logger.info(f"Initialized with remotes: {self.remotes}")
    
    def search_packages(self, pattern: str) -> List[ConanRef]:
        """Search for packages matching a pattern across all remotes"""
        all_packages = []
        
        for remote in self.remotes:
            try:
                cmd = ["conan", "search", pattern, "-r", remote]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                
                packages = []
                for line in result.stdout.split('\n'):
                    line = line.strip()
                    if line and '@' in line and '/' in line:
                        try:
                            packages.append(ConanRef.parse(line))
                        except ValueError as e:
                            logger.warning(f"Failed to parse reference '{line}': {e}")
                            continue
                
                all_packages.extend(packages)
                logger.info(f"Found {len(packages)} packages in remote '{remote}'")
                
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to search packages in remote '{remote}': {e}")
                continue
        
        # Remove duplicates while preserving order
        seen = set()
        unique_packages = []
        for pkg in all_packages:
            pkg_key = str(pkg)
            if pkg_key not in seen:
                seen.add(pkg_key)
                unique_packages.append(pkg)
        
        return unique_packages
    
# def get_package_info(self, ref: ConanRef) -> Optional[PackageInfo]:
#     """Get detailed information about a package using conan graph explain"""
#     for remote in self.remotes:
#         try:
#             # Use conan graph explain to get package dependencies
#             cmd = ["conan", "graph", "explain", f"--requires={ref}", "-r", remote, "--format=json"]
#             result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
#             # Parse JSON output
#             import json
#             data = json.loads(result.stdout)
            
#             requirements = []
#             # The structure depends on the actual JSON output from conan graph explain
#             # You might need to adjust this based on actual output
#             if "graph" in data:
#                 for node_id, node in data["graph"].get("nodes", {}).items():
#                     node_ref = node.get("ref", "")
#                     if node_ref and node_ref != str(ref):  # Don't include self
#                         try:
#                             dep_ref = ConanRef.parse(node_ref)
#                             requirements.append(Requirement(ref=dep_ref))
#                         except ValueError as e:
#                             logger.warning(f"Failed to parse dependency '{node_ref}': {e}")
            
#             logger.info(f"Successfully got package info for {ref} from remote '{remote}'")
#             return PackageInfo(ref=ref, requirements=requirements)
            
#         except subprocess.CalledProcessError as e:
#             logger.warning(f"Failed to get package info for {ref} from remote '{remote}': {e}")
#             continue
    
#     logger.error(f"Failed to get package info for {ref} from any remote")
#     return None
    def get_package_info(self, ref: ConanRef) -> Optional[PackageInfo]:
    """Get detailed information about a package using conan graph explain"""
    import shutil
    import json
    
    # Get the full path to conan
    conan_path = shutil.which("conan")
    if not conan_path:
        logger.error("Conan executable not found in PATH")
        return None
    
    for remote in self.remotes:
        try:
            cmd = [conan_path, "graph", "explain", f"--requires={ref}", "-r", remote, "--format=json"]
            
            # Get current environment
            env = os.environ.copy()
            
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True,
                env=env,
                timeout=60
            )
            
            # Check if we have valid output before parsing JSON
            if not result.stdout or result.stdout.strip() == "":
                logger.warning(f"Empty output from conan command for {ref} on remote '{remote}'")
                continue
            
            try:
                # Parse JSON output
                data = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON output for {ref} on remote '{remote}': {e}")
                logger.warning(f"Raw output: {result.stdout[:200]}...")
                continue
            
            # Check if data is valid
            if not data or not isinstance(data, dict):
                logger.warning(f"Invalid JSON data structure for {ref} on remote '{remote}'")
                continue
            
            requirements = []
            if "graph" in data and "nodes" in data["graph"]:
                for node_id, node in data["graph"]["nodes"].items():
                    node_ref = node.get("ref", "")
                    if node_ref and node_ref != str(ref) and "conanfile" not in node_ref.lower():
                        try:
                            dep_ref = ConanRef.parse(node_ref)
                            requirements.append(Requirement(ref=dep_ref))
                        except ValueError as e:
                            logger.warning(f"Failed to parse dependency '{node_ref}': {e}")
            
            logger.info(f"Successfully got package info for {ref} from remote '{remote}'")
            return PackageInfo(ref=ref, requirements=requirements)
            
        except subprocess.CalledProcessError as e:
            logger.warning(f"Command failed for {ref} on remote '{remote}'")
            logger.warning(f"Return code: {e.returncode}")
            logger.warning(f"stdout: {e.stdout}")
            logger.warning(f"stderr: {e.stderr}")
            continue
        except subprocess.TimeoutExpired:
            logger.warning(f"Command timed out for {ref} on remote '{remote}'")
            continue
        except Exception as e:
            logger.warning(f"Unexpected error for {ref} on remote '{remote}': {e}")
            continue
    
    logger.error(f"Failed to get package info for {ref} from any remote")
    return None



    
    def get_available_versions(self, package_name: str) -> List[str]:
        """Get all available versions for a package across all remotes"""
        packages = self.search_packages(f"{package_name}*")
        versions = []
        for pkg in packages:
            if pkg.name == package_name:
                versions.append(pkg.version)
        return sorted(set(versions), key=lambda v: self._version_key(v))
    
    def test_remote_connectivity(self) -> Dict[str, bool]:
        """Test connectivity to all configured remotes"""
        results = {}
        for remote in self.remotes:
            try:
                cmd = ["conan", "remote", "list"]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                # Check if remote exists in the output
                results[remote] = remote in result.stdout
            except subprocess.CalledProcessError:
                results[remote] = False
        
        return results
    
    @staticmethod
    def _version_key(version: str) -> Tuple:
        """Convert version string to tuple for sorting"""
        parts = []
        for part in version.split('.'):
            # Try to convert to int, fallback to string for sorting
            try:
                parts.append(int(part))
            except ValueError:
                # Handle non-numeric parts like 'beta4', 'rc1', etc.
                parts.append(part)
        return tuple(parts)


class DependencyGraph:
    """Represents a dependency graph"""
    
    def __init__(self):
        self.nodes: Dict[str, PackageInfo] = {}
        self.edges: Dict[str, Set[str]] = defaultdict(set)
        self.conflicts: List[Tuple[ConanRef, ConanRef, str]] = []
    
    def add_package(self, package_info: PackageInfo):
        """Add a package to the graph"""
        key = f"{package_info.ref.name}"
        self.nodes[key] = package_info
        
        for dep in package_info.dependencies:
            dep_key = f"{dep.name}"
            self.edges[key].add(dep_key)
    
    def detect_conflicts(self) -> List[Tuple[str, List[ConanRef]]]:
        """Detect version conflicts in the dependency graph"""
        conflicts = []
        package_versions = defaultdict(list)
        
        # Collect all required versions for each package
        for package_info in self.nodes.values():
            for dep in package_info.dependencies:
                package_versions[dep.name].append((dep, package_info.ref))
        
        # Check for version conflicts
        for package_name, requirements in package_versions.items():
            versions = set(req[0].version for req in requirements)
            if len(versions) > 1:
                conflicts.append((package_name, [req[0] for req in requirements]))
        
        return conflicts
    
    def topological_sort(self) -> List[str]:
        """Perform topological sort of the dependency graph"""
        in_degree = defaultdict(int)
        
        # Calculate in-degrees
        for node in self.nodes:
            in_degree[node] = 0
        
        for node, deps in self.edges.items():
            for dep in deps:
                in_degree[dep] += 1
        
        # Kahn's algorithm
        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for neighbor in self.edges[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(result) != len(self.nodes):
            raise ValueError("Circular dependency detected")
        
        return result


class DependencySolver:
    """Main solver for dependency conflicts"""
    
    def __init__(self, remote_client: ConanRemoteClient):
        self.remote_client = remote_client
        self.graph = DependencyGraph()
    
    def solve_dependencies(self, main_conanfile: str) -> Dict[str, ConanRef]:
        """
        Solve dependency conflicts by finding optimal compatible versions
        
        Args:
            main_conanfile: Path to the main conanfile.py
            
        Returns:
            Dictionary mapping package names to their resolved versions
        """
        logger.info("Starting dependency resolution...")
        
        # Parse main conanfile
        main_package = ConanfileParser.parse_conanfile(main_conanfile)
        self.graph.add_package(main_package)
        
        # Recursively resolve dependencies
        visited = set()
        self._resolve_recursive(main_package.dependencies, visited)
        
        # Detect conflicts
        conflicts = self.graph.detect_conflicts()
        if not conflicts:
            logger.info("No conflicts detected!")
            return self._build_solution()
        
        logger.info(f"Found {len(conflicts)} conflicts, attempting resolution...")
        
        # Attempt to resolve conflicts
        resolved_versions = self._resolve_conflicts(conflicts)
        
        return resolved_versions
    
    def _resolve_recursive(self, dependencies: List[ConanRef], visited: Set[str]):
        """Recursively resolve dependencies"""
        for dep in dependencies:
            dep_key = str(dep)
            if dep_key in visited:
                continue
            
            visited.add(dep_key)
            
            # Get package info from remote
            package_info = self.remote_client.get_package_info(dep)
            if package_info:
                self.graph.add_package(package_info)
                self._resolve_recursive(package_info.dependencies, visited)
    
    def _resolve_conflicts(self, conflicts: List[Tuple[str, List[ConanRef]]]) -> Dict[str, ConanRef]:
        """Resolve version conflicts by finding compatible versions"""
        resolved = {}
        
        # First, check which packages have force=True requirements
        forced_versions = self._get_forced_versions()
        
        for package_name, conflicting_refs in conflicts:
            logger.info(f"Resolving conflict for {package_name}")
            
            # Check if any requirement is forced
            if package_name in forced_versions:
                forced_ref = forced_versions[package_name]
                logger.info(f"Using forced version for {package_name}: {forced_ref}")
                resolved[package_name] = forced_ref
                continue
            
            # Get all available versions
            available_versions = self.remote_client.get_available_versions(package_name)
            
            # Find a version that satisfies all requirements
            compatible_version = self._find_compatible_version(
                package_name, conflicting_refs, available_versions
            )
            
            if compatible_version:
                # Create the resolved reference (using first ref's user/channel)
                first_ref = conflicting_refs[0]
                resolved_ref = ConanRef(
                    package_name, compatible_version, 
                    first_ref.user, first_ref.channel
                )
                resolved[package_name] = resolved_ref
                logger.info(f"Resolved {package_name} to version {compatible_version}")
            else:
                logger.warning(f"Direct resolution failed for {package_name}, trying backtracking...")
                # Try backtracking: modify parent packages to find compatible versions
                backtrack_result = self._backtrack_resolve_conflict(package_name, conflicting_refs)
                
                if backtrack_result:
                    resolved.update(backtrack_result)
                    logger.info(f"Backtracking successful for {package_name}")
                else:
                    logger.error(f"Could not resolve conflict for {package_name} even with backtracking")
                    # Use the highest version as fallback
                    if available_versions:
                        fallback_version = available_versions[-1]
                        first_ref = conflicting_refs[0]
                        resolved_ref = ConanRef(
                            package_name, fallback_version,
                            first_ref.user, first_ref.channel
                        )
                        resolved[package_name] = resolved_ref
        
        return resolved
    
    def _get_forced_versions(self) -> Dict[str, ConanRef]:
        """Get all packages that have force=True requirements"""
        forced = {}
        
        for package_info in self.graph.nodes.values():
            for req in package_info.requirements:
                if req.force:
                    forced[req.ref.name] = req.ref
                    logger.info(f"Found forced requirement: {req.ref} (force=True)")
        
        return forced
    
    def _backtrack_resolve_conflict(self, conflicting_package: str, 
                                  conflicting_refs: List[ConanRef]) -> Optional[Dict[str, ConanRef]]:
        """
        Advanced backtracking resolution: Try different versions of parent packages
        to find a combination where they all require the same version of the conflicting package.
        
        For example, if lib_a/1.0 needs lib_b/3.0 and lib_c/2.1 needs lib_b/3.1,
        try different versions of lib_a and lib_c to see if any combination agrees on lib_b.
        """
        logger.info(f"Starting backtracking for {conflicting_package}")
        
        # Find which packages depend on the conflicting package
        parent_packages = self._find_parent_packages(conflicting_package, conflicting_refs)
        
        if not parent_packages:
            return None
        
        # Get all available versions for each parent package
        parent_versions = {}
        forced_parents = self._get_forced_versions()
        
        for parent_name in parent_packages:
            if parent_name in forced_parents:
                # Skip forced packages - they can't be changed
                logger.info(f"Skipping forced parent package: {parent_name}")
                continue
            
            versions = self.remote_client.get_available_versions(parent_name)
            if versions:
                parent_versions[parent_name] = versions
                logger.info(f"Found {len(versions)} versions for {parent_name}: {versions}")
        
        if not parent_versions:
            logger.info("No changeable parent packages found")
            return None
        
        # Try different combinations of parent versions
        return self._try_version_combinations(conflicting_package, parent_versions)
    
    def _find_parent_packages(self, target_package: str, conflicting_refs: List[ConanRef]) -> Set[str]:
        """Find which packages depend on the target package"""
        parents = set()
        
        for package_info in self.graph.nodes.values():
            for req in package_info.requirements:
                if req.ref.name == target_package:
                    parents.add(package_info.ref.name)
        
        return parents
    
    def _try_version_combinations(self, conflicting_package: str, 
                                parent_versions: Dict[str, List[str]]) -> Optional[Dict[str, ConanRef]]:
        """
        Try different combinations of parent package versions to find compatibility
        """
        import itertools
        
        parent_names = list(parent_versions.keys())
        version_lists = [parent_versions[name] for name in parent_names]
        
        # Limit combinations to avoid exponential explosion
        max_combinations = 50
        combination_count = 0
        
        logger.info(f"Trying version combinations for parents: {parent_names}")
        
        # Try combinations starting with the most recent versions
        for combination in itertools.product(*version_lists):
            combination_count += 1
            if combination_count > max_combinations:
                logger.warning(f"Reached maximum combinations limit ({max_combinations})")
                break
            
            logger.info(f"Trying combination {combination_count}: {dict(zip(parent_names, combination))}")
            
            # Test this combination
            compatible_version = self._test_version_combination(
                conflicting_package, parent_names, combination
            )
            
            if compatible_version:
                # Build the solution
                solution = {}
                
                # Add parent package versions
                for i, parent_name in enumerate(parent_names):
                    version = combination[i]
                    # Use original user/channel from existing reference
                    original_ref = self._find_original_ref(parent_name)
                    if original_ref:
                        solution[parent_name] = ConanRef(
                            parent_name, version, 
                            original_ref.user, original_ref.channel
                        )
                
                # Add the resolved conflicting package
                original_ref = self._find_original_ref(conflicting_package)
                if original_ref:
                    solution[conflicting_package] = ConanRef(
                        conflicting_package, compatible_version,
                        original_ref.user, original_ref.channel
                    )
                
                logger.info(f"Found compatible combination: {solution}")
                return solution
        
        return None
    
    def _test_version_combination(self, conflicting_package: str, 
                                parent_names: List[str], 
                                parent_versions: Tuple[str, ...]) -> Optional[str]:
        """
        Test if a specific combination of parent versions results in 
        compatible requirements for the conflicting package
        """
        required_versions = []
        
        for i, parent_name in enumerate(parent_names):
            parent_version = parent_versions[i]
            
            # Get the original reference to construct the test reference
            original_ref = self._find_original_ref(parent_name)
            if not original_ref:
                continue
            
            test_ref = ConanRef(parent_name, parent_version, original_ref.user, original_ref.channel)
            
            # Get dependencies for this version
            package_info = self.remote_client.get_package_info(test_ref)
            if not package_info:
                continue
            
            # Check what version of the conflicting package this requires
            for dep in package_info.dependencies:
                if dep.name == conflicting_package:
                    required_versions.append(dep.version)
                    break
        
        # Check if all parent packages agree on the same version
        if required_versions and len(set(required_versions)) == 1:
            agreed_version = required_versions[0]
            logger.info(f"All parents agree on {conflicting_package}/{agreed_version}")
            return agreed_version
        
        return None
    
    def _find_original_ref(self, package_name: str) -> Optional[ConanRef]:
        """Find the original reference for a package name"""
        for package_info in self.graph.nodes.values():
            if package_info.ref.name == package_name:
                return package_info.ref
            for req in package_info.requirements:
                if req.ref.name == package_name:
                    return req.ref
        return None
    
    def _find_compatible_version(self, package_name: str, 
                               conflicting_refs: List[ConanRef], 
                               available_versions: List[str]) -> Optional[str]:
        """Find a version that's compatible with all requirements"""
        
        # For now, implement a simple strategy:
        # Try to find the highest version that satisfies the minimum requirements
        
        min_versions = []
        for ref in conflicting_refs:
            min_versions.append(ref.version)
        
        # Find the highest minimum version
        min_versions.sort(key=lambda v: self.remote_client._version_key(v))
        target_min_version = min_versions[-1]
        
        # Find compatible versions (>= target_min_version)
        compatible_versions = [
            v for v in available_versions 
            if self.remote_client._version_key(v) >= self.remote_client._version_key(target_min_version)
        ]
        
        if compatible_versions:
            return compatible_versions[0]  # Return the lowest compatible version
        
        return None
    
    def _build_solution(self) -> Dict[str, ConanRef]:
        """Build the final solution from the resolved graph"""
        solution = {}
        for package_info in self.graph.nodes.values():
            solution[package_info.ref.name] = package_info.ref
        return solution


def main():
    """Example usage of the dependency solver with multiple remotes"""
    
    # Create the solver with multiple remotes
    remotes = ["conancenter", "mycompany", "artifactory", "local"]
    remote_client = ConanRemoteClient(remotes)
    solver = DependencySolver(remote_client)
    
    # Test remote connectivity
    connectivity = remote_client.test_remote_connectivity()
    print("Remote connectivity status:")
    for remote, status in connectivity.items():
        print(f"  {remote}: {'✓' if status else '✗'}")
    
    # Example conanfile path
    conanfile_path = "conanfile.py"
    
    if not os.path.exists(conanfile_path):
        print("Creating example conanfile.py...")
        example_conanfile = '''from conans import ConanFile

class MainPackage(ConanFile):
    name = "main_package"
    version = "1.0"
    
    def requirements(self):
        self.requires("lib_a/1.0@user/stable", force=True)
        self.requires("lib_b/2.0@user/stable")
        self.requires("lib_d/3.0@user/stable", force=True)
        self.requires("lib_f/2.5@user/stable", force=True)
        self.requires("lib_x/1.4@user/stable")
        self.requires("pack_age_this/0.5.4@something/beta4")
        self.requires("pack_age_this/0.5.4-6@something/beta4")
    
    def build(self):
        pass
'''
        with open(conanfile_path, 'w') as f:
            f.write(example_conanfile)
    
    try:
        # Solve dependencies
        solution = solver.solve_dependencies(conanfile_path)
        
        print("\n=== DEPENDENCY RESOLUTION SOLUTION ===")
        for package_name, resolved_ref in solution.items():
            print(f"{package_name}: {resolved_ref}")
        
        # Print forced requirements
        print("\n=== FORCED REQUIREMENTS ===")
        for package_info in solver.graph.nodes.values():
            for req in package_info.requirements:
                if req.force:
                    print(f"{req.ref} (force=True)")
        
        # Print conflicts that were found
        conflicts = solver.graph.detect_conflicts()
        if conflicts:
            print("\n=== CONFLICTS DETECTED ===")
            for package_name, conflicting_refs in conflicts:
                print(f"{package_name}:")
                for ref in conflicting_refs:
                    print(f"  - {ref}")
        
    except Exception as e:
        logger.error(f"Error during dependency resolution: {e}")


if __name__ == "__main__":
    main()

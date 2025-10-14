#!/usr/bin/env python3
"""
scarab-infra helper CLI for quick environment setup.
"""

from __future__ import annotations

import argparse
import getpass
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
except ImportError:  # pragma: no cover - packaging is usually available
    SpecifierSet = None
    Version = None

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_TRACE_URL = "https://drive.google.com/uc?id=1tfKL7wYK1mUqpCH8yPaPVvxk2UIAJrOX"
TRACE_ARCHIVE_NAME = "simpoint_traces.tar.gz"
TRACE_DIR_NAME = "simpoint_traces"
ENV_NAME = "scarabinfra"
MINICONDA_URL = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
OPTIONAL_TITLES = {
    "Optional Slurm installation",
    "Optional ghcr.io login",
}


def load_infra_utilities():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts import utilities as infra_utils  # type: ignore
    except ImportError as exc:  # pragma: no cover - missing optional dependency
        raise StepError(
            f"Unable to import scarab-infra utilities: {exc}. Did you run `./sci --init` and ensure required packages are installed?"
        ) from exc
    return infra_utils


class StepError(RuntimeError):
    """Raised when a bootstrap step cannot be completed."""


def print_heading(title: str) -> None:
    print(f"\n==> {title}")


def info(message: str) -> None:
    if not message:
        return
    for line in message.splitlines():
        print(f"   {line}")


def run_command(
    cmd: Iterable[str],
    *,
    check: bool = True,
    capture: bool = False,
    input_data: Optional[str] = None,
    cwd: Optional[Path] = None,
) -> Optional[str]:
    cmd_list = list(cmd)
    print(f"$ {' '.join(cmd_list)}")
    try:
        completed = subprocess.run(
            cmd_list,
            check=check,
            text=True,
            input=input_data,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT if capture else None,
            cwd=str(cwd) if cwd else None,
        )
    except FileNotFoundError as exc:
        raise StepError(f"Command not found: {cmd_list[0]}") from exc
    except subprocess.CalledProcessError as exc:
        output = exc.stdout if capture else ""
        raise StepError(
            f"{' '.join(cmd_list)} failed with exit code {exc.returncode}\n{output}"
        ) from exc
    if capture:
        return completed.stdout or ""
    return None


def ensure_repo_on_path() -> None:
    repo_str = str(REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def import_repo_module(module: str):
    ensure_repo_on_path()
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise StepError(f"Unable to import module '{module}'") from exc


def read_descriptor(descriptor_name: str):
    infra_utils = load_infra_utilities()
    path = REPO_ROOT / "json" / f"{descriptor_name}.json"
    if not path.is_file():
        raise StepError(f"Descriptor not found at {path}")
    data = infra_utils.read_descriptor_from_json(str(path))
    if data is None:
        raise StepError(f"Failed to read descriptor {path}")
    return path, data


def resolve_conda_env_path() -> Tuple[Path, Optional[Path]]:
    candidates: List[Path] = []
    user_conda = Path.home() / "miniconda3" / "bin" / "conda"
    if user_conda.exists():
        candidates.append(user_conda)
    system_conda = shutil.which("conda")
    if system_conda:
        system_path = Path(system_conda)
        if system_path not in candidates:
            candidates.append(system_path)
    if not candidates:
        raise StepError("Conda not available. Run `sci --init` first.")

    for candidate in candidates:
        try:
            proc = subprocess.run(
                [str(candidate), "env", "list", "--json"],
                capture_output=True,
                text=True,
                check=True,
            )
            data = json.loads(proc.stdout or "{}")
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            continue
        for env in data.get("envs", []):
            if Path(env).name == ENV_NAME:
                return Path(env), candidate

    raise StepError("scarabinfra environment missing; run `sci --init` first.")


def reexec_in_conda_env() -> None:
    if os.environ.get("SCI_IN_CONDA") == "1":
        return

    env_path, conda_bin = resolve_conda_env_path()
    env_bin = env_path / "bin"
    python_bin = env_bin / "python"
    if not python_bin.exists():
        raise StepError(f"Python not found in environment at {env_path}")

    env = dict(os.environ)
    env["SCI_IN_CONDA"] = "1"
    env["PATH"] = f"{env_bin}:{env.get('PATH', '')}"
    env["CONDA_PREFIX"] = str(env_path)
    env["CONDA_DEFAULT_ENV"] = ENV_NAME

    if conda_bin:
        env["CONDA_EXE"] = str(conda_bin)
        try:
            base_prefix = env_path.parent
            if base_prefix.name == "envs":
                base_prefix = base_prefix.parent
            env["CONDA_PYTHON_EXE"] = str(base_prefix / "bin" / "python")
        except Exception:
            pass

    cmd = [str(python_bin), str(Path(__file__).resolve()), *sys.argv[1:]]
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


def handle_descriptor_action(descriptor_name: str, action: str) -> None:
    path, descriptor = read_descriptor(descriptor_name)
    dtype = descriptor.get("descriptor_type")
    infra_dir = str(REPO_ROOT)

    if dtype == "simulation":
        sim_module = import_repo_module("scripts.run_simulation")
        dbg_lvl = 3 if action == "launch" else 2
        sim_action = {
            "launch": "launch",
            "simulate": "simulate",
            "kill": "kill",
            "info": "info",
            "clean": "clean",
        }.get(action)
        if sim_action is None:
            raise StepError(f"Unsupported action '{action}' for simulation descriptors")
        sim_module.run_simulation_command(str(path), sim_action, dbg_lvl=dbg_lvl, infra_dir=infra_dir)
        return

    if dtype == "trace":
        trace_module = import_repo_module("scripts.run_trace")
        dbg_lvl = 3 if action in {"launch", "trace"} else 2
        trace_action = {
            "launch": "launch",
            "trace": "trace",
            "kill": "kill",
            "info": "info",
            "clean": "clean",
        }.get(action)
        if trace_action is None:
            raise StepError(f"Unsupported action '{action}' for trace descriptors")
        trace_module.run_trace_command(str(path), trace_action, dbg_lvl=dbg_lvl, infra_dir=infra_dir)
        return

    if dtype == "perf":
        if action != "launch":
            raise StepError("Perf descriptors only support interactive launch")
        perf_module = import_repo_module("scripts.run_perf")
        perf_module.run_perf_command(str(path), "launch", dbg_lvl=3, infra_dir=infra_dir)
        return

    raise StepError(f"Unsupported descriptor type '{dtype}' for action '{action}'")


def confirm(prompt: str, *, default: bool) -> bool:
    if not sys.stdin.isatty():
        return default
    suffix = " [Y/n]" if default else " [y/N]"
    while True:
        response = input(f"{prompt}{suffix} ").strip().lower()
        if not response:
            return default
        if response in {"y", "yes"}:
            return True
        if response in {"n", "no"}:
            return False
        print("Please respond with 'y' or 'n'.")


def conda_env_exists() -> bool:
    try:
        output = run_command(["conda", "env", "list", "--json"], capture=True)
    except StepError:
        return False
    try:
        data = json.loads(output or "{}")
    except json.JSONDecodeError:
        return False
    envs = data.get("envs", [])
    for env_path in envs:
        if Path(env_path).name == ENV_NAME:
            return True
    return False


def ensure_docker(_: argparse.Namespace) -> Tuple[bool, str]:
    docker = shutil.which("docker")
    if docker:
        return True, f"Docker already available at {docker}"
    apt = shutil.which("apt-get")
    sudo = shutil.which("sudo")
    if not apt or not sudo:
        return False, "Docker not found and automatic installation only supports apt-based systems with sudo."
    try:
        run_command(["sudo", "apt-get", "update"])
        run_command(["sudo", "apt-get", "install", "-y", "docker.io"])
        if shutil.which("systemctl"):
            run_command(["sudo", "systemctl", "enable", "--now", "docker"], check=False)
        elif shutil.which("service"):
            run_command(["sudo", "service", "docker", "start"], check=False)
    except StepError as exc:
        return False, str(exc)
    docker = shutil.which("docker")
    if not docker:
        return False, "Docker installation attempted but docker binary not found."
    try:
        run_command(["docker", "--version"], check=False)
    except StepError as exc:
        return False, str(exc)
    return True, "Docker installed."


def configure_docker_permissions(_: argparse.Namespace) -> Tuple[bool, str]:
    sock = Path("/var/run/docker.sock")
    if not sock.exists():
        for candidate in (["sudo", "systemctl", "start", "docker"], ["sudo", "service", "docker", "start"]):
            if shutil.which(candidate[0]):
                run_command(candidate, check=False)
                if sock.exists():
                    break
        if not sock.exists():
            return False, "Docker socket not found; ensure the docker daemon is running."
    mode = sock.stat().st_mode & 0o777
    if mode == 0o666:
        return True, "Docker socket already has 0666 permissions."
    if shutil.which("sudo"):
        try:
            run_command(["sudo", "chmod", "666", str(sock)])
        except StepError as exc:
            return False, str(exc)
        return True, "Updated /var/run/docker.sock permissions."
    return False, "sudo not available to adjust /var/run/docker.sock permissions."


def ensure_conda_env(args: argparse.Namespace) -> Tuple[bool, str]:
    if not shutil.which("conda"):
        return False, "conda command not found. Install Miniconda or Anaconda."
    yaml_path = REPO_ROOT / "quickstart_env.yaml"
    if not yaml_path.exists():
        return False, f"Missing {yaml_path}"

    env_exists = conda_env_exists()
    if env_exists:
        needs_update, reason = conda_env_needs_update(yaml_path)
        if needs_update:
            info(f"Updating scarabinfra conda environment ({reason}).")
            try:
                run_command(
                    [
                        "conda",
                        "env",
                        "update",
                        "--name",
                        ENV_NAME,
                        "--file",
                        str(yaml_path),
                        "--prune",
                    ]
                )
            except StepError as exc:
                return False, f"Failed to update conda environment: {exc}"
            return True, f"Updated existing conda environment ({reason})."

        return True, "Conda environment already up to date."

    if not env_exists:
        try:
            run_command(["conda", "env", "create", "-f", str(yaml_path)])
        except StepError as exc:
            return False, f"Failed to create conda environment: {exc}"
        return True, "Created conda environment 'scarabinfra'."

    return True, "Conda environment already up to date."


def validate_conda_env(_: argparse.Namespace) -> Tuple[bool, str]:
    if not shutil.which("conda"):
        return False, "conda command not found."
    if not conda_env_exists():
        return False, "scarabinfra environment missing; rerun the create step."
    try:
        run_command(["conda", "run", "-n", ENV_NAME, "python", "-c", "import sys"], check=True)
    except StepError as exc:
        return False, f"Failed to run python inside scarabinfra environment: {exc}"
    return True, "Conda environment validated. Activate with `conda activate scarabinfra` when needed."


def parse_spec_dependencies(yaml_path: Path) -> Tuple[List[str], List[str]]:
    conda_deps: List[str] = []
    pip_deps: List[str] = []
    in_dependencies = False
    in_pip_block = False
    with yaml_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("dependencies:"):
                in_dependencies = True
                in_pip_block = False
                continue
            if not in_dependencies:
                continue
            if line.startswith("- "):
                item = line[2:].strip()
                if item == "pip:":
                    in_pip_block = True
                    continue
                if in_pip_block:
                    pip_deps.append(item)
                else:
                    conda_deps.append(item)
            elif line == "pip:":
                in_pip_block = True
    return conda_deps, pip_deps


def export_conda_packages() -> Tuple[List[str], List[str]]:
    export_cmd = [
        "conda",
        "env",
        "export",
        "--name",
        ENV_NAME,
        "--json",
        "--no-builds",
    ]
    try:
        output = run_command(export_cmd, capture=True)
    except StepError:
        # Fallback: retry without --no-builds for older conda versions.
        fallback_cmd = [
            "conda",
            "env",
            "export",
            "--name",
            ENV_NAME,
            "--json",
        ]
        try:
            output = run_command(fallback_cmd, capture=True)
        except StepError as exc:
            raise StepError(f"Unable to export conda environment: {exc}") from exc
    try:
        data = json.loads(output or "{}")
    except json.JSONDecodeError as exc:
        raise StepError("Failed to parse conda export output") from exc
    conda_deps: List[str] = []
    pip_deps: List[str] = []
    for entry in data.get("dependencies", []):
        if isinstance(entry, str):
            conda_deps.append(entry)
        elif isinstance(entry, dict):
            pip_deps.extend(entry.get("pip", []))
    return conda_deps, pip_deps


def split_conda_dep(dep: str) -> Tuple[str, Optional[str]]:
    parts = dep.split("=")
    name = parts[0]
    version = parts[1] if len(parts) >= 2 else None
    return name, version


def find_conda_dep(actual: List[str], name: str) -> Optional[Tuple[str, Optional[str]]]:
    for entry in actual:
        entry_name, entry_version = split_conda_dep(entry)
        if entry_name == name:
            return entry_name, entry_version
    return None


PIP_SPEC_PATTERN = re.compile(r"^([A-Za-z0-9_.-]+)(?:\[[^\]]+\])?(?:(==|!=|>=|<=|>|<|~=)\s*([^,;\s]+))?")


def parse_pip_spec(spec: str) -> Tuple[str, Optional[str], Optional[str]]:
    match = PIP_SPEC_PATTERN.match(spec.strip())
    if not match:
        return spec.strip().lower(), None, None
    name, operator, version = match.groups()
    return name.lower(), operator, version.strip() if version else None


def version_key(value: str) -> Tuple:
    parts = [part for part in re.split(r"[._-]", value) if part]
    key: List[object] = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part)
    return tuple(key)


def compare_versions(actual: str, operator: str, expected: str) -> bool:
    if actual is None or expected is None:
        return True
    if SpecifierSet is not None:
        try:
            spec = SpecifierSet(f"{operator}{expected}")
            return spec.contains(actual, prereleases=True)
        except Exception:  # pragma: no cover - fall back to manual comparison
            pass
    actual_key = version_key(actual)
    expected_key = version_key(expected)
    if operator in {"==", "="}:
        return actual_key == expected_key
    if operator == "!=":
        return actual_key != expected_key
    if operator == ">=":
        return actual_key >= expected_key
    if operator == ">":
        return actual_key > expected_key
    if operator == "<=":
        return actual_key <= expected_key
    if operator == "<":
        return actual_key < expected_key
    if operator == "~=":
        # approximate: same major component and >= expected
        return actual_key[:1] == expected_key[:1] and actual_key >= expected_key
    return True


def build_actual_pip_map(actual_pip: List[str]) -> Dict[str, Tuple[Optional[str], Optional[str], str]]:
    result: Dict[str, Tuple[Optional[str], Optional[str], str]] = {}
    for spec in actual_pip:
        name, operator, version = parse_pip_spec(spec)
        # For equality operators, strip leading '=' characters to get version number.
        cleaned_version = None
        if version:
            cleaned_version = version.lstrip("=")
        result[name] = (operator, cleaned_version, spec)
    return result


def conda_env_needs_update(yaml_path: Path) -> Tuple[bool, str]:
    expected_conda, expected_pip = parse_spec_dependencies(yaml_path)
    try:
        actual_conda, actual_pip = export_conda_packages()
    except StepError as exc:
        return True, str(exc)

    issues: List[str] = []

    for expected in expected_conda:
        name, expected_version = split_conda_dep(expected)
        match = find_conda_dep(actual_conda, name)
        if not match:
            issues.append(f"missing {expected}")
            continue
        _, actual_version = match
        if expected_version and (
            not actual_version or not actual_version.startswith(expected_version)
        ):
            issues.append(
                f"{name} version mismatch (expected {expected_version}, found {actual_version or 'unknown'})"
            )

    actual_pip_map = build_actual_pip_map(actual_pip)
    for expected_pip_pkg in expected_pip:
        name, operator, version = parse_pip_spec(expected_pip_pkg)
        actual_entry = actual_pip_map.get(name)
        if not actual_entry:
            issues.append(f"pip:{expected_pip_pkg} missing")
            continue
        actual_operator, actual_version, raw_actual = actual_entry
        if operator and version:
            if actual_version:
                if not compare_versions(actual_version, operator, version):
                    issues.append(
                        f"pip:{name} version mismatch (expected {operator}{version}, found {actual_version})"
                    )
            elif operator in {"==", "="}:
                issues.append(
                    f"pip:{name} requires exact version {operator}{version} but environment provides {raw_actual}"
                )

    if issues:
        return True, ", ".join(issues)
    return False, "Environment in sync with quickstart_env.yaml."


def ensure_ssh_key(args: argparse.Namespace) -> Tuple[bool, str]:
    ssh_dir = Path.home() / ".ssh"
    try:
        ssh_dir.mkdir(mode=0o700, exist_ok=True)
    except OSError as exc:
        return False, f"Could not create {ssh_dir}: {exc}"
    public_keys = sorted(ssh_dir.glob("id_*.pub"))
    if public_keys:
        return True, f"Found existing SSH key at {public_keys[0]}"
    if not shutil.which("ssh-keygen"):
        return False, "ssh-keygen not found; install the OpenSSH client."
    if not confirm(
        "No GitHub SSH key found. Generate a new ed25519 key now?",
        default=True,
    ):
        return False, "SSH key not generated."
    comment = ""
    if sys.stdin.isatty():
        comment = input("GitHub email or comment (optional): ").strip()
    key_path = ssh_dir / "id_ed25519"
    if key_path.exists():
        return False, f"{key_path} already exists; remove or rename it before rerunning."
    cmd = ["ssh-keygen", "-t", "ed25519", "-f", str(key_path), "-N", ""]
    if comment:
        cmd.extend(["-C", comment])
    try:
        run_command(cmd)
    except StepError as exc:
        return False, str(exc)
    pub_key = Path(f"{key_path}.pub")
    message = (
        f"Generated SSH key at {pub_key}. Upload it to GitHub: https://github.com/settings/ssh/new\n"
        f"Use `cat {pub_key}` to copy the key."
    )
    return True, message


def ensure_traces(_: argparse.Namespace) -> Tuple[bool, str]:
    trace_home = Path(
        os.environ.get("trace_home")
        or (Path.home() / "traces")
    ).expanduser()
    try:
        trace_home.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return False, f"Could not create trace directory {trace_home}: {exc}"
    extracted_dir = trace_home / TRACE_DIR_NAME
    if extracted_dir.exists():
        return True, f"Trace directory already exists at {extracted_dir}"
    archive = trace_home / TRACE_ARCHIVE_NAME
    if not archive.exists():
        if not confirm(
            f"SimPoint traces not found under {trace_home}. Download now?",
            default=False,
        ):
            return True, "Skipped trace download (user opted to handle manually)."
        download_cmd = [
            "conda",
            "run",
            "-n",
            ENV_NAME,
            "python",
            "-m",
            "gdown",
            "--fuzzy",
            DEFAULT_TRACE_URL,
            "-O",
            str(archive),
        ]
        try:
            run_command(download_cmd)
        except StepError as exc:
            if shutil.which("gdown"):
                try:
                    run_command(
                        [
                            "gdown",
                            "--fuzzy",
                            DEFAULT_TRACE_URL,
                            "-O",
                            str(archive),
                        ]
                    )
                except StepError as fallback_exc:
                    return False, f"Failed to download traces: {fallback_exc}"
            else:
                return False, f"Failed to download traces: {exc}"
    try:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=trace_home)
    except (tarfile.TarError, OSError) as exc:
        return False, f"Failed to extract {archive}: {exc}"
    return True, f"Installed traces under {extracted_dir}"


def ensure_conda_installed(_: argparse.Namespace) -> Tuple[bool, str]:
    target_dir = Path.home() / "miniconda3"
    target_bin = target_dir / "bin"
    target_conda = target_bin / "conda"
    target_python = target_bin / "python"

    if target_conda.exists():
        os.environ["PATH"] = f"{target_bin}:{os.environ.get('PATH', '')}"
        os.environ["CONDA_EXE"] = str(target_conda)
        os.environ["CONDA_PYTHON_EXE"] = str(target_python)
        return True, f"Using existing Miniconda at {target_dir}. Add `{target_bin}` to PATH in future shells."

    def existing_conda_info() -> Tuple[Optional[str], Optional[str], bool]:
        conda_bin = shutil.which("conda")
        if not conda_bin:
            return None, None, False
        try:
            output = run_command(["conda", "info", "--json"], capture=True)
        except StepError:
            return conda_bin, None, False
        try:
            info = json.loads(output or "{}")
        except json.JSONDecodeError:
            return conda_bin, None, False
        root_prefix = info.get("root_prefix")
        pkgs_dirs = info.get("pkgs_dirs", [])
        under_home = False
        if root_prefix:
            try:
                Path(root_prefix).resolve().relative_to(Path.home().resolve())
                under_home = True
            except ValueError:
                under_home = False
        writable = False
        if root_prefix and os.access(root_prefix, os.W_OK):
            for pkgs_dir in pkgs_dirs:
                if os.access(pkgs_dir, os.W_OK):
                    writable = True
                    break
        return conda_bin, root_prefix, writable and under_home

    conda_path, root_prefix, is_user_writable = existing_conda_info()
    if is_user_writable and conda_path:
        return True, f"Conda already installed at {conda_path}."

    if conda_path:
        location = root_prefix or conda_path
        print(f"Found existing conda at {location}, but installing user-local Miniconda for this workflow.")

    if not confirm(
        "Conda not found. Install Miniconda locally?",
        default=True,
    ):
        return False, "Conda installation skipped; install Miniconda manually and rerun."

    installer_path = Path.home() / "miniconda3-installer.sh"

    try:
        run_command(["curl", "-fsSL", MINICONDA_URL, "-o", str(installer_path)])
    except StepError as exc:
        return False, f"Failed to download Miniconda installer: {exc}"

    install_cmd = ["bash", str(installer_path), "-b", "-p", str(target_dir)]
    if target_dir.exists():
        install_cmd.append("-u")

    try:
        run_command(install_cmd)
    except StepError as exc:
        return False, f"Failed to install Miniconda: {exc}"
    finally:
        try:
            if installer_path.exists():
                installer_path.unlink()
        except OSError:
            pass

    os.environ["PATH"] = f"{target_bin}:{os.environ.get('PATH', '')}"
    os.environ["CONDA_EXE"] = str(target_conda)
    os.environ["CONDA_PYTHON_EXE"] = str(target_python)

    if shutil.which("conda"):
        return True, f"Installed Miniconda at {target_dir}. Add `{target_bin}` to PATH in future shells."

    return False, (
        f"Miniconda installed at {target_dir}, but `conda` not on PATH. Add {target_bin} to PATH and rerun."
    )


def ghcr_credentials_present() -> Tuple[bool, str]:
    config_path = Path.home() / ".docker" / "config.json"
    if not config_path.is_file():
        return False, "Docker config missing; ghcr.io credentials not found."
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "Could not parse Docker config; ghcr.io credentials unknown."
    auths = data.get("auths", {})
    cred_helpers = data.get("credHelpers", {})
    creds_present = "ghcr.io" in auths or "ghcr.io" in cred_helpers
    if creds_present:
        return True, "ghcr.io credentials already configured."
    return False, "ghcr.io credentials not configured."


def prebuilt_image_present() -> Tuple[bool, str]:
    tag_path = REPO_ROOT / "last_built_tag.txt"
    if not tag_path.is_file():
        return False, "No last_built_tag.txt found; cannot verify prebuilt image."
    tag = tag_path.read_text(encoding="utf-8").strip()
    if not tag:
        return False, "last_built_tag.txt empty; cannot verify prebuilt image."
    repos = ["allbench_traces", "ghcr.io/litz-lab/scarab-infra/allbench_traces"]
    patterns = {f"{repo}:{tag}" for repo in repos}
    try:
        output = run_command(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture=True,
            check=False,
        ) or ""
    except StepError:
        return False, "Unable to query local Docker images."
    images = {line.strip() for line in output.splitlines() if line.strip()}
    for pattern in patterns:
        if pattern in images:
            return True, f"Prebuilt image {pattern} already pulled."
    return False, f"Prebuilt image allbench_traces:{tag} not found locally."


def run_build_scarab(descriptor_name: str) -> int:
    infra_utils = load_infra_utilities()
    descriptor_path = REPO_ROOT / "json" / f"{descriptor_name}.json"
    if not descriptor_path.is_file():
        raise StepError(f"Descriptor not found at {descriptor_path}")
    descriptor = infra_utils.read_descriptor_from_json(str(descriptor_path))
    if descriptor is None:
        raise StepError(f"Failed to read descriptor {descriptor_path}")
    if descriptor.get("descriptor_type") != "simulation":
        raise StepError("`--build` currently supports simulation descriptors only.")

    scarab_path = descriptor.get("scarab_path")
    if not scarab_path:
        raise StepError("Descriptor missing `scarab_path`.")
    if not Path(scarab_path).exists():
        raise StepError(f"scarab_path '{scarab_path}' does not exist.")

    docker_home = descriptor.get("root_dir")
    if not docker_home:
        raise StepError("Descriptor missing `root_dir`.")
    if not Path(docker_home).exists():
        raise StepError(f"root_dir '{docker_home}' does not exist.")

    architecture = descriptor.get("architecture")
    if not architecture:
        raise StepError("Descriptor missing `architecture`.")

    simulations = descriptor.get("simulations") or []
    if not simulations:
        raise StepError("Descriptor contains no simulations to derive docker image.")

    workloads_path = REPO_ROOT / "workloads" / "workloads_db.json"
    workloads = infra_utils.read_descriptor_from_json(str(workloads_path))
    if workloads is None:
        raise StepError("Failed to read workloads/workloads_db.json.")

    docker_prefix_list = infra_utils.get_image_list(simulations, workloads)
    if not docker_prefix_list:
        raise StepError("Unable to determine docker image from descriptor simulations.")

    build_mode: str = descriptor.get("scarab_build") or "opt"
    if build_mode not in {"opt", "dbg"}:
        raise StepError("Build mode must be 'opt' or 'dbg'.")

    try:
        githash = run_command(["git", "rev-parse", "--short", "HEAD"], capture=True, check=True, input_data=None)
    except StepError as exc:
        raise StepError("Failed to obtain scarab-infra git hash.") from exc
    if githash is None:
        raise StepError("Git hash query returned no output.")
    githash = githash.strip()

    user = getpass.getuser()

    print_heading("Build Scarab")
    info(f"Descriptor: {descriptor_path.name}")
    info(f"Mode: make {build_mode}")
    info(f"Scarab path: {scarab_path}")

    try:
        infra_utils.prepare_simulation(
            user,
            scarab_path,
            build_mode,
            docker_home,
            descriptor.get("experiment", "build"),
            architecture,
            docker_prefix_list,
            githash,
            str(REPO_ROOT),
            ["current"],
            interactive_shell=True,
            dbg_lvl=2,
            stream_build=True,
        )
    except subprocess.CalledProcessError as exc:
        raise StepError(
            f"Scarab build failed (exit {exc.returncode}).\nSTDOUT:{exc.stdout}\nSTDERR:{exc.stderr}"
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise StepError(f"Scarab build failed: {exc}") from exc

    info("Scarab build completed successfully.")
    return 0


def run_build_image(workload_group: str) -> int:
    if not workload_group:
        raise StepError("Provide a workload group name (see ./sci --list).")
    workloads_dir = REPO_ROOT / "workloads"
    target_dir = workloads_dir / workload_group
    if not target_dir.is_dir():
        raise StepError(
            f"Workload group '{workload_group}' not found under {workloads_dir}."
        )
    if not shutil.which("docker"):
        raise StepError("Docker CLI not available; install Docker before building images.")

    # Ensure no local edits to shared docker context dirs.
    status_output = run_command(
        [
            "git",
            "status",
            "--porcelain",
            str(REPO_ROOT / "common"),
            str(target_dir),
        ],
        capture=True,
        cwd=REPO_ROOT,
    )
    dirty = [line for line in (status_output or "").splitlines() if line and not line.startswith("??")]
    if dirty:
        details = "\n".join(dirty[:10])
        raise StepError(
            f"There are uncommitted changes in ./common or ./workloads/{workload_group}.\n"
            "Commit, stash, or revert them before building.\n"
            f"Sample status entries:\n{details}"
        )

    githash = run_command(["git", "rev-parse", "--short", "HEAD"], capture=True, cwd=REPO_ROOT)
    if not githash:
        raise StepError("Unable to determine git hash for tagging the image.")
    githash = githash.strip()
    current_ref = f"{workload_group}:{githash}"

    def current_images() -> set[str]:
        output = run_command(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture=True,
        )
        return {line.strip() for line in (output or "").splitlines() if line.strip()}

    images = current_images()
    print_heading("Build Docker Image")
    info(f"Workload group: {workload_group}")
    info(f"Target tag: {current_ref}")

    if current_ref in images:
        info(f"Image {current_ref} already available; nothing to do.")
        return 0

    tag_path = REPO_ROOT / "last_built_tag.txt"
    last_hash = tag_path.read_text(encoding="utf-8").strip() if tag_path.is_file() else ""
    base_ref = f"{workload_group}:{last_hash}" if last_hash else None

    if base_ref and base_ref not in images:
        remote_ref = f"ghcr.io/litz-lab/scarab-infra/{workload_group}:{last_hash}"
        try:
            info(f"Pulling prebuilt image {remote_ref}")
            run_command(["docker", "pull", remote_ref])
            run_command(["docker", "tag", remote_ref, base_ref])
            run_command(["docker", "rmi", remote_ref], check=False)
            images = current_images()
        except StepError:
            info(f"Unable to pull {remote_ref}; will build from Dockerfile instead.")
            base_ref = None

    dockerfile_path = target_dir / "Dockerfile"
    if not dockerfile_path.is_file():
        raise StepError(f"Dockerfile not found at {dockerfile_path}")

    rebuild_required = True
    if base_ref and base_ref in images:
        diff_output = run_command(
            [
                "git",
                "diff",
                last_hash,
                "--",
                str(REPO_ROOT / "common"),
                str(target_dir),
            ],
            capture=True,
            cwd=REPO_ROOT,
        )
        rebuild_required = bool(diff_output and diff_output.strip())

    if rebuild_required:
        info("Changes detected or no base image available; building from source.")
        run_command(
            [
                "docker",
                "build",
                str(REPO_ROOT),
                "-f",
                str(dockerfile_path),
                "--no-cache",
                "-t",
                current_ref,
            ],
        )
    else:
        info(f"No Dockerfile changes since {last_hash}; retagging {base_ref} -> {current_ref}")
        run_command(["docker", "tag", base_ref, current_ref])

    info(f"Docker image ready: {current_ref}")
    return 0


def maybe_install_slurm(_: argparse.Namespace) -> Tuple[bool, str]:
    # Check if Slurm is already installed via common binaries.
    if shutil.which("sinfo") or shutil.which("squeue"):
        return True, "Slurm already installed."
    print(
        "Slurm enables cluster scheduling (squeue/sbatch). The apt package installs the"
        " services but leaves /etc/slurm-llnl/slurm.conf and the munge key unconfigured."
        " You must update those files manually before using Slurm."
    )
    if not confirm(
        "Install Slurm components (optional)?",
        default=False,
    ):
        return True, "Skipped Slurm installation."
    apt = shutil.which("apt-get")
    sudo = shutil.which("sudo")
    if apt and sudo:
        try:
            run_command(["sudo", "apt-get", "update"])
            run_command(["sudo", "apt-get", "install", "-y", "slurm-wlm"])
            return True, "Installed slurm-wlm via apt-get. Configure /etc/slurm-llnl and munge before use."
        except StepError as exc:
            return False, str(exc)
    return False, "Automated Slurm setup supported only on apt-based systems. See docs/slurm_install_guide.md."


def maybe_docker_login(_: argparse.Namespace) -> Tuple[bool, str]:
    if not shutil.which("docker"):
        return False, "Docker not available; cannot login to ghcr.io."
    creds_present, creds_msg = ghcr_credentials_present()
    image_present, image_msg = prebuilt_image_present()
    if creds_present:
        message = creds_msg
        if image_present:
            message += f" {image_msg}"
        return True, message
    if image_present:
        return True, image_msg
    print(
        "Logging in to ghcr.io lets the helper pull prebuilt images instead of rebuilding"
        " them locally. Provide a GitHub Personal Access Token with read:packages scope."
    )
    if not confirm(
        "Login to ghcr.io to pull prebuilt images (optional)?",
        default=False,
    ):
        return True, "Skipped ghcr.io login."
    username = (
        os.environ.get("GH_USERNAME")
        or os.environ.get("GITHUB_USERNAME")
        or ""
    ).strip()
    token = (
        os.environ.get("GHCR_TOKEN")
        or os.environ.get("GITHUB_TOKEN")
        or ""
    ).strip()
    if not username and sys.stdin.isatty():
        username = input("GitHub username: ").strip()
    if not token:
        token = getpass.getpass("GitHub token (with read:packages): ").strip()
    if not username or not token:
        return False, "Provide --gh-username/--gh-token or set GH_USERNAME/GHCR_TOKEN for login."
    try:
        run_command(
            ["docker", "login", "ghcr.io", "-u", username, "--password-stdin"],
            input_data=f"{token}\n",
        )
    except StepError as exc:
        return False, str(exc)
    return True, "Authenticated with ghcr.io."


def run_init(args: argparse.Namespace) -> int:
    steps = [
        ("Install Docker", ensure_docker),
        ("Configure Docker permissions", configure_docker_permissions),
        ("Install Miniconda", ensure_conda_installed),
        ("Create scarabinfra conda env", ensure_conda_env),
        ("Validate conda env activation", validate_conda_env),
        ("Ensure GitHub SSH key", ensure_ssh_key),
        ("Download simpoint traces", ensure_traces),
        ("(Optional) Slurm installation", maybe_install_slurm),
        ("(Optional) ghcr.io login to pull pre-built images from GitHub Container Registry (recommended)", maybe_docker_login),
    ]
    summary: List[Tuple[str, bool, str]] = []
    for title, func in steps:
        print_heading(title)
        try:
            success, message = func(args)
        except StepError as exc:
            success, message = False, str(exc)
        info(message)
        summary.append((title, success, message))
    print("\nInit summary:")
    for title, success, message in summary:
        status = "ok" if success else "failed"
        print(f"- [{status}] {title}: {message}")
    failures = [
        title for title, success, _ in summary if not success and title not in OPTIONAL_TITLES
    ]
    if failures:
        print("\nThe following required steps failed:")
        for title in failures:
            print(f"- {title}")
        print("Resolve the issues above and rerun `sci --init`.")
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sci",
        description="scarab-infra helper CLI.",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Run environment bootstrap steps.",
    )
    parser.add_argument(
        "--build-scarab",
        dest="build_scarab",
        metavar="DESCRIPTOR",
        help="Build scarab sources defined in json/<DESCRIPTOR>.json.",
    )
    parser.add_argument(
        "--build-image",
        dest="build_image",
        metavar="WORKLOAD_GROUP",
        help="Build or retag a workload Docker image for the given group.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List workload group names (requires jq and python modules).",
    )
    parser.add_argument(
        "--interactive",
        dest="interactive",
        metavar="DESCRIPTOR",
        help="Open an interactive shell for the workloads in json/<DESCRIPTOR>.json.",
    )
    parser.add_argument(
        "--trace",
        metavar="DESCRIPTOR",
        help="Collect traces using the configuration in json/<DESCRIPTOR>.json.",
    )
    parser.add_argument(
        "--sim",
        metavar="DESCRIPTOR",
        help="Run simulations defined in json/<DESCRIPTOR>.json.",
    )
    parser.add_argument(
        "--kill",
        metavar="DESCRIPTOR",
        help="Kill active simulations for json/<DESCRIPTOR>.json.",
    )
    parser.add_argument(
        "--status",
        metavar="DESCRIPTOR",
        help="Show run and node status for json/<DESCRIPTOR>.json.",
    )
    parser.add_argument(
        "--clean",
        metavar="DESCRIPTOR",
        help="Remove containers and state for json/<DESCRIPTOR>.json.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    requested = [
        bool(args.init),
        bool(args.build_scarab),
        bool(args.build_image),
        bool(args.list),
        bool(args.interactive),
        bool(args.trace),
        bool(args.sim),
        bool(args.kill),
        bool(args.status),
        bool(args.clean),
    ]
    if sum(requested) > 1:
        print("Use only one primary action per invocation.")
        return 1

    needs_env = any([
        args.build_scarab,
        args.build_image,
        args.interactive,
        args.trace,
        args.sim,
        args.kill,
        args.status,
        args.clean,
    ])
    if needs_env:
        reexec_in_conda_env()

    if args.build_scarab:
        try:
            return run_build_scarab(args.build_scarab)
        except StepError as exc:
            print(exc)
            return 1
    if args.build_image:
        try:
            return run_build_image(args.build_image)
        except StepError as exc:
            print(exc)
            return 1
    if args.init:
        return run_init(args)
    if args.list:
        workloads_path = REPO_ROOT / "workloads" / "workloads_db.json"
        try:
            with workloads_path.open(encoding="utf-8") as handle:
                workloads = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Failed to read workloads database: {exc}")
            return 1

        use_color = sys.stdout.isatty()
        green = "\033[92m" if use_color else ""
        red = "\033[31m" if use_color else ""
        reset = "\033[0m" if use_color else ""

        def walk(prefix: str, node: Dict[str, object]) -> None:
            sim_info = node.get("simulation") if isinstance(node, dict) else None
            if isinstance(sim_info, dict):
                print(prefix)
                for mode, details in sim_info.items():
                    if mode == "prioritized_mode":
                        continue
                    image_name = details.get("image_name", "?") if isinstance(details, dict) else "?"
                    print(f"    <{green}{mode}{reset} : {red}{image_name}{reset}>")
                return
            if isinstance(node, dict):
                for key, value in node.items():
                    next_prefix = f"{prefix}/{key}" if prefix else key
                    if isinstance(value, dict):
                        walk(next_prefix, value)

        header = "Simulation mode"
        image_label = "Docker image name to build"
        if use_color:
            header = f"{green}{header}{reset}"
            image_label = f"{red}{image_label}{reset}"
        print(f"Workload    <{header} : {image_label}>")
        print("----------------------------------------------------------")
        if isinstance(workloads, dict):
            walk("", workloads)
        return 0
    if args.interactive:
        try:
            handle_descriptor_action(args.interactive, "launch")
            return 0
        except (StepError, RuntimeError) as exc:
            print(exc)
            return 1
    if args.trace:
        try:
            handle_descriptor_action(args.trace, "trace")
            return 0
        except (StepError, RuntimeError) as exc:
            print(exc)
            return 1
    if args.sim:
        try:
            handle_descriptor_action(args.sim, "simulate")
            return 0
        except (StepError, RuntimeError) as exc:
            print(exc)
            return 1
    if args.kill:
        try:
            handle_descriptor_action(args.kill, "kill")
            return 0
        except (StepError, RuntimeError) as exc:
            print(exc)
            return 1
    if args.status:
        try:
            handle_descriptor_action(args.status, "info")
            return 0
        except (StepError, RuntimeError) as exc:
            print(exc)
            return 1
    if args.clean:
        try:
            handle_descriptor_action(args.clean, "clean")
            return 0
        except (StepError, RuntimeError) as exc:
            print(exc)
            return 1
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

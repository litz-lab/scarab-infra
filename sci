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
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

try:
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version
except ImportError:  # pragma: no cover - packaging is usually available
    SpecifierSet = None
    Version = None

REPO_ROOT = Path(__file__).resolve().parent
ENV_NAME = "scarabinfra"
MINICONDA_URL = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
OPTIONAL_TITLES = {
    "Optional Slurm installation",
    "Optional ghcr.io login",
}

COOKIES_CACHE_PATH = Path.home() / ".cache" / "gdown" / "cookies.txt"
_COOKIES_STATUS = {"prompted": False, "skipped": False}
WORKLOADS_DB_PATH = REPO_ROOT / "workloads" / "workloads_db.json"


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


def extract_descriptor_expectations(descriptor: Dict[str, Any]) -> Tuple[Set[str], Set[str]]:
    workloads: Set[str] = set()
    workloads_db: Optional[Dict[str, Any]] = None

    def ensure_workloads_db() -> Dict[str, Any]:
        nonlocal workloads_db
        if workloads_db is not None:
            return workloads_db
        try:
            with WORKLOADS_DB_PATH.open(encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (OSError, json.JSONDecodeError):
            loaded = {}
        workloads_db = loaded if isinstance(loaded, dict) else {}
        return workloads_db

    def add_workloads_from_subsuite(
        subsuite_node: Optional[Dict[str, Any]],
        *,
        required_type: Optional[str] = None,
    ) -> None:
        if not isinstance(subsuite_node, dict):
            return
        for name, payload in subsuite_node.items():
            if isinstance(payload, dict):
                if required_type:
                    sim_info = payload.get("simulation")
                    if (
                        not isinstance(sim_info, dict)
                        or required_type not in sim_info
                    ):
                        continue
                workloads.add(str(name))

    simulations = descriptor.get("simulations")
    if isinstance(simulations, list):
        for entry in simulations:
            if isinstance(entry, dict):
                workload = entry.get("workload")
                if workload:
                    workloads.add(str(workload))
                    continue
                suite = entry.get("suite")
                subsuite = entry.get("subsuite")
                sim_type = entry.get("simulation_type")
                if suite and workload is None:
                    db = ensure_workloads_db()
                    suite_node = db.get(str(suite))
                    if not isinstance(suite_node, dict):
                        continue
                    if subsuite:
                        add_workloads_from_subsuite(
                            suite_node.get(str(subsuite)),
                            required_type=str(sim_type) if sim_type else None,
                        )
                    else:
                        for subsuite_node in suite_node.values():
                            if isinstance(subsuite_node, dict):
                                add_workloads_from_subsuite(
                                    subsuite_node,
                                    required_type=str(sim_type) if sim_type else None,
                                )

    configurations = descriptor.get("configurations")
    config_names: Set[str] = set()
    if isinstance(configurations, dict):
        config_names = {str(name) for name in configurations.keys()}

    return workloads, config_names


def collect_stats_for_visualization(descriptor_path: Path, stats_path: Path) -> bool:
    stat_script = REPO_ROOT / "scarab_stats" / "stat_collector.py"
    if not stat_script.is_file():
        print(f"Stat collector script not found at {stat_script}.")
        return False

    python_cmd: List[str] = []
    sys_python = Path(sys.executable) if sys.executable else None
    if sys_python and sys_python.exists():
        python_cmd = [sys.executable]
    else:
        try:
            env_path, conda_bin = resolve_conda_env_path()
            env_python = env_path / "bin" / "python"
            if env_python.exists():
                python_cmd = [str(env_python)]
            elif conda_bin:
                python_cmd = [str(conda_bin), "run", "-n", ENV_NAME, "python"]
        except StepError:
            python_cmd = []

    if not python_cmd:
        print("Unable to locate python executable for stat collection.")
        return False

    env = os.environ.copy()
    tmp_dir = env.get("TMPDIR")
    if not tmp_dir or not Path(tmp_dir).is_dir():
        tmp_path = stats_path.parent / "tmp"
        try:
            tmp_path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            print(f"Failed to prepare temporary directory {tmp_path}: {exc}")
            return False
        tmp_dir = str(tmp_path)
    env["TMPDIR"] = tmp_dir

    cmd = python_cmd + [
        str(stat_script),
        "-d",
        str(descriptor_path),
        "-o",
        str(stats_path),
    ]
    print("Refreshing collected stats via stat_collector.py...")
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        print(f"Stat collection failed with exit code {exc.returncode}.")
        return False
    except FileNotFoundError as exc:
        print(f"Failed to launch stat_collector.py: {exc}")
        return False

    return True


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


def handle_descriptor_action(descriptor_name: str, action: str, dbg_override: Optional[int] = None) -> None:
    path, descriptor = read_descriptor(descriptor_name)
    dtype = descriptor.get("descriptor_type")
    infra_dir = str(REPO_ROOT)

    def resolve_dbg(default: int) -> int:
        return dbg_override if dbg_override is not None else default

    if dtype == "simulation":
        sim_module = import_repo_module("scripts.run_simulation")
        dbg_lvl = resolve_dbg(3 if action == "launch" else 2)
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
        dbg_lvl = resolve_dbg(3 if action in {"launch", "trace"} else 2)
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
        perf_module.run_perf_command(str(path), "launch", dbg_lvl=resolve_dbg(3), infra_dir=infra_dir)
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


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    value = float(num_bytes)
    unit = 0
    while value >= 1024 and unit < len(units) - 1:
        value /= 1024
        unit += 1
    if unit == 0:
        return f"{int(value)} {units[unit]}"
    return f"{value:.1f} {units[unit]}"


def format_duration(seconds: float) -> str:
    if seconds < 1:
        return "<1s"
    minutes, sec = divmod(int(seconds + 0.5), 60)
    if minutes == 0:
        return f"{sec}s"
    hours, minutes = divmod(minutes, 60)
    if hours == 0:
        return f"{minutes}m {sec}s"
    return f"{hours}h {minutes}m"


def load_workloads_file(filename: str) -> Dict[str, Any]:
    path = REPO_ROOT / "workloads" / filename
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise StepError(f"Missing workloads file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise StepError(f"Failed to parse {path}: {exc}") from exc


def ensure_gdown_cookies(*, force_prompt: bool = False, initial_prompt: bool = False) -> bool:
    global _COOKIES_STATUS
    target = COOKIES_CACHE_PATH
    if target.exists():
        info(f"Using browser cookies at {target}")
        _COOKIES_STATUS["prompted"] = True
        _COOKIES_STATUS["skipped"] = False
        return True

    if _COOKIES_STATUS["skipped"]:
        return False

    if not sys.stdin.isatty():
        info(
            f"gdown requires an authenticated cookies file at {target}. Upload cookies.txt to that path and rerun."
        )
        _COOKIES_STATUS["skipped"] = True
        return False

    if _COOKIES_STATUS["prompted"] and not force_prompt:
        return False

    _COOKIES_STATUS["prompted"] = True
    header = (
        "Google Drive downloads require an authenticated cookies.txt file."
        if initial_prompt
        else "Google Drive downloads may require authentication on headless servers."
    )
    print(
        header
        + f"\nUpload your exported cookies.txt to this machine and enter its path to copy it into {target}."
        + "\nType 'skip' to continue without cookies (downloads may fail)."
    )
    while True:
        user_path = input("Path to cookies.txt: ").strip()
        if not user_path:
            print("Please enter a path or type 'skip'.")
            continue
        if user_path.lower() in {"skip", "s"}:
            info("Skipping cookie setup; Google Drive may block some downloads.")
            _COOKIES_STATUS["skipped"] = True
            return False
        source = Path(user_path).expanduser()
        if not source.is_file():
            print(f"File not found: {source}. Try again or type 'skip'.")
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        except OSError as exc:
            print(f"Failed to install cookies: {exc}. Try again or type 'skip'.")
            continue
        info(f"Stored cookies for gdown at {target}")
        _COOKIES_STATUS["skipped"] = False
        return True


def build_gdown_command(file_id: str, output_path: Path, *, use_conda: bool) -> List[str]:
    url = f"https://drive.google.com/uc?id={file_id}"
    cmd: List[str] = []
    if use_conda:
        cmd.extend([
            "conda",
            "run",
            "-n",
            ENV_NAME,
            "python",
            "-m",
            "gdown",
        ])
    else:
        cmd.append("gdown")

    cmd.append("--fuzzy")
    cmd.extend([url, "-O", str(output_path)])
    return cmd


def download_trace_file(file_id: str, output_path: Path) -> None:
    commands = [
        build_gdown_command(file_id, output_path, use_conda=True),
    ]
    if shutil.which("gdown"):
        commands.append(build_gdown_command(file_id, output_path, use_conda=False))
    last_exc: Optional[StepError] = None
    attempted_cookie_setup = False
    while True:
        retry_requested = False
        for cmd in commands:
            try:
                run_command(cmd)
                return
            except StepError as exc:
                last_exc = exc
                if not attempted_cookie_setup:
                    attempted_cookie_setup = True
                    if ensure_gdown_cookies():
                        info("Retrying download using browser cookies.")
                        retry_requested = True
                        break
        if retry_requested:
            continue
        break
    if last_exc:
        message = str(last_exc)
        if "Cannot retrieve the public link" in message or "Failed to retrieve file url" in message:
            if COOKIES_CACHE_PATH.exists():
                message += (
                    "\nGoogle Drive still refused the download even with cookies. "
                    "Verify that the trace is shared with the account used to export cookies, "
                    "or try again later when the quota resets."
                )
            else:
                message += (
                    "\nInstall cookies.txt exported from a browser session with access and rerun."
                )
        raise StepError(message)
    raise StepError("Unknown gdown failure")


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


def ensure_docker_running(_: argparse.Namespace) -> Tuple[bool, str]:
    def docker_is_responding() -> bool:
        try:
            run_command(["docker", "info"], capture=True)
            return True
        except StepError:
            return False

    def docker_service_active() -> bool:
        systemctl = shutil.which("systemctl")
        if systemctl:
            try:
                result = subprocess.run(
                    [systemctl, "is-active", "--quiet", "docker"],
                    check=False,
                )
                if result.returncode == 0:
                    return True
            except OSError:
                pass

        service = shutil.which("service")
        if service:
            try:
                result = subprocess.run(
                    [service, "docker", "status"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                if result.returncode == 0:
                    return True
            except OSError:
                pass

        return False

    if docker_is_responding() or docker_service_active():
        return True, "Docker daemon already running."

    start_commands: List[List[str]] = []
    systemctl = shutil.which("systemctl")
    service = shutil.which("service")
    sudo = shutil.which("sudo")

    if sudo and systemctl:
        start_commands.append(["sudo", "systemctl", "start", "docker"])
    elif systemctl:
        start_commands.append(["systemctl", "start", "docker"])
    elif sudo and service:
        start_commands.append(["sudo", "service", "docker", "start"])
    elif service:
        start_commands.append(["service", "docker", "start"])

    for command in start_commands:
        if shutil.which(command[0]) is None:
            continue
        try:
            run_command(command, check=False)
        except StepError:
            continue
        if docker_is_responding() or docker_service_active():
            return True, "Docker daemon started."

    return False, "Unable to automatically start the docker daemon; start it manually and rerun."


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


def ensure_conda_shell_hook(_: argparse.Namespace) -> Tuple[bool, str]:
    conda_bin = shutil.which("conda")
    if not conda_bin:
        return False, "conda command not found."
    try:
        base_output = run_command(["conda", "info", "--base"], capture=True)
    except StepError as exc:
        return False, f"Failed to determine conda base prefix: {exc}"
    conda_base = (base_output or "").strip()
    if not conda_base:
        return False, "conda info --base returned an empty path."
    hook_path = Path(conda_base) / "etc" / "profile.d" / "conda.sh"
    if not hook_path.exists():
        return False, f"Expected conda hook at {hook_path}."

    bashrc_path = Path.home() / ".bashrc"
    if bashrc_path.exists():
        try:
            existing = bashrc_path.read_text(encoding="utf-8")
        except OSError as exc:
            return False, f"Unable to read {bashrc_path}: {exc}"
        if (
            "conda shell.bash hook" in existing
            or "conda initialize" in existing
            or ">>> scarab-infra conda setup >>>" in existing
        ):
            return True, f"Conda shell hook already configured in {bashrc_path}."

    snippet = (
        "\n# >>> scarab-infra conda setup >>>\n"
        f'if [ -x "{conda_bin}" ]; then\n'
        f'    eval "$("{conda_bin}" shell.bash hook)"\n'
        "fi\n"
        "# <<< scarab-infra conda setup <<<\n"
    )
    try:
        with bashrc_path.open("a", encoding="utf-8") as handle:
            handle.write(snippet)
    except OSError as exc:
        return False, f"Failed to update {bashrc_path}: {exc}"
    return True, f"Added conda shell hook to {bashrc_path}."


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

    workloads_db = load_workloads_file("workloads_db.json")
    try:
        workloads_top = load_workloads_file("workloads_top_simp.json")
    except StepError:
        workloads_top = {}

    use_top_requested = confirm(
        "Download only the top 3 simpoints from workloads_top_simp.json? (Choose 'No' for all simpoints)",
        default=True,
    )
    use_top_three = bool(workloads_top) and use_top_requested
    if use_top_requested and not workloads_top:
        info("Top-3 simpoint metadata not found; defaulting to all simpoints from workloads_db.json.")

    source_data: Dict[str, Any] = workloads_top if use_top_three else workloads_db
    if not isinstance(source_data, dict) or not source_data:
        return True, "No workloads defined for trace download."

    ensure_gdown_cookies(force_prompt=True, initial_prompt=True)

    downloads = 0
    existing = 0
    missing_count = 0
    missing_examples: List[str] = []
    undownloadable_workloads: Dict[str, int] = {}
    failures: List[str] = []

    excluded_suites = {"deprecated", "dc_java", "fleetbench", "mongo-perf"}
    for suite, subsuites in source_data.items():
        if suite in excluded_suites:
            info(f"Skipping suite '{suite}' by default.")
            continue
        if not isinstance(subsuites, dict):
            continue
        db_subsuites = workloads_db.get(suite)
        if not isinstance(db_subsuites, dict):
            failures.append(f"Suite '{suite}' missing from workloads_db.json")
            continue

        suite_plan: Dict[str, List[Tuple[str, List[Dict[str, Any]]]]] = {}
        suite_missing: Dict[str, int] = {}
        suite_size_bytes = 0

        print_heading(f"Trace suite: {suite}")

        for subsuite, workloads in subsuites.items():
            if not isinstance(workloads, dict):
                continue
            db_workloads = db_subsuites.get(subsuite)
            if not isinstance(db_workloads, dict):
                failures.append(
                    f"Subsuite '{suite}/{subsuite}' missing from workloads_db.json"
                )
                continue
            subsuite_plan: List[Tuple[str, List[Dict[str, Any]]]] = []
            subsuite_missing = False
            for workload, workload_payload in workloads.items():
                if not isinstance(workload_payload, dict):
                    continue
                db_workload_entry = db_workloads.get(workload)
                if not isinstance(db_workload_entry, dict):
                    failures.append(
                        f"Workload '{suite}/{subsuite}/{workload}' missing from workloads_db.json"
                    )
                    continue
                all_simpoints = db_workload_entry.get("simpoints", [])
                if not isinstance(all_simpoints, list) or not all_simpoints:
                    continue
                if use_top_three:
                    top_simpoints = workload_payload.get("simpoints", [])
                    cluster_ids = {
                        simpoint.get("cluster_id")
                        for simpoint in top_simpoints
                        if isinstance(simpoint, dict) and simpoint.get("cluster_id") is not None
                    }
                    if not cluster_ids:
                        continue
                    selected_simpoints = [
                        simpoint
                        for simpoint in all_simpoints
                        if isinstance(simpoint, dict)
                        and simpoint.get("cluster_id") in cluster_ids
                    ]
                    missing_clusters = cluster_ids - {
                        simpoint.get("cluster_id")
                        for simpoint in selected_simpoints
                    }
                    if missing_clusters:
                        failures.append(
                            f"Missing drive_id mapping for {suite}/{subsuite}/{workload}: {sorted(missing_clusters)}"
                        )
                        continue
                else:
                    selected_simpoints = [
                        simpoint for simpoint in all_simpoints if isinstance(simpoint, dict)
                    ]

                missing_ids_for_workload = [
                    simpoint.get("cluster_id")
                    for simpoint in selected_simpoints
                    if not simpoint.get("drive_id")
                ]
                workload_key = f"{suite}/{subsuite}/{workload}"
                if missing_ids_for_workload:
                    missing_count += len(missing_ids_for_workload)
                    suite_missing[workload_key] = len(missing_ids_for_workload)
                    subsuite_missing = True
                    for cid in missing_ids_for_workload:
                        if len(missing_examples) < 5:
                            missing_examples.append(f"{workload_key}:{cid}")
                    continue

                subsuite_plan.append((workload, selected_simpoints))
                for simpoint in selected_simpoints:
                    size_value = simpoint.get("size_bytes")
                    if isinstance(size_value, int) and size_value > 0:
                        suite_size_bytes += size_value

            if subsuite_plan:
                suite_plan[subsuite] = subsuite_plan
            elif subsuite_missing:
                info(
                    f"No downloadable traces for subsuite '{suite}/{subsuite}'; add drive_id entries to enable."
                )

        if suite_missing:
            for key, count in suite_missing.items():
                undownloadable_workloads[key] = count

        prompt = f"Download traces for suite '{suite}'"
        if suite_size_bytes > 0:
            prompt += f" (~{format_size(suite_size_bytes)})"

        if not suite_plan:
            print(f"{prompt}? [skipped - no drive_id entries]")
            info(
                f"Add drive_id entries for suite '{suite}' to download its traces for future use."
            )
            continue

        if suite_missing:
            info(
                f"Suite '{suite}' has workload(s) without drive_id entries; they will be skipped until populated."
            )

        prompt += "?"
        default_choice = False
        if not confirm(prompt, default=default_choice):
            info(f"Skipped downloads for suite '{suite}' by user choice.")
            continue

        suite_downloaded_bytes = 0
        suite_downloads = 0
        suite_existing = 0
        suite_start = time.monotonic()
        for subsuite, workloads in suite_plan.items():
            subsuite_prompt = f"  Download subsuite '{suite}/{subsuite}'?"
            subsuite_size = 0
            for _, selected_simpoints in workloads:
                for simpoint in selected_simpoints:
                    size_value = simpoint.get("size_bytes")
                    if isinstance(size_value, int) and size_value > 0:
                        subsuite_size += size_value
            if subsuite_size > 0:
                subsuite_prompt += f" (~{format_size(subsuite_size)})"
            if not confirm(subsuite_prompt + "", default=False):
                info(f"Skipped subsuite '{suite}/{subsuite}'.")
                continue

            for workload, selected_simpoints in workloads:
                for simpoint in selected_simpoints:
                    cluster_id = simpoint.get("cluster_id")
                    file_id = simpoint.get("drive_id")
                    if cluster_id is None or not file_id:
                        continue
                    target_path = (
                        trace_home
                        / suite
                        / subsuite
                        / workload
                        / "traces"
                        / "simp"
                        / f"{cluster_id}.zip"
                    )
                    if target_path.exists():
                        existing += 1
                        suite_existing += 1
                        continue
                    try:
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                    except OSError as exc:
                        failures.append(f"Failed to create {target_path.parent}: {exc}")
                        continue
                    try:
                        download_trace_file(str(file_id), target_path)
                    except StepError as exc:
                        failures.append(
                            f"Download failed for {suite}/{subsuite}/{workload}:{cluster_id}: {exc}"
                        )
                        continue
                    downloads += 1
                    suite_downloads += 1
                    size_value = simpoint.get("size_bytes")
                    if isinstance(size_value, int) and size_value > 0:
                        suite_downloaded_bytes += size_value

        elapsed = time.monotonic() - suite_start
        summary_details: List[str] = []
        if suite_downloads:
            download_detail = f"{suite_downloads} new trace(s)"
            if suite_downloaded_bytes > 0:
                download_detail += f" (~{format_size(suite_downloaded_bytes)})"
            summary_details.append(download_detail)
        else:
            summary_details.append("no new downloads")
        if suite_existing:
            summary_details.append(f"{suite_existing} already present")
        summary = f"[{suite}] Completed in {format_duration(elapsed)}"
        if summary_details:
            summary += "; " + "; ".join(summary_details)
        info(summary + ".")

    if failures:
        return False, failures[0]

    skipped_msg = ""
    if missing_count:
        sample = ", ".join(missing_examples)
        extra = " (and more)" if missing_count > len(missing_examples) else ""
        skipped_msg = (
            f" Skipped {missing_count} trace(s) missing drive_id: {sample}{extra}."
        )
    if undownloadable_workloads:
        workload_list = list(undownloadable_workloads.keys())
        example_str = ", ".join(workload_list[:5])
        extra = " (and more)" if len(workload_list) > 5 else ""
        info(
            "Workload(s) skipped due to missing drive_id entries: "
            + example_str
            + extra
        )

    if downloads and existing:
        return True, f"Downloaded {downloads} trace(s); {existing} already present." + skipped_msg
    if downloads:
        return True, f"Downloaded {downloads} trace(s)." + skipped_msg
    if existing:
        return True, f"All requested traces already present ({existing})." + skipped_msg
    if skipped_msg:
        return True, skipped_msg.strip()
    return True, "No traces selected for download."


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
            ["scarab_current"],
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
    if sys.stdin.isatty():
        print_heading("Prepare Google Drive cookies.txt")
        info(
            "Google limits bulk downloads from shared folders; providing browser cookies lets gdown reuse your session when Drive throttles direct access. "
            "If you can still open the files in your browser, exporting cookies.txt may unblock automated downloads."
        )
        info(
            "For users who need to download traces from the shared Google Drive, prepare cookies.txt before continuing:"
            "\n1. On your local machine, open the shared folder in a logged-in browser session."
            "\n2. Install a cookies export extension (e.g. 'Get cookies.txt LOCALLY')."
            "\n3. Use the extension's 'Export All Cookies' button to save the folder cookies as cookies.txt."
            "\n4. Upload cookies.txt to this server before continuing (for example: scp cookies.txt user@host:~/.cache/gdown/cookies.txt)."
        )
        if not confirm("Ready to continue with the setup now?", default=True):
            print("Re-run `./sci --init` after cookies.txt is uploaded.")
            return 0
    steps = [
        ("Install Docker", ensure_docker),
        ("Start Docker daemon", ensure_docker_running),
        ("Configure Docker permissions", configure_docker_permissions),
        ("Install Miniconda", ensure_conda_installed),
        ("Create scarabinfra conda env", ensure_conda_env),
        ("Configure conda shell hook", ensure_conda_shell_hook),
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


def run_visualize(descriptor_name: str) -> int:
    try:
        descriptor_path, descriptor = read_descriptor(descriptor_name)
    except StepError as exc:
        print(exc)
        return 1

    if descriptor.get("descriptor_type") != "simulation":
        print("Visualization only supports simulation descriptors.")
        return 1

    root_dir = descriptor.get("root_dir")
    experiment_name = descriptor.get("experiment")
    if not root_dir or not experiment_name:
        print("Descriptor must include 'root_dir' and 'experiment'.")
        return 1

    stats_path = Path(root_dir) / "simulations" / experiment_name / "collected_stats.csv"
    if not stats_path.is_file():
        print(f"No collected stats found at {stats_path}. Attempting to collect now...")
        if not collect_stats_for_visualization(descriptor_path, stats_path):
            print("Automatic stat collection failed; run the stat collector manually.")
            return 1
        if not stats_path.is_file():
            print("Stat collection completed, but stats file is still missing.")
            return 1
        print("Stat collection completed successfully.")

    try:
        import scarab_stats
    except ImportError as exc:
        print(f"Failed to import scarab_stats: {exc}")
        return 1

    try:
        importlib.reload(scarab_stats)
        from scarab_stats import stat_aggregator
    except ImportError as exc:
        print(f"Failed to access stat_aggregator: {exc}")
        return 1

    def load_experiment():
        agg = stat_aggregator()
        try:
            exp = agg.load_experiment_csv(str(stats_path))
        except Exception as exc:  # pragma: no cover - depends on user data
            print(f"Failed to load stats from {stats_path}: {exc}")
            return None, None
        return agg, exp

    aggregator, experiment = load_experiment()
    if not aggregator or not experiment:
        return 1

    try:
        workloads = sorted(experiment.get_workloads())
        configs = sorted(experiment.get_configurations())
    except Exception as exc:  # pragma: no cover - depends on user data
        print(f"Failed to inspect experiment data: {exc}")
        return 1

    expected_workloads, expected_configs = extract_descriptor_expectations(descriptor)
    missing_workloads = expected_workloads - set(workloads)
    missing_configs = expected_configs - set(configs)

    if (missing_workloads or missing_configs) and (expected_workloads or expected_configs):
        if missing_workloads:
            print(
                f"Stats file missing workloads defined in descriptor: {', '.join(sorted(missing_workloads))}"
            )
        if missing_configs:
            print(
                f"Stats file missing configurations defined in descriptor: {', '.join(sorted(missing_configs))}"
            )
        if collect_stats_for_visualization(descriptor_path, stats_path):
            aggregator, experiment = load_experiment()
            if not aggregator or not experiment:
                return 1
            try:
                workloads = sorted(experiment.get_workloads())
                configs = sorted(experiment.get_configurations())
            except Exception as exc:  # pragma: no cover - depends on user data
                print(f"Failed to inspect experiment data: {exc}")
                return 1

            remaining_workloads = expected_workloads - set(workloads)
            remaining_configs = expected_configs - set(configs)
            if remaining_workloads or remaining_configs:
                print("Refreshed stats still missing descriptor entries; continuing with available data.")
        else:
            print("Unable to refresh stats automatically; continuing with existing data.")

    if expected_workloads:
        workloads_set = set(workloads)
        filtered_workloads = sorted(workloads_set & expected_workloads)
        if not filtered_workloads:
            print("None of the workloads specified in the descriptor are present in the stats file.")
            return 1
        extra_workloads = workloads_set - expected_workloads
        if extra_workloads:
            print(
                "Ignoring workloads not listed in the descriptor: "
                + ", ".join(sorted(extra_workloads))
            )
        workloads = filtered_workloads

    if expected_configs:
        configs_set = set(configs)
        filtered_configs = sorted(configs_set & expected_configs)
        if not filtered_configs:
            print("None of the configurations specified in the descriptor are present in the stats file.")
            return 1
        extra_configs = configs_set - expected_configs
        if extra_configs:
            print(
                "Ignoring configurations not listed in the descriptor: "
                + ", ".join(sorted(extra_configs))
            )
        configs = filtered_configs

    if not workloads:
        print("No workloads found in the stats file.")
        return 1
    if not configs:
        print("No configurations found in the stats file.")
        return 1

    stats_to_plot = descriptor.get("visualize_counters") or ["Periodic_IPC"]
    if not isinstance(stats_to_plot, list) or not stats_to_plot:
        print("Descriptor field 'visualize_counters' must be a non-empty list when provided.")
        return 1

    def normalize_counter_entry(entry: object) -> Optional[Dict[str, object]]:
        """
        Convert visualize_counters entries into a normalized plot request.

        Supported formats:
        - \"stat_name\"
        - [\"stat_a\", \"stat_b\", ...] (stacked when length > 1)
        - {\"stats\": [...], \"type\": \"stacked\"} (optional \"title\", \"y_label\", \"name\")
        """
        if isinstance(entry, str):
            return {"type": "single", "stats": [entry]}
        if isinstance(entry, (list, tuple, set)):
            stats = [str(stat) for stat in entry if stat is not None]
            if not stats:
                return None
            plot_type = "stacked" if len(stats) > 1 else "single"
            return {"type": plot_type, "stats": stats}
        if isinstance(entry, dict):
            raw_stats = entry.get("stats")
            if raw_stats is None:
                raw_stats = entry.get("stacked")
            if raw_stats is None:
                print("Skipping visualize entry without 'stats' field:", entry)
                return None
            if isinstance(raw_stats, str):
                stats = [raw_stats]
            else:
                stats = [str(stat) for stat in raw_stats if stat is not None]
            if not stats:
                return None
            plot_type = entry.get("type") or entry.get("mode")
            if plot_type not in {"single", "stacked"}:
                plot_type = "stacked" if len(stats) > 1 else "single"
            request: Dict[str, object] = {"type": plot_type, "stats": stats}
            if "title" in entry:
                request["title"] = str(entry["title"])
            if "y_label" in entry:
                request["y_label"] = str(entry["y_label"])
            if "name" in entry:
                request["name"] = str(entry["name"])
            return request
        print(f"Skipping unsupported visualize entry type: {entry!r}")
        return None

    plot_requests: List[Dict[str, object]] = []
    for raw_entry in stats_to_plot:
        normalized = normalize_counter_entry(raw_entry)
        if not normalized:
            continue
        plot_requests.append(normalized)

    if not plot_requests:
        print("No valid entries found in 'visualize_counters'.")
        return 1

    baseline: Optional[str] = None
    raw_baseline = descriptor.get("visualize_baseline")
    if isinstance(raw_baseline, str):
        candidate = raw_baseline.strip()
        if candidate:
            if candidate in configs:
                baseline = candidate
            else:
                print(
                    f"Configured visualize_baseline '{candidate}' is not present in the stats file; "
                    "defaulting to the first available configuration."
                )
        else:
            print("Descriptor field 'visualize_baseline' is empty; defaulting to the first configuration.")
    elif raw_baseline is not None:
        print(
            "Descriptor field 'visualize_baseline' must be a string when provided; "
            "defaulting to the first configuration."
        )
    if baseline is None:
        baseline = configs[0]

    def safe_filename(stem: str) -> str:
        return "".join(c if c.isalnum() or c in {"-", "_"} else "_" for c in stem)

    print(f"Using stats file: {stats_path}")
    print(f"Workloads: {len(workloads)}; Configurations: {len(configs)}; Speedup baseline: {baseline}")

    available_stats = set()
    try:
        available_stats = set(experiment.get_stats())
    except Exception:
        pass

    for request in plot_requests:
        stats_list = [str(stat) for stat in request["stats"]]  # type: ignore[index]
        missing_stats: Set[str] = set()
        if available_stats:
            missing_stats = {stat for stat in stats_list if stat not in available_stats}
        if missing_stats:
            print(
                f"Skipping {stats_list}: not present in collected stats "
                f"({', '.join(sorted(missing_stats))})."
            )
            continue

        plot_type = request["type"]  # type: ignore[index]
        custom_stem = request.get("name") if isinstance(request, dict) else None
        if not custom_stem:
            custom_stem = "_".join(stats_list)

        if plot_type == "stacked" and len(stats_list) > 1:
            stem_safe = safe_filename(str(custom_stem))
            stacked_output = stats_path.with_name(f"{stem_safe}_stacked.png")
            title = str(request.get("title", ""))  # type: ignore[call-arg]
            y_label = str(request.get("y_label", "Value"))  # type: ignore[call-arg]
            print(f"Plotting stacked {stats_list} → {stacked_output.name}")
            try:
                aggregator.plot_stacked(
                    experiment,
                    stats_list,
                    workloads,
                    configs,
                    title=title,
                    y_label=y_label,
                    plot_name=str(stacked_output),
                )
            except Exception as exc:  # pragma: no cover - matplotlib backend dependent
                print(f"Failed to generate stacked plot for {stats_list}: {exc}")
            continue

        stat_name = stats_list[0]
        stat_safe = safe_filename(stat_name)
        baseline_safe = safe_filename(baseline)
        ipc_output = stats_path.with_name(f"{stat_safe}_ipc.png")
        speedup_output = stats_path.with_name(f"{stat_safe}_speedup_vs_{baseline_safe}.png")

        print(f"Plotting '{stat_name}' → {ipc_output.name}, {speedup_output.name}")

        try:
            aggregator.plot_workloads(
                experiment,
                [stat_name],
                workloads,
                configs,
                title=str(request.get("title", "")),  # type: ignore[call-arg]
                y_label=str(request.get("y_label", stat_name)),  # type: ignore[call-arg]
                x_label="Workloads",
                average=True,
                plot_name=str(ipc_output),
            )
            aggregator.plot_workloads_speedup(
                experiment,
                [stat_name],
                workloads,
                configs,
                speedup_baseline=baseline,
                title="",
                y_label=f"{stat_name} Speedup (%)",
                x_label="Workloads",
                average=True,
                plot_name=str(speedup_output),
            )
        except Exception as exc:  # pragma: no cover - matplotlib backend dependent
            print(f"Failed to generate plots for '{stat_name}': {exc}")

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
        "--visualize",
        metavar="DESCRIPTOR",
        help="Plot IPC and speedup for collected stats in json/<DESCRIPTOR>.json.",
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
    parser.add_argument(
        "--debug-level",
        dest="debug_level",
        type=int,
        choices=[1, 2, 3],
        help="Override descriptor command verbosity (1=errors, 2=warnings, 3=info).",
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
        bool(args.visualize),
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
        args.visualize,
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
            handle_descriptor_action(args.interactive, "launch", dbg_override=args.debug_level)
            return 0
        except (StepError, RuntimeError) as exc:
            print(exc)
            return 1
    if args.trace:
        try:
            handle_descriptor_action(args.trace, "trace", dbg_override=args.debug_level)
            return 0
        except (StepError, RuntimeError) as exc:
            print(exc)
            return 1
    if args.sim:
        try:
            handle_descriptor_action(args.sim, "simulate", dbg_override=args.debug_level)
            return 0
        except (StepError, RuntimeError) as exc:
            print(exc)
            return 1
    if args.visualize:
        return run_visualize(args.visualize)
    if args.kill:
        try:
            handle_descriptor_action(args.kill, "kill", dbg_override=args.debug_level)
            return 0
        except (StepError, RuntimeError) as exc:
            print(exc)
            return 1
    if args.status:
        try:
            handle_descriptor_action(args.status, "info", dbg_override=args.debug_level)
            return 0
        except (StepError, RuntimeError) as exc:
            print(exc)
            return 1
    if args.clean:
        try:
            handle_descriptor_action(args.clean, "clean", dbg_override=args.debug_level)
            return 0
        except (StepError, RuntimeError) as exc:
            print(exc)
            return 1
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())

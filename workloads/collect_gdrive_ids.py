#!/usr/bin/env python3
"""Populate ``drive_id`` and ``size_bytes`` fields in workloads_db.json using the Google Drive API."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from urllib.error import HTTPError

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
TRACE_SUFFIX = "traces/simp"
DEVICE_CODE_ENDPOINT = "https://oauth2.googleapis.com/device/code"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"


class CollectError(RuntimeError):
    """Raised when Drive ID collection cannot be completed."""


def debug(message: str) -> None:
    print(message)


def load_client_info(credentials_path: Path) -> Dict[str, str]:
    try:
        config = json.loads(credentials_path.read_text())
    except json.JSONDecodeError as exc:
        raise CollectError(f"Invalid credentials JSON: {exc}") from exc
    for key in ("installed", "web"):
        client_info = config.get(key)
        if client_info:
            break
    else:  # pragma: no cover - unexpected structure
        raise CollectError("Credentials file missing 'installed' configuration block.")
    required = {"client_id"}
    missing = required - client_info.keys()
    if missing:
        raise CollectError(f"Credentials file missing fields: {sorted(missing)}")
    return client_info


def run_console_oauth(flow: InstalledAppFlow) -> Credentials:
    auth_url, _ = flow.authorization_url(prompt="consent", access_type="offline")
    print("\nOpen the following URL in any browser:\n")
    print(auth_url)
    print()
    code = input("Enter the authorization code: ").strip()
    if not code:
        raise CollectError("No authorization code provided for console OAuth flow.")
    flow.fetch_token(code=code)
    return flow.credentials


def run_device_oauth(client_info: Dict[str, str]) -> Credentials:
    payload = urllib.parse.urlencode(
        {
            "client_id": client_info["client_id"],
            "scope": " ".join(SCOPES),
        }
    ).encode()
    try:
        with urllib.request.urlopen(DEVICE_CODE_ENDPOINT, data=payload) as resp:
            device_data = json.loads(resp.read().decode())
    except HTTPError as exc:  # pragma: no cover - network error handling
        detail = exc.read().decode() if hasattr(exc, "read") else ""
        raise CollectError(f"Device authorization request failed: {exc} {detail}") from exc

    verification_url = device_data.get("verification_url") or device_data.get("verification_uri")
    user_code = device_data["user_code"]
    device_code = device_data["device_code"]
    interval = int(device_data.get("interval", 5))

    print("\nFollow the steps below to authorize access:")
    print(f"  1. Open: {verification_url}")
    print(f"  2. Enter code: {user_code}")
    print("  3. Approve the requested permissions.\n")

    token_payload = {
        "client_id": client_info["client_id"],
        "device_code": device_code,
        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
    }
    if "client_secret" in client_info:
        token_payload["client_secret"] = client_info["client_secret"]

    while True:
        time.sleep(interval)
        try:
            with urllib.request.urlopen(
                TOKEN_ENDPOINT,
                data=urllib.parse.urlencode(token_payload).encode(),
            ) as resp:
                token_data = json.loads(resp.read().decode())
        except HTTPError as exc:  # pragma: no cover
            detail = exc.read().decode() if hasattr(exc, "read") else ""
            raise CollectError(f"Token polling failed: {exc} {detail}") from exc

        if "error" in token_data:
            error = token_data["error"]
            if error == "authorization_pending":
                continue
            if error == "slow_down":
                interval += 5
                continue
            if error == "access_denied":
                raise CollectError("Device authorization denied by user.")
            raise CollectError(f"Device authorization failed: {error}")

        expires_in = int(token_data.get("expires_in", 0))
        creds = Credentials(
            token=token_data.get("access_token"),
            refresh_token=token_data.get("refresh_token"),
            token_uri=TOKEN_ENDPOINT,
            client_id=client_info.get("client_id"),
            client_secret=client_info.get("client_secret"),
            scopes=SCOPES,
        )
        if expires_in:
            creds.expiry = _dt.datetime.utcnow() + _dt.timedelta(seconds=expires_in)
        return creds


def load_credentials(credentials_path: Path, auth_mode: str) -> Credentials:
    token_path = credentials_path.with_name("token.json")
    creds: Optional[Credentials] = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            client_info = load_client_info(credentials_path)
            flow = InstalledAppFlow.from_client_config({"installed": client_info}, SCOPES)
            if auth_mode == "device":
                debug("Starting OAuth device authorization flow…")
                creds = run_device_oauth(client_info)
            elif auth_mode == "console":
                debug("Starting OAuth console flow…")
                creds = run_console_oauth(flow)
            else:
                try:
                    creds = flow.run_local_server(port=0)
                except Exception:
                    debug("Browser-based auth failed; retrying with console OAuth flow.")
                    creds = run_console_oauth(flow)
            if auth_mode == "device" and not creds:
                raise CollectError("Device flow did not return credentials.")
        if not creds:
            raise CollectError("Failed to obtain Google Drive credentials.")
        token_path.write_text(creds.to_json())
    return creds


def build_drive_service(credentials_path: Path, auth_mode: str) -> any:
    creds = load_credentials(credentials_path, auth_mode)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_children(service, folder_id: str) -> Iterable[Dict[str, Any]]:
    query = f"'{folder_id}' in parents and trashed = false"
    page_token = None
    while True:
        response = (
            service.files()
            .list(
                q=query,
                fields="nextPageToken, files(id, name, mimeType, size)",
                pageSize=1000,
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            .execute()
        )
        for item in response.get("files", []):
            yield item
        page_token = response.get("nextPageToken")
        if not page_token:
            break


def crawl_drive_tree(service, root_folder_id: str) -> Dict[str, Dict[str, Any]]:
    mapping: Dict[str, Dict[str, Any]] = {}
    queue: Deque[Tuple[str, str]] = deque([(root_folder_id, "")])

    while queue:
        folder_id, prefix = queue.popleft()
        for item in list_children(service, folder_id):
            name = item["name"]
            path = f"{prefix}/{name}" if prefix else name
            if item["mimeType"] == "application/vnd.google-apps.folder":
                queue.append((item["id"], path))
            else:
                size_value: Optional[int]
                try:
                    size_value = int(item.get("size")) if item.get("size") is not None else None
                except (TypeError, ValueError):
                    size_value = None
                mapping[path] = {"id": item["id"], "size": size_value}
    return mapping


def load_workloads(path: Path) -> Dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise CollectError(f"Workloads file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CollectError(f"Failed to parse JSON at {path}: {exc}") from exc


def update_workloads_with_ids(workloads: Dict, drive_map: Dict[str, Dict[str, Any]]) -> int:
    updated = 0
    for suite, subsuites in workloads.items():
        if not isinstance(subsuites, dict):
            continue
        for subsuite, workloads_dict in subsuites.items():
            if not isinstance(workloads_dict, dict):
                continue
            for workload, payload in workloads_dict.items():
                if not isinstance(payload, dict):
                    continue
                simpoints = payload.get("simpoints", [])
                if not isinstance(simpoints, list):
                    continue
                for sim in simpoints:
                    if not isinstance(sim, dict):
                        continue
                    cluster_id = sim.get("cluster_id")
                    if cluster_id is None:
                        continue
                    drive_path = f"{suite}/{subsuite}/{workload}/{TRACE_SUFFIX}/{cluster_id}.zip"
                    mapping_entry = drive_map.get(drive_path)
                    if not mapping_entry:
                        continue
                    file_id = mapping_entry.get("id")
                    size_bytes = mapping_entry.get("size")
                    if file_id and sim.get("drive_id") != file_id:
                        sim["drive_id"] = file_id
                        updated += 1
                    if size_bytes is not None and sim.get("size_bytes") != size_bytes:
                        sim["size_bytes"] = size_bytes
    return updated


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate drive_id and size metadata in workloads_db.json")
    parser.add_argument("--folder-id", required=True, help="Google Drive folder ID containing suite subdirectories")
    parser.add_argument("--credentials", default="credentials.json", help="Path to OAuth client credentials JSON")
    parser.add_argument("--workloads", default="workloads/workloads_db.json", help="Input workloads DB JSON file")
    parser.add_argument("--output", default="workloads/workloads_db_with_ids.json", help="Destination for the updated JSON")
    auth_group = parser.add_mutually_exclusive_group()
    auth_group.add_argument("--auth-console", action="store_true", help="Use manual console-based OAuth flow")
    auth_group.add_argument("--auth-device", action="store_true", help="Use OAuth device authorization flow (requires TV/limited-input credentials)")
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    credentials_path = Path(args.credentials)
    if not credentials_path.exists():
        raise CollectError(f"Credentials file not found: {credentials_path}")

    if args.auth_device:
        auth_mode = "device"
    elif args.auth_console:
        auth_mode = "console"
    else:
        auth_mode = "browser"

    debug("Connecting to Google Drive API…")
    service = build_drive_service(credentials_path, auth_mode)

    debug("Crawling Drive folder hierarchy…")
    drive_map = crawl_drive_tree(service, args.folder_id)
    debug(f"Discovered {len(drive_map)} files.")

    workloads_path = Path(args.workloads)
    workloads = load_workloads(workloads_path)

    debug("Mapping drive IDs onto workloads…")
    updated = update_workloads_with_ids(workloads, drive_map)
    debug(f"Updated {updated} simpoint entries with drive_id values.")

    output_path = Path(args.output)
    output_path.write_text(json.dumps(workloads, indent=2, separators=(",", ":")))
    debug(f"Wrote updated workloads JSON to {output_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except CollectError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

"""
Convenience Launcher for Rainfall Analysis Dashboard Project.
Checks environment, validates datasets, starts Flask server,
verifies HTTP health, and launches Google Chrome/default browser.
"""

import os
import sys
import time
import socket
import signal
import webbrowser
import subprocess
import threading
import urllib.request
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from config import SERVER_HOST, SERVER_PORT, DISTRICT_CSV_PATH, SUBDIVISION_CSV_PATH
from app import app
import data_loader
from werkzeug.serving import make_server


def find_available_port(host: str = "127.0.0.1", default_port: int = 5000) -> int:
    """Checks port availability and finds an open TCP port starting with default_port."""
    for port in [default_port] + list(range(5001, 5025)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue

    # Fallback to ephemeral port allocated by OS
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def wait_for_server_ready(url: str, timeout_seconds: float = 6.0) -> bool:
    """Polls the /api/health endpoint to confirm server is actively responding."""
    start_time = time.time()
    health_url = f"{url}/api/health"
    while time.time() - start_time < timeout_seconds:
        try:
            req = urllib.request.Request(health_url, headers={"User-Agent": "HealthCheck"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.1)
    return False


def launch_chrome_or_browser(url: str) -> bool:
    """
    Launches Google Chrome directly if available, with fallbacks to system default browser.
    Non-blocking execution ensures it never hangs the launcher.
    """
    chrome_candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        os.path.expandvars(r"%PROGRAMFILES%\Google\Chrome\Application\chrome.exe"),
    ]

    # 1. Attempt launching Google Chrome directly
    for candidate in chrome_candidates:
        if os.path.exists(candidate):
            try:
                subprocess.Popen(
                    [candidate, url],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"[✓] Successfully opened dashboard in Google Chrome!", flush=True)
                return True
            except Exception as e:
                print(f"[!] Note on Chrome launch: {e}", flush=True)
                break

    # 2. Fallback to Windows Shell Association
    try:
        subprocess.Popen(
            ["cmd", "/c", "start", "", url],
            shell=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"[✓] Successfully opened dashboard via Windows default browser!", flush=True)
        return True
    except Exception:
        pass

    # 3. Fallback to standard webbrowser module
    try:
        webbrowser.open(url)
        print(f"[✓] Opened dashboard via webbrowser module!", flush=True)
        return True
    except Exception as e:
        print(f"[!] Automatic browser open failed ({e}). Please open URL manually.", flush=True)
        return False


def main():
    print("=" * 72, flush=True)
    print("   [+] HYDROCLIM INDIA: RAINFALL & MONSOON CLIMATE INTELLIGENCE", flush=True)
    print("=" * 72, flush=True)

    # 1. Validate original source CSV files exist without modifying them
    if not SUBDIVISION_CSV_PATH.exists():
        print(f"[!] Error: Missing subdivision CSV at {SUBDIVISION_CSV_PATH}", flush=True)
        sys.exit(1)
    if not DISTRICT_CSV_PATH.exists():
        print(f"[!] Error: Missing district CSV at {DISTRICT_CSV_PATH}", flush=True)
        sys.exit(1)

    print(f"[✓] Verified source dataset: {SUBDIVISION_CSV_PATH.name}", flush=True)
    print(f"[✓] Verified source dataset: {DISTRICT_CSV_PATH.name}", flush=True)

    # 2. Pre-warm data caches in memory
    print("[+] Pre-warming in-memory datasets...", end=" ", flush=True)
    data_loader.get_metadata()
    data_loader.load_subdivision_data()
    data_loader.load_district_data()
    print("Done! (Cache Ready)", flush=True)

    # 3. Select open port
    target_port = find_available_port(SERVER_HOST, SERVER_PORT)
    if target_port != SERVER_PORT:
        print(f"[!] Port {SERVER_PORT} is in use; automatically selected port {target_port}.", flush=True)
    else:
        print(f"[✓] Port {target_port} is available.", flush=True)

    url = f"http://{SERVER_HOST}:{target_port}"

    # 4. Start Werkzeug multithreaded server on a background thread
    server = make_server(SERVER_HOST, target_port, app, threaded=True)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # 5. Verify server readiness via HTTP health check
    print(f"[+] Verifying server readiness at {url}...", end=" ", flush=True)
    if not wait_for_server_ready(url, timeout_seconds=5.0):
        print("Failed!", flush=True)
        print("[!] Server did not respond to health check in time.", flush=True)
        server.shutdown()
        sys.exit(1)
    print("Server is Ready (HTTP 200)!", flush=True)

    # 6. Launch browser directly
    launch_chrome_or_browser(url)

    # 7. Print exact localhost URL prominently
    print("\n" + "=" * 72, flush=True)
    print("   [★] DASHBOARD SERVER ACTIVE AND READY!", flush=True)
    print(f"   [★] Open this exact URL in Chrome: {url}", flush=True)
    print("=" * 72, flush=True)
    print("[+] Press Ctrl+C to stop the server.\n", flush=True)

    # 8. Handle shutdown gracefully
    def shutdown_handler(signum, frame):
        print("\n[+] Gracefully shutting down dashboard server...", flush=True)
        server.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown_handler(None, None)


if __name__ == "__main__":
    main()

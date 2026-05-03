import os
import sys
import time
import json
import socket
import sqlite3
import subprocess
import threading
import runpy
import multiprocessing
from paths import _path, BASE_DIR

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

SYSTEM_NAME = "Hearth AI - Home Care Monitoring System"

_SEV_TAGS = {'normal': '[OK]', 'warning': '[!]', 'danger': '[!!]'}


def _banner(subtitle=None):

    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"   {SYSTEM_NAME}")
    print(f"   {subtitle or f'Version: {VERSION}'}")


# pipe subprocess stdout to our terminal in real time
def _drain_output(proc):
    try:
        for line in iter(proc.stdout.readline, b''):
            sys.stdout.write(line.decode('utf-8', errors='replace'))
            sys.stdout.flush()
    except Exception:
        pass


def _launch(label, script, *, pipe=True, fatal=False, extra_env=None):

    print(label)
    try:
        env = {**os.environ}
        if pipe:
            env['PYTHONUNBUFFERED'] = '1'
        if extra_env:
            env.update(extra_env)
        final_env = env if (pipe or extra_env) else None
        args = [sys.executable] + (['-u'] if pipe else []) + [_path(script)]
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE if pipe else subprocess.DEVNULL,
            stderr=subprocess.STDOUT if pipe else subprocess.DEVNULL,
            env=final_env,
        )
        if pipe:
            threading.Thread(
                target=_drain_output, args=(proc,), daemon=True).start()
        print(f"      [OK] PID {proc.pid}")
        return proc
    except Exception as e:
        print(f"      [ERROR] {e}")
        if fatal:
            sys.exit(1)
        return None


# kill anything squatting on our port from a previous crash
def _free_port(port: int):
    try:
        result = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True
        )
        pids = result.stdout.strip().split()
        for pid in pids:
            try:
                subprocess.run(["kill", "-9", pid], check=False)
            except Exception:
                pass
        if pids:
            time.sleep(0.5)
    except Exception:
        pass


def _wait_for_server(host, port, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                print(f"      [OK] Server accepting connections on {host}:{port}")
                return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.5)
    return False


def main():
    _banner(f"Version {VERSION}")
    print("  Hearth AI — Web Dashboard Launcher")
    print("  " + "=" * 44)
    print()

    try:
        from ui.gui import run_gui_in_thread
    except ImportError as e:
        print(f"[ERROR] Could not import ui.gui: {e}")
        print("       Ensure Flask is installed: pip install flask")
        input("Press Enter to exit...")
        sys.exit(1)

    from model.engine import CHECKPOINT_PATH as _CKPT_PATH

    # no checkpoint found — offer inline training before proceeding
    if not os.path.exists(_CKPT_PATH):
        print("[INFO] No trained model found.")
        print()
        print("  [T] Train model now")
        print("  [Q] Quit")
        print()
        try:
            choice = input("Choice: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            return

        if choice != 'T':
            return

        # collect training parameters
        try:
            p_raw = input("\nNumber of patients  (default 200): ").strip()
            tr_patients = int(p_raw) if p_raw else 200
            tr_patients = max(1, min(tr_patients, 10000))

            d_raw = input("Number of days      (default 60):  ").strip()
            tr_days = int(d_raw) if d_raw else 60
            tr_days = max(1, min(tr_days, 730))

            r_raw = input("Sensor reads/hour   (default 12):  ").strip()
            tr_rph = int(r_raw) if r_raw else 12
            tr_rph = max(1, min(tr_rph, 60))
        except (ValueError, EOFError, KeyboardInterrupt):
            print("[ERROR] Invalid input.")
            return

        print()
        print(f"[1/2] Generating training data ({tr_patients} patients, {tr_days} days, {tr_rph} reads/hr)...")
        try:
            from data.generator import generate_to_db
            generate_to_db(
                num_patients=tr_patients,
                num_days=tr_days,
                readings_per_hour=tr_rph,
            )
        except Exception as e:
            print(f"[ERROR] Data generation failed: {e}")
            return

        print("[2/2] Training TabNet model...")
        try:
            from model.engine import get_engine
            get_engine().train_from_db()
        except Exception as e:
            print(f"[ERROR] Training failed: {e}")
            return

        print("[OK] Model trained successfully.\n")

    try:
        n_raw = input("\nLive patients  [1-500]  (default 50): ").strip()
        n_patients = int(n_raw) if n_raw else 50
        n_patients = max(1, min(n_patients, 500))

        t_raw = input("Tick rate (s)  [0.5-30] (default 2.0): ").strip()
        tick  = float(t_raw) if t_raw else 2.0
        tick  = max(0.5, min(tick, 30.0))

        p_raw = input("Dashboard port          (default 8050): ").strip()
        port  = int(p_raw) if p_raw else 8050
    except (ValueError, EOFError, KeyboardInterrupt):
        print("[ERROR] Invalid input.")
        return

    live_env = {
        'LIVE_N_PATIENTS':   str(n_patients),
        'LIVE_TICK_SECONDS': f"{tick:.1f}",
        'HEARTH_LIVE_MODE':  '1',
    }

    server = _launch("[1/3] Launching AI Server...", os.path.join("server", "ai_server.py"),
                     fatal=True, extra_env=live_env)
    from config import SERVER_PORT
    if server is None or not _wait_for_server('127.0.0.1', SERVER_PORT):
        print("[ERROR] AI Server did not become ready within 60 s. Aborting.")
        if server is not None:
            server.terminate()
        return

    sim = _launch("[2/3] Starting Live Simulator...",
                  os.path.join("iot", "simulator.py"), extra_env=live_env)

    print("[3/3] Starting Web Dashboard...")
    _free_port(port)
    run_gui_in_thread(port=port, open_browser=True)

    print(f" HEARTH AI LIVE + DASHBOARD — {n_patients} patients @ {tick:.1f}s/tick")
    print(f" Dashboard: http://localhost:{port}  |  Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
            if server is not None and server.poll() is not None:
                print("\n[ERROR] AI Server stopped unexpectedly.")
                break
            if sim is not None and sim.poll() is not None:
                print("\n[ERROR] Live Simulator exited unexpectedly (code %d)." % sim.returncode)
                break
    except KeyboardInterrupt:
        print("\n\nStopping Dashboard + Live Monitoring...")
    finally:
        for name, proc in [("Live Simulator", sim), ("AI Server", server)]:
            if proc and proc.poll() is None:
                proc.terminate()
                print(f"[OK] {name} stopped.")
        print("[INFO] Dashboard halted.")
        time.sleep(1)


if __name__ == "__main__":
    multiprocessing.freeze_support()

    if getattr(sys, 'frozen', False) and len(sys.argv) > 1:
        target_script = sys.argv[-1]
        if target_script.endswith(('ai_server.py', 'simulator.py')):
            runpy.run_path(target_script, run_name="__main__")
            sys.exit(0)
    # -------------------------------------

    main()
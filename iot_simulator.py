
import asyncio
import json
import os
import random
import struct
import sys
from datetime import datetime
from typing import List, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

HOST         = "127.0.0.1"
PORT         = 65432
LIVE_MODE    = os.environ.get("HEARTH_LIVE_MODE", "0") == "1"
N_PATIENTS   = int(os.environ.get("LIVE_N_PATIENTS", "50"))
TICK_SECONDS = float(os.environ.get("LIVE_TICK_SECONDS", "2.0"))
SECONDS_PER_DAY = 1.0



class AsyncTCPClient:

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self._connected = False

    async def connect(self, retries: int = 5, delay: float = 2.0) -> bool:
        for attempt in range(retries):
            try:
                self.reader, self.writer = await asyncio.open_connection(
                    self.host, self.port
                )
                self._connected = True
                mode = "LIVE" if LIVE_MODE else "SIMULATOR"
                print(f"[{mode}] Connected to AI Server at {self.host}:{self.port}")
                return True
            except (ConnectionRefusedError, OSError) as e:
                if attempt < retries - 1:
                    print(f"[CONN] Connection refused (attempt {attempt+1}/{retries}). "
                          f"Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    print(f"[CONN] AI Server unreachable after {retries} attempts: {e}")
                    return False
        return False

    async def send_framed(self, data: bytes) -> bool:
        if not self._connected or self.writer is None:
            return False
        try:
            frame = struct.pack(">I", len(data)) + data
            self.writer.write(frame)
            await self.writer.drain()
            return True
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            print(f"[CONN] Send failed: {e}")
            self._connected = False
            return False

    async def reconnect(self) -> bool:
        await self.close()
        print("[CONN] Attempting reconnection...")
        return await self.connect(retries=3, delay=1.0)

    async def close(self):
        if self.writer:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass
        self._connected = False
        self.reader = None
        self.writer = None



def _nan_to_none(v):
    if isinstance(v, float) and v != v:
        return None
    return v


class Patient:

    def __init__(self, patient_id: int):
        from data_generator import (
            _VITALS, _PROFILE_WEIGHTS, _base_vitals,
            _healthy_reading, _unhealthy_reading, _critical_reading,
            inject_sensor_dropout,
        )
        self._VITALS = _VITALS
        self._base_vitals = _base_vitals
        self._healthy_reading = _healthy_reading
        self._unhealthy_reading = _unhealthy_reading
        self._critical_reading = _critical_reading
        self._inject_sensor_dropout = inject_sensor_dropout

        _PROFILES, _WEIGHTS = zip(*_PROFILE_WEIGHTS)
        self.patient_id = patient_id
        self.profile = random.choices(_PROFILES, weights=_WEIGHTS, k=1)[0]
        self.base = self._base_vitals(self.profile)
        self.noise_rng = random.Random(patient_id * 31337)
        self.crit_left = 0

    def next_reading(self) -> dict:
        v = self._VITALS[self.profile]
        hour = datetime.now().hour

        if self.crit_left > 0:
            raw = self._critical_reading(self.base)
            self.crit_left -= 1
        elif random.random() < v["critical_prob"]:
            raw = self._critical_reading(self.base)
            self.crit_left = random.randint(1, 3)
        elif random.random() < v["unhealthy_prob"]:
            raw = self._unhealthy_reading(self.profile, self.base)
        else:
            raw = self._healthy_reading(self.profile, self.base, hour)

        raw = self._inject_sensor_dropout(raw, self.noise_rng)

        return {
            "patient_id":   self.patient_id,
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "heart_rate":   _nan_to_none(raw["hr"]),
            "systolic_bp":  raw["sbp"],
            "diastolic_bp": raw["dbp"],
            "body_temp":    _nan_to_none(raw["temp"]),
            "spo2":         _nan_to_none(raw["spo2"]),
            "activity":     raw["activity"],
        }


async def broadcast_live(client: AsyncTCPClient, patients: List[Patient]):
    tick = 0
    while True:
        tick += 1
        start = asyncio.get_event_loop().time()
        sim_date = datetime.now().strftime("%Y-%m-%d")

        readings = [p.next_reading() for p in patients]
        payload = {"sim_date": sim_date, "readings": readings}
        data_bytes = json.dumps(payload).encode("utf-8")

        success = await client.send_framed(data_bytes)
        if not success:
            if await client.reconnect():
                success = await client.send_framed(data_bytes)
            if not success:
                print(f"[LIVE] Tick {tick}: send failed — skipping")
                await asyncio.sleep(TICK_SECONDS)
                continue

        print(f"[TICK {tick:>4}] {datetime.now().strftime('%H:%M:%S')} | "
              f"{len(readings)} patients | tick={TICK_SECONDS:.1f}s")

        elapsed = asyncio.get_event_loop().time() - start
        await asyncio.sleep(max(0.0, TICK_SECONDS - elapsed))


async def run_live_mode():
    from data_generator import _PROFILE_WEIGHTS
    _PROFILES, _ = zip(*_PROFILE_WEIGHTS)

    print("=" * 60)
    print("   Hearth AI — Live Real-Time Simulator")
    print("=" * 60)
    print(f"\n[LIVE] Initialising {N_PATIENTS} patients...")

    patients = [Patient(i + 1) for i in range(N_PATIENTS)]

    profile_counts = {}
    for p in patients:
        profile_counts[p.profile] = profile_counts.get(p.profile, 0) + 1
    for prof, count in sorted(profile_counts.items()):
        print(f"       {prof:<12} {count} patients")

    print(f"\n[LIVE] Tick rate : {TICK_SECONDS:.1f}s per broadcast")
    print(f"[LIVE] Target    : {HOST}:{PORT}")
    print()

    client = AsyncTCPClient(HOST, PORT)
    if not await client.connect():
        sys.exit(1)

    print("[LIVE] Streaming started — press Ctrl+C to stop.\n")
    try:
        await broadcast_live(client, patients)
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("\n[LIVE] Stopped by user.")
    except Exception as e:
        print(f"\n[LIVE] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.close()
        print("[LIVE] Connection closed.")



async def broadcast_replay(client: AsyncTCPClient, dates: List[str]):
    from data_logger import get_last_reading_per_patient_for_date

    total_days = len(dates)
    for index, sim_date in enumerate(dates):
        start_time = asyncio.get_event_loop().time()

        readings = get_last_reading_per_patient_for_date(sim_date)
        if not readings:
            print(f"[DAY {index+1}/{total_days}] {sim_date} | No readings (skipped)")
            continue

        payload = {"sim_date": sim_date, "readings": readings}
        data_bytes = json.dumps(payload).encode("utf-8")

        success = await client.send_framed(data_bytes)
        if not success:
            if await client.reconnect():
                success = await client.send_framed(data_bytes)
            if not success:
                print(f"[SIMULATOR] Failed to send day {sim_date} after reconnect")
                continue

        print(f"[DAY {index+1}/{total_days}] {sim_date} | "
              f"Sent {len(readings):,} patients (end-of-day snapshot)")

        elapsed = asyncio.get_event_loop().time() - start_time
        await asyncio.sleep(max(0.0, SECONDS_PER_DAY - elapsed))


async def run_replay_mode():
    from data_logger import get_dates_available

    print("=" * 60)
    print("   Hearth AI — Async IoT Edge Simulator (Replay)")
    print("=" * 60)
    print()

    dates = get_dates_available()
    if not dates:
        print("[SIMULATOR] Error: No sensor data found in database.")
        print("            Run data_generator.py to seed the database first.")
        sys.exit(1)

    sim_start = os.environ.get("SIM_START_DATE")
    sim_end = os.environ.get("SIM_END_DATE")

    if sim_start:
        dates = [d for d in dates if d >= sim_start]
        print(f"[SIMULATOR] SIM_START_DATE={sim_start}: "
              f"replaying from {dates[0] if dates else '(none)'}")
    if sim_end:
        dates = [d for d in dates if d < sim_end]
        print(f"[SIMULATOR] SIM_END_DATE={sim_end}: "
              f"replaying up to (not including) {sim_end}")

    if not dates:
        print("[SIMULATOR] No dates remain after applying filters.")
        sys.exit(1)

    print(f"[SIMULATOR] Found {len(dates)} days of historical data")
    print(f"[SIMULATOR] Broadcast rate: 1 day per {SECONDS_PER_DAY:.1f} second(s)")
    print()

    client = AsyncTCPClient(HOST, PORT)
    if not await client.connect():
        sys.exit(1)

    print("[SIMULATOR] Beginning telemetry broadcast...\n")
    try:
        await broadcast_replay(client, dates)
        print("\n[SIMULATOR] Reached end of historical data.")
    except asyncio.CancelledError:
        print("\n[SIMULATOR] Broadcast cancelled.")
    except KeyboardInterrupt:
        print("\n[SIMULATOR] Stopped by user.")
    except Exception as e:
        print(f"\n[SIMULATOR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await client.close()
        print("[SIMULATOR] Connection closed.")



async def main():
    if LIVE_MODE:
        await run_live_mode()
    else:
        await run_replay_mode()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[SIMULATOR] Interrupted.")

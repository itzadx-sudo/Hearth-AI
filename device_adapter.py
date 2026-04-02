import asyncio
import json
import struct
from datetime import datetime

LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 65431
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 65432

# maps vendor-specific field names to our standard schema
VENDOR_KEY_MAP = {
    "vendor_bpm":      "heart_rate",
    "vendor_oxygen":   "spo2",
    "vendor_movement": "activity",
    "alt_temp":        "body_temp",
    "bed_sys":         "systolic_bp",
    "bed_dia":         "diastolic_bp",
}


def translate_payload(raw_payload: list) -> list:
    standardized = []
    for reading in raw_payload:
        clean = {VENDOR_KEY_MAP.get(k, k): v for k, v in reading.items()}
        if "patient_id" not in clean or "timestamp" not in clean:
            print(f"[WARNING] Adapter: dropped reading missing patient_id/timestamp: {list(clean.keys())}")
            continue
        standardized.append(clean)
    return standardized


_server_writer: asyncio.StreamWriter | None = None
_server_lock:   asyncio.Lock | None = None


def _get_lock() -> asyncio.Lock:
    global _server_lock
    if _server_lock is None:
        _server_lock = asyncio.Lock()
    return _server_lock


async def _get_server_connection() -> asyncio.StreamWriter:
    global _server_writer
    if _server_writer is not None and not _server_writer.is_closing():
        return _server_writer
    _, writer = await asyncio.open_connection(SERVER_HOST, SERVER_PORT)
    _server_writer = writer
    print(f"[CONN] Adapter connected to AI Server at {SERVER_HOST}:{SERVER_PORT}")
    return _server_writer


async def _close_server_connection():
    global _server_writer
    if _server_writer is not None:
        try:
            _server_writer.close()
            await _server_writer.wait_closed()
        except Exception:
            pass
        _server_writer = None


async def forward_to_server(clean_batch: list):
    # length-prefixed framing, same as iot_simulator
    payload = {"sim_date": datetime.now().strftime("%Y-%m-%d"), "readings": clean_batch}
    data  = json.dumps(payload).encode("utf-8")
    frame = struct.pack(">I", len(data)) + data

    async with _get_lock():
        for attempt in range(2):
            try:
                writer = await _get_server_connection()
                writer.write(frame)
                await writer.drain()
                return
            except (ConnectionRefusedError, ConnectionResetError, BrokenPipeError, OSError) as exc:
                print(f"[WARN] Adapter: forward attempt {attempt+1} failed: {exc}")
                await _close_server_connection()
                if attempt == 0:
                    continue
                print(f"[WARNING] Adapter could not reach AI Server on port {SERVER_PORT}.")


async def handle_device_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
    addr = writer.get_extra_info("peername")
    print(f"[CONN] Device connected: {addr}")
    try:
        raw = await reader.read(1_048_576)
        if raw:
            clean_batch = translate_payload(json.loads(raw.decode("utf-8")))
            if clean_batch:
                await forward_to_server(clean_batch)
    except json.JSONDecodeError:
        print(f"[ERROR] Adapter received malformed JSON from {addr}.")
    except asyncio.IncompleteReadError:
        pass
    except Exception as exc:
        print(f"[ERROR] Adapter error handling device {addr}: {exc}")
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        print(f"[DISC] Device disconnected: {addr}")


async def start_adapter():
    server = await asyncio.start_server(handle_device_connection, LISTEN_HOST, LISTEN_PORT)
    addr = server.sockets[0].getsockname()
    print(f"[OK] Device Adapter Online — listening on {addr[0]}:{addr[1]}")
    print(f"     Forwarding translated data → AI Server port {SERVER_PORT}")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(start_adapter())
    except KeyboardInterrupt:
        print("\n[INFO] Device Adapter shutting down.")

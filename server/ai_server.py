import asyncio
import json
import os
import struct
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from model.engine import (get_engine, TabNetEngine, LABEL_TO_IDX, IDX_TO_LABEL,
                           DEVICE, _DEVICE_LABEL, VITALS, N_FEATURES,
                           normalize_vitals_tanaka)
from config import (SERVER_HOST as HOST, SERVER_PORT as PORT, MAX_MSG_BYTES,
                    PREDICT_EVERY_N, EXERTION_BIAS_HR, EXERTION_BIAS_SBP,
                    CRITICAL_CONF_THRESHOLD, LOW_CONFIDENCE_THRESHOLD)

LIVE_MODE = os.environ.get("HEARTH_LIVE_MODE") == "1"

if LIVE_MODE:
    from data import logger as live_db


class Colors:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"

    @classmethod
    def status(cls, status: str) -> str:
        if status == "Critical":   return cls.RED + cls.BOLD
        if status == "Unhealthy":  return cls.YELLOW
        return cls.GREEN

    @classmethod
    def feature(cls, importance: float) -> str:
        if importance > 0.25: return cls.RED
        if importance > 0.15: return cls.YELLOW
        return cls.CYAN


class AsyncHearthServer:
    def __init__(self):
        self.engine: TabNetEngine = get_engine()
        if self.engine.is_ready:
            print(f"{Colors.GREEN}[OK]{Colors.RESET} Model loaded")
        else:
            print(f"{Colors.YELLOW}[INFO]{Colors.RESET} Model not trained — using NEWS2 fallback")
            print(f"       Train via: python tabnet_engine.py")

        print(f"{Colors.CYAN}[INFO]{Colors.RESET} Compute device: {_DEVICE_LABEL.get(DEVICE.type, DEVICE.type)}")
        print(f"{Colors.CYAN}[INFO]{Colors.RESET} Tanaka normalization: ΔHR={EXERTION_BIAS_HR}, ΔSBP={EXERTION_BIAS_SBP}")

        self._stats = {
            "total_readings": 0, "total_patients": 0,
            "critical_count": 0, "unhealthy_count": 0, "healthy_count": 0,
        }
        self._tick       = 0
        self._session_id = datetime.now().strftime("%Y%m%d_%H%M%S") if LIVE_MODE else None
        if LIVE_MODE:
            live_db.init_live_db()
            print(f"{Colors.CYAN}[LIVE]{Colors.RESET} Session ID : {self._session_id}")
            print(f"{Colors.CYAN}[LIVE]{Colors.RESET} Storage    : hearth_live.db")
            print(f"{Colors.CYAN}[LIVE]{Colors.RESET} Predictions: every {PREDICT_EVERY_N} ticks (needs ≥3 ticks history)")

    async def start(self):
        server = await asyncio.start_server(self._handle_client, HOST, PORT)
        addr   = server.sockets[0].getsockname()
        print(f"\n{Colors.GREEN}[OK]{Colors.RESET} Hearth AI Server listening on {addr[0]}:{addr[1]}")
        print(f"{Colors.BOLD}{'━' * 51}{Colors.RESET}\n")
        async with server:
            await server.serve_forever()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info("peername")
        print(f"{Colors.CYAN}[CONN]{Colors.RESET} Client connected: {addr}")
        try:
            while True:
                header  = await reader.readexactly(4)
                msg_len = struct.unpack(">I", header)[0]
                if msg_len > MAX_MSG_BYTES:
                    print(f"{Colors.RED}[ERROR]{Colors.RESET} Frame too large ({msg_len:,} bytes) — dropping connection")
                    break
                raw     = await reader.readexactly(msg_len)
                await self._process_payload(raw)
        except asyncio.IncompleteReadError:
            pass
        except ConnectionResetError:
            pass
        except Exception as e:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Client handler: {e}")
            import traceback
            traceback.print_exc()
        finally:
            writer.close()
            await writer.wait_closed()
            print(f"{Colors.CYAN}[DISC]{Colors.RESET} Client disconnected: {addr}")

    async def _process_payload(self, raw: bytes):
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} Malformed JSON payload — skipping")
            return

        sim_date = payload.get("sim_date", "Unknown")
        readings = payload.get("readings", [])
        if not readings:
            return

        start_time          = asyncio.get_event_loop().time()
        normalized_readings = readings

        feature_names = VITALS + ["activity", "delta_hr", "delta_spo2"]
        n             = len(normalized_readings)

        patient_reading_indices: Dict[str, List[int]] = defaultdict(list)
        for i, r in enumerate(normalized_readings):
            patient_reading_indices[str(r.get("patient_id", i))].append(i)

        patient_last_idx: Dict[str, int] = {
            pid: idxs[-1] for pid, idxs in patient_reading_indices.items()
        }

        # model path: batch inference on GPU, rule path: NEWS2 per-reading
        if self.engine.is_ready and self.engine.model is not None:
            X = self.engine.impute_all_patients(patient_reading_indices, normalized_readings)
            self.engine.model.eval()
            with torch.no_grad():
                status_logits, _, att = self.engine.model(X, return_attention=True)
                probs        = F.softmax(status_logits, dim=1)
                _indices_t   = probs.argmax(dim=1)
                _confs_t     = probs.gather(1, _indices_t.unsqueeze(1)).squeeze(1)

            indices     = _indices_t.cpu().numpy()
            confidences = _confs_t.cpu().numpy()
            attention   = att.cpu().numpy() if att is not None else \
                          np.full((n, N_FEATURES), 1.0 / N_FEATURES, dtype=np.float32)
        else:
            from model.engine import derive_severity
            indices     = np.zeros(n, dtype=np.int64)
            confidences = np.full(n, 0.80, dtype=np.float32)
            attention   = np.full((n, N_FEATURES), 1.0 / N_FEATURES, dtype=np.float32)
            for i, r in enumerate(normalized_readings):
                indices[i] = LABEL_TO_IDX[derive_severity(r)]

        patient_results = {"Healthy": 0, "Unhealthy": 0, "Critical": 0}
        for pid, idx in patient_last_idx.items():
            patient_results[IDX_TO_LABEL[int(indices[idx])]] += 1

        high_attention_features = defaultdict(float)
        for i in range(len(attention)):
            for j, feat in enumerate(feature_names):
                high_attention_features[feat] += attention[i, j]

        critical_patients = []
        low_conf_downgraded = 0
        for pid, idx in patient_last_idx.items():
            predicted_label = IDX_TO_LABEL[int(indices[idx])]
            conf = float(confidences[idx])

            # downgrade low-confidence criticals
            if predicted_label == "Critical" and conf < LOW_CONFIDENCE_THRESHOLD:
                predicted_label = "Unhealthy"
                indices[idx] = LABEL_TO_IDX["Unhealthy"]  # mutate so DB store is consistent
                low_conf_downgraded += 1
                print(f"    {Colors.YELLOW}[LOW-CONF]{Colors.RESET} Patient {pid}: "
                      f"Critical→Unhealthy (conf={conf:.2f} < {LOW_CONFIDENCE_THRESHOLD})")

            if predicted_label == "Critical" and conf >= CRITICAL_CONF_THRESHOLD:
                critical_patients.append({
                    "pid":  pid,
                    "conf": conf,
                    "attention": {feat: float(attention[idx, j]) for j, feat in enumerate(feature_names)},
                    "vitals": readings[idx],
                })

        inference_time = (asyncio.get_event_loop().time() - start_time) * 1000

        self._stats["total_readings"]  += len(readings)
        self._stats["total_patients"]  += len(patient_last_idx)
        self._stats["critical_count"]  += patient_results["Critical"]
        self._stats["unhealthy_count"] += patient_results["Unhealthy"]
        self._stats["healthy_count"]   += patient_results["Healthy"]

        n_patients = len(patient_last_idx)
        print(f"\n{Colors.BOLD}[DAY] {sim_date}{Colors.RESET} | "
              f"{n_patients} patients | {len(readings)} readings | {inference_time:.1f}ms")
        print(f"  {Colors.GREEN}* Healthy:{Colors.RESET}   {patient_results['Healthy']:>3} / {n_patients} patients")
        print(f"  {Colors.YELLOW}* Unhealthy:{Colors.RESET} {patient_results['Unhealthy']:>3} / {n_patients} patients")
        print(f"  {Colors.RED}* Critical:{Colors.RESET}  {patient_results['Critical']:>3} / {n_patients} patients")

        if critical_patients:
            print(f"\n  {Colors.RED}{Colors.BOLD}⚠ CRITICAL PATIENTS (conf ≥ {CRITICAL_CONF_THRESHOLD}):{Colors.RESET}")
            for cp in critical_patients[:5]:
                vitals = cp["vitals"]
                att    = cp["attention"]
                parts  = []
                hr  = vitals.get("heart_rate")
                sbp = vitals.get("systolic_bp")
                spo = vitals.get("spo2")
                tmp = vitals.get("body_temp")
                if hr  is not None and not (isinstance(hr,  float) and hr  != hr):  parts.append(f"HR:{hr:.0f}")
                if sbp is not None: parts.append(f"BP:{sbp:.0f}/{vitals.get('diastolic_bp', 0):.0f}")
                if spo is not None and not (isinstance(spo, float) and spo != spo): parts.append(f"SpO2:{spo:.0f}")
                if tmp is not None and not (isinstance(tmp, float) and tmp != tmp): parts.append(f"T:{tmp:.1f}°C")
                print(f"    {Colors.RED}-{Colors.RESET} Patient {cp['pid']} "
                      f"(Conf: {cp['conf']:.2f}) | {' | '.join(parts) if parts else 'Vitals: N/A'}")
                if att:
                    top = sorted(att.items(), key=lambda x: -x[1])[:3]
                    print(f"      {Colors.CYAN}Attention:{Colors.RESET} {', '.join(f'{f}:{v:.2f}' for f, v in top)}")
            if len(critical_patients) > 5:
                print(f"    {Colors.YELLOW}... and {len(critical_patients) - 5} more{Colors.RESET}")
            if low_conf_downgraded > 0:
                print(f"    {Colors.YELLOW}[LOW-CONF] {low_conf_downgraded} Critical prediction(s) "
                      f"downgraded → Unhealthy (conf < {LOW_CONFIDENCE_THRESHOLD}){Colors.RESET}")

        if high_attention_features and len(readings) > 10:
            print(f"\n  {Colors.CYAN}Feature Importance (Day Aggregate):{Colors.RESET}")
            total_att = sum(high_attention_features.values())
            if total_att > 0:
                norm_att  = {k: v / total_att for k, v in high_attention_features.items()}
                top_feats = sorted(norm_att.items(), key=lambda x: -x[1])[:4]
                for feat, importance in top_feats:
                    bar   = "█" * int(importance * 20) + "░" * (20 - int(importance * 20))
                    color = Colors.feature(importance)
                    print(f"    {color}{feat:<12}{Colors.RESET} {bar} {importance:.3f}")
        print()

        if LIVE_MODE:
            self._tick += 1
            tick_time   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._handle_live_mode(tick_time, readings, patient_last_idx, indices, confidences, attention, feature_names)


    def _handle_live_mode(self, tick_time: str, readings: list,
                          patient_last_idx: Dict[str, int],
                          indices, confidences, attention, feature_names):
        # store tick + maybe predict
        tick_results = []
        for pid, idx in patient_last_idx.items():
            r = readings[idx]
            att_dict = {feat: round(float(attention[idx, j]), 4)
                        for j, feat in enumerate(feature_names)}
            tick_results.append({
                "patient_id":   int(pid),
                "status":       IDX_TO_LABEL[int(indices[idx])],
                "confidence":   float(confidences[idx]),
                "heart_rate":   r.get("heart_rate"),
                "systolic_bp":  r.get("systolic_bp"),
                "diastolic_bp": r.get("diastolic_bp"),
                "body_temp":    r.get("body_temp"),
                "spo2":         r.get("spo2"),
                "activity":     r.get("activity"),
                "attention":    att_dict,
            })

        live_db.store_tick_results(
            self._session_id, self._tick, tick_time, tick_results
        )

        # need at least 7 ticks of history before predictions make sense
        if self._tick >= 7 and self._tick % PREDICT_EVERY_N == 0:
            patient_ids = list(patient_last_idx.keys())
            self._compute_live_predictions(tick_time, patient_ids)

    def _compute_live_predictions(self, tick_time: str, patient_ids: list):
        high_risk  = []
        n_assessed = 0

        for pid in patient_ids:
            window = live_db.get_patient_window(
                self._session_id, int(pid), limit=7
            )
            if len(window) < 7:
                continue

            result = self.engine.predict_risk(str(pid), window)
            if result is None:
                continue

            risk_score = result.get("risk_score", 0.0)
            conf = max(risk_score, 1.0 - risk_score)
            if result.get("risk_label") == "HIGH RISK" and conf < LOW_CONFIDENCE_THRESHOLD:
                # downgrade both label AND score so they stay consistent
                result["risk_label"] = "LOW RISK"
                result["risk_score"] = round(min(risk_score, LOW_CONFIDENCE_THRESHOLD - 0.01), 4)
                print(f"    {Colors.YELLOW}[LOW-CONF]{Colors.RESET} Patient {pid}: "
                      f"HIGH RISK→LOW RISK (conf={conf:.2f} < {LOW_CONFIDENCE_THRESHOLD})")

            live_db.store_prediction(
                self._session_id, self._tick, tick_time, int(pid), result
            )
            n_assessed += 1
            if result.get("risk_score", 0) >= 0.5:
                high_risk.append((pid, result))

        self._print_prediction_summary(n_assessed, high_risk)

    def _print_prediction_summary(self, n_assessed: int, high_risk: list):
        if n_assessed == 0:
            return
        print(f"  {Colors.MAGENTA}{Colors.BOLD}> 7-TICK PREDICTIONS  "
              f"(tick {self._tick}){Colors.RESET}")
        print(f"    {Colors.CYAN}Assessed : {n_assessed} patients{Colors.RESET}")
        if high_risk:
            print(f"    {Colors.RED}HIGH RISK: {len(high_risk)} patient(s){Colors.RESET}")
            for pid, result in high_risk[:5]:
                score   = result.get("risk_score", 0)
                factors = result.get("top_factors", [])
                fstr    = ", ".join(factors[:3]) if factors else "—"
                print(f"      {Colors.RED}-{Colors.RESET} Patient {pid:<6} "
                      f"score={score:.3f}  [{fstr}]")
            if len(high_risk) > 5:
                print(f"      {Colors.YELLOW}... and {len(high_risk)-5} more{Colors.RESET}")
        else:
            print(f"    {Colors.GREEN}No HIGH RISK patients at this horizon{Colors.RESET}")
        print()

    def print_session_summary(self):
        print(f"\n{Colors.BOLD}{'━' * 51}{Colors.RESET}")
        print(f"{Colors.BOLD}SESSION SUMMARY{Colors.RESET}")
        print(f"  Total Readings:  {self._stats['total_readings']:,}")
        print(f"  Total Patients:  {self._stats['total_patients']:,}")
        print(f"  {Colors.GREEN}Healthy:{Colors.RESET}        {self._stats['healthy_count']:,}")
        print(f"  {Colors.YELLOW}Unhealthy:{Colors.RESET}      {self._stats['unhealthy_count']:,}")
        print(f"  {Colors.RED}Critical:{Colors.RESET}       {self._stats['critical_count']:,}")
        if LIVE_MODE and self._session_id:
            try:
                summary = live_db.get_session_summary(self._session_id)
                preds   = live_db.get_latest_predictions(self._session_id)
                high_r  = [p for p in preds if p.get("risk_score", 0) >= 0.5]
                print(f"\n  {Colors.MAGENTA}LIVE SESSION  ({self._session_id}){Colors.RESET}")
                print(f"  Ticks recorded  : {summary['total_ticks']}")
                print(f"  Patients tracked: {summary['total_patients']}")
                print(f"  Predictions     : {len(preds)}")
                print(f"  {Colors.RED}HIGH RISK at exit: {len(high_r)}{Colors.RESET}")
                print(f"  {Colors.CYAN}Saved to: hearth_live.db{Colors.RESET}")
            except Exception:
                pass
        print(f"{Colors.BOLD}{'━' * 51}{Colors.RESET}")


async def main():
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}   Hearth AI — Async Real-Time Monitoring Server{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}\n")
    server = AsyncHearthServer()
    try:
        await server.start()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}[INFO]{Colors.RESET} Server shutting down...")
        server.print_session_summary()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.CYAN}[INFO]{Colors.RESET} Server stopped.")


import csv
import math
import os
import random
import sqlite3
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

from constants import news2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# better hr max formula for older adults than 220-age
def tanaka_hr_max(age: int) -> float:
    return 208.0 - 0.7 * age


def get_exertional_bias(age: int) -> Dict[str, float]:
    hr_max = tanaka_hr_max(age)
    
    hr_reserve = hr_max - 70
    delta_hr = 0.25 * hr_reserve
    delta_hr = max(8, min(delta_hr, 20))
    
    delta_sbp = 15.0
    
    return {
        "delta_hr": delta_hr,
        "delta_sbp": delta_sbp,
        "hr_max": hr_max,
    }


EXERTION_BIAS = {
    "heart_rate": 15.0,
    "systolic_bp": 15.0,
}



_PROFILE_WEIGHTS = [
    ("independent", 0.30),
    ("managed",     0.50),
    ("frail",       0.20),
]

_VITALS = {
    "independent": {
        "hr_base":   (60, 84),
        "sbp_base":  (115, 138),
        "dbp_base":  (60, 78),
        "temp_base": (36.0, 37.2),
        "spo2_base": (95, 99),
        "age_range": (65, 75),
        "active_prob":    0.25,
        "unhealthy_prob": 0.010,
        "critical_prob":  0.0001,
        "hr_active_boost":   (8, 20),
        "sbp_active_boost":  (12, 25),
        "dbp_active_boost":  (2, 8),
    },
    "managed": {
        "hr_base":   (64, 90),
        "sbp_base":  (128, 152),
        "dbp_base":  (62, 82),
        "temp_base": (36.0, 37.2),
        "spo2_base": (93, 97),
        "age_range": (70, 82),
        "active_prob":    0.12,
        "unhealthy_prob": 0.050,
        "critical_prob":  0.0005,
        "hr_active_boost":  (5, 15),
        "sbp_active_boost": (8, 20),
        "dbp_active_boost": (2, 6),
    },
    "frail": {
        "hr_base":   (62, 95),
        "sbp_base":  (128, 160),
        "dbp_base":  (58, 82),
        "temp_base": (35.8, 36.9),
        "spo2_base": (91, 95),
        "age_range": (78, 95),
        "active_prob":    0.08,
        "unhealthy_prob": 0.150,
        "critical_prob":  0.002,
        "hr_active_boost":  (3, 10),
        "sbp_active_boost": (8, 18),
        "dbp_active_boost": (1, 5),
    },
}

_UNHEALTHY_TRIGGERS = [
    "geriatric_fever",
    "tachycardia",
    "bp_spike",
    "mild_hypoxia",
    "combined_mild",
    "orthostatic_hypotension",
]

_CRITICAL_EVENTS = [
    "geriatric_cardiac_distress",
    "severe_hypoxia",
    "severe_fever",
    "hypertensive_crisis",
    "severe_bradycardia",
    "silent_sepsis",
]



def _sample_status(hr: float, sbp: float, spo2: float, temp: float,
                   is_active: bool = False) -> str:
    score, max_single = news2_score(
        hr, sbp, temp, spo2, is_active=is_active,
        exertion_bias_hr=EXERTION_BIAS["heart_rate"],
        exertion_bias_sbp=EXERTION_BIAS["systolic_bp"]
    )
    
    if score >= 5 or max_single >= 3:
        return "Critical"
    if score >= 2:
        return "Unhealthy"

    # catch tachycardia + low BP combo that NEWS2 misses in elderly
    eff_hr  = max(hr  - EXERTION_BIAS["heart_rate"], 25.0) if is_active else hr
    eff_sbp = max(sbp - EXERTION_BIAS["systolic_bp"], 50.0) if is_active else sbp
    if eff_hr > 100 and eff_sbp < 110:
        return "Critical"

    return "Healthy"



def _base_vitals(profile: str) -> Dict[str, float]:
    v = _VITALS[profile]
    return {
        "hr":   random.uniform(*v["hr_base"]),
        "sbp":  random.uniform(*v["sbp_base"]),
        "dbp":  random.uniform(*v["dbp_base"]),
        "temp": random.uniform(*v["temp_base"]),
        "spo2": random.randint(*v["spo2_base"]),
        "age":  random.randint(*v["age_range"]),
    }


# BP and HR naturally rise during the day, dip at night
def _circadian_offset(hour: int) -> Dict[str, float]:
    t_norm = (hour - 6) / 24.0 * 2 * math.pi
    sbp_offset = 8.0 * math.sin(t_norm) - 4.0 * math.cos(t_norm)
    hr_offset  = 5.0 * math.sin(t_norm) - 2.5 * math.cos(t_norm)
    t_temp = (hour - 5) / 24.0 * 2 * math.pi
    temp_offset = 0.35 * math.sin(t_temp)
    return {"sbp": sbp_offset, "hr": hr_offset, "temp": temp_offset}


_ACTIVITY_WEIGHTS = {
    "independent": [0.30, 0.25, 0.20, 0.15, 0.07, 0.03],
    "managed":     [0.45, 0.25, 0.18, 0.09, 0.02, 0.01],
    "frail":       [0.60, 0.25, 0.12, 0.03, 0.00, 0.00],
}


def _healthy_reading(profile: str, base: dict, hour: int) -> dict:
    v   = _VITALS[profile]
    cir = _circadian_offset(hour)
    age = base.get("age", 75)

    # shared noise makes vitals correlate (sick = everything shifts together)
    shared_noise = random.gauss(0, 1.0)
    is_night = (hour >= 22 or hour <= 5)

    if is_night:
        activity_level = 0
    else:
        activity_level = random.choices(
            range(6), weights=_ACTIVITY_WEIGHTS[profile], k=1
        )[0]

    is_active = activity_level >= 3
    mild_active = activity_level in (1, 2)

    hr   = base["hr"]   + cir["hr"]   + shared_noise * 1.5 + random.gauss(0, 2.0)
    sbp  = base["sbp"]  + cir["sbp"]  + shared_noise * 2.5 + random.gauss(0, 2.5)
    dbp  = base["dbp"]  + shared_noise * 1.5 + random.gauss(0, 1.8)
    temp = base["temp"] + cir["temp"] + random.gauss(0, 0.07)
    spo2 = base["spo2"] + random.randint(-1, 1)

    if is_active:
        exertion = get_exertional_bias(age)
        intensity_scale = 0.7 + (activity_level - 3) * 0.20
        hr   += random.uniform(
            exertion["delta_hr"] * 0.6 * intensity_scale,
            exertion["delta_hr"] * 1.4 * intensity_scale,
        )
        sbp  += random.uniform(
            v["sbp_active_boost"][0] * intensity_scale,
            v["sbp_active_boost"][1] * intensity_scale,
        )
        dbp  += random.uniform(
            v["dbp_active_boost"][0] * intensity_scale,
            v["dbp_active_boost"][1] * intensity_scale,
        )
        spo2  = max(spo2 - random.randint(0, 2), 94)
    elif mild_active:
        hr  += random.uniform(3, 8)
        sbp += random.uniform(3, 8)

    hr   = max(28,  round(hr))
    sbp  = max(60,  round(sbp))
    dbp  = max(30,  min(round(dbp), sbp - 5))
    temp = round(max(33.0, min(temp, 43.0)), 1)
    spo2 = min(100, max(60, round(spo2)))

    status = _sample_status(hr, sbp, spo2, temp, is_active=is_active)
    return {
        "hr": hr, "sbp": sbp, "dbp": dbp, "temp": temp, "spo2": spo2,
        "activity": activity_level,
        "status": status,
        "age": age,
    }


def _unhealthy_reading(profile: str, base: dict) -> dict:
    trigger = random.choice(_UNHEALTHY_TRIGGERS)
    cir     = _circadian_offset(random.randint(6, 22))

    hr   = base["hr"]   + cir["hr"]   + random.gauss(0, 2)
    sbp  = base["sbp"]  + cir["sbp"]  + random.gauss(0, 2)
    dbp  = base["dbp"]  + random.gauss(0, 2)
    temp = base["temp"] + cir["temp"] + random.gauss(0, 0.08)
    spo2 = float(base["spo2"])

    if trigger == "geriatric_fever":
        temp = round(random.uniform(37.5, 38.3), 1)
        hr   = hr + random.uniform(8, 18)
    elif trigger == "orthostatic_hypotension":
        sbp  = sbp - random.uniform(20, 35)
        hr   = hr + random.uniform(8, 18)
    elif trigger == "tachycardia":
        hr = random.uniform(91, 130)
        if random.random() < 0.5:
            sbp = sbp + random.uniform(8, 18)
    elif trigger == "bp_spike":
        sbp = random.uniform(101, 135)
        dbp = dbp + random.uniform(6, 14)
    elif trigger == "mild_hypoxia":
        spo2 = random.uniform(92, 95)
        hr   = hr + random.uniform(6, 16)
    elif trigger == "combined_mild":
        hr   = hr + random.uniform(18, 30)
        temp = round(random.uniform(37.5, 38.4), 1)

    hr   = max(28,  round(hr))
    sbp  = max(60,  round(sbp))
    dbp  = max(30,  min(round(dbp), sbp - 5))
    temp = round(max(33.0, min(temp, 43.0)), 1)
    spo2 = min(100, max(60, round(spo2)))

    status = _sample_status(hr, sbp, spo2, temp, is_active=False)
    return {
        "hr": hr, "sbp": sbp, "dbp": dbp, "temp": temp, "spo2": spo2,
        "activity": 0,
        "status": status,
        "age": base.get("age", 75),
    }


# critical events are the ones that should trigger alerts
def _critical_reading(base: dict) -> dict:
    event = random.choice(_CRITICAL_EVENTS)

    hr   = base["hr"]   + random.gauss(0, 2)
    sbp  = base["sbp"]  + random.gauss(0, 3)
    dbp  = base["dbp"]  + random.gauss(0, 2)
    temp = base["temp"] + random.gauss(0, 0.1)
    spo2 = float(base["spo2"])

    if event == "geriatric_cardiac_distress":
        hr  = random.uniform(91, 108)
        sbp = random.uniform(88, 109)
        dbp = dbp - random.uniform(5, 15)
    elif event == "silent_sepsis":
        spo2 = random.uniform(88, 94)
        hr   = hr + random.uniform(12, 22)
        temp = round(random.uniform(35.5, 37.2), 1)
        sbp  = sbp - random.uniform(8, 18)
    elif event == "severe_hypoxia":
        spo2 = random.uniform(72, 91)
        hr   = hr + random.uniform(18, 38)
    elif event == "severe_fever":
        temp = round(random.uniform(39.1, 40.2), 1)
        hr   = random.randint(112, 145)
        spo2 = random.uniform(90, 94)
    elif event == "hypertensive_crisis":
        sbp = random.uniform(220, 260)
        dbp = dbp + random.uniform(20, 40)
        hr  = hr  + random.uniform(10, 28)
    elif event == "severe_bradycardia":
        hr  = random.uniform(18, 40)
        sbp = sbp - random.uniform(10, 30)

    hr   = max(10,  round(hr))
    sbp  = max(50,  round(sbp))
    dbp  = max(25,  min(round(dbp), sbp - 5))
    temp = round(max(33.0, min(temp, 43.0)), 1)
    spo2 = min(100, max(50, round(spo2)))

    status = _sample_status(hr, sbp, spo2, temp, is_active=False)
    return {
        "hr": hr, "sbp": sbp, "dbp": dbp, "temp": temp, "spo2": spo2,
        "activity": 0,
        "status": status,
        "age": base.get("age", 75),
    }



# simulate real sensor failures - wrist HR sensors drop a lot
HR_DROPOUT        = 0.385
SPO2_DROPOUT      = 0.250
SPO2_DROPOUT_FEVER= 0.420  # sweaty fingers = worse pulse ox readings
TEMP_DROPOUT      = 0.050


def inject_sensor_dropout(reading: dict, noise_rng: random.Random) -> dict:
    is_febrile = reading.get("temp") is not None and reading["temp"] >= 37.5
    spo2_drop = SPO2_DROPOUT_FEVER if is_febrile else SPO2_DROPOUT
    
    hr_val   = reading["hr"]   if noise_rng.random() > HR_DROPOUT   else float('nan')
    spo2_val = reading["spo2"] if noise_rng.random() > spo2_drop    else float('nan')
    temp_val = reading["temp"] if noise_rng.random() > TEMP_DROPOUT else float('nan')
    
    return {
        "hr": hr_val,
        "sbp": reading["sbp"],
        "dbp": reading["dbp"],
        "temp": temp_val,
        "spo2": spo2_val,
        "activity": reading["activity"],
        "status": reading["status"],
        "age": reading.get("age", 75),
    }



def _generate_patient_chunk(args):
    patient_ids, profiles_chunk, num_days, readings_per_hour, seed_offset = args

    random.seed(42 + seed_offset)
    np.random.seed(42 + seed_offset)
    noise_rng = random.Random(1337 + seed_offset)

    readings_per_day = readings_per_hour * 24
    interval = timedelta(minutes=60.0 / readings_per_hour)
    drain_per_reading = 0.5 / readings_per_day
    start_dt = datetime(2024, 1, 1, 0, 0, 0)

    batch = []
    counts = {"Healthy": 0, "Unhealthy": 0, "Critical": 0}

    for patient_id, profile in zip(patient_ids, profiles_chunk):
        v = _VITALS[profile]
        base = _base_vitals(profile)
        battery = random.uniform(70, 100)
        crit_left = 0
        patient_time = start_dt

        for _day in range(num_days):
            for _slot in range(readings_per_day):
                hour = patient_time.hour

                if crit_left > 0:
                    reading = _critical_reading(base)
                    crit_left -= 1
                elif random.random() < v["critical_prob"]:
                    reading = _critical_reading(base)
                    crit_left = random.randint(1, 4)
                elif random.random() < v["unhealthy_prob"]:
                    reading = _unhealthy_reading(profile, base)
                else:
                    reading = _healthy_reading(profile, base, hour)

                counts[reading["status"]] = counts.get(reading["status"], 0) + 1

                battery = max(5.0, battery - drain_per_reading)
                if random.random() < 0.0005:
                    battery = random.uniform(85, 100)

                ts = patient_time.strftime("%Y-%m-%d %H:%M:%S")

                reading_with_nan = inject_sensor_dropout(reading, noise_rng)

                batch.append((
                    patient_id, ts, ts[:10],
                    reading_with_nan["hr"],
                    reading_with_nan["sbp"],
                    reading_with_nan["dbp"],
                    reading_with_nan["temp"],
                    reading_with_nan["spo2"],
                    reading_with_nan["activity"],
                ))
                patient_time += interval

    return batch, counts



def generate_to_db(num_patients: int, num_days: int, readings_per_hour: int = 15,
                   batch_size: int = 10_000, seed: int = 42):
    from multiprocessing import Pool
    from data import logger as _dl

    readings_per_day = readings_per_hour * 24
    total = num_patients * num_days * readings_per_day
    print(f"\nGenerating {total:,} records "
          f"({num_patients} patients × {num_days} days "
          f"× {readings_per_hour} rdgs/hr)  …")
    print(f"[INFO] Tanaka Formula enabled for age-based exertion modeling")
    print(f"[INFO] Random seed: {seed} "
          f"({'training' if seed == 42 else 'testing / custom'})")
    print(f"[INFO] Sensor dropouts injected as NaN: HR={HR_DROPOUT*100:.0f}%, "
          f"SpO2={SPO2_DROPOUT*100:.0f}%, Temp={TEMP_DROPOUT*100:.0f}%")

    random.seed(seed)
    np.random.seed(seed)
    _dl.ensure_sensor_db(_dl.SENSOR_DB_PATH)

    profile_names = [n for n, _ in _PROFILE_WEIGHTS]
    profile_wts = [w for _, w in _PROFILE_WEIGHTS]
    profiles = random.choices(profile_names, weights=profile_wts, k=num_patients)
    patient_ids = list(range(1, num_patients + 1))

    n_workers = max(1, os.cpu_count() or 1)
    chunk_size = min(50, (num_patients + n_workers - 1) // n_workers)
    chunks = [
        (
            patient_ids[i: i + chunk_size],
            profiles[i: i + chunk_size],
            num_days, readings_per_hour, i,
        )
        for i in range(0, num_patients, chunk_size)
    ]
    print(f"[INFO] {n_workers} CPU cores | {len(chunks)} chunks | "
          f"≤{chunk_size} patients each\n")

    _INSERT_SQL = (
        "INSERT INTO sensor_data "
        "(patient_id, timestamp, sim_date, heart_rate, systolic_bp, "
        "diastolic_bp, body_temp, spo2, activity) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )

    conn = sqlite3.connect(_dl.SENSOR_DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-32000")

    total_written = 0
    all_counts = {"Healthy": 0, "Unhealthy": 0, "Critical": 0}

    try:
        with Pool(processes=n_workers) as pool:
            for chunk_idx, (rows, counts) in enumerate(
                    pool.imap_unordered(_generate_patient_chunk, chunks)):
                for i in range(0, len(rows), batch_size):
                    conn.executemany(_INSERT_SQL, rows[i: i + batch_size])
                    conn.commit()
                    total_written += len(rows[i: i + batch_size])
                for k in counts:
                    all_counts[k] = all_counts.get(k, 0) + counts[k]

                pct = (chunk_idx + 1) / len(chunks) * 100
                print(f"  [{pct:5.1f}%] Chunk {chunk_idx+1}/{len(chunks)} | "
                      f"{total_written:>12,} records written")
    finally:
        conn.close()

    print(f"\n[OK] {total_written:,} records written to hearth_sensor.db")
    print(f"[INFO] NaN values injected for sensor dropout simulation\n")

    total_r = sum(all_counts.values())
    print(f"  Label Distribution (NEWS2):")
    for status in ("Healthy", "Unhealthy", "Critical"):
        c = all_counts.get(status, 0)
        pct = c / total_r * 100 if total_r > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {status:<10}: {c:>12,} ({pct:5.2f}%) {bar}")



def generate_classifier_csv(num_records: int,
                            filename: str = "classifier_training_data.csv"):
    print(f"\n[INFO] Generating {num_records:,} readings with Tanaka Formula …")
    print(f"[INFO] NaN dropouts: HR={HR_DROPOUT*100:.0f}%, SpO2={SPO2_DROPOUT*100:.0f}%, "
          f"Temp={TEMP_DROPOUT*100:.0f}%")

    noise_rng = random.Random(1337)
    profile_names = [n for n, _ in _PROFILE_WEIGHTS]
    profile_wts = [w for _, w in _PROFILE_WEIGHTS]

    rows = []
    counts = {"Healthy": 0, "Unhealthy": 0, "Critical": 0}

    for i in range(num_records):
        profile = random.choices(profile_names, weights=profile_wts, k=1)[0]
        base = _base_vitals(profile)
        v = _VITALS[profile]
        rand = random.random()
        hour = random.randint(0, 23)

        if rand < v["critical_prob"]:
            reading = _critical_reading(base)
        elif rand < v["unhealthy_prob"]:
            reading = _unhealthy_reading(profile, base)
        else:
            reading = _healthy_reading(profile, base, hour)

        counts[reading["status"]] = counts.get(reading["status"], 0) + 1

        reading_with_nan = inject_sensor_dropout(reading, noise_rng)

        rows.append({
            "patient_id": (i // 48) + 1,
            "heart_rate": reading_with_nan["hr"],
            "systolic_bp": reading_with_nan["sbp"],
            "diastolic_bp": reading_with_nan["dbp"],
            "body_temp": reading_with_nan["temp"],
            "spo2": reading_with_nan["spo2"],
            "activity": reading_with_nan["activity"],
            "age": reading_with_nan.get("age", 75),
            "status": reading["status"],
        })

    headers = ["patient_id", "heart_rate", "systolic_bp", "diastolic_bp",
               "body_temp", "spo2", "activity", "age", "status"]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] {num_records:,} records → {filename}")
    total_r = sum(counts.values())
    print(f"\n  Label Distribution (NEWS2-validated):")
    for status in ("Healthy", "Unhealthy", "Critical"):
        c = counts.get(status, 0)
        pct = c / total_r * 100 if total_r > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {status:<10}: {c:>8,} ({pct:5.2f}%) {bar}")



def _prompt_int(prompt: str, default: int, lo: int, hi: int) -> int:
    try:
        raw = input(f"  {prompt} (default {default}): ").strip()
        if not raw:
            return default
        val = int(raw)
        if not (lo <= val <= hi):
            print(f"    Out of range ({lo}–{hi}). Using {default}.")
            return default
        return val
    except (ValueError, EOFError):
        return default



if __name__ == "__main__":
    from data import logger as _dl

    print("=" * 58)
    print("  Hearth AI — Data Generator (Tanaka Formula + NaN Dropout)")
    print("=" * 58)
    print()
    print("  [1] Training Data  → hearth_sensor.db  (trainer + simulator)")
    print("  [2] Testing Data   → hearth_test.db   (separate evaluation dataset)")
    print("  [0] Exit")
    print()

    try:
        sub = input("  Select (0-2): ").strip()
    except (EOFError, KeyboardInterrupt):
        sys.exit(0)

    if sub == "0":
        sys.exit(0)

    if sub not in ("1", "2"):
        print("  [ERROR] Invalid choice.")
        sys.exit(1)

    is_training = (sub == "1")
    label   = "Training" if is_training else "Testing"
    db_name = "hearth_sensor.db" if is_training else "hearth_test.db"
    db_path = os.path.join(BASE_DIR, db_name)

    print(f"\n  {label} Data Generator — target: {db_name}")
    print(f"  Profiles: independent 30% / managed 50% / frail 20%")
    print(f"  Sensor dropout: HR {HR_DROPOUT*100:.0f}%, "
          f"SpO2 {SPO2_DROPOUT*100:.0f}%, Temp {TEMP_DROPOUT*100:.0f}%")
    print()

    num_patients      = _prompt_int("Number of patients   [1-5000]", 50,  1, 5000)
    num_days          = _prompt_int("Days per patient     [1-730] ", 30,  1,  730)
    readings_per_hour = _prompt_int("Readings per hour    [1-20]  ", 15,  1,   20)

    total = num_patients * num_days * readings_per_hour * 24
    print(f"\n  Will generate {total:,} records → {db_name}")

    try:
        if input("\n  Proceed? (Y/n): ").strip().lower() == "n":
            sys.exit(0)
    except (EOFError, KeyboardInterrupt):
        sys.exit(0)

    orig_path = _dl.SENSOR_DB_PATH
    _dl.SENSOR_DB_PATH = db_path
    try:
        generate_to_db(num_patients, num_days, readings_per_hour)
        print(f"\n[OK] {label} dataset ready → {db_name}")
    except Exception as exc:
        print(f"[ERROR] Generation failed: {exc}")
    finally:
        _dl.SENSOR_DB_PATH = orig_path

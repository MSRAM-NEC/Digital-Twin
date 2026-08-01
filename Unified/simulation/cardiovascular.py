"""
Layer 5: Cardiovascular System Model
Simulates the heart and circulatory system response to drug effects.

Models implemented:
- Windkessel circulation model (arterial pressure dynamics)
- Simplified cardiac electrophysiology (action potential duration)
- Cardiac output computation
- ECG waveform generation
- Arrhythmia probability estimation
"""

import numpy as np
from scipy.integrate import solve_ivp


def windkessel_model(heart_rate: float, stroke_volume: float,
                     arterial_compliance: float = 1.5,
                     peripheral_resistance: float = 1.0,
                     t_max: float = 5.0, n_points: int = 1000) -> dict:
    """
    3-Element Windkessel model of arterial pressure.

    Parameters:
        heart_rate: beats per minute
        stroke_volume: mL per beat
        arterial_compliance: arterial compliance (mL/mmHg)
        peripheral_resistance: total peripheral resistance (mmHg·s/mL)
        t_max: simulation time (seconds)
        n_points: number of time points
    """
    T = 60.0 / heart_rate  # cardiac cycle period
    t_systole = T * 0.35   # systolic phase duration

    # Characteristic impedance
    Zc = 0.05  # mmHg·s/mL

    def cardiac_flow(t):
        """Pulsatile cardiac output as flow rate."""
        phase = t % T
        if phase < t_systole:
            # Sinusoidal ejection pattern
            return (stroke_volume / t_systole) * np.sin(np.pi * phase / t_systole)
        return 0.0

    def odes(t, y):
        P = y[0]  # arterial pressure
        Q_in = cardiac_flow(t)
        dPdt = (Q_in - P / peripheral_resistance) / arterial_compliance
        return [dPdt]

    t_span = (0, t_max)
    t_eval = np.linspace(0, t_max, n_points)
    y0 = [80.0]  # initial arterial pressure

    sol = solve_ivp(odes, t_span, y0, t_eval=t_eval, method="RK45", max_step=0.001)

    pressure = sol.y[0]

    # Extract systolic and diastolic after stabilization
    stable_start = int(n_points * 0.4)
    systolic = float(np.max(pressure[stable_start:]))
    diastolic = float(np.min(pressure[stable_start:]))
    mean_arterial = diastolic + (systolic - diastolic) / 3

    return {
        "time_seconds": sol.t.tolist(),
        "arterial_pressure_mmHg": pressure.tolist(),
        "systolic_pressure": round(systolic, 1),
        "diastolic_pressure": round(diastolic, 1),
        "mean_arterial_pressure": round(mean_arterial, 1),
        "pulse_pressure": round(systolic - diastolic, 1),
    }


def generate_ecg_waveform(heart_rate: float, qt_interval_ms: float = 400,
                           duration_seconds: float = 10.0,
                           sampling_rate: int = 500) -> dict:
    """
    Generate synthetic ECG waveform (Lead II approximation).

    Models P wave, QRS complex, and T wave with adjustable QT interval.
    """
    n_points = int(duration_seconds * sampling_rate)
    t = np.linspace(0, duration_seconds, n_points)
    ecg = np.zeros(n_points)

    T_cycle = 60.0 / heart_rate  # RR interval in seconds
    qt_sec = qt_interval_ms / 1000.0

    for beat_start in np.arange(0, duration_seconds, T_cycle):
        for i, ti in enumerate(t):
            dt = ti - beat_start
            if dt < 0 or dt > T_cycle:
                continue

            # P wave (atrial depolarization)
            p_center = 0.1
            p_width = 0.04
            if abs(dt - p_center) < 3 * p_width:
                ecg[i] += 0.15 * np.exp(-((dt - p_center) ** 2) / (2 * p_width ** 2))

            # QRS complex
            qrs_center = 0.2
            # Q wave
            q_center = qrs_center - 0.015
            if abs(dt - q_center) < 0.02:
                ecg[i] -= 0.1 * np.exp(-((dt - q_center) ** 2) / (2 * 0.005 ** 2))

            # R wave
            r_width = 0.008
            if abs(dt - qrs_center) < 0.03:
                ecg[i] += 1.2 * np.exp(-((dt - qrs_center) ** 2) / (2 * r_width ** 2))

            # S wave
            s_center = qrs_center + 0.015
            if abs(dt - s_center) < 0.02:
                ecg[i] -= 0.25 * np.exp(-((dt - s_center) ** 2) / (2 * 0.006 ** 2))

            # T wave (ventricular repolarization)
            t_center = qrs_center + qt_sec * 0.6
            t_width = qt_sec * 0.12
            if abs(dt - t_center) < 4 * t_width:
                ecg[i] += 0.3 * np.exp(-((dt - t_center) ** 2) / (2 * t_width ** 2))

    # Add baseline noise
    noise = np.random.normal(0, 0.02, n_points)
    ecg += noise

    return {
        "time_seconds": t.tolist(),
        "voltage_mV": ecg.tolist(),
        "heart_rate": heart_rate,
        "rr_interval_ms": round(T_cycle * 1000, 1),
        "qt_interval_ms": qt_interval_ms,
        "qtc_interval_ms": round(qt_interval_ms / np.sqrt(T_cycle), 1),
    }


def compute_cardiac_output(heart_rate: float, contractility_pct: float,
                           preload_factor: float = 1.0) -> dict:
    """
    Compute cardiac output and related hemodynamic parameters.

    Parameters:
        heart_rate: bpm
        contractility_pct: percentage of normal contractility (100 = normal)
        preload_factor: multiplier for venous return (1.0 = normal)
    """
    # Stroke volume depends on contractility and preload (Frank-Starling)
    base_sv = 70  # mL, normal stroke volume
    sv = base_sv * (contractility_pct / 100) * preload_factor

    # Adjust SV for extreme heart rates (decreased filling time)
    if heart_rate > 120:
        filling_penalty = 1.0 - 0.005 * (heart_rate - 120)
        sv *= max(0.5, filling_penalty)
    elif heart_rate < 40:
        sv *= 1.2  # compensatory increase at very low HR

    cardiac_output = heart_rate * sv / 1000  # L/min
    ejection_fraction = min(75, max(15, sv / 120 * 100))  # approximate EF%

    # Organ perfusion estimates
    total_co = cardiac_output
    organ_perfusion = {
        "brain": round(total_co * 0.15, 2),
        "heart": round(total_co * 0.05, 2),
        "kidneys": round(total_co * 0.22, 2),
        "liver": round(total_co * 0.25, 2),
        "skeletal_muscle": round(total_co * 0.20, 2),
        "skin": round(total_co * 0.05, 2),
        "other": round(total_co * 0.08, 2),
    }

    return {
        "heart_rate_bpm": round(heart_rate, 1),
        "stroke_volume_mL": round(sv, 1),
        "cardiac_output_Lmin": round(cardiac_output, 2),
        "ejection_fraction_pct": round(ejection_fraction, 1),
        "organ_perfusion_Lmin": organ_perfusion,
    }


def estimate_arrhythmia_risk(qt_interval_ms: float, heart_rate: float,
                              herg_binding: float = 0.0,
                              drug_target_interactions: dict = None) -> dict:
    """
    Estimate arrhythmia risk based on QT prolongation and drug-ion channel interactions.
    """
    # QTc correction (Bazett's formula)
    rr_interval = 60.0 / max(heart_rate, 30)
    qtc = qt_interval_ms / (1000 * np.sqrt(rr_interval)) * 1000

    # Base risk from QTc
    if qtc > 500:
        qt_risk = 0.8
    elif qtc > 480:
        qt_risk = 0.5
    elif qtc > 460:
        qt_risk = 0.25
    elif qtc > 440:
        qt_risk = 0.1
    else:
        qt_risk = 0.02

    # Additional risk from ion channel interactions
    ion_channel_risk = 0.0
    if drug_target_interactions:
        interactions = drug_target_interactions.get("interactions", {})
        herg = interactions.get("hERG", {}).get("binding_strength", 0)
        nav = interactions.get("Nav1.5", {}).get("binding_strength", 0)
        cav = interactions.get("Cav1.2", {}).get("binding_strength", 0)

        ion_channel_risk = 0.3 * herg + 0.15 * nav + 0.1 * cav

    # Combined risk
    total_risk = min(0.95, qt_risk + ion_channel_risk * (1 - qt_risk))

    # Risk category
    if total_risk > 0.6:
        category = "high"
        recommendation = "Avoid drug or use with continuous ECG monitoring"
    elif total_risk > 0.3:
        category = "moderate"
        recommendation = "Use with caution; periodic ECG monitoring recommended"
    elif total_risk > 0.1:
        category = "low"
        recommendation = "Standard monitoring; check baseline QTc"
    else:
        category = "minimal"
        recommendation = "No special cardiac monitoring needed"

    # Torsades de Pointes (TdP) risk
    tdp_risk = total_risk * 0.3  # TdP is a subset of arrhythmia risk

    return {
        "qtc_interval_ms": round(qtc, 1),
        "arrhythmia_probability": round(total_risk, 4),
        "tdp_risk": round(tdp_risk, 4),
        "risk_category": category,
        "recommendation": recommendation,
        "risk_factors": {
            "qt_prolongation": round(qt_risk, 4),
            "ion_channel_blockade": round(ion_channel_risk, 4),
        },
    }


def simulate_cardiovascular(pd_effects: dict, drug_target_interactions: dict = None) -> dict:
    """
    Full cardiovascular simulation pipeline.

    Takes PD effects and simulates:
    - Arterial pressure waveform
    - ECG waveform
    - Cardiac output
    - Arrhythmia risk
    """
    # Extract cardiac parameters from PD effects
    effects_at_cmax = pd_effects.get("effects_at_cmax", {})
    heart_rate = effects_at_cmax.get("heart_rate", 72)
    systolic = effects_at_cmax.get("systolic_bp", 120)
    diastolic = effects_at_cmax.get("diastolic_bp", 80)
    contractility = effects_at_cmax.get("contractility", 100)
    qt_interval = effects_at_cmax.get("qt_interval", 400)

    # Clamp to physiological ranges
    heart_rate = max(30, min(200, heart_rate))
    contractility = max(20, min(150, contractility))
    qt_interval = max(300, min(600, qt_interval))

    # Compute stroke volume and adjust peripheral resistance
    stroke_volume = 70 * (contractility / 100)
    peripheral_resistance = max(0.3, (systolic / 120) * 1.0)
    compliance = max(0.5, 1.5 * (120 / max(systolic, 60)))

    # Windkessel model
    windkessel = windkessel_model(
        heart_rate, stroke_volume,
        arterial_compliance=compliance,
        peripheral_resistance=peripheral_resistance
    )

    # ECG waveform
    ecg = generate_ecg_waveform(heart_rate, qt_interval, duration_seconds=5.0)

    # Cardiac output
    co = compute_cardiac_output(heart_rate, contractility)

    # Arrhythmia risk
    arrhythmia = estimate_arrhythmia_risk(
        qt_interval, heart_rate,
        drug_target_interactions=drug_target_interactions
    )

    # Overall cardiac state assessment
    cardiac_state = "normal"
    warnings = []

    if heart_rate < 50:
        cardiac_state = "bradycardia"
        warnings.append("Significant bradycardia detected")
    elif heart_rate > 100:
        cardiac_state = "tachycardia"
        warnings.append("Tachycardia detected")

    if systolic < 90:
        cardiac_state = "hypotension"
        warnings.append("Hypotension risk")
    elif systolic > 140:
        warnings.append("Hypertension not fully controlled")

    if contractility < 60:
        warnings.append("Reduced cardiac contractility")

    if qt_interval > 460:
        warnings.append(f"Prolonged QT interval ({qt_interval:.0f} ms)")

    if arrhythmia["arrhythmia_probability"] > 0.3:
        warnings.append("Elevated arrhythmia risk")

    return {
        "cardiac_parameters": {
            "heart_rate_bpm": round(heart_rate, 1),
            "systolic_bp_mmHg": round(systolic, 1),
            "diastolic_bp_mmHg": round(diastolic, 1),
            "contractility_pct": round(contractility, 1),
            "qt_interval_ms": round(qt_interval, 1),
        },
        "windkessel": windkessel,
        "ecg": ecg,
        "cardiac_output": co,
        "arrhythmia_risk": arrhythmia,
        "cardiac_state": cardiac_state,
        "warnings": warnings,
    }

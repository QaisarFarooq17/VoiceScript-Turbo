from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

import sounddevice as sd


def get_input_device_info(input_device: Optional[int]) -> dict:
    device = sd.query_devices(input_device, "input")
    if not isinstance(device, dict):
        raise RuntimeError("Unable to read input device information")
    return device


def pick_sample_rate(input_device: Optional[int], requested_sample_rate: int) -> int:
    device = get_input_device_info(input_device)
    default_rate = int(round(float(device.get("default_samplerate", 0)) or 0))

    candidates = []
    if requested_sample_rate > 0:
        candidates.append(int(requested_sample_rate))
    if default_rate > 0:
        candidates.append(default_rate)
    candidates.extend([16000, 48000, 44100, 32000, 22050])

    # Keep order while removing duplicates.
    unique_candidates = list(dict.fromkeys(candidates))

    for sample_rate in unique_candidates:
        try:
            sd.check_input_settings(
                device=input_device,
                channels=1,
                dtype="float32",
                samplerate=sample_rate,
            )
            return sample_rate
        except Exception:
            continue

    raise RuntimeError(
        "No compatible sample rate found for input device. "
        "Run --list-devices and choose another device with --input-device."
    )


def describe_selected_device(input_device: Optional[int]) -> Tuple[int, str]:
    if input_device is None:
        selected_idx = int(sd.default.device[0])
    else:
        selected_idx = int(input_device)
    info = get_input_device_info(selected_idx)
    return selected_idx, str(info.get("name", "unknown"))


def boost_quiet_audio(
    audio: np.ndarray,
    target_rms: float = 0.08,
    max_gain: float = 20.0,
    silence_threshold: float = 0.002,
) -> Tuple[np.ndarray, float, float, float]:
    signal = audio.reshape(-1).astype(np.float32)
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak < silence_threshold:
        return signal, 1.0, peak, 0.0

    rms = float(np.sqrt(np.mean(signal * signal) + 1e-12))
    if rms <= 0:
        return signal, 1.0, peak, rms

    gain = min(max(target_rms / rms, 1.0), max_gain)
    boosted = np.clip(signal * gain, -1.0, 1.0)
    return boosted, float(gain), peak, rms

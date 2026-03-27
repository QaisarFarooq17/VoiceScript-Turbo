# Personal Voice-to-Text Bot (NVIDIA + Whisper Large-v3-Turbo)

Local GPU-accelerated speech-to-text bot using `faster-whisper` and your conda environment.

## What this gives you
- Fast transcription with `large-v3-turbo` on NVIDIA GPU.
- Optional VAD filtering to ignore silence.
- CLI script for transcribing audio files.
- Clean project structure for extension into a real-time mic bot.

## Project layout
- `src/transcribe.py` - Main CLI transcription script.
- `src/live_mic.py` - Real-time microphone transcription script.
- `src/live_gui.py` - Desktop GUI (start/stop, frequency graph, transcript, copy).
- `requirements.txt` - Python dependencies.
- `.env.example` - Runtime defaults.
- `outputs/` - Transcription results.

## 1) Pick a working conda env
I checked your machine and only this env currently has a working Python binary:

`/home/qaisar/miniconda3/envs/Finn_Plus`

Activate with:

```bash
source /home/qaisar/miniconda3/bin/activate /home/qaisar/miniconda3/envs/Finn_Plus
```

Note: `/home/qaisar/miniconda3/envs/finn_env/` currently appears incomplete (missing valid Python target), so use `Finn_Plus` unless you repair `finn_env`.

## 2) Install dependencies
From project root:

```bash
pip install -r requirements.txt
```

## 3) Run transcription

```bash
python -m src.transcribe \
  --input /absolute/path/to/audio.wav \
  --output outputs/result.txt \
  --model large-v3-turbo \
  --device cuda \
  --compute-type float16 \
  --language en
```

Or with the helper script:

```bash
./run_transcribe.sh /absolute/path/to/audio.wav
```

## 4) Run real-time microphone bot

```bash
./run_live_mic.sh
```

If you want Urdu (example):

```bash
./run_live_mic.sh ur
```

To list input devices:

```bash
/home/qaisar/miniconda3/envs/Finn_Plus/bin/python -m src.live_mic --list-devices
```

Then select one explicitly (example index `0`):

```bash
/home/qaisar/miniconda3/envs/Finn_Plus/bin/python -m src.live_mic --input-device 0 --language en --vad
```

The script now auto-selects a valid sample rate for your microphone (prevents `Invalid sample rate [PaErrorCode -9997]`).

## 5) Run GUI (start/stop + graph + copy)

```bash
./run_live_gui.sh
```

GUI features:
- `Start Speaking` and `Stop Speaking` controls.
- Live frequency graph (FFT).
- Live transcript text window.
- `Copy Text` button to copy the generated transcript.
- `Clear Text` button to reset transcript view instantly.
- Language dropdown (including `Urdu`, `English`, and `Auto Detect`).
- `Whisper Mode` plus `Mic Boost` slider to amplify very low speaking volume.

For whisper/very low voice in GUI:
- Keep `Whisper Mode` enabled.
- Increase `Mic Boost` (for example `20x` to `30x`) if text is still missing.

For whisper mode in CLI:

```bash
./run_live_mic.sh --target-rms 0.10 --max-gain 28 --silence-threshold 0.0012
```

## Useful flags
- `--vad` enables voice activity detection.
- `--beam-size` defaults to `1` for speed.
- `--task` supports `transcribe` or `translate`.

## Notes
- First run downloads model weights and can take a bit.
- If GPU is unavailable, use `--device cpu --compute-type int8`.
- For best speed/accuracy balance on NVIDIA, keep `--device cuda --compute-type float16`.

## Troubleshooting
- `run_transcribe.sh: command not found` means your shell does not search the current directory by default. Use `./run_transcribe.sh ...` (or `./scripts/run_transcribe.sh ...`).
- `bash: run_transcribe.sh: No such file or directory` from project root means the file is in `scripts/`; use `bash scripts/run_transcribe.sh ...` or the root wrapper `./run_transcribe.sh ...`.
- `pip install Finn_Plus` fails because `Finn_Plus` is a local conda environment name, not a pip package.
- `Invalid sample rate [PaErrorCode -9997]`: the app now auto-negotiates a supported sample rate, but you can still choose a specific mic index using `--input-device`.

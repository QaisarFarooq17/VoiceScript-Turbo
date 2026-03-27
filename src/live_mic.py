import argparse
import signal
import tempfile
import threading
from pathlib import Path
from queue import Empty, Queue
from typing import List

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from src.audio_utils import boost_quiet_audio, describe_selected_device, pick_sample_rate

try:
    import sounddevice as sd
except ImportError as exc:
    raise RuntimeError(
        "sounddevice is required for live microphone mode. Install with: pip install sounddevice"
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time microphone transcription with faster-whisper."
    )
    parser.add_argument("--model", default="large-v3-turbo", help="Whisper model name")
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Inference device",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        help="CTranslate2 compute type (float16, int8_float16, int8)",
    )
    parser.add_argument("--language", default="en", help="Language code or empty for auto")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=3.0,
        help="Audio window duration per transcription pass",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=0,
        help="Capture sample rate. Use 0 for auto device-compatible selection.",
    )
    parser.add_argument("--beam-size", type=int, default=1, help="Beam size")
    parser.add_argument("--vad", action="store_true", help="Enable VAD filter")
    parser.add_argument(
        "--target-rms",
        type=float,
        default=0.09,
        help="Target RMS after amplification for quiet voice capture",
    )
    parser.add_argument(
        "--max-gain",
        type=float,
        default=24.0,
        help="Maximum amplification factor for low-volume speech",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=0.0018,
        help="Peak threshold below which audio is treated as silence",
    )
    parser.add_argument(
        "--output",
        default="outputs/live_transcript.txt",
        help="Append transcript lines to this file",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List audio devices and exit",
    )
    parser.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="Optional sounddevice input index",
    )
    return parser.parse_args()


def list_devices() -> None:
    print(sd.query_devices())


def write_temp_wav(audio: np.ndarray, sample_rate: int) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sample_rate)
        return tmp.name


def main() -> None:
    args = parse_args()

    if args.list_devices:
        list_devices()
        return

    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    stop_event = threading.Event()
    audio_queue: Queue[np.ndarray] = Queue()

    def on_sigint(_signum: int, _frame: object) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, on_sigint)

    def callback(indata: np.ndarray, frames: int, time: object, status: object) -> None:
        del frames, time
        if status:
            print(f"Audio callback status: {status}")
        audio_queue.put(indata.copy())

    sample_rate = pick_sample_rate(args.input_device, args.sample_rate)
    blocksize = int(sample_rate * args.chunk_seconds)
    language = args.language.strip() or None
    device_idx, device_name = describe_selected_device(args.input_device)

    print(
        f"Starting microphone stream. device={device_idx} ({device_name}), "
        f"sample_rate={sample_rate}. Press Ctrl+C to stop."
    )

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        callback=callback,
        blocksize=blocksize,
        device=args.input_device,
    ):
        while not stop_event.is_set():
            chunks: List[np.ndarray] = []
            try:
                while True:
                    chunks.append(audio_queue.get(timeout=0.2))
                    if sum(chunk.shape[0] for chunk in chunks) >= blocksize:
                        break
            except Empty:
                if not chunks:
                    continue

            audio = np.concatenate(chunks, axis=0).squeeze()
            if audio.ndim != 1:
                audio = audio.reshape(-1)

            boosted_audio, _, peak, _ = boost_quiet_audio(
                audio,
                target_rms=args.target_rms,
                max_gain=args.max_gain,
                silence_threshold=args.silence_threshold,
            )

            if peak < args.silence_threshold:
                continue

            wav_path = write_temp_wav(boosted_audio, sample_rate)
            try:
                segments, _ = model.transcribe(
                    wav_path,
                    language=language,
                    beam_size=args.beam_size,
                    vad_filter=args.vad,
                    task="transcribe",
                )
                text = " ".join(seg.text.strip() for seg in segments).strip()
                if text:
                    print(text)
                    with output_path.open("a", encoding="utf-8") as f:
                        f.write(text + "\n")
            finally:
                Path(wav_path).unlink(missing_ok=True)

    print("Stopped.")


if __name__ == "__main__":
    main()

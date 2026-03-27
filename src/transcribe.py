import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from faster_whisper import WhisperModel


def parse_args() -> argparse.Namespace:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="GPU-accelerated local transcription with faster-whisper."
    )
    parser.add_argument("--input", required=True, help="Path to input audio/video file")
    parser.add_argument(
        "--output",
        default="outputs/transcript.txt",
        help="Where to write plain text transcript",
    )
    parser.add_argument(
        "--json-output",
        default="outputs/transcript.segments.json",
        help="Where to write JSON segments",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("WHISPER_MODEL", "large-v3-turbo"),
        help="Whisper model name",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("WHISPER_DEVICE", "cuda"),
        choices=["cuda", "cpu", "auto"],
        help="Inference device",
    )
    parser.add_argument(
        "--compute-type",
        default=os.getenv("WHISPER_COMPUTE_TYPE", "float16"),
        help="CTranslate2 compute type (e.g. float16, int8_float16, int8)",
    )
    parser.add_argument(
        "--language",
        default=os.getenv("WHISPER_LANGUAGE", "en"),
        help="Language code (e.g. en, ur). Leave empty for auto-detect.",
    )
    parser.add_argument(
        "--task",
        default=os.getenv("WHISPER_TASK", "transcribe"),
        choices=["transcribe", "translate"],
        help="Whisper task",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=int(os.getenv("WHISPER_BEAM_SIZE", "1")),
        help="Beam search width; 1 is fastest",
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        default=os.getenv("WHISPER_VAD", "true").lower() in {"1", "true", "yes"},
        help="Enable VAD filtering to reduce silence hallucinations",
    )
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable VAD even if enabled by environment",
    )

    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def serialize_segments(segments: Any) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    for segment in segments:
        data.append(
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text.strip(),
                "avg_logprob": getattr(segment, "avg_logprob", None),
                "no_speech_prob": getattr(segment, "no_speech_prob", None),
            }
        )
    return data


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    text_out = Path(args.output).expanduser().resolve()
    json_out = Path(args.json_output).expanduser().resolve()
    ensure_parent(text_out)
    ensure_parent(json_out)

    use_vad = args.vad and not args.no_vad
    language = args.language.strip() or None

    print(
        f"Loading model={args.model} device={args.device} compute_type={args.compute_type} ..."
    )
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    print(f"Transcribing: {input_path}")
    segments, info = model.transcribe(
        str(input_path),
        language=language,
        task=args.task,
        beam_size=args.beam_size,
        vad_filter=use_vad,
    )

    segment_list = serialize_segments(segments)
    transcript = " ".join(s["text"] for s in segment_list).strip()

    text_out.write_text(transcript + "\n", encoding="utf-8")
    json_out.write_text(json.dumps(segment_list, indent=2, ensure_ascii=False), encoding="utf-8")

    detected = info.language if hasattr(info, "language") else "unknown"
    prob = info.language_probability if hasattr(info, "language_probability") else None

    print("Done.")
    print(f"Text output : {text_out}")
    print(f"JSON output : {json_out}")
    print(f"Detected language: {detected} (p={prob})")
    print(f"Segments: {len(segment_list)}")


if __name__ == "__main__":
    main()

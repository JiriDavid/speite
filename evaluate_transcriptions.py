#!/usr/bin/env python3
"""
Evaluate transcription quality on sample clips using word error rate.
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

from speite.config import settings
from speite.core import SpeechToTextService
from speite.utils import AudioPreprocessor


FAST_EVAL_TRANSCRIBE_OPTIONS = {
    "beam_size": 1,
    "best_of": 1,
    "temperature": 0.0,
    "condition_on_previous_text": False,
    "without_timestamps": True,
    "verbose": False,
}

STRONG_DECODE_OPTIONS = {
    "beam_size": 5,
    "best_of": 5,
    "patience": 0.2,
    "temperature": 0.0,
    "condition_on_previous_text": False,
    "without_timestamps": True,
    "verbose": False,
}


def resolve_manifest_path(manifest_arg: Path) -> Path:
    """Resolve the manifest path with a fallback to the evaluation directory."""
    candidates = []

    if manifest_arg.is_absolute():
        candidates.append(manifest_arg)
    else:
        candidates.append((Path.cwd() / manifest_arg).resolve())
        candidates.append((Path.cwd() / "evaluation" / manifest_arg).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Manifest file not found. Checked:\n"
        f"{searched}\n\n"
        "Try either:\n"
        "  python evaluate_transcriptions.py evaluation/manifest.example.json\n"
        "or:\n"
        "  python evaluate_transcriptions.py manifest.example.json"
    )


def normalize_text(text: str) -> str:
    """Normalize text for stable WER comparisons."""
    normalized = text.lower()
    normalized = re.sub(r"[^a-z0-9']+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def word_error_stats(reference: str, hypothesis: str) -> tuple[int, int, float]:
    """Return edit count, reference word count, and WER."""
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()

    if not ref_words:
        raise ValueError("Reference transcript is empty after normalization")

    rows = len(ref_words) + 1
    cols = len(hyp_words) + 1
    distance = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        distance[i][0] = i
    for j in range(cols):
        distance[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            substitution_cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            distance[i][j] = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                distance[i - 1][j - 1] + substitution_cost,
            )

    edits = distance[-1][-1]
    word_count = len(ref_words)
    return edits, word_count, edits / word_count


def average_logprob(transcription: dict) -> Optional[float]:
    """Compute the average segment log probability if available."""
    segments = transcription.get("segments") or []
    logprobs = [segment.get("avg_logprob") for segment in segments if segment.get("avg_logprob") is not None]
    if not logprobs:
        return None
    return float(np.mean(logprobs))


DECODE_PROFILES = {
    "fast": FAST_EVAL_TRANSCRIBE_OPTIONS,
    "default": {},
    "strong": STRONG_DECODE_OPTIONS,
}


def build_transcribe_options(decode_profile: str) -> dict:
    """Return a copy of the decode options for the requested profile."""
    if decode_profile not in DECODE_PROFILES:
        raise ValueError(f"Unsupported decode profile: {decode_profile}")
    return dict(DECODE_PROFILES[decode_profile])


def resolve_prompt_value(prompt: Optional[str], prompt_file: Optional[Path]) -> Optional[str]:
    """Resolve prompt text from inline value or file reference."""
    if prompt and prompt_file:
        raise ValueError("Provide either --prompt or --prompt-file, not both")
    if prompt_file:
        if not prompt_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
        return prompt_file.read_text(encoding="utf-8").strip()
    return prompt.strip() if prompt else None


def resolve_path(manifest_path: Path, value: str) -> Path:
    """Resolve a manifest-relative path."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def load_reference(entry: dict, manifest_path: Path) -> str:
    """Load reference text from inline text or file path."""
    if entry.get("reference_text"):
        return entry["reference_text"]

    reference_path_value = entry.get("reference_path")
    if not reference_path_value:
        raise ValueError("Manifest entry requires reference_text or reference_path")

    reference_path = resolve_path(manifest_path, reference_path_value)
    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference transcript not found for clip '{entry.get('name', 'unknown')}'. "
            f"Expected: {reference_path}"
        )
    return reference_path.read_text(encoding="utf-8")


def load_manifest(manifest_path: Path) -> list[dict]:
    """Load and validate the evaluation manifest."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Manifest must be a JSON array of clip definitions")
    return payload


def preprocess_audio(
    preprocessor: AudioPreprocessor,
    audio_path: Path,
    profile: str,
) -> tuple[np.ndarray, float]:
    """Load and preprocess audio for a given evaluation profile."""
    audio_data, sample_rate = preprocessor.load_audio(str(audio_path))
    audio_data = preprocessor.resample_audio(audio_data, sample_rate)
    duration_seconds = len(audio_data) / settings.sample_rate

    if profile == "none":
        audio_data = np.asarray(audio_data, dtype=np.float32)
        preprocessor.validate_audio(audio_data)
        return audio_data, duration_seconds

    return preprocessor.preprocess_array(audio_data, profile=profile), duration_seconds


def get_audio_duration_seconds(audio_path: Path) -> float:
    """Read audio duration without running the full preprocessing pipeline."""
    preprocessor = AudioPreprocessor()
    audio_data, sample_rate = preprocessor.load_audio(str(audio_path))
    return len(audio_data) / sample_rate if sample_rate else 0.0


def summarize_workload(manifest_path: Path, entries: list[dict], profiles: list[str]) -> None:
    """Print the expected evaluation workload before running inference."""
    total_audio_seconds = 0.0
    for entry in entries:
        audio_path = resolve_path(manifest_path, entry["audio_path"])
        total_audio_seconds += get_audio_duration_seconds(audio_path)

    total_runs = len(entries) * len(profiles)
    total_runtime_hours = (total_audio_seconds * len(profiles)) / 3600

    print(
        f"Evaluating {len(entries)} clip(s) across {len(profiles)} profile(s) "
        f"for {total_runs} run(s).",
        flush=True,
    )
    print(
        f"Total source audio per pass: {total_audio_seconds / 60:.1f} min; "
        f"aggregate audio processed: {total_runtime_hours:.2f} hours.",
        flush=True,
    )


def evaluate_profile(
    service: SpeechToTextService,
    preprocessor: AudioPreprocessor,
    manifest_path: Path,
    entries: list[dict],
    profile: str,
    transcribe_options: dict,
    decode_profile: str,
    fallback_options: Optional[dict],
    fallback_profile_name: Optional[str],
    retry_on_low_confidence: bool,
    low_avg_logprob_threshold: float,
) -> list[dict]:
    """Evaluate one preprocessing profile across all manifest entries."""
    results = []

    for index, entry in enumerate(entries, start=1):
        audio_path = resolve_path(manifest_path, entry["audio_path"])
        reference_text = load_reference(entry, manifest_path)
        clip_name = entry.get("name", audio_path.name)

        print(
            f"[{profile}] clip {index}/{len(entries)}: {clip_name}",
            flush=True,
        )

        processed_audio, duration_seconds = preprocess_audio(
            preprocessor,
            audio_path,
            profile,
        )

        print(
            f"[{profile}] preprocessed {duration_seconds / 60:.1f} min of audio; starting transcription...",
            flush=True,
        )

        started = time.perf_counter()
        transcription = service.transcribe(processed_audio, **dict(transcribe_options))
        elapsed = time.perf_counter() - started

        avg_logprob = average_logprob(transcription)
        redecoded = False

        needs_retry = (
            retry_on_low_confidence
            and fallback_options is not None
            and avg_logprob is not None
            and avg_logprob < low_avg_logprob_threshold
        )

        if needs_retry:
            print(
                f"[{profile}] low confidence (avg_logprob {avg_logprob:.2f} < {low_avg_logprob_threshold}); re-decoding with {fallback_profile_name or 'fallback'} settings...",
                flush=True,
            )
            retry_started = time.perf_counter()
            transcription = service.transcribe(processed_audio, **dict(fallback_options))
            retry_elapsed = time.perf_counter() - retry_started
            elapsed += retry_elapsed
            avg_logprob = average_logprob(transcription)
            redecoded = True

        edits, word_count, wer = word_error_stats(
            reference_text,
            transcription["text"],
        )

        results.append(
            {
                "clip": clip_name,
                "profile": profile,
                "edits": edits,
                "reference_words": word_count,
                "wer": wer,
                "duration_seconds": duration_seconds,
                "elapsed_seconds": elapsed,
                "real_time_factor": elapsed / duration_seconds if duration_seconds else 0.0,
                "hypothesis": transcription["text"],
                "avg_logprob": avg_logprob,
                "decode_profile": decode_profile,
                "redecoded": redecoded,
                "fallback_profile": fallback_profile_name if redecoded else None,
            }
        )

        print(
            f"[{profile}] finished {clip_name} in {elapsed:.1f}s "
            f"(RTF {elapsed / duration_seconds if duration_seconds else 0.0:.2f}, WER {wer * 100:.2f}%)",
            flush=True,
        )

    return results


def print_results(results: list[dict]) -> None:
    """Print per-clip and per-profile summaries."""
    print("\nPer-clip results")
    print("-" * 96)
    print(f"{'Profile':<10} {'Clip':<24} {'WER':>8} {'RTF':>8} {'Edits':>8} {'Words':>8}")
    print("-" * 96)
    for item in results:
        print(
            f"{item['profile']:<10} {item['clip']:<24} {item['wer'] * 100:>7.2f}% "
            f"{item['real_time_factor']:>7.2f} {item['edits']:>8} {item['reference_words']:>8}"
        )

    print("\nProfile summary")
    print("-" * 72)
    for profile in sorted({item["profile"] for item in results}):
        profile_items = [item for item in results if item["profile"] == profile]
        total_edits = sum(item["edits"] for item in profile_items)
        total_words = sum(item["reference_words"] for item in profile_items)
        average_rtf = sum(item["real_time_factor"] for item in profile_items) / len(profile_items)
        aggregate_wer = total_edits / total_words if total_words else 0.0
        print(
            f"{profile:<10} WER={aggregate_wer * 100:>7.2f}% "
            f"avg_RTF={average_rtf:>6.2f} clips={len(profile_items)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Speite transcriptions with word error rate",
    )
    parser.add_argument(
        "manifest",
        type=Path,
        help="Path to a JSON manifest with audio/reference pairs",
    )
    parser.add_argument(
        "--model",
        default="tiny",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model to evaluate (default: %(default)s)",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["none", "offline", "live"],
        choices=["none", "offline", "live"],
        help="Preprocessing profiles to compare",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path to write raw evaluation results as JSON",
    )
    parser.add_argument(
        "--decode-profile",
        choices=["fast", "default", "strong"],
        default="fast",
        help="Decode profile: fast (beam1/best1), default (Whisper defaults), strong (beam5/best5)",
    )
    parser.add_argument(
        "--accurate-decode",
        action="store_true",
        help="Deprecated: same as --decode-profile default",
    )
    parser.add_argument(
        "--retry-low-avg-logprob",
        action="store_true",
        help="Re-decode clips whose avg_logprob drops below the threshold using the strong profile",
    )
    parser.add_argument(
        "--low-avg-logprob-threshold",
        type=float,
        default=-1.0,
        help="Avg logprob threshold to trigger a re-decode (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        help="Initial prompt text to bias decoding (applied to all clips)",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="File containing the initial prompt text",
    )
    args = parser.parse_args()

    manifest_path = resolve_manifest_path(args.manifest)
    entries = load_manifest(manifest_path)
    summarize_workload(manifest_path, entries, args.profiles)

    decode_profile = args.decode_profile
    if args.accurate_decode:
        decode_profile = "default"

    transcribe_options = build_transcribe_options(decode_profile)

    prompt_value = resolve_prompt_value(args.prompt, args.prompt_file)
    if prompt_value:
        transcribe_options["initial_prompt"] = prompt_value

    fallback_profile_name: Optional[str] = None
    fallback_options: Optional[dict] = None
    if args.retry_low_avg_logprob and decode_profile != "strong":
        fallback_profile_name = "strong"
        fallback_options = build_transcribe_options("strong")
        if prompt_value:
            fallback_options["initial_prompt"] = prompt_value

    service = SpeechToTextService(model_name=args.model)
    service.load_model()
    preprocessor = AudioPreprocessor()

    all_results = []
    for profile in args.profiles:
        print(f"Starting profile: {profile}", flush=True)
        all_results.extend(
            evaluate_profile(
                service,
                preprocessor,
                manifest_path,
                entries,
                profile,
                transcribe_options,
                decode_profile,
                fallback_options,
                fallback_profile_name,
                args.retry_low_avg_logprob,
                args.low_avg_logprob_threshold,
            )
        )

    print_results(all_results)

    if args.json_output:
        args.json_output.write_text(
            json.dumps(all_results, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
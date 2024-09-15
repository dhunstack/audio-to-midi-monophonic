from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from .fileio import load_all_features, create_dir_if_not_exist, save_json
from .config import (
    OUTPUT_DIR_NAME,
    SEGMENT_THRESHOLD,
    MIN_NOTE_DURATION,
    MIN_VELOCITY,
    ONSET_THRESHOLD,
    MIN_SPLIT_DURATION,
    TRIM_THRESHOLD,
)
from .midi_framework import features_to_midi
from .feature_extractor import run as features_extractor_run


def run(
    audio_dir,
    feature_dir=None,
    output_dir=None,
    no_output_subfolders=False,
    segment_threshold=SEGMENT_THRESHOLD,
    min_note_duration=MIN_NOTE_DURATION,
    min_velocity=MIN_VELOCITY,
    onset_threshold=ONSET_THRESHOLD,
    min_split_duration=MIN_SPLIT_DURATION,
    trim_threshold=TRIM_THRESHOLD,
):

    audio_dir = Path(audio_dir)

    if not feature_dir:
        feature_dir = (
            audio_dir if audio_dir.is_dir() else audio_dir.parent
        ) / OUTPUT_DIR_NAME
    feature_dir = Path(feature_dir)

    # Check if feature directory exists
    if not feature_dir.exists():
        features_extractor_run(audio_dir, feature_dir)

    if not output_dir:
        output_dir = (
            audio_dir if audio_dir.is_dir() else audio_dir.parent
        ) / OUTPUT_DIR_NAME
    output_dir = Path(output_dir)

    # Create output directory if it does not exist
    create_dir_if_not_exist(output_dir)

    if audio_dir.is_file():
        wav_files = [audio_dir] if audio_dir.suffix == ".wav" else []
    else:
        wav_files = audio_dir.rglob("*.wav")

    for path in wav_files:

        print(f"Generating MIDI for {path.stem}")

        file_feature_dir = feature_dir / path.stem

        if not file_feature_dir.exists():
            print(f"Features not found in {file_feature_dir}")
            features_extractor_run(path, feature_dir)

        file_output_dir = output_dir if no_output_subfolders else output_dir / path.stem
        create_dir_if_not_exist(file_output_dir)

        onset_activations, time, frequency, confidence, rms = load_all_features(
            file_feature_dir
        )
        midi = features_to_midi(
            onset_activations,
            time,
            frequency,
            confidence,
            rms,
            segment_threshold=SEGMENT_THRESHOLD,
            min_note_duration=MIN_NOTE_DURATION,
            min_velocity=MIN_VELOCITY,
            onset_threshold=ONSET_THRESHOLD,
            min_split_duration=MIN_SPLIT_DURATION,
            trim_threshold=TRIM_THRESHOLD,
        )
        midi.write(str(file_output_dir / f"{path.stem}.mid"))

        print(f"MIDI file saved in {file_output_dir}")

    config = {
        "segment_threshold": segment_threshold,
        "min_note_duration": min_note_duration,
        "min_velocity": min_velocity,
        "onset_threshold": onset_threshold,
        "min_split_duration": min_split_duration,
        "trim_threshold": trim_threshold,
    }
    # Save config
    save_json(output_dir / "config.json", config)


def main():
    """
    This is a script for generating MIDI files from audio files in a directory.

    Usage:
    python midi_generator.py <path_to_audio_dir>
    """

    parser = ArgumentParser()
    parser.add_argument(
        "audio_dir", type=Path, help="Path to the directory containing audio files"
    )
    parser.add_argument(
        "--feature_dir",
        type=Path,
        help="Path to the directory containing feature files",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Path to the directory where the MIDI files will be saved",
    )
    parser.add_argument(
        "--no_output_subfolders",
        action="store_true",
        help="Do not create subfolders in the output directory",
    )
    parser.add_argument(
        "--segment_threshold",
        type=float,
        default=SEGMENT_THRESHOLD,
        help="Threshold for segmenting the note boundary activations",
    )
    parser.add_argument(
        "--min_note_duration",
        type=float,
        default=MIN_NOTE_DURATION,
        help="Minimum duration of a note",
    )
    parser.add_argument(
        "--min_velocity",
        type=int,
        default=MIN_VELOCITY,
        help="Minimum velocity of a note",
    )
    parser.add_argument(
        "--onset_threshold",
        type=float,
        default=ONSET_THRESHOLD,
        help="Threshold for onset detection",
    )
    parser.add_argument(
        "--min_split_duration",
        type=float,
        default=MIN_SPLIT_DURATION,
        help="Minimum duration for splitting the note boundaries",
    )
    parser.add_argument(
        "--trim_threshold",
        type=float,
        default=TRIM_THRESHOLD,
        help="Threshold for trimming the note boundaries",
    )

    args = parser.parse_args()

    run(
        args.audio_dir,
        args.feature_dir,
        args.output_dir,
        args.no_output_subfolders,
        args.segment_threshold,
        args.min_note_duration,
        args.min_velocity,
        args.onset_threshold,
        args.min_split_duration,
        args.trim_threshold,
    )


if __name__ == "__main__":
    main()

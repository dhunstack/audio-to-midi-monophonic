from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from .fileio import load_all_features
from .config import OUTPUT_DIR_NAME
from .midi_framework import features_to_midi
from .feature_extractor import run as features_extractor_run


def run(audio_dir):

    if audio_dir.is_file():
        output_dir = Path(audio_dir).parent / OUTPUT_DIR_NAME
        wav_files = [audio_dir]
    else:
        output_dir = Path(audio_dir) / OUTPUT_DIR_NAME
        wav_files = Path(audio_dir).rglob("*.wav")

    # Check if output directory exists
    if not output_dir.exists():
        features_extractor_run(audio_dir)

    for path in wav_files:

        print(f"Processing file {path.stem}")

        file_output_dir = output_dir / path.stem
        if not file_output_dir.exists():
            print(f"Features not found in {file_output_dir}")
            features_extractor_run(path)

        onset_activations, time, frequency, confidence, rms = load_all_features(
            file_output_dir
        )
        midi = features_to_midi(onset_activations, time, frequency, confidence, rms)
        midi.write(str(file_output_dir / f"{path.stem}.mid"))

        print(f"MIDI file saved in {file_output_dir}")


def main():

    parser = ArgumentParser(
        description=main.__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "audio_dir", type=Path, help="Path to the directory containing audio files"
    )

    args = parser.parse_args()

    run(args.audio_dir)


if __name__ == "__main__":
    main()

from .feature_utils import FeaturesExtractor
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from .fileio import create_dir_if_not_exist, save_all_features
from .config import OUTPUT_DIR_NAME


def run(audio_dir, crepe_model_capacity="full"):
    features_extractor = FeaturesExtractor(crepe_model_capacity)

    if audio_dir.is_file():
        output_dir = Path(audio_dir).parent / OUTPUT_DIR_NAME
        wav_files = [audio_dir]
    else:
        output_dir = Path(audio_dir) / OUTPUT_DIR_NAME
        wav_files = Path(audio_dir).rglob("*.wav")

    create_dir_if_not_exist(output_dir)

    for path in wav_files:

        print(f"Processing file {path.stem}")

        file_output_dir = output_dir / path.stem
        if file_output_dir.exists():
            print(f"Features previously extracted in {file_output_dir}")
            continue
        onset_activations, time, frequency, confidence, rms = (
            features_extractor.get_all_features(str(path))
        )
        save_all_features(
            file_output_dir, onset_activations, time, frequency, confidence, rms
        )

        print(f"Features saved in {file_output_dir}")


def main():
    """
    This is a script for generating and saving features from audio files in a directory.
    The features include onset times, time, frequency, confidence and RMS energy.
    The features are saved in 'outputs' directory in the same directory as the audio files.
    Each audio file has its own directory in the 'output' directory.
    The features are saved as pickle files.

    Usage:
    python main.py <path_to_audio_dir_or_file>
    """

    parser = ArgumentParser(
        description=main.__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "audio_dir", type=Path, help="Path to the directory containing audio files"
    )
    parser.add_argument(
        "--model-capacity",
        "-c",
        choices=["tiny", "full"],
        default="full",
        help="Model capacity of CREPE",
    )

    args = parser.parse_args()

    run(args.audio_dir, args.model_capacity)


if __name__ == "__main__":
    main()

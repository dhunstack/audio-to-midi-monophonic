
# Audio to MIDI Monophonic

This Python project generates monophonic MIDI files from audio files.
The script extracts onsets, pitch and rms energy from audio, processes them, and creates corresponding MIDI files.

## Quick Usage

Install with pip the provided `.whl` file.

Generate MIDI for audio directory or single wav file with following command -

```bash
python -m audiotomidimonophonic <path_to_audio_dir_or_file>
```

The outputs will be saved in the `outputs` directory at the provided path.

## Requirements

- Python 3.x (tested with Python 3.10)
- The following Python libraries:

  ```bash
    python = "^3.10.14"
    madmom = "^0.17.dev0"
    crepe = "^0.0.16"
    librosa = "^0.10.2.post1"
    pretty_midi = "^0.2.10"
    numpy = "^1.26.4"
    scipy = "^1.14.1"
  ```

## Installation

Build via `pyproject.toml` file using `poetry`:

```bash
poetry build
poetry install
```

 OR
 
Set up a virtual environment and manually install dependencies -

```bash
pip install numpy scipy librosa crepe madmom pretty_midi
```

## Detailed Usage

```bash
python -m audiotomidimonophonic <path_to_audio_dir> [--feature_dir <path_to_feature_dir>] [--output_dir <path_to_output_dir>] [--no_output_subfolders] [--segment_threshold <float>] [--min_note_duration <float>] [--min_velocity <int>] [--onset_threshold <float>] [--min_split_duration <float>] [--trim_threshold <float>]
```

### Arguments:

- **audio_dir**: Path to the directory containing `.wav` files for MIDI generation.
- **--feature_dir**: (Optional) Path to the directory containing precomputed features. If not provided, features will be computed automatically.
- **--output_dir**: (Optional) Path to save the generated MIDI files. Defaults to a subdirectory within `audio_dir`.
- **--no_output_subfolders**: (Optional) If set, prevents creating subfolders in the output directory for each audio file.
- **--segment_threshold**: (Optional) Threshold for segmenting note boundary activations (default from `config.py`).
- **--min_note_duration**: (Optional) Minimum duration of a note in seconds (default from `config.py`).
- **--min_velocity**: (Optional) Minimum velocity of a note (default from `config.py`).
- **--onset_threshold**: (Optional) Threshold for onset detection (default from `config.py`).
- **--min_split_duration**: (Optional) Minimum duration for splitting note boundaries (default from `config.py`).
- **--trim_threshold**: (Optional) Threshold for trimming note boundaries (default from `config.py`).

### Example Usage:

1. To generate MIDI files for all `.wav` files in a directory:
   
   ```bash
   python -m audiotomidimonophonic /path/to/audio/files
   ```

2. To specify a custom output directory:
   
   ```bash
   python -m audiotomidimonophonic /path/to/audio/files --output_dir /path/to/output
   ```

3. To skip creating subfolders for each file:
   
   ```bash
   python -m audiotomidimonophonic /path/to/audio/files --no_output_subfolders
   ```

### Configurable Parameters:

- **SEGMENT_THRESHOLD**: Controls the threshold for segmenting note boundary activations.
- **MIN_NOTE_DURATION**: Minimum note duration in seconds.
- **MIN_VELOCITY**: Minimum velocity value for a note.
- **ONSET_THRESHOLD**: Threshold for onset detection.
- **MIN_SPLIT_DURATION**: Minimum duration for note splits.
- **TRIM_THRESHOLD**: Threshold for trimming note boundaries.

These parameters can be set via the command line or the `config.py` file.

## File Structure

```
├── midi_generator.py          # Main script for MIDI generation
├── config.py                  # Configuration settings and thresholds
├── fileio.py                  # Helper functions for file I/O
├── midi_framework.py          # Functions for converting features to MIDI
├── feature_extractor.py       # Main script for feature extraction
└── feature_utils.py           # Functions for extracting features from audio files
```


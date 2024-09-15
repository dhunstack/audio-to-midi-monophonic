import pretty_midi
import numpy as np
import librosa
import scipy
from .config import (
    SEGMENT_THRESHOLD,
    MIN_NOTE_DURATION,
    MIN_VELOCITY,
    FPS,
    ONSET_THRESHOLD,
    TRIM_THRESHOLD,
)


class Note:
    def __init__(self, pitch, start, end, velocity):
        self.pitch = pitch
        self.start = start
        self.end = end
        self.velocity = velocity

    def __repr__(self):
        return f"Note(pitch={self.pitch}, start={self.start}, end={self.end}, velocity={self.velocity})"


def features_to_midi(onset_activations, time, frequency, confidence, rms):

    # Scale rms amplitude to [0, 127]
    midi_velocity = rms_to_velocity(rms)

    # Convert frequency to MIDI pitch and get the pitch gradient
    midi_pitch, pitch_gradient = compute_midi_pitch_and_gradient(frequency)

    # Compute note segments
    note_segments = compute_note_segments(confidence, pitch_gradient)

    # Create Note instances
    notes = create_notes(note_segments, midi_pitch, midi_velocity)

    # Merge adjacent notes with similar pitch
    notes = merge_notes(notes)

    # Remove short and low amplitude notes
    notes = remove_short_quiet_notes(notes)

    # Split Notes with strong onset activations
    onset_activations = threshold_onset_activations(onset_activations)
    notes = split_notes(notes, onset_activations)

    # Trim Note boundaries
    notes = trim_notes(notes, midi_velocity, trim_threshold=TRIM_THRESHOLD)

    # Get MIDI object
    midi = make_midi(notes, time)

    return midi


def rms_to_velocity(rms):
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    rms_clip = np.clip(rms, 0, rms_mean + 6 * rms_std)
    midi_velocity = np.interp(rms_clip, (0, rms_clip.max()), (0, 127))
    return midi_velocity


def compute_midi_pitch_and_gradient(frequency):
    midi_pitch = librosa.hz_to_midi(frequency)
    pitch_gradient = np.abs(np.gradient(midi_pitch))
    pitch_gradient_scaled = np.interp(
        pitch_gradient, (pitch_gradient.min(), pitch_gradient.max()), (0, 1)
    )
    return midi_pitch, pitch_gradient_scaled


def compute_note_segments(
    confidence, pitch_gradient, segment_threshold=SEGMENT_THRESHOLD
):
    """
    Compute note segments from the confidence and the scaled pitch gradient

    Parameters
    ----------
    confidence : np.ndarray
        Confidence of the pitch tracker
    pitch_gradient_scaled : np.ndarray
        Scaled pitch gradient
    segment_threshold : float
        Threshold for segmenting notes

    Returns
    -------
    note_segments : list
        List of tuples containing the start and end indices of the note segments
    """

    # Compute the segmentation signal
    seg_signal = (1 - confidence) * pitch_gradient
    scaled_seg_signal = np.interp(
        seg_signal, (seg_signal.min(), seg_signal.max()), (0, 1)
    )

    # Find boundary points and their widths
    boundary_points, _ = scipy.signal.find_peaks(
        scaled_seg_signal, distance=4, prominence=segment_threshold
    )
    _, _, boundary_starts, boundary_ends = scipy.signal.peak_widths(
        scaled_seg_signal, boundary_points
    )

    # Convert boundary points to note starts and ends
    # End of boundary is the start of the note
    # Start of boundary is the end of the note
    note_starts = [0] + [int(p) for p in np.round(boundary_ends)]
    note_ends = [int(p) for p in np.round(boundary_starts)] + [len(confidence)]

    # Get note segments
    note_segments = [(s, e) for s, e in zip(note_starts, note_ends) if e - s > 1]
    return note_segments


def create_notes(note_segments, midi_pitch, midi_velocity):
    notes = []
    for start, end in note_segments:
        pitch = np.median(midi_pitch[start:end])
        velocity = np.max(midi_velocity[start:end])
        notes.append(Note(pitch, start, end, velocity))
    return notes


def merge_notes(notes):
    f_notes = []
    comb = []

    def combine_notes():
        if len(comb) == 1:
            f_notes.append(comb[0])
        elif len(comb) > 1:
            start = comb[0].start
            end = comb[-1].end
            pitch = np.median([note.pitch for note in comb])
            velocity = np.max([note.velocity for note in comb])
            f_notes.append(Note(pitch, start, end, velocity))

    for n1, n2 in zip(notes, notes[1:]):
        if np.abs(n2.pitch - n1.pitch) < 0.5:
            comb.append(n1)
        else:
            comb.append(n1)
            combine_notes()
            comb = []

    combine_notes()  # Combine the last notes if remaining
    return f_notes


def remove_short_quiet_notes(
    notes, min_duration=MIN_NOTE_DURATION, min_velocity=MIN_VELOCITY
):
    f_notes = [
        note
        for note in notes
        if note.end - note.start > min_duration * FPS and note.velocity > min_velocity
    ]
    return f_notes


def threshold_onset_activations(onset_activations, onset_threshold=ONSET_THRESHOLD):
    onset_indices = scipy.signal.find_peaks(
        onset_activations, distance=4, height=onset_threshold
    )[0]
    onset_activations = np.zeros_like(onset_activations)
    onset_activations[onset_indices] = 1
    return onset_activations


def split_notes(notes, onset_activations, min_duration=MIN_NOTE_DURATION):
    f_notes = []

    for note in notes:
        if np.any(onset_activations[note.start : note.end]):
            split_indices = np.where(onset_activations[note.start : note.end])[0]
            split_indices = split_indices + note.start
            split_indices = np.append(split_indices, note.end)
            prev_split = note.start

            for split in split_indices:
                if split - prev_split > min_duration * FPS:
                    f_notes.append(Note(note.pitch, prev_split, split, note.velocity))
                    prev_split = split

        else:
            f_notes.append(note)

    return f_notes


def trim_notes(notes, midi_velocity, trim_threshold=TRIM_THRESHOLD):

    for note in notes:
        start = note.start
        end = note.end
        while start < end and midi_velocity[start] < trim_threshold:
            start += 1
        while start < end and midi_velocity[end - 1] < trim_threshold:
            end -= 1
        note.start = start
        note.end = end

    return notes


def make_midi(notes, time):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(0)
    for note in notes:
        note_instance = pretty_midi.Note(
            velocity=int(round(note.velocity)),
            pitch=int(round(note.pitch)),
            start=time[note.start],
            end=time[note.end - 1],
        )
        instrument.notes.append(note_instance)
    midi.instruments.append(instrument)
    return midi


def write_midi(midi, output_path):
    midi.write(output_path)

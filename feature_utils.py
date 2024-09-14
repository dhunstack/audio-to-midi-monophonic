import madmom
import crepe
import librosa

class FeaturesExtractor:
    def __init__(self, crepe_model_capacity='full'):
        self.madmom_onset_generator = madmom.features.onsets.CNNOnsetProcessor()
        self.crepe_model_capacity = crepe_model_capacity

    def get_onset_activations(self, audio_path):
        """
        Extract onset times from audio file using madmom

        Args:
            audio_path (str): Path to the audio file
        
        Returns:
            np array : Onset times
        """
        onset_activations = self.madmom_onset_generator(audio_path)
        return onset_activations
    
    def get_crepe_outputs(self, audio_path):
        """
        Extract pitch activations from audio file using crepe

        Args:
            audio_path (str): Path to the audio file

        Returns:
            np array : Time
            np array : Frequency
            np array : Confidence
        """
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)    # Crepe requires 16kHz sampling rate
        time, frequency, confidence, _ = crepe.predict(audio, sr, viterbi=True, model_capacity=self.crepe_model_capacity)
        return time, frequency, confidence
    
    def get_rms_energy(self, audio_path):
        """
        Extract RMS energy from audio file using librosa

        Args:
            audio_path (str): Path to the audio file

        Returns:
            np array : RMS energy
        """
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        hop_length = sr//100   # 10ms hop length to match pitch and onset activations
        rms = librosa.feature.rms(y=audio, frame_length=2*hop_length, hop_length=hop_length)[0]
        return rms

    def get_all_features(self, audio_path):
        """
        Extract all features from audio file

        Args:
            audio_path (str): Path to the audio file

        Returns:
            np array : Onset times
            np array : Time
            np array : Frequency
            np array : Confidence
            np array : RMS energy
        """
        onset_activations = self.get_onset_activations(audio_path)
        time, frequency, confidence = self.get_crepe_outputs(audio_path)
        rms = self.get_rms_energy(audio_path)
        
        return onset_activations, time, frequency, confidence, rms
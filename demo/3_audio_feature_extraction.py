from tempocnn.classifier import TempoClassifier
from tempocnn.feature import read_features
import librosa
import numpy as np
import os
import pandas as pd
import scipy.stats

print()
print('#*' * 24)
print('🔵 Export the 19 audio features of each video')
print('*#' * 24)


def extract_and_save_features_from_file_list(
        audio_path_list: list[str],
) -> None:
    # Ensure the output directory exists
    output_dir: str = 'audio/features'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize dictionaries to aggregate 1D features if saving as CSV
    feature_csv_data: dict[str, list] = {}
    feature_max_lengths: dict[str, any] = {}

    # Initialize TempoClassifier model for tempo estimation
    tempo_classifier = TempoClassifier('cnn')

    # Loop through all audio files
    for index, audio_path in enumerate(audio_path_list):
        audio_name: str = os.path.splitext(os.path.basename(audio_path))[0]  # Get file name without extension
        print('-' * 48)
        print(f'📼 Video {(index + 1)}/{len(commercials_df)}:', commercial_id)
        # Open the single audio file
        y, sr = librosa.load(audio_path, sr=None)  # Load audio
        onset_env: np.ndarray = librosa.onset.onset_strength(
            y=y,
            sr=sr,
        )  # Onset Strength for tempo estimation
        # New params for reducing tempogram file size
        hop_length: int = 1024  # The default is 512
        tempogram_onset_env: np.ndarray = librosa.onset.onset_strength(
            y=y,
            sr=sr,
            hop_length=hop_length,
        )
        prior_uniform = scipy.stats.uniform(30, 300)  # uniform distribution over 30-300 BPM for tempo estimation
        prior_lognorm = scipy.stats.lognorm(
            loc=np.log(120),
            scale=120,
            s=1,
        )  # lognorm distribution for tempo estimation

        # Feature extraction
        features: dict[str, any] = {
            'chroma_stft': librosa.feature.chroma_stft(y=y, sr=sr),
            'chroma_cqt': librosa.feature.chroma_cqt(y=y, sr=sr),
            'chroma_cens': librosa.feature.chroma_cens(y=y, sr=sr),
            'melspectrogram': librosa.feature.melspectrogram(y=y, sr=sr),
            'mfcc': librosa.feature.mfcc(y=y, sr=sr),
            'rms': librosa.feature.rms(y=y)[0],
            'spectral_centroid': librosa.feature.spectral_centroid(y=y, sr=sr)[0],
            'spectral_bandwidth': librosa.feature.spectral_bandwidth(y=y, sr=sr)[0],
            'spectral_contrast': librosa.feature.spectral_contrast(y=y, sr=sr),
            'spectral_flatness': librosa.feature.spectral_flatness(y=y)[0],
            'spectral_rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr)[0],
            'poly_features': librosa.feature.poly_features(y=y, sr=sr, order=1)[1],  # First Order
            'tonnetz': librosa.feature.tonnetz(y=y, sr=sr),  # Tonal Centroids
            'zero_crossing_rate': librosa.feature.zero_crossing_rate(y=y)[0],
            'tempogram': librosa.feature.tempogram(y=y, sr=sr, onset_envelope=tempogram_onset_env),
            'librosa_static_tempo_default_prior': librosa.feature.tempo(onset_envelope=onset_env, sr=sr),
            'librosa_static_tempo_uniform_prior': librosa.feature.tempo(onset_envelope=onset_env, sr=sr,
                                                                        prior=prior_uniform),
            'librosa_static_tempo_lognorm_prior': librosa.feature.tempo(onset_envelope=onset_env, sr=sr,
                                                                        prior=prior_lognorm),
            'tempocnn_static_tempo': np.array(
                [tempo_classifier.estimate_tempo(read_features(audio_path), interpolate=False)]),
        }

        for feature_name in features.keys():
            # Ensure the feature output folder exists
            feature_dir: str = os.path.join(output_dir, f'{feature_name}_files')
            os.makedirs(feature_dir, exist_ok=True)

        # Save features
        for feature_name, feature_data in features.items():
            output_path: str = os.path.join(output_dir, f'{feature_name}_files', f'{audio_name}.{feature_name}.npz')
            np.savez_compressed(output_path, feature_data)

            print(f'🎶 Saved `{output_path}`')


commercials_df: pd.DataFrame = pd.read_csv('general/commercials.csv')

audio_path_list: list[str] = list()
for index, row in commercials_df.iterrows():
    commercial_id = row['commercial_id']
    audio_path_list.append(f'videos/{commercial_id}.mp4')

extract_and_save_features_from_file_list(
    audio_path_list=audio_path_list,
)

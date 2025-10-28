from faster_whisper import WhisperModel, BatchedInferencePipeline
from nltk.corpus import stopwords
from pydub import AudioSegment
from scipy.io import wavfile
from sklearn.feature_extraction.text import TfidfVectorizer
import ast
import csv
import nltk
import numpy as np
import os
import pandas as pd
import regex
import shutil
import stanza
import tensorflow as tf
import tensorflow_hub as hub

print()
print('#*' * 24)
print('🔵 1. Audio transcription')
print('➡️ Find the “Speech” presence in each video,\n'
      '➡️ transcribe the found speech.')
print('*#' * 24)

commercials_df: pd.DataFrame = pd.read_csv('general/commercials.csv')
# Load product types in Italian to give as a prompt to Whisper
product_types_df: pd.DataFrame = pd.read_csv(
    'general/product_types.csv',
    usecols=['product_type_key', 'product_type_it'],
    index_col='product_type_key',
)
# Create `tmp_wav` folder
print('-' * 48)
print('📂️ Create `tmp_wav` folder')
os.makedirs('tmp_wav', exist_ok=True)
# Create `text` folder
print('-' * 48)
print('📂️ Create `text` folder')
os.makedirs('text', exist_ok=True)
# Create `transcriptions` folder
print('-' * 48)
print('📂️ Create `text/transcriptions` folder')
os.makedirs('text/transcriptions', exist_ok=True)

print('-' * 48)
print('⏳️ Load YAMNet model from TensorFlow Hub (please wait…)')
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# The audio event class ID for “Speech”
speech_mid: str = '/m/09x0r'

audio_class_map_path: str = yamnet_model.class_map_path().numpy().decode('utf-8')
audio_class_map_df: pd.DataFrame = pd.read_csv(audio_class_map_path)

# Filter audio_class_map_df keeping only the “Speech” class ('/m/09x0r')
audio_class_map_df = audio_class_map_df[audio_class_map_df['mid'] == speech_mid]

# Whisper model name
whisper_model_name: str = 'large-v3'
# Initialize Whisper model
print('-' * 48)
print('⏳️ Initialize Whisper model (please wait…)')
whisper_model: WhisperModel = WhisperModel(model_size_or_path=whisper_model_name, device='cpu', compute_type='int8')

batched_model: BatchedInferencePipeline = BatchedInferencePipeline(model=whisper_model)

# The average log probability of the predictions for the words in this segment, indicating overall confidence.
# Set the minimum value in order to exclude segments with low confidence.
min_avg_logprob: float = -1

speech_class_confidence_score_df: pd.DataFrame = pd.DataFrame()
transcription_df_list: list[pd.DataFrame] = list()
for index, row in commercials_df.iterrows():
    commercial_id: str = row['commercial_id']
    # Get title, brand, product_type_key, and product type description (it)
    title: str = row['title']
    brand: str = row['brand']
    product_type_key: int = row['product_type_key']
    product_type_it: str = product_types_df.loc[product_type_key]['product_type_it']
    print('-' * 48)
    print(f'📼 Video {(index + 1)}/{len(commercials_df)}:', commercial_id)
    # Extract WAV audio track from MP4 video
    video_file_path: str = f'videos/{commercial_id}.mp4'
    wav_file_path: str = f'tmp_wav/{commercial_id}.wav'
    print(f'📄 Exported `wav_file_path`')
    audio: AudioSegment = AudioSegment.from_file(video_file_path, format='mp4')
    audio = audio.set_channels(1)  # Convert to mono
    audio = audio.set_frame_rate(16000)  # Set frame rate to 16 kHz
    audio.export(wav_file_path, format='wav')

    sample_rate, wav_data = wavfile.read(wav_file_path, 'rb')

    # The wav_data needs to be normalized to values in [-1.0, 1.0] (as stated in the model's documentation).
    waveform: np.ndarray = wav_data / tf.int16.max

    # Run the model, check the output.
    scores, embeddings, spectrogram = yamnet_model(waveform)
    scores_np = scores.numpy()
    spectrogram_np = spectrogram.numpy()

    # Total number of frames
    total_frames: int = scores_np.shape[0]

    # Calculate percentage for Speech audio event class
    new_row: dict = {'commercial_id': commercial_id}
    index: int = audio_class_map_df.loc[0, 'index']
    event_frames = scores_np[:, index] > 0.5  # Threshold for event detection
    speech_class_confidence_score: np.float64 = np.sum(event_frames) / total_frames

    new_row[speech_mid] = speech_class_confidence_score
    # Convert the row in a dataframe
    new_row_df: pd.DataFrame = pd.DataFrame([new_row])

    # Concatenate the new row to the speech_class_confidence_score_df
    speech_class_confidence_score_df = pd.concat([speech_class_confidence_score_df, new_row_df], ignore_index=True)

    # If “Speech” score is 0, continue

    transcription_path = f'text/transcriptions/{commercial_id}.transcription.csv'
    # Create a CSV file in "write" mode
    with open(transcription_path, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)

        # CSV header
        csv_writer.writerow([
            'commercial_id',
            # 'title',
            'brand',
            'transcription_list',
            'avg_logprob_list',
            'word_count_list',
            'speech_timecode_list',
            'speech_duration_list',
            'speech_speed_wps_list',
        ])
        # If the audio does not contain speech write an empty row and go on
        if speech_class_confidence_score == 0:
            csv_writer.writerow([
                commercial_id,  # commercial_id
                # title,  # title
                brand,  # brand
                [],  # transcription_list
                [],  # avg_logprob_list
                [],  # word_count_list
                [],  # speech_timecode_list
                [],  # speech_duration_list
                [],  # speech_speed_wps_list
            ])
            print('📄 No transcription found (“Speech” score = 0)')
            continue

        segments, info = batched_model.transcribe(
            audio=wav_file_path,
            beam_size=5,
            language='it',
            initial_prompt=title + '. ' + brand + '. ' + product_type_it,
            batch_size=8,
            log_progress=False,
            multilingual=True,
            suppress_blank=True,
            word_timestamps=True,
            hotwords=brand,
        )
        # print('==SEGMENTS==', segments, info)
        # segments = list(segments)  # The transcription will actually run here.
        result_text_list: list[str] = []
        result_avg_logprob_list: list[float] = []
        result_word_count_list: list[int] = []
        result_speech_timecode_list: list[tuple[float, float]] = []
        result_speech_duration_list: list[float] = []
        result_speech_speed_wps_list: list[float] = []
        for segment in segments:
            # Exclude segments with low confidence.
            if segment.avg_logprob <= min_avg_logprob:
                continue
            result_text_list.append(segment.text.strip())
            result_avg_logprob_list.append(segment.avg_logprob)
            word_count: int = len(segment.words)
            result_word_count_list.append(word_count)
            # Covert np.float64 to float with float()
            result_speech_timecode_list.append((float(segment.start), float(segment.end)))
            speech_duration: float = round(float(segment.end - segment.start), 2)
            result_speech_duration_list.append(speech_duration)
            result_speech_speed_wps_list.append(round(float(word_count / speech_duration), 2))
        print(f'📝 Found {len(result_text_list)} segment(s)')
        for index_t, t in enumerate(result_text_list):
            print(f'[Segment {index_t + 1}]', result_text_list[index_t])

        # If transcription is successful
        if len(result_text_list) > 0:
            # Save details in a new row
            csv_writer.writerow([
                commercial_id,
                # title,
                brand,
                result_text_list,  # transcription_list
                result_avg_logprob_list,  # avg_logprob_list
                result_word_count_list,  # word_count_list
                result_speech_timecode_list,  # speech_timecode_list
                result_speech_duration_list,  # speech_duration_list
                result_speech_speed_wps_list,  # speech_speed_wps_list
                # end - start,
            ])
            # has_been_transcribed = True
        else:
            # If transcription fails, write an empty row
            csv_writer.writerow([
                commercial_id,  # commercial_id
                # title,  # title
                brand,  # brand
                [],  # transcription_list
                [],  # avg_logprob_list
                [],  # word_count_list
                [],  # speech_timecode_list
                [],  # speech_duration_list
                [],  # speech_speed_wps_list
            ])
        print(f'📝 Saved `{transcription_path}`')
    # Load the transcription CSV file and save it to a DataFrame
    transcription_df: pd.DataFrame = pd.read_csv(transcription_path)
    # Append the transcription_df DataFrame to transcription_df_list
    transcription_df_list.append(transcription_df)

    # Delete temporary wav file
    os.remove(wav_file_path)
    print(f'🗑️ Deleted `{wav_file_path}`')

# Concatenate the transcription DataFrames in a unique DataFrame and save it
transcriptions_df: pd.DataFrame = pd.concat(transcription_df_list).reset_index(drop=True)
transcriptions_df.to_csv('text/transcriptions.csv', index=False)

# Delete `tmp_wav` folder
shutil.rmtree('tmp_wav')
print('-' * 48)
print(f'🗑️ Deleted `tmp_wav` folder')

# Delete `transcriptions` folder
shutil.rmtree('text/transcriptions')
print('-' * 48)
print(f'🗑️ Deleted `text/transcriptions` folder')

# Rename the '/m/09x0r' (speech_mid) as 'speech_class_confidence_score'
speech_class_confidence_score_df.rename(columns={speech_mid: 'speech_class_confidence_score'}, inplace=True)
# Save `speech_class_confidence_score.csv` file
speech_class_confidence_score_df.to_csv(f'audio/speech_class_confidence_score.csv', index=False)
print('-' * 48)
print(f'📄 Saved `audio/speech_class_confidence_score.csv`')

################

print()
print('#*' * 24)
print('🔵 2. Lemmatization')
print('➡️ Lemmatize each transcription,\n'
      '➡️ save lemmas alphabetically ordered.')
print('*#' * 24)

# Download necessary resources
print('-' * 48)
print('⏳️ Download Stanza Italian model (please wait…)')
stanza.download('it')  # Download Italian model

# Load Italian NLP model
print('-' * 48)
print('⏳️ Load Italian NLP model (please wait…)')
nlp = stanza.Pipeline(
    lang='it',
    processors='tokenize,mwt,pos,lemma,depparse,ner',
)

lemma_info_collection: list[str] = []
for index, row in transcriptions_df.iterrows():
    commercial_id: str = row['commercial_id']
    print('-' * 48)
    print(f'📼 Video {(index + 1)}/{len(commercials_df)}:', commercial_id)
    transcription_list_str: str = row['transcription_list']
    transcription_list: list[str] = ast.literal_eval(transcription_list_str)
    for index_t, transcription in enumerate(transcription_list):
        # Process text
        doc = nlp(transcription)

        # Store detailed lemma info
        lemma_info: list[dict] = list()
        # ✅ POS & Morphology → Identify verbs, nouns, adjectives, and grammatical features.
        # ✅ Dependency Relation → Extract syntactic relationships (e.g., subject-verb-object structure).
        # ✅ Sentence ID → Analyze word usage per sentence.
        # ✅ Head Word ID → Understand sentence structure by linking words to their governing words.
        for sent in doc.sentences:
            for word in sent.words:
                if word.upos != 'PUNCT':  # Exclude punctuation
                    lemma_info.append({
                        'commercial_id': commercial_id,
                        # Remove hyphens and/or dots at the beginning and end of original words and lemmas
                        # 'original': regex.sub(pattern='^[-.]+|[-.]+$', repl='', string=word.text),
                        # Store lemma in lowercase if UPOS is not 'PROPN'
                        'lemma': regex.sub(pattern='^[-.]+|[-.]+$', repl='',
                                           string=(word.lemma if word.upos == 'PROPN' else word.lemma.lower())),
                        'POS': word.xpos,  # Fine-grained POS (e.g., 'NOUN', 'VERB')
                        'UPOS': word.upos,  # Universal POS
                        # 'morphology': word.feats,  # Morphological info (gender, number, tense, etc.)
                        # 'dependency': word.deprel,  # Dependency relation
                        # 'head_word_id': word.head,  # Governing word ID
                        # 'sentence_id': sent.index  # Sentence number
                    })
        lemma_info_collection.extend(lemma_info)

# Convert lemma_info_collection_df to DataFrame
lemma_info_collection_df = pd.DataFrame(lemma_info_collection)

# Sort lemmas alphabetically within each commercial_id, case-insensitive
lemma_info_collection_df = lemma_info_collection_df.groupby(
    'commercial_id',
    group_keys=False,
    sort=False,
).apply(
    lambda g: g.sort_values('lemma', key=lambda x: x.str.lower())
    .reset_index(drop=True))

# Save lemma_info_collection_df to CSV
lemma_info_collection_df.to_csv('text/lemmas.csv', index=False)

# Delete `text/transcriptions.csv`
# os.remove('text/transcriptions.csv')
# print(f'🗑️ Deleted `text/transcriptions.csv`')

###################

print()
print('#*' * 24)
print('🔵 3. Calculate the tf-idf values of each lemma')
print('➡️ Calculate the tf-idf values,\n'
      '➡️ update `lemmas.csv`.')
print('*#' * 24)

# Download necessary resources
print('-' * 48)
print('⏳️ Download nltk  package `punkt` (please wait…)')
nltk.download('punkt')

print('-' * 48)
print('⏳️ Download nltk  package `stopwords` (please wait…)')
nltk.download('stopwords')

# Italian stopwords
stop_words: list[str] = list(stopwords.words('italian'))

## Create collection of document to analyze
doc_collection: dict = dict()
commercial_id_list: list[str] = list()
for index, row in lemma_info_collection_df.iterrows():
    commercial_id: str = row['commercial_id']
    commercial_id_list.append(commercial_id) if commercial_id not in commercial_id_list else None
    # Add empty doc to doc_collection
    if commercial_id not in doc_collection:
        doc_collection[commercial_id] = []
    doc_collection[commercial_id].append(row['lemma'])

doc_collection = {k: ' '.join(v) for k, v in doc_collection.items()}

## Calculate tf-idf
vectorizer: TfidfVectorizer = TfidfVectorizer(
    stop_words=stop_words,
    lowercase=False,
)
# Generate TF-IDF matrix
tfidf_matrix: any = vectorizer.fit_transform(doc_collection.values())
# Get feature names (words)
feature_names: np.ndarray = vectorizer.get_feature_names_out()
# Convert to DataFrame for better visualization
tfidf_df: pd.DataFrame = pd.DataFrame(
    tfidf_matrix.toarray(),
    index=commercial_id_list,
    columns=feature_names,
)  # .reset_index()
tfidf_df.index.name = 'commercial_id'


## Add corresponding tfidf to each lemma
# Try to fetch the tf-idf values using DataFrame indexing
def fetch_tfidf(row):
    try:
        return tfidf_df.at[row['commercial_id'], row['lemma']]
    except KeyError:
        key_error_lemmas.add(row['lemma'])
        return None  # Or np.nan if needed


# Initialize the set to track missing lemmas
key_error_lemmas: set[str] = set()

# Apply the function across all rows
lemma_info_collection_df['tf_idf'] = lemma_info_collection_df.apply(fetch_tfidf, axis=1)

# Print missing lemmas
print('-' * 48)
print('⚠️ Lemmas without tf-idf:', key_error_lemmas)

# Save to CSV
lemma_info_collection_df.to_csv('text/lemmas.csv', index=False)
print('-' * 48)
print(f'📄 Updated `text/lemmas.csv`')

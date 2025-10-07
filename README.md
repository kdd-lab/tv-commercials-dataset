# Italian TV Commercials Dataset – Methodology

This repository provides a **demo pipeline** that illustrates the methodology used to create the  
[Italian TV Commercials Dataset (1980–2024)](https://doi.org/10.5281/zenodo.17127245)  
(over 10,000 commercials with video, audio, colour, and text annotations).

The goal of this repository is to show how the dataset was generated starting from raw **MP4 video files**.  
It includes a **Jupyter Notebook** and supporting Python scripts that reproduce the main steps of the pipeline:
scene segmentation, thumbnail extraction, colour palette mapping, audio feature extraction, and text processing.

---

## Requirements

The environment requires Python ≥3.11 an environment.yml file is provided for Conda users. 

Install with:
```bash
conda env create -f environment.yml
conda activate tv-demo
```

---

## Repository Structure

```
.
├── demo_notebook.ipynb                          # walkthrough of the entire pipeline
├── 1_color_and_thumb_extraction.py
├── 2_ref_palette_idf_calculation.py
├── 3_audio_feature_extraction.py
├── 4_audio_transcription_and_lemmatization.py
├── initial_data/
│   └── commercials_initial_metadata.csv         # (user-provided)
├── videos/                                      # raw MP4 commercials (user-provided)
├── general/                                     # output: commercials.csv, scenes.csv, etc.
├── colors/                                      # output: reference palettes + IDFs
├── audio/                                       # output: features and speech confidence
├── text/                                        # output: lemmas and transcriptions
└── thumbnails/                                  # output: scene thumbnails
```

---

## Before Starting

1. Collect your **commercial videos** in MP4 format.  
   - Place them inside the `videos/` folder.  
   - Each filename must correspond to the chosen `commercial_id`.  

2. Fill in the metadata file:  
   ```
   initial_data/commercials_initial_metadata.csv
   ```
   Required fields:
   - `commercial_id`
   - `title`
   - `brand`
   - `nice_class`
   - `product_type_key`
   - `year`
   - `lustrum`
   - `source`

---

## Pipeline Steps

The pipeline consists of four main stages:

### 1. Colour and Thumbnail Extraction
- Reads each video and enriches the metadata (`avg_frame_rate`, `aspect_ratio`).
- Splits video into **scenes** and extracts:
  - A **representative thumbnail** per scene (WEBP, 180px height).
  - A **scene-level colour palette** (up to 32 colours).
- Outputs:
  - `general/commercials.csv`  
  - `general/scenes.csv`  
  - `colors/commercial_palettes.csv`  
  - `thumbnails/<commercial_id>/...`

Run:
```bash
%run '1_color_and_thumb_extraction.py'
```

---

### 2. Reference Palette IDF Calculation
- Computes **Inverse Document Frequencies (IDFs)** for each colour in the basic, essential, and extended palettes.
- Outputs:
  - `colors/basic_palette_idfs.csv`
  - `colors/essential_palette_idfs.csv`
  - `colors/extended_palette_idfs.csv`

Run:
```bash
%run '2_ref_palette_idf_calculation.py'
```

---

### 3. Audio Feature Extraction
- Extracts **19 audio features** (spectral, chromatic, rhythmic, etc.) using `librosa` and `tempocnn`.
- Saves results as `.npz` files under `audio/features/<feature_name>_files/`.

Run:
```bash
%run '3_audio_feature_extraction.py'
```

---

### 4. Audio Transcription and Lemmatization
- Detects **speech presence** (`audio/speech_class_confidence_scores.csv`).
- Transcribes speech to text and performs:
  - **Lemmatization**
  - **POS tagging**
  - **TF-IDF weighting**
- Outputs:
  - `audio/speech_class_confidence_score.csv`
  - `text/transcriptions.csv`
  - `text/lemmas.csv`

Run:
```bash
%run '4_audio_transcription_and_lemmatization.py'
```

#### Legal notes

For legal and copyright reasons, the column containing the raw speech transcriptions in text/transcriptions.csv must not be distributed or reused outside the project workflow. Only derived features (e.g., lemmas, frequency counts, TF-IDF values, or psycholinguistic annotations) should be retained and shared for analysis. The raw transcriptions should therefore be discarded after processing.

---

## Notebook

To see and execute the entire workflow step by step, open:

```
script_pipeline.ipynb
```

This notebook runs the pipeline scripts sequentially, showing intermediate results (metadata tables, colour palettes, spectrograms, sample transcriptions).

---


## Reference

If you use this methodology or the dataset, please cite:

> Fadda D., Bellante G., Rinzivillo S., et al.  
> *A dataset of four decades of Italian TV commercials: visual, audio, and linguistic feature descriptors*  
> Zenodo (2025). DOI: [10.5281/zenodo.17127245](https://doi.org/10.5281/zenodo.17127245)

---

## License

The code and demo pipeline are released under the **Apache License, Version 2.0**.  
You may obtain a copy of the License at:

> [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, the software is distributed on an  
*“AS IS” basis*, without warranties or conditions of any kind, either express or implied.  
See the accompanying `LICENSE` file for detailed terms and conditions.

The dataset itself is distributed separately under its own license,  
as indicated in the Zenodo record:

> [https://doi.org/10.5281/zenodo.17127245](https://doi.org/10.5281/zenodo.17127245)

### Acknowledgment

This work was carried out within the **SoBigData.it** project  
*“Strengthening the Italian RI for Social Mining and Big Data Analytics”*  
(Prot. IR0000013 – Avviso n. 3264 del 28/12/2021),  
funded by the **European Union – NextGenerationEU**  
under the **National Recovery and Resilience Plan (PNRR)**.

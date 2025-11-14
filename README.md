# Decibel_duel_PS
# Decibel Duel — Audio Classification

## 1) Drive access & path setup

* The notebook mounts Google Drive and sets `base_dir` to the Decibel Duel directory. It constructs `train_dir` and `test_dir` paths used throughout the notebook.
* It prints existence checks and sample file listings so the code can programmatically confirm that expected folders and files are present.

---

## 2) Audio inspection (single-file demonstration)

* Uses `librosa.load` to read an example `.wav` file (`audio_file_path`). `librosa.display.waveshow` is used to plot the waveform for quick visual inspection.
* `IPython.display.Audio` is used to play the sample inside the notebook.
* For comparison, the code also loads the same file with `scipy.io.wavfile.read` to inspect raw integer waveform data and plots it with `matplotlib`.
---

## 3) MFCC feature extraction (simple tabular features)

* `features_extractor(file_name)` uses `librosa.load(..., res_type='kaiser_fast')` and computes `librosa.feature.mfcc` with `n_mfcc=40`. It then takes the mean across time frames (`np.mean(mfccs_features.T, axis=0)`) to create a fixed-length 40-dimensional feature vector per file.
* The code iterates over every file in each label folder (with `tqdm` for progress), extracts MFCC features, and accumulates them into `extracted_features` as `[features, class_label]` pairs.
* These pairs are converted into a `pandas.DataFrame` and then into `X` and `y` arrays for modelling.
---

## 4) Label encoding & train/test split (baseline pipeline)

* Labels are transformed into one-hot vectors using `LabelEncoder` followed by `to_categorical`.
* The code demonstrates splitting `X`/`y` into training and validation sets using `train_test_split(test_size=0.2, random_state=0)`.
---

## 5) Enhanced mel-spectrogram extraction (CNN-ready 2D inputs)

The `extract_enhanced_melspec` function performs several important tasks:

* **Load & fix duration:** loads audio at `sr=22050` and enforces a fixed `duration` (default 5s). This yields consistent input shapes (important for CNNs).
* **Augmentation (optional):** when `augment=True`, the function randomly applies time-stretch and pitch-shift with probabilistic branches. These augmentations create varied inputs for training and TTA during inference.
* **Padding/trimming:** pads shorter audio with zeros or truncates longer audio to exactly `sr * duration` samples.
* **Mel-spectrogram computation:** computes mel-spectrogram with `n_mels=128`, `n_fft=2048`, `hop_length=512`, and limits `fmax=8000` to focus on most relevant frequencies for environmental sounds.
* **Convert to dB & normalization:** converts power spectrogram to decibels and normalizes each sample by its mean and standard deviation.
* **Return shape:** returns a 2D array `(n_mels, time_frames)` that is later expanded with `[..., np.newaxis]` to become a shape suitable for Conv2D (`(n_mels, time_frames, 1)`).
---

## 6) Building the CNN model (architecture details)

The `build_enhanced_model(input_shape, num_classes)` function constructs a deep CNN with the following key features:

* **Input layer:** `Input(shape=input_shape)` where `input_shape` usually equals `(n_mels, time_frames, 1)`.
* **Repeated Conv blocks:** five main blocks, progressively increasing channel counts (32 -> 64 -> 128 -> 256 -> 512). Each block contains two `Conv2D` layers (except the final block), `BatchNormalization`, `MaxPooling2D`, and `Dropout`.
* **Global pooling:** `GlobalAveragePooling2D` reduces the 2D feature maps to a single vector per filter, reducing overfitting and parameter count compared to flattening.
* **Dense head:** several dense layers (`1024`, `512`, `256`) with `BatchNormalization`, `Dropout`, and `l2` kernel regularisation on the large dense layers. These layers learn higher-level combinations of the convolutional features.
* **Output:** `Dense(num_classes, activation='softmax')` for multi-class classification.

**Key design choices:**

* `BatchNormalization` helps faster convergence and stability.
* Dropout and `l2` regulariser reduce overfitting.
* Deep architecture captures hierarchical audio features from short time-frequency patterns to longer-term statistics.

---

## 7) K‑Fold training & callbacks

* Uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` to create balanced folds by label.
* For each fold the code:

  * Builds a fresh model with `build_enhanced_model`.
  * Compiles the model with `Adam(1e-4)` and `CategoricalCrossentropy(label_smoothing=0.1)`.
  * Uses two important callbacks:

    * `EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)` — helps stop training when accuracy stops improving and keeps the best weights.
    * `ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7)` — lowers learning rate when validation loss plateaus.
  * Trains with `epochs=100`, `batch_size=32` (but training can stop earlier due to early stopping).
  * Evaluates fold validation accuracy and saves the model as `model_fold_{fold}.h5`.
---

## 8) Model saving & handling out-of-GPU scenarios

* The notebook saves each fold's model to Drive. A commented-out code block shows how to later `load_model` for inference if training was interrupted or GPU was unavailable.
---

## 9) Ensemble inference + Test-Time Augmentation (TTA)

* Loads or uses the `models_list` of trained fold models.
* For each test file it:

  * Extracts a base mel-spectrogram (`augment=False`) and run predictions through each model.
  * Creates additional predictions using `augment=True` (two TTA samples per model in this code) to simulate different acoustic variants.
  * Collects all predictions, averages them (`np.mean(all_preds, axis=0)`), and chooses the argmax as the final class.
  * Maps predicted class index back to class name using the `LabelEncoder` (`le.classes_`).
* Aggregates results into `predictions` list of dicts and finally writes `submission.csv`.
---

## 10) Error handling and sanity checks in the code

* The feature extraction loops are wrapped in `try`/`except` to skip problematic files and print errors.
* The code checks for file extensions (`.wav`, `.mp3`) and verifies folder existence early.
* When loading augmentations return `None` for failures so the loop can continue without crashing.
---

## How I feel about this project

I really enjoyed doing this project. I truly tried my best to improve the accuracy. I think there are a few clear ways the accuracy can be improved, and I will try to implement and test those improvements soon. Honestly, I spent most of my time trying to improve the accuracy of PS1, so I didn’t get enough time to study for PS2 :( I tried my best and ended up achieving an accuracy of about 98.37%. However, due to the deadline, it shows as a late submission, but I hope you will still consider it. I’ll also try to study more and learn about AI/ML in general so I can perform better in future tasks. Thank you for :)

---

*End of code explanation README.*


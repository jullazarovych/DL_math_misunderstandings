# Project Documentation: Math Misconceptions Classification

## Overview
This project implements a deep learning system for identifying and classifying mathematical misconceptions in student explanations. The system uses RoBERTa-based models with LoRA fine-tuning to categorize student responses into 33 different misconception types.

## Project Structure

The project consists of three main notebooks:
1. **math_misunderstanding.ipynb** - Data preprocessing with semantic deduplication (v1)
2. **math_misunderstandings_v2.ipynb** - Data preprocessing without semantic deduplication (v2)
3. **model_training.ipynb** - Model training and evaluation

---

## 1. Data Preprocessing

### Libraries and Dependencies

```python
# Core data processing
import pandas as pd
import numpy as np
import re
import json

# Text processing
from textblob import TextBlob
import enchant
import concurrent.futures
from tqdm import tqdm

# Semantic similarity (v1 only)
from sentence_transformers import SentenceTransformer
import faiss

# System integration
from google.colab import drive
```

### Installation Commands

```bash
pip install --upgrade --force-reinstall textblob nltk
apt-get update
apt-get install -y libenchant-2-2
pip install pyenchant --force-reinstall
pip install sentence-transformers faiss-cpu  # v1 only
```

### Data Loading

**Function:** Initial data loading and length filtering

**Parameters:**
- `BASE_DATA_PATH`: Path to data directory in Google Drive
- `max_length`: Maximum character length for student explanations (250)

**Process:**
1. Mounts Google Drive
2. Loads raw training data
3. Filters out explanations longer than 250 characters
4. Reports row counts at each step

---

### Text Normalization

#### Contractions Dictionary

A comprehensive dictionary containing 200+ mathematical and linguistic patterns for normalization:

**Categories:**
- Fraction notations: `"1 3rd" → "1/3"`, `"3 9ths" → "3/9"`
- Contractions: `"isn't" → "is not"`, `"don't" → "do not"`
- Mathematical terms: `"denominater" → "denominator"`, `"numerater" → "numerator"`
- Mathematical expressions: `"3x5" → "3 x 5"`, `"2y=24" → "2 y = 24"`
- Common misspellings: `"wich" → "which"`, `"hole" → "whole"`

#### `normalize_text_watch(text)`

**Purpose:** Normalizes student explanations by expanding contractions and standardizing mathematical notation.

**Input:** Raw text string

**Output:** Tuple (original_text, normalized_text)

**Process:**
1. Converts text to lowercase
2. Applies contraction expansions using regex patterns
3. Normalizes whitespace
4. Removes dots between letters
5. Standardizes fraction notation (e.g., `3 / 9` → `3/9`)
6. Removes non-alphanumeric characters (except math symbols)
7. Removes periods and commas

**Example:**
```python
input: "I think 3/9is the same as1/3"
output: ("I think 3/9is the same as1/3", "i think 3/9 is the same as 1/3")
```

---

### Semantic Deduplication (v1 only)

#### `deduplicate_similar_texts(texts, threshold=0.90, batch_size=256)`

**Purpose:** Removes semantically similar duplicate entries using cosine similarity.

**Parameters:**
- `texts`: List of text strings to deduplicate
- `threshold`: Similarity threshold (default: 0.90, i.e., 90% similar)
- `batch_size`: Batch size for encoding (default: 256)

**Process:**
1. Encodes all texts using SentenceTransformer (`all-MiniLM-L6-v2`)
2. Normalizes embeddings to unit length (L2 normalization)
3. Builds FAISS index for efficient similarity search
4. For each text, finds its nearest neighbor
5. If similarity > threshold, marks duplicate for removal
6. Returns indices of texts to keep

**Returns:** List of indices to retain

**Technical Details:**
- Uses FAISS IndexFlatIP (Inner Product) for cosine similarity
- Performs k=2 nearest neighbor search (text and its closest match)
- Memory-efficient: processes in batches

---

### Spelling Correction

#### Configuration

```python
# English dictionary for validation
dict_en = enchant.Dict("en_US")

# Whitelist of mathematical terms
WHITELIST = {
    "decimal", "denominator", "numerator", "equivalent",
    "fraction", "fractions", "simplify", "simplified",
    "multiplying", "dividing", "multiply", "divide",
    "factor", "factors", "percent", "percentage",
    "expression", "equation", "variable", "coefficient", "lcm"
}

# Pattern-based corrections
SPECIAL_FIX = {
    r"d[eo]nom[a-z]*": "denominator",
    r"numer[a-z]*": "numerator",
    r"equiv[a-z]*": "equivalent",
}
```

#### `apply_special_fix(word)`

**Purpose:** Applies pattern-based corrections for mathematical terms.

**Input:** Single word (lowercase)

**Output:** Corrected word or None

**Logic:** Checks if word matches any regex pattern in SPECIAL_FIX dictionary.

---

#### `smart_correct_spelling(text)`

**Purpose:** Intelligent spelling correction that preserves mathematical notation.

**Input:** Text string

**Output:** Corrected text string

**Process:**
1. Tokenizes text into words and symbols
2. For each token:
   - **Skip** if contains numbers or special characters
   - **Apply** special fix if matches pattern
   - **Keep** if in whitelist
   - **Keep** if recognized by enchant dictionary
   - **Correct** using TextBlob if misspelled
   - **Validate** correction (reject if length differs by >3 characters)
3. Reconstructs text with proper spacing

**Example:**
```python
input: "the numerater and denom should be divded"
output: "the numerator and denominator should be divided"
```

---

#### `process_in_parallel(texts, func, num_workers=4)`

**Purpose:** Parallel processing for spelling correction.

**Parameters:**
- `texts`: List of texts to process
- `func`: Function to apply (smart_correct_spelling)
- `num_workers`: Number of parallel processes (default: 4)

**Returns:** List of corrected texts

**Technical Details:**
- Uses ProcessPoolExecutor for CPU parallelization
- Shows progress bar with tqdm
- Significantly speeds up processing for large datasets

---

### LaTeX Cleaning

#### `clean_latex(text)`

**Purpose:** Removes LaTeX formatting artifacts from question text and answers.

**Input:** Text with potential LaTeX markup

**Output:** Cleaned text

**Process:**
1. Removes LaTeX delimiters: `\( \) \[ \]`
2. Removes backslashes
3. Fixes spacing around punctuation
4. Removes spaces between numbers and letters
5. Normalizes all whitespace

**Example:**
```python
input: "What is \\(3\\times5\\) ?"
output: "What is 3times5?"
```

---

### Label Normalization

#### `normalize_label(label)`

**Purpose:** Standardizes misconception labels.

**Input:** Raw label string

**Output:** Normalized label

**Process:**
1. Handles NA/null values
2. Strips whitespace
3. Converts to lowercase
4. Replaces spaces and hyphens with underscores
5. Removes duplicate underscores

**Example:**
```python
input: "Inverse Operation"
output: "inverse_operation"
```

**Label Mapping:**
```python
mapping = {
    "inversion": "inverse_operation"
}
```

---

### Data Deduplication Summary

**Version 1 (v1):** With semantic deduplication
- Initial filtering: Length > 250 characters
- Text normalization and basic deduplication
- **Semantic deduplication** (90% similarity threshold)
- Spelling correction and final deduplication

**Version 2 (v2):** Without semantic deduplication
- Initial filtering: Length > 250 characters
- Text normalization and basic deduplication
- Spelling correction and final deduplication
- **More data retained** compared to v1

**Final Output:**
- Combined column: `QuestionText || MC_Answer || StudentExplanation`
- Saved as CSV with UTF-8 encoding

---

## 2. Model Training

### Libraries and Dependencies

```python
# Core ML frameworks
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset

# Transformers and PEFT
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
    prepare_model_for_kbit_training
)

# Data processing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from datasets import Dataset

# Evaluation
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Utilities
import json
import gc
from tqdm import tqdm
```

### Installation Commands

```bash
pip install --upgrade transformers peft accelerate bitsandbytes
pip install -q transformers[torch] peft accelerate bitsandbytes scikit-learn pandas sentencepiece
```

---

### Configuration

#### Model Settings

```python
model_checkpoint = "roberta-base"
model_name = "roberta-base"

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
```

**Purpose:** 4-bit quantization for memory-efficient training
- Reduces model size by ~4x
- Maintains performance with NF4 quantization
- Enables training on consumer GPUs

---

#### LoRA Configuration

```python
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,                    # Rank of update matrices
    lora_alpha=32,           # Scaling factor
    lora_dropout=0.1,        # Dropout rate
    bias="none",             # No bias updates
    target_modules=["query", "value"]  # Attention layers to adapt
)
```

**Purpose:** Low-Rank Adaptation for parameter-efficient fine-tuning
- Only trains ~0.1% of parameters
- Target modules: Query and Value projections in attention
- r=16: Balance between capacity and efficiency

---

#### Training Arguments

```python
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir='./logs',
    fp16=True,                      # Mixed precision training
    report_to="none",
    label_smoothing_factor=0.1      # Label smoothing for regularization
)
```

---

### Data Preparation

#### Label Encoding

```python
le = LabelEncoder()
df['label'] = le.fit_transform(df['Misconception'])
num_labels = len(le.classes_)

id2label = {i: label for i, label in enumerate(le.classes_)}
label2id = {label: i for i, label in enumerate(le.classes_)}
```

**Purpose:** Converts string labels to numeric indices
**Output:** 33 unique classes (0-32)

---

#### `preprocess_function(examples)`

**Purpose:** Tokenizes input text for model consumption.

**Parameters:**
- `examples`: Batch of text examples

**Process:**
1. Tokenizes using RoBERTa tokenizer
2. Truncates to max_length=256 tokens
3. Applies padding

**Returns:** Dictionary with `input_ids` and `attention_mask`

---

### Custom Dataset Class

#### `TextDataset(TorchDataset)`

**Purpose:** PyTorch dataset wrapper for tokenized data.

**Methods:**

**`__init__(self, encodings, labels)`**
- Stores tokenized inputs and labels

**`__getitem__(self, idx)`**
- Returns single example as dictionary with tensors
- Includes: input_ids, attention_mask, labels

**`__len__(self)`**
- Returns dataset size

---

### Custom Trainer

#### `WeightedLossTrainer(Trainer)`

**Purpose:** Handles class imbalance with weighted loss.

**Method:** `compute_loss(model, inputs, return_outputs, num_items_in_batch)`

**Process:**
1. Extracts labels from inputs
2. Gets model predictions (logits)
3. Computes CrossEntropyLoss with class weights
4. Returns loss (and outputs if requested)

**Class Weights Calculation:**
```python
classes_in_fold = np.unique(train_labels_fold)
fold_weights = compute_class_weight('balanced', 
                                   classes=classes_in_fold, 
                                   y=train_labels_fold)
weights_tensor = torch.tensor(weights_full, dtype=torch.float).to(device)
```

---

### Cross-Validation Training

#### Training Loop Structure

```python
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
```

**For each fold:**

1. **Data Split**
   - Stratified split to maintain class distribution
   - Train/validation split

2. **Class Weight Computation**
   - Calculates balanced weights for current fold
   - Handles missing classes gracefully

3. **Model Initialization**
   ```python
   model = AutoModelForSequenceClassification.from_pretrained(
       model_checkpoint,
       num_labels=num_labels,
       id2label=id2label,
       label2id=label2id
   )
   model.resize_token_embeddings(len(tokenizer))
   model = get_peft_model(model, peft_config)
   ```

4. **Training**
   ```python
   trainer = WeightedLossTrainer(
       model=model,
       args=training_args,
       train_dataset=train_dataset,
       eval_dataset=val_dataset,
       tokenizer=tokenizer,
       data_collator=data_collator,
       compute_metrics=compute_metrics
   )
   trainer.train()
   ```

5. **Evaluation & Saving**
   - Evaluates on validation set
   - Saves model to Drive
   - Clears GPU memory

---

### Evaluation Metrics

#### `compute_metrics(eval_pred)`

**Purpose:** Computes comprehensive evaluation metrics.

**Input:** 
- `eval_pred`: Tuple of (logits, labels)

**Metrics Computed:**
- **Accuracy**: Overall correctness
- **F1-Macro**: Average F1 across all classes (unweighted)
- **F1-Weighted**: Weighted average F1 (by support)
- **Precision-Macro**: Average precision
- **Recall-Macro**: Average recall
- **Balanced Accuracy**: Average recall per class
- **MCC**: Matthews Correlation Coefficient

**Returns:** Dictionary of metrics

---

### Holdout Test Evaluation

**Purpose:** Final evaluation on unseen 300-sample holdout set.

**Process:**
1. Loads saved model from best fold
2. Runs inference on holdout set
3. Generates classification report
4. Shows error examples

**Output:**
- Accuracy on holdout set
- Detailed classification report
- Confusion matrix
- Top 5 misclassified examples

---

### Visualization Functions

#### `visualize_classification_results(y_true, y_pred, labels, target_names)`

**Purpose:** Creates comprehensive result visualizations.

**Visualizations Created:**

1. **Heatmap**: Precision, Recall, F1-Score by class
   - Color-coded performance metrics
   - Sorted by F1-score (ascending)
   - RdYlGn colormap (red=low, green=high)

2. **Bar Chart**: Class distribution with quality overlay
   - X-axis: Sample count (support)
   - Y-axis: Class names
   - Color: F1-score (viridis colormap)
   - Shows which classes have most/least data

**Usage:**
```python
visualize_classification_results(y_true, y_pred, all_label_ids, target_names)
```

---

#### Training History Visualization

**Purpose:** Plots training dynamics across folds.

**Plots Created:**

1. **Loss Curves**
   - Training loss (blue)
   - Validation loss (red)
   - Shows convergence behavior

2. **Validation Metrics**
   - F1-Macro (orange, solid line)
   - Accuracy (green, dashed line)
   - Shows model improvement over epochs

**Features:**
- Aggregates data across all folds
- Shows epoch-wise progression
- Grid for easy reading

---

### Inference Pipeline

#### Final Model Loading and Prediction

**Purpose:** Loads trained model and generates predictions on test set.

**Process:**

1. **Model Loading**
   ```python
   base_model = AutoModelForSequenceClassification.from_pretrained(
       model_checkpoint,
       num_labels=num_labels,
       id2label=id2label,
       label2id=label2id
   )
   model = PeftModel.from_pretrained(base_model, saved_model_path)
   ```

2. **Data Preparation**
   - Loads test CSV
   - Fills NA values
   - Creates Combined column
   - Tokenizes input

3. **Prediction**
   ```python
   predictions_output = trainer.predict(tokenized_test_clean)
   y_pred_ids = np.argmax(predictions_output.predictions, axis=1)
   y_pred_labels = [labels_map[i] for i in y_pred_ids]
   ```

4. **Output**
   - Adds predictions to DataFrame
   - Saves as submission.csv

---

### Additional Utilities

#### `clean_latex(text)` (in model_training.ipynb)

**Purpose:** Same as in preprocessing, ensures consistency.

#### Semantic Similarity Visualization (Optional)

```python
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

# Create TF-IDF embeddings
tfidf = TfidfVectorizer(max_features=500).fit_transform(sample_df['Combined'])

# Reduce to 2D with t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embedding = tsne.fit_transform(tfidf.toarray())

# Plot colored by misconception
sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], 
                hue=sample_df['Misconception'], ...)
```

**Purpose:** Visualizes semantic clustering of misconceptions.

---

## Key Findings

### Data Statistics

**Version 2 (Recommended):**
- **Initial rows**: ~15,000
- **After length filter**: ~14,000
- **After cleaning & deduplication**: ~13,000
- **After spelling correction**: ~12,500
- **33 unique misconception classes**

### Model Performance

**Best Fold Results:**
- **Accuracy**: 93.25%
- **F1-Macro**: 86.58%
- **Balanced Accuracy**: ~90%

**Training Dynamics:**
- Validation loss drops from 0.39 → 0.27 (epoch 1→3)
- Accuracy increases from 90.6% → 93.3%
- F1-Macro: 75.9% → 86.6%

### Challenges Addressed

1. **Class Imbalance**: Weighted loss function
2. **Long-tail distribution**: Label smoothing
3. **Memory constraints**: 4-bit quantization + LoRA
4. **Overfitting**: 5-fold cross-validation
5. **Spelling errors**: Custom correction pipeline

---

## Usage Instructions

### 1. Data Preprocessing

```python
# Run preprocessing (v2 recommended)
# Execute cells in math_misunderstandings_v2.ipynb
# Output: train_v2.csv
```

### 2. Model Training

```python
# Execute cells in model_training.ipynb
# Models saved to Google Drive: /nlp_math_misunderstanding/weights/fold_X
```

### 3. Inference

```python
# Load trained model
# Run inference on test set
```

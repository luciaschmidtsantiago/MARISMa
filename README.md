# MARISMa

MARISMa is a pipeline for cleaning, organizing, labeling, and analyzing MALDI-TOF mass spectrometry data for microbial identification and antimicrobial resistance (AMR) research. The project includes scripts for data cleaning, quality checking, statistics generation, AMR label matching, and a demonstration use case to facilitate immediate use and reproducibility.

---

## Folder Structure

```
MARISMa/
├── 1_cleaner.py           # Cleans and organizes raw data
├── 2_checker.py           # Checks for duplicates and empty spectra
├── 3_stats.py             # Generates detailed statistics and CSVs
├── 4_AMR_labeler.py       # Matches and labels AMR data
├── Plug&Play/             # Plug & Play spectrum processing utilities (see below)
├── README.md              # This file
```

---

## Plug&Play

The `Plug&Play` folder contains reusable spectrum processing utilities, including:

- **spectrum.py**:  
  Classes and functions for spectrum representation, binning, normalization, trimming, smoothing, baseline correction, peak detection, and more.  
  These tools can be used independently for custom spectrum preprocessing pipelines.
- **rf_maldi_model.joblib**:  
  Pretrained Random Forest classifier. To train a new one, use simple_classifier.ipynb
**Usage Example:**
```python
import joblib
model = joblib.load('models/pretrained_model.pkl')
# Use model.predict(...) as needed
```

### Jupyter Notebook

A Jupyter notebook is provided for interactive data exploration, preprocessing, and classification of MALDI-TOF spectra using the MARISMa pipeline.

You can find it in the repository as:

- `Plug&Play/simple_classifier.ipynb`

**Notebook Features:**
- Loads Bruker MALDI-TOF spectra for a specified genus (and optional species) and years.
- Applies preprocessing: variance stabilization, smoothing, baseline correction, normalization, trimming, and binning.
- Extracts features and builds labels.
- Trains a Random Forest classifier for species-level identification.
- Evaluates performance and saves the trained model.

To launch the notebook:
```sh
jupyter notebook Plug&Play/simple_classifier.ipynb
```

---

## Main Scripts

### 1. Data Cleaning

**`1_cleaner.py`**  
Cleans and organizes raw MALDI-TOF data by year.  
**Usage:**  
```sh
python3 1_cleaner.py [root] [base_dest_dir] year1 [year2 ...]
```
- If `root` and `base_dest_dir` are omitted, defaults are used.

### 2. Data Checking

**`2_checker.py`**  
Checks for empty and duplicate spectra, and generates hash and stats files.  
**Usage:**  
```sh
python3 2_checker.py [--base_dest_dir BASE] year1 [year2 ...]
```

### 3. Statistics

**`3_stats.py`**  
Generates detailed statistics JSON and a CSV with identifiers, target positions, genus, species, year, and path.  
**Usage:**  
```sh
python3 3_stats.py --base_dir BASE year1 [year2 ...]
```

### 4. AMR Labeling

**`4_AMR_labeler.py`**  
Matches AMR labels to valid identifiers, checks genus/species, and adds file paths.

---

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- tqdm

Install dependencies with:
```sh
pip install pandas numpy matplotlib tqdm
```

---

## Notes

- Paths and file names may need to be adapted to your environment.
- See comments in each script for further customization.

---

## License

(C) Inés López-Mareca and Lucía Schmidt-Santiago. All rights reserved.

---

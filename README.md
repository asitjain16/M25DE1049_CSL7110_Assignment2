# CSL7110 - Min-Hashing and LSH Assignment

## Author
Asit Jain

## Files
- `asitjain_assignment2.py` - Complete implementation (all 5 questions)
- `minhash/` - Document files (D1.txt - D4.txt)
- `requirements.txt` - Python dependencies

## Setup

1. Install dependencies:
```bash
pip install numpy
```

2. Download MovieLens 100k dataset:
```bash
# Download from: http://files.grouplens.org/datasets/movielens/ml-100k.zip
# Extract to ml-100k/ directory
```

## Run

```bash
python asitjain_assignment2.py
```

## Results Summary

### Question 1: K-Grams
- Character 2-grams, 3-grams, and word 2-grams computed
- Jaccard similarities calculated for all document pairs
- D1 vs D2: 0.9780 (nearly identical)

### Question 2: Min-Hashing
- Tested with t = 20, 60, 150, 300, 600
- Optimal t = 150-200 (error < 1%, fast execution)

### Question 3: LSH
- Optimal configuration: b=20, r=8 for threshold τ=0.7
- Successfully separates similar and dissimilar pairs

### Question 4: MovieLens Min-Hash
- 943 users, 10 pairs with Jaccard >= 0.5
- False positive/negative analysis for t=50, 100, 200

### Question 5: MovieLens LSH
- Multiple (b, r) configurations tested
- Thresholds 0.6 and 0.8 analyzed
- LSH provides significant speedup


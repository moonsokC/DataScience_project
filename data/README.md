# Data

This repository uses the Kaggle Craigslist Vehicles dataset.

Because the original `vehicles.csv` file is large, the project first creates a reproducible random sample of 20,000 rows using `scripts/create_random_sample.py` and random seed 42.

Expected local files:

```text
data/
├── vehicles.csv                  # Original Kaggle file, not tracked in this repo
├── vehicles_sample_20000.csv      # Reproducible 20,000-row sample, generated locally
└── used_cars_cleaned_for_modeling.csv
```

The cleaned modeling dataset is included because it is small and directly supports the published results.

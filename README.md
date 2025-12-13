# 360° Adaptive Streaming with Language-Guided Semantic Priors

This repository contains the source code for a 6-week undergraduate research project to build and evaluate a novel 360° adaptive streaming client. The system leverages edge-offloaded semantic priors, lightweight on-device personalization, and a contextual bandit controller to optimize the trade-off between Quality of Experience (QoE) and on-device energy consumption.

---

### Key System Components
-   **Edge Semantic Priors:** Offline generation of K-class semantic tile maps from video content.
-   **On-Device Personalization:** Learns a per-user preference vector `w` over semantic classes.
-   **Cross-Modal FoV Predictor:** A tiny Transformer model that fuses head motion and semantic priors.
-   **Contextual Bandit Controller:** Dynamically adapts update rates, offloading, and saccade-gating.
-   **Two-Stage ABR:** A prefetch/enhance scheduler based on a Multiple-Choice Knapsack Problem (MCKP).

---

### Directory Structure
```
.
├── data/              # Placeholder for datasets (ignored by git)
├── docs/              # Project reports, diagrams, and final presentation
├── notebooks/         # Jupyter notebooks for exploration and analysis
├── scripts/           # Standalone scripts (data processing, experiment runners)
├── src/               # Main source code
│   ├── common/        # Shared utilities (logger, tiling geometry)
│   ├── personalize/   # Personalization head training and fusion
│   ├── modeling/     # FoV prediction model
│   └── semantics/     # Semantic prior generation
├── tests/             # Unit tests for key components
├── .gitignore         # Files and directories ignored by git
├── README.md          # This file
└── requirements.txt   # Python project dependencies
```

---

### Getting Started

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd Semantic-Viewport-Prediction
    ```

2.  **Create a Python virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### AVTrack360 Ingestion & Alignment (Week 1)

We ingest AVTrack360 head–motion logs and standardize them under data/standardized/.

1. Parse and standardize

    ```
   python -m ingest.scripts.avtrack360_loader .\data\raw\avtrack360\<user_number>.json [--debugging]
    ```
   
This produces (example):

    data/standardized/user10_clip6.parquet – normalized angles + raw timestamps (sec)
    data/standardized/user10_clip6_60hz.parquet – resampled to a uniform 60 Hz timeline (time_s)
    data/standardized/user10_clip6_60hz_vel.parquet – angular velocities on the 60 Hz grid

2. Alignment report (data health check)

    ```
    python -m ingest.scripts.make_alignment_report \
        data/raw/avtrack360/10.json \
        --clip 6 \
        --user 10
    ```

This generates:

    reports/alignment/user10_clip6/user10_clip6_alignment.json
    reports/alignment/user10_clip6/user10_clip6_alignment.png
    reports/alignment/user10_clip6/user10_clip6_summary.md

The report includes:

    raw sampling interval histogram (Δt in ms)
    raw timestamp vs sample index (“drift”)
    angle sanity ranges (raw deg + normalized rad)
    60 Hz cadence check
    a heuristic missing–data estimate

3. Dev subset

For quick experiments, we use a pre-processed slice:

    data/dev/avtrack360_user10_clip6/
        raw/10.json
        standardized/{user10_clip6.parquet, user10_clip6_60hz.parquet, user10_clip6_60hz_vel.parquet}
        reports/{user10_clip6_alignment.json, user10_clip6_alignment.png, user10_clip6_summary.md}

### Building Semantic Priors Tile-grid from Video (Week 2)

This command runns the `build_priors.py` script under `./scripts` to build and output the PyTorch tensor file (.pt) that contains the Semantic Priors map of tiles.

The shape of this tensor is (sequence length, number of semantic classes, rows, columns) which should look something like `[T, 11, 4, 6]`

```
python ./src/semantics/build_priors.py .\data\videos\<video number>.mp4 [--debugging]
```

### Running Head Motion Dataset (Week 3)

This module is used to convert the user motion sequence data in the parquet files generated from AVTrack360 Ingestion & Alignment into a tensor that can be concatenated with the semantic prior tensor of a video.

```
python ./scripts/run_dataset.py <video_number>
```

### Traiing and Running the Fusion model for One Video (Week 4)

Running the fusion model will take an 80/20 training split on the user motion tensors output by the Head Motion Dataset for a given video before outputting predictions for the 2 testing users.

Results are output to `./results/clip_<video_number>`

```
python -m scripts.train_fusion <video_number> --epochs <number_of_epochs>
```

---

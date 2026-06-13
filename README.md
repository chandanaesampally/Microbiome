# Bio-Perceiver OS

Microbiome-to-Macronutrient Mapping Surrogate — a neural food recommendation system that maps gut microbiome data (4 phylum-level ratios) + clinical metadata (age, BMI, TDEE, IBS severity, diet preference) to ranked food affinities across 850 Indian dishes.

---

## Required Folder Structure

After cloning, your repo must look **exactly** like this:

```
bio-perceiver/
│
├── app.py                        ← the only runtime script
├── requirements.txt
├── Procfile
├── README.md
│
├── resource/
│   ├── food_catalog_v5.csv       ← 850-food database (upload this)
│   └── abns_surrogate_weights.pth ← trained model weights (upload this)
│
└── images/
    ├── 1.jpg
    ├── 2.jpg
    ├── ...
    └── 850.jpg                   ← food images (upload all)
```

> **Do NOT include** patient_manifold_v2.csv or food_target_matrix.csv — those are training-only files.

---

## Local Run

```bash
pip install -r requirements.txt
python app.py
# Open http://127.0.0.1:8050
```

---

## Deploy on Render.com (Free — Recommended)

1. Push this repo to GitHub (include `resource/` and `images/` folders)
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Set these fields:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:server --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
   - **Instance Type:** Free
5. Click Deploy — Render reads the `Procfile` automatically

Your app will be live at `https://your-app-name.onrender.com`

---

## Deploy on Hugging Face Spaces (Also Free)

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) → Create Space
2. Choose **Gradio** SDK (then switch to Docker) OR choose **Docker**
3. Upload all files keeping the same folder structure
4. Add a `README.md` with `sdk: docker` in the frontmatter

---

## Model Architecture

**BioPerceiverFusionNet** — Dual-stream Tabular ResNet

```
Input: BIO[1,4] + CLIN[1,5]
  │
  ├── Bio Stream:   4 → 128 (ResBlock) → 64
  ├── Clin Stream:  5 → 64  (ResBlock) → 32
  │
  └── Fusion:       96 → 256 (ResBlock) → 850 foods  [Sigmoid]
```

- **Total Parameters:** ~430,000
- **Training Data:** 3,949 synthetic patient profiles × 850 foods
- **Loss:** AsymmetricMSE (over-penalty = 2×, under-penalty = 1×)
- **Modality Dropout:** 25% clinical dropout + 10% biological dropout during training

---

## Files NOT Needed for Deployment

These were used only to generate data and train the model:

| File | Purpose |
|------|---------|
| `1_catalog.py` | Generates food_catalog_v5.csv |
| `2_patient_manifold.py` | Generates patient_manifold_v2.csv |
| `3_target_matrix.py` | Generates food_target_matrix.csv |
| `4_architecture.py` | Architecture preview/test |
| `5_dataset_check.py` | Data audit |
| `6_evaluation.py` | NDCG / ablation evaluation |
| `7_trainer.py` | Model training |
| `image.py` / `image2.py` | Image downloading scripts |

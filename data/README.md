# Dataset

Download a waste classification dataset. A good option:

- [Waste Classification Data (Kaggle)](https://www.kaggle.com/datasets/techsash/waste-classification-data)
  ~2,500 images across 2 categories (O = Organic, R = Recyclable)

Or any dataset with folders named `recyclable/`, `organic/`, `landfill/`.

Structure expected:
```
data/
  train/
    recyclable/  *.jpg
    organic/     *.jpg
    landfill/    *.jpg
  val/
    recyclable/  *.jpg
    organic/     *.jpg
    landfill/    *.jpg
```

Then run:
```bash
pip install -r api/requirements.txt
python model/train.py
python api/app.py
```

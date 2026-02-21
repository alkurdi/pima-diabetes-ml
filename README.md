# PIMA Diabetes ML Project (Structured)

المشروع الآن يدعم **Multiple Runs** مع تتبع كامل لكل تجربة.

## المراحل الأساسية

- `src/preprocessing.py`
- `src/models.py`
- `src/train.py`
- `src/evaluate.py`

## تتبع التجارب (Multiple Runs)

كل تشغيل تدريب ينشئ مجلدًا مستقلًا:

- `results/runs/<run_id>/`

ويحتوي على:

- `train_log.txt`
- `eval_log.txt` (بعد التقييم)
- `metrics.json`
- `metrics.csv`
- `run_config.json`
- `model.pkl`

> إذا لم تمرر `--run_id`، يتم توليد `run_id` تلقائيًا (timestamp).

## أوامر التشغيل

```bash
python src/train.py
python src/evaluate.py
```

تشغيل مخصص:

```bash
python src/train.py --run_id exp_001 --seed 42 --data_path data/diabetes.csv --model logreg
python src/evaluate.py --run_id exp_001 --seed 42 --data_path data/diabetes.csv
```

## Summary تجميعي

بعد كل تقييم يتم تحديث:

- `results/summary.csv`

ويحتوي صفًا لكل run بالأعمدة:

- `run_id, model_name, seed, accuracy, f1, roc_auc, timestamp`


## Evaluate with threshold

```bash
python src/evaluate.py --run_id test1 --threshold 0.5
```

ينتج داخل كل run: `predictions.csv`, `roc_curve.csv`, `roc_curve.png`, `confusion_matrix.csv`, `confusion_matrix.png`, و `eval_config.json`.

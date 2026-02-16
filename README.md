# PIMA Diabetes ML Project (Structured)

تنظيم مشروع التنبؤ بالسكري إلى ملفات واضحة مع منع **data leakage** باستخدام `sklearn.Pipeline`.

## Structure

- `src/preprocessing.py`
  - تعريف أعمدة الميزات والهدف.
  - فصل `X` و `y`.
  - تنفيذ `make_train_test_split` قبل أي preprocessing.
  - بناء `ColumnTransformer` يحتوي `SimpleImputer` + `StandardScaler` داخل Pipeline.
- `src/models.py`
  - Registry للموديلات (`logreg`, `rf`).
  - بناء الـPipeline الكامل (`preprocess` ثم `model`).
  - مساحات Hyperparameter للـGridSearch.
- `src/train.py`
  - قراءة البيانات.
  - split أولًا.
  - مقارنة عدة نماذج باستخدام `StratifiedKFold` وقياسات `ROC-AUC` و`F1`.
  - حفظ نتائج المقارنة في جدول واحد (`artifacts/model_comparison.csv`).
  - تدريب نموذج نهائي (مع/بدون tuning) وحفظه كملف `pickle`.
- `src/evaluate.py`
  - تحميل الموديل المحفوظ.
  - تقييمه على holdout split بنفس الإعدادات.

## Leakage Safety Checklist

- ✅ **Scaling يتم بعد split**: لأن `StandardScaler` داخل الـPipeline، والـPipeline يتم `fit` على `X_train` فقط.
- ✅ **Imputation داخل Pipeline**: `SimpleImputer` موجود داخل preprocessing pipeline.
- ✅ **عدم استخدام test data أثناء التدريب**: `fit`/`GridSearchCV`/`StratifiedKFold CV` يتم على train split فقط.

## Run

```bash
# مقارنة كل النماذج وحفظ جدول النتائج
python src/train.py --compare-models --cv 5

# تدريب نموذج محدد (مع tuning اختياري)
python src/train.py --model logreg --tune

# تقييم النموذج المحفوظ
python src/evaluate.py --model-path artifacts/model.pkl
```

> غيّر `--random-state` و`--test-size` (بنفس القيم) لإعادة إنتاج نفس التقسيم.


## Outputs

- Trained model: `artifacts/model.pkl`
- Metrics: `results/metrics.json` and `results/metrics.csv`
- Run config: `results/run_config.json`

- Reproducibility seed is fixed (default 42) for `numpy`, `random`, and sklearn components through `random_state`.

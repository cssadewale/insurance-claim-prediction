# Insurance Claim Prediction Fix Notes

## Fixed

- Added Python 3.12 deployment pinning and compatible binary-wheel dependencies.
- Made the model cache path relative to `app.py`, not the process working directory.
- Added graceful handling for failed Google Drive downloads.
- Added protection against caching an HTML permission/quota page as a `.joblib` file.
- Added model deserialization error reporting and a 13-feature schema check.

## Required external artifact

The repository intentionally does not contain `best_random_forest_model.joblib`; the application downloads it from the configured Google Drive URL. The Drive file must be shared as “Anyone with the link” and must remain the exact 13-feature model expected by the app.

## Remaining model-quality limitation

The app still contains manually recorded scaling statistics and one-hot feature engineering. The production-grade fix is to retrain and export one complete sklearn preprocessing/model pipeline, then replace the hard-coded constants with that pipeline.

# Model Weights

This directory documents the trained model used by CardioSentinel.

## Expected Model

The current application has been associated with the checkpoint `ecg_cnn_v2.pt`. Confirm its actual location and compatibility with the current `core.py` before running the application.

## Setup

1. Obtain the model checkpoint from an authorized team source.
2. Place it at the path expected by the model-loading code.
3. Verify that the checkpoint loads successfully.
4. Run a prediction on a known test window and check the output format.

## Sharing and Redistribution

Do not commit model weights until the team confirms that redistribution is permitted and that the file size is suitable for the chosen hosting service.

If weights are not included, explain how authorized users can obtain them. Never commit secrets or credentials alongside a model.

## Limitations

The checkpoint is experimental and must not be treated as a clinically validated diagnostic model.

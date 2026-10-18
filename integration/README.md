# Coordinated-package integration test

Run with the updated PixelCue wheel, Pillow, NumPy and OpenCV installed:

    python -m pytest -q integration/test_hybrid_pipeline.py

This is separate from the dependency-free PrismMoves core suite, which runs under
`tests/` without installing PixelCue or neural frameworks. No missing-dependency
skip is used. The update bundle validates this integration against its actual
built PixelCue/PrismMoves wheels before distribution. Synthetic model adapters
exercise semantic provenance/order; they are not pretrained accuracy tests.

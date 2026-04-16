"""Import smoke tests — catch missing dependencies and circular imports.

Mirrors `Section 1` of the original `scripts/test_all.sh` harness.
Runs on CPU; no GPU required.
"""

from __future__ import annotations


def test_import_stieltjes_attention():
    import stieltjes_attention as s

    assert "stieltjes_attention" in s.__all__
    assert "stieltjes_attention_ref" in s.__all__
    assert "StieltjesAttention" in s.__all__


def test_import_nanogpt_data():
    from nanogpt.data import (
        CLRSDataset,
        TaskConfig,
        VOCAB_SIZE,
        PAD,
        SEPARATOR,
    )

    assert SEPARATOR == 256
    assert PAD == 257
    assert VOCAB_SIZE == 258


def test_import_nanogpt_model():
    from nanogpt.model import GPT, GPTConfig  # noqa: F401


def test_import_nanogpt_train():
    import nanogpt.train  # noqa: F401


def test_import_nanogpt_train_curriculum():
    import nanogpt.train_curriculum  # noqa: F401


def test_import_nanogpt_eval_accuracy():
    import nanogpt.eval_accuracy  # noqa: F401


def test_import_nanogpt_analyze():
    import nanogpt.analyze  # noqa: F401

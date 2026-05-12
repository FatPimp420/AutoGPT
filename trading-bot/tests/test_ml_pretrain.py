"""Smoke tests for the batch pre-training script and saved model files.

Verifies that:
1. ``pretrain_all`` creates .pkl files in the target directory.
2. A saved model can be loaded and ``generate_signal`` runs without error.
3. The loaded model's ``predict_proba`` output is well-formed.

Run from the trading-bot root:
    python -m pytest tests/test_ml_pretrain.py -v
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob

import pytest

from backtest.synthetic import generate_ohlcv
from ml.model import MLSignalModel
from ml.pretrain_all import pretrain_all, STRATEGIES, SYMBOLS, _model_filename
from strategies.rsi import RSIStrategy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    """Run pretrain_all into a temp directory and return the path."""
    out = str(tmp_path_factory.mktemp("ml_models"))
    # Use fewer days to keep the test fast
    pretrain_all(days=30, output_dir=out)
    return out


# ---------------------------------------------------------------------------
# 1. All expected .pkl files are created
# ---------------------------------------------------------------------------


def test_all_pkl_files_created(model_dir):
    """pretrain_all must create one .pkl file per strategy × symbol."""
    expected = len(STRATEGIES) * len(SYMBOLS)
    pkl_files = glob.glob(os.path.join(model_dir, "*.pkl"))
    assert len(pkl_files) == expected, (
        f"Expected {expected} .pkl files, found {len(pkl_files)}: {pkl_files}"
    )


def test_expected_filenames_present(model_dir):
    """Each expected filename must exist on disk."""
    for strategy_name in STRATEGIES:
        for symbol in SYMBOLS:
            fname = _model_filename(symbol, strategy_name)
            full = os.path.join(model_dir, fname)
            assert os.path.isfile(full), f"Missing model file: {full}"


# ---------------------------------------------------------------------------
# 2. Saved model can be loaded and generate_signal called
# ---------------------------------------------------------------------------


def test_load_and_generate_signal(model_dir):
    """Load one saved model (BTC/USDT + rsi) and call generate_signal."""
    pkl_path = os.path.join(model_dir, _model_filename("BTC/USDT", "rsi"))
    assert os.path.isfile(pkl_path), f"BTC_USDT_rsi.pkl not found at {pkl_path}"

    strategy = RSIStrategy()
    loaded = MLSignalModel(base_strategy=strategy)
    loaded.load(pkl_path)

    # Generate a small synthetic OHLCV window and run inference
    df = generate_ohlcv("BTC/USDT", n_candles=200)
    enriched = strategy.add_indicators(df.copy())

    signal = loaded.generate_signal(enriched)
    assert signal in ("buy", "sell", None), (
        f"generate_signal must return 'buy', 'sell', or None; got {signal!r}"
    )


# ---------------------------------------------------------------------------
# 3. predict_proba is well-formed after loading
# ---------------------------------------------------------------------------


def test_loaded_predict_proba(model_dir):
    """predict_proba on a loaded model returns proper keys that sum to 1."""
    pkl_path = os.path.join(model_dir, _model_filename("ETH/USDT", "macd"))
    assert os.path.isfile(pkl_path), f"ETH_USDT_macd.pkl not found at {pkl_path}"

    from strategies.macd import MACDStrategy

    strategy = MACDStrategy()
    loaded = MLSignalModel(base_strategy=strategy)
    loaded.load(pkl_path)

    df = generate_ohlcv("ETH/USDT", n_candles=200)
    enriched = strategy.add_indicators(df.copy())
    proba = loaded.predict_proba(enriched)

    assert set(proba.keys()) == {"buy", "sell", "hold"}, (
        f"predict_proba keys must be {{buy, sell, hold}}; got {set(proba.keys())}"
    )
    total = sum(proba.values())
    assert abs(total - 1.0) < 1e-6, f"Probabilities must sum to 1.0; got {total}"


# ---------------------------------------------------------------------------
# 4. pretrain_all returns accuracy dict with correct keys
# ---------------------------------------------------------------------------


def test_pretrain_returns_accuracy_dict(model_dir):
    """The return value of pretrain_all must map each key to a float in [0, 1]."""
    # Re-run for a single combo to inspect the return value cheaply
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        results = pretrain_all(days=15, output_dir=tmpdir)

    expected_count = len(STRATEGIES) * len(SYMBOLS)
    assert len(results) == expected_count, (
        f"Expected {expected_count} result entries, got {len(results)}"
    )
    for key, acc in results.items():
        assert isinstance(acc, float), f"Accuracy for '{key}' is not a float: {acc!r}"
        assert 0.0 <= acc <= 1.0, f"Accuracy for '{key}' out of [0,1]: {acc}"

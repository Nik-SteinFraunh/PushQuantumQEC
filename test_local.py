# -*- coding: utf-8 -*-
"""
test_local.py

Local tests for the QISKIT track (no hardware, no token needed):

    make_stim_circuit()  →  stim_to_qiskit()  →  Aer  →  counts_to_measurement_array()
                         →  extract_syndromes()  →  PyMatching

Run:
    python test_local.py          # runs everything and prints a summary
    pytest -q test_local.py       # same tests through pytest

Requires: stim, pymatching, numpy, qiskit, qiskit-aer (see requirements.txt).

NOTE: only the Z basis is tested through Qiskit. X-basis circuits use RX / MX,
which stim_to_qiskit() / get_qubit_lists() do not handle yet.
"""

import numpy as np
import pymatching
import stim
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from extract_syndromes import extract_syndromes
from internal_helpers import counts_to_measurement_array, get_meas_order
from stim_to_qiskit import stim_to_qiskit
from surface_code import DEFAULT_NOISE, make_stim_circuit


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def aer_noise_model(p: float = 0.003) -> NoiseModel:
    """Simple Aer noise model roughly matching DEFAULT_NOISE."""
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p, 1), ["h"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p, 2), ["cx"])
    nm.add_all_qubit_readout_error(ReadoutError([[1 - p, p], [p, 1 - p]]))
    return nm


def run_aer(
    distance: int = 3,
    rounds: int = 3,
    shots: int = 5_000,
    noise_model: NoiseModel = None,
    seed: int = 1234,
) -> tuple[dict, stim.Circuit]:
    """
    Runs the no-reset Z-memory circuit on Aer, exactly like run_qiskit() would on
    hardware, and returns (syndromes dict, stim circuit that ran).
    """
    stim_circ = make_stim_circuit(distance, rounds, "Z", no_reset=True, noise=None)
    qc, _, meas_order = stim_to_qiskit(stim_circ)
    assert meas_order == get_meas_order(stim_circ), "measurement order mismatch"

    sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    qc_t = transpile(qc, basis_gates=["h", "cx", "measure", "reset"], optimization_level=0)
    counts = sim.run(qc_t, shots=shots).result().get_counts()

    raw = counts_to_measurement_array(counts, stim_circ.num_measurements, shots)
    return extract_syndromes(raw, stim_circ, print_summary=False), stim_circ


def decode(syndromes: dict, distance: int, rounds: int, noise: dict = DEFAULT_NOISE) -> tuple[float, float]:
    """Decodes with PyMatching using the DEM of the *no-reset* noisy circuit. Returns (LER, 1-sigma)."""
    dem = make_stim_circuit(distance, rounds, "Z", no_reset=True, noise=noise) \
        .detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    pred = matcher.decode_batch(syndromes["det_events"])
    fails = (pred != syndromes["obs_flips"]).any(axis=1)
    ler = fails.mean()
    return float(ler), float(np.sqrt(ler * (1 - ler) / len(fails)))


# ─────────────────────────────────────────────────────────────────────────────
#  1. make_stim_circuit: argument validation
# ─────────────────────────────────────────────────────────────────────────────

def test_invalid_arguments():
    bad_calls = [
        (dict(distance=4, rounds=1), ValueError),
        (dict(distance=1, rounds=1), ValueError),
        (dict(distance=3.0, rounds=1), TypeError),
        (dict(distance=3, rounds=0), ValueError),
        (dict(distance=3, rounds=1, basis="Y"), ValueError),
        (dict(distance=3, rounds=1, noise={"foo": 0.1}), ValueError),
        (dict(distance=3, rounds=1, noise={"after_clifford_depolarization": 2}), ValueError),
        (dict(distance=3, rounds=1, noise=[0.1]), TypeError),
    ]
    for kwargs, exc in bad_calls:
        try:
            make_stim_circuit(**kwargs)
        except exc:
            continue
        raise AssertionError(f"{kwargs} should raise {exc.__name__}")


def test_noise_dict_not_mutated():
    noise = dict(DEFAULT_NOISE)
    make_stim_circuit(3, 2, noise=noise)
    make_stim_circuit(3, 2, no_reset=False, noise=noise)
    assert noise == DEFAULT_NOISE


# ─────────────────────────────────────────────────────────────────────────────
#  2. No-reset circuit (Stim only)
# ─────────────────────────────────────────────────────────────────────────────

def test_no_reset_structure():
    for d, r in [(3, 1), (3, 3), (5, 3)]:
        c = make_stim_circuit(d, r, "Z", no_reset=True)
        ref = make_stim_circuit(d, r, "Z", no_reset=False)
        assert "MR" not in str(c.flattened()), "no MR allowed in the no-reset circuit"
        assert c.num_detectors == ref.num_detectors
        assert c.num_measurements == ref.num_measurements
        assert c.num_observables == ref.num_observables


def test_no_reset_detectors_deterministic():
    """Noiseless: no detector may fire. Noisy: DEM must build (fails on non-deterministic detectors)."""
    for basis in "ZX":
        for d, r in [(3, 1), (3, 4), (5, 3)]:
            c = make_stim_circuit(d, r, basis, no_reset=True)
            det, obs = c.compile_detector_sampler().sample(500, separate_observables=True)
            assert not det.any() and not obs.any(), (basis, d, r)
            make_stim_circuit(d, r, basis, no_reset=True, noise=DEFAULT_NOISE) \
                .detector_error_model(decompose_errors=True)


def test_no_reset_ler_matches_reset():
    """Stim + PyMatching: no-reset and reset circuits should give a similar logical error rate."""
    d, r, shots = 3, 3, 20_000
    lers = []
    for no_reset in (False, True):
        c = make_stim_circuit(d, r, "Z", no_reset=no_reset, noise=DEFAULT_NOISE)
        m = pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))
        det, obs = c.compile_detector_sampler(seed=1).sample(shots, separate_observables=True)
        lers.append((m.decode_batch(det) != obs).any(axis=1).mean())
    print(f"  Stim LER (d={d}, r={r}): reset={lers[0]:.4f}  no-reset={lers[1]:.4f}")
    assert abs(lers[0] - lers[1]) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
#  3. Qiskit track: Stim → Qiskit → Aer → syndromes → decoder
# ─────────────────────────────────────────────────────────────────────────────

def test_qiskit_noiseless():
    """Noiseless Aer run of the converted circuit: zero detection events, zero logical flips."""
    for r in (1, 3):
        syn, _ = run_aer(3, r, shots=1_000, noise_model=None)
        assert syn["det_events"].sum() == 0, f"rounds={r}: detectors fired without noise"
        assert syn["obs_flips"].sum() == 0, f"rounds={r}: logical flips without noise"


def test_qiskit_noisy_decoding():
    """Noisy Aer run: decoding must clearly reduce the logical error rate."""
    d, r = 3, 3
    syn, _ = run_aer(d, r, shots=5_000, noise_model=aer_noise_model(0.003))
    raw_ler = syn["obs_flips"].mean()
    ler, err = decode(syn, d, r)
    print(f"  Aer (d={d}, r={r}): mean syndrome weight={syn['det_events'].sum(1).mean():.3f}  "
          f"undecoded={raw_ler:.4f}  decoded LER={ler:.4f} ± {err:.4f}")
    assert ler < raw_ler, "decoder should improve the logical error rate"


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [(name, fn) for name, fn in list(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    raise SystemExit(1 if failed else 0)

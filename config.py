from dataclasses import dataclass
import numpy as np

GOLDEN_RATIO = (1 + 5**0.5) / 2     # 1.618...; note that 1/GR == GR-1 == 0.618
# As of numpy 1.17, Generator is preferable function for doing random numbers
RNG = np.random.default_rng()   # Random float uniformly distributed over [0, 1)

@dataclass
class FullConfig:
    # Intentially make numbers not match; demonstrate real-world imperfection
    MIN_INTEGER, MAX_INTEGER = 1, 100_000_000
    ARRAY_SIZE = 10_000_000
    N_SIMULATIONS = 1_000_000   # i.e. Number of targets

    # Uniform distribution random integers for "targets" and "sorted array"
    TARGETS_UNIFORM = RNG.integers(MIN_INTEGER, MAX_INTEGER, endpoint=True, size=N_SIMULATIONS)
    # NOTE: I will just reuse the same sorted array for all N_SIMULATIONS... no loss of generality?
    ARRAY_UNIFORM = np.sort(RNG.integers(MIN_INTEGER, MAX_INTEGER, endpoint=True, size=ARRAY_SIZE))

    # Normal distribution random integers for "targets" and "sorted array"
    mu = (MAX_INTEGER-MIN_INTEGER) / 2    # Set center of bell curve at center of range [MIN_INTEGER, MAX_INTEGER]
    sigma = (MAX_INTEGER-MIN_INTEGER) / 6   # Set std such that 3 std each direction (99.7%) is "bounds" of our range
    TARGETS_NORMAL = RNG.normal(mu, sigma, size=N_SIMULATIONS).clip(MIN_INTEGER, MAX_INTEGER).round()   # Clamp + round
    ARRAY_NORMAL = np.sort(RNG.normal(mu, sigma, size=ARRAY_SIZE).clip(MIN_INTEGER, MAX_INTEGER).round())

@dataclass
class TrivialConfig:
    MIN_INTEGER, MAX_INTEGER = 1, 1_000_000
    ARRAY_SIZE = 1_000_000
    N_SIMULATIONS = 1_000_000   # i.e. Number of targets

    # NOTE: I will just reuse the same sorted array for all N_SIMULATIONS... no loss of generality?
    ARRAY_UNIFORM = np.arange(MIN_INTEGER, MAX_INTEGER+1)
    assert len(ARRAY_UNIFORM) == ARRAY_SIZE
    ARRAY_NORMAL = ARRAY_UNIFORM

    # Uniform distribution random integers for "targets"
    TARGETS_UNIFORM = RNG.integers(MIN_INTEGER, MAX_INTEGER, endpoint=True, size=N_SIMULATIONS)

    # Normal distribution random integers for "targets"
    mu = (MAX_INTEGER-MIN_INTEGER) / 2    # Set center of bell curve at center of range [MIN_INTEGER, MAX_INTEGER]
    sigma = (MAX_INTEGER-MIN_INTEGER) / 6   # Set std such that 3 std each direction (99.7%) is "bounds" of our range
    TARGETS_NORMAL = RNG.normal(mu, sigma, size=N_SIMULATIONS).clip(MIN_INTEGER, MAX_INTEGER).round()   # Clamp + round

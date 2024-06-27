import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Qt5Agg')   # matplotlib 3.5 changed default backend and PyCharm freezes; revert to avoid problem
from section_search import binary_search, fibonacci_search, golden_section_search_inside, golden_section_search_outside
import argparse


parser = argparse.ArgumentParser(description='GOLDEN_SECTION project experiment script.')
# Add arguments
parser.add_argument('--targets_mode', type=str, default='TRIVIAL', 
                    help=("Select TRIVIAL (default) or FULL experiment mode for the distribution of "
                          "targets (i.e. values to search for within the sorted array)!\n"
                          "TRIVIAL just searches every possible value once; intentionally simple.\n"
                          "FULL enables sampling targets from uniform and normal distributions, "
                          "which is meant to be more relevant to real-world usage."),
                    choices=['TRIVIAL', 'FULL'])
parser.add_argument('--array_mode', type=str, default='TRIVIAL', 
                    help=("Select TRIVIAL (default) or FULL experiment mode for the distribution of "
                          "values in the sorted array (i.e. the array within which we search for the targets)!\n"
                          "TRIVIAL just assigns values equal to the indexes, i.e. a sorted array of size 100 would "
                          "just have values of 1 to 100. This is intentionally simple; note how in reality this "
                          "scenario wouldn't actually require search.\n"
                          "FULL enables sampling values from uniform and normal distributions to fill "
                          "the array, which is meant to be more relevant to real-world usage."),
                    choices=['TRIVIAL', 'FULL'])
# Parse the arguments
args = parser.parse_args()


# [MANUAL] CONFIGURATIONS ######################################################

MIN_INTEGER, MAX_INTEGER = 1, 1_000_000
N_ARRAY_SAMPLES = 1_000_000     # i.e. Number of values to sample for sorted array
ARRAY_DUPLICATES = False    # False means array size may be smaller than number of samples
N_TARGET_SIMULATIONS = 1_000_000    # i.e. Number of targets; not affected by duplicates


# Generate inputs for the simulations ##########################################

# Constants and derived values
GOLDEN_RATIO = (1 + 5**0.5) / 2     # 1.618...; note that 1/GR == GR-1 == 0.618
# As of numpy 1.17, Generator is preferable function for doing random numbers
rng = np.random.default_rng()   # Random float uniformly distributed over [0, 1)
# Normal distribution prep
mu = (MAX_INTEGER-MIN_INTEGER) / 2    # Set center of bell curve at center of range [MIN_INTEGER, MAX_INTEGER]
sigma = (MAX_INTEGER-MIN_INTEGER) / 6   # Set std such that 3 std each direction (99.7%) is "bounds" of our range

# Create targets and sorted array; the lists are work queues needed for 'FULL' mode
TARGETS_LIST = []
ARRAY_LIST = []

if args.targets_mode == 'TRIVIAL':
    TARGETS_TRIVIAL = np.arange(MIN_INTEGER, MAX_INTEGER+1)
    assert len(TARGETS_TRIVIAL) == N_TARGET_SIMULATIONS
    TARGETS_LIST.append(TARGETS_TRIVIAL)
elif args.targets_mode == 'FULL':
    # Uniform and normal distribution random integers for "targets"
    # Note that we would expect duplicate (non-unique) targets
    TARGETS_UNIFORM = rng.integers(MIN_INTEGER, MAX_INTEGER, endpoint=True, size=N_TARGET_SIMULATIONS)
    TARGETS_NORMAL = rng.normal(mu, sigma, size=N_TARGET_SIMULATIONS).clip(MIN_INTEGER, MAX_INTEGER).round()
    TARGETS_LIST.append(TARGETS_UNIFORM)
    TARGETS_LIST.append(TARGETS_NORMAL)

    # VISUALIZE target integer distributions
    fig, ax = plt.subplots()
    ax.hist(TARGETS_UNIFORM, color='C3', alpha=0.5, label='Uniform Distribution')   # Should look like a rectangle block
    ax.hist(TARGETS_NORMAL, color='C4', alpha=0.5, label='Normal Distribution (with clamping and rounding)') # Bell curve
    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.set_xlabel('Random Number from Generator')
    ax.set_ylabel('Frequency (# of Occurrences)')
    ax.set_title(f"Sanity Check: Here are the \"target\" integers we will try to locate\n"
                 f"via recursive search algos on range [{MIN_INTEGER:,.0f}, {MAX_INTEGER:,.0f}]")
    ax.legend()
    fig.tight_layout()
    plt.show()  # First figure of script may not show without this
else:
    raise ValueError("IMPOSSIBLE")

if args.array_mode == 'TRIVIAL':
    ARRAY_TRIVIAL = np.arange(MIN_INTEGER, MAX_INTEGER+1)
    assert len(ARRAY_TRIVIAL) == N_ARRAY_SAMPLES
    assert len(TARGETS_TRIVIAL) == N_TARGET_SIMULATIONS     # Assert connection
    ARRAY_LIST.append(ARRAY_TRIVIAL)
elif args.array_mode == 'FULL':
    # NOTE: I will just reuse the same sorted array for all N_SIMULATIONS... no loss of generality?
    # Uniform and normal distribution random integers for "sorted array"
    # Note that we would expect duplicate (non-unique) array values, even when they are sorted
    ARRAY_UNIFORM = np.sort(rng.integers(MIN_INTEGER, MAX_INTEGER, endpoint=True, size=ARRAY_SIZE))
    ARRAY_NORMAL = np.sort(rng.normal(mu, sigma, size=ARRAY_SIZE).clip(MIN_INTEGER, MAX_INTEGER).round())
    ARRAY_LIST.append(ARRAY_UNIFORM)
    ARRAY_LIST.append(ARRAY_NORMAL)

    # VISUALIZE array integer distributions
    fig, ax = plt.subplots()
    ax.hist(ARRAY_UNIFORM, color='C3', alpha=0.5, label='Uniform Distribution')     # Should look like a rectangle block
    ax.hist(ARRAY_NORMAL, color='C4', alpha=0.5, label='Normal Distribution (with clamping and rounding)')  # Bell curve
    ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.set_xlabel('Random Number from Generator')
    ax.set_ylabel('Frequency (# of Occurrences)')
    ax.set_title("Sanity Check: Here are the integers filling the \"sorted arrays\",\n"
                 "inside which we will search for the \"targets\"")
    ax.legend()
    fig.tight_layout()
    plt.show()  # First figure of script may not show without this
else:
    raise ValueError("IMPOSSIBLE")


# Run the simulations ##########################################################

# Experiment variables - we'll want to do all 4x3x3=36 combinations
# Note that the "trivial" types are technically special cases, independent of the distributions
search_types = ['binary', 'fibonacci', 'golden_inside', 'golden_outside']
search_dispatch = {
    'binary': binary_search,
    'fibonacci': fibonacci_search,
    'golden_inside': golden_section_search_inside,
    'golden_outside': golden_section_search_outside
}
array_types = ['trivial', 'uniform', 'normal']
array_dispatch = {
    'trivial': ARRAY_TRIVIAL,
    'uniform': ARRAY_UNIFORM,
    'normal': ARRAY_NORMAL
}
target_types = ['trivial', 'uniform', 'normal']
target_dispatch = {
    'trivial': TARGETS_TRIVIAL,
    'uniform': TARGETS_UNIFORM,
    'normal': TARGETS_NORMAL
}

# Numpy "vectorized" versions of search algos we can apply to the arrays of targets
apply_search_to_targets_dict = {}
for search_type in search_types:
    for array_type in array_types:
        apply_search_to_targets_dict[search_type, array_type] = \
            np.vectorize(lambda n: search_dispatch[search_type](n, array_dispatch[array_type], 
                                                                verbose=False, force_return=True), 
                         otypes=[float])

# Finally, run the search algos on the targets and record results
results_dict = {}
counter, total_count = 1, len(search_types)*len(target_types)*len(array_types)
for search_type in search_types:
    for target_type in target_types:
        for array_type in array_types:
            print(f"({counter}/{total_count}) Calculating {search_type}, {target_type}, {array_type}...")
            results_dict[search_type, target_type, array_type] = \
                apply_search_to_targets_dict[search_type, array_type](target_dispatch[target_type])
            print("Done.")
            counter += 1


# Results - Binary vs. Golden ##################################################

# TODO: Add trivial targets? Oh wait, no it's literally the exact same as uniform! But more stable for inside/outside...
array_mode = 'normal'
uniform_binary_cuts = pd.Series(results_dict['binary', 'uniform', array_mode])
normal_binary_cuts = pd.Series(results_dict['binary', 'normal', 'trivial'])
uniform_fibonacci_cuts = pd.Series(results_dict['fibonacci', 'uniform', array_mode])
normal_fibonacci_cuts = pd.Series(results_dict['fibonacci', 'normal', array_mode])
uniform_golden_inside_cuts = pd.Series(results_dict['golden_inside', 'uniform', array_mode])
normal_golden_inside_cuts = pd.Series(results_dict['golden_inside', 'normal', array_mode])
uniform_golden_outside_cuts = pd.Series(results_dict['golden_outside', 'uniform', array_mode])
normal_golden_outside_cuts = pd.Series(results_dict['golden_outside', 'normal', array_mode])

# Table
means_table = \
    pd.DataFrame({'Binary': [uniform_binary_cuts.mean(), normal_binary_cuts.mean()],
                  'Fibonacci': [uniform_fibonacci_cuts.mean(), normal_fibonacci_cuts.mean()],
                  'Golden (Inside)': [uniform_golden_inside_cuts.mean(), normal_golden_inside_cuts.mean()],
                  'Golden (Outside)': [uniform_golden_outside_cuts.mean(), normal_golden_outside_cuts.mean()]},
                 index=['Uniform', 'Normal'])
print(means_table)

# Visualize number-of-splits distributions - histogram, control for integer bins


def aesthetic_integer_bins(int_arr):
    """ Generate numpy array of bin boundaries given a numpy array of integers to be binned.
        This is surprisingly tricky in default np.histogram() because
        1) final bin is inclusive of final bin boundary rather than exclusive like every other bin right bound
        2) graphically "centering" bin on the integer involves left and right bounds fractionally around integer
    :param int_arr: numpy array of integers to be binned
    :return: numpy array of bin boundaries (usually float, as it includes fractional bounds around integers)
    """
    if int_arr.dtype != int:
        # Try removing NaNs - easiest way is to use pandas dropna()
        int_arr = pd.Series(int_arr).dropna().astype(int)   # Explicitly cast to int as well
    unique_ints = np.unique(int_arr)
    step = np.min(np.diff(unique_ints))     # Usually 1 when dealing with consecutive integers
    # From half a step below min to half a step above max, allowing each integer to be "centered"
    # Note extra step added to right bound because np.arange() excludes rightmost step
    return np.arange(np.min(unique_ints)-step/2, np.max(unique_ints)+step/2+step, step)


# Uniform
fig, ax = plt.subplots()
histogram_group = [uniform_binary_cuts, uniform_fibonacci_cuts, uniform_golden_inside_cuts, uniform_golden_outside_cuts]
ax.hist(histogram_group, bins=aesthetic_integer_bins(pd.concat(histogram_group)), histtype='bar',
        label=[
            f"Binary (0.5); Mean {means_table.loc[('Uniform', 'Binary')]:.2f}",
            f"Fibonacci (0.618) \"Left\"; Mean {means_table.loc[('Uniform', 'Fibonacci')]:.2f}",
            f"Golden (0.618) \"Inside\"; Mean {means_table.loc[('Uniform', 'Golden (Inside)')]:.2f}",
            f"Golden (0.618) \"Outside\"; Mean {means_table.loc[('Uniform', 'Golden (Outside)')]:.2f}"
        ])
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_xlabel('# of Splits to Arrive at Number')
ax.set_ylabel('Frequency (# of Occurrences)')
ax.set_title(f"Results: Uniform Distribution, {N_SIMULATIONS:,.0f} Samples\n"
             f"Ratio splits of [0, {MAX_INTEGER:,.0f}] needed to arrive at random number")
ax.legend()
fig.tight_layout()
plt.show()

# Normal
fig, ax = plt.subplots()
histogram_group = [normal_binary_cuts, normal_fibonacci_cuts, normal_golden_inside_cuts, normal_golden_outside_cuts]
ax.hist(histogram_group, bins=aesthetic_integer_bins(pd.concat(histogram_group)), histtype='bar',
        label=[
            f"Binary (0.5); Mean {means_table.loc[('Normal', 'Binary')]:.2f}",
            f"Fibonacci (0.618) \"Left\"; Mean {means_table.loc[('Normal', 'Fibonacci')]:.2f}",
            f"Golden (0.618) \"Inside\"; Mean {means_table.loc[('Normal', 'Golden (Inside)')]:.2f}",
            f"Golden (0.618) \"Outside\"; Mean {means_table.loc[('Normal', 'Golden (Outside)')]:.2f}"
        ])
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_xlabel('# of Splits to Arrive at Number')
ax.set_ylabel('Frequency (# of Occurrences)')
ax.set_title(f"Results: Normal Distribution, {N_SIMULATIONS:,.0f} Samples\n"
             f"Ratio splits of [0, {MAX_INTEGER:,.0f}] needed to arrive at random number")
ax.legend()
fig.tight_layout()
plt.show()

# Print statement
binary_supremacy_uniform = \
    (means_table.loc[('Uniform', 'Fibonacci')] - means_table.loc[('Uniform', 'Binary')]) \
    / means_table.loc[('Uniform', 'Binary')]
binary_supremacy_normal = \
    (means_table.loc[('Normal', 'Fibonacci')] - means_table.loc[('Normal', 'Binary')]) \
    / means_table.loc[('Normal', 'Binary')]
print(f"Under uniform distribution,\n"
      f"  binary search is {binary_supremacy_uniform*100:.2f}% more efficient\n"
      f"  than Fibonacci search over [{MIN_INTEGER:,.0f}, {MAX_INTEGER:,.0f}], {N_SIMULATIONS:,.0f} simulations.")
print(f"Under normal distribution,\n"
      f"  binary search is {binary_supremacy_normal*100:.2f}% more efficient\n"
      f"  than Fibonacci search over [{MIN_INTEGER:,.0f}, {MAX_INTEGER:,.0f}], {N_SIMULATIONS:,.0f} simulations.")



# NOTE: This is a sanity check on how many targets actually exist in the sorted array; constant across each row...
n_table = \
    pd.DataFrame({'Binary': [pd.Series(results_dict['binary', 'uniform', 'uniform']).dropna().shape[0],
                             pd.Series(results_dict['binary', 'uniform', 'normal']).dropna().shape[0],
                             pd.Series(results_dict['binary', 'normal', 'uniform']).dropna().shape[0],
                             pd.Series(results_dict['binary', 'normal', 'normal']).dropna().shape[0]],
                  'Fibonacci': [pd.Series(results_dict['fibonacci', 'uniform', 'uniform']).dropna().shape[0],
                                pd.Series(results_dict['fibonacci', 'uniform', 'normal']).dropna().shape[0],
                                pd.Series(results_dict['fibonacci', 'normal', 'uniform']).dropna().shape[0],
                                pd.Series(results_dict['fibonacci', 'normal', 'normal']).dropna().shape[0]],
                  'Golden (Inside)': [pd.Series(results_dict['golden_inside', 'uniform', 'uniform']).dropna().shape[0],
                                      pd.Series(results_dict['golden_inside', 'uniform', 'normal']).dropna().shape[0],
                                      pd.Series(results_dict['golden_inside', 'normal', 'uniform']).dropna().shape[0],
                                      pd.Series(results_dict['golden_inside', 'normal', 'normal']).dropna().shape[0]],
                 'Golden (Outside)': [pd.Series(results_dict['golden_outside', 'uniform', 'uniform']).dropna().shape[0],
                                      pd.Series(results_dict['golden_outside', 'uniform', 'normal']).dropna().shape[0],
                                      pd.Series(results_dict['golden_outside', 'normal', 'uniform']).dropna().shape[0],
                                      pd.Series(results_dict['golden_outside', 'normal', 'normal']).dropna().shape[0]]},
                 index=['Uniform Targets, Uniform Array',
                        'Uniform Targets, Normal Array',
                        'Normal Targets, Uniform Array',
                        'Normal Targets, Normal Array'])
print(n_table)


# Results - Binary vs. Golden ##################################################

# Table
means_table = \
    pd.DataFrame({'Binary': [np.nanmean(results_dict['uniform', 'binary', 'uniform']),
                             np.nanmean(results_dict['uniform', 'binary', 'normal']),
                             np.nanmean(results_dict['normal', 'binary', 'uniform']),
                             np.nanmean(results_dict['normal', 'binary', 'normal'])],
                  'Fibonacci': [np.nanmean(results_dict['uniform', 'fibonacci', 'uniform']),
                                np.nanmean(results_dict['uniform', 'fibonacci', 'normal']),
                                np.nanmean(results_dict['normal', 'fibonacci', 'uniform']),
                                np.nanmean(results_dict['normal', 'fibonacci', 'normal'])],
                  'Golden (Inside)': [np.nanmean(results_dict['uniform', 'golden_inside', 'uniform']),
                                      np.nanmean(results_dict['uniform', 'golden_inside', 'normal']),
                                      np.nanmean(results_dict['normal', 'golden_inside', 'uniform']),
                                      np.nanmean(results_dict['normal', 'golden_inside', 'normal'])],
                  'Golden (Outside)': [np.nanmean(results_dict['uniform', 'golden_outside', 'uniform']),
                                       np.nanmean(results_dict['uniform', 'golden_outside', 'normal']),
                                       np.nanmean(results_dict['normal', 'golden_outside', 'uniform']),
                                       np.nanmean(results_dict['normal', 'golden_outside', 'normal'])]},
                 index=['Uniform Targets, Uniform Array',
                        'Uniform Targets, Normal Array',
                        'Normal Targets, Uniform Array',
                        'Normal Targets, Normal Array'])
print(means_table)

# NOTE: This is a sanity check on how many targets actually exist in the sorted array; constant across each row...
n_table = \
    pd.DataFrame({'Binary': [pd.Series(results_dict['uniform', 'binary', 'uniform']).dropna().shape[0],
                             pd.Series(results_dict['uniform', 'binary', 'normal']).dropna().shape[0],
                             pd.Series(results_dict['normal', 'binary', 'uniform']).dropna().shape[0],
                             pd.Series(results_dict['normal', 'binary', 'normal']).dropna().shape[0]],
                  'Fibonacci': [pd.Series(results_dict['uniform', 'fibonacci', 'uniform']).dropna().shape[0],
                                pd.Series(results_dict['uniform', 'fibonacci', 'normal']).dropna().shape[0],
                                pd.Series(results_dict['normal', 'fibonacci', 'uniform']).dropna().shape[0],
                                pd.Series(results_dict['normal', 'fibonacci', 'normal']).dropna().shape[0]],
                  'Golden (Inside)': [pd.Series(results_dict['uniform', 'golden_inside', 'uniform']).dropna().shape[0],
                                      pd.Series(results_dict['uniform', 'golden_inside', 'normal']).dropna().shape[0],
                                      pd.Series(results_dict['normal', 'golden_inside', 'uniform']).dropna().shape[0],
                                      pd.Series(results_dict['normal', 'golden_inside', 'normal']).dropna().shape[0]],
                 'Golden (Outside)': [pd.Series(results_dict['uniform', 'golden_outside', 'uniform']).dropna().shape[0],
                                      pd.Series(results_dict['uniform', 'golden_outside', 'normal']).dropna().shape[0],
                                      pd.Series(results_dict['normal', 'golden_outside', 'uniform']).dropna().shape[0],
                                      pd.Series(results_dict['normal', 'golden_outside', 'normal']).dropna().shape[0]]},
                 index=['Uniform Targets, Uniform Array',
                        'Uniform Targets, Normal Array',
                        'Normal Targets, Uniform Array',
                        'Normal Targets, Normal Array'])
print(n_table)

# Visualize number-of-splits distributions - histogram, control for integer bins


def aesthetic_integer_bins(int_arr):
    """ Generate numpy array of bin boundaries given a numpy array of integers to be binned.
        This is surprisingly tricky in default np.histogram() because
        1) final bin is inclusive of final bin boundary rather than exclusive like every other bin right bound
        2) graphically "centering" bin on the integer involves left and right bounds fractionally around integer
    :param int_arr: numpy array of integers to be binned
    :return: numpy array of bin boundaries (usually float, as it includes fractional bounds around integers)
    """
    if int_arr.dtype != int:
        # Try removing NaNs - easiest way is to use pandas dropna()
        int_arr = pd.Series(int_arr).dropna().astype(int)   # Explicitly cast to int as well
    unique_ints = np.unique(int_arr)
    step = np.min(np.diff(unique_ints))     # Usually 1 when dealing with consecutive integers
    # From half a step below min to half a step above max, allowing each integer to be "centered"
    # Note extra step added to right bound because np.arange() excludes rightmost step
    return np.arange(np.min(unique_ints)-step/2, np.max(unique_ints)+step/2+step, step)


# Uniform
# TODO: Figure out which charts/stats are actually important!
fig, ax = plt.subplots()
# ax.hist(uniform_binary_cuts, bins=aesthetic_integer_bins(uniform_binary_cuts), alpha=0.5,
#         label=f"Binary (0.5); Mean {means_table.loc[('Uniform', 'Binary')]:.2f}")
# ax.hist(uniform_golden_inside_cuts, bins=aesthetic_integer_bins(uniform_golden_inside_cuts), alpha=0.5,
#         label=f"Golden (0.618) \"Inside\"; Mean {means_table.loc[('Uniform', 'Golden (Inside)')]:.2f}")
# ax.hist(uniform_golden_outside_cuts, bins=aesthetic_integer_bins(uniform_golden_outside_cuts), alpha=0.5,
#         label=f"Golden (0.618) \"Outside\"; Mean {means_table.loc[('Uniform', 'Golden (Outside)')]:.2f}")
ax.hist(results_dict['uniform', 'binary', 'uniform'], bins=aesthetic_integer_bins(results_dict['uniform', 'binary', 'uniform']), alpha=0.5,
        label=f"Binary (0.5) Uniform-Uniform; Mean {means_table.loc[('Uniform Targets, Uniform Array', 'Binary')]:.2f}")
ax.hist(results_dict['uniform', 'binary', 'normal'], bins=aesthetic_integer_bins(results_dict['uniform', 'binary', 'normal']), alpha=0.5,
        label=f"Binary (0.5) Uniform-Normal; Mean {means_table.loc[('Uniform Targets, Normal Array', 'Binary')]:.2f}")
ax.hist(results_dict['normal', 'binary', 'uniform'], bins=aesthetic_integer_bins(results_dict['normal', 'binary', 'uniform']), alpha=0.5,
        label=f"Binary (0.5) Normal-Uniform; Mean {means_table.loc[('Normal Targets, Uniform Array', 'Binary')]:.2f}")
ax.hist(results_dict['normal', 'binary', 'normal'], bins=aesthetic_integer_bins(results_dict['normal', 'binary', 'normal']), alpha=0.5,
        label=f"Binary (0.5) Normal-Normal; Mean {means_table.loc[('Normal Targets, Normal Array', 'Binary')]:.2f}")

# ax.hist(results_dict['uniform', 'binary', 'uniform'], bins=aesthetic_integer_bins(results_dict['uniform', 'binary', 'uniform']), alpha=0.5,
#         label=f"Binary (0.5) Uniform-Uniform; Mean {means_table.loc[('Uniform Targets, Uniform Array', 'Binary')]:.2f}")
# ax.hist(results_dict['uniform', 'binary', 'normal'], bins=aesthetic_integer_bins(results_dict['uniform', 'binary', 'normal']), alpha=0.5,
#         label=f"Binary (0.5); Mean {means_table.loc[('Uniform', 'Binary')]:.2f}")
# ax.hist(results_dict['uniform', 'binary', 'uniform'], bins=aesthetic_integer_bins(results_dict['uniform', 'binary', 'uniform']), alpha=0.5,
#         label=f"Binary (0.5); Mean {means_table.loc[('Uniform', 'Binary')]:.2f}")
# ax.hist(results_dict['uniform', 'binary', 'normal'], bins=aesthetic_integer_bins(results_dict['uniform', 'binary', 'normal']), alpha=0.5,
#         label=f"Binary (0.5); Mean {means_table.loc[('Uniform', 'Binary')]:.2f}")
# ax.hist(uniform_golden_inside_cuts, bins=aesthetic_integer_bins(uniform_golden_inside_cuts), alpha=0.5,
#         label=f"Golden (0.618) \"Inside\"; Mean {means_table.loc[('Uniform', 'Golden (Inside)')]:.2f}")
# ax.hist(uniform_golden_outside_cuts, bins=aesthetic_integer_bins(uniform_golden_outside_cuts), alpha=0.5,
#         label=f"Golden (0.618) \"Outside\"; Mean {means_table.loc[('Uniform', 'Golden (Outside)')]:.2f}")
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_xlabel('# of Splits to Arrive at Number')
ax.set_ylabel('Frequency (# of Occurrences)')
ax.set_title(f"Results: Uniform Distribution, {N_SIMULATIONS:,.0f} Samples\n"
             f"Ratio splits of [0, {MAX_INTEGER:,.0f}] needed to arrive at random number")
ax.legend()
fig.tight_layout()

# Normal
fig, ax = plt.subplots()
ax.hist(normal_binary_cuts, bins=aesthetic_integer_bins(normal_binary_cuts), alpha=0.5,
        label=f"Binary (0.5); Mean {means_table.loc[('Normal', 'Binary')]:.2f}")
ax.hist(normal_golden_inside_cuts, bins=aesthetic_integer_bins(normal_golden_inside_cuts), alpha=0.5,
        label=f"Golden (0.618) \"Inside\"; Mean {means_table.loc[('Normal', 'Golden (Inside)')]:.2f}")
ax.hist(normal_golden_outside_cuts, bins=aesthetic_integer_bins(normal_golden_outside_cuts), alpha=0.5,
        label=f"Golden (0.618) \"Outside\"; Mean {means_table.loc[('Normal', 'Golden (Outside)')]:.2f}")
ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
ax.set_xlabel('# of Splits to Arrive at Number')
ax.set_ylabel('Frequency (# of Occurrences)')
ax.set_title(f"Results: Normal Distribution, {N_SIMULATIONS:,.0f} Samples\n"
             f"Ratio splits of [0, {MAX_INTEGER:,.0f}] needed to arrive at random number")
ax.legend()
fig.tight_layout()

# Print statement
binary_supremacy_uniform = \
    (means_table.loc[('Uniform', 'Golden (Outside)')] - means_table.loc[('Uniform', 'Binary')]) \
    / means_table.loc[('Uniform', 'Binary')]
binary_supremacy_normal = \
    (means_table.loc[('Normal', 'Golden (Outside)')] - means_table.loc[('Normal', 'Binary')]) \
    / means_table.loc[('Normal', 'Binary')]
print(f"Under uniform distribution,\n"
      f"  binary search is {binary_supremacy_uniform*100:.2f}% more efficient\n"
      f"  than Fibonacci search over [0, {MAX_INTEGER:,.0f}], {N_SIMULATIONS:,.0f} simulations.")
print(f"Under normal distribution,\n"
      f"  binary search is {binary_supremacy_normal*100:.2f}% more efficient\n"
      f"  than Fibonacci search over [0, {MAX_INTEGER:,.0f}], {N_SIMULATIONS:,.0f} simulations.")

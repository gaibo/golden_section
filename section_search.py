from config import GOLDEN_RATIO
from typing import Sequence, Any, TypeVar, Protocol, Optional


# NOTE: This is me trying to practice Python type-hinting and generic types...
class SupportsLT(Protocol):
    def __lt__(self, __other: Any, /) -> bool: ...  # Double underscore for positional-only arg


CT = TypeVar('CT', bound=SupportsLT)    # "Comparable" generic type! (in Python 3 implementations all you need is <)


def section_search(target: CT, sorted_arr: Sequence[CT], split: float = 0.5, 
                   use_insideoutside: bool = False, split_insideoutside: str = 'inside', 
                   verbose: bool = True, force_return: bool = False, return_index: bool = False) -> Optional[int]:
    """ Generalization of binary search such that the split (0.5 for binary) may be changed.
        This becomes somewhat convoluted as it introduces asymmetry - do you want the 0.7 split on left or right?
        (Fibonacci search for example, for Fibonacci sequence math reasons, keeps ~0.618 split on left side always.)
        My answer is to introduce an option to keep split fraction "inside" or "outside", which requires
        storing state on whether the previous split sent you "left" or "right". e.g.,

        [----------] (10 difference between bounds), split=0.7; initial inside/outside N/A, previous left/right N/A
        Split 1 - arbitrary because no state; let it be "inside", as if previous "right": [-------|---]
        Split 2a - inside, previous left:  [--|-----]
        Split 2b - inside, previous right:          [--|-]
        Split 2c - outside, previous left: [-----|--]
        Split 2d - outside, previous right:         [-|--]

        I'm imagining "inside" vs. "outside" as a way someone would try to bias splits
        to benefit in certain distributions - if you know numbers near the center of the
        bounds are more common, could you get there faster by playing with this parameter?
        Answer: Empirically, a split >0.5 does better on normal distribution with "outside" vs. "inside", but both
                are significantly worse than just binary search. e.g., on a million uniform integers [1, 1,000,000],
                Mean # of Splits  Binary 0.5  Golden 0.618 (Inside)  Golden 0.618 (Outside)
                Uniform            19.951814              20.701467               20.701008
                Normal             19.948935              20.987887               20.528782
                Note inside/outside perform the exact same on a uniform distribution, as you would expect!
                On uniform, Fibonacci search (0.618) is about 3.8% worse here than binary search (0.5).
    :param target: (comparable) value to search for (within the array)
    :param sorted_arr: pre-sorted array of (indexable, comparable) values
    :param split: fractional split of bounds; set 0.5 for binary search
    :param use_insideoutside: set True to use my persistent "inside/outside" split system; set False for static system
    :param split_insideoutside: 'inside' or 'outside' split concept defined in description above
    :param verbose: set True for print statements describing splits
    :param force_return: set True to force return of non-None even when target not found
    :param return_index: set True to return index within array of the found value; this is what search should return...
    :return: integer number of comparisons (think splits) needed to arrive at target within sorted array
             or None if target is not found within array; if return_index is set, it's integer number of index instead!
    """
    assert 0 < split < 1
    assert split_insideoutside in ['inside', 'outside']

    # Initialize
    i_left: int = 0  # Left lower bound index
    i_right: int = len(sorted_arr) - 1   # Right upper bound index
    completed_comparisons: int = 0
    prev_leftright = 'right'    # Arbitrarily intialized; note difference it makes when switching inside/outside

    # Perform binary search until failure condition: boundary indexes overlap (still need to check at index if equals)
    while i_left <= i_right:
        split_size_floor = int(split * (i_right - i_left))  # "floor" because left/right confusing later

        # Complicated cases describing how someone might use a ratio split in relation to "center" of range
        if (use_insideoutside is False
                or (split_insideoutside == 'inside' and prev_leftright == 'right')
                or (split_insideoutside == 'outside' and prev_leftright == 'left')):
            # Put split from left to right - e.g. split=0.7 -> [-------|---]
            i_mid: int = i_left + split_size_floor     # int in [i_left, i_right-1]
        elif ((split_insideoutside == 'inside' and prev_leftright == 'left')
              or (split_insideoutside == 'outside' and prev_leftright == 'right')):
            # Put split from right to left - e.g. split=0.7 -> [---|-------]
            i_mid: int = i_right - split_size_floor    # int in [i_left+1, i_right]
        else:
            raise ValueError(f"IMPOSSIBLE: ratio split case ('{split_insideoutside}', '{prev_leftright}')")

        # Split towards target's appropriate bounds
        value_left: CT = sorted_arr[i_left]
        value_right: CT = sorted_arr[i_right]
        value_mid: CT = sorted_arr[i_mid]
        completed_comparisons += 1  # Not done yet, but want this ahead of verbose
        if verbose:
            print(f"Evaluation {completed_comparisons}: target {target}")
            print(f"\tindexes: [{i_left}, {i_mid}] | [{i_mid}, {i_right}]")
            print(f"\tvalues:  [{value_left}, {value_mid}] | [{value_mid}, {value_right}]")
        if target < value_mid:
            if verbose:
                print("Went LEFT!")
            i_right = i_mid - 1     # May now violate i_right > i_left!
            prev_leftright = 'left'
        elif target > value_mid:
            if verbose:
                print("Went RIGHT!")
            i_left = i_mid + 1  # May now violate i_left < i_right!
            prev_leftright = 'right'
        else:
            if verbose:
                print(f"****FOUND {value_mid} at index {i_mid} after {completed_comparisons} comparisons!")
            return completed_comparisons if not return_index else i_mid

    # Unsuccessful at finding target in sorted_arr
    if verbose:
        print(f"****UNFOUND {target} after all {completed_comparisons} comparisons!")
    if not force_return:
        return None
    else:
        return completed_comparisons if not return_index else i_right   # Note how i_right now is index with value less


def binary_search(target, sorted_arr, **kwargs):
    return section_search(target, sorted_arr, 0.5, use_insideoutside=False, **kwargs)


def fibonacci_search(target, sorted_arr, **kwargs):
    # Not strictly Fibonacci search because I skip straight to golden ratio...
    # Note that default use_insideoutside=False puts split on the left, as desired
    return section_search(target, sorted_arr, split=1/GOLDEN_RATIO, use_insideoutside=False, **kwargs)


def golden_section_search_inside(target, sorted_arr, **kwargs):
    return section_search(target, sorted_arr, split=1/GOLDEN_RATIO, use_insideoutside=True,
                          split_insideoutside='inside', **kwargs)


def golden_section_search_outside(target, sorted_arr, **kwargs):
    return section_search(target, sorted_arr, split=1/GOLDEN_RATIO, use_insideoutside=True,
                          split_insideoutside='outside', **kwargs)

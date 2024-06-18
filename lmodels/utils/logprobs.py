import numpy as np
import numpy.typing as npt


def concatenate_logprobs(
    first: npt.NDArray[np.float_], second: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """
    Concatenates two arrays of logprobs across the first dimension.

    ### Parameters
    --------------
    `first`: the first array of logprobs to concatenate.
    `second`: the second array of logprobs to concatenate.

    ### Returns
    --------------
    A numpy array with the concatenated logprobs. The shape of the returned array is
    `(n_context_first + n_context_second, n_samples, n_tokens)`, where `n_samples` is the
    maximum number of samples between `first` and `second`, and `n_tokens` is the maximum
    number of tokens between `first` and `second`. The array is filled with `np.nan` where
    the original arrays did not have values.
    """

    n_context_first, n_samples_first, n_tokens_first = first.shape
    n_context_second, n_samples_second, n_tokens_second = second.shape

    n_samples = max(n_samples_first, n_samples_second)
    n_tokens = max(n_tokens_first, n_tokens_second)

    # We extend the arrays with `np.nan`
    first_extended = np.full((n_context_first, n_samples, n_tokens), np.nan)
    first_extended[:, :n_samples_first, :n_tokens_first] = first
    second_extended = np.full((n_context_second, n_samples, n_tokens), np.nan)
    second_extended[:, :n_samples_second, :n_tokens_second] = second

    # We return the concatenated arrays
    return np.concatenate([first_extended, second_extended], axis=0)

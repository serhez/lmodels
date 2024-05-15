from __future__ import annotations

from dataclasses import dataclass


@dataclass(kw_only=True)
class Usage:
    """Usage statistics for a model."""

    n_calls: int = 0
    """The number of calls to the model's forward pass."""

    n_tokens_context: int = 0
    """The number of tokens in the context."""

    n_tokens_output: int = 0
    """The number of tokens in the output."""

    def __add__(self, other: Usage | dict[str, int] | None) -> Usage:
        """
        Add two usage statistics together.

        ### Parameters
        --------------
        `other`: the usage statistics to add.
        - If the usage is a dictionary and a key is missing, it is assumed to be 0; the required keys are:
            - `n_calls`: the number of calls to add.
            - `n_tokens_context`: the number of tokens in the context to add.
            - `n_tokens_output`: the number of tokens in the output to add.

        ### Returns
        -----------
        The updated object.
        """

        if isinstance(other, dict):
            return Usage(
                n_calls=self.n_calls + other.get("n_calls", 0),
                n_tokens_context=self.n_tokens_context
                + other.get("n_tokens_context", 0),
                n_tokens_output=self.n_tokens_output + other.get("n_tokens_output", 0),
            )

        return Usage(
            n_calls=self.n_calls + other.n_calls if other is not None else self.n_calls,
            n_tokens_context=self.n_tokens_context + other.n_tokens_context
            if other is not None
            else self.n_tokens_context,
            n_tokens_output=self.n_tokens_output + other.n_tokens_output
            if other is not None
            else self.n_tokens_output,
        )

    def __radd__(self, other: Usage | dict[str, int] | None) -> Usage:
        """
        Add two usage statistics together.

        ### Parameters
        --------------
        `other`: the usage statistics to add.
        - If the usage is a dictionary and a key is missing, it is assumed to be 0; the required keys are:
            - `n_calls`: the number of calls to add.
            - `n_tokens_context`: the number of tokens in the context to add.
            - `n_tokens_output`: the number of tokens in the output to add.

        ### Returns
        -----------
        The updated object.
        """

        return self + other

    def __iadd__(self, other: Usage | dict[str, int] | None) -> Usage:
        """
        Add two usage statistics together in place.

        ### Parameters
        --------------
        `usage`: the usage statistics to add.

        ### Returns
        -----------
        The updated object.

        ### Notes
        - If the usage is a dictionary and a key is missing, it is assumed to be 0; the required keys are:
            - `n_calls`: the number of calls to add.
            - `n_tokens_context`: the number of tokens in the context to add.
            - `n_tokens_output`: the number of tokens in the output to add.
        """

        if isinstance(other, dict):
            self.n_calls += other.get("n_calls", 0)
            self.n_tokens_context += other.get("n_tokens_context", 0)
            self.n_tokens_output += other.get("n_tokens_output", 0)
        elif other is not None:
            self.n_calls += other.n_calls
            self.n_tokens_context += other.n_tokens_context
            self.n_tokens_output += other.n_tokens_output

        return self

    def reset(self) -> Usage:
        """
        Reset the usage statistics.

        ### Returns
        -----------
        The reset object.
        """

        self.n_calls = 0
        self.n_tokens_context = 0
        self.n_tokens_output = 0

        return self

    def add_call(self, n_tokens_context: int, n_tokens_output: int) -> Usage:
        """
        Add a call to the usage statistics.

        ### Parameters
        --------------
        `n_tokens_context`: the number of tokens in the context.
        `n_tokens_output`: the number of tokens in the output.

        ### Returns
        -----------
        The updated object.
        """

        self.n_calls += 1
        self.n_tokens_context += n_tokens_context
        self.n_tokens_output += n_tokens_output

        return self

    def add(self, n_calls: int, n_tokens_context: int, n_tokens_output: int) -> Usage:
        """
        Add usage statistics to the current statistics.

        ### Parameters
        --------------
        `n_calls`: the number of calls to add.
        `n_tokens_context`: the number of tokens in the context to add.
        `n_tokens_output`: the number of tokens in the output to add.

        ### Returns
        -----------
        The updated object.
        """

        self.n_calls += n_calls
        self.n_tokens_context += n_tokens_context
        self.n_tokens_output += n_tokens_output

        return self

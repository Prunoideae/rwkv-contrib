from torch import Tensor


class LogitsTransformer:
    """
    Transform a tensor of logits into a transformed tensor of logits.

    An example is filtering the logits by a certain grammar.
    """

    def transform(self, logits: Tensor) -> Tensor:
        """
        Transform the logits.
        """
        ...


class LogitsSampler:
    """
    Samples the logits.
    """

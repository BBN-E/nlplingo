import torch
from torch.nn import ParameterList, Parameter


class Elmo(torch.nn.Module):
    """
    Computes a parameterised scalar mixture of N tensors, `mixture = gamma * sum(s_k * tensor_k)`
    where `s = softmax(w)`, with `w` and `gamma` scalar parameters.
    In addition, if `do_layer_norm=True` then apply layer normalization to each tensor
    before weighting.
    """

    def __init__(
            self,
            mixture_size,
            do_layer_norm=False,
            trainable=True,
    ):
        super().__init__()
        self.mixture_size = mixture_size
        self.do_layer_norm = do_layer_norm

        initial_scalar_parameters = [0.0] * mixture_size

        self.scalar_parameters = ParameterList(
            [
                Parameter(
                    torch.FloatTensor([initial_scalar_parameters[i]]), requires_grad=trainable
                )
                for i in range(mixture_size)
            ]
        )
        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=trainable)

    def forward(self, tensors):
        """
        tensors.shape = num feature layers * [batch size, num tokens, rep dim]
        """

        def _do_layer_norm(tensor, broadcast_mask, num_elements_not_masked):
            tensor_masked = tensor * broadcast_mask
            mean = torch.sum(tensor_masked) / num_elements_not_masked
            variance = (
                    torch.sum(((tensor_masked - mean) * broadcast_mask) ** 2) / num_elements_not_masked
            )
            return (tensor - mean) / torch.sqrt(variance + 1e-12)

        normed_weights = torch.nn.functional.softmax(
            torch.cat([parameter for parameter in self.scalar_parameters]), dim=0
        )
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)

        pieces = []
        for weight, tensor in zip(normed_weights, tensors):
            pieces.append(weight * tensor)
        return self.gamma * sum(pieces)

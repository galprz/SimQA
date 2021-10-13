from torchtext.data.example import Example


def sort_by_indices(x: Example) -> int:
    return x.indices

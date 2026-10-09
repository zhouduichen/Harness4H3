def test_tensor_stats_is_finite_and_digest_stable():
    import torch

    from tools.wan_p0_compare import tensor_stats

    value = torch.tensor([[[[[1.0, 3.0]]]]], dtype=torch.float16)
    first = tensor_stats("latent", value)
    second = tensor_stats("latent", value.clone())
    assert first == second
    assert first["shape"] == [1, 1, 1, 1, 2]
    assert first["finite"] is True
    assert first["mean"] == 2.0

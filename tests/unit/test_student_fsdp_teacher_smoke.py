from tools.student_fsdp_teacher_smoke import _target_from_latent_options


def test_smoke_can_use_the_remote_h3_cache_latent_shape():
    target = _target_from_latent_options(16, 16)

    assert (target.latent_height, target.latent_width) == (16, 16)


def test_smoke_keeps_student_target_defaults_without_overrides():
    target = _target_from_latent_options(None, None)

    assert (target.latent_height, target.latent_width) == (32, 32)

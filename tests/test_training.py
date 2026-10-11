from pathlib import Path
from types import SimpleNamespace

import pytest

from polyfid.models import ARM_SPECS, arm_spec
from polyfid.training import (
    CheckpointSchedule,
    PretrainingRecipe,
    TokenProgress,
    TrainingState,
    learning_rate_candidates,
    save_training_checkpoint,
)


def test_seven_arm_protocol_encodes_only_intended_differences():
    assert tuple(ARM_SPECS) == ("C", "B0", "A0", "A_att", "A_gcl", "A_eqs", "B_eqs")
    assert not arm_spec("C").train_on_pi1m
    assert arm_spec("B0").initialization == "scratch"
    assert arm_spec("A0").initialization == "warm"
    assert arm_spec("A_att").masking == "boundary"
    assert arm_spec("A_gcl").relation_weight == 0.1
    assert arm_spec("A_eqs").temperature == arm_spec("A_gcl").temperature == 0.1
    assert arm_spec("B_eqs").relation_objective == arm_spec("A_eqs").relation_objective


def test_progress_counts_non_padding_tokens_and_optimizer_steps():
    progress = TokenProgress()
    assert progress.update([[1, 1, 0], [1, 0, 0]], optimizer_step=True) == 3
    assert progress.non_padding_tokens == 3
    assert progress.optimizer_steps == 1
    assert TokenProgress.from_dict(progress.to_dict()) == progress


def test_checkpoint_schedule_emits_each_predeclared_fraction_once_and_restores():
    schedule = CheckpointSchedule(1000)
    assert schedule.newly_reached(9) == ()
    assert schedule.newly_reached(10) == (1,)
    assert schedule.newly_reached(260) == (5, 10, 25)
    restored = CheckpointSchedule.from_dict(schedule.to_dict())
    assert restored.newly_reached(500) == (50,)
    assert restored.newly_reached(1000) == (100,)
    assert restored.newly_reached(1000) == ()


def test_training_state_round_trip_preserves_resume_position():
    state = TrainingState("A_eqs", 42, epoch=2, global_step=109, non_padding_tokens=9876)
    assert TrainingState.from_dict(state.to_dict()) == state


def test_failed_checkpoint_write_keeps_the_previous_checkpoint(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    target = tmp_path / "last.pt"
    target.write_bytes(b"previous checkpoint")
    temporary = target.with_name(f".{target.name}.tmp")

    def fail_after_partial_write(payload, path):
        Path(path).write_bytes(b"partial checkpoint")
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(torch, "save", fail_after_partial_write)
    component = SimpleNamespace(state_dict=lambda: {})
    with pytest.raises(RuntimeError, match="simulated write failure"):
        save_training_checkpoint(
            target,
            model=component,
            optimizer=component,
            state=TrainingState("A_att", 13),
        )

    assert target.read_bytes() == b"previous checkpoint"
    assert not temporary.exists()


def test_checkpoint_stages_locally_before_serialized_publish(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    staging_dir = tmp_path / "node-local"
    staging_dir.mkdir()
    target = tmp_path / "shared" / "last.pt"
    write_lock = tmp_path / "shared" / ".checkpoint-write.lock"
    original_save = torch.save
    save_paths = []

    def record_save_path(payload, path):
        save_paths.append(Path(path))
        original_save(payload, path)

    monkeypatch.setenv("SLURM_TMPDIR", str(staging_dir))
    monkeypatch.setattr(torch, "save", record_save_path)
    component = SimpleNamespace(state_dict=lambda: {})
    save_training_checkpoint(
        target,
        model=component,
        optimizer=component,
        state=TrainingState("A_att", 13, global_step=50),
        write_lock_path=write_lock,
    )

    payload = torch.load(target, map_location="cpu", weights_only=False)
    assert payload["training_state"]["global_step"] == 50
    assert save_paths[0].parent == staging_dir
    assert list(staging_dir.iterdir()) == []
    assert write_lock.is_file()
    assert not target.with_name(f".{target.name}.tmp").exists()


def test_failed_shared_publish_keeps_the_previous_checkpoint(tmp_path, monkeypatch):
    import polyfid.training.state as state_module

    staging_dir = tmp_path / "node-local"
    staging_dir.mkdir()
    target = tmp_path / "shared" / "last.pt"
    target.parent.mkdir()
    target.write_bytes(b"previous checkpoint")

    def fail_after_partial_copy(source, destination):
        Path(destination).write_bytes(b"partial checkpoint")
        raise OSError("simulated shared write failure")

    monkeypatch.setenv("SLURM_TMPDIR", str(staging_dir))
    monkeypatch.setattr(state_module.shutil, "copyfile", fail_after_partial_copy)
    component = SimpleNamespace(state_dict=lambda: {})
    with pytest.raises(OSError, match="simulated shared write failure"):
        save_training_checkpoint(
            target,
            model=component,
            optimizer=component,
            state=TrainingState("A_att", 13),
            write_lock_path=target.parent / ".checkpoint-write.lock",
        )

    assert target.read_bytes() == b"previous checkpoint"
    assert list(staging_dir.iterdir()) == []
    assert not target.with_name(f".{target.name}.tmp").exists()


def test_common_training_recipe_and_learning_rate_search_spaces():
    recipe = PretrainingRecipe()
    assert recipe.effective_anchor_batch_size == (
        recipe.micro_batch_size * recipe.gradient_accumulation
    )
    assert recipe.epochs == 3
    assert learning_rate_candidates("warm") == (1e-5, 3e-5, 1e-4)
    assert learning_rate_candidates("scratch") == (3e-5, 1e-4, 3e-4)


def test_projection_and_objective_when_torch_is_available():
    torch = pytest.importorskip("torch")
    from polyfid.models.projection import RelationProjectionHead
    from polyfid.training.objectives import paper104_objective

    projection = RelationProjectionHead()
    hidden = torch.randn(3, 768)
    projected = projection(hidden)
    assert projected.shape == (3, 256)
    assert torch.allclose(projected.norm(dim=-1), torch.ones(3), atol=1e-5)

    logits = torch.randn(2, 4, 17, requires_grad=True)
    labels = torch.tensor([[-100, 2, -100, -100], [-100, 3, -100, -100]])
    loss = paper104_objective(
        logits,
        labels,
        anchor_projections=projected,
        positive_projections=projected.clone(),
        hard_negative_projections=torch.roll(projected, 1, 0),
    )
    assert torch.allclose(loss.total, loss.mlm + 0.1 * loss.relation)
    loss.total.backward()

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
import h5py
import numpy as np
from torch.optim import SGD
from torch.optim.lr_scheduler import LinearLR

from model3_o2_dynamic.config import load_config
from model3_o2_dynamic.contracts import validate_contract
from model3_o2_dynamic.models.model3_o2_wam import Model3O2DynamicWAM, Model3O2WAM
from model3_o2_dynamic.models.response_adapter import ResponseAdapterBank
from model3_o2_dynamic.pipeline import STAGES, build_pipeline_plan
from model3_o2_dynamic.stage1 import collect as stage1_collect
from model3_o2_dynamic.stage1 import carrier as stage1_carrier
from model3_o2_dynamic.stage1 import teacher as stage1_teacher
from model3_o2_dynamic.stage1.cache import CurrentHiddenReader, current_hidden
from model3_o2_dynamic.stage1.common import motion_labels, target_pose
from model3_o2_dynamic.stage1.contracts import Stage1DataConfig, Stage1ContractError, task_subset_for_suite
from model3_o2_dynamic.stage1.selection import (
    _require_test_permission,
    coverage_summary,
    normalize_splits,
    select_candidates_for_split,
)
from model3_o2_dynamic.stage1.preflight import validate_demonstration_file
from model3_o2_dynamic.stage1.train import ResponseWarmupModel, response_loss
from model3_o2_dynamic.stage1.export import save_adapter_export
from model3_o2_dynamic.trainer import CoupledAdapterScheduler
from model3_o2_dynamic.prepare_stage2 import prepare_stage2_config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_response_adapter_is_exact_identity_at_initialization() -> None:
    torch.manual_seed(11)
    adapters = ResponseAdapterBank()
    hidden = torch.randn(2, 392, 1536)
    assert torch.equal(adapters.apply(8, hidden), hidden)
    assert torch.count_nonzero(adapters.residual(16, hidden)) == 0


def test_response_adapter_apply_preserves_module_traversal_protocol() -> None:
    adapters = ResponseAdapterBank()
    visited: list[torch.nn.Module] = []

    result = adapters.apply(lambda module: visited.append(module))

    assert result is adapters
    assert any(isinstance(module, torch.nn.LayerNorm) for module in visited)
    assert any(isinstance(module, torch.nn.Linear) for module in visited)


def test_stage1_response_loss_stops_at_h_and_reaches_only_a_q() -> None:
    torch.manual_seed(12)
    model = ResponseWarmupModel(predictor_width=16)
    hidden = torch.randn(1, 3, 392, 1536, requires_grad=True)
    actions = torch.randn(1, 4, 8, 7)
    targets = torch.randn(1, 3, 3, 4, 256)
    loss, metrics = response_loss(
        model,
        current_hidden=hidden,
        actions=actions,
        targets=targets,
        teacher_timestep=250,
        beta_anchor=1.0,
    )
    loss.backward()

    assert hidden.grad is None
    assert metrics["loss_response"].isfinite()
    assert all(parameter.grad is not None for parameter in model.parameters())
    # With Q restricted to r=A(sg(h)), the zero-initialized up projection is
    # the first deployed tensor that must receive response supervision.
    for layer in (8, 16, 24):
        up_bias_grad = model.adapters.adapter(layer).up.bias.grad
        assert up_bias_grad is not None
        assert torch.count_nonzero(up_bias_grad) > 0


def test_stage1_predictor_receives_only_adapter_residual_not_identity_skip() -> None:
    torch.manual_seed(121)
    model = ResponseWarmupModel(predictor_width=16)
    with torch.no_grad():
        model.adapters.adapter(8).up.bias.fill_(0.25)
    hidden = torch.randn(2, 3, 392, 1536)
    actions = torch.randn(2, 4, 8, 7)
    targets = torch.randn(2, 3, 3, 4, 256)
    observed: list[torch.Tensor] = []
    hook = model.predictors["8"].register_forward_pre_hook(
        lambda _module, args: observed.append(args[0].detach().clone())
    )
    try:
        response_loss(
            model,
            current_hidden=hidden,
            actions=actions,
            targets=targets,
            teacher_timestep=250,
            beta_anchor=1.0,
        )
    finally:
        hook.remove()

    expected_residual = model.adapters.residual(8, hidden[:, 0].detach())
    identity_skip = hidden[:, 0] + expected_residual
    assert len(observed) == 3
    assert torch.allclose(observed[0], expected_residual)
    assert not torch.allclose(observed[0], identity_skip)


def test_dynamic_adapter_replaces_adapted_before_query_memory_projection(monkeypatch) -> None:
    torch.manual_seed(13)
    model = Model3O2DynamicWAM.__new__(Model3O2DynamicWAM)
    torch.nn.Module.__init__(model)
    model.response_adapters = ResponseAdapterBank()
    model._response_adapters_enabled = True
    model._response_adapters_trainable = True
    with torch.no_grad():
        # A nonzero residual makes the position check observable without a Wan
        # model or any routed-query/Action-DiT path.
        model.response_adapters.adapter(8).up.bias.fill_(0.25)
    hidden = torch.randn(1, 392, 1536)
    backbone = hidden - 0.5
    states = [
        {"layer_idx": layer, "backbone": backbone.clone(), "adapted": hidden.clone(), "delta": hidden - backbone}
        for layer in (8, 16, 24)
    ]
    monkeypatch.setattr(
        Model3O2WAM,
        "_build_multilayer_action_fusion_inputs",
        lambda self, video_token_slice=None: states,
    )

    output = model._build_multilayer_action_fusion_inputs()
    assert torch.allclose(output[0]["adapted"], hidden + 0.25)
    assert torch.allclose(output[0]["delta"], hidden + 0.25 - backbone)
    assert torch.equal(states[0]["adapted"], hidden)
    with model.response_adapters_disabled():
        disabled = model._build_multilayer_action_fusion_inputs()
    assert disabled is states


def test_sharded_current_hidden_reader_is_bounded_and_exact(tmp_path: Path) -> None:
    first = torch.full((1, 3, 392, 1536), 1.0, dtype=torch.bfloat16)
    second = torch.full((1, 3, 392, 1536), 2.0, dtype=torch.bfloat16)
    first_path = tmp_path / "first.pt"
    second_path = tmp_path / "second.pt"
    torch.save({"current_hidden": first}, first_path)
    torch.save({"current_hidden": second}, second_path)
    payload = {
        "current_hidden": None,
        "current_hidden_shards": [
            {"path": str(first_path), "sha256": _sha256(first_path), "shape": [1, 3, 392, 1536]},
            {"path": str(second_path), "sha256": _sha256(second_path), "shape": [1, 3, 392, 1536]},
        ],
        "current_hidden_index": torch.tensor([[0, 0], [1, 0]], dtype=torch.int64),
    }
    reader = CurrentHiddenReader(payload, max_cached_shards=1)
    payload["_current_hidden_reader"] = reader
    values = current_hidden(payload, torch.tensor([1, 0], dtype=torch.long))

    assert values.dtype == torch.bfloat16
    assert float(values[0, 0, 0, 0]) == 2.0
    assert float(values[1, 0, 0, 0]) == 1.0
    assert len(reader._loaded) == 1


def test_test_split_is_explicitly_sealed_before_adapter_export(tmp_path: Path) -> None:
    assert normalize_splits(("validation", "train")) == ("train", "validation")
    with pytest.raises(Stage1ContractError, match="sealed"):
        _require_test_permission(("test",), None)
    export = tmp_path / "adapter.pt"
    export.touch()
    _require_test_permission(("test",), export)


def test_motion_selection_is_deterministic_and_meets_quotas() -> None:
    config = Stage1DataConfig()
    candidates: dict[int, dict[int, list[dict[str, object]]]] = {0: {}}
    labels = {
        "object_motion": True,
        "robot_motion": True,
        "camera_motion": True,
        "contact_interaction": True,
        "static_control": False,
    }
    for progress_bin in range(config.states_per_demo):
        candidates[0][progress_bin] = [
            {"source_index": 99, "motion_labels": labels},
            {"source_index": progress_bin, "motion_labels": labels},
        ]
    selected = select_candidates_for_split(candidates, config=config)
    assert [row["source_index"] for row in selected] == list(range(10))
    assert coverage_summary(selected, config)["passed"] is True


def test_motion_selection_repairs_same_bin_coverage_without_lowering_quotas() -> None:
    config = Stage1DataConfig()
    candidates: dict[int, dict[int, list[dict[str, object]]]] = {0: {}}
    non_object = {
        "object_motion": False,
        "robot_motion": True,
        "camera_motion": True,
        "contact_interaction": True,
        "static_control": False,
    }
    object_only = {
        "object_motion": True,
        "robot_motion": False,
        "camera_motion": False,
        "contact_interaction": False,
        "static_control": False,
    }
    for progress_bin in range(config.states_per_demo):
        candidates[0][progress_bin] = [
            {"source_index": 100 + progress_bin, "motion_labels": non_object},
        ]
        if progress_bin < 3:
            candidates[0][progress_bin].append(
                {"source_index": progress_bin, "motion_labels": object_only}
            )

    selected = select_candidates_for_split(candidates, config=config)

    assert [row["source_index"] for row in selected[:3]] == [0, 1, 2]
    assert coverage_summary(selected, config)["passed"] is True


def test_task_subset_preserves_registered_order_and_rejects_unknown_tasks() -> None:
    selected = task_subset_for_suite(
        "libero_10",
        (
            "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
            "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        ),
    )

    assert selected == (
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    )
    with pytest.raises(Stage1ContractError, match="unknown Dynamic Stage-1 tasks"):
        task_subset_for_suite("libero_10", ("not_a_libero_task",))


def test_branch_restore_reseeds_and_restores_static_model_state() -> None:
    class FakeModel:
        pass

    class FakeData:
        def __init__(self) -> None:
            self.ctrl = np.array([3.0, -2.0])
            self.qacc_warmstart = np.array([4.0, 5.0])
            self.qfrc_applied = np.array([6.0, 7.0])
            self.xfrc_applied = np.array([[8.0, 9.0]])

    class FakeSim:
        def __init__(self) -> None:
            self.model = FakeModel()
            self.data = FakeData()
            self.state = np.array([0.0, 0.0])
            for index, name in enumerate(stage1_collect.STATIC_MODEL_STATE_ARRAYS):
                setattr(self.model, name, np.array([float(index)]))

        def set_state_from_flattened(self, state) -> None:
            self.state = np.asarray(state, dtype=np.float64).copy()

        def forward(self) -> None:
            pass

    class FakeController:
        def __init__(self) -> None:
            self.initial_joint = np.array([-1.0, -1.0])
            self.updated_with: np.ndarray | None = None

        def update_initial_joints(self, joints) -> None:
            self.initial_joint = np.asarray(joints, dtype=np.float64).copy()
            self.updated_with = self.initial_joint.copy()

    class FakeRobot:
        def __init__(self, sim: FakeSim) -> None:
            self.sim = sim
            self.controller = FakeController()

        @property
        def _joint_positions(self) -> np.ndarray:
            return self.sim.state.copy()

    class FakeRawEnv:
        def __init__(self) -> None:
            self.sim = FakeSim()
            self.control_timestep = 1.0
            self.model_timestep = 1.0
            self.robots = [FakeRobot(self.sim)]
            self.success_checks = 0
            self.post_processes = 0

        def _pre_action(self, *args, **kwargs) -> None:
            pass

        def _post_action(self, *args, **kwargs):
            return 0.0, False, {}

        def _update_observables(self, *, force: bool) -> None:
            assert force is True

        def _get_observations(self):
            return {"synthetic": self.sim.state.copy()}

        def _check_success(self) -> None:
            self.success_checks += 1

        def _post_process(self) -> None:
            self.post_processes += 1

    class FakeEnv:
        def __init__(self) -> None:
            self.seeds: list[int] = []
            self.env = FakeRawEnv()

        def seed(self, value: int) -> None:
            self.seeds.append(value)

        def reset(self) -> None:
            for name in stage1_collect.STATIC_MODEL_STATE_ARRAYS:
                getattr(self.env.sim.model, name)[...] = -1.0

        def set_state(self, state) -> None:
            self.env.sim.set_state_from_flattened(state)

        def get_sim_state(self):
            return self.env.sim.state.copy()

    env = FakeEnv()
    model_state = stage1_collect._capture_static_model_state(env)
    state = np.array([1.0, 2.0])
    result = stage1_collect._restore_branch(env, state, Stage1DataConfig(), model_state)

    assert env.seeds == [42]
    assert result["restore_max_abs"] == 0.0
    assert np.array_equal(result["obs"]["synthetic"], state)
    controller = env.env.robots[0].controller
    assert np.array_equal(controller.updated_with, state)
    assert np.array_equal(controller.initial_joint, state)
    for name in ("ctrl", "qacc_warmstart", "qfrc_applied", "xfrc_applied"):
        assert np.count_nonzero(getattr(env.env.sim.data, name)) == 0
    assert env.env.success_checks == 1
    assert env.env.post_processes == 1
    for name, value in model_state.items():
        assert np.array_equal(getattr(env.env.sim.model, name), value)


def test_collection_resume_requires_exact_current_provenance(tmp_path: Path) -> None:
    config = Stage1DataConfig()
    selection_path = tmp_path / "selection.pt"
    static_model_path = tmp_path / "static_model_state.pt"
    payload = {
        "schema_version": 1,
        "track_id": "model3_o2_dynamic",
        "artifact_kind": "stage1_branch_collection_shard",
        "task": "synthetic_task",
        "split": "train",
        "demo_id": 3,
        "selection_path": str(selection_path),
        "selection_sha256": "selection-sha",
        "source_restore_contract": "seeded_static_osc_source_sync_v1",
        "renderer_context_contract": stage1_collect.COLLECTION_RENDERER_CONTEXT_CONTRACT,
        "static_model_state_path": str(static_model_path),
        "static_model_state_sha256": "static-sha",
        "data_contract": config.as_dict(),
    }
    kwargs = {
        "task": "synthetic_task",
        "split": "train",
        "demo_id": 3,
        "selection_path": selection_path,
        "selection_sha256": "selection-sha",
        "static_model_path": static_model_path,
        "static_model_state_sha256": "static-sha",
        "config": config,
    }

    assert stage1_collect._matches_current_collection_provenance(payload, **kwargs)
    stale = {**payload, "selection_sha256": "old-selection-sha"}
    assert not stage1_collect._matches_current_collection_provenance(stale, **kwargs)
    legacy = {key: value for key, value in payload.items() if key != "source_restore_contract"}
    assert not stage1_collect._matches_current_collection_provenance(legacy, **kwargs)


def test_carrier_and_teacher_resume_require_exact_current_collection_provenance(tmp_path: Path) -> None:
    collection_path = tmp_path / "collection.pt"
    collection_path.write_bytes(b"collection-v1")
    current_path = collection_path.resolve()
    current_sha256 = _sha256(collection_path)
    payload = {
        "collection_path": str(current_path),
        "collection_sha256": current_sha256,
    }

    for downstream in (stage1_carrier, stage1_teacher):
        assert downstream._matches_current_collection_provenance(
            payload,
            collection_path=current_path,
            collection_sha256=current_sha256,
        )
        assert not downstream._matches_current_collection_provenance(
            {**payload, "collection_path": str(tmp_path / "other_collection.pt")},
            collection_path=current_path,
            collection_sha256=current_sha256,
        )

    collection_path.write_bytes(b"collection-v2")
    rebuilt_collection_sha256 = _sha256(collection_path)
    assert rebuilt_collection_sha256 != current_sha256
    for downstream in (stage1_carrier, stage1_teacher):
        assert not downstream._matches_current_collection_provenance(
            payload,
            collection_path=current_path,
            collection_sha256=rebuilt_collection_sha256,
        )
        assert not downstream._matches_current_collection_provenance(
            {"collection_path": str(current_path)},
            collection_path=current_path,
            collection_sha256=rebuilt_collection_sha256,
        )


def test_collection_env_uses_hard_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    import libero.libero.envs as libero_envs

    observed: dict[str, object] = {}

    class FakeOffscreenEnv:
        def __init__(self, **kwargs) -> None:
            observed.update(kwargs)
            self.seed_value: int | None = None

        def seed(self, value: int) -> None:
            self.seed_value = value

    monkeypatch.setattr(libero_envs, "OffScreenRenderEnv", FakeOffscreenEnv)
    env = stage1_collect._make_env(bddl_path=Path("/tmp/task.bddl"), resolution=256)

    assert observed["hard_reset"] is True
    assert env.seed_value == 42


def test_collection_repeat_error_includes_source_and_pixel_context(monkeypatch: pytest.MonkeyPatch) -> None:
    config = Stage1DataConfig()
    base = {
        "restore_max_abs": 0.0,
        "current_images": np.zeros((2, 1, 1, 3), dtype=np.uint8),
        "current_proprio": np.zeros(8, dtype=np.float32),
        "render_tick_images": np.zeros((32, 2, 1, 1, 3), dtype=np.uint8),
        "stage_images": np.zeros((4, 2, 1, 1, 3), dtype=np.uint8),
        "render_capture_substeps": [],
        "terminal_state": np.zeros(3, dtype=np.float64),
        "target_position_key": "target_pos",
        "target_position": np.zeros(3, dtype=np.float64),
        "done_steps": [],
    }
    def clone(render_value: int) -> dict[str, object]:
        output = {key: (value.copy() if isinstance(value, np.ndarray) else value) for key, value in base.items()}
        output["render_tick_images"][0, 0, 0, 0, 0] = render_value
        return output

    outputs = [clone(0) for _ in range(4)] + [clone(1), clone(2), clone(3)]

    monkeypatch.setattr(stage1_collect, "_run_branch", lambda *args, **kwargs: outputs.pop(0))
    selection_record = {
        "sample_id": 1232,
        "task": "synthetic_task",
        "task_position": 2,
        "instruction": "synthetic instruction",
        "demo_id": 23,
        "split": "train",
        "source_index": 55,
        "episode_progress": 0.5,
        "progress_bin": 2,
        "motion_labels": {},
        "target_position_key": "target_pos",
        "source_state": np.zeros(3, dtype=np.float64),
        "expert_actions": np.zeros((8, 7), dtype=np.float32),
    }

    with pytest.raises(
        RuntimeError,
        match=r"after 4 attempts.*sample_id=1232.*render_max_abs=3.*render_changed=1",
    ):
        stage1_collect._build_record(object(), selection_record, config, {})


def test_collection_repeat_uses_first_exact_expert_consensus(monkeypatch: pytest.MonkeyPatch) -> None:
    config = Stage1DataConfig()
    base = {
        "restore_max_abs": 0.0,
        "current_images": np.zeros((2, 1, 1, 3), dtype=np.uint8),
        "current_proprio": np.zeros(8, dtype=np.float32),
        "render_tick_images": np.zeros((32, 2, 1, 1, 3), dtype=np.uint8),
        "stage_images": np.zeros((4, 2, 1, 1, 3), dtype=np.uint8),
        "render_capture_substeps": [],
        "terminal_state": np.zeros(3, dtype=np.float64),
        "target_position_key": "target_pos",
        "target_position": np.zeros(3, dtype=np.float64),
        "done_steps": [],
    }

    def clone(render_value: int) -> dict[str, object]:
        output = {key: (value.copy() if isinstance(value, np.ndarray) else value) for key, value in base.items()}
        output["render_tick_images"][0, 0, 0, 0, 0] = render_value
        return output

    outputs = [clone(0) for _ in range(4)] + [clone(1), clone(0)]
    monkeypatch.setattr(stage1_collect, "_run_branch", lambda *args, **kwargs: outputs.pop(0))
    selection_record = {
        "sample_id": 1232,
        "task": "synthetic_task",
        "task_position": 2,
        "instruction": "synthetic instruction",
        "demo_id": 23,
        "split": "train",
        "source_index": 55,
        "episode_progress": 0.5,
        "progress_bin": 2,
        "motion_labels": {},
        "target_position_key": "target_pos",
        "source_state": np.zeros(3, dtype=np.float64),
        "expert_actions": np.zeros((8, 7), dtype=np.float32),
    }

    record = stage1_collect._build_record(object(), selection_record, config, {})

    assert record["expert_repeat_attempt_count"] == 3
    assert record["repeat_render_images_exact"] is True
    assert int(record["render_tick_images"][0, 0, 0, 0, 0, 0]) == 0


def test_target_pose_uses_exact_goal_object_keys_and_stable_primary() -> None:
    observation = {
        "first_target_pos": np.array([1.0, 2.0, 3.0]),
        "first_target_to_robot0_eef_pos": np.array([9.0, 9.0, 9.0]),
        "second_target_pos": np.array([4.0, 5.0, 6.0]),
        "second_target_quat": np.array([0.0, 0.0, 0.0, 1.0]),
    }
    position, quaternion, key = target_pose(observation, ("fixed_fixture", "first_target", "second_target"))

    assert key == "first_target_pos"
    assert np.array_equal(position, np.array([1.0, 2.0, 3.0]))
    assert quaternion is None


def test_motion_labels_detect_any_goal_target_object_motion() -> None:
    config = Stage1DataConfig()
    identity_quaternion = np.array([0.0, 0.0, 0.0, 1.0])
    common = {
        "eef_position": np.zeros(3),
        "eef_quaternion": identity_quaternion,
        "wrist_position": np.zeros(3),
        "wrist_rotation": np.eye(3),
        "target_contact_signature": frozenset(),
    }
    initial = {
        **common,
        "target_poses": (
            (np.zeros(3), identity_quaternion, "first_target_pos"),
            (np.zeros(3), identity_quaternion, "second_target_pos"),
        ),
    }
    moved = {
        **common,
        "target_poses": (
            (np.zeros(3), identity_quaternion, "first_target_pos"),
            (np.array([config.object_motion_m + 1.0e-3, 0.0, 0.0]), identity_quaternion, "second_target_pos"),
        ),
    }

    labels = motion_labels([initial, moved], config)

    assert labels["object_motion"] is True


def test_stage1_demonstration_preflight_reads_only_file_metadata(tmp_path: Path) -> None:
    source = tmp_path / "complete.hdf5"
    with h5py.File(source, "w") as handle:
        data = handle.create_group("data")
        data.attrs["bddl_file_name"] = "bddl_files/libero_10/task.bddl"
        for demo_id in range(50):
            data.create_group(f"demo_{demo_id}")
    result = validate_demonstration_file(
        source,
        task="synthetic_long_task",
        config=Stage1DataConfig(),
    )
    assert result["demo_count"] == 50
    assert result["payload_read"] is False

    incomplete = tmp_path / "incomplete.hdf5"
    with h5py.File(incomplete, "w") as handle:
        data = handle.create_group("data")
        data.attrs["bddl_file_name"] = "bddl_files/libero_10/task.bddl"
        for demo_id in range(49):
            data.create_group(f"demo_{demo_id}")
    with pytest.raises(Stage1ContractError, match="demo groups are incomplete"):
        validate_demonstration_file(
            incomplete,
            task="synthetic_long_task",
            config=Stage1DataConfig(),
        )


def test_adapter_scheduler_keeps_exact_lr_ratio() -> None:
    base = torch.nn.Parameter(torch.tensor(1.0))
    adapter = torch.nn.Parameter(torch.tensor(1.0))
    gate = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = SGD(
        [
            {"params": [base], "lr": 1.0e-3, "group_name": "o2_peft"},
            {"params": [adapter], "lr": 1.0e-4, "group_name": "response_adapters"},
            {"params": [gate], "lr": 1.0e-3, "group_name": "o2_layer_readout_gate"},
        ]
    )
    inherited = LinearLR(optimizer, start_factor=0.5, end_factor=1.0, total_iters=2)
    scheduler = CoupledAdapterScheduler(
        inherited,
        optimizer,
        base_group_index=0,
        adapter_group_index=1,
        scale=0.1,
        additional_group_scales={2: 1.0},
    )
    optimizer.step()
    scheduler.step()
    assert optimizer.param_groups[1]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"] * 0.1)
    assert optimizer.param_groups[2]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"])


def test_pipeline_dry_plan_defers_test_until_after_fixed_export() -> None:
    config_path = Path("model3_o2_dynamic/configs/libero_object.json")
    plan = build_pipeline_plan(
        config_path=config_path,
        run_id="dynamic-unit-dry-plan",
        stage1_device="cuda:0",
    )
    assert plan["execute_required"] is True
    assert plan["stages"].index("stage1_response_warmup") < plan["stages"].index("select_test")
    assert tuple(plan["stages"]) == STAGES
    assert load_config(config_path).base.training.gpu_ids == (0, 1, 2, 3)


@pytest.mark.parametrize(
    ("config_path", "gate_boundary"),
    [
        ("model3_o2_dynamic/configs/libero_long_fast.json", (0, 1)),
        ("model3_o2_dynamic/configs/libero_object.json", (30000, 30001)),
    ],
)
def test_global_dynamic_contracts_keep_residual_only_and_suite_gate_boundaries(
    config_path: str,
    gate_boundary: tuple[int, int],
) -> None:
    dynamic = load_config(config_path)
    result = validate_contract(dynamic, check_paths=False)
    assert result["passed"] is True
    assert (dynamic.schedule.freeze_through_step, dynamic.schedule.first_adapter_update_step) == (5000, 5001)
    assert (
        dynamic.schedule.gate_freeze_through_step,
        dynamic.schedule.first_gate_update_step,
    ) == gate_boundary
    assert dynamic.schedule.gate_lr_scale == 1.0


def test_adapter_only_export_materializes_a_strict_stage2_config(tmp_path: Path) -> None:
    template = Path("model3_o2_dynamic/configs/libero_object.json").resolve()
    dynamic = load_config(template)
    export = tmp_path / "stage1_adapter.pt"
    save_adapter_export(
        export,
        ResponseAdapterBank(),
        source_identity={
            "model3_warmstart_sha256": dynamic.initialization.model3_checkpoint_sha256,
            "original_o2_tensor_sha256": "a" * 64,
        },
        normalization_identity={"normalization_fit_split": "train"},
    )
    output = tmp_path / "stage2" / "config.json"
    result = prepare_stage2_config(
        template_config=template,
        adapter_export=export,
        output_config=output,
    )

    materialized = load_config(output)
    assert output.is_file()
    assert result["stage2_config"] == str(output)
    assert materialized.initialization.stage_role == "stage2_joint"
    assert materialized.initialization.response_adapter_export == export.resolve()


def test_stage2_materialization_rejects_export_without_residual_only_provenance(tmp_path: Path) -> None:
    template = Path("model3_o2_dynamic/configs/libero_long_fast.json").resolve()
    dynamic = load_config(template)
    export = tmp_path / "legacy_identity_skip_export.pt"
    save_adapter_export(
        export,
        ResponseAdapterBank(),
        source_identity={
            "model3_warmstart_sha256": dynamic.initialization.model3_checkpoint_sha256,
            "original_o2_tensor_sha256": "b" * 64,
        },
        normalization_identity={"normalization_fit_split": "train"},
    )
    payload = torch.load(export, map_location="cpu", weights_only=False)
    payload.pop("stage1_predictor_input")
    torch.save(payload, export)

    with pytest.raises(ValueError, match="stage1_predictor_input"):
        prepare_stage2_config(
            template_config=template,
            adapter_export=export,
            output_config=tmp_path / "stage2" / "config.json",
        )

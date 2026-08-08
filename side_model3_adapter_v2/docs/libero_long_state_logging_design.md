# LIBERO Long Simulator-State Logging Design

Date: 2026-08-08

## Scope

This is an evaluation-only diagnostic design. It does not change training,
policy inference, action generation, success metrics, or the official video
retention rule. Its purpose is to make a later failure analyzer distinguish
visible but ambiguous events such as a missed grasp, a failed placement, an
unclosed drawer, and an unrecovered disturbance.

## Verified Current Recorder

The current single-process evaluator creates a recorder immediately after
`set_init_state`, records the initial state, then records one post-step state
after every settling and policy action. It writes the terminal record before
returning from the episode. This is a valid transition alignment:

```text
state[0] -- action[0] --> state[1] -- action[1] --> ... --> state[T]
```

The implementation is in
`model3/third_party/light_wam/experiments/libero/eval_libero_single.py` and is
enabled by `EVALUATION.record_simulator_state`. Episode files use the stable
identity:

```text
<eval_output>/libero_10/simulator_states/task_<task>/episode_<trial>.json
```

The JSON recorder currently persists:

- simulator time, action, settling versus policy phase, policy query index and
  action-chunk index;
- end-effector position/quaternion and gripper position;
- pose of every movable object, fixture, and task site;
- every named MuJoCo joint position;
- every benchmark goal predicate evaluated by LIBERO itself;
- final success and task/trial provenance.

This is enough to verify goal transitions, object/fixture motion, drawer or
microwave joint movement, stove predicate state, and whether a policy began a
new chunk after a failed predicate.

## First State Analyzer

The first task-aware analyzer consumes this JSON directly. It reads only policy
actions for action events, uses the last settling state as the motion baseline,
and maps native predicate truth through the Long task registry. The registry
contains explicit stage requirements and the drawer/microwave placement-before-
closure diagnostic prerequisites.

Its `grasp_failed` and `grasp_alignment_failure` labels are deliberately
kinematic trajectory proxies: close action, target proximity, target/EEF
co-motion, and subsequent retreat. They are useful for triage but are not
claimed as contact or authoritative grasp truth. Traces without enough evidence
are sent to manual review.

## Signals Still Needed For Contact-Level Attribution

The current JSON schema does not yet retain raw MuJoCo `qpos/qvel`, robot joint
velocity, gripper velocity, fingerpad grasp state, or contact pairs. Add these
only in a follow-up evaluation-only change, behind the existing recorder flag:

| Signal | Capture source | Why it matters |
| --- | --- | --- |
| `sim_qpos`, `sim_qvel` | `env.sim.data.qpos/qvel` | replayable mechanical state and drift magnitude |
| `robot_joint_pos`, `robot_joint_vel`, `gripper_qvel` | observation keys or `env.robots[0]` / MuJoCo joint addresses | distinguish arm motion from a stationary failed approach |
| `grasp_state[target]` | robosuite `task_env._check_grasp(task_env.robots[0].gripper, object)` | authoritative two-finger grasp, rather than image inference |
| `contact_pairs` | `env.sim.data.contact[:env.sim.data.ncon]` translated with `geom_id2name` | identify table/basket/drawer collisions and disturbances |
| `goal_predicate_state` | existing `task_env._eval_predicate` | retain as the benchmark-defined truth source |

Store numeric time-series arrays in one compressed `.npz` per episode and keep
the current JSON as a compact schema/provenance sidecar. The ndarray file should
contain fixed-shape state histories; JSON should contain entity names, goal
predicate strings, array shapes, task/trial identity and final success. This
avoids repeated structural JSON while keeping the trace inspectable.

## Minimal Recorder Extension

The existing recorder insertion points are already correct:

1. After `set_init_state`, write `state[0]` with `action=null`.
2. Before each `env.step`, retain the exact action and metadata that will be
   executed.
3. After each `env.step`, capture `state[t+1]`, predicate truth and contacts.
4. Write after terminal `done`, so the terminal state is retained.

The extension should use the existing `LiberoSimulatorStateRecorder`, not add a
second evaluation loop or render pass. It must not derive stages from BDDL list
order: the Long stage registry remains the source for any physical-progress
interpretation, while `_eval_predicate` supplies only benchmark predicate truth.

## Deterministic Downstream Rules

`side_model3_adapter/scripts/analyze_libero_state_failures.py` implements the
first ruleset. It evaluates only `policy` records and writes the exact triggered
events, configured thresholds, and source state path. The implemented rules are:

- **grasp failed:** a close action occurs near an unsatisfied target, then the
  end effector retreats while the target does not co-move;
- **grasp alignment failure:** at least three independent close attempts target
  the same object without verified carry;
- **placement failure:** verified target/end-effector co-motion is followed by
  release or departure while the native `In` or `On` predicate remains false;
- **premature drawer/microwave closure:** Task $3$/$9$ reaches the native
  closure predicate while its placement prerequisite remains false; this emits
  both `placement_failure` and `outcome_awareness_failure`;
- **lost stove invariant:** Task $8$ starts with `Turnon(flat_stove_1)` and a
  later native false predicate emits `mechanism_interaction_failure`;
- **environment disturbance:** a registry-listed movable distractor remains
  displaced relative to the final settling state across the configured number
  of policy records;
- **recovery failure:** repeated failed grasp alignment with a persistent
  distractor disturbance leaves at least two later policy-query opportunities
  without recovery of that target predicate.

The first two labels explicitly carry `evidence_basis=kinematic_trajectory_proxy`.
The optional recorder extension can upgrade grasp and disturbance attribution
from proxies to direct contact evidence.

The final two rules need the exact Long task registry in
`configs/libero_long_stage_rules.json`; task $3$ and task $9$, for example,
need object placement before closure as a physical-progress hint, while the
benchmark still evaluates their goal predicates as an unordered conjunction.

## Evidence Sources

- Evaluator control loop and recorder placement:
  `model3/third_party/light_wam/experiments/libero/eval_libero_single.py`.
- Current JSON recorder:
  `model3/libero_simulator_state.py`.
- Env-level MuJoCo state access:
  `/data/users/lizhe/LIBERO/libero/libero/envs/env_wrapper.py`.
- Native success and predicate evaluation:
  `/data/users/lizhe/LIBERO/libero/libero/envs/problems/libero_kitchen_tabletop_manipulation.py`.
- Native object pose API:
  `/data/users/lizhe/LIBERO/libero/libero/envs/object_states/base_object_states.py`.
- Native two-finger grasp semantics:
  `robosuite/environments/manipulation/manipulation_env.py` in the eval conda
  environment.

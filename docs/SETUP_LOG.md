# Setup log

Every setup decision, version, workaround and deviation (CLAUDE.md rule 6). Append; don't rewrite history.

## 2026-10-07: repo scaffold (written on macOS, not yet run on the Windows/AC machine)

### Upstream
- `third_party/assetto_corsa_gym` is a submodule pinned at `ef0fa502bdd257fc2affad9361c02b3dbdfe1632` (2025-08-02). Not edited in place. No `/patches` needed so far.
- Upstream code was read before writing ours. Findings that shaped the design:
  - The env accepts `ac_configs_path` (`AssettoCorsaEnv/assettoCorsa.py:make_ac_env`). We point it at `generated/AssettoCorsaConfigs`, an overlay whose `tracks/config.yaml` sets `ref_lap_file` to our centerline. That swaps the reference line without editing upstream.
  - The reference line is loaded by `ReferenceLap(ref_lap_file)`. It reads columns `pos_x,pos_y`, and the last point must precede the first (no duplicated closing point). It drives:
    - `state["gap"]`: signed distance to the line, via `gap_cpu.get_gap`
    - the reward term `use_reference_line_in_reward`: `r = speed * (1 - |gap| / 12) / 300`
    - the curvature look-ahead in the observation, indexed by `LapDist = NormalizedSplinePosition * TrackLength`

    With our overlay, upstream's `gap` therefore measures distance to the centerline.
  - The agent loop (`algorithm/discor/discor/agent.py`) calls `env.set_actions(a)`, then `env.step(None)`. It calls `env.close()` at the end of every episode and uses the returned stats dict. It reads `env._max_episode_steps` and `env.states`. Our wrapper forwards all of these, and writes its reward into `env.state` and `env.states[-1]` so upstream's parquet logs, `summary.csv` and best-reward checkpoint see it.
  - The upstream frame is `world_position_x = AC z`, `world_position_y = AC x` (`sensors_par/structures.py`), and `yaw = -physics.heading`. Because of this we don't assume any steering sign: `scripts/smoke_gym.py` measures it.
  - Actions are 3-D `[steer, pedal, brake]` in [-1, 1]; pedal and brake use -1 = released. `use_relative_actions: True` makes them deltas scaled by a per-step rate limit (`max_steer_rate` deg/s for steering).
  - Resets go through CSP: the plugin calls `ac.ext_resetCar()` (`sensors_par.py`).
  - The plugin folder is **`sensors_par`**. INSTALL.md also writes `sensor_par`, but AC loads `apps/python/<name>/<name>.py` and the file is `sensors_par.py`, so the destination must be `apps/python/sensors_par`.
  - The plugin config (`sensors_par/config.py`) has `sampling_freq = 50`, which must equal the AC FPS cap and is asserted when vJoy runs on the client. It also has `ego_sampling_freq = 25`.
  - Upstream `train.py` asserts CUDA. Ours (`acrl_indy.train`) allows CPU.
  - Upstream `requirements.txt` omits `scipy`, which `reference_lap.py` and `track.py` import. It is added in our `requirements.txt`.
- Upstream ships `indianapolis_sp.csv` (bounds plus racing line, 2611 rows), `indianapolis_sp-racing_line.csv`, and `cars/ks_mazda_miata/{steer_map,brake_map}.csv`. Steer map: steer = ±1 corresponds to ±302.4° of steering wheel.
- The occupancy grid `indianapolis_sp_0.1m.pkl` is gitignored upstream. It is downloaded by `scripts/prepare_assets.py` from HF `dasgringuen/assettoCorsaGym` (116,704,469 bytes; the URL was verified to resolve) into the upstream tracks folder. That folder ignores `*.pkl`, so the submodule stays clean.
- `indianapolis_sp` here is the **oval**, a rounded rectangle about 4.07 km long, not a road course (see `runs/track_indianapolis_sp.png`).

### Centerline (Phase 3)
- Method: both bounds are resampled to 0.5 m arc length. Each left sample is paired with the nearest right sample within ±40 m of its arc-length-proportional partner, and the midpoints are taken. The result is smoothed with a Gaussian (σ = 3 m), closed, resampled to 1 m, and rotated so point 0 is at the start of upstream's racing line, running in its direction.
- **Deviation from CLAUDE.md wording:** CLAUDE.md says to resample both bounds to equal arc length and take midpoints. Pure arc-length pairing drifts through corners because the inside bound is shorter (here 4020.6 m vs 4114.0 m), so the nearest-neighbour refinement was added.
- Result: 4067 points, 4067.1 m (AC spline `TrackLength` 4066.0 m).
  - 100% of points are inside the bound quads.
  - Minimum margin to a bound is 6.12 m.
  - Maximum off-center is 0.035 m.
  - It differs from naive row-wise midpoints by at most 0.04 m.

### Choices
- **RL algorithm:** upstream SAC (`discor.algorithm.SAC` + `discor.agent.Agent`) with our env wrapper, as CLAUDE.md prefers.
  - Tradeoff: it's tied to upstream's old `gym==0.21` API and logging. In exchange we keep upstream's tested real-time loop (actions are applied before the gradient update, then it waits for the next sim tick).
- **Baseline actions:** `use_relative_actions: False` (absolute), so pure-pursuit outputs map 1:1 to vJoy. RL keeps upstream's default `True`.
- **Baseline speed:** a constant 100 km/h target, plus a corner cap `v <= sqrt(7 m/s² / curvature)` (about 150 km/h on these corners, so it rarely binds on the oval).
- **Reward:** `1.0 * progress_m - 0.1 * lat² - 1.0 * Δsteer²`, terminate if |lat| > 4 m. Upstream's out-of-track and low-speed terminations stay on. `termination_penalty` is 0.
- `Agent.memory_size` lowered from 8M to 1M (8M would need about 8 GB RAM).
- **Lap completed** = forward progress ≥ one centerline length from the spawn. AC's `LapCount` isn't used, because the hotlap spawn starts before the line.

### Must verify on the Windows machine
- [ ] `baseline.steer_sign`: run `scripts/smoke_gym.py` and copy the printed value into `configs/indy.yml`.
- [ ] `baseline.steer_ratio` (15) and `wheelbase_m` (2.265) for `ks_mazda_miata`. They are reasonable guesses because AC car data is encrypted. If the baseline cuts or overshoots corners, tune `steer_ratio`.
- [ ] Hotlap spawn: the car should be stationary, aligned with the track, before the start line, and `send_reset_at_start` should return it there.
- [ ] Actual packet rate is about 25 Hz (`scripts/smoke_client.py`).

### Offline verification done (macOS, Python 3.9.25, gym 0.21.0, torch 2.8.0 CPU)
- The upstream requirements install cleanly with CLAUDE.md's recipe: `setuptools==65.5.0`, `"cython<3"`, `"wheel<0.40.0"`, `pip==24.0`, then `pip install -r requirements.txt`.
- `tests/test_pipeline.py` passes (8 tests) against upstream's real `AssettoCorsaEnv`, with `tests/fake_client.py` standing in for AC (a kinematic car on the real centerline, with the real occupancy grid). The tests cover:
  - the centerline being used as the reference line
  - steer-sign detection
  - the reset smoke test
  - a baseline lap: PASS, lateral RMS 0.03 m, at 99 km/h
  - the wrong steer sign failing
  - the lateral-error termination
  - the upstream SAC agent running two episodes and saving

  This proves the plumbing, not the real-car behaviour.

---

<!-- Phase 0 audit output is appended below by: python scripts/audit_env.py --log -->

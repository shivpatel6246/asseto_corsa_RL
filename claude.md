# Project: RL agent drives one lap of Indianapolis on the centerline (Assetto Corsa)

## Goal (definition of done)
A trained policy, starting from the hotlap spawn on `indianapolis_sp`, completes **one full lap** without leaving the track while tracking the **geometric centerline** (midpoint of left/right track bounds), not AC's AI racing line.

Success metrics (log these for every eval):
- Lap completed: yes/no
- Lateral error from centerline: RMS < 1.0 m, max < 3.0 m
- No out-of-track events

## Platform facts
- OS: Windows. Assetto Corsa is installed via Steam (default path `C:\Program Files (x86)\Steam\steamapps\common\assettocorsa\`; verify it, don't assume).
- AC runs in **real time**. Env steps are wall-clock bound at 25 Hz (`ego_sampling_freq`). Compute is not the bottleneck. Do not suggest cloud/Colab.
- Base environment: https://github.com/dasGringuen/assetto_corsa_gym (Gym interface, SAC baseline, AC plugin `sensor_par`, vJoy for control input, CSP needed for car resets).
- Secondary reference only (read, don't build on): https://github.com/Jurredr/ACRL

## Repo layout to create
```
/third_party/assetto_corsa_gym   # git submodule, upstream, DO NOT edit in place
/src/acrl_indy/                  # our code: centerline, reward wrapper, controllers, training entry
/configs/indy.yml                # our config overrides
/scripts/                        # smoke tests, eval, plotting
/runs/                           # outputs (gitignored)
/docs/SETUP_LOG.md               # record every setup decision, version, and deviation
```
Override upstream behavior via wrappers or subclasses in `src/`. If an upstream change is truly unavoidable, put it in a patch file under `/patches` and document it.

## Rules for Claude Code
1. **You cannot see or click inside Assetto Corsa or Content Manager.** Every GUI step is a HUMAN step. When you reach one, stop, print a numbered checklist for the human, and wait for confirmation. Do not guess that it was done.
2. Writing into `Program Files (x86)` and `Documents\Assetto Corsa` may need an elevated shell. Prefer printing exact copy commands for the human over failing silently.
3. Verify, don't assume: track folder names, car folder names, which track and car configs upstream ships, and how upstream loads the reference line. Read the code first.
4. Do not start long training until every smoke test in Phase 2 passes.
5. Set `disable_wandb: True` unless the human says otherwise. Log to CSV and TensorBoard locally.
6. Keep `docs/SETUP_LOG.md` updated: Python, torch, and CUDA versions, the GPU, and every workaround.

---

## Phase 0: Environment audit (you)
- Report the Windows version, the GPU (`nvidia-smi`), and the driver/CUDA version.
- Locate the AC install. Check that these exist:
  - `content\tracks\indianapolis_sp` (if missing, STOP and tell the human; the track may not be base content)
  - `content\cars\ks_mazda_miata` (stock car, chosen because it's slow and forgiving)
- Check whether Content Manager, CSP and vJoy are installed, as far as the filesystem can tell.
- Python: upstream pins Python 3.9.13 + torch 1.12.1 + CUDA 11.6. Create a conda env `acrl` with Python 3.9.
  - If the GPU is too new for torch 1.12/cu116 (e.g. RTX 50-series), install the newest torch that supports both Python 3.9 and the GPU, or use a CPU build. Networks are small MLPs, so CPU is acceptable. Record the choice.
  - Upstream install steps: `setuptools==65.5.0`, `"cython<3"`, `"wheel<0.40.0"`, `pip==24.0`, then `pip install -r requirements.txt`. The Visual C++ Build Tools may be needed; check for them.
- Add the upstream repo as a submodule and install its requirements.

## Phase 1: Assetto Corsa setup (HUMAN, you provide the checklist)
Based on upstream `INSTALL.md` (Windows). Print it for the human, with paths resolved to their machine:
1. Install vJoy.
2. Copy plugin folder `third_party/assetto_corsa_gym/assetto_corsa_gym/AssettoCorsaPlugin/plugins/sensor_par` (verify the exact folder name, since upstream docs spell it two ways) to `<AC>\apps\python\sensor_par`.
3. From `AssettoCorsaPlugin\windows-libs`: copy `Vjoy.ini` and `WASD.ini` to `Documents\Assetto Corsa\cfg\controllers\savedsetups`, and copy the DLL/lib folders to `<AC>\system\x64`.
4. Install Content Manager, then install Custom Shaders Patch (Settings > Custom Shaders Patch). This is required for resets.
5. In AC: enable the `sensor_par` UI module, and load the vJoy controls preset.
6. Video: framerate limit 50 FPS. Run windowed, and keep the AC window visible during runs.
7. Session: Challenge > Hotlap, track `indianapolis_sp`, car `ks_mazda_miata`. Assists: auto gearbox ON, auto clutch ON, TC/ESC/ABS OFF, damage OFF, fuel and tyre wear OFF, tyre blankets ON, ideal conditions (optimum surface, time multiplier 1x), penalties ON.
8. Debug log: `Documents\Assetto Corsa\logs\py_log.txt`

## Phase 2: Smoke tests (you write the scripts, human runs AC)
Convert upstream `test_client.ipynb` and `test_gym.ipynb` into scripts in `/scripts`:
- `smoke_client.py`: connect and print telemetry (speed, position, track progress) for 10 s.
- `smoke_gym.py`: `env.reset()`, then apply constant small throttle and zero steer for ~5 s; confirm the car moves and observations change.
- `smoke_reset.py`: drive off track on purpose and confirm the episode terminates and the reset works (this needs CSP).
- Confirm the track assets are present for `indianapolis_sp` (occupancy grid pickle, reference line, bounds). If missing, download them from the HF dataset `dasgringuen/assettoCorsaGym`, path `AssettoCorsaConfigs/tracks/*`, into the upstream tracks folder. Otherwise generate them with upstream `generate_track.ipynb` (human has AC running).
- Confirm a car config exists for `ks_mazda_miata` in `AssettoCorsaConfigs/cars`. If it doesn't, report which cars are supported and stop.

## Phase 3: Centerline
- Read how upstream loads `track bounds` and the reference line, and how `use_reference_line_in_reward` uses it.
- Build `src/acrl_indy/centerline.py`:
  - Resample the left and right bounds to equal arc length, take midpoints, smooth lightly, close the loop, and resample to ~1 m spacing.
  - Save it in the same format as upstream's reference line so it can be swapped in.
- `scripts/plot_track.py`: plot the bounds, the AC reference line, and the new centerline. Save a PNG to `/runs`. Sanity check that the centerline stays inside the bounds everywhere.

## Phase 4: Classical baseline (do this BEFORE RL)
Purpose: prove the plumbing, the action mapping and the centerline are correct, independent of learning. If this fails, RL will fail for reasons that have nothing to do with RL.
- `src/acrl_indy/pure_pursuit.py`: pure-pursuit steering on the centerline (speed-scaled lookahead), plus a PI speed controller (target ~ a modest constant speed first, e.g. 100 km/h).
- Check how upstream actions work (`use_relative_actions: True` means actions are deltas, and steering is rate-limited by `max_steer_rate`). Either disable relative actions for the baseline or integrate correctly. Document which you chose.
- `scripts/run_baseline.py`: drive one lap, then log lateral error and speed against track progress, and the lap result.
- Pass criteria = the project's success metrics. Report the results.

## Phase 5: RL (SAC)
- Use the upstream SAC (`train.py`) with our wrapper and config, or a clean SAC in `src/` that wraps the upstream env. Prefer the upstream SAC first; note the tradeoff.
- Reward (implement as a wrapper, keep it documented and configurable):
  - `+ progress along the centerline per step` (in metres)
  - `- w_lat * lateral_error^2`
  - `- w_dsteer * (Δsteer)^2` (smoothness; upstream also has `penalize_actions_diff`)
  - terminate if |lateral_error| > threshold (start at ~4 m) or out of track; keep low-speed termination
- Config in `configs/indy.yml`: track `indianapolis_sp`, car `ks_mazda_miata`, our centerline as the reference, `disable_wandb: True`, `checkpoint_freq` lowered (e.g. 20_000 steps), and `num_steps` capped for the first run (e.g. 300k = ~3.3 h of sim time at 25 Hz).
- Log the episode return, progress per episode, lateral RMS, and the best lap.
- `scripts/eval.py`: load a checkpoint, run 1 lap deterministically, and report the success metrics. Plot lateral error against progress next to the baseline.
- If there's no progress after ~100k steps, stop and diagnose (look at the reward breakdown and action histograms) rather than just running longer.

## Phase 6: Report
Write `docs/RESULTS.md`: setup versions, baseline vs RL metrics, plots, wall-clock training time, and what broke.
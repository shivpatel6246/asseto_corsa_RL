# acrl_indy: one lap of Indianapolis on the centerline

An RL agent (SAC) that drives the `ks_mazda_miata` around `indianapolis_sp` in Assetto Corsa. It starts from the hotlap spawn and tracks the **geometric centerline** (the midpoint of the track bounds), not AC's racing line. It is built on [assetto_corsa_gym](https://github.com/dasGringuen/assetto_corsa_gym), which is included as a submodule and not edited.

The spec is in `claude.md`. Decisions, versions and deviations are in `docs/SETUP_LOG.md`.

**Success:** the lap is completed, lateral error RMS < 1.0 m, max < 3.0 m, and no out-of-track events.

## Layout
```
third_party/assetto_corsa_gym   upstream submodule (env, plugin, SAC); untouched
src/acrl_indy/                  our code
  centerline.py                 build/check/project the centerline
  reward.py                     reward wrapper (progress - lat² - Δsteer², |lat| > 4 m terminates)
  pure_pursuit.py               classical baseline controller
  envs.py, config.py            upstream env against our configs overlay; config merging
  train.py                      SAC training entry (python -m acrl_indy.train)
configs/indy.yml                our overrides on top of upstream config.yml
data/                           generated centerline (committed)
scripts/                        audit, checklist, assets, smoke tests, baseline, eval, plots
generated/, runs/               overlay with the 117 MB grid, and outputs (gitignored)
```

## Getting it running (Windows machine with AC)
The phases follow `claude.md`. Don't skip ahead: training waits until every Phase 2 smoke test passes.

### Phase 0: audit and Python env
```
git clone --recurse-submodules https://github.com/shivpatel6246/asseto_corsa_RL.git
cd asseto_corsa_RL
python scripts/audit_env.py --log
```
The audit reports Windows/GPU/CUDA, AC paths, the track and car folders, CSP/CM/vJoy, and a recommended torch build. **If the track `indianapolis_sp` is missing from `content/tracks`, stop.**

Then create the env. Use the torch line from the audit; the one below is upstream's pin for RTX 30-series and older:
```
conda create -n acrl python=3.9 -y
conda activate acrl
python -m pip install setuptools==65.5.0 "cython<3" "wheel<0.40.0" pip==24.0
pip install -r third_party/assetto_corsa_gym/requirements.txt
pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
pip install -e .
```

### Phase 1: AC setup (human)
`python scripts/phase1_checklist.py` prints the steps with paths for that machine:
- vJoy
- copying the `sensors_par` plugin, presets and libs
- Content Manager and CSP
- enabling the UI module and the Vjoy preset
- 50 FPS, windowed
- Challenge > Hotlap with `indianapolis_sp` and `ks_mazda_miata`, plus the assist settings

### Phase 2: assets and smoke tests (AC running in the hotlap session)
```
python scripts/prepare_assets.py   # grid download, centerline, configs overlay
python -m pytest tests -q          # offline checks against a fake AC
python scripts/smoke_client.py     # telemetry for 10 s at ~25 Hz
python scripts/smoke_gym.py        # car moves; prints baseline.steer_sign -> put it in configs/indy.yml
python scripts/smoke_reset.py      # off-track termination and CSP reset
```

### Phase 3: centerline check
`python scripts/plot_track.py` writes `runs/track_indianapolis_sp.png`.

### Phase 4: baseline (before any RL)
`python scripts/run_baseline.py` drives one lap with pure pursuit and a PI speed controller, then reports the success metrics. Tune it with overrides, e.g. `baseline.target_speed_kmh=80 baseline.steer_ratio=16`.

### Phase 5: RL
```
python -m acrl_indy.train
tensorboard --logdir runs
python scripts/eval.py runs/<run>/model/best_reward
```
- The first run is capped at 300k steps (about 3.3 h), with checkpoints every 20k in `runs/<run>/model/checkpoints/`.
- Logs go to `runs/<run>/summary.csv`, plus TensorBoard (`summary/`, `summary_centerline/`). wandb is off.
- If the best episode progress is under 500 m after 100k steps, training stops on its own. Then run `python scripts/diagnose.py runs/<run>`.
- `eval.py` plots lateral error against progress next to the latest baseline.

# Assetto Corsa RL

SAC agent that learns to drive in Assetto Corsa. It reads car state from AC's shared memory and drives through a vJoy virtual joystick. Training is real time: the game is never paused.

```
AC shared memory ──► AssettoCorsaEnv (gymnasium) ──► SAC (stable-baselines3)
        ▲                                              │
        └──────── vJoy: steer=X, throttle=Y, brake=Z ◄─┘
```

| Path | What |
|---|---|
| `ac_rl/shared_memory.py` | ctypes structs for `acpmf_physics/graphics/static` |
| `ac_rl/controls.py` | vJoy output and the reset hotkey |
| `ac_rl/track.py` | reference line, used for track-relative observations |
| `ac_rl/env.py` | the Gymnasium env: observations, reward, termination, real-time pacing |
| `train.py` / `evaluate.py` | train and resume / watch a checkpoint |
| `scripts/` | setup check, vJoy binding helper, reference-line recorder |
| `config.yaml` | all tunables |
| `tests/` | offline tests against a fake AC (no Windows needed) |

## Setup (Windows, on the machine running AC)

1. **Python env** (3.10–3.12 recommended):
   ```
   python -m venv .venv && .venv\Scripts\activate
   pip install torch --index-url https://download.pytorch.org/whl/cu121   # if you have an NVIDIA GPU
   pip install -r requirements.txt
   python -m pytest tests          # sanity check, no AC needed
   ```
   If `pyvjoy` can't find its DLL, copy `C:\Program Files\vJoy\x64\vJoyInterface.dll` into `.venv\Lib\site-packages\pyvjoy\`.

2. **vJoy**: open *Configure vJoy*. Enable device 1 with at least the X, Y and Z axes.

3. **Bind the axes in AC** (Content Manager → Settings → Assetto Corsa → Controls, input method *Wheel*):
   - Click the Steering bind, then run `python scripts/vjoy_bind_helper.py steer` and switch to CM.
   - Do the same with `throttle` and `brake`.
   - Use `python scripts/vjoy_bind_helper.py test` to check that all three move.
   - Set steering lock / degrees to something sensible (e.g. 540°), and turn off deadzones and filtering.
   - Save it as a preset, e.g. `vjoy_rl`.

4. **Assists**: turn on the **automatic gearbox** (the agent doesn't shift). ABS and TC on is easiest to start with.

5. **Reset hotkey**: in AC/CM or a CSP app, bind *restart session* (or *teleport to start/pits*) to a key combo. Put that combo in `config.yaml` → `env.reset.hotkey`, e.g. `["ctrl", "r"]`. Key names are pydirectinput names. While it's empty, the env asks you to reset by hand in the terminal, which is fine for a first test but not for real training. Hotkeys only reach AC when **the AC window has focus**.

6. **Session**: pick a car and track and start a *Practice* or *Hotlap* session. Running AC windowed is handy so you can watch the terminal.

7. **Record a reference line**: run `python scripts/record_line.py` and drive one clean lap yourself. Recording runs from one start/finish crossing to the next and saves `data/lines/<track>__<layout>.npz`.

8. **Check it all**: `python scripts/check_setup.py`. Fix anything marked FAIL. The velocity checks confirm the shared-memory struct layout matches your AC version.

## Train

```
python train.py
tensorboard --logdir runs
python train.py --resume runs/<run>/checkpoints/sac_100000_steps.zip
python evaluate.py runs/<run>/checkpoints/sac_100000_steps.zip
```

Click into the AC window during the 5 s countdown. Ctrl+C saves a checkpoint and the replay buffer. Pausing AC (Esc) is safe: the env holds neutral inputs and waits.

Gradient updates happen at the end of each episode, while the car sits at the start after the reset. Expect a short pause there.

## Design

- **Action** `[steer, pedal]` ∈ [-1, 1]². pedal > 0 is throttle, < 0 is brake.
- **Observation** (22 + 2 + len(lookahead)):
  - car state: speed, local velocity, angular velocity, g-forces, wheel slip, rpm, gear, inputs, tyres out, previous action
  - from the reference line: lateral offset, heading error, and how much the line turns at each lookahead distance
  - all of these are independent of the track's frame and position.
- **Reward**: `0.1 × metres of progress` along AC's track spline, minus a penalty per step off track (more than 2 wheels out) and a small steering-rate penalty.
- **Episode ends** when the car is off track for more than 1 s, stuck below 5 km/h for 4 s, goes backwards 30 m, or the time limit (180 s) runs out. Damage-based termination is optional.
- **Control rate**: 20 Hz.

## Things to verify on first run

These couldn't be tested off Windows (the tests use a fake AC):
- **Steering direction**: `vjoy_bind_helper.py test` should turn the wheel left then right. If it's inverted in AC, tick *invert* in the binding.
- **Reset hotkey**: watch that each episode end actually puts the car back at the start. The env raises an error if the car never comes to rest within `reset.timeout_s`.
- **Struct layout**: `check_setup.py` checks the velocity fields. If the local-velocity check fails, your AC build's physics struct differs.

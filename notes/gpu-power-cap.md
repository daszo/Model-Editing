# GPU power cap — how to lower / raise the watts

Quick reference for capping this box's RTX 3090 to trade speed for **less
heat, quieter fans, and steadier clocks** during long training runs. Companion
to `gpu-cooling.md` (the physical-cooling side).

The single lever is `nvidia-smi -pl <watts>` ("power limit"). Lower watts →
cooler chip → the fan backs off → quieter, but slower. It's safe and fully
reversible; nothing is written permanently.

---

## The commands

Check the current cap and live state:
```bash
nvidia-smi --query-gpu=power.draw,power.limit,temperature.gpu,fan.speed,clocks.sm --format=csv
```

Lower the cap (quieter / cooler):
```bash
sudo nvidia-smi -pl 150      # quiet mode — fan drops off 100%
```

Raise it back (faster):
```bash
sudo nvidia-smi -pl 200      # sweet spot: no thermal throttle, near-full speed
sudo nvidia-smi -pl 350      # full power (this card's default/max)
```

Allowed range on this card is **100–350 W**. Any value in between works
(e.g. `-pl 175`), so you can fine-tune.

---

## What each setting does on THIS card (measured 2026-09-13, t5-base training, case door open)

| cap | step time | ~time for 32 epochs | fan | temp | feel |
|-----|-----------|---------------------|-----|------|------|
| **350 W** (max) | 0.173 s/step | ~17 h | 100% | 71 °C | fastest, loud, surges/throttles |
| **200 W** | 0.194 s/step | ~19 h | 100% | 71 °C | ~full speed, no thermal throttle, still loud |
| **150 W** | 0.362 s/step | ~35 h | **74%** | 58–60 °C | ~half speed, **quiet enough to sleep near** |

Rules of thumb for this box:
- **Want it done fast, don't care about noise:** `-pl 350` (or 200 to avoid the
  thermal boost-collapse surging at almost no speed cost).
- **Want to sleep / need it quiet:** `-pl 150`. The chip runs cool enough that
  the fan finally drops below 100%. Slower, but if it runs overnight the extra
  time is free.
- The **fan is temperature-driven**, not directly settable here (already pinned
  at 100% under load, and there's no coolbits/X fan control). Lowering the power
  cap is the *only* way to make the fan quieter — cool the chip and it eases off.

---

## Good to know

- **Resets on reboot.** The cap is not persistent — after a restart the card is
  back at 350 W. Re-apply the command if you rebooted.
- **Passwordless sudo works** for `nvidia-smi` on this box, so the commands run
  without a prompt.
- The `persistence mode is disabled` warning after setting a cap is harmless —
  it does not undo the cap.
- Capping is safe: it only *lowers* the ceiling. It can't overheat or damage the
  card; worst case it just runs slower.

### Optional: make a cap survive reboots
Only if you want it permanent. Create a tiny systemd service:
```bash
sudo tee /etc/systemd/system/gpu-powercap.service >/dev/null <<'UNIT'
[Unit]
Description=Cap RTX 3090 power limit
After=multi-user.target

[Service]
Type=oneshot
ExecStart=/usr/bin/nvidia-smi -pl 200

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl enable --now gpu-powercap.service
```
Change `200` to whatever you want. Disable later with
`sudo systemctl disable --now gpu-powercap.service`.

---

See also: `gpu-cooling.md` (dust/paste/pads/airflow — the physical fixes) and
`repro-benchmarks.md` (why cooling dominates training time on this box).

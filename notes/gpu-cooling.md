# RTX 3090 cooling — diagnosis, fix ladder, and where to learn

Why this note exists: benchmarking the DOME reproduction showed the GPU is
**thermal-throttling**, and that costs ~3–4× on training wall-clock (the
t5-large run is ~7.5 d throttled; recovering the clock moves it toward ~4 d).
The card has an old water-cooling setup, which is the most likely cause. This is
fixable, and the fixes range from zero-risk software tweaks to a re-paste.

---

## Start here (TL;DR)

1. **Do the Tier-0 software steps first** — zero risk, might recover a chunk of
   clock without opening the card: `sudo nvidia-smi -pm 1`, then
   `sudo nvidia-smi -pl 250`, max the pump/fans, confirm the pump actually runs.
   Re-run the 2-min benchmark to see the effect.
2. **Don't open anything** until you've watched a full teardown of your *exact*
   card model.
3. **Watch order:** teardown of your model → repaste → thermal-pad replacement
   (the one that covers *measuring pad thickness* — matters a lot on a 3090) →
   drain/refill if it's a custom loop.
4. **Beginner-friendly first:** JayzTwoCents. **Most rigorous / 3090-specific:**
   Gamers Nexus. (Full list below.)

**To get specific videos:** tell me your exact card model (e.g. "EVGA RTX 3090
FTW3 Hybrid" vs "custom loop with an EK block") and whether it's a **sealed AIO**
or a **custom open loop**, and I'll web-search the current teardown/repaste
videos for that exact card instead of generic ones.

---

## The evidence (from the driver itself)

`nvidia-smi -q -d TEMPERATURE` and `-d PERFORMANCE` during a sustained run:

```
GPU Target Temperature : 83 °C        ← driver downclocks to hold this
SW Thermal Slowdown    : ~27 min cumulative   ← it IS thermal-throttling
SM clock               : 1950 MHz cold → collapses to ~500 MHz within seconds
Power draw             : ~160 W of a 350 W budget   ← NOT power-limited
Memory temp            : N/A (this card doesn't expose GDDR6X temp)
Persistence Mode       : Disabled
```

**Reading it:** the card hits 83 °C, so the driver drops clocks to stay there,
even though it's only using ~160 W of 350 W. A *power*-limited card would pull
the full 350 W and hold clocks. Yours leaves half the power budget unused and
clocks down anyway → **the cooler can't move heat off the die fast enough.** That
is a heat-transport problem, confirmed by the active `SW Thermal Slowdown`.

Note the coolant can be cold while the *die* is hot — that's the classic
signature of dried thermal paste or dried memory pads, not a "the water is hot"
problem.

---

## Fix ladder — start at the top (safe), stop when it's good enough

Re-run the 2-minute benchmark after each change to see if it helped:
`./.venv/bin/python notes/benchmarks/bench_train.py --model_name t5-large`
(watch the `s/step` number — lower is better; goal is well under ~1.7).

### Tier 0 — software & housekeeping (zero risk, do first)
- `sudo nvidia-smi -pm 1` — enable persistence mode.
- `sudo nvidia-smi -pl 250` — cap power to 250 W. Counter-intuitive but it often
  *raises* sustained clocks by stopping the thermal oscillation and holding a
  steadier operating point. (Revert with `sudo nvidia-smi -pl 350`.)
- Max the pump and radiator-fan speeds; confirm the **pump actually runs** (feel
  for vibration / listen). A dead or weak pump is a common failure on old loops.
- Clean dust from the radiator fins and improve case airflow (intake in front of
  the rad, clear exhaust path).

### Tier 1 — re-paste + new thermal pads (cheap, ~1 hr, the high-value fix)
Most likely the real cause on an old water-cooled card:
- **Thermal paste** between the die and the water block dries/pumps out over
  years → hot die, cold coolant. Fresh paste (or a PTM7950 phase-change pad)
  fixes this. This is *the* single highest-value maintenance step.
- **Memory/VRM thermal pads** dry out. The 3090's GDDR6X throttles at
  ~95–105 °C and is cooled through pads; dried pads throttle the card while the
  core reads a benign temp. Replace pads at the same time.
  - ⚠️ **Pad thickness matters.** GPU pads come in specific thicknesses
    (e.g. 1.0 / 1.5 / 2.0 mm). Note/measure what comes off and match it — wrong
    thickness = no contact = worse. Take photos before disassembly.

### Tier 2 — loop maintenance (only if it's a custom/open loop)
- Drain, flush, refill with fresh coolant; bleed trapped air; check/replace the
  pump if flow is weak. Air pockets and old coolant both kill heat transport.
- If it's a **sealed AIO/hybrid** cooler (not a custom loop), you can't really
  service the loop — a worn-out AIO pump (they last ~5–7 yrs) means replacing the
  cooler, or converting to an air/aftermarket cooler.

**First, figure out which you have:** a *custom loop* has separate parts —
reservoir, tubing you can open, a fill port. A *sealed AIO/hybrid* is one closed
unit (pump+block+hoses+rad) with no fill port. The maintenance path differs.

---

## Where to learn (channels I'd trust; search these, they're all active)

You're right to be wary — do Tier 0 first, and only open the card once you've
watched a full teardown for *your specific* card model. Reputable sources:

- **Gamers Nexus** (YouTube) — the most rigorous. Search
  "Gamers Nexus GPU repaste" and, very relevant to you,
  "Gamers Nexus RTX 3090 thermal pads / memory temperatures". Thorough, correct,
  explains *why*.
- **JayzTwoCents** (YouTube) — the most beginner-friendly for this. Search
  "JayzTwoCents how to repaste your GPU" and "JayzTwoCents drain and refill
  custom loop". Calm, step-by-step, aimed at first-timers.
- **der8auer** (YouTube) — expert on paste/repaste technique (he runs Thermal
  Grizzly). Great once you want to do it *precisely*; also good on PTM7950 pads.
- **Linus Tech Tips** (YouTube) — accessible water-cooling builds/maintenance
  overviews; good for the mental model, lighter on precision.
- **iFixit** (website, ifixit.com) — teardown guides + the general repair skills
  (anti-static, tools, screw discipline) and the toolkit if you don't have one.
- **EKWB blog / Thermal Grizzly guides** (websites) — vendor guides on loop
  maintenance and paste/pad application; useful reference, mind the sales pitch.

**Watch, in order:** (1) a full teardown of *your exact* GPU model before
touching anything; (2) a repaste guide; (3) a thermal-pad-replacement guide that
covers measuring pad thickness; (4) if custom loop, a drain/refill guide.

### Beginner safety / prep
- Photograph every step; bag screws by location. Note pad thicknesses.
- Anti-static: work on a hard surface, touch bare metal / use a strap, avoid
  carpet.
- Don't force fittings or over-torque the block; snug, not gorilla.
- Buy: quality paste (or PTM7950), a thermal-pad kit in the right thicknesses,
  91 %+ isopropyl alcohol, lint-free wipes, a precision screwdriver set.
- Give yourself an unhurried afternoon, not a rushed hour.

---

## How to know it worked
Before: sustained ~1.7 s/step (t5-large, throttled). After a fix, re-run the
benchmark; every 2× of clock recovered is roughly 2× off the training week.
Also re-check `nvidia-smi -q -d PERFORMANCE` — you want `SW Thermal Slowdown` to
stop climbing under load, and the SM clock to hold high instead of collapsing.

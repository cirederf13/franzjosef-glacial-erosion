#!/usr/bin/env python3
"""Build glacial_cycle_smb_rescaled_lgm_v4.txt from glacial_cycle_smb.txt: anchor-preserving linear rescale of the ELA and accmax columns."""

ANCHOR = 1826.32
ORIG_LGM_MIN = 690.32

SCALE = 0.994777  # fitted, see docstring
TARGET_LGM_MIN = ANCHOR - SCALE * (ANCHOR - ORIG_LGM_MIN)  # = 696.25

RAMP_START_ELA = 1100.0
RAMP_END_ELA = TARGET_LGM_MIN  # accmax = ACCMAX_MIN at this (the new LGM min)
ACCMAX_FULL = 3.0
ACCMAX_MIN = 2.64  # fitted, see docstring (raw fit: 2.6376)

src = "glacial_cycle_smb.txt"
dst = "glacial_cycle_smb_rescaled_lgm_v4.txt"

with open(src) as f:
    lines = f.readlines()

header, rows = lines[0], lines[1:]

out_lines = [header]
for line in rows:
    t, gradabl, gradacc, ela, accmax = line.split()
    t, gradabl, gradacc, ela, accmax = (
        float(t), float(gradabl), float(gradacc), float(ela), float(accmax)
    )

    new_ela = ANCHOR - SCALE * (ANCHOR - ela)

    if new_ela >= RAMP_START_ELA:
        new_accmax = ACCMAX_FULL
    else:
        frac = (RAMP_START_ELA - new_ela) / (RAMP_START_ELA - RAMP_END_ELA)
        frac = min(frac, 1.0)  # clip in case of any float overshoot past the LGM min
        new_accmax = ACCMAX_FULL - frac * (ACCMAX_FULL - ACCMAX_MIN)

    out_lines.append(
        f"{t:.2f} {gradabl:.2f} {gradacc:.2f} {new_ela:.2f} {new_accmax:.2f}\n"
    )

with open(dst, "w") as f:
    f.writelines(out_lines)

# sanity report
new_min_row = min(rows, key=lambda l: float(l.split()[3]))
orig_min_ela = float(new_min_row.split()[3])
orig_min_t = float(new_min_row.split()[0])
new_min_ela = ANCHOR - SCALE * (ANCHOR - orig_min_ela)
print(f"scale factor: {SCALE:.6f}")
print(f"LGM row (t={orig_min_t}): orig ela={orig_min_ela:.2f} -> new ela={new_min_ela:.2f}")
print(f"present-day anchor row (t=-70.0) unchanged: {ANCHOR}")
n_below_1100 = sum(
    1 for l in out_lines[1:] if float(l.split()[3]) < RAMP_START_ELA
)
n_reduced = sum(
    1 for l in out_lines[1:] if float(l.split()[4]) < ACCMAX_FULL
)
print(f"rows with new ela < {RAMP_START_ELA}: {n_below_1100}")
print(f"rows with reduced accmax: {n_reduced}")
ela_vals = [float(l.split()[3]) for l in out_lines[1:]]
accmax_vals = [float(l.split()[4]) for l in out_lines[1:]]
print(f"ela range: {min(ela_vals):.2f} .. {max(ela_vals):.2f}")
print(f"accmax range: {min(accmax_vals):.2f} .. {max(accmax_vals):.2f}")
print(f"wrote {dst}")

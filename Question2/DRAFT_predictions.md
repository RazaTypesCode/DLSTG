# DRAFT predictions – rewrite in your own words, then commit BEFORE running the experiments they cover

These are drafts of Q2.5 and Q2.11 based on the handout, the reach calculation (model A = 8 px/step) and
standard properties of Moving MNIST. They are predictions, not results: edit them so they say what *you*
believe. The wording you commit is the wording that goes in the report, even if it turns out wrong.
Fill the [brackets] from your Figure 1 once you have it.

## Q2.5 (commit before running `stage_A2.py`)

I expect model A to pass the mirror test reasonably well and the rotation test noticeably worse. Figure 1(a)
should show a roughly flat direction histogram (about 1/18 per bin), so mirrored and rotated trajectories
already occur in training, and any failure should come from digit appearance rather than from motion; a
mirrored digit is a slightly unfamiliar shape, a rotated one more so, so I expect C_R and E_R to exceed C_M and
E_M, with the worst cases at overlapping digits. For speed, the median flow speed in Figure 1(b) is about
[X] px/frame, so 2x ≈ [2X] and 3x ≈ [3X] px/frame against model A's reach of 8 px per step; I expect A to
still track at 1x, to start degrading at 2x if [2X] is close to 8 px, and to break down at 3x, where [3X] exceeds 8 px
(digits stall, blur or duplicate).

## Q2.11 (≤ 3 sentences; commit before training B, C, D)

Model B should reduce C_M, C_R and the transformed errors (appearance failure) but not fix 2x/3x. Model C should
fix the speeds that lie within the 8-px reach (2x) but I expect it still to fail at 3x. Model D, with its 16-px
reach, should fix the remaining 3x failure, while none of B, C, D should by themselves make C_M, C_R exactly
zero or remove blur at overlapping digits (only the symmetrised wrapper does the former).

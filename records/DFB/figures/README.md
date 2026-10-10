# Architecture-diagram assets (generated from FF++, not stock images)

Every panel here was produced by `scripts_gen_arch_assets.py` from the real
FaceForensics++ c23 data and the trained checkpoint
`records/DFB/models_ffpp_c23/best.pth`, using the same code paths the model runs at
inference (`DFBTriStreamDataset` for preprocessing, `Stream2`'s own BayarConv /
`FFTBranch._spectrum` / Haar kernels for the frequency panels, `Stream3V2` +
`LocalizationHead` for `M'` and the predicted mask). Nothing is re-implemented
approximately and nothing is illustrative.

Four sets are included, one per FF++ manipulation. **`arch_assets_NeuralTextures/`
reads best** — its ground-truth mask is face-shaped rather than a rectangle, so the
localization panels are the most informative. `arch_assets/` is Deepfakes.

Each folder's `SOURCE.txt` records the exact clip, frame, bbox and checkpoint used,
plus the model's `p(fake)` for that sample.

## Where each asset goes in the diagram

| Diagram box | Asset |
|---|---|
| DeepFakeBench Data | `01_raw_frame.png`, `04b_landmark_heatmap_gray.png`, `05_gt_mask.png` |
| Preprocessing → Landmark BBox Crop | `02_landmark_bbox.png` (81 landmarks + the 25 %-margin crop window) |
| Preprocessing → Heatmap *H* Generation | `04_landmark_heatmap.png` (or `04c_heatmap_on_face.png`) |
| Stream 1: Temporal — clip input | `11_frame_stack.png` (perspective stack) or `10_filmstrip.png` (8 of the T=32 frames) |
| Stream 2 — BayarConv | `20_bayar_residual.png` (mean abs residual), `21_bayar_grid.png` (8 highest-energy channels) |
| Stream 2 — FFT | `22_fft_spectrum.png` (DC-notched log spectrum, exactly `FFTBranch._spectrum`) |
| Stream 2 — DWT | `23_dwt_subbands.png` (LL/LH/HL/HH quad) or the four `23_dwt_*.png` singles |
| Stream 3 — face input | `03_face_crop.png` (fake), `03b_face_crop_real.png` (same identity, real) |
| Stream 3 — Conv Stem output (pre-gate) | `30_spatial_prepool.png` |
| Stream 3 — Landmark-Attention gate | `31_landmark_gate.png` / `31b_landmark_gate_smooth.png` |
| **Spatial Map *M′*** | `32_spatial_map_Mprime.png` (true 14×14 grid) or `32d_..._smooth.png`; `32c_Mprime_on_face.png` for an overlay |
| Localization Head → mask | `40_pred_mask_prob.png` (sigmoid, true 28×28), `41_pred_mask_binary.png`, `42_pred_mask_on_face.png` |
| Ground truth, for a side-by-side | `05_gt_mask.png`, `05b_gt_mask_on_face.png` |

`*_smooth.png` variants are bicubic-upsampled companions — use them when the panel is
shrunk into a small diagram box; the default files keep the true feature-map resolution
(14×14 for `M′`, 28×28 for the mask) visible.

`99_contact_sheet.png` in each folder is a labelled index of everything in it.

`triforensics_arch_fixed.png` is the corrected architecture diagram (`M′` → Localization
Head, `f_s` → Additive Injection).

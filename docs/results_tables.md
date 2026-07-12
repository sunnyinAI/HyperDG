# Controlled multi-seed results

Mean +/- sample SD over 4 seeds x 4 held-out domains. Parentheses show raw/Holm-adjusted paired p-values versus the matched zero-penalty control.

| Condition | pacs | vlcs | officehome | terraincognita |
|---|---:|---:|---:|---:|
| Zero-penalty control | 84.04 +/- 7.10 | 75.63 +/- 13.32 | 67.05 +/- 10.26 | 43.90 +/- 8.49 |
| Full HyperDG (alpha_var=0.2) | 84.50 +/- 7.45 (p=0.534/1.000) | 75.47 +/- 14.14 (p=0.713/1.000) | 66.03 +/- 9.98 (p=0.013/0.153) | 43.17 +/- 4.97 (p=0.631/1.000) |
| Curvature frozen (no learning) | 84.67 +/- 7.42 (p=0.301/1.000) | 76.27 +/- 13.17 (p=0.237/1.000) | 66.13 +/- 9.78 (p=0.020/0.200) | 44.30 +/- 6.74 (p=0.784/1.000) |
| No variance reg (alpha_var=0) | 84.57 +/- 6.92 (p=0.422/1.000) | 75.04 +/- 14.25 (p=0.348/1.000) | 66.03 +/- 9.75 (p=0.017/0.185) | 43.90 +/- 6.94 (p=1.000/1.000) |

#!/bin/zsh
# exp17: full bench sequence (strictly serial, machine otherwise idle).
set -e
cd "$(dirname "$0")"
./build/bcn bench --curve a --out ../results/exp17_curve_a.csv && echo A_DONE
./build/bcn bench --curve ap --out ../results/exp17_curve_ap.csv && echo AP_DONE
./build/bcn bench --curve b --out ../results/exp17_curve_b.csv && echo B_DONE
./build/bcn bench --curve c --out ../results/exp17_curve_c.csv && echo C_DONE
./build/bcn bench --curve tlgl --net ../networks/t_lgl.txt --out ../results/exp17_tlgl.csv --dump ../results/exp17_crosscheck/res_tlgl.txt && echo ALL_DONE

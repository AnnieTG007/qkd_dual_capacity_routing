# data/

Optional CSV data files for classical capacity and GNPy results.

## Classical capacity CSV format

```csv
u,v,classical_capacity_gbps
0,1,400
0,2,800
...
```

- `u`, `v`: node indices (0-based)
- `classical_capacity_gbps`: per-edge classical capacity in Gb/s
- Undirected — `(u, v)` and `(v, u)` are treated as the same edge

## GNPy result CSV format

### Direct capacity

```csv
u,v,classical_capacity_gbps
0,1,375.2
0,2,412.8
...
```

### GSNR-based (auto-detected by presence of `gsnr_db` column)

```csv
u,v,gsnr_db,bandwidth_ghz
0,1,18.5,75
0,2,22.1,75
...
```

- `gsnr_db`: GSNR in dB (generalised signal-to-noise ratio)
- `bandwidth_ghz`: per-channel bandwidth in GHz (can be omitted if all links use the same value; global default is 75 GHz)
- GSNR → Shannon capacity: `B_hz * log2(1 + gsnr_linear / margin_linear) / 1e9`

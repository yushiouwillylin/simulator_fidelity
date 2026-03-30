# Dataset Archives

This directory stores the zip archives for the datasets used by the reproduction notebooks, along with the unpack script that restores them into `data/`.

## Archives

- `worldvalue_data.zip`: source archive for `data/worldvalue/` and `data/worldvaluesbench/`.
- `eedi_data.zip`: source archive for `data/eedi/`.
- `opinionqa_data.zip`: source archive for `data/opinionqa/`.
- `unpack_reproduction_data.py`: extracts the archives into the internal bundle data tree.

## How To Restore

From the repository root:

```bash
python datasets/unpack_reproduction_data.py
```

For WorldValue, the default extraction mode is the minimal paper-reproduction layout:

```bash
python datasets/unpack_reproduction_data.py --dataset worldvalue --worldvalue-layout minimal
```

Use the full upstream layout only if you also want the original benchmark splits and auxiliary outputs:

```bash
python datasets/unpack_reproduction_data.py --dataset worldvalue --worldvalue-layout full
```

The notebooks expect these folders to be restored inside the reproduction bundle:

- `worldvalue_data.zip` extracts to `data/worldvalue/` and the required minimal subset of `data/worldvaluesbench/` by default
- `eedi_data.zip` extracts to `data/eedi/`
- `opinionqa_data.zip` extracts to `data/opinionqa/`

For the minimal WorldValue restore, the largest remaining file is `data/worldvaluesbench/F00011356-WVS_Cross-National_Wave_7_csv_v6_0.zip`. This is already the compressed raw WVS archive; the minimal layout intentionally keeps it zipped rather than expanding it to a much larger CSV tree.


## UMAR Data

The UMAR building-control reproduction uses repo-local CSV inputs under `data/umar/` rather than a zip archive restored by `unpack_reproduction_data.py`.

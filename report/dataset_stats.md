# Hindi Dataset Statistics

## Collection & cleaning pipeline (per source)

| Source | Raw docs | Kept after cleaning | Dropped (clean filters) | Dropped (language-ID) |
|---|---|---|---|---|
| ai4bharat_hi_subset | 1609711 | 1546751 | 62530 | 430 |
| indiccorp_v2 | 4982046 | 4787397 | 193350 | 1299 |
| sangraha_synthetic_hin_deva | 200329 | 192837 | 6507 | 985 |

## Per-source counts (final corpus, after dedup)

| Source | Documents | Words |
|---|---|---|
| ai4bharat_hi_subset | 1545080 | 92078181 |
| indiccorp_v2 | 3231017 | 192291065 |
| sangraha_synthetic_hin_deva | 192689 | 109732706 |

## Deduplication

Documents removed by dedup (exact + near, combined): 26.8%

## Manual vs. downloaded token split

0% manual / 100% public downloaded. Per the project's agreed scope, the assignment's usual >=20% manual-collection requirement was dropped in favor of reaching the ~500M-token target entirely from public Hugging Face datasets (IndicCorpV2, zicsx/ai4bharat-hi-subset, and ai4bharat/sangraha's synthetic/hin_Deva split as a top-up).

## Train / validation / test splits

| Split | Documents | Words |
|---|---|---|
| train | 4869410 | 386133037 |
| val | 49688 | 3956205 |
| test | 49688 | 4012710 |

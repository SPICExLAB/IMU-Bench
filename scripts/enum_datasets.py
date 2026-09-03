import pyarrow.dataset as pa_ds
import pyarrow.compute as pc

CANDIDATES = ["pamap2", "prism", "recofit", "samosa", "shoaib", "ut_watch",
              "utd_mhad", "wisdm", "motionsense", "usc_had", "dsads", "opportunity"]
d = pa_ds.dataset('/projects/b53017/vbc1591/IMULM_master_dataset_v2.parquet', format='parquet')
for name in CANDIDATES:
    try:
        t = d.to_table(columns=['dataset_activity_label'],
                       filter=(pa_ds.field('dataset') == name))
        if t.num_rows == 0:
            print(f"[{name}] ABSENT", flush=True)
            continue
        vc = pc.value_counts(t['dataset_activity_label'])
        pairs = sorted(((str(x['values']).replace('_',' ').lower(), int(x['counts']))
                        for x in vc.to_pylist()), key=lambda p: -p[1])
        pairs = [(k, v) for k, v in pairs if k not in ('nan','none','unknown','-1','None')]
        print(f"[{name}] rows={t.num_rows:,} labels({len(pairs)}): {pairs}", flush=True)
    except Exception as e:
        print(f"[{name}] ERR {e}", flush=True)

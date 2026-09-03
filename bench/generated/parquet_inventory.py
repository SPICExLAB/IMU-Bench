import pyarrow.dataset as ds, pyarrow.compute as pc, json
d=ds.dataset("/projects/b53017/vbc1591/IMULM_master_dataset_v2.parquet", format="parquet")
names=pc.unique(d.to_table(columns=["dataset"])["dataset"]).to_pylist()
out={}
for n in sorted(names):
    t=d.to_table(columns=["subject_id","session_id","dataset_activity_id","gyro_x"], filter=pc.field("dataset")==n)
    _m=pc.mean(pc.greater(pc.abs(t["gyro_x"]),1e-9)).as_py()
    nz=float(_m) if _m is not None else 0.0
    out[n]=dict(subjects=len(pc.unique(t["subject_id"])), sessions=len(pc.unique(t["session_id"])),
                classes=len(pc.unique(t["dataset_activity_id"])), samples=t.num_rows,
                hours=round(t.num_rows/50/3600,1), gyro_nonzero=round(nz,3))
    print(f"{n:24s} subj {out[n]['subjects']:>4} sess {out[n]['sessions']:>5} classes {out[n]['classes']:>3} "
          f"{out[n]['hours']:>7.1f} h  gyro_nonzero {out[n]['gyro_nonzero']}", flush=True)
json.dump(out, open("/gpfs/home/vbc1591/IMU-SSL/bench/generated/parquet_inventory_dsid.json","w"), indent=1)

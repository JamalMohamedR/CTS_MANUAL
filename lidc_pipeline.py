"""
fix_trainer.py
==============
Fixes the nnUNetTrainerFGOversample signature to match your installed
nnU-Net version, then restarts training.

Run:
    python fix_trainer.py
"""

import os, sys, subprocess, inspect

# ── Match paths from lidc_pipeline.py ────────────────────────────────────────
BASE_DIR     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nnunet_workspace")
RAW_DIR      = os.path.join(BASE_DIR, "nnUNet_raw")
PREPROCESSED = os.path.join(BASE_DIR, "nnUNet_preprocessed")
RESULTS_DIR  = os.path.join(BASE_DIR, "nnUNet_results")
DATASET_ID   = "001"
FOLD         = 0
# ─────────────────────────────────────────────────────────────────────────────

env = os.environ.copy()
env["nnUNet_raw"]          = RAW_DIR
env["nnUNet_preprocessed"] = PREPROCESSED
env["nnUNet_results"]      = RESULTS_DIR
env["nnUNet_n_proc_DA"]    = "2"
env["OMP_NUM_THREADS"]     = "1"
env["MKL_NUM_THREADS"]     = "1"


def write_correct_trainer():
    try:
        import nnunetv2
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
    except ImportError:
        print("[ERROR] nnunetv2 not installed"); sys.exit(1)

    trainer_dir = os.path.join(
        os.path.dirname(nnunetv2.__file__),
        "training", "nnUNetTrainer"
    )

    # Inspect the actual __init__ signature of the installed nnUNetTrainer
    sig    = inspect.signature(nnUNetTrainer.__init__)
    params = list(sig.parameters.keys())   # e.g. ['self','plans','configuration','fold','dataset_json','device']
    print(f"[INFO] nnUNetTrainer.__init__ signature: {params}")

    # Build the super().__init__ call using only the params that exist
    # (exclude 'self', always pass plans/configuration/fold/dataset_json)
    call_args = ["plans", "configuration", "fold", "dataset_json"]
    if "unpack_dataset" in params:
        call_args.append("unpack_dataset")
    if "device" in params:
        call_args.append("device")

    # Build __init__ def args — match whatever the parent has
    def_args = ", ".join(
        f"{p}=None" if p not in ("self","plans","configuration","fold","dataset_json")
        else p
        for p in params
        if p != "self"
    )
    super_args = ", ".join(call_args)

    code = f"""from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerFGOversample(nnUNetTrainer):
    \"\"\"
    Forces 33% of patches per batch to be centred on foreground (nodule) voxels.
    Essential for highly imbalanced datasets like LIDC-IDRI where nodules are
    < 0.01% of voxels. Without this, Dice stays at 0 throughout training.
    \"\"\"

    def __init__(self, {def_args}):
        super().__init__({super_args})
        # At least 1 in 3 patches per batch must contain a nodule voxel
        self.oversample_foreground_percent = 0.33
"""

    out_path = os.path.join(trainer_dir, "nnUNetTrainerFGOversample.py")
    with open(out_path, "w") as f:
        f.write(code)

    print(f"[OK] Trainer written → {out_path}")
    print(f"     def __init__(self, {def_args})")
    print(f"     super().__init__({super_args})")
    return "nnUNetTrainerFGOversample"


def train(trainer_name):
    cmd = (
        f"nnUNetv2_train {DATASET_ID} 3d_fullres {FOLD} "
        f"-tr {trainer_name} --npz -num_gpus 1"
    )
    print(f"\n[CMD] {cmd}")
    r = subprocess.run(cmd, shell=True, env=env)
    if r.returncode != 0:
        print("\n[ERROR] Training failed — see message above.")
        sys.exit(r.returncode)
    print("\n[OK] Training complete.")
    print("[NEXT] python lidc_pipeline.py --mode predict")


if __name__ == "__main__":
    print("=" * 60)
    print("  Fix nnUNetTrainerFGOversample + Resume Training")
    print("=" * 60)
    trainer = write_correct_trainer()
    train(trainer)
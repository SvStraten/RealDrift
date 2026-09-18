import sys
import os
import re
import glob
import random
import inspect
import datetime
import argparse
from collections import Counter, defaultdict

import torch
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Random seed for training/generation")
parser.add_argument("--repo-path", type=str,
                     default=os.environ.get("TF_DECODER_REPO_PATH",
                                             "/vast.mnt/home/20182633/ICPM/TF-decoder-0096"),
                     help="Path to the TF-decoder repo checkout (contains src/, configs/).")
parser.add_argument("--data-root", type=str, default=None,
                     help="Parent directory holding data_{dataset}_rank{N} folders. "
                          "Defaults to the repo path's parent directory (matches the "
                          "original ../data_rank{N} convention).")
parser.add_argument("--datasets", type=str, default="bpic12,emergency",
                     help="Comma-separated subset of {bpic12,emergency} to run.")
parser.add_argument("--ranks", type=str, default=None,
                     help="Comma-separated rank numbers to run (default: all ranks "
                          "found under --data-root for each requested dataset).")
parser.add_argument("--max-epochs", type=int, default=200)
parser.add_argument("--use-beam-search", action="store_true", default=True)
parser.add_argument("--beam-size", type=int, default=4)
args, _ = parser.parse_known_args()

SEED = args.seed
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print(f"Using seed: {SEED}")


REPO_PATH = os.path.abspath(args.repo_path)
assert os.path.isdir(os.path.join(REPO_PATH, "src")), f"src/ not found under {REPO_PATH} -- check --repo-path"

DATA_ROOT = os.path.abspath(args.data_root) if args.data_root else os.path.abspath(os.path.join(REPO_PATH, ".."))

os.chdir(REPO_PATH)
sys.path.insert(0, REPO_PATH)
sys.path.insert(0, os.path.join(REPO_PATH, "src"))

print("cwd:", os.getcwd())
print("data root:", DATA_ROOT)


_original_load = torch.load
def _patched_load(*load_args, **load_kwargs):
    load_kwargs["weights_only"] = False   # force override, not setdefault
    return _original_load(*load_args, **load_kwargs)
torch.load = _patched_load

import hydra
from omegaconf import OmegaConf
from lightning import Trainer
from models.components.cvae.vae2 import VAE  

_decode_src = inspect.getsource(VAE.decode)
assert "self.decoder(z, c" in _decode_src and "**kwargs" in _decode_src, (
    "VAE.decode (models/components/cvae/vae2.py) does not forward **kwargs to "
    "self.decoder -- the beam-search passthrough is missing. Fix vae2.py first."
)
print("Confirmed: VAE.decode (vae2) forwards kwargs to the decoder (beam search fix is active).")


DATASET_CONFIG = {
    "bpic12": {"data_yaml": "configs/data/bpic.yaml", "subfolder": "bpic2012_a"},
    "emergency": {"data_yaml": "configs/data/emergency.yaml", "subfolder": "emergency_ORT"},
    "sepsis": {"data_yaml": "configs/data/sepsis.yaml", "subfolder": "sepsis"},
}


def discover_ranks(dataset: str, data_root: str = DATA_ROOT):
    pattern = os.path.join(data_root, f"data_{dataset}_rank*")
    ranks = []
    for p in glob.glob(pattern):
        m = re.search(rf"data_{dataset}_rank(\d+)$", p)
        if m:
            ranks.append(int(m.group(1)))
    return sorted(ranks)


def count_cases_in_full_csv(dataset: str, rank: int, data_root: str = DATA_ROOT) -> int:
    """Total number of traces in this concept's sublog."""
    cfg = DATASET_CONFIG[dataset]
    csv_path = os.path.join(data_root, f"data_{dataset}_rank{rank}", cfg["subfolder"],
                             f"{cfg['subfolder']}.csv")
    df = pd.read_csv(csv_path, sep=";", usecols=["Case ID"], keep_default_na=False, na_values=[])
    return df["Case ID"].nunique()


def compute_max_trace_length(dataset: str, rank: int, data_root: str = DATA_ROOT) -> int:
    cfg = DATASET_CONFIG[dataset]
    csv_path = os.path.join(data_root, f"data_{dataset}_rank{rank}", cfg["subfolder"],
                             f"{cfg['subfolder']}.csv")
    df = pd.read_csv(csv_path, sep=";", usecols=["Case ID"], keep_default_na=False, na_values=[])
    return int(df.groupby("Case ID").size().max())


def build_datamodule(dataset: str, rank: int, data_root: str = DATA_ROOT):
    cfg_info = DATASET_CONFIG[dataset]
    cfg = OmegaConf.load(cfg_info["data_yaml"])
    OmegaConf.set_struct(cfg, False)
    data_dir = os.path.join(data_root, f"data_{dataset}_rank{rank}")
    cfg.data_dir = os.path.abspath(data_dir)
    cfg.dataset_name = cfg_info["subfolder"]  # explicit, don't rely on the class default
    cfg.max_trace_length = compute_max_trace_length(dataset, rank, data_root)
    return hydra.utils.instantiate(cfg)


def build_model(num_labels: int):
    model_cfg = OmegaConf.load("configs/model/tf_decoder.yaml")
    OmegaConf.set_struct(model_cfg, False)
    model_cfg.vae.c_dim = max(num_labels, 1)
    return hydra.utils.instantiate(model_cfg)

def get_latest_checkpoint(dataset: str, rank: int):
    pattern = f"training_runs/{dataset}/rank{rank}/lightning_logs/version_*/checkpoints/*.ckpt"
    ckpts = glob.glob(pattern)
    return max(ckpts, key=os.path.getmtime) if ckpts else None


def epoch_of_checkpoint(ckpt_path):
    if ckpt_path is None:
        return -1
    m = re.search(r"epoch=(\d+)", os.path.basename(ckpt_path))
    return int(m.group(1)) if m else -1


def fix_device_attrs(module, device):
    for m in module.modules():
        if hasattr(m, "device") and isinstance(getattr(m, "device"), torch.device):
            m.device = device

def constrained_activity_sequence(act_logits, activity_names, w_activities, eot_name="EOT"):
    seq_len = act_logits.shape[0]
    open_acts = set()
    chosen = []
    start_positions = {}

    for j in range(seq_len):
        logits = act_logits[j].clone()
        for idx, name in enumerate(activity_names):
            if name == eot_name:
                continue
            base, _, lc = name.rpartition('-')
            if base not in w_activities:
                continue
            if lc == 'START' and base in open_acts:
                logits[idx] = float('-inf')
            elif lc == 'COMPLETE' and base not in open_acts:
                logits[idx] = float('-inf')

        chosen_idx = int(torch.argmax(logits).item())
        chosen_name = activity_names[chosen_idx]

        if chosen_name == eot_name:
            chosen.append(chosen_idx)
            break

        base, _, lc = chosen_name.rpartition('-')
        if base in w_activities:
            if lc == 'START':
                open_acts.add(base)
                start_positions[base] = len(chosen)
            elif lc == 'COMPLETE':
                open_acts.discard(base)
        chosen.append(chosen_idx)

    if open_acts:
        drop_positions = {start_positions[b] for b in open_acts if b in start_positions}
        chosen = [c for k, c in enumerate(chosen) if k not in drop_positions]

    return chosen


def check_alternation(activity_token_names, w_activities):
    per_act = defaultdict(list)
    for name in activity_token_names:
        base, _, lc = name.rpartition('-')
        if base in w_activities and lc in ('START', 'COMPLETE'):
            per_act[base].append(lc)
    for base, lcs in per_act.items():
        if len(lcs) % 2 != 0:
            return False
        for i in range(0, len(lcs), 2):
            if lcs[i] != 'START' or lcs[i + 1] != 'COMPLETE':
                return False
    return True

def generate_n_traces(lightning_module, datamodule, num_traces: int,
                       use_beam_search: bool = True, beam_size: int = 4, batch_size: int = 256,
                       max_gap_minutes: float = 60 * 24 * 14):
    vae = lightning_module.model
    vae.eval()
    device = next(vae.parameters()).device

    test_set = datamodule.data_test

    activity_names = [test_set.n2activity[i] for i in range(len(test_set.n2activity))]
    w_activities = {name.rsplit('-', 1)[0] for name in activity_names if name.endswith('-START')}
    print(f"W_* activities detected from vocab: {sorted(w_activities)}")

    train_set = datamodule.data_train
    label_counts = Counter(tuple(y.tolist()) for y in train_set.y)
    label_proportions = {train_set.onehot2label[lbl]: c / len(train_set.y) for lbl, c in label_counts.items()}
    print(f"Label proportions used for generation (from {len(train_set.y)} training traces):", label_proportions)

    sampled_labels = random.choices(
        list(label_proportions.keys()), weights=list(label_proportions.values()), k=num_traces
    )
    c_list = [test_set.label2onehot[label] for label in sampled_labels]
    labels_tensor = torch.stack(c_list)

    new_data = []
    n_wellformed = 0

    with torch.no_grad():
        for start in range(0, num_traces, batch_size):
            end = min(start + batch_size, num_traces)
            c = labels_tensor[start:end].to(device)
            z = torch.randn(end - start, vae.z_dim, device=device)

            out = vae.decode(z, c, use_beam_search=use_beam_search, beam_size=beam_size)
            if len(out) == 5:
                attrs, acts, ts, ress, _ = out
            else:
                attrs, acts, ts, ress = out
            ts = ts.clamp_min(0)

            if start == 0:
                print(f"[diagnostic] acts shape={tuple(acts.shape)} dtype={acts.dtype} "
                      f"min={acts.min().item():.4f} max={acts.max().item():.4f} "
                      f"sums_to_one(dim=-1)~{torch.allclose(acts[0,0].sum(), torch.tensor(1.0), atol=1e-2)}")

            trace_attrs = {}
            for attr in test_set.trace_attributes:
                if attr["type"] == "categorical":
                    attrs[attr["name"]] = torch.argmax(attrs[attr["name"]], dim=1)
                else:
                    attrs[attr["name"]] = attr["min_value"] + attrs[attr["name"]] * (attr["max_value"] - attr["min_value"])
                trace_attrs[attr["name"]] = attrs[attr["name"]]

            ress_idx = torch.argmax(ress[:, :, :-1], dim=2)

            for i in range(end - start):
                global_i = start + i
                case_label = sampled_labels[global_i]
                case_attrs = {k: v[i] for k, v in trace_attrs.items()}
                start_datetime = pd.to_datetime(test_set.log["time:timestamp"]).min()
                start_offset = datetime.timedelta(minutes=case_attrs["relative_timestamp_from_start"].item())
                current_time = start_datetime + start_offset
                case_id = f"GEN{global_i}"

                chosen_indices = constrained_activity_sequence(acts[i], activity_names, w_activities)

                token_names_for_case = []
                for j, act_idx in enumerate(chosen_indices):
                    activity_name = activity_names[act_idx]
                    if activity_name == "EOT":
                        break
                    token_names_for_case.append(activity_name)

                    res_idx = ress_idx[i][j] if j < ress_idx.shape[1] else ress_idx[i][-1]
                    resource_name = test_set.n2resource[res_idx.item()]

                    cat_attrs = {k: test_set.i2s[k][v.item()] for k, v in case_attrs.items() if k in test_set.i2s}
                    num_attrs = {k: v.item() for k, v in case_attrs.items() if k not in test_set.i2s}

                    time_value = ts[i][j].item() * test_set.highest_ts if j < ts.shape[1] else 0.0
                    clamped = min(time_value, max_gap_minutes)
                    current_time += datetime.timedelta(minutes=clamped)

                    new_data.append({
                        "case:concept:name": case_id, "concept:name": activity_name,
                        "org:resource": resource_name,
                        "time:timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                        **cat_attrs, **num_attrs, "case:label": case_label,
                    })

                if check_alternation(token_names_for_case, w_activities):
                    n_wellformed += 1

    df = pd.DataFrame(new_data)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
    print(f"Generated {df['case:concept:name'].nunique()} traces, {len(df)} events")
    print(f"Well-formed START/COMPLETE alternation: {n_wellformed}/{num_traces} "
          f"({n_wellformed/num_traces*100:.1f}%)")
    return df

def main():
    requested_datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    for d in requested_datasets:
        assert d in DATASET_CONFIG, f"Unknown dataset '{d}', expected one of {list(DATASET_CONFIG)}"

    requested_ranks = None
    if args.ranks:
        requested_ranks = {int(r.strip()) for r in args.ranks.split(",") if r.strip()}

    results = {}

    for dataset in requested_datasets:
        ranks = discover_ranks(dataset)
        if requested_ranks is not None:
            ranks = [r for r in ranks if r in requested_ranks]
        if not ranks:
            print(f"No data_{dataset}_rank* folders found under {DATA_ROOT} -- skipping {dataset}. "
                  f"Run prepare_concept_datasets.py first.")
            continue

        for rank in ranks:
            print(f"\n=== {dataset}: concept rank {rank} ===")

            num_traces = count_cases_in_full_csv(dataset, rank)
            print(f"{dataset} rank{rank}: will generate {num_traces} traces "
                  f"(== total traces in this concept's sublog)")

            ckpt_path = get_latest_checkpoint(dataset, rank)
            saved_epoch = epoch_of_checkpoint(ckpt_path)
            max_epochs_this_rank = max(args.max_epochs, saved_epoch + 1)

            datamodule = build_datamodule(dataset, rank)
            num_labels = len(datamodule.dataset_info.labels)
            print(f"{dataset} rank{rank}: {num_labels} distinct label(s) "
                  f"({datamodule.dataset_info.labels}) -> c_dim={max(num_labels, 1)}, "
                  f"max_trace_length={datamodule.hparams.max_trace_length}")

            cfg_info = DATASET_CONFIG[dataset]
            split_dir = os.path.join(DATA_ROOT, f"data_{dataset}_rank{rank}", cfg_info["subfolder"])
            known_labels = set(datamodule.dataset_info.labels)
            for split_name, split_path in [("TRAIN", datamodule.dataset_info.train_path),
                                            ("VAL", datamodule.dataset_info.val_path),
                                            ("TEST", datamodule.dataset_info.test_path)]:
                split_labels = set(pd.read_csv(split_path, sep=";", usecols=["label"], keep_default_na=False, na_values=[])["label"].unique())
                unexpected = split_labels - known_labels
                if unexpected:
                    raise RuntimeError(
                        f"{dataset} rank{rank}: {split_name} split ({split_path}) contains label "
                        f"value(s) {unexpected} not present in the FULL file's label set "
                        f"{known_labels} (read from {split_dir}/{cfg_info['subfolder']}.csv). "
                        f"The full/{split_name} files are out of sync -- re-run "
                        f"prepare_concept_datasets.py and re-transfer {split_dir}/ before retrying."
                    )

            model = build_model(num_labels)

            trainer = Trainer(accelerator="gpu", devices=1, max_epochs=max_epochs_this_rank,
                               gradient_clip_val=1.0,
                               default_root_dir=f"./training_runs/{dataset}/rank{rank}")

            if ckpt_path is not None:
                print(f"{dataset} rank{rank}: found checkpoint at epoch {saved_epoch}, "
                      f"resuming with max_epochs={max_epochs_this_rank}")
                trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
            else:
                print(f"{dataset} rank{rank}: no checkpoint found, training from scratch "
                      f"(max_epochs={max_epochs_this_rank})")
                trainer.fit(model=model, datamodule=datamodule)

            datamodule.setup(stage="test")
            fix_device_attrs(model.model, next(model.model.parameters()).device)

            generated_df = generate_n_traces(model, datamodule, num_traces=num_traces,
                                              use_beam_search=args.use_beam_search, beam_size=args.beam_size)
            out_path = f"generated_{dataset}_rank{rank}.csv"
            generated_df.to_csv(out_path, sep=";", index=False)
            print(f"Wrote {out_path}")
            results[(dataset, rank)] = generated_df

    print("\n========================================================")
    print(f"COMPLETE -- ran {len(results)} (dataset, rank) combinations")
    print("========================================================")
    return results


if __name__ == "__main__":
    main()
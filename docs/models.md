# External models and runtimes (not bundled)

Configure their common root with `scripts/configure_local_paths.py --model-root`.
The default below is `assets/models`. Sources were actually used by the
project; the package does not download or substitute newer releases.

| Asset | Public source and study version | Required local layout |
|---|---|---|
| ChemBERTa | https://huggingface.co/DeepChem/ChemBERTa-100M-MLM | `chemberta_100m_mlm/`, model/config/tokenizer files from that repository |
| TransPolymer | https://github.com/ChangwenXu98/TransPolymer, public repository acquired 2026-08; Git LFS weights in `ckpt/pretrain.pt` | `external_baselines/transpolymer/source/`; copy released Hugging Face model-directory contents to `checkpoint/`; RoBERTa tokenizer files to `tokenizer/roberta-base/` |
| PerioGT | https://github.com/wuyuhui-zju/PerioGT, version 1.0; checkpoint https://zenodo.org/records/17035498 | `external_baselines/periogt/source/`; released light `base.pth` under `checkpoint/pretrained_ckpt/pretrained/light/base.pth`; scaler under `source/PerioGT_common/datasets/pretrain/scaler_all.pkl` |
| GRIN | https://github.com/liugangcode/torch-molecule, v0.1.7 | `external_baselines/grin/source/` containing `torch_molecule/`; no public pretrained initialization is used by the study's supervised GRIN runner |
| Chemprop | https://github.com/chemprop/chemprop; **runtime 1.6.1**, not the separately inspected v2.1.2 source | Installed `chemprop==1.6.1`; no baseline source tree required |

Obtain the ChemBERTa repository with the installed Hugging Face CLI, e.g.
`huggingface-cli download DeepChem/ChemBERTa-100M-MLM --local-dir assets/models/chemberta_100m_mlm`.
Download each external release according to its official README. TransPolymer's
`pretrain.pt` is a released pretrained-model location; the runner expects its
`config.json` and `pytorch_model.bin` in the local `checkpoint/` directory.
Its tokenizer starts from official `FacebookAI/roberta-base` tokenizer assets
and is used by the released `PolymerSmilesTokenizer`, not a replacement tokenizer.

External source licenses remain upstream: TransPolymer/GRIN/PerioGT source
repositories record MIT; the PerioGT model record uses CC-BY-4.0. Confirm source
terms for the actual downloaded version. No external source code or weights
are redistributed by this lightweight package. Exact upstream revision IDs
were not recorded for every repository; version/date descriptions are not
claimed to uniquely identify future changing repository contents.

The first-party baseline adapters preserve the original workflow: TransPolymer
FP64, length 512; Paper104 FP32 embedding forward, length 256; PerioGT periodic
prompt plus the released 1,613-descriptor scaler; GRIN repetition augmentation.
The local GRIN adapter uses tested PyTorch `scatter_reduce` in place of two
unavailable compiled max-aggregation extension calls. These are documented
runtime bridges, not renamed alternative methods. Supplemental TransPolymer
vocabulary assets for OPV/PE-I/PE-II were unavailable, so those external
results are absent rather than approximated with a base vocabulary.

## Optional baseline runtime

The core editable install does not install every external baseline. In a compatible
environment, use the baseline extra and install DGL explicitly for PerioGT:

```bash
python -m pip install -e '.[baselines]'
python -m pip install 'dgl==1.1.3+cu121' --find-links https://data.dgl.ai/wheels/cu121/repo.html
```

The DGL command specifies the archived CUDA 12.1 wheel, not a CPU-only package or
a promise of compatibility with every machine. `dgllife` does not install DGL
transitively. `requirements.txt` records the entire observed dependency stack;
installation on a fresh machine has not been validated for this publication copy.
The upstream source trees and required files in the table must also be acquired.

Paper104's trained representations, frozen heads and adapted models are not bundled
and currently have no public download URL. Training produces them beneath the
configured checkpoint/output roots. Downloading the original ChemBERTa checkpoint
does not provide the A_eqs/A_gcl trained states used in the retained result tables.

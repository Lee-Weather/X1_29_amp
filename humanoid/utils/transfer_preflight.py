"""Pin exp3.4's initial transfer source instead of silently starting from zero."""
import hashlib
import json
from pathlib import Path


def prepare_initial_transfer(args, train_cfg, root):
    template = getattr(train_cfg.runner, 'transfer_manifest', None)
    if not template:
        return
    manifest_path = Path(template.format(LEGGED_GYM_ROOT_DIR=str(root)))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    path = Path(args.ckpt_path) if getattr(args, 'ckpt_path', None) else None
    if path is None or not path.is_file():
        raise ValueError('exp3.4 initial transfer requires an exact --ckpt_path to exp2.4a-cont2 model_26998.pt')
    expected = manifest['transfer_source']['sha256']
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError('exp3.4 transfer checkpoint SHA256 mismatch; do not use pre-axis-fix or unrelated models')
    for filename, info in manifest['files'].items():
        asset = manifest_path.parent / filename
        if hashlib.sha256(asset.read_bytes()).hexdigest() != info['sha256']:
            raise ValueError('exp3.4 reference checksum mismatch: ' + filename)
    # Direct loading already creates fresh policy/estimator optimizers. Enforce
    # a fresh discriminator as well; this applies to training, not playback.
    args.disc_fresh = True
    print('[TRANSFER] verified exp2.4a-cont2 source; fresh discriminator and all optimizers; '
          'additional updates={}'.format(args.max_iterations or train_cfg.runner.max_iterations))

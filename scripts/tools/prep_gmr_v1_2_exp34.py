"""Create B3-only phase/demo assets and pin the post-axis-fix transfer source."""
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'resources/motions/processed'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    sources = [('ref_lib_v1_2_exp33_phase.pt', 'ref_lib_v1_2_exp34_phase.pt'),
               ('amp_v1_2_exp33.pt', 'amp_v1_2_exp34.pt')]
    manifest = dict(experiment='exp3.4', segment_weights={'walk_forward': 1.0}, files={})
    for source, destination in sources:
        library = torch.load(OUT / source, map_location='cpu', weights_only=False)
        forward = library['walk_forward']
        torch.save({'walk_forward': forward}, OUT / destination)
        manifest['files'][destination] = dict(source=source, source_sha256=digest(OUT / source),
            sha256=digest(OUT / destination), frames=len(forward['dof_pos']), fps=float(forward['fps']))
    model = ROOT / 'czy/data/exp2.4a_cont2/model_26998.pt'
    checkpoint = torch.load(model, map_location='cpu', weights_only=False)
    state = checkpoint['model_state_dict']
    if state['actor.0.weight'].shape[1] != 557 or state['actor.6.weight'].shape[0] != 29:
        raise ValueError('transfer source does not match the legacy 98-feature DH policy')
    if int(checkpoint['iter']) != 26997:
        raise ValueError('unexpected exp2.4a-cont2 checkpoint iteration')
    manifest['transfer_source'] = dict(file='model_26998.pt', local_path=model.relative_to(ROOT).as_posix(),
        experiment='exp2.4a-cont2', sha256=digest(model), bytes=model.stat().st_size,
        internal_iter=int(checkpoint['iter']), completed_iterations=int(checkpoint['iter']) + 1,
        fresh_discriminator=True, fresh_optimizers=True)
    (OUT / 'v1_2_exp34_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()

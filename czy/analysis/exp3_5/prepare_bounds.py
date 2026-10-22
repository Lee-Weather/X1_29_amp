"""Build exp3.5 upper-body bounds from the verified healthy-model replay."""
import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
reference=ROOT/'resources/motions/processed/ref_lib_v1_2_exp34_phase.pt'
source=ROOT/'czy/analysis/exp3_4/preflight_20261021_144449/b3_initial_isaac_diag.csv'
lib=torch.load(reference,weights_only=False,map_location='cpu')['walk_forward']
with source.open(newline='',encoding='utf-8-sig') as stream:rows=list(csv.DictReader(stream))
limits={j.attrib['name']:(float(j.find('limit').attrib['lower']),float(j.find('limit').attrib['upper']))
 for j in ET.parse(ROOT/'resources/robots/x1/urdf/X1_29DOF_physically_mirrored.urdf').getroot().findall('joint') if j.find('limit') is not None}
bounds={}
for i,name in enumerate(lib['dof_names']):
 if not any(p in name for p in ('lumbar','shoulder','elbow','wrist')):continue
 v=np.array([float(r['pos_des_raw_'+name]) for r in rows]);q=lib['dof_pos'][:,i].numpy();lo,hi=limits[name]
 bounds[name]=[max(lo,min(float(np.quantile(v,.005)),float(q.min()))-.05),min(hi,max(float(np.quantile(v,.995)),float(q.max()))+.05)]
result={'source':'model_26998 B3 initial nominal replay; 0.5/99.5 percentiles union B3 range, margin .05 rad, intersect URDF hard bounds','bounds':bounds}
(ROOT/'resources/motions/processed/v1_2_exp35_target_bounds.json').write_text(json.dumps(result,indent=2)+'\n')
provenance={'source_csv_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),'bounded_joints':len(bounds)}
(ROOT/'czy/analysis/exp3_5/bounds_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(provenance)

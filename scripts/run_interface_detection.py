#!/usr/bin/env python3
"""Run an explicitly bound geometry-aware interface model on one raw image."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tfree_detection.expert_supervision import contained_path,normalize_image,sha256_file,write_new_json
from tfree_detection.interface_release import InterfaceResearchRuntime
from tfree_detection.preprocessing import load_grayscale

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True);p.add_argument("--binding",required=True)
    p.add_argument("--output",required=True);p.add_argument("--device",default="cpu")
    p.add_argument("--frame-index",type=int);p.add_argument("--dataset-path");p.add_argument("--projection",choices=["mean","sum","max"])
    a=p.parse_args();source=contained_path(ROOT,a.source);output=contained_path(ROOT,a.output)
    if output.exists():raise FileExistsError("Refusing to overwrite detection output")
    image=load_grayscale(source,frame_index=a.frame_index,dataset_path=a.dataset_path,projection=a.projection)
    runtime=InterfaceResearchRuntime(ROOT,a.binding,device=a.device,allow_experimental=True)
    result=runtime.predict(normalize_image(image));output.mkdir(parents=True)
    path=output/"prediction.npz"
    with path.open("xb") as stream:np.savez_compressed(stream,**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
    write_new_json(output/"receipt.json",{**{k:v for k,v in result.items() if not isinstance(v,np.ndarray)},
        "source":str(source.relative_to(ROOT)),"source_sha256":sha256_file(source),
        "binding":a.binding,"binding_sha256":sha256_file(contained_path(ROOT,a.binding)),
        "arrays_sha256":sha256_file(path),"frame_index":a.frame_index,"dataset_path":a.dataset_path})
    print(output,result["decision"],flush=True)


if __name__=="__main__":main()

import argparse
from pathlib import Path

import torch


ENCODEC_PATH = "https://dl.fbaipublicfiles.com/encodec/v0/encodec_24khz-d7cc33bc.th"

parser = argparse.ArgumentParser()
parser.add_argument("--download-dir", type=str, required=True)

if __name__ == "__main__":
    args = parser.parse_args()
    out_dir = Path(args.download_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    print(" ### Downloading EnCodec weights...")
    state_dict = torch.hub.load_state_dict_from_url(
        ENCODEC_PATH,
        map_location="cpu",
        check_hash=True
    )
    with open(out_dir / Path(ENCODEC_PATH).name, "wb") as fout:
        torch.save(state_dict, fout)

    print("Done.")
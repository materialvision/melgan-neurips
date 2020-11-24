#!/usr/bin/env python3
import subprocess
from pathlib import Path

import click
from joblib import Parallel, delayed


def convert(input, output, samplerate):
    command = ["ffmpeg", "-i", str(input), "-y", "-ar", str(samplerate), str(output)]
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT,)
    except subprocess.CalledProcessError as exc:
        print(f"Return code: {exc.returncode}\n", exc.output)
        raise


@click.command()
@click.option("--in-dir", "-i", required=True)
@click.option("--out-dir", "-o", required=True)
@click.option("--samplerate", "-s", type=int, default=22050)
def main(in_dir, out_dir, samplerate):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    in_paths = list(Path(in_dir).rglob("*.*"))
    out_paths = [out_dir / in_path.relative_to(in_dir) for in_path in in_paths]

    for sub_dir in set(out_path.parent for out_path in out_paths):
        sub_dir.mkdir(exist_ok=True, parents=True)

    Parallel(n_jobs=-1, verbose=2)(
        delayed(convert)(in_path, out_path, samplerate)
        for in_path, out_path in zip(in_paths, out_paths)
    )


if __name__ == "__main__":
    main()

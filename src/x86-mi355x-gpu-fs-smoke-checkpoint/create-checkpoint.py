# Copyright (c) 2026 The Regents of The University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer; redistributions in
# binary form must reproduce the above copyright notice, this list of
# conditions and the following disclaimer in the documentation and/or other
# materials provided with the distribution; neither the name of the copyright
# holders nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written
# permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Create the MI355X GPUFS smoke-kernel checkpoint resource."""

import argparse
import runpy
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gem5-root",
    type=Path,
    required=True,
    help="Path to a gem5 checkout containing the MI355X GPUFS config",
)
parser.add_argument(
    "--resource-directory",
    type=Path,
    default=Path(__file__).resolve().parent / "resources",
)
parser.add_argument(
    "--mode",
    choices=("create-loader", "create-kernel"),
    required=True,
)
parser.add_argument(
    "--loader-checkpoint",
    type=Path,
    help="Initialized loader checkpoint produced by the loader stage",
)
parser.add_argument(
    "--kernel-binary",
    type=Path,
    help="gfx950 code object to load before the final checkpoint",
)
parser.add_argument(
    "--checkpoint-output",
    type=Path,
    required=True,
    help="Directory in which to save the generated checkpoint",
)
args = parser.parse_args()

gem5_root = args.gem5_root.resolve()
config = gem5_root / "tests" / "gem5" / "gpu" / "configs" / "mi355x_gpu.py"
if not config.is_file():
    parser.error(f"MI355X GPUFS config not found: {config}")
sys.path.insert(0, str(config.parent))

source_directory = Path(__file__).resolve().parent
config_args = [
    str(config),
    "--resource-directory",
    str(args.resource_directory),
    "--mode",
    args.mode,
    "--checkpoint-output",
    str(args.checkpoint_output),
]

if args.mode == "create-loader":
    if args.loader_checkpoint or args.kernel_binary:
        parser.error(
            "--loader-checkpoint and --kernel-binary are only valid for the "
            "kernel stage"
        )
    config_args.extend(
        (
            "--gpu-application-binary",
            str(source_directory / "hip-checkpoint-runner.py"),
        )
    )
else:
    if not args.loader_checkpoint or not args.kernel_binary:
        parser.error(
            "the kernel stage requires --loader-checkpoint and --kernel-binary"
        )
    config_args.extend(
        (
            "--checkpoint-directory",
            str(args.loader_checkpoint),
            "--gpu-kernel-binary",
            str(args.kernel_binary),
        )
    )

sys.argv = config_args
runpy.run_path(str(config), run_name="__main__")

import argparse, sys, os
from utils import logger

from backend import prepare


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=str, help="Path to the Model.")
    parser.add_argument(
        "--mode", required=True, type=str, help="Mode i.e u64|i64|float."
    )
    parser.add_argument(
        "--scale", required=True, type=int, help="Scale for computation", default=0
    )
    parser.add_argument(
        "--backend", required=True, type=str, help="Backend to compile model to."
    )
    # parser.add_argument(
    #     "--generate", required=True, type=str, choices=["code", "executable"]
    # )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Prepare a BackendRep for the Model.
    backendrep = prepare(args.path)

    # Export the Model as Secfloat and writes to a cpp file
    backendrep.export_model(args.mode, args.scale, args.backend)

    # if args.generate == "executable":
    #     logger.info("Starting Compilation.")
    #     os.system(f"lib_secfloat/compile_secfloat.sh {args.path[:-5]}_secfloat.cpp")
    #     logger.info(f"Output Binary generated : {args.path[:-5]}_secfloat.out")


if __name__ == "__main__":
    main()

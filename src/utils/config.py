"""
This file handles the configuration yaml file
"""
import yaml, argparse, munch

parser = argparse.ArgumentParser(description="")
parser.add_argument("--config_path", default="config.yml")
args = parser.parse_args()
config = munch.munchify(yaml.safe_load(open(args.config_path, "r")))

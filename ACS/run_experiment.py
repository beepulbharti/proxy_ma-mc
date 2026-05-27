# primary imports
import os
import pickle
import argparse

# local imports
from utils import run_one_seed

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        choices=["ACSPubcov_no_sex", "ACSIncome_no_race"],
        help="Experiment name",
    )

    parser.add_argument(
        "--classifier",
        type=str,
        required=True,
        help="Classifier to use",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    seeds = [1, 78, 89, 10, 567]

    exp_name = args.exp
    classifier = args.classifier
    min_group_size = 0

    results_dir = os.path.join("results", exp_name)
    os.makedirs(results_dir, exist_ok=True)

    results = {}

    for k, seed in enumerate(seeds):
        results[k] = run_one_seed(
            seed=seed,
            exp_name=exp_name,
            min_group_size=min_group_size,
            classifier=classifier,
        )

    with open(os.path.join(results_dir, classifier + "_mc_results.pkl"), "wb") as f:
        pickle.dump(results, f)

if __name__ == "__main__":
    main()
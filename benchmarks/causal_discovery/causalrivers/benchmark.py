import sys
import time

from hydra import compose, initialize
from omegaconf import DictConfig

from tools.scoring_tools import score
from tools.tools import (
    benchmarking,
    load_joint_samples,
    save_run,
    standard_preprocessing,
)


# Example script to benchmark causal discovery methods.
def main(cfg: DictConfig):

    if cfg.method.name == "tgraank":
        from tools.baseline_methods import var_tgraank as cd_method
    elif cfg.method.name == "var":
        from tools.baseline_methods import var_baseline as cd_method
    else:
        raise ValueError("SPECIFY AND LOAD YOUR OWN METHOD HERE")

    start = time.time()
    print("Loading data...")
    # First, we load the full dataset from path and preprocess according to config..
    test_data, test_labels = load_joint_samples(
        cfg, preprocessing=standard_preprocessing if cfg.dt_preprocess else None
    )
    selection = {
        "restrict_to": cfg.restrict_to,
        "sample_count": len(test_data),
        "nodes": [[str(c) for c in sample.columns] for sample in test_data],
        "rows": [len(sample) for sample in test_data],
        "time_ranges": [[str(sample.index[0]), str(sample.index[-1])] for sample in test_data],
        "capture": "before method invocation",
    }

    # Then we simply run the specified method on all samples.
    print("Performing Causal Discovery...")
    preds = benchmarking(test_data, cfg, cd_method)

    out = score(
        preds,
        test_labels,
        remove_autoregressive=cfg.remove_diagonal,
        name=cfg.method.name,
    )
    stop_time = time.time() - start
    print(out)

    if cfg.save_full_out:
        print("Saving...")
        run_path = save_run(out, stop_time, preds, cfg)
        from tools.provenance import environment_record, write_run_record

        write_run_record(
            run_path,
            cfg,
            "python benchmark.py " + " ".join(sys.argv[1:]),
            selection,
            environment_record(),
        )
    print("Done", stop_time)


if __name__ == "__main__":
    # 🚀 The Fix: Collect runtime CLI arguments directly from the terminal stream
    cli_overrides = sys.argv[1:]

    with initialize(version_base=None, config_path="config"):
        # Explicitly forward the overrides array into Hydra's composition boundary
        _cfg = compose(config_name="benchmark.yaml", overrides=cli_overrides)
        print("Resolved Configuration Profile:")
        print(_cfg)
    main(_cfg)

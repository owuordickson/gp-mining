import datetime

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
    
    
    if cfg.method.name == "var":
        from tools.baseline_methods import var_baseline as cd_method
    elif cfg.method.name == "tgraank":
        from tools.baseline_methods import var_tgraank as cd_method
    else:
        raise ValueError("SPECIFY AND LOAD YOUR OWN METHOD HERE")

    start = datetime.datetime.now()
    print("Loading data...")
    # First, we load the full dataset from path and preprocess according to config..
    test_data, test_labels = load_joint_samples(
        cfg, preprocessing=standard_preprocessing if cfg.dt_preprocess else None
    )

    # Then we simply run the specified method on all samples.
    print("Performing Causal Discovery...")
    preds = benchmarking(test_data, cfg, cd_method)

    out = score(
        preds,
        test_labels,
        remove_autoregressive=cfg.remove_diagonal,
        name=cfg.method.name,
    )
    stop_time = datetime.datetime.now() - start
    print(out)

    if cfg.save_full_out:
        print("Saving...")
        save_run(out, stop_time, preds, cfg)
    print("Done", stop_time)


if __name__ == "__main__":
    with initialize(version_base=None, config_path="config"):
        _cfg = compose(config_name='benchmark.yaml')
        print(_cfg)

    main(_cfg)

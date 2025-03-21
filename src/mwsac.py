import argparse
from copy import deepcopy

import numpy as np

from xuance.common import get_configs
from xuance.environment import make_envs
from xuance.torch.agents import MASAC_Agents
from xuance.torch.utils.operations import set_seed
import wandb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="./src/configs/masac",
                      help="配置文件路径")
    parser.add_argument("--config_name", type=str, default='mw',
                      help="配置文件名称,如果指定则会在config_path下寻找该文件")
    return parser.parse_args()

def run(args):
    if args.config_name is not None:
        config_file = f"{args.config_path}/{args.config_name}.yaml"
    else:
        config_file = args.config_path
    configs_dict = get_configs(file_dir=config_file)
    configs = argparse.Namespace(**configs_dict)

    set_seed(configs.seed)
    envs = make_envs(configs)  # Make parallel environments.
    Agent = MASAC_Agents(config=configs, envs=envs)  # Create a PPO agent from XuanCe.

    if configs.benchmark:

        def env_fn():
            configs_test = deepcopy(configs)
            configs_test.parallels = configs_test.test_episode
            return make_envs(configs_test)

        train_steps = configs.running_steps // configs.parallels
        eval_interval = configs.eval_interval // configs.parallels
        test_episode = configs.test_episode
        num_epoch = int(train_steps / eval_interval)

        test_scores = Agent.test(env_fn, test_episode)
        Agent.save_model(model_name="best_model.pth")
        best_scores_info = {
            "mean": np.mean(test_scores),
            "std": np.std(test_scores),
            "step": Agent.current_step,
        }
        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            Agent.train(eval_interval)
            test_scores = Agent.test(env_fn, test_episode)

            if np.mean(test_scores) > best_scores_info["mean"]:
                best_scores_info = {
                    "mean": np.mean(test_scores),
                    "std": np.std(test_scores),
                    "step": Agent.current_step,
                }
                # save best model
                Agent.save_model(model_name="best_model.pth")
        # end benchmarking
        print(
            "Best Model Score: %.2f, std=%.2f"
            % (best_scores_info["mean"], best_scores_info["std"])
        )
        wandb.run.summary["best_score"] = best_scores_info["mean"]
    else:
        if configs.test:

            def env_fn():
                configs.parallels = configs.test_episode
                return make_envs(configs)

            Agent.load_model(path=Agent.model_dir_load)
            scores = Agent.test(env_fn, configs.test_episode)
            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
            print("Finish testing.")
        else:
            Agent.train(configs.running_steps // configs.parallels)
            Agent.save_model("final_train_model.pth")
            print("Finish training!")

    Agent.finish()

    import requests

    requests.get("https://api.day.app/Ya5CADvAuDWf5NR4E8ZGt5/masac训练完成")


if __name__ == "__main__":
    args = parse_args()
    """
    get_arguments(
        method=parser.method,
        env=parser.env,
        env_id=parser.env_id,
        config_path=parser.config,
        parser_args=parser,
    )
    """
    run(args)

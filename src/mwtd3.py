import argparse

from xuance.common import get_configs
from xuance.environment import make_envs
from xuance.torch.agents import MATD3_Agents

if __name__ == "__main__":  # 新增保护块
    configs_dict = get_configs(file_dir="./xuance/configs/matd3/multiwalker.yaml")
    configs = argparse.Namespace(**configs_dict)

    envs = make_envs(configs)  # Make parallel environments.
    Agent = MATD3_Agents(config=configs, envs=envs)  # Create a PPO agent from XuanCe.
    Agent.train(
        configs.running_steps // configs.parallels
    )  # Train the model for numerous steps.
    Agent.save_model("final_train_model.pth")  # Save the model to model_dir.
    Agent.finish()  # Finish the training.

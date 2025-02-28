from xuance.common import Optional
from xuance.environment.multi_agent_env.mpe import MPE_Env
from xuance.environment.utils import EnvironmentDict

REGISTRY_MULTI_AGENT_ENV: Optional[EnvironmentDict] = {
    "mpe": MPE_Env,
}

try:
    from xuance.environment.multi_agent_env.drones import Drones_MultiAgentEnv

    REGISTRY_MULTI_AGENT_ENV["Drones"] = Drones_MultiAgentEnv
except Exception as error:
    REGISTRY_MULTI_AGENT_ENV["Drones"] = str(error)

try:
    from xuance.environment.multi_agent_env.football import GFootball_Env

    REGISTRY_MULTI_AGENT_ENV["Football"] = GFootball_Env
except Exception as error:
    REGISTRY_MULTI_AGENT_ENV["Football"] = str(error)

try:
    from xuance.environment.multi_agent_env.robotic_warehouse import RoboticWarehouseEnv

    REGISTRY_MULTI_AGENT_ENV["RoboticWarehouse"] = RoboticWarehouseEnv
except Exception as error:
    REGISTRY_MULTI_AGENT_ENV["RoboticWarehouse"] = str(error)

try:
    from xuance.environment.multi_agent_env.starcraft2 import StarCraft2_Env

    REGISTRY_MULTI_AGENT_ENV["StarCraft2"] = StarCraft2_Env
except Exception as error:
    REGISTRY_MULTI_AGENT_ENV["StarCraft2"] = str(error)

try:
    from xuance.environment.multi_agent_env.multiwalker import MultiWalkerEnv

    REGISTRY_MULTI_AGENT_ENV["MultiWalker"] = MultiWalkerEnv
except Exception as error:
    REGISTRY_MULTI_AGENT_ENV["MultiWalker"] = str(error)
    raise error

__all__ = [
    "REGISTRY_MULTI_AGENT_ENV",
]

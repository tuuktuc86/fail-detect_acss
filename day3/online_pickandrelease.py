
import set_env
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
import ppo

def main():
    """메인 함수"""
    # 환경 갯수(1개로 고정)
    

    env = set_env.make_env()
    obs, _ = env.reset()
    device = env.device
    print(f"Environment reset. Number of environments: {env.unwrapped.num_envs}")
    memory = RandomMemory(memory_size=96, num_envs=env.num_envs, device=device)
    models = {}
    models["policy"] = ppo.Shared(env.observation_space, env.action_space, device)
    models["value"] = models["policy"]  # same instance: shared model


    # configure and instantiate the agent (visit its documentation to see all the options)
    # https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
    
    agent = PPO(models=models,
                memory=memory,
                cfg=ppo.set_config(env = env, device=device),
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)


    # configure and instantiate the RL trainer
    cfg_trainer = {"timesteps": 67200, "headless": True}
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

    # start training
    trainer.train() 
# 메인 함수 실행
if __name__ == "__main__":
    main()
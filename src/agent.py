"""
Agent definition
"""
import torch 
from memory import *
from procgen_wrappers import *
from utils import save_video, save_model, print_rewards
from casual import *
from a2c import *
from torch import nn
from model import *
from tqdm import tqdm 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, mod):
        self.mod = mod

    def play(self, env_name, ne, start_level, num_levels, total_steps):

        env = make_env(
            n_env=ne,
            name = env_name,
            start_level= start_level,
            num_levels=num_levels, 
        )


        eval_env = make_env(  
            n_env=1,
            name = env_name,
            start_level= num_levels,
            num_levels= start_level, 
        )

        match self.mod:
            case "a2c":
                model = A2C(Model(3, 128), 128, env.action_space.n, .5, device)
                model, (steps, rewards) = self.train_model(
                    model = model, 
                    env=env, 
                    device = device, 
                    num_epochs= 3, 
                    batch_size=512, 
                    lr = 1e-4, 
                    eps = 1e-4,
                    num_steps=256,
                    total_steps = total_steps,
                    eval_env = eval_env, 
                )

                print_rewards(steps, rewards, f"./{self.mod}_{env_name}")
                save_model(model, f"./{self.mod}_{env_name}.pt")

            case "random":
                model = RandomAgent(env.action_space)

                model, (steps, rewards) = self.train_model(
                    model = model, 
                    env=env, 
                    device = device, 
                    num_epochs= 3, 
                    batch_size=512, 
                    lr = 1e-4, 
                    eps = 1e-4,
                    num_steps=256,
                    total_steps = total_steps,
                    eval_env = eval_env, 
                )

                print_rewards(steps, rewards, f"./{self.mod}_{env_name}")
            case _ :
                print("Not implemented")

        video_env = make_env(
            n_env =1,
            name = env_name,
            start_level = num_levels,
            num_levels= 0, #start_level
        )

        obs = video_env.reset() 
        total_reward, _ = self.evaluate_model(
            model=model,
            eval_env=video_env,
            obs = obs, 
            num_steps =1024,
            video = True,
            video_fp = f"./{self.mod}_{env_name}.mp4"
        )
        print(total_reward)

    def train_model(self, model, env, device, num_epochs, batch_size, lr, eps, num_steps, total_steps, eval_env=None):

        steps = []
        train_rew = []
        test_rew = []
        obs = env.reset()
        eval_obs = eval_env.reset()
        step=0
        match self.mod:
            case "a2c":
                model.to(device=device)

                optimizer = torch.optim.Adam(model.parameters(), lr=lr, eps=eps)

                memory = Memory(env.observation_space.shape, num_steps, env.num_envs, device)
                print("start train")
                pbar = tqdm(desc="train loop",  total= total_steps)
                while step < total_steps:

                    model.eval()
                    for _ in range(num_steps):
                        #populate memory
                        action, value, _ = model.act(obs)
                        next_obs, rew, done, info =env.step(action)
                        memory.store(obs, action,rew, done, info, value)
                        obs = next_obs

                    steps.append(step)
                    train_rew.append(memory.get_reward())

                    test_r, eval_obs = self.evaluate_model(model, eval_env, eval_obs, num_steps=num_steps)
                    test_rew.append(test_r)

                    step += env.num_envs * num_steps
                    pbar.update(step)

                    _, value,_ = model.act(obs)
                    memory.store_last(obs, value)
                    
                    memory.compute_return_advantage()
                    #policy optimization
                    model.train()

                    for ep in range(num_epochs):
                        generator = memory.get_generator(batch_size)

                        for batch in generator:
                            b_obs, b_act, b_ret, b_del, b_adv, b_val = batch 
                            policy, value = model(b_obs)
                            
                            loss = model.loss(batch, policy, value)
                            loss.backward()

                            torch.nn.utils.clip_grad_norm_(model.parameters(), model.grad_eps)

                            #torch.nn.utils.crip_grad_norm()
                            optimizer.step()
                            optimizer.zero_grad()
                pbar.close()
                print("end train")
                rewards = (torch.stack(train_rew).cpu().detach().numpy(), torch.stack(test_rew).cpu().detach().numpy())
            case "random":
                print("start ")
                pbar = tqdm(desc="train loop",  total= total_steps)
                while step <total_steps:
                    action = model.act(obs)
                    next_obs, rew, done, info = env.step(action)

                    steps.append(step)
                    train_rew.append(rew)
                    step += env.num_envs*num_steps 
                pbar.close()
                print("end")
                rewards = train_rew
            case _:
                print("not implemented")

        return model, (steps, rewards)

    def evaluate_model(self, model, eval_env, obs, num_steps=256, video=False, video_fp=None):
        frames=[]
        total_reward=[]
        match self.mod:
            case "a2c":
                model.eval()
                for _ in range(num_steps):
                    action, value,_= model.act(obs)
                    obs, rew, done, _ = eval_env.step(action)
                    total_reward.append(torch.Tensor(rew))

                    if video: 
                        frame = (torch.Tensor(eval_env.render(mode="rgb_array"))*255.0).byte()
                        frames.append(frame)
                if video:
                    save_video(frames, video_fp)

            case "random":
                for _ in range(num_steps):
                    action = model.act(obs)
                    obs,rew,done, info = eval_env.step(action)
                    total_reward.append(torch.Tensor(rew))
                    if video:
                        frame = (torch.Tensor(eval_env.render(mode="rgb_array"))*255.0).byte()
                        frames.append(frame)
                if video:
                    save_video(frames, video_fp)
            case _:
                print("not implemented")
            
        return torch.stack(total_reward).sum(0), obs

    def replay(self, file, game, num_levels, start_level):

        rep_env = make_env(
            n_env=1,
            name = game,
            start_level = num_levels,
            num_levels = start_level,
        )
        
        model = A2C(Model(3, 128), 128, rep_env.action_space.n, .5, device)
        model.load_state_dict(torch.load(file))
        model.to(device)
        obs = rep_env.reset()
        total_rew, _ = self.evaluate_model(
            model=model,
            eval_env = rep_env,
            obs = obs,
            num_steps=1024,
            video = True,
            video_fp = f"./rep_{game}.mp4"
        )
        print(total_rew)

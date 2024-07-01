"""
Utility functions to print graphs,save models and videos
"""
import torch 
import imageio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def save_video(frames, filepath):
    frames = torch.stack(frames)
    imageio.mimsave(filepath, frames, fps=25)

def print_rewards(steps, rewards, path):
    mean=None
    std=None
    if isinstance(rewards, tuple):
        train = rewards[0]
        test = rewards[1]

        mean_train_rew = np.zeros(len(steps))
        mean_test_rew = np.zeros(len(steps))
        for x in range(len(steps)):
            mean_train_rew[x] = train[x].mean()
            mean_test_rew[x] = test[x].mean()
        mean= pd.DataFrame(
            {
                'steps': steps,
                'rewards_train':mean_train_rew,
                'rewards_test':mean_test_rew,
            }
        )
    else:
        mean_train_rew = np.zeros(len(steps))
        for x in range(len(steps)):
            mean_train_rew[x] = rewards[x].mean()
        mean= pd.DataFrame(
            {
                'steps': steps,
                'rewards':mean_train_rew, 
            }
        )
    sns.set(style="darkgrid", font_scale=1)
    fig, ax = plt.subplots(figsize=(15,15))#,figsize=(15,15))
    fig1, ax1 = plt.subplots(figsize=(15,15))
    ax.relim(visible_only=True)
    ax.set_xlabel('steps')
    ax.set_ylabel('rewards')
    if isinstance(rewards, tuple):
        sns.lineplot(ax = ax, data= mean, x="steps", y="rewards_train", legend='brief', label="mean_train")
        sns.lineplot(ax = ax, data= mean, x="steps", y="rewards_test", legend='brief', label="mean_test")
    else:
        sns.lineplot(ax = ax, data= mean, x="steps", y="rewards", legend='brief', label="mean")
    fig.savefig(f"{path}_train.png")


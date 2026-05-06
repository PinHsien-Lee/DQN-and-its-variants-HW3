import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Gridworld import Gridworld
import random
from collections import deque
import copy

def get_state(game):
    state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 50.0
    return torch.from_numpy(state_).float()

def train():
    game = Gridworld(size=4, mode='static')
    
    l1, l2, l3, l4 = 64, 150, 100, 4
    model = nn.Sequential(
        nn.Linear(l1, l2),
        nn.ReLU(),
        nn.Linear(l2, l3),
        nn.ReLU(),
        nn.Linear(l3, l4)
    )
    
    loss_fn = nn.MSELoss()
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    gamma = 0.9
    epsilon = 1.0
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    
    epochs = 1000
    losses = []
    
    # Experience Replay Buffer
    mem_size = 1000
    batch_size = 200
    replay = deque(maxlen=mem_size)
    
    print("Training Naive DQN with Experience Replay on static mode...")
    
    for i in range(epochs):
        game = Gridworld(size=4, mode='static')
        state1 = get_state(game)
        status = 1
        
        while status == 1:
            qval = model(state1)
            qval_ = qval.data.numpy()
            
            if random.random() < epsilon:
                action_ = np.random.randint(0, 4)
            else:
                action_ = np.argmax(qval_)
                
            action = action_set[action_]
            game.makeMove(action)
            
            state2 = get_state(game)
            reward = game.reward()
            
            # Check if game is over (pit or goal)
            if reward != -1:
                status = 0
                
            done = True if status == 0 else False
            replay.append((state1, action_, reward, state2, done))
            
            state1 = state2
            
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                
                state1_batch = torch.cat([m[0] for m in minibatch])
                action_batch = torch.Tensor([m[1] for m in minibatch])
                reward_batch = torch.Tensor([m[2] for m in minibatch])
                state2_batch = torch.cat([m[3] for m in minibatch])
                done_batch = torch.Tensor([m[4] for m in minibatch])
                
                Q1 = model(state1_batch)
                with torch.no_grad():
                    Q2 = model(state2_batch)
                
                Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2, dim=1)[0])
                X = Q1.gather(dim=1, index=action_batch.long().unsqueeze(dim=1)).squeeze()
                
                loss = loss_fn(X, Y.detach())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                
        if epsilon > 0.1:
            epsilon -= (1/epochs)
            
        if (i+1) % 100 == 0:
            print(f"Epoch {i+1}/{epochs} completed. Avg Loss: {np.mean(losses[-100:]) if losses else 0:.4f}")
            
    print("Training finished!")
    return model

if __name__ == "__main__":
    trained_model = train()

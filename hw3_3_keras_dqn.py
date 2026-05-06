import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random
from Gridworld import Gridworld

def get_state(game):
    # Flatten the state and add minor noise
    state_ = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64) / 50.0
    return state_.astype(np.float32)

def build_model():
    inputs = layers.Input(shape=(64,))
    x = layers.Dense(150, activation='relu')(inputs)
    x = layers.Dense(100, activation='relu')(x)
    outputs = layers.Dense(4, activation='linear')(x)
    return keras.Model(inputs=inputs, outputs=outputs)

def train_keras_dqn():
    print("Training Keras DQN on random mode with Training Tips...")
    
    # Training Tips: Learning Rate Scheduling and Gradient Clipping
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=2000,
        decay_rate=0.9
    )
    
    # clipnorm=1.0 applies Gradient Clipping
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
    loss_fn = keras.losses.MeanSquaredError()
    
    model = build_model()
    target_model = build_model()
    target_model.set_weights(model.get_weights())
    
    gamma = 0.9
    epsilon = 1.0
    epsilon_min = 0.1
    epochs = 2000
    sync_freq = 100
    
    mem_size = 2000
    batch_size = 200
    replay = deque(maxlen=mem_size)
    
    action_set = {0: 'u', 1: 'd', 2: 'l', 3: 'r'}
    losses = []
    step_count = 0
    
    for i in range(epochs):
        # Using 'random' mode as requested for HW3-3
        game = Gridworld(size=4, mode='random')
        state1 = get_state(game)
        status = 1
        
        while status == 1:
            step_count += 1
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action_ = np.random.randint(0, 4)
            else:
                qval = model(state1, training=False)
                action_ = np.argmax(qval[0])
                
            action = action_set[action_]
            game.makeMove(action)
            
            state2 = get_state(game)
            reward = game.reward()
            
            if reward != -1:
                status = 0
                
            done = True if status == 0 else False
            replay.append((state1, action_, reward, state2, done))
            
            state1 = state2
            
            # Experience Replay Training Step
            if len(replay) > batch_size:
                minibatch = random.sample(replay, batch_size)
                
                state1_batch = np.vstack([m[0] for m in minibatch])
                action_batch = np.array([m[1] for m in minibatch])
                reward_batch = np.array([m[2] for m in minibatch])
                state2_batch = np.vstack([m[3] for m in minibatch])
                done_batch = np.array([m[4] for m in minibatch])
                
                # Double DQN logic in Keras
                q1 = model(state1_batch, training=False).numpy()
                q2_curr = model(state2_batch, training=False).numpy()
                q2_target = target_model(state2_batch, training=False).numpy()
                
                best_actions = np.argmax(q2_curr, axis=1)
                
                # Target Q-value = r + gamma * Q_target(s', argmax Q(s', a))
                target_q_values = reward_batch + gamma * (1 - done_batch) * q2_target[np.arange(batch_size), best_actions]
                
                # Update Q1 with the calculated target for the taken actions
                y_true = q1.copy()
                y_true[np.arange(batch_size), action_batch] = target_q_values
                
                with tf.GradientTape() as tape:
                    y_pred = model(state1_batch, training=True)
                    loss = loss_fn(y_true, y_pred)
                    
                # Calculate gradients and apply them (with clipping via the optimizer)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                
                losses.append(loss.numpy())
                
            if step_count % sync_freq == 0:
                target_model.set_weights(model.get_weights())
                
        # Epsilon decay
        if epsilon > epsilon_min:
            epsilon -= (1.0 / epochs)
            
        if (i+1) % 200 == 0:
            avg_loss = np.mean(losses[-200:]) if losses else 0
            curr_lr = float(optimizer.learning_rate(optimizer.iterations))
            print(f"Epoch {i+1}/{epochs} - Avg Loss: {avg_loss:.4f} - LR: {curr_lr:.6f} - Epsilon: {epsilon:.2f}")

    print("Keras Training Finished!")
    return model, losses

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    trained_model, losses = train_keras_dqn()
    
    plt.figure(figsize=(10,5))
    
    def moving_average(a, n=100):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    plt.plot(moving_average(losses), label='Keras DQN Loss')
    plt.title("Keras DQN Training Loss (Random Mode, Smoothed)")
    plt.xlabel("Training Steps")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.savefig("keras_dqn_loss.png")
    plt.close()
    
    print("Saved training loss plot to keras_dqn_loss.png")

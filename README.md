Project Overview

The main goal of this project is to create a DQL-based agent that learns to trade a given stock over a period. The agent is trained using historical HFT data and is tested on unseen data to evaluate its performance. The agent is designed to handle high-frequency trading environments with detailed order book information.

Tabular Q-Learning:

	•	State Representation: In Tabular Q-Learning, the state space is typically discrete and finite, allowing each state-action pair to be stored explicitly in a table (Q-table). The Q-table maintains the estimated value (Q-value) of taking an action from a given state.
	•	Scalability: Tabular Q-Learning is limited by the size of the state space. As the state space grows, the Q-table becomes impractically large, leading to inefficient learning and high memory consumption. This approach is feasible only for problems with a small and well-defined state space.
	•	Generalization: Tabular Q-Learning does not generalize well to unseen states, as it requires the agent to visit every state-action pair to learn effectively.

Deep Q-Learning (DQL):

	•	State Representation: Deep Q-Learning overcomes the limitations of Tabular Q-Learning by using a deep neural network (DNN) to approximate the Q-function. Instead of storing Q-values in a table, the DNN learns to estimate Q-values based on a continuous or large discrete state space.
	•	Scalability: DQL can handle high-dimensional state spaces (such as those encountered in stock trading) where tabular methods would fail. This makes it suitable for complex environments with large, continuous, or unbounded state spaces.
	•	Generalization: The use of a neural network allows DQL to generalize across similar states, making it more efficient in environments where the agent cannot possibly explore every state-action pair. This generalization is particularly beneficial in trading environments where the state space is vast due to various market conditions and indicators.

Neural Network Structure Used in This Code

In this project, the Q-function is approximated using a deep neural network (DNN) with the following structure:

	1.	Input Layer:
	•	Features: The input layer takes in features that represent the current state of the environment. These features include various technical indicators such as RSI, MACD, Bollinger Bands, and other market data like price levels and trading volume.
	•	Dimensionality: The number of input neurons corresponds to the number of features used to describe the state.
	2.	Hidden Layers:
	•	The network consists of multiple fully connected hidden layers (dense layers). These hidden layers capture the complex, non-linear relationships between the state features and the Q-values. The depth and size of these hidden layers can be tuned based on the complexity of the task and the amount of data available.
	•	Activation Functions: Rectified Linear Units (ReLU) are typically used as the activation functions in the hidden layers to introduce non-linearity, allowing the network to learn intricate patterns in the data.
	3.	Output Layer:
	•	The output layer consists of a single neuron for each possible action (e.g., buy, sell, hold). Each output neuron represents the estimated Q-value for taking that action in the given state.
	•	The network produces a Q-value for each action, and the action with the highest Q-value is selected as the optimal action at each step.

Training Process

The neural network is trained using a variant of Q-learning called Deep Q-Learning with Experience Replay. Here’s how it works:

	1.	Experience Replay: The agent’s experiences (state, action, reward, next state) are stored in a replay buffer. During training, random samples from this buffer are used to update the network, which helps to break the correlation between consecutive experiences and stabilize learning.
	2.	Target Network: A separate target network is used to compute the target Q-values during training. This target network is periodically updated with the weights of the main network, which helps in stabilizing training by preventing rapid oscillations in Q-value updates.
	3.	Loss Function: The network is trained using a loss function that minimizes the difference between the predicted Q-values and the target Q-values computed using the Bellman equation. The optimizer (e.g., Adam) adjusts the network weights to reduce this loss over time.
	4.	Exploration-Exploitation Trade-off: The agent balances exploration (trying new actions) and exploitation (choosing the best-known action) using an epsilon-greedy strategy, where the probability of choosing a random action decreases as the agent becomes more confident in its learned policy.

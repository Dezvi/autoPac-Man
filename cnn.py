from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense


class PolicyGradientCNN:

    def __init__(self, state_space, action_space, epsilon_min, epsilon_decay):
        self.action_space = action_space
        self.state_space = state_space
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        buildModel()
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        self.model.compile(loss = None, optimizer=adam, metrics=['mae'])
        self.model.summary()

    def buildModel():
        # We have 3 inputs, a state, an action, and a reward of that action in that state 
        last_state = Input(shape=(self.state_space,), name='input')
        last_action = Input(shape=(self.action_size,), name='last_action')
        last_reward = Input(shape=(1,), name='reward')
        # How we are using an image as an input we need convolutions
        f = Conv2D(32, 8, strides=(4, 4), activation = 'relu', input_shape=input_shape, kernel_initializer='glorot_uniform')(x)
        f = Conv2D(64, 4, strides=(2, 2), activation = 'relu', input_shape=input_shape, kernel_initializer='glorot_uniform')(x)
        f = Conv2D(64, 3, strides=(1, 1), activation = 'relu', input_shape=input_shape, kernel_initializer='glorot_uniform')(x)
        f = Flatten()(f)
        f = Dense(512, activation = 'relu', kernel_initializer='glorot_uniform')(f)
        # We predict an action as an output with the size of the action_space
        action_pred = Dense(self.action_space, activation = 'softmax', kernel_initializer='glorot_uniform')(f)
        self.model = Model(inputs=[last_state, last_action, last_reward], outputs = [action_pred])
        self.model.add_loss(self.custom_loss(action_pred, last_action, last_reward))


    # This loss function is a policy gradient loss function
    def customLoss(action_pred, last_action, last_reward):
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = action_pred, labels = last_action)
        loss = tf.reduce_mean(neg_log_prob * last_reward)
        return loss

    # To choose an action we need to have some exploration, we make this posible by an epsilon
    def chooseAction(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, epsilon)
            
        r = np.random.random()
        if r > self.epsilon:
            print(" ********************* CHOOSING A PREDICTED ACTION **********************")
            actions = np.ones((1, self.action_space))
            rewards = np.ones((1, 1))
            state = np.array(state).reshape(1, self.state_space)
            pred = self.model.predict([state, actions, rewards])
            action = pred
        else:
            print("******* CHOOSING A RANDOM ACTION *******")
            action = np.random.choice(range(self.action_space))  # select action w.r.t the actions prob
        return action


    # Update our target network
    def trainTarget(self, states, actions, discounted_episode_rewards):
        self.model.fit([states, actions, discounted_episode_rewards])

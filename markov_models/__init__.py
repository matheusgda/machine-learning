import numpy as np

class HiddenMarkovModel:

    def __init__(self, transitions, emissions, state_labels, observation_labels):
        self.m = transitions
        self.e = emissions
        self.s_labels = state_labels
        self.o_labels = observation_labels
        self.states = len(state_labels)


    # dynamic programming implmentation to be reused by alpha and beta
    #  sel_fun gets transition probabilities for states
    def dynamic_probability(self, sequence, sel_fun, prior=None):
        T = len(sequence)
        state_probability = np.ones((T + 1, self.states))

        # initializing accepting a prior distribution
        if prior is None:
            state_probability[0] = state_probability[0] / self.states
        else:
            state_probability[0] = prior

        #state_probability[0] = self.e[sequence[0]] # uses emissions of first observation
        for t in range(1, T + 1):
            t_a = t - 1
            o = sequence[t_a]
            probs = state_probability[t_a] # previous alpha
            for k in range(self.states):
                transition_prob = sel_fun(k) # get transition vector
                emission_prob = self.e[o][k]
                state_probability[t][k] = emission_prob * np.dot(transition_prob, probs)
        return state_probability[T]


    # dynamic programming approach to solve optimum sequence query
    def query_hidden_sequence(self, sequence, prior=None):
        prior_d = prior # allows user defined prior
        T = len(sequence)
        if prior is None:
             prior_d = np.ones(self.states) / self.states
        delta = np.ones((T, self.states))
        delta[0] = np.multiply(self.e[sequence[0]], prior_d)
        solutions = np.zeros((T,self.states), dtype='i4')
        for t in range(1, T):
            max_v = np.zeros(self.states)
            for k in range(self.states): # compute inner delta for all states
                val = np.multiply(self.m[k], delta[t-1])
                arg = np.argmax(val)
                max_v[k] = val[arg]
                solutions[t-1][k] = arg
            delta[t] = np.multiply(self.e[sequence[t]], max_v)
        return (self.backtrace_solution(solutions, delta[T - 1]), np.max(delta[T - 1]))


    # look backwards from the future
    def beta(self, sequence, prior=None):
        return self.dynamic_probability(np.flip(sequence, 0), lambda k: self.m[k], prior)


    # look foward from the past
    def alpha(self, sequence, prior=None):
        return self.dynamic_probability(sequence, lambda k: self.m[:][k], prior)


    # uses alpha to obtain the probability of the whole sequence
    def sequence_probability_a(self, sequence, prior=None):
        return np.sum(self.alpha(sequence, prior))


    # uses beta to obtain the probability of whole sequences of observations
    def sequence_probability_b(self, sequence, prior=None):
        return np.sum(self.beta(sequence, prior))


    # compact method to get most probable hidden sequence using user defined labels for
    #  states
    def hidden_sequence(self, observations, prior=None):
        seq = list()
        hs = self.query_hidden_sequence(observations, prior)
        for i in hs[0]:
            seq.append(self.s_labels[i])
        return (seq, hs[1])

    def state_probability_at(self, sequence, time):
        T = len(sequence)

        # note that emission probability is taken into account twice
        alpha = self.alpha(sequence[: time + 1])
        beta = self.beta(sequence[time: ])
        return np.multiply(alpha, beta) / np.sum(np.multiply(alpha, beta))



    def most_probable_state(self, sequence, time):
        states = self.state_probability_at(sequence, time)
        arg = np.argmax(states)
        return (self.s_labels[arg], states[arg])


    # iterate backwards over solutions to obtain the best sequence
    def backtrace_solution(self, solutions, delta):
        arg = np.argmax(delta)
        T = len(solutions)
        seq = np.ones(T, dtype='i4')
        for i in range(0, T):
            j = T - i - 1
            seq[j] = solutions[j][arg]
            arg = seq[j]
        return seq

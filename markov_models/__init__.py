import numpy as np

class HiddenMarkovModel:

    def __init__(self, transitions, emissions, state_labels, observation_labels):
        self.m = transitions
        self.e = emissions
        self.s_labels = state_labels
        self.o_labels = observation_labels
        self.states = len(state_labels)



    # dynamic programming implmentation to be reused by alpha and beta
    def state_probability(self, sequence, sel_fun):
        T = len(sequence)
        values = np.zeros((T, self.states))
        values[0] = self.e[sequence[0]] # uses emissions of first observation
        for t in range(1, T):
            for k in range(self.states):
                o = sequence[t]
                transition_prob = sel_fun(k) # get transition vector
                probs = values[t-1] # previous alpha
                emission_prob = self.e[o][k]
                values[t][k] = emission_prob * np.dot(transition_prob, probs)
        return values[T - 1]


    # look backwards from the future
    def beta(self, sequence):
        print(sequence, np.flip(sequence, 0), "addsf")
        return self.state_probability(np.flip(sequence, 0), lambda k: self.m[k])


    # look foward from the past
    def alpha(self, sequence):
        return self.state_probability(sequence, lambda k: self.m[:][k])


    # g
    def alpha_sequence_probability(self, sequence):
        return np.sum(self.alpha(sequence))

    def beta_sequence_probability(self, sequence):
        return np.sum(self.beta(sequence))

    def state_probability_at(self, sequence, time, state):
        return 0


    def most_probable_state(self, sequence, state):
        return 0




# class DynamicProgrammer(self,)

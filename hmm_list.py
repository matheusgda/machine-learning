from markov_models import HiddenMarkovModel
import numpy as np

transitions = np.array([[0.3, 0.7], [0.7,0.3]]) # coin A, coin B
emissions = np.array([[0.9,0.1],[0.1,0.9]]) # state head; state tail
state_labels = ["Coin A", "Coin B"]
observation_labels = ["Head", "Tails"]

sequence1 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
sequence2 = np.array([0, 1, 0, 1, 0, 1, 0 ,1 ,0 ,1, 0])

hmm = HiddenMarkovModel(transitions, emissions, state_labels, observation_labels)
print(hmm.sequence_probability_a(sequence1), hmm.sequence_probability_b(sequence1))
print(hmm.sequence_probability_a(sequence2), hmm.sequence_probability_b(sequence2))

print("Alpha for sequence 1", hmm.alpha(sequence1))
print("Alpha for sequence 2", hmm.alpha(sequence2))

print("\nTests for state probability")

print(sequence1[5], sequence2[5])
print(hmm.state_probability_at(sequence1, 4))
print(hmm.state_probability_at(sequence2, 4))


print("\nSolving most probable sequence!")
print(hmm.query_hidden_sequence(sequence1))
print(hmm.query_hidden_sequence(sequence2))

print(hmm.hidden_sequence(sequence1))
print(hmm.hidden_sequence(sequence2))

print("\nQUESTION 5\n")
obs1 = np.array([0,1,0,0,1])
print("Part 1")
print("MPS:\n", hmm.hidden_sequence(obs1))
print("At third choice\n", hmm.most_probable_state(obs1, 3))


print("Part2")
print("MPS:\n", hmm.hidden_sequence(sequence2))
print("MPS:\n", hmm.hidden_sequence(sequence2, [0.2,0.8]))





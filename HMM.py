import numpy as np
import math

class HMM:
    def __init__(self, A, B, pi, o_stataes, h_states ):
        self.A = A
        self.B = B
        self.pi = pi
        self.o_stataes = o_stataes
        self.h_states = h_states

    def HMM_Forward(self, observation):
        # theta = []
        delta = []
        for i, observe in enumerate(observation):
            delta_temp = []
            # theta_temp = []
            for prev_state in h_states:
                if i == 0:
                    delta_temp.append(self.pi[h_states.index(prev_state)] * self.B[h_states.index(prev_state)][o_stataes.index(observe)]) 
                    # theta_temp = [0,0,0]
                else:
                    delta_temp_temp = []
                    for next_state in h_states:
                        # print (delta[i - 1][k],B[j][C.index(O[i])])
                        delta_temp_temp.append(delta[i - 1][h_states.index(next_state)] *
                        self.B[h_states.index(prev_state)][o_stataes.index(observe)] *
                        self.A[h_states.index(next_state)][h_states.index(prev_state)])
                    # print(delta_temp_temp)
                    # print(max(delta_temp_temp))
                    delta_temp.append(sum(delta_temp_temp))
                    # theta_temp.append(np.argmax(delta_temp_temp) + 1)
            # print(delta_temp)
            delta.append(delta_temp)
            # theta.append(theta_temp)
            # print(theta_temp)
        return delta

    def HMM_Viterbi(self, observation):
        # print(theta)

        theta = [[0,0,0]]
        delta = []

        delta.append(list(self.pi * self.B[:,o_stataes.index(observation[0])]))
        for i  in range(1, len(observation)):
            delta_temp = []
            theta_temp = []
            for prev_state in h_states:
                delta_temp_temp = delta[i - 1] *  self.A[:,h_states.index(prev_state)] * self.B[h_states.index(prev_state)][o_stataes.index(observation[i])]
                delta_temp.append(max(delta_temp_temp))
                theta_temp.append(self.h_states[np.argmax(delta_temp_temp)])
            # print(delta_temp)
            delta.append(delta_temp)
            theta.append(theta_temp)
            # print(theta_temp)


        #Best State
        q_star = []
        # print(delta)
        q_star.append(self.h_states[np.argmax(delta[-1])])
        # print(q_star)
        for i in range(len(theta) - 1, 0, -1):
            # print(q_star[-1])
            # print(theta[i][q_star[-1]])
            q_star = [theta[i][self.h_states.index(q_star[0])]] + q_star

        # print(q_star)
        return delta, q_star

    def observeGenerating(self, index):
        observation = []

        state = np.random.choice(self.h_states,1, p = self.pi)[0]
        observe = np.random.choice(self.o_stataes,1, p = self.B[h_states.index(state)])[0]
        observation.append(observe)
        # print(0 , state, observe)
        for i in range(100):
            state = np.random.choice(self.h_states,1, p = self.A[h_states.index(state)])[0]
            observe = np.random.choice(self.o_stataes,1, p = self.B[h_states.index(state)])[0]
            observation.append(observe)
            # print(i , state, observe)
            if observe == "S":
                break
        # print(observe)
        print("Sequence "+ str(index + 1) +" of observations:",",".join(observation),"\n")


A1 = [[1, 0, 0, 0] , [0, 0, 0, 1], [0, 0.4, 0.3, 0.3], [0.3, 0.2, 0.2, 0.3]]
B1 = [[1, 0, 0, 0, 0], [0, 0.5, 0.5, 0, 0], [0, 0.2, 0.2, 0.3, 0.3],[0, 0, 0, 0.5, 0.5]]

A2 = np.array([[1, 0, 0, 0] , [0.1, 0.3, 0.5, 0.1], [0.1, 0.4, 0.3, 0.2], [0.1, 0.4, 0.2, 0.3]])
B2 = np.array([[1, 0, 0, 0, 0], [0, 0, 0.5, 0, 0.5], [0, 0, 0.5, 0.5, 0],[0, 0.5, 0, 0, 0.5]])
pi1 = [0, 1.0, 0, 0]
pi2 = [0, 0 , 0 , 1]
o_stataes = ["S", "A" , "B", "C", "D"]
h_states = [1,2,3, 4]
O = [['A', 'D', 'C', 'B', 'D', 'C', 'C', 'S'],['B', 'D', 'S'], ['B', 'C', 'C', 'B', 'D', 'D', 'C', 'A', 'C', 'S'],
['A', 'C', 'D', 'S'], ['A', 'D', 'A', 'C', 'S'], [ 'D', 'B', 'B', 'S'] , ['A', 'B', 'S'],
['D', 'D', 'B', 'D', 'D', 'B', 'A', 'C', 'C', 'D', 'A', 'B', 'B', 'C', 'D', 'B', 'B', 'B', 'S'],
['D', 'B', 'D', 'S'], ['A', 'A', 'A', 'A', 'D', 'C', 'B', 'S']]
O2 = ['B', 'D', 'S']

# print("aaaaaaaaaaa")
# print(A[:,0])
hmm1 = HMM(A1, B1, pi1, o_stataes, h_states)
hmm2 = HMM(A2, B2, pi2, o_stataes, h_states)
#Question 1
print("Question 1")
for i in range(10):
    hmm1.observeGenerating(i)


#Question 2
for observe in O:
    print(*observe, sep = "," )
    delta1 = hmm1.HMM_Forward(observe)
    delta2 = hmm2.HMM_Forward(observe)
    print("P(O|HMM1)", sum(delta1[-1]))
    print("P(O|HMM2)", sum(delta2[-1]))
    if sum(delta1[-1]) > sum(delta2[-1]):
        print(*observe, " belongs to HMM 1\n", sep = "," )
    else:
        print(*observe, " belongs to HMM 2\n", sep = "," )


#Question 3
print("Question 3: Best hidden states for above observations using HMM2 \n")
for observe in O:
    # print(observe)
    delta_viterbi, q_star = hmm2.HMM_Viterbi(observe)
    print(",".join(observe),":",",".join(map(str, q_star)),"\n")
    # print(sep = ",")
# print (delta)
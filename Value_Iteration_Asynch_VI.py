import numpy as np
import heapq
from Solvers.Abstract_Solver import AbstractSolver, Statistics


class ValueIteration(AbstractSolver):
    def __init__(self, env, eval_env, options):
        assert str(env.observation_space).startswith("Discrete"), (
            str(self) + " cannot handle non-discrete state spaces"
        )
        assert str(env.action_space).startswith("Discrete"), (
            str(self) + " cannot handle non-discrete action spaces"
        )
        super().__init__(env, eval_env, options)
        self.V = np.zeros(env.observation_space.n)
        #self.P = {}

    def train_episode(self):
            
        for each_state in range(self.env.observation_space.n):
            # Do a one-step lookahead to find the best action
            bAction = self.one_step_lookahead(each_state)
            bestAction = np.max(bAction)
            
            self.V[each_state] = bestAction
        
        self.statistics[Statistics.Rewards.value] = np.sum(self.V)
        self.statistics[Statistics.Steps.value] = -1

    def __str__(self):
        return "Value Iteration"

    def one_step_lookahead(self, state: int):
        A = np.zeros(self.env.action_space.n)
        for a in range(self.env.action_space.n):
            for prob, next_state, reward, done in self.env.P[state][a]:
                A[a] += prob * (reward + self.options.gamma * self.V[next_state])
        return A

    def create_greedy_policy(self):
        
        def policy_fn(state):
            values = self.one_step_lookahead(state)
            #numA = len(values)
            chooseAction = np.argmax(values)
            return chooseAction


        return policy_fn


class AsynchVI(ValueIteration):
    def __init__(self, env, eval_env, options):
        super().__init__(env, eval_env, options)
        self.pq = PriorityQueue()
        self.pred = {}
        for s in range(self.env.observation_space.n):
            A = self.one_step_lookahead(s)
            best_action_value = np.max(A)
            self.pq.push(s, -abs(self.V[s] - best_action_value))
            for a in range(self.env.action_space.n):
                for prob, next_state, reward, done in self.env.P[s][a]:
                    if prob > 0:
                        if next_state not in self.pred.keys():
                            self.pred[next_state] = set()
                        if s not in self.pred[next_state]:
                            try:
                                self.pred[next_state].add(s)
                            except KeyError:
                                self.pred[next_state] = set()

    def train_episode(self):    
        cState = self.pq.pop()
        OS_state = self.one_step_lookahead(cState)
        vS_state = np.max(OS_state)
        self.V[cState] = vS_state
        
        for pState in self.pred[cState]:
            OS_pState = self.one_step_lookahead(pState)
            vS_pState = np.max(OS_pState)
            self.pq.update(pState, -abs(self.V[pState] - vS_pState))

        self.statistics[Statistics.Rewards.value] = np.sum(self.V)
        self.statistics[Statistics.Steps.value] = -1

    def __str__(self):
        return "Asynchronous VI"


class PriorityQueue:

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)
            
            


import numpy as np
import numpy.linalg
from Solvers.Abstract_Solver import AbstractSolver, Statistics

class PolicyIteration(AbstractSolver):

    def __init__(self, env, eval_env, options):
        assert str(env.observation_space).startswith( 'Discrete' ), str(self) + \
                                                                    " cannot handle non-discrete state spaces"
        assert str(env.action_space).startswith('Discrete'), str(self) + " cannot handle non-discrete action spaces"
        super().__init__(env, eval_env, options)
        self.V = np.zeros(env.observation_space.n)
        # Start with a random policy
        self.policy = np.ones([env.observation_space.n, env.action_space.n]) / env.action_space.n

    def train_episode(self):
        self.policy_eval()
        
        for s in range(self.env.observation_space.n):
            bAction = self.one_step_lookahead(s)
            best = np.argmax(bAction)

            self.policy[s] = np.eye(self.env.action_space.n)[best]
            

        self.statistics[Statistics.Rewards.value] = np.sum(self.V)
        self.statistics[Statistics.Steps.value] = -1

    def __str__(self):
        return "Policy Iteration"

    def one_step_lookahead(self, state):
        A = np.zeros(self.env.action_space.n)
        for a in range(self.env.action_space.n):
            for prob, next_state, reward, done in self.env.P[state][a]:
                A[a] += prob * (reward + self.options.gamma * self.V[next_state])
        return A

    def policy_eval(self):
        A=np.zeros((self.env.observation_space.n, self.env.observation_space.n))
        B=np.zeros(self.env.observation_space.n)
        
        
        for state in range(self.env.observation_space.n):
            for action, action_prob in enumerate(self.policy[state]):
                for prob, next_state, reward, done in self.env.P[state][action]:
                    A[state][next_state]+=action_prob*prob*self.options.gamma
                    B[state]+=action_prob*prob*reward
                    
        self.V=np.linalg.solve((np.eye(self.env.observation_space.n)-A),B)
                    
        

                         
    def create_greedy_policy(self):
        
        def policy_fn(state):
            return np.argmax(self.policy[state])

        return policy_fn
        

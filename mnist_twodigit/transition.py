import torch
import statistics

'''Stores transition matrix p(s' | s,a) in tabular form.  '''

class Transition:

    def __init__(self, ncodes): 
        self.ncodes = ncodes
        self.reset()

    def reset(self):

        self.state_transition = torch.zeros(self.ncodes,3,self.ncodes)

        self.tr_lst = []
        self.trn_lst = []
        for j in range(0,self.ncodes):
            self.tr_lst.append([])
            self.trn_lst.append([])


    def update(self, ind_last, ind_new, a1, y1, y1_):


        for j in range(0, ind_last.shape[0]):
            self.state_transition[ind_last.flatten()[j], a1[j]+1, ind_new.flatten()[j]] += 1

        for j in range(0,self.ncodes):
            self.tr_lst[j] += y1[ind_last.flatten()==j].data.cpu().numpy().tolist()
            self.trn_lst[j] += y1_[ind_new.flatten()==j].data.cpu().numpy().tolist()


    def print_codes(self): 

        for j in range(0,self.ncodes):

            if len(self.tr_lst[j]) > 0:
                print('last', j, self.tr_lst[j], 'mode', statistics.mode(self.tr_lst[j]))
            if len(self.trn_lst[j]) > 0:
                print('next', j, self.trn_lst[j], 'mode', statistics.mode(self.trn_lst[j]))

    def print_modes(self):

        mode_lst = []
        moden_lst = []
        for j in range(0,self.ncodes):
            if len(self.tr_lst[j]) == 0:
                mode_lst.append(-1)
            else:
                mode_lst.append(statistics.mode(self.tr_lst[j]))#torch.Tensor(tr_lst[j]).mode()[0])

            if len(self.trn_lst[j]) == 0:
                moden_lst.append(-1)
            else:
                moden_lst.append(statistics.mode(self.trn_lst[j]))#torch.Tensor(trn_lst[j]).mode()[0])

        corr = 0
        incorr = 0

        coverage = torch.zeros(10,3)

        print('state transition matrix!')
        for a in range(0,3):
            for k in range(0,self.state_transition.shape[0]):
                if self.state_transition[k,a].sum().item() > 0:
                    print(mode_lst[k], a-1, 'argmax', moden_lst[self.state_transition[k,a].argmax()], 'num', self.state_transition[k,a].sum().item())

                    num = self.state_transition[k,a].sum().item()
                    s1 = mode_lst[k]
                    s2 = moden_lst[self.state_transition[k,a].argmax()]
                    nex = s1 + (a-1)
                    nex = min(nex, 9)
                    nex = max(nex, 0)
                    coverage[nex,a] = 1.0
                    if nex == s2:
                        corr += num
                    else:
                        incorr += num


        print('transition acc', corr*1.0 / (corr+incorr))
        print('coverage', coverage.sum() * 1.0 / (10*3))   

    def select_goal(self):

        code_count = self.state_transition.sum(dim=(1,2))

        reward = 1.0 / torch.sqrt(code_count+0.1)

        #print('reward', reward)

        return reward

    '''
        p(s' | a,s)

        max_a * p(s' | a,s)

    '''
    def select_policy(self, init_state, reward):

        eps = 0.0001
        counts = self.state_transition+eps
        probs = counts / counts.sum(dim=2, keepdim=True)

        best_action = -1
        best_value = -1

        for a in range(0,3):
            val = 0.0
            for sn in range(self.ncodes):
                val += reward[sn] * probs[init_state, a, sn]

            print('a,v', a-1, val)

            if val > best_value:
                best_value = val
                best_action = a

        #print('probs', probs)

        #print('probs at init state', probs[init_state])

        #raise Exception('done')


        best_action = torch.Tensor([best_action]).long() - 1

        return best_action







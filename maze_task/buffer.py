import random
import torch

class Buffer:

    '''
        Stores (a, y1, y1_, c1, y2, y2_, c2, x, x_) as matrices.  
    '''
    def __init__(self, ep_length, max_k):
        self.a = []
        self.y1 = []
        self.y1_ = []
        # self.c1 = []
        # self.c2 = []
        self.y2 = []
        self.y2_ = []
        self.x = []
        self.x_ = []

        self.steps = []

        self.ep_length = ep_length
        self.max_k = max_k

        self.num_ex = 0

    '''
        
    '''
    # def add_example(self, a, y1, y1_, c1, y2, y2_, c2, x, x_, step):

    def add_example(self, a, y1, y1_,  y2, y2_, x, x_, step):

        self.a.append(a)
        self.y1.append(y1)
        self.y1_.append(y1_)
        self.y2.append(y2)
        self.y2_.append(y2_)
        # self.c1.append(c1)
        # self.c2.append(c2)
        self.x.append(x)
        self.x_.append(x_)

        self.steps.append(step)

        self.num_ex += 1

    '''
        Returns bs of (a, y1, y1_, x, x_)


        

    '''
    def sample_batch(self, bs, indlst=None, klim=None): 


        ba = []
        by1 = []
        by1_ = []
        bx = []
        bx_ = []
        bk = []

        #if indlst is None:
        #    bs = min(self.num_ex, bs)


        for k in range(0, bs):

            if indlst is None:
                j = random.randint(0, self.num_ex-1)
            else:
                j = indlst[k]

            a,y1,y1_,x,x_,step = self.sample_ex(j)

            maxk = min(self.max_k, self.ep_length - step)
            maxk = min(maxk, self.num_ex - j - 1)
            maxk = max(maxk, 1)


            if klim is not None:
                maxk = min(maxk, klim)

            randk = random.randint(1, maxk)

            if randk != 1:
                j_n = j + randk
                _, _, y1_n, _, x_n, step_n = self.sample_ex(j_n)
                assert step_n > step
                y1_ = y1_n
                x_ = x_n


            ba.append(a.cuda())
            by1.append(y1.cuda())
            by1_.append(y1_.cuda())
            bx.append(x)
            bx_.append(x_)
            bk.append(torch.Tensor([randk]).long().cuda())

        ba = torch.cat(ba, dim=0).cuda()
        by1 = torch.cat(by1, dim=0).cuda()
        by1_ = torch.cat(by1_, dim=0).cuda()
        bx = torch.cat(bx, dim=0).cuda()
        bx_ = torch.cat(bx_, dim=0).cuda()
        bk = torch.cat(bk, dim=0).cuda()

        return ba, by1, by1_, bx, bx_, bk

    def sample_ex(self, j): 

        return (self.a[j], self.y1[j], self.y1_[j], self.x[j], self.x_[j], self.steps[j])






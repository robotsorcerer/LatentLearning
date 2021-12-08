import random
import torch

class Buffer:

    '''
        Stores (a, y1, y1_, c1, y2, y2_, c2, x, x_) as matrices.  
    '''
    def __init__(self):
        self.a = []
        self.y1 = []
        self.y1_ = []
        self.c1 = []
        self.c2 = []
        self.y2 = []
        self.y2_ = []
        self.x = []
        self.x_ = []

        self.num_ex = 0

    '''
        
    '''
    def add_example(self, a, y1, y1_, c1, y2, y2_, c2, x, x_):

        self.a.append(a)
        self.y1.append(y1)
        self.y1_.append(y1_)
        self.y2.append(y2)
        self.y2_.append(y2_)
        self.c1.append(c1)
        self.c2.append(c2)
        self.x.append(x)
        self.x_.append(x_)

        self.num_ex += 1

    '''
        Returns bs of (a, y1, y1_, x, x_)
    '''
    def sample_batch(self, bs, indlst=None): 

        ba = []
        by1 = []
        by1_ = []
        bx = []
        bx_ = []

        #if indlst is None:
        #    bs = min(self.num_ex, bs)


        for k in range(0, bs):

            if indlst is None:
                j = random.randint(0, self.num_ex-1)
            else:
                j = indlst[k]

            a,y1,y1_,x,x_ = self.sample_ex(j)

            ba.append(a.cuda())
            by1.append(y1.cuda())
            by1_.append(y1_.cuda())
            bx.append(x)
            bx_.append(x_)

        ba = torch.cat(ba, dim=0).cuda()
        by1 = torch.cat(by1, dim=0).cuda()
        by1_ = torch.cat(by1_, dim=0).cuda()
        bx = torch.cat(bx, dim=0).cuda()
        bx_ = torch.cat(bx_, dim=0).cuda()


        return ba, by1, by1_, bx, bx_

    def sample_ex(self, j): 

        return (self.a[j], self.y1[j], self.y1_[j], self.x[j], self.x_[j])







# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:24:52 2019

@author: u4ft
"""

PARSER = False

import env
import random
import numpy as np
import random # random samples from different batches (experience replay)
import os # For loading and saving brain
import copy
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim # for using stochastic gradient descent
import torch.autograd as autograd # Conversion from tensor (advanced arrays) to avoid all that contains a gradient
# We want to put the tensor into a variable taht will also contain a
# gradient and to this we need:
from torch.autograd import Variable

if PARSER:
    import spacy
    
'''    
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)'''

words = '''
_ I you someone them
move give speak hit
birth die
IF AND OR NOT YES TRUE because
TRY can should
want believe
before after WHILE soon later _times 
here there
very almost like 
rarely seldom sometimes often always
_ed will
equal big small positive negative zero one two three smaller bigger
first next last word _list
no few some many ALL 
same different
that like of 
when what who where why which whom whither how how-much
join separate oppose friend enemy
revenge betray
ask answer order obey forbid allow
call promise swear bless curse
err forget teach learn
just free caring holy loyal respect
mate feed
mated mymate mateable isMateage taskmate mate_me
step1 step2 step3 step4 step
op_compare op_random_choice op_out_mate
em_nodes people
Q X
'''.split()

# Question = ? eXecute = !

namedict = {}
s,v,o,t = range(4)

idx = 1
LEN = 50
vec = {}
ivec = {}
for w in words:
    idx += 1
    globals()[w] = idx
    vec[idx] =  w #np.random.random(LEN)
    
def coin(p):
    return random.random() < p


#writeNodes
#loadNodes

pop = env.pop
    
cnt=0
DNA_LEN = 50
WM_LEN = 40*3 + 40
TASK_LEN = 500
N_LAYERS = 3


'''
class NNtask(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(100, 30)
        self.output = nn.Linear(30, 10)
  
    def forward(self, x):
        x = self.hidden(x)
        x = F.sigmoid(x)
        x = self.output(x)
        return x
        
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        '''


class NNSupervised(nn.Module):
    def __init__(self):
        super().__init__()
        #self.hidden = nn.Linear(100, 30)
        self.output = nn.Linear(WM_LEN, WM_LEN)
  
    def forward(self, x):
        #x = self.hidden(x)
        #x = F.sigmoid(x)
        x = self.output(x)
        return x


nn_before = NNSupervised()
nn_after = NNSupervised()
nn_bdi = NNSupervised()
nn_can = NNSupervised()
        

class Z():
    def __init__(self, s=None, v = None, o = None, k=1, op=None ,t = None, 
                 txt = None):
        self.s, self.v, self.o, self.t, self.k, self.op = s,v,o,k,op,t
        self.txt = txt

def save():
    #nnTask.load('a.th')
    pass

def load():
    #nnTask.save('a.th')
    
    pass

VERBS = [birth, die] + [give, hit] + [speak, believe, want, TRY]

def generategoal():
    s = I  if coin(0.5) else random.choice(pop)    
    v1 = random.choice(VERBS)
    o1 = random.choice(pop)
    if v1 in [birth, die]:
        return Z(s=s, v=v1)
    if v1 in [give, hit]:
        return Z(s=s, v=v1, o = o1)
    if v1 in [speak, believe, want, TRY]:
        s2 = random.choice(pop)
        v2 = random.choice([birth, die, give, hit])
        o2 = random.choice(pop)
        return Z(s=s, v=v1, txt = Z(s=s2, v=v2,o=o2), o = o1)
        
    
def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class Agent():
        
    def  __init__(self,m=None,f=None, gender=None):
        global cnt
        self.id = cnt
        cnt += 1
        self.gender = gender if gender is not None else random.choice(['M','F'])
        self.e = 8
        #self.birth =0
        self.dna = np.random.random(DNA_LEN)
        
        self.child = None
        self.nn = np.zeros(500)
        self.task = np.zeros(TASK_LEN)
        self.w = np.zeros((500,500))
        self.wm = torch.zeros(2,WM_LEN)
        self.ctx = torch.zeros(5,WM_LEN)
        self.em1 = []
        
        self.log = []
        self.i = 0
        self.sociosexuality = random.random()
        self.soulmate = None
        self.action = []
        self.goals = torch.zeros(10,WM_LEN)
        self.wantmate = None
        self.name = str(random.random()*1e8)
        namedict[self.name] = self
        
    def __str__(self):
        return 'a' + str(self.id)
        
    def fitness_age(self):        
         if self.age < 20:  
            return self.age/20
         else:
            return max(0, (20 - self.age/2) /20)   



    def f_isMateAge(self):
        return     self.age < 20 and self.age<50 
    
    def f_mateable(self,x):
        return  x.gender != self.gender and x.age > 25 and x.age<50
    
        
    def utility_self(self): 
        u =0
        if self.nn[mated]: u +=1
        u += min(1, self.e/10)
        
        self.nn[mated] = False
        return u
    
    def inclusive_fitness(self):
        def fitness_age(self):
            return 0.5 + self.age/40 if self.age2 < 20 else max(0, 1.5 - self.age/40)
        u = 0
        for p in pop:
            u+= p.utility_self()* env.kin[self, p] * p.fitness_age()
        return u    
            
    def input1(self, foo):
        if isinstance(foo, tuple) or isinstance(foo, list) :
            foo = Z(foo[0], foo[1], foo[2] if len(foo)>2 else None)
        foo.t = env.T    
        if foo.v == mated:
            self.mated = True
            return       
        em_write(self,foo)  
    
      
    #def step(self): pass #self.action = xxx
    
    def step(self):    
        #self.action = []
        nn, wm, w, task, goals, ctx = self.nn, self.wm, self.w, self.task, self.goals, self.ctx
        
        for tt in range(20):
            if len(self.log) >0:
                inp = self.log[self.i % len(self.log)]
                #calc_reward inp
                #update weights
                em_write(inp)
                
    
    
            '''for i in range(N_LAYERS):
                i100 = 100*i
                ctx[i100+100, i100 +200] = conv(ctx[ i100, i100+100 ])'''
                
            '''wm = conv( conv, conv( inp))
            centroid -> 
            winner-take-all -> next level'''
            
            
            step1(self)
    
            #task = nn_task( torch.cat((task, wm), 0) )  # sigmoid ?
            w2 = [0.9**x for x in range(TASK_LEN)]
            #task += nnTask( task, wm, ctx)* w
            '''norm(task)  
            t1 = env.T -self.birth   
            for i in range(10):
                t1 = int(t1/2)
                if t1 % 2 ==0: 
                    task[i*10: i*10+10] = nnTask( task, wm, ctx)'''
                    
                    
            # wm-ops
            # few-shot learning
    
               
     
            #attention
    
            #chunk a,b,c -> X #unchunk X -> ab,c
            # I told b that a hit c but he did not believe
            
            nA = 10
            task[random.randint(0,15)] = 1
            probs = task[:15]
            probs = np.random.random(nA)
            probs = normalized(probs, axis=0,order=1)[0]
    
            op = np.random.choice(nA, p=probs )
            #op = softmax( task[:10] )
            if op  in [0,1,2,3]:
                v =  ['mate','give','hit','speak'][op]
                #ag = agent_from_wm( wm[0]) 
                ag = random.choice(pop)
                self.action.append( Z(s=self, v=v ,o=ag) )
            if op  in [4,5,6,7]:
                #wm[1] = supervised_out([nn_before, nn_after, nn_bdi, nn_can][op-4], wm[0] )  
                net = [nn_before, nn_after, nn_bdi, nn_can][op-4]
                #wm[0] = torch.from_numpy( np.zeros([100]) ,  dtype=torch.float64)
                wm[0] = torch.from_numpy( np.zeros([WM_LEN,]) ).float()
                wm[1] = net( wm[0] ) 
            if op==8:
                # spreading activation     
                #nn = np.matmul(w, nn)
               # nn = normalized(nn, axis=0)   # 
               pass
            if op==9:
                shift = torch.rand( [WM_LEN,WM_LEN] ) # task[1]        
                wm[1] = torch.mv( shift , wm[0]) 
                #print(c)
                    # nn_magnitude
            if op == 10:
                goals.append( g)
            if op == 11 :   
                goals.remove(g)            
            if op== 12:
               # wm = search_em( wm )
               pass
            if op==13:
                pass
                #g = find_g( s, g0)
            #if op == 5
            #    wm2 = nn( op_vec, wm) 

            #value = compare( wm, g)
           # Q[g] = value
            
            u = self.inclusive_fitness()
    
            # planning:   x causes goal => goal2 = x
    
            '''supervised(nn_before, wm[1], wm[0])     # nn[before]=1, train
            supervised(nn_after, wm[0], wm[1])
            b= foo[s]
            supervised( can, wm[0] ,b, foo)
            
            #bg, bw = inspect_goal_want(b)
            unsupervised(nn_bdi, bg, bw, foo)'''

    def add_goal(self, x):
        r = 300 + random.randint(0,20)
        self.w[r,node(x.s)]= x.k
        self.w[r,node(x.v)]= x.k
        self.w[r,node(x.o)]= x.k

    def spread(self):
        self.nn = np.matmul( self.nn, self.w)

    def zero(self):
        #for x in range(500):
        self.nn = np.zeros(500)    
               
def supervised(nn, a, b):
    lr = 0.01  # learn rate
    for i, j in itertools.product(range(WM_LEN), range(WM_LEN)):
        nn[i,j]+= lr*a[i]*b[j]                                   #hebbian
    
    
    
def supervised_out(nn, a,b):  
    for j in range(WM_LEN):
        b[j]=0
    for i, j in itertools.product(range(WM_LEN), range(WM_LEN)):
        b[j]+= wm[i,j]*a[j]
        
def unsupervised(nn, vec):
    pass
    # vae
    


def compare(s, g):
    return np.cos(s,g)
    
def sample(self, nn, inp, n_samples=3):
    previous = np.zeros()
    
    
    for i in n_samples:
        previous+= n_i
        inhibit = norm(previous / i)            
        #inhibit previous samples ]
        z0 = random.random()
        nn[j] += inp + z0
        nn[j] -= inhibit[j]
    out, p = nn(inp, z0)
    #p < 0.3 ?
    #uncertainty   -> nn[uncertain] =1
 
def random_choice(): 
    #weight?
    bag = set()
    for x in range(500):
        if nn[x][1] >= 1:
            bag.add(x)
    
   # wm[0] = random.choice(bag)
    
    
   

def em_write(self, event):
    w = self.w
    node = random.randint(400,500-1)
    w[ em_nodes ][node]=1 # this is a episodic memory
    s = event.s.id if isinstance(event.s, Agent) else 0  
    
    self.em1.append(event)
    # w[node][s] =1  w[node][v] =1

def conflict():
    pass





   
def node(x, group=None):
    if isinstance(x, str):
        return vec[x]
    if isinstance(x, Agent):
        return (x.id%100) + 300
    if group == 'time':
        return x % 100
    return 0

def rand_mem(self):
    return random.choice(self.em1)
    
def seek_mem(self, x):
    for y in self.em1:
        if y.s == s and y.v == s.v and y.o == x.o: return y
    return None
    
    nn = self.nn 
    self.zero()
    nn[node(x.s)] = 1
    nn[node(x.t)] = 1
    nn[node(x.o)] = 1
    self.spread()
    idx = 1 #argmax( nn[400:500])
    if  nn[idx] ==3:
        zeros()
        nn[idx] = 1
        self.spread()
        
    
    
def step1(self): #concepts, biases, etc
    nn = self.nn    


# mating pair bond
    p = self.sociosexuality
    if self.soulmate is not None and last_mate > 10 and coin(0.1):
         self.soulmate = random.choice( [x for x in pop if self.f_mateable(x)])
    #u = p* mated + (1-p)*mated[soulmate]
    self.add_goal( Z(s=self, v=mate, k= p))
    self.add_goal(  Z(s=self, v=mate, o=self.soulmate , k= (1-p) ))
    
# reciprocity
    # x cause u(y) up -> x help y
    #  x help y/I  -> y/I help x   
 
    #addInfer('x help y => y help x')
        
    #self.append_goal( Z(s=self, v=help, o=x))

    
# in/outgroup
    #I e_ group  -> help group -> x hit group -> hurt x
    
# parenting
    #child e < 2 fitness score 
    feed = [x.id for x in pop if env.kin[x,self] > 0.2 and x.e< 4 and x.age<20 and x != self]
    if len(feed) >0:
         self.append_goal( Z(s=self, v=give, o=random.choice(feed)))
    #predict child_dead -> u-- -> help_child -> feed child
    
    
# believe/want, and or not yes no, can should try, if because
   
   #speak
    opt = 1 #random.randint(10) 
    if opt == 1:
           # s v o -ed -> "s v o ed" -> s v o -ed 
       '''t1 = random.randint(env.T-5, env.T)
       self.zero()
       #nn[epmem] = 1
       #nn[node(t1,'time')] = 1    
       #nn[sample] = 1
       x = random.randint(400,500-1) 
       nn[x]=1
       self.spread()
       self.action.append( [speak, self])'''
       
       if len(self.em1) >0:
           mem = random.choice( self.em1)
           txt = copy.copy(mem) 
           txt.t = _ed
           self.action.append( Z(s=self, v=speak, o = random.sample(pop,3), txt=mem ) )
           
           
    if opt ==2:
           #s v o will -> "s v o will" -> s v o will        
       nn[expect] = 1
       nn[sample]=1
       speak, s, v, o
       
    if opt in [3,4,5,6]:
            #x believe p "x believe p" -> x believe p
       v = [believe,want, can, should, TRY][opt-4]          
       s = random.choice(pop)
       nn[v]=1
       nn[sample]=1
       speak, s, v, s2,v2,o2
    if opt ==7:
       t1 = random.randint(env.T-5, env.T)
       nn[mem] = 1
       nn[t] = t1    
       nn[sample] = 1
       
       [nn_cause]
       
       
      # speak s v o, because, s2, v2, o2 
 
    if opt == 8:
       nn[event] = 1    
       nn[sample] = 1
       run[nn_fut]
       
       
       #speak IF s v o,  , s2, v2, o2 
     
       #a and b / a or b / yes / no
    
    #hear
    ask = seek_mem(self,Z(v='speak', op='?', o=self,  t = env.T) ) # t=now x ask y
    if ask is not None:
         txt = ask.txt
         
         answer = seek_mem(txt)
         # wh? -->   # em_expr(x) speak x
         
         if '_' in [txt.s, txt.v, txt.o, txt.txt, txt.t]:
             actions.append([speak, txt if answer is not None else ''])
             pass
         else: #  y/n?
             #actions.append([speak , 'yes' if answer is not None else 'no'])     
             pass
        
    cmd = seek_mem(self,Z(v='speak', op='!', o=self,  t = env.T))
    if cmd is not None: 
         append_goal(cmd.txt)
        
    info = seek_mem(self,Z(v='speak', op=None, o=self,  t = env.T)   )
    if info is not None:
        em_write(info.txt)


# some common-sense 
       #x speak y -> z believe y
    
# model of above
    #nn[my_attention] = 

# play dream imagine

def name2agent(name):
    return namedict[ name.lowercase() ]
    
def txt2op(txt):
   'spacy'
        
    
def op2txt(z):
    z2 = z
    v = vec[z.v]
    if t == '_ed': v = v+'_ed'
    if t == will: v = 'will '+v
    if z.v in [birth ,die]:
         return str(z.s) +' ' + v
    if z.v in [mate, give, hit]:
         return str(z.s) +' '+ v+' '+ str(z.o)
    if z.v in [ believe, want]:  
         return str(z.s) +' '+ v+ ' that '+ op2txt(z.txt)
    if z.v in [speak]  :  
        return str(z.s) +' '+ v + ' to '+ str(z.o)+' that ' + '('+op2txt(z.txt)+')'
    if z.v in [ TRY,can,should]:
        txt2 = copy.copy(z)
        if txt2.s == s:
             txt2.s = ''
        return str(z.s) +' '+ v+  str(s2) + '('+op2txt(txt2)+')'     
    

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
QU X
'''.split()

# Question = ? eXecute = !

namedict = {}
s,v,o,t = range(4)

pop = env.pop
    
cnt=0
DNA_LEN = 50
WM_LEN = 40*3 + 40
TASK_LEN = 500
N_LAYERS = 3
VEC_LEN = 50

idx = 1
LEN = 50
toWord = {}
toNPVec = {}
ivec = {}
for w in words:
    idx += 1
    globals()[w] = idx
    toWord[idx] =  w #np.random.random(LEN)
    toNPVec[idx] = np.random.random(VEC_LEN)
    
toNPVec[None] = np.zeros( VEC_LEN)    
def coin(p):
    return random.random() < p

COMPRESS = np.random.random((5*VEC_LEN, VEC_LEN))

#writeNodes
#loadNodes




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

class NNPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        #self.hidden = nn.Linear(100, 30)
        self.output = nn.Linear(210,5*VEC_LEN) #nn.Linear(WM_LEN, WM_LEN)
  
    def forward(self, x):
        #x = self.hidden(x)
        #x = F.sigmoid(x)
        x = self.output(x)
        return x
    
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
    
class NNFuture(nn.Module):
    def __init__(self):
        super().__init__()
        #self.hidden = nn.Linear(100, 30)
        self.output = nn.Linear(5*VEC_LEN, 5*VEC_LEN)
  
    def forward(self, x):
        #x = self.hidden(x)
        #x = F.sigmoid(x)
        x = self.output(x)
        return x    
    
class NNtorch(nn.Module):
    def __init__(self):
        super().__init__()
        #self.hidden = nn.Linear(100, 30)
        self.output = nn.Linear(5*VEC_LEN, 5*VEC_LEN)
  
    def forward(self, x):
        #x = self.hidden(x)
        #x = F.sigmoid(x)
        x = self.output(x)
        return x   
    
class NNAction(nn.Module):
    def __init__(self):
        super().__init__()
        #self.hidden = nn.Linear(100, 30)
        self.output = nn.Linear(VEC_LEN, 5*VEC_LEN)
  
    def forward(self, x):
        #x = self.hidden(x)
        #x = F.sigmoid(x)
        x = self.output(x)
        return x      
    
class NNQ(nn.Module):
    def __init__(self):
        super().__init__()
        #self.hidden = nn.Linear(100, 30)
        self.output = nn.Linear( 5*VEC_LEN, 1)
  
    def forward(self, x):
        #x = self.hidden(x)
        #x = F.sigmoid(x)
        x = self.output(x)
        return x     

future = NNFuture()
cause = NNFuture()
nn_action = NNAction()
nn_before = NNSupervised()
nn_after = NNSupervised()
nn_bdi = NNSupervised()
nn_can = NNSupervised()
policy = NNPolicy()
Q = NNQ()
NN = NNtorch()
        

class Z():
    def __init__(self, s=None, v = None, o = None, k=1, op=None ,t = None, 
                 txt = None, m=None, mod = None):
        self.s, self.v, self.o, self.t, self.k, self.op = s,v,o,k,op,t
        self.txt ,self.m, self.mod= txt,m,mod
        self.confidence = 1
        
    def __str__(self):
        ret = ''
        if self.s is not None: ret+= 's= ' + str(self.s)+ ', '   
        if self.v is not None: ret+= 'v= ' + toWord[self.v]+ ', ' 
        if self.o is not None: ret+= 'o= ' + str(self.o)+ ', '   
        if self.t is not None: ret+= 't= ' + str(self.t)+ ', '         
        if self.k is not None: ret+= 'k= ' + str(self.k)+ ', '   
        if self.op is not None: ret+= 'op= ' + str(self.op)+ ', '                      
        if self.txt is not None: ret+= 'txt= (' + str(self.txt)+ '), ' 
        if self.m is not None: ret+= 'm= ' + str(self.m)+ ', '
        if self.mod is not None: ret+= 'mod= ' + str(self.mod)+ ', ' 
        return ret

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
        self.em = []
        self.em_future = []
        self.mem = {}
        self.state= torch.zeros(WM_LEN)
        self.future = future
        
        self.NN = NN
        self.log = []
        self.i = 0
        self.sociosexuality = random.random()
        self.soulmate = None
        self.action = []
        self.goals = torch.zeros(10,WM_LEN)
        self.wantmate = None
        self.name = str(random.random()*1e8)
        self.Q = Q
        self.cause = cause
        namedict[self.name] = self
        
        toNPVec[self] = np.random.random(VEC_LEN)
        
    def step0(self):
       for a in pop:
            if a not in self.mem: self.mem[a] = {believe:[],want:[]}
  
    
    def Agent2Vec(self,a):
        return torch.rand(VEC_LEN)
    
    def vec(self, o):
        if isinstance(o, str):   return toNPVec[o]
        if isinstance(o, Agent): return Agent2Vec(o)  
        if isinstance(o, Z): return Agent2Vec(o) 
        
    def vecT(self, t):
        if t is None: t = 0
        ret= np.zeros(VEC_LEN)
        idx = int(min(VEC_LEN-1, max(0, VEC_LEN /2 + t - env.T)))
        ret[ idx] = 1
        return ret
        
    def toVec(self,  inp):
        if inp is None: return np.zeros(5*VEC_LEN)
         # a said to b that c loves d
         # s=a v=say t=-ed, o=b, txt=(c loves d)
        X = VEC_LEN
        
        v = np.zeros(5*X)
        v[0*X: 1*X] = self.Agent2Vec(inp.s)
        v[1*X: 2*X] = toNPVec[inp.v]
        v[2*X: 3*X] = self.Agent2Vec(inp.o)
        v[3*X: 4*X] = self.vecT(inp.t)              
        v[4*X: 5*X] = np.matmul( self.toVec(inp.txt), COMPRESS  )
        return np.random.random(5*VEC_LEN)
               
    
    def sampleActions(self, qty):  
        ret = []         
        for i in range(10):
            z = torch.rand(VEC_LEN)
            ret.append( policy(torch.cat( (self.state, z))) )            
        ret = list(set(ret))    
        return ret[:qty]                
    
    def getConsequences(self, a ):
        return future( a )
    
         
    def updateRL(self, a,c,u) :
        
        # update policy, NN, 
        #update supervised
        pass  
            
        
    def toZ(self, tens):
        return self.generateRandomEvent(0)
        
    def find_em(self, a):
        if len(self.em)==0: return None
            
        if isinstance(a, Z): 
            a = self.toVec(a)
        d = []
        norma=np.linalg.norm(a)
        for i in range(0, len(self.em)):
            d.append(np.abs(np.linalg.norm(a - self.em[:, i]))/norma)
        idxmin = np.argmin(d)
        nearest = self.em[:, idxmin]
        dist = np.abs(np.linalg.norm(a - nearest))/norma

        return nearest if dist < 0.5 else None
        
        
       
    def getPastEvent(self, dt = -1):
        ev = Z(t = env.T - random.randint(0,10) )
        return self.find_em(ev)
        
    def predictFutureEvent(self):
        if len(self.em_future) ==0: return None
        return random.choice(self.em_future)
    
    def predictFuture(self, a):
        tens = self.future( torch.from_numpy(self.toVec(a)).float() )
        return self.toZ( tens )
    
    def generateRandomEvent(self, i=1):
        ev = Z()
        ev.s = random.choice(pop)
        ev.o = random.choice(pop)
        ev.t = random.choice( [_ed, will] + list(range(env.T-9,env.T+9)) )
        ev.v = random.choice( [mate,give, hit, speak, believe, want])
        if i>0:
            ev.txt = self.generateRandomEvent(i-1)
        return ev    
        
    def getCause(self,a): 
        z = np.random.random(VEC_LEN)
        tens= self.cause(a) 
        return self.toZ( tens )
        
    def getPossibleAction(self,a):
        z = np.random.random(VEC_LEN)
        tens= nn_action(self.Agent2Vec (a))  #, z)
        return self.toZ( tens )     
        
        
    def getBelief(self, a)  :  
        #self.wm[believe] = 1
        #self.wm[a] = 1
        if len(  self.mem[a][want] ) ==0: return None
        return random.choice (self.mem[a][believe] )
        
    def getGoal(self,a):  
        #self.wm[want] = 1
        #self.wm[a] = 1
        if len(  self.mem[a][want] ) ==0: return None
        return random.choice ( self.mem[a][want] ) 
        
    def supervised(self, nn, inp, out):
        pass
    
    def getQuestion(self):
        ev = self.generateRandomEvent()
        ev.mod = '?'
        if coin(0.5):
            op = random.randint(0,4)
            if op==0: ev.s = _
            if op==1: ev.v = _
            if op==2: ev.t = _
            if op==3: ev.o = _            
        return ev
    
    def getCmd(self):
        ev = self.generateRandomEvent()
        ev.mod = '!'
        return ev
        
                
    def addEpMem(self,ev)  :
        self.em.append(ev)
        
        
    def addEpMemPast(self,ev):
        self.em.append(ev)
        
    def  addPssibleFuture(self,ev):
        self.future.append(ev)
        supervised(self.future, state, ev)
                    
    def setPredictModel(self,a,b):
         supervised(self.future, a, b)
        
    def setPossibleAction(self,a, txt):
        supervised(bdi, (a, a_state), txt)
    
        
    def writeBelief(self,a, txt):
        nn[belief] = 1
        nn[a] = 1
        setNN(txt)
        
        
    def writeGoal(self,a, txt) :
        nn[goal] = 1
        nn[a] = 1
        setNN(txt)
        
        ''' nn[a][10:20] += txt'''
        '''nn[a,goal] += txt'''
       
    def answer(self,ev) : 
        out = self.find_em(ev)
        if '_' in [ev.s, ev.v, ev.t, ev.o, ev.txt]:  # wh-               
            if out is None:
                return None
            if '_' == ev.s:  return out.s
            if '_' == ev.v:  return out.v
            if '_' == ev.t:  return out.t
            if '_' == ev.o:  return out.o
            if '_' == ev.txt:  return out.txt                    
        else:   # yes/no
            return 'yes' if out is not None else 'not'
        
        
    def setGoal(self,ev) :
        self.goal.append(ev)
    
    
    
        
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
   
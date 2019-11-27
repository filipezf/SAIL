# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:24:52 2019

@author: u4ft
"""

import env
import random
import numpy as np
import random
import os 
import copy
import itertools


pop = env.pop

def coin(p):
    return random.random() < p
    
'''nn_before = NNSupervised()
nn_after = NNSupervised()
nn_bdi = NNSupervised()
nn_can = NNSupervised()'''
        

# some common-sense 
       #x speak y -> z believe y
    
# model of above
    #nn[my_attention] = 

# play dream imagine
    
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
        return 'a_' + str(self.id)                        
      
    def step(self):  
        actions = self.actions = []
        
        if coin(0.33):
            actions.append( speakDummy() )
            
            

        for inp in self.log:
                   # a said to b that c loves d
                   # s=a v=say t=-ed, o=b, txt=(c loves d)
                   # [ a b c d ab ac ... VEC998 ]
           
            if coin(0.33):
                hearDummy( inp )
             vinp = toVec( inp)    

       
            # model-free
            vout = NN ( vinp )
            
            actions.append( toAction( vout ) )
            
            #model based:        
           #plan:
              vec1 -> vec2    a,b,c attrib
              u = Q(pred)
       # a ignore b - nn -> a like I -> a mate I -> $

              # if, because, try, can, 
              # believe, want -ed, will

    
           x = future(x)
           b,d,i = bdi(b,d,i)
           wm = op(task, wm)
                
    
          
    
          goal: b friend <-
          say to b a lies 
    
         # x.e < 10 => goals.append( feed x) -> c give x
    
           net = [nn_before, nn_after, nn_bdi, nn_can][op-4]
           wm[0] = torch.from_numpy( np.zeros([WM_LEN,]) ).float()
           wm[1] = net( wm[0] ) 
                #if op in           goals.append( wm)
                #if op == 11 :      goals.remove(wm)            
                #if op== 12:        wm = search_em( ctx)               
                #if op==13:         wm = find_goal( g, ctx)
                #if op == 5         wm2 = nn( op_vec, wm) 
    
       
       # compare

       update_weight( task, u)
        
        
        # update supervised networks
        
          supervised(nn_before, wm[1], wm[0])     # nn[before]=1, train
            supervised(nn_after, wm[0], wm[1])
            b= foo[s]
            supervised( can, wm[0] ,b, foo)
            
            #bg, bw = inspect_goal_want(b)
            unsupervised(nn_bdi, bg, bw, foo)
        
        
         self.log = []
     
             
    def input1(self, foo):
        if foo.v == mated:
            self.mated = True
            return       
        s#elf.em_write(foo)
        self.log.append( foo )
        
        
    def speakDummy(self):
        i = random.randint(-1,8)
        
        if i==-1:
            ev = getPastEvent(-1)
            ev.t = None
            return ev
        
        if i==0:
            ev = getPastEvent()
            ev.t = _ed
            return ev
        
        if i==1:
            ev = predictFutureEvent()
            ev.t = will
            return ev
        
        if i==2: 
            a = generateRandomEvent()
            b = predictFuture(a)            
            return Z(s = a, v= IF, txt=b)
            
        if i==3: 
            a = getPastEvent()
            b = getCause(a)            
            return Z(s = a, v= because, txt=b)

        if i==4:
            a = random.choice(pop)
            ev = getPossibleAction(a)
            ev.m = can
            return ev

        if i==5:
            a = random.choice(pop)
            ev = getBelief(a)
            return Z(s = a, v= believe, txt=ev)            
            
        if i==6:
            a = random.choice(pop)
            ev = getGoal(a)
            return Z(s = a, v= want, txt=ev)         
        
        
    def hearDummy(self, ev):
        v = ev.v
        if t== None:
            ev = getPastEvent(-1)
            ev.t = None
            return ev
        
        if t == _ed:
            ev = getPastEvent()
            ev.t = _ed
            return ev
        
        if t == will:
            ev = predictFutureEvent()
            ev.t = will
            return ev
        
        if v==IF: 
            a = generateRandomEvent()
            b = predictFuture(a)            
            return Z(s = a, v= IF, txt=b)
            
        if v==because: 
            a = getPastEvent()
            b = getCause(a)            
            return Z(s = a, v= because, txt=b)

        if m==can:
            a = random.choice(pop)
            ev = getPossibleAction(a)
            ev.m = can
            return ev

        if v==believe:
            a = random.choice(pop)
            ev = getBelief(a)
            return Z(s = a, v= believe, txt=ev)            
            
        if v==want:
            a = random.choice(pop)
            ev = getGoal(a)
            return Z(s = a, v= want, txt=ev)    
        
        
        
        
        
        
        
        
        
        




    
    

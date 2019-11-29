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
            
            
            act = toAction( vout )
            if act.confidence > 0.5:
                actions.append( act )
            
            #model based / plan        
            
            
        if coin(0.3):            
        #if max( [act.confidence for act in actions]) < 0.5:
        
            options = []
            for a  in self.sampleActions(3):                
                c = self.getConsequences(a )
                u = Q(c)
                options.append( a ,u) )
                
            options.sort(key:lambda :x[0]) 
                
            actions.append( options[0][1] )         
        
        
        self.updateRL(a,c,u)        
        self.updateSupervisedNetworks(a,c)
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
            ev = self.getPastEvent(-1)
            ev.t = None
            return ev
        
        if i==0:
            ev = self.getPastEvent()
            ev.t = _ed
            return ev
        
        if i==1:
            ev = self.predictFutureEvent()
            ev.t = will
            return ev
        
        if i==2: 
            a = self.generateRandomEvent()
            b = self.predictFuture(a)            
            return Z(s = a, v= IF, txt=b)
            
        if i==3: 
            a = self.getPastEvent()
            b = self.getCause(a)            
            return Z(s = a, v= because, txt=b)

        if i==4:
            a = random.choice(pop)
            ev = self.getPossibleAction(a)
            ev.m = can
            return ev

        if i==5:
            a = random.choice(pop)
            ev = self.getBelief(a)
            return Z(s = a, v= believe, txt=ev)            
            
        if i==6:
            a = random.choice(pop)
            ev = self.getGoal(a)
            return Z(s = a, v= want, txt=ev)    
        
        if i==7:
            a = self.getQuestion()
            a.mod = '?'
            return a
        
        if i==8:
            a = self.getcmd()
            a.mod = '!'
            return a
        
        
    def hearDummy(self, ev):
        v,t,m,mod = ev.v, ev.t, ev.m, ev.mod
        if t== None:    self.addEpMem(ev)  
        elif t == _ed:    self.addEpMemPast(ev)
        elif t == will:  self. addPssibleFuture(ev)
        elif v == IF:    self.setPredictModel(a,b)
        elif v==because:   self.setPredictModel(a,b)
        elif m==can:      self.setPossibleAction(a, txt)
        elif v==believe:  self.informBelief(a, txt)
        elif v==want:     self.informGoal(a, txt)        
        elif mod == '?':  self.answer(ev)        
        elif mod == '!': self.setGoal(ev)

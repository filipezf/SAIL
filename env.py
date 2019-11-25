# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 12:37:07 2019

@author: u4ft
"""


pop = []
kin = {}

T = 0

#import pytorch
import random
import numpy as np
import agmodel as ag

DNA_LEN = 50
give, hit, mate , speak= 'give hit mate speak'.split()

Agent = ag.Agent


#test agent
cnt=0


def coin(p=0.5):
    return random.random()<p
    

def mix(dna1, dna2):
    return np.random.randint(0,2,DNA_LEN)
    
def create_pop():    
    for i in range(10):
        a = Agent(gender='M')
        b = Agent(gender='F') 
        pop.append(a)
        pop.append(b)       
        a.age = b.age = 20 + 30* random.random()
        a.birth = b.birth =  int(-2000.0*a.age/60)
        
    for i in range(10):
        m = random.choice([x for x in pop if x.gender == 'M'])
        f = random.choice([x for x in pop if x.gender == 'F'])
        c = Agent(m,f)
        c.age = 0 + 30* random.random()
        c.birth =  int(-2000.0*a.age/60)
        pop.append(c)

    for b in pop:
        for c in pop:
            kin[b,c] = max(0, np.dot(b.dna, c.dna)/DNA_LEN -0.5 * 2)


def learn():
    for a in pop:
        a.learn()

def step(T):  
    QTY = 50  
    num_observers =1

    for a in pop:
        #for z in range(15):
        #    a.dt()
        a.step()
        for act in a.action:
            b = act[1]
            if act[0] == give: 
                a.e -= 1
                b.e += 2 if QTY >0 else 1
                QTY -= 1
               # print(a, a.e, b, b.e)
           # if act[0] == hit:  
            #    b.e -= 2
            if act[0] == mate:
                if b.wantmate !=a:
                    a.wantmate = b
                else:
                    if a.gender != b.gender and a.age > 20 and a.age < 50 and b.age>20 and b.age<50:
                        a.input1([a,'mated',b])
                        b.input1([b,'mated',a])
                        m = a if a.gender == 'M'   else b
                        f = a if a.gender == 'F' else b
                        if f. child is None and coin(0.5):
                            f.child = Agent(m,f) 
                            f.child.dna = mix(m.dna, f.dna)
                            f.child.birth = T + 50 # 2000* 0.9
            if act[0] == speak:
                pass
            audience = [a] + random.sample(pop, num_observers) 
            audience += b if isinstance(b, list) else [b]


            for x in audience:
                x.input1([a,act[0], act[1]])


    for a in pop[:]:
        
            
        a.age = int( (T- a.birth)/2000.0 * 60 )
        a.wantmate = None
        if coin(0.01): 
            a.e = int( a.e*random.random())  # sick        
        f = a.age/80
        if coin(f*f*f):                     # old
            a.e -= random.random()*5
        if a.e < 0:
            pop.remove(a)
            #print(a.age)
            for x in pop:
                x.input1([a, 'die'])
                
        if a.child is not None and a.child.birth == T:
            c = a.child
            c.age = 0
            pop.append( c)
            a.child = None
            
           # print(T, len(pop), a.id, a.age, c.id)
            for x in pop:
                x.input1([a,'birth', c])          
        
 

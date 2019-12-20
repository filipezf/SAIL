# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 10:58:32 2019

@author: u4ft
"""

'''
Language aquisition milestones
------------------------------
basic ontology
references (symbols)
n-ary relationships (social, dynamical)
propositions ( true/false statements)
speech acts (interpersonal)
narratives (multi-statement procedural/declarative)
'''


#generic verbs: v1 v2 v3 v20 ?

def train():
    for i in range(500):
        for p in pop:
            train_a(p)
        env.step()
        
        
        
#TODO: option to learn from input speech 
#and reinforcement when speaks relevant text        


def e_level(ag):
    if ag.e < 5: return 'low'
    if ag.e < 10: return 'mid'
    return 'high'


def age_level(ag):
    if ag.e < 20: return 'young'
    if ag.e < 40: return 'adult'
    return 'old'
    


def train_a(ag):    
    
    # bias: hear x -> put x into txt buffer ?
    # bias: extend from single entity to generic entity
    
     a = random.choice(pop)
      
         # agent has mapping from value to axis

    #energy age gender live;dead

    ag.setTxt( a ,'be', 'energy', e_level(a.e) )     # s v o o2
    ag.setTxt( a ,'be', 'age' ,age_level(a.age) )    
    ag.setTxt( a ,'be' ,'gender', a.gender )  
    ag.setTxt( a ,'be', 'live' if a in pop else 'dead' )      

    # give hit

    v = random.choice( ['give' ,'hit'] )
    x,y = random.choice(pop),  random.choice(pop)
    ag.setInput(x ,v, y)
    ag.setLang(  x ,v ,y )
    
    # speak
    
    v = speak
    txt = pickEvent()
    ag.setInput(x, v ,y)             # s v o o2
    ag.setLang(  x , v ,y, txt)
    
    
    #while before after
    
    svo = ag.pickEvent()
    svo2 = ag.pickEvent()
    
    w= WHILE
    if svo.t < svo2.t: w = before
    if svo.t > svo2.t: w = after    
    ag.setLang(  svo ,w ,svo2)
    
    
    # -ed will soon later 
    
    svo = ag.pickEvent()   
    ag.setLang( svo, _ed ) 
        
    svo = createEvent()
    dt = random.randint(0,10)
    w = ''
    if coin(0.5):
        w = soon if dt < 5 else later        
    ag.setTimeout( setInput , dt , svo )
    ag.setLang(svo, will, w)
    
    
    # and or not
    
    a,b,o = random.sample(pop,3)
    v= random.choice( ['give' ,'hit'] )
    
    ag.setInput( [(a ,v, o), (b ,v ,o) ] )
    ag.setLang( (AND ,a, b) ,v, o )
    
    ag.setInput( [(a ,v, o), (b ,v ,o), (b ,v ,o) ] )
    ag.setLang( (AND ,a, b,c) ,v, o )
    
    ag.setInput([a v o, a v2 o ] )
    ag.setLang( a (AND v v2) o )
         
    ag.setLang( (OR a b) v o )
    x = a if coin(0.5) else b
    setTimeout( ag.setInput( x v o)  )

    #    a NOT vo -> b vo c vo
    #    x v y ? NOT
    #    x v y ? YES
        
    

    # if because
    
    svo = ag.pickEvent()
    svo2 = ag.pickCause(svo)
    if svo2 is not None:
         ag.setLang( svo2 ,because, svo )
         
    svo = ag.generateEvent()
    svo2 = ag.imagineFuture(svo)     
    ag.setLang( IF, svo ,svo2 )


    # questions

    # believe want can should
    
    b = ag if coin(0.5) else random.choice(pop)
    ag.setLang( b, believe, b.getBelief() )
    ag.setLang( b, want,  b.getWant() )
    ag.setLang( b, can, getPossibleAction() )
    ag.setLang( b, should, getGoodAction() )
      
     # ask / answer

    #knowledge, uncertainty?
    #    (Permit p, q ,txt) ==  (say p, q ( PERM txt)
        
     
    
    b,c,d,e = random.sample(pop,4)
    v,v1, v2, v3 = random.sample( ['give' ,'hit', 'v1', 'v2', 'v3'],4)
        
    # order
    ag.setInput(b ,speak, (c, v1, '!') )
    ag.setInput(c ,v1 )
    ag.setLang( b ,order ,(c ,v1), t=4 )
    
    ag.setInput( d speak (d, v2, '!') , t=1)
    ag.setInput( e, v3 , t=2)
    ag.setInput( d, hit, e, t=3)    
    ag.setLang( d, order (e ,v2), t=4 )
    
     
     # forbid
     ag.setInput (  b, speak, (c ,NOT, v1, '!') )
     ag.setLang( b ,forbid, c, v1)
     
     
     # member_of
     G = random.sample(pop, 5)     
     ag.setLang( c, member_of, g)   # innate concept ?
          
     
     # join
     ag.setTxt( d ,NOT, member_of, G )
     ag.setLang( d ,join, G)
     ag.setTxt(c ,member_of, G)
     
     #friend
     ag.setTxt( c ,be ,e ,low)
     ag.setInput(b ,give, c)
     ag.setLang( b ,friend, c)
      
     #enemy / antonym
     ag.setTxt( b, NOT, friend, c)
     ag.setTxt( c , 'be' ,e, low)
     ag.setInput( b ,hit, c)
     ag.setLang( b ,enemy, c)
            
     
     # revenge
     ag.setTxt( a, hurt, b)
     ag.setTxt( b ,hurt a )
     ag.setLang( b revenge, a)

        
     #betray
        #  ,(betray a, b)== (believe ,a ,(friend, b,a)) => hurt(b,a)
         
         
     # lie
     ag.setInput( a, v ,b)
     ag.setTxt( c ,believe,( a ,v, b))
     ag.setInput( c ,speak ,(f ,h, d))
     ag.setInput( d ,speak ,(a, h, b))
     ag.setLang( c ,lie )
     ag.setLang( d ,truth_tell )     
       
     # pretend      
     # suppose      
     # communism      
     # free-will
     # definiiton     
     # abstract
     
      'this sentence is false'
     
     #self-referential 
     # infinity of primes
     # twin prime conjecture
     # w-category
     # non-Hausdoff manifold
     
     """justice freedom care holyness loyalty respect accidents, agency, agreement, alternatives, apologies,
     arbitration, attempts, bias, blame, coercion, commensuration, conflict, constraints, conventions,
     costs, crimes, culpability, culture, debt, deception, decisions, dependence, deterrents, distractions,
     domination, duress, duty, escalation, excuses, exoneration, failures, fairness, false beliefs,
     forgiveness, freedom, goals, goodness, identity, ignorance, impairment, impartiality, innocence, 
     intervention, justifications, mental models,mercy, mistakes, moral rules, norms, paragons, passion, 
     persons, plans, pReferences, prohibitions, punishment, recklessness, reparations, reputation,
     retaliation, shame, side-effects, strategies, temptation, tort, trust, universals, utility, 
     values, vengeance, virtues, and will"""
    
    
    ag.setInput( b, die )
    ag.setInput( c ,speak , ag, ( b, will 'o?') )
    
    ag.setInput(c, speak  ,(IF ,(ag ,die), (ag ,will ,'o?' ) ))
    #setInput( c speak 'If you die then what will you do ?')
    
    
    # understand/generate metaphors
    # create narratives


def self_play(N):
     for t in range(N):
        #set random goal ?
        env.step()
   





    
    
    
    

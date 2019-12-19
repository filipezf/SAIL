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


def age_level():
    if ag.e < 20: return 'young'
    if ag.e < 40: return 'adult'
    return 'old'
    
def setLang(ag, ev):
    ag.hear(ev )
    ag.setTxt( ev )
    ag.desiredOut( speak ev ) 

def train_a(ag):    
    
    # bias: hear x -> put x into txt buffer ?
    # bias: extend from single entity to generic entity
    
        a = rA()
      
         # agent has mapping from value to axis
    ag.setTxt( a 'be' 'energy' e_level(a.e) )
    ag.setTxt( a 'be' 'age' age_level(a.age) )    
    ag.setTxt( a 'be' 'gender' a.gender )  
    ag.setTxt( a 'be' 'live' if a in pop else 'dead' )      


    v = give/hit
    ag.setInput(x v y)
   setLang(ag,  x v y )
    
    v = speak
    ag.setInput(x v y)
    setLang(ag,  x speak y txt)
    
    svo = pickEvent()
    svo2 = pickEvent()
    
    w= WHILE
    if svo.t < svo2.t: w = before
    if svo.t > svo2.t: w = after    
    setLang(ag,  svo w svo2)
    
    svo = ag.pickEvent()   
    setLang(ag,  speak svo -ed ) 
        
    svo=createEvent()
    dt = random.randint(0,10)
    w = ''
    if coin(0.5):
        w = soon if dx < 5 else later        
    setTimeout( svo , dt )
    setLang(svo will w)
    
    
    
    ag.setInput([a v o, b v o ] )
    setLang( (AND a b) v o )
    
    ag.setInput([a v o, b v o, cvo ] )
    setLang( (AND a b c) v o )
    
    ag.setInput([a v o, a v2 o ] )
    setLang( a (AND v v2) o )
         
    setLang( (OR a b) v o )
    x = a if coin(0.5) else b
    setTimeout( ag.setInput( x v o)  )

    #    a NOT vo -> b vo c vo
    #    x v y ? NOT
    #    x v y ? YES
        
        
    svo = ag.pickEvent()
    svo2 = ag.pickCause(svo)
    if svo2 is not None:
         setLang( svo2 because svo )
         
    svo = ag.generateEvent()
    svo2 = ag.imagineFuture(svo)     
    setLang( IF svo svo2 )


    # questions

    # TODO
    
    b = ag if coin(0.5) else rA()
    b = rA()
    setLang( b believe b.getBelief() )
    setLang( b want  b.getWant() )
    setLang( b can getPossibleAction() )
    setLang( b should getGoodAction() )
      
     # ask / answer

    #knowledge, uncertainty?
    #    (Permit p, q ,txt) ==  (say p, q ( PERM txt)
        
     
    # order
    setInput(b speak (c v1 !) )
    setInput(c v1 )
    setLang( b order (c v1), t=4 )
    
    setInput( d speak (d v2 !) , t=1)
    setInput( e v3 , t=2)
    setInput( d hit e, t=3)
    
    setLang( d order (e v2), t=4 )
    

     
     # forbid
     setInput (  b speak (c NOT v1!) )
     setLang( b forbid c v1)
     
     
     # member_of
     setLang( c member_of g)
     
     
     
     # join
     setTxt( c NOT member_of G )
     setLang( c join G)
     setTxt(c member_of G)
     
     #friend
      setTxt( c e low)
      setInput(b give c)
      setLang( b friend c)
      
     #enemy / antonym
      setTxt( b NOT friend c)
      setTxt( c e low)
      setInput( b give c)
     setLang( b enemy c)
      
      
     
     # revenge
     setTxt( a hurt b)
     setTxt( b hurt a )
      setLang( b revenge, a)

        
     #betray
     setTxt( a hurt b)
     setTxt( b hurt a )
     setLang( b revenge, a)
      
         ,(betray a, b)== (believe ,a ,(friend, b,a)) => hurt(b,a)
         
         
     # lie
     setInput( a h b)
     setTxt( c believe a h b)
     setInput( c speak f h d)
     setInput( d speak a h b)
     setLang( c lie )
     setLang( d truth_tell )     
       
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
    
    
    setInput( b die -ed)
    setInput(c speak ag ( b will v?) )
    
    ag.setInput((c speak  IF (ag die) (ag will v?) ))
    #setInput( c speak 'If you die then you will ?')
    
    
    # understand/generate metaphors
    # create narratives


def self_play(N):
     for t in range(N):
        #set random goal ?
        env.step()
   

import env
import agmodel
#import lang


if __name__ == "__main__":
    
    env.create_pop()
    #print( agmodel.op2txt(  agmodel.generategoal( )) )
    
   # x = 1/0
    #agmodel.load()
    
    
    for a in env.pop:
        a.add_goal( agmodel.generategoal() )
        a.add_goal( agmodel.generategoal() )
        
    for t in range(2):
        env.step(t)
       
   # agmodel.save('v1.pk')
   
   
   
   
            

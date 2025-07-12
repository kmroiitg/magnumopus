import numpy as np
import pandas as pd

def readx(h,columns,file):
    name = np.arange(columns)
    name = [str(i) for i in name ] 
    df = pd.read_csv(file, sep='\s+', header=h,names = name)
    matrix = df.values

    return matrix

    
            


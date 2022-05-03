# Enter your code here. Read input from STDIN. Print output to STDOUT
from scipy import stats

def spearmanRankCorrelationCoefficient(x,y):
    xSort=sorted(x)
    ySort=sorted(y)
    rankOfX={}
    rankOfY={}
    for i in xSort:
        rankOfX[i]= xSort.index(i)+1
    
    for i in ySort:
        rankOfY[i]= ySort.index(i)+1
    length= len(x)         
    d=[]
    for i in range(length):
        temp=rankOfX[x[i]]- rankOfY[y[i]]
        d.append(temp**2)
                
    numerator = 6* sum(d)
    denominator = length * ((length ** 2 ) -1)         
    
    return round(1-(numerator/float(denominator)),3)
# https://github.com/abrahamalbert18/HackerRank-Solutions-in-Python/blob/master/Day%207:%20Spearman's%20Rank%20Correlation%20Coefficient.py
# n=input()
# x=map(float,raw_input().split())
# y=map(float,raw_input().split())
if __name__ == "__main__":
    x = [1,2,3]
    y = [1,2,3]
                
    # s = spearmanRankCorrelationCoefficient(x,y)      
    score = stats.spearmanr(x,y)
    print(score)  
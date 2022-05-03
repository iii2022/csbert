import numpy as np
import matplotlib.pyplot as plt
import pickle
# import required libraries
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
 
#   https://www.askpython.com/python/normal-distribution
# creating the dataset
# data = {'C':20, 'C++':15, 'Java':30,
#         'Python':35}
# courses = list(data.keys())
# values = list(data.values())
with open("stc.pkl", "rb") as f:
    score = pickle.load(f)
# courses = []
# values = []
mean, std = 0,0
max1,min1 = 0,0
for k,v in score.items():
    # print(v)
    x = np.array(v)
    v = x.astype(np.float)
    # for js in v:
    #     courses

    # courses = v
    # values = 
    #Calculate mean and Standard deviation.
    # print(v)
    mean = np.mean(v)
    std = np.std(v)
    max1 = np.amax(v)
    min1 = np.amin(v)
    # break
 
    # Creating the distribution
    # data = np.arange(1,10,0.01)
    print(min1,max1,mean,std)
    data = np.arange(min1,max1, 0.01)
    pdf = norm.pdf(data , loc = mean , scale = std )
    
    #Visualizing the distribution
    
    sb.set_style('whitegrid')
    sb.lineplot(data, pdf , color = 'black')
    plt.xlabel('Score')
    plt.ylabel('Probability Density')
    plt.title(k[3:])
    
    # fig = plt.figure(figsize = (10, 5))
    
    # # creating the bar plot
    # plt.bar(courses, values, color ='maroon',
    #         width = 0.4)
    
    # plt.xlabel("Courses offered")
    # plt.ylabel("No. of students enrolled")
    # plt.title("Students enrolled in different courses")
    # # plt.show()
    plt.savefig("fig/{}.png".format(k[3:]))
    plt.clf()
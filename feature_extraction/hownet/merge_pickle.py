import pickle
from commons.util import list_files

if __name__ == "__main__":
    fs = list_files("./conceptnet.hownet/hassubevent-aligment.pkl/","*.pkl")
    # fs = list_files("./usedfor_man/man/","*.pkl")
    res = []
    for name in fs:
        with open(name,"rb") as f:
            data = pickle.load(f)
            res += data
    
    # with open("./usedfor_man/test.pkl","wb") as f:
    with open("conceptnet.hownet/HasSubevent.pn.model.pkl","wb") as f:
        pickle.dump(res, f)

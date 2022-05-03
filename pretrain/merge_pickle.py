import pickle, os
from glob import glob

# https://stackoverflow.com/questions/18394147/recursive-sub-folder-search-and-return-files-in-a-list-python
def list_files(directory, suffix = '*.txt'):
	# directory/     the dir
	# **/       every file and dir under my_path
	# *.txt     every file that ends with '.txt'
	return sorted(glob(directory + '/**/%s' % suffix, recursive=True))


if __name__ == "__main__":
    # fs = list_files("/ldata/name/common/PartOf_pandn_instance_noxiaoqi/","*.pkl")
    fs = list_files("/ldata/name/common/Causes_pandn_instance/","*.pkl")
    # fs = list_files("./usedfor_man/man/","*.pkl")
    res = []
    for name in fs:
        with open(name,"rb") as f:
            data = pickle.load(f)
            res += data
    
    # with open("./usedfor_man/test.pkl","wb") as f:
    os.makedirs("/ldata/name/common/Causes_pandn_instance/all/", exist_ok=True)
    with open("/ldata/name/common/Causes_pandn_instance/all/test.pkl","wb") as f:
        pickle.dump(res, f)

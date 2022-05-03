import gzip, json, re, os
import numpy as np
from tqdm import tqdm


def change(files, reweight):
    res = {}
    c = 0
    for relation, fp in files.items():
        predictions = np.load(fp, allow_pickle=True).item()
        for id_, score_ in tqdm(predictions.items()):
            if "PartOf" in id_:
                c += 1
            if id_ in reweight:
                # # ########## CSBERT < score
                # if float(reweight[id_][2]) >= 1:
                #     if score_ < 0.5 and "PartOf" in id_:
                #         print(id_, reweight[id_][2], score_)
                # else:
                #     if float(reweight[id_][2]) - score_ > 0.5 and "PartOf" in id_:
                #         print(id_, reweight[id_][2], score_)
                scaler = 0.296
                #### CSBERT > score
                # if score_ - float(reweight[id_][2]) * scaler > 0.01 and "PartOf" in id_:
                #     print(id_, float(reweight[id_][2]) * scaler, score_)
                # #### CSBERT < score
                if score_ - float(reweight[id_][2]) * scaler > 0.01 and "PartOf" in id_:
                    # print(id_, float(reweight[id_][2]) * scaler, score_)
                    res[id_] = [float(reweight[id_][2]) * scaler, score_]
    print("=====", c)
    return res


if __name__ == "__main__":
    print("load reweight")
    reweight = np.load("reweight.pkl.npy", allow_pickle=True).item()

    # 循环载入每一种predict关系类型的 {id:score}，更新score
    # bert + hownet
    files = {
        # "/r/RelatedTo":"/ldata/name/common/transformers-master/examples/text-classification/reweight_relation.pkl.npy",
        "/r/Synonym":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/synonym-noxiaoqi/reweight_relation.pkl.npy",
        "/r/Antonym":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/antonym-noxiaoqi/reweight_relation.pkl.npy",
        "/r/DerivedFrom":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/derivedfrom-noxiaoqi/reweight_relation.pkl.npy",
        "/r/HasSubevent":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/hassubevent-noxiaoqi/reweight_relation.pkl.npy",
        "/r/UsedFor":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/usedfor-noxiaoqi/reweight_relation.pkl.npy",
        "/r/AtLocation":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/atlocation-noxiaoqi/reweight_relation.pkl.npy",
        "/r/FormOf":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/formof-noxiaoqi/reweight_relation.pkl.npy",
        "/r/IsA":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/isa-noxiaoqi/reweight_relation.pkl.npy",
        "/r/CapableOf":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/capableof-noxiaoqi/reweight_relation.pkl.npy",
        "/r/HasContext":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/hascontext-noxiaoqi/reweight_relation.pkl.npy",
        "/r/PartOf":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/partof-noxiaoqi/reweight_relation.pkl.npy",
        "/r/Causes":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/causes-noxiaoqi/reweight_relation.pkl.npy",
        "/r/HasPrerequisite":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/hasprerequisite-noxiaoqi/reweight_relation.pkl.npy",
    }
    hownet_bert = change(files, reweight)

    # only hownet
    files = {
        # "/r/RelatedTo":"/ldata/name/common/transformers-master/examples/text-classification/reweight_relation.pkl.npy",
        "/r/Synonym":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/synonym-onlyhownet/reweight_relation.pkl.npy",
        "/r/Antonym":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/antonym-onlyhownet/reweight_relation.pkl.npy",
        "/r/DerivedFrom":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/derivedfrom-onlyhownet/reweight_relation.pkl.npy",
        "/r/HasSubevent":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/hassubevent-onlyhownet/reweight_relation.pkl.npy",
        "/r/UsedFor":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/usedfor-onlyhownet/reweight_relation.pkl.npy",
        "/r/AtLocation":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/atlocation-onlyhownet/reweight_relation.pkl.npy",
        "/r/FormOf":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/formof-onlyhownet/reweight_relation.pkl.npy",
        "/r/IsA":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/isa-onlyhownet/reweight_relation.pkl.npy",
        "/r/CapableOf":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/capableof-onlyhownet/reweight_relation.pkl.npy",
        "/r/HasContext":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/hascontext-onlyhownet/reweight_relation.pkl.npy",
        "/r/PartOf":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/partof-onlyhownet/reweight_relation.pkl.npy",
        "/r/Causes":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/causes-onlyhownet/reweight_relation.pkl.npy",
        "/r/HasPrerequisite":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/hasprerequisite-onlyhownet/reweight_relation.pkl.npy",
    }

    print("updating ")
    hownet = change(files, reweight)


    # print(err, "not in reweight")


    # only triple
    files = {
        # "/r/RelatedTo":"/ldata/name/common/transformers-master/examples/text-classification/reweight_relation.pkl.npy",
        "/r/Synonym":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/synonym-onlytriple/reweight_relation.pkl.npy",
        "/r/Antonym":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/antonym-onlytriple/reweight_relation.pkl.npy",
        "/r/DerivedFrom":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/derivedfrom-onlytriple/reweight_relation.pkl.npy",
        "/r/HasSubevent":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/hassubevent-onlytriple/reweight_relation.pkl.npy",
        "/r/UsedFor":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/usedfor-onlytriple/reweight_relation.pkl.npy",
        "/r/AtLocation":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/atlocation-onlytriple/reweight_relation.pkl.npy",
        "/r/FormOf":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/formof-onlytriple/reweight_relation.pkl.npy",
        "/r/IsA":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/isa-onlytriple/reweight_relation.pkl.npy",
        "/r/CapableOf":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/capableof-onlytriple/reweight_relation.pkl.npy",
        "/r/HasContext":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/hascontext-onlytriple/reweight_relation.pkl.npy",
        "/r/PartOf":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/partof-onlytriple/reweight_relation.pkl.npy",
        "/r/Causes":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/causes-onlytriple/reweight_relation.pkl.npy",
        "/r/HasPrerequisite":"/ldata/name/common/transformers-master/examples/text-classification/predictions-softmax/hasprerequisite-onlytriple/reweight_relation.pkl.npy",
    }

    bert = change(files, reweight)

    res = []
    c = 0
    
    for k,v in hownet.items():

        # if k in bert and k in hownet_bert:
        if k in bert:
            if bert[k][1] - v[1] < -0.2:
                print(k,v[0],'hownet',v[1],"bert",bert[k][1])
        # if True:
            # res.append([k, v[0], v[1], bert[k][1]])
            # print([k, v[0], "hownet", v[1], "bert", bert[k][1], "hownet-bert" , hownet_bert[k][1]])
            # c += 1
    # print(len(hownet_bert), len(hownet), len(bert))
    # print(1.0*c/len(hownet), c)
    
import pickle
import os;
import treelib;
import  re;
import bz2;
import os;
import urllib.request;
import json;
import numpy as np


from treelib import Node, Tree;
global id_num


def get_nokuohao_sememe(target):
    begin=target.find("{");
    end=target.find("}")
    sememe=target[begin+1:end]
    if(end<=begin):
        print("get_nokuohao_sememe wrong")
    return sememe;

get_nokuohao_sememe("{1}232")

#创造节点的时候注意节点id唯一性。HowNet Taxonomy EventRoleAndFeatures.txt中的cause出现了两次。

def match_kuohao(s,begin):#输入字符串s，位置begin，如果当前位置不是左括号，返回当前位置，否则按照左右括号加减法，返回第一个终结右括号的位置。
    if(s[begin:begin+1]!='{'):#长度越界的话依然判断成立
        return begin;
    else:
        i=1;
        j=begin+1;
        while((i>0)&(j<len(s))):
            if(s[j:j+1]=='{'):
                i=i+1;
            elif(s[j:j+1]=='}'):
                i=i-1;
            j=j+1;
    return j-1;

def entity_def_analyse1(s):#输入：{AnimalHuman|动物:HostOf={Ability|能力}{Name|姓名}{Wisdom|智慧},{speak|说:agent={~}},{think|思考:agent={~}}}
    any=[];
    if((s.find(":"))<0):      #如果只有第一特征，没有冒号，则只返回第一特征.如果连第一特征都没有，则同样返回长度为1的列表，只不过第一个元素为空字符串。
        # s=s.replace('{','');
        # s=s.replace("}","");
        any.append(s);
        return any;

    #any.append(s);#如果存在冒号，首先数组第一项为描述的这个句子本身。
    any.append(s[s.find("{"):s.find(":")]+"}");#数组第一项为第一特征,{thing|万物},带大括号
    s=s[s.find(":")+1:s.rfind("}")];#s变为第一特征后面的部分
    begin=0;
    i=0;
    while(i<len(s)):
        if(s[int(i):int(i+1)]==","):   #如果遇到逗号，那么RelateTo={tree|树},{eat|吃:patient={~}},这两类进入数组
            any.append(s[int(begin):int(i)]);
            begin=i+1;
        if (s[int(i):int(i + 1)] == "{"):#如果先遇到左括号，那么遍历到右括号为止。
            left=1;
            while (left>0&i<len(s)):
                i=i+1;
                if (s[int(i):int(i + 1)] == "{"):
                    left=left+1;
                if (s[int(i):int(i + 1)] == "}"):
                    left=left-1;
        if (((i + 1) == len(s)) & (i >= begin)):#最后一项没有逗号，根据字符串长度来判断结束
            any.append(s[int(begin):int(i + 1)]);
            begin=i+1;
        i=i+1;
   # print(any);
    return any;

def entity_def_analyse2(data,id,f):#输入的data为analyse1返回的结果，id为该行所定义的词语，用大括号括起来。
    if(len(data)==1):#如果只有第一特征或者第一特征都是空字符串，那么返回长度为1的dict。
        dict={'class':data[0]};
        dict['id']=id;
        return dict;
    dict = {'class':data[0]};#class代表第一特征
    dict['id'] = id;#id代表改行所定义的单词，用大括号括起来
    i=1;
    j=0;
    while(i<len(data)):#data：<class 'list'>: ['{thing|万物}', 'HostOf={Appearance|外观}', '{perception|感知:content={~}}']
        data[i] = data[i].replace('{~}', id);#将句子中的~用词语替代
        if((data[i][0]!='{')&((data[i].find("=")) > 0)):#第一种情况，HostOf={Ability|能力}{Name|姓名}{Wisdom|智慧},{speak|说:agent={~}},
            begin=data[i].find('=')+1;
            end=match_kuohao(data[i],begin);
            k=0;
            while((end>begin)&(data[i][end:end+1]=='}')):#循环内关系（谓词）都是一样的，HostOf={Ability|能力}{Name|姓名}{Wisdom|智慧}

                if(data[i][begin:end].find(':')>0):#whole的情况 {fittings|配件} {component|部分:modifier={other|另}{unnecessary|不必要},whole={entity|实体:{contain|包含:OfPart={~}}}}
                    id_value=entity_def_analyse1(data[i][begin:end+1])[0];
                    f.write("    &&    "+id+"    "+data[i][0:data[i].find("=")]+"    "+id_value);
                    dict[chr(48+k)+"    "+data[i][0:data[i].find("=")]]=entity_def_analyse2(entity_def_analyse1(data[i][begin:end+1]),id_value,f);
                else:
                    f.write("    &&    "+id + "    " + data[i][0:data[i].find("=")] + "    " + data[i][begin:end+1]);
                    dict[data[i][0:data[i].find("=")] +"    "+chr(48 + k)] =data[i][begin:end+1];
                begin=end+1;
                end=match_kuohao(data[i],begin);
                k=k+1;
        elif(data[i][0]=='{'):
            id_value=entity_def_analyse1(data[i])[0];
            dict[chr(48+j)+"    "+'identifier']=entity_def_analyse2(entity_def_analyse1(data[i]),id_value,f);
            j=j+1;


        i=i+1;
    return dict;

class Word:#增加node的可打印项
    def __init__(self,data,id):
        self.definition=data;
        self.id=id;

def get_entity():
    s2 = "/Users/name/Desktop/学习3/HowNet 2012/Data/HowNet Taxonomy Entity.txt";
    f2 = open(s2, "r", encoding="GB18030", errors="ignore");
    s3 = "/Users/name/Desktop/学习3/HowNet 2012/Data/1Entity.txt";
    f3 = open(s3, "w");
    entity = Tree();
    for j in range(8000000):
        line1 = f2.readline();
        if (len(line1) == 0):
            break;
            # print(line1);
        begin = line1.find("{");
        end = line1.find("}");
        sense = line1[begin:end + 1];
        f3.write(sense);
        data = line1[line1.find("{", begin + 1):line1.rfind('}') + 1];
        data = entity_def_analyse1(data);
        i = line1.find("{") / 2 - 1;
        i = int(i);  # 获得当前行被定义词语的层级深度，根节点为0
        if (i == 0):
            entity.create_node(sense, sense.lower());
            node = entity.get_node(sense.lower());
            node.data = Word(entity_def_analyse2(data, sense.lower(), f3),sense.lower());#node的可打印项。node本身和id
            # print(node.data);
            path = [];
            path.append(entity.get_node(sense.lower()));
        else:
            f3.write("    &&    " + sense.lower() + "    " + "subclass of" + "    " + path[i - 1].data.id);
            entity.create_node(sense, sense.lower(), parent=path[i - 1], data=line1[end + 1:len(line1) - 1]);#此处的data是乱写的，马上更新
            node = entity.get_node(sense.lower());
            node.data = Word(entity_def_analyse2(data, sense.lower(), f3),sense.lower());
            ##print(node.data)
            if (i <= (len(path) - 1)):
                path[i] = entity.get_node(sense.lower());
            else:
                path.append(entity.get_node(sense.lower()));

        f3.write("\n");

    f2.close();
    f3.close();
    return entity



def event_def_analyse1(s):#输入：{AnimalHuman|动物:HostOf={Ability|能力}{Name|姓名}{Wisdom|智慧},{speak|说:agent={~}},{think|思考:agent={~}}}
    any=[];
    if((s.find(":"))<0):      #如果只有第一特征，没有冒号，则只返回第一特征.如果连第一特征都没有，则同样返回长度为1的列表，只不过第一个元素为空字符串。
        # s=s.replace('{','');
        # s=s.replace("}","");
        any.append(s);
        return any;

    #any.append(s);#如果存在冒号，首先数组第一项为描述的这个句子本身。
    any.append(s[s.find("{"):s.find(":")]+"}");#数组第一项为第一特征,{thing|万物},带大括号
    s=s[s.find(":")+1:s.rfind("}")];#s变为第一特征后面的部分
    begin=0;
    i=0;
    while(i<len(s)):
        if(s[int(i):int(i+1)]==","):   #如果遇到逗号，那么RelateTo={tree|树},{eat|吃:patient={~}},这两类进入数组
            any.append(s[int(begin):int(i)]);
            begin=i+1;
        if (s[int(i):int(i + 1)] == "{"):#如果先遇到左括号，那么遍历到右括号为止。
            left=1;
            while (left>0&i<len(s)):
                i=i+1;
                if (s[int(i):int(i + 1)] == "{"):
                    left=left+1;
                if (s[int(i):int(i + 1)] == "}"):
                    left=left-1;
        if (((i + 1) == len(s)) & (i >= begin)):#最后一项没有逗号，根据字符串长度来判断结束
            any.append(s[int(begin):int(i + 1)]);
            begin=i+1;
        i=i+1;
   # print(any);
    return any;

def event_def_analyse2(data,id,f):#输入的data为analyse1返回的结果，id为该行所定义的词语，用大括号括起来。
    if(len(data)==1):#如果只有第一特征或者第一特征都是空字符串，那么返回长度为1的dict。
        dict={'class':data[0]};
        dict['id']=id;
        return dict;
    dict = {'class':data[0]};#class代表第一特征
    dict['id'] = id;#id代表改行所定义的单词，用大括号括起来
    i=1;
    j=0;
    while(i<len(data)):#data：<class 'list'>: ['{thing|万物}', 'HostOf={Appearance|外观}', '{perception|感知:content={~}}']
        data[i] = data[i].replace('{~}', id);#将句子中的~用词语替代
        if((data[i][0]!='{')&((data[i].find("=")) > 0)):#第一种情况，HostOf={Ability|能力}{Name|姓名}{Wisdom|智慧},{speak|说:agent={~}},
            begin=data[i].find('=')+1;
            end=match_kuohao(data[i],begin);
            k=0;
            while((end>begin)&(data[i][end:end+1]=='}')):#循环内关系（谓词）都是一样的，HostOf={Ability|能力}{Name|姓名}{Wisdom|智慧}

                if(data[i][begin:end].find(':')>0):#whole的情况 {fittings|配件} {component|部分:modifier={other|另}{unnecessary|不必要},whole={entity|实体:{contain|包含:OfPart={~}}}}
                    id_value=entity_def_analyse1(data[i][begin:end+1])[0];
                    f.write("    &&    "+id+"    "+data[i][0:data[i].find("=")]+"    "+id_value);
                    dict[chr(48+k)+"    "+data[i][0:data[i].find("=")]]=entity_def_analyse2(entity_def_analyse1(data[i][begin:end+1]),id_value,f);
                else:
                    f.write("    &&    "+id + "    " + data[i][0:data[i].find("=")] + "    " + data[i][begin:end+1]);
                    dict[data[i][0:data[i].find("=")] +"    "+chr(48 + k)] =data[i][begin:end+1];
                begin=end+1;
                end=match_kuohao(data[i],begin);
                k=k+1;
        elif(data[i][0]=='{'):
            id_value=entity_def_analyse1(data[i])[0];
            dict[chr(48+j)+"    "+'identifier']=entity_def_analyse2(entity_def_analyse1(data[i]),id_value,f);
            j=j+1;


        i=i+1;
    return dict;

def get_event():
    s2 = "/Users/name/Desktop/学习3/HowNet 2012/Data/HowNet Taxonomy Event.txt";
    f2 = open(s2, "r", encoding="GB18030", errors="ignore");
    s3 = "/Users/name/Desktop/学习3/HowNet 2012/Data/1Event.txt";
    f3 = open(s3, "w");
    entity = Tree();
    for j in range(1000000):
        line1 = f2.readline();
        if (len(line1) == 0):
            break;
        #print(line1);
        begin = line1.find("{");
        end = line1.find("}");
        sense = line1[begin:end + 1];
        f3.write(sense);
        data = line1[line1.find("{", begin + 1):line1.rfind('}') + 1];
        #print(data);
        i = line1.find("{") / 2 - 1;
        i = int(i);
        if (i == 0):
            data = event_def_analyse1(data);
            entity.create_node(sense+"    "+chr(48), sense.lower()+"    "+chr(48));
            node = entity.get_node(sense.lower()+"    "+chr(48));
            node.data = Word(event_def_analyse2(data, sense.lower()+"    "+chr(48),f3),sense.lower());
            #print(node.data);
            path = [];
            path.append(entity.get_node(sense.lower()+"    "+chr(48)));
        else:

            f3.write("    &&    "+sense+"    "+"subclass of"+"    "+path[i-1].data.id);
            k=0;

            while(data.find(';')>=0):
                data1=data[0:data.find(';')];
                data=data[data.find(';')+1:len(data)];
                entity.create_node(sense+"    "+chr(48+k), sense.lower()+"    "+chr(48+k), parent=path[i - 1]);
                node = entity.get_node(sense.lower()+"    "+chr(48+k));
                data1 = event_def_analyse1(data1);
                node.data = Word(event_def_analyse2(data1, sense.lower()+"    "+chr(48+k),f3),sense);
                if (i <= (len(path) - 1)):
                    path[i] = entity.get_node(sense.lower()+"    "+chr(48));
                else:
                    path.append(entity.get_node(sense.lower()+"    "+chr(48)));
                k=k+1;
                f3.write("######");

            data1 = data;
            entity.create_node(sense+"    "+chr(48+k), sense.lower() + "    " + chr(48 + k), parent=path[i - 1]);
            node = entity.get_node(sense.lower() + "    " + chr(48 + k));
            data1 = event_def_analyse1(data1);
            node.data = Word(event_def_analyse2(data1, sense.lower()+"    "+chr(48+k),f3),sense);
            if (i <= (len(path) - 1)):
                path[i] = entity.get_node(sense.lower()+"    "+chr(48));
            else:
                path.append(entity.get_node(sense.lower()+"    "+chr(48)));

        f3.write("\n");




    f2.close();
    f3.close();
    return entity;

def get_attribute():
    s2 = "/Users/name/Desktop/学习3/HowNet 2012/Data/HowNet Taxonomy Attribute.txt";
    f2 = open(s2, "r", encoding="GB18030", errors="ignore");
    s3 = "/Users/name/Desktop/学习3/HowNet 2012/Data/1Attribute.txt";
    f3 = open(s3, "w");
    entity = Tree();
    for j in range(8000000):
        line1 = f2.readline();
        if (len(line1) == 0):
            break;
            # print(line1);
        begin = line1.find("{");
        end = line1.find("}");
        sense = line1[begin:end + 1];
        f3.write(sense);
        data = line1[line1.find("{", begin + 1):line1.rfind('}') + 1];
        data = entity_def_analyse1(data);
        i = line1.find("{") / 2 - 1;
        i = int(i);  # 获得当前行被定义词语的层级深度，根节点为0
        if (i == 0):
            entity.create_node(sense, sense.lower());
            node = entity.get_node(sense.lower());
            node.data = Word(entity_def_analyse2(data, sense.lower(), f3), sense.lower());  # node的可打印项。node本身和id
            # print(node.data);
            path = [];
            path.append(entity.get_node(sense.lower()));
        else:
            f3.write("    &&    " + sense.lower() + "    " + "subclass of" + "    " + path[i - 1].data.id);
            entity.create_node(sense, sense.lower(), parent=path[i - 1],
                               data=line1[end + 1:len(line1) - 1]);  # 此处的data是乱写的，马上更新
            node = entity.get_node(sense.lower());
            node.data = Word(entity_def_analyse2(data, sense.lower(), f3), sense.lower());
            ##print(node.data)
            if (i <= (len(path) - 1)):
                path[i] = entity.get_node(sense.lower());
            else:
                path.append(entity.get_node(sense.lower()));

        f3.write("\n");

    f2.close();
    f3.close();
    return entity

def get_attributevalue():
    s2 = "/Users/name/Desktop/学习3/HowNet 2012/Data/HowNet Taxonomy AttributeValue.txt";
    f2 = open(s2, "r", encoding="GB18030", errors="ignore");
    s3 = "/Users/name/Desktop/学习3/HowNet 2012/Data/1AttributeValue.txt";
    f3 = open(s3, "w");
    entity = Tree();
    for j in range(8000000):
        line1 = f2.readline();
        if (len(line1) == 0):
            break;
            # print(line1);
        begin = line1.find("{");
        end = line1.find("}");
        sense = line1[begin:end + 1];
        f3.write(sense);
        data = line1[line1.find("{", begin + 1):line1.rfind('}') + 1];
        data = entity_def_analyse1(data);
        i = line1.find("{") / 2 - 1;
        i = int(i);  # 获得当前行被定义词语的层级深度，根节点为0
        if (i == 0):
            entity.create_node(sense, sense.lower());
            node = entity.get_node(sense.lower());
            node.data = Word(entity_def_analyse2(data, sense.lower(), f3), sense.lower());  # node的可打印项。node本身和id
            # print(node.data);
            path = [];
            path.append(entity.get_node(sense.lower()));
        else:
            f3.write("    &&    " + sense.lower() + "    " + "subclass of" + "    " + path[i - 1].data.id);
            entity.create_node(sense, sense.lower(), parent=path[i - 1],
                               data=line1[end + 1:len(line1) - 1]);  # 此处的data是乱写的，马上更新
            node = entity.get_node(sense.lower());
            node.data = Word(entity_def_analyse2(data, sense.lower(), f3), sense.lower());
            ##print(node.data)
            if (i <= (len(path) - 1)):
                path[i] = entity.get_node(sense.lower());
            else:
                path.append(entity.get_node(sense.lower()));

        f3.write("\n");

    f2.close();
    f3.close();
    return entity

def get_SecondaryFeature():
    s2 = "/Users/name/Desktop/学习3/HowNet 2012/Data/HowNet Taxonomy SecondaryFeature.txt";
    f2 = open(s2, "r", encoding="GB18030", errors="ignore");
    s3 = "/Users/name/Desktop/学习3/HowNet 2012/Data/1SecondaryFeature.txt";
    f3 = open(s3, "w");
    entity = Tree();
    for j in range(8000000):
        line1 = f2.readline();
        if (len(line1) == 0):
            break;
            # print(line1);
        begin = line1.find("{");
        end = line1.find("}");
        sense = line1[begin:end + 1];
        f3.write(sense);
        data = line1[line1.find("{", begin + 1):line1.rfind('}') + 1];
        data = entity_def_analyse1(data);
        i = line1.find("{") / 2 - 1;
        i = int(i);  # 获得当前行被定义词语的层级深度，根节点为0
        if (i == 0):
            entity.create_node(sense, sense.lower());
            node = entity.get_node(sense.lower());
            node.data = Word(entity_def_analyse2(data, sense.lower(), f3), sense.lower());  # node的可打印项。node本身和id
            # print(node.data);
            path = [];
            path.append(entity.get_node(sense.lower()));
        else:
            f3.write("    &&    " + sense.lower() + "    " + "subclass of" + "    " + path[i - 1].data.id);
            entity.create_node(sense, sense.lower(), parent=path[i - 1],
                               data=line1[end + 1:len(line1) - 1]);  # 此处的data是乱写的，马上更新
            node = entity.get_node(sense.lower());
            node.data = Word(entity_def_analyse2(data, sense.lower(), f3), sense.lower());
            ##print(node.data)
            if (i <= (len(path) - 1)):
                path[i] = entity.get_node(sense.lower());
            else:
                path.append(entity.get_node(sense.lower()));

        f3.write("\n");

    f2.close();
    f3.close();
    return entity

def get_EventRoleAndFeatures():
    s2 = "/Users/name/Desktop/学习3/HowNet 2012/Data/HowNet Taxonomy EventRoleAndFeatures.txt";
    f2 = open(s2, "r", encoding="GB18030", errors="ignore");
    s3 = "/Users/name/Desktop/学习3/HowNet 2012/Data/1EventRoleAndFeatures.txt";
    f3 = open(s3, "w");
    entity = Tree();
    for j in range(8000000):
        line1 = f2.readline();
        if (len(line1) == 0):
            break;
            # print(line1);
        begin = line1.find("{");
        end = line1.find("}");
        sense = line1[begin:end + 1];
        f3.write(sense);
        data = line1[line1.find("{", begin + 1):line1.rfind('}') + 1];
        data = entity_def_analyse1(data);
        i = line1.find("{") / 2 - 1;
        i = int(i);  # 获得当前行被定义词语的层级深度，根节点为0
        if (i == 0):
            entity.create_node(sense, sense.lower());
            node = entity.get_node(sense.lower());
            node.data = Word(entity_def_analyse2(data, sense.lower(), f3), sense.lower());  # node的可打印项。node本身和id
            # print(node.data);
            path = [];
            path.append(entity.get_node(sense.lower()));
        else:
            f3.write("    &&    " + sense.lower() + "    " + "subclass of" + "    " + path[i - 1].data.id);
            entity.create_node(sense, sense.lower(), parent=path[i - 1],
                               data=line1[end + 1:len(line1) - 1]);  # 此处的data是乱写的，马上更新
            node = entity.get_node(sense.lower());
            node.data = Word(entity_def_analyse2(data, sense.lower(), f3), sense.lower());
            ##print(node.data)
            if (i <= (len(path) - 1)):
                path[i] = entity.get_node(sense.lower());
            else:
                path.append(entity.get_node(sense.lower()));

        f3.write("\n");

    f2.close();
    f3.close();







sememe_2000_embedding={};
def get_sememe_2000_embedding(sememe_2000_embedding,tree_Taxonomy,root_identifier):

    nodes = tree_Taxonomy.all_nodes()
    root = tree_Taxonomy.get_node(root_identifier);
    for node1 in nodes:
        vec={};
        par1 = node1;
        rate_now=1.0;
        vec[get_nokuohao_sememe(par1.tag)] = rate_now;
        rate_now=0*rate_now;
        while (par1 != root):
            par1 = tree_Taxonomy.parent(par1.identifier);
            rate_now=rate_now/len(tree_Taxonomy.children(par1.identifier))
            vec[get_nokuohao_sememe(par1.tag)]=rate_now
        sememe_2000_embedding[get_nokuohao_sememe(node1.tag)]=vec;


    return sememe_2000_embedding;

def get_sememe2index(sememe_2000_embedding):
    sememe2index = {};
    index=0;
    for sememe in sememe_2000_embedding:
        sememe2index[sememe] =index;
        index+=1;
    return sememe2index;

def get_sememe_2000_embedding_array(sememe_2000_embedding,sememe2index):
    length=len(sememe2index);
    sememe_2000_embedding_array={}
    for sememe in sememe_2000_embedding:
        vec = [0.0 for _ in range(length)]
        data=sememe_2000_embedding[sememe];
        for sememe2 in data:
            vec[sememe2index[sememe2]]=data[sememe2];
        vec=np.array(vec)
        sememe_2000_embedding_array[sememe]=vec;
    return sememe_2000_embedding_array;


def get_tree_html(tree_dict_node,tree_ago,identifier):
    global id_num
    id_num+=1;
    children=tree_ago.children(identifier)
    tree_dict_node["id"] = id_num
    tag=tree_ago.get_node(identifier).tag
    begin=tag.find("{")
    end=tag.rfind("}")
    tag=tag[begin+1:end]
    tree_dict_node["name"]=tag;
    tree_dict_node["nodes"]=[]

    for node_i in children:
        dict_i={}
        tree_dict_node["nodes"].append(dict_i)
        identifier_i=node_i.identifier
        get_tree_html(dict_i,tree_ago,identifier_i)







id_num=0

entity=get_entity();
event=get_event();
attribute=get_attribute();
attributevalue=get_attributevalue();

entity_dict={}
event_dict={}
attribute_dict={}
attributevalue_dict={}
sememe_dict={}


get_tree_html(event_dict,event,"{event|事件}    0")
get_tree_html(entity_dict,entity,"{entity|实体}")

get_tree_html(attribute_dict,attribute,"{attribute|属性}")
get_tree_html(attributevalue_dict,attributevalue,"{attributevalue|属性值}")
sememe_dict["id"] = 0
sememe_dict["name"]="义原"
sememe_dict["nodes"]=[]
sememe_dict["nodes"].append(entity_dict)
sememe_dict["nodes"].append(event_dict)
sememe_dict["nodes"].append(attribute_dict)
sememe_dict["nodes"].append(attributevalue_dict)

print(sememe_dict)
print(event_dict)
j = json.dumps(sememe_dict,ensure_ascii=False)
print(j)
with open('sememe.json', 'w') as f:
    json.dump(sememe_dict, f,ensure_ascii=False)  # 会在目录下生成









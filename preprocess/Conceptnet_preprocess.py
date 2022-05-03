import pickle


s="/Users/name/Desktop/知识库/Conceptnet/assertions.csv";
f =open(s, "r", errors="ignore");
s2="en_conceptnet";
f2 =open(s2, "w", errors="ignore");
s3="zh_conceptnet";
f3 =open(s3, "w", errors="ignore");
while(1):
    line= f.readline();
    if (len(line) == 0):
        break;
    items=line.strip().split("\t")
    head=items[2]
    tail=items[3]
    r=items[1]
    lang=head.split("/")[2]
    lang2 = tail.split("/")[2]
    if(lang=="en" and lang2=="en"):
        f2.write(line)
    if(lang=="zh" and lang2=="zh"):
        f3.write(line)
















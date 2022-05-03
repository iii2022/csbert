import pickle
s2="en_conceptnet";
f2 =open(s2, "r", errors="ignore");
s3="en_conceptnet_2";
f3 =open(s3, "w", errors="ignore");
while(1):
    line= f2.readline();
    if (len(line) == 0):
        break;
    items=line.strip().split("\t")
    head=items[2]
    tail=items[3]
    r=items[1]
    head=head.replace("/c/en/","")
    tail=tail.replace("/c/en/","")
    r=r.replace("/r/","")
    f3.write(r+"\t"+head+"\t"+tail+"\n")


















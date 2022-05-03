export PYTHONPATH=/data1/name/github/mycommon/:$PYTHONPATH
echo $PYTHONPATH
nohup python3 parse.py \
--start 0 --end 10000000 --file_name triples-en-0.csv > 0 
# &
# nohup python3 parse.py \
# --start 10000000 --end 20000000 --file_name triples-en-1.csv > 1 &
# nohup python3 parse.py \
# --start 20000000 --end 30000000 --file_name triples-en-2.csv > 2 &
# nohup python3 parse.py \
# --start 30000000 --end 40000000 --file_name triples-en-3.csv > 3

# python3 parse.py \
# --start 10000000 --end 20000000 --file_name triples-en-1.csv

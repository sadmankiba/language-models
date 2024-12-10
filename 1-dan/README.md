# Assignment 1

Run with `--emb_file='glove.6B.300d.txt'` for result with Glove embedding. Test accuracy is close with and without Glove embedding. Run as follows-
```sh
CAMPUSID="9084608083"
mkdir -p $CAMPUSID
python main.py --train=data/sst-train.txt --dev=data/sst-dev.txt --test=data/sst-test.txt --dev_out=$CAMPUSID/sst-dev-output.txt --test_out=$CAMPUSID/sst-test-output.txt`
python main.py --train=data/cfimdb-train.txt --dev=data/cfimdb-dev.txt --test=data/cfimdb-test.txt --dev_out=CAMPUSID/cfimdb-dev-output.txt --test_out=CAMPUSID/cfimdb-test-output.txt` --emb_file='glove.6B.300d.txt'
```

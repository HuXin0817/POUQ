wget https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/deep1M.tar.gz --no-check-certificate
tar -xzvf deep1M.tar.gz
rm deep1M.tar.gz

wget https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/word2vec.tar.gz --no-check-certificate
tar -xzvf word2vec.tar.gz
rm word2vec.tar.gz

wget https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/msong.tar.gz --no-check-certificate
tar -xzvf msong.tar.gz
rm msong.tar.gz

wget https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/gist.tar.gz --no-check-certificate
tar -xzvf gist.tar.gz
rm gist.tar.gz

wget https://www.cse.cuhk.edu.hk/systems/hash/gqr/dataset/imagenet.tar.gz --no-check-certificate
tar -xzvf imagenet.tar.gz
rm imagenet.tar.gz

wget -P sift/ https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/sift1M/sift_base.fvecs --no-check-certificate
wget -P sift/ https://huggingface.co/datasets/hhy3/ann-datasets/resolve/main/sift1M/sift_query.fvecs --no-check-certificate

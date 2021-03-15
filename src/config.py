class Config(object):
    data_dir = "../data"
    embedding_zip_filepath = f"{data_dir}/cc.ja.300.vec.gz"
    embedding_filepath = f"{data_dir}/cc.ja.300.vec"
    embedding_download_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ja.300.vec.gz"
from elasticsearch import Elasticsearch, helpers
import csv


def create_index(es_object, index_name):
    """ 创建新的索引"""
    es_object.indices.create(index=index_name, ignore=400)


def generate_data(source_file):
    """ 从CSV文件生成数据 """
    with open(source_file, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            yield {
                "_index": "34051_wiki",  # 索引名称
                "_source": row,
            }


def bulk_index_data(es_object, source_file):
    """ 批量索引数据 """
    resp = helpers.bulk(
        es_object,
        generate_data(source_file),
        chunk_size=1000,  # 每次发送1000条记录
        request_timeout=120
    )
    print("成功索引数据:", resp)


if __name__ == '__main__':
    es = Elasticsearch(hosts=["localhost:9200"])  # 调整为你的Elasticsearch的配置
    index_name = "34051_wiki"

    # 确保索引存在
    if not es.indices.exists(index=index_name):
        create_index(es, index_name)

        # 文件路径
    source_csv_file = "path/to/your/data.csv"  # 更改为你的CSV文件的路径

    # 执行批量索引
    bulk_index_data(es, source_csv_file)
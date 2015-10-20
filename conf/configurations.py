__author__ = 'tmkasun'
import json

# TODO: remove absolute path
with open(
        '/Users/tmkasun/Documents/uni/Level4_Semester2/IN 4410 - Big Data Analytics/labs/bigdata_project/spark/conf/datapackage.json',
        'r') as data_schema:
    json_schema = json.load(data_schema)
    schema = json_schema['resources'][0]['schema']['fields']

project = {
    'data_file': "data/automobile_data.csv",
    'sample_data_file': "data/sample_data.csv",
    'schema': schema
}
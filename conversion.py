import pandas as pd

def json_to_parquet(json_file, parquet_file):
    df = pd.read_json(json_file, lines=True)
    
    df.to_parquet(parquet_file, engine='pyarrow')

json_file_business = './yelp_academic_dataset_business.json'
parquet_file_business = './yelp_academic_dataset_business.parquet'
json_to_parquet(json_file_business, parquet_file_business)


json_file_tip = './yelp_academic_dataset_tip.json'
parquet_file_tip = './yelp_academic_dataset_tip.parquet'
json_to_parquet(json_file_tip, parquet_file_tip)
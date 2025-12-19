from pipelines.data_loader import DataLoader
from utils.config import RAW_DATA_DIR

loader = DataLoader(RAW_DATA_DIR)
success, msg = loader.initialize_kaggle_api()
print(f'Init: {msg}')

if success:
    datasets = loader.search_kaggle_datasets('iris', max_results=3)
    if datasets:
        print(f'Found {len(datasets)} datasets')
    else:
        print('No datasets found')
else:
    print(f'Failed: {msg}')

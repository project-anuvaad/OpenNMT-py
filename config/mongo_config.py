import os

MONGODB_SETTINGS = {
    'db': 'anuvada_benchmarks',
    'host': os.environ.get('mongo_ip','localhost'),
    'port': 27017
}
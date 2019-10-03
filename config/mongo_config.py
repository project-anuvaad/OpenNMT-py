import os

MONGODB_SETTINGS = {
    'db': 'anuvada_benchmarks',
    'host': os.environ.get('MONGO_IP','localhost'),
    'port': 27017
}
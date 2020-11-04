#!/usr/bin/env python3
"""Script that provides some stats about Nginx logs stored in MongoDB"""
from pymongo import MongoClient


if __name__ == "__main__":
    client = MongoClient('mongodb://127.0.0.1:27017')

    logs = client.logs.nginx
    num_of_docs = logs.count_documents({})

    print(str(num_of_docs), "logs")
    print("Methods:")

    for method in ["GET", "POST", "PUT", "PATCH", "DELETE"]:
        num_of_met = logs.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, num_of_met))

    path = {"method": "GET", "path": "/status"}

    num_of_path = logs.count_documents(path)
    print(str(num_of_path), "status check".format())

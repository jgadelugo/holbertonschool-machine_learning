#!/usr/bin/env python3
"""create a method that returns the list of ships that can
hold a given number of passengers"""
import requests


def availableShips(passengerCount):
    """create a method that returns the list of ships that can hold
    a given number of passengers"""
    BASE_URL = "https://swapi-api.hbtn.io/api/"
    ENDPOINT = "starships/"
    page_url = BASE_URL + ENDPOINT

    ships = []
    while (page_url):
        res = requests.get(page_url)
        results = res.json()['results']
        for ship in results:
            pass_size = ship['passengers']
            pass_size = pass_size.replace(',', '')
            if pass_size.isnumeric():
                if int(pass_size) >= passengerCount:
                    ships.append(ship['name'])
        page_url = res.json()['next']
    return ships

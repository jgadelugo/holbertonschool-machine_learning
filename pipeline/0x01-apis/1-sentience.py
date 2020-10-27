#!/usr/bin/env python3
"""create a method that returns the list of names of the home
planets of all sentient species"""
import requests


def sentientPlanets():
    """create a method that returns the list of names of the home
    planets of all sentient species"""
    BASE_URL = "https://swapi-api.hbtn.io/api/"
    ENDPOINT = "species/"
    page_url = BASE_URL + ENDPOINT

    planets = []
    while (page_url):
        res = requests.get(page_url)
        results = res.json()['results']
        for species in results:
            classi = species['classification']
            desi = species['designation']
            if 'sentient' in [classi, desi]:
                p_url = species['homeworld']
                if p_url:
                    planet = requests.get(p_url)
                    planets.append(planet.json()['name'])
        page_url = res.json()['next']
    return planets

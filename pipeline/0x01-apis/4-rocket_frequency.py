#!/usr/bin/env python3
"""script that displays the number of launches per rocket"""
import requests


if __name__ == '__main__':
    BASE_URL = "https://api.spacexdata.com/v4"
    ENDPOINT = "/launches"
    page_url = BASE_URL + ENDPOINT

    res = requests.get(page_url)

    launches = res.json()

    rocket_launch = {}
    rocket_ids = {}

    for lau in launches:
        r_id = lau['rocket']

        if r_id not in rocket_ids.keys():
            url = "https://api.spacexdata.com/v4/rockets/{}".format(r_id)
            res_rocket = requests.get(url).json()
            rocket_ids[r_id] = res_rocket['name']
        rocket = rocket_ids[r_id]
        if rocket in rocket_launch.keys():
            rocket_launch[rocket] += 1
        else:
            rocket_launch[rocket] = 1

    s_rockets = sorted(rocket_launch, key=lambda k: (-rocket_launch[k], k))
    for rocket in s_rockets:
        print("{}: {}".format(rocket, rocket_launch[rocket]))

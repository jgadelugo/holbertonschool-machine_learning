#!/usr/bin/env python3
"""script that displays the upcoming launch with these information"""
import requests


if __name__ == '__main__':
    BASE_URL = "https://api.spacexdata.com/v4"
    ENDPOINT = "/launches/upcoming"
    page_url = BASE_URL + ENDPOINT

    res = requests.get(page_url)
    res.status_code

    launches = res.json()

    launches = sorted(launches, key=lambda k: k['date_unix'])
    launch = launches[0]

    launch_name = launch['name']
    date = launch['date_local']

    r_id = launch['rocket']
    url = "https://api.spacexdata.com/v4/rockets/{}".format(r_id)
    res_rocket = requests.get(url).json()
    rocket_name = res_rocket['name']

    lp_id = launch["launchpad"]
    url = "https://api.spacexdata.com/v4/launchpads/{}".format(lp_id)
    res_pad = requests.get(url).json()
    launchpad_name = res_pad['name']
    launchpad_locality = res_pad['locality']

    text = f"{launch_name} ({date}) {rocket_name} - {launchpad_name}"
    text1 = " ({launchpad_locality})"
    print(text + text1)

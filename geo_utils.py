"""
geo_utils.py
-------------
Small helper for the multi-site / nationwide pipeline: picks an
appropriate LOCAL metric (UTM) CRS for a given lat/lon, so terrain
features (slope, TWI, distances) come out in consistent real-world units
(metres) no matter which part of the country an event is in.

India spans UTM zones 42N-47N roughly. Using a single fixed CRS
(like the Pune-specific EPSG:32643) across the whole country would
distort distances/areas the further you get from that zone -- so for
the multi-site pipeline we compute the correct zone per event instead.
"""

import math


def get_utm_epsg(lon, lat):
    """
    Returns the EPSG code for the UTM zone containing (lon, lat).
    Valid for anywhere on Earth; India falls in the northern-hemisphere
    UTM zones (326xx).
    """
    zone_number = int(math.floor((lon + 180) / 6) + 1)
    hemisphere_code = 326 if lat >= 0 else 327  # north vs south
    return f"EPSG:{hemisphere_code}{zone_number:02d}"

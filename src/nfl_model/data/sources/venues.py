"""Static NFL stadium metadata.

One row per stadium with the inputs the model actually consumes:
``lat`` / ``lon`` for weather and travel, ``roof`` / ``surface`` for the
dome and turf flags, and ``pass_axis_bearing_deg`` for projecting wind
speed onto the long axis of the field (the only direction that matters
for deep-passing offenses).

Stadium IDs use the PFR ``stadium_id`` code where one exists (e.g.
``KAN00`` for Arrowhead) so cross-source joins on ``stadium_id`` work
without further mapping. Where PFR doesn't expose one we use a
team-keyed synthetic ID — those are noted in comments.
"""

from __future__ import annotations

import pandas as pd

from nfl_model.data.warehouse import upsert_dataframe
from nfl_model.logging import get_logger

log = get_logger("data.sources.venues")


# pass_axis_bearing_deg is the compass bearing of the long axis of the
# field, used to project wind speed onto the passing direction. Values
# from public stadium-orientation surveys; approximate but consistent.
# Where a value is unknown we use 0.0 (no pass-axis adjustment).
_VENUES: list[dict] = [
    # AFC East
    {"stadium_id": "BUF00", "stadium_name": "Highmark Stadium",        "team": "BUF", "city": "Orchard Park",  "state": "NY", "lat": 42.7738, "lon": -78.7869, "elevation_ft":  610, "roof": "outdoors", "surface": "a_turf",    "pass_axis_bearing_deg":  40.0, "timezone": "America/New_York"},
    {"stadium_id": "MIA00", "stadium_name": "Hard Rock Stadium",       "team": "MIA", "city": "Miami Gardens", "state": "FL", "lat": 25.9580, "lon": -80.2389, "elevation_ft":    7, "roof": "outdoors", "surface": "grass",     "pass_axis_bearing_deg":  50.0, "timezone": "America/New_York"},
    {"stadium_id": "FOX00", "stadium_name": "Gillette Stadium",        "team": "NE",  "city": "Foxborough",    "state": "MA", "lat": 42.0909, "lon": -71.2643, "elevation_ft":  235, "roof": "outdoors", "surface": "fieldturf", "pass_axis_bearing_deg":  20.0, "timezone": "America/New_York"},
    {"stadium_id": "EAS00", "stadium_name": "MetLife Stadium",         "team": "NYJ", "city": "East Rutherford","state": "NJ","lat": 40.8135, "lon": -74.0744, "elevation_ft":    7, "roof": "outdoors", "surface": "fieldturf", "pass_axis_bearing_deg":  35.0, "timezone": "America/New_York"},

    # AFC North
    {"stadium_id": "BAL00", "stadium_name": "M&T Bank Stadium",        "team": "BAL", "city": "Baltimore",     "state": "MD", "lat": 39.2780, "lon": -76.6227, "elevation_ft":   56, "roof": "outdoors", "surface": "grass",     "pass_axis_bearing_deg":   0.0, "timezone": "America/New_York"},
    {"stadium_id": "CIN00", "stadium_name": "Paycor Stadium",          "team": "CIN", "city": "Cincinnati",    "state": "OH", "lat": 39.0954, "lon": -84.5161, "elevation_ft":  493, "roof": "outdoors", "surface": "fieldturf", "pass_axis_bearing_deg":  90.0, "timezone": "America/New_York"},
    {"stadium_id": "CLE00", "stadium_name": "Cleveland Browns Stadium","team": "CLE", "city": "Cleveland",     "state": "OH", "lat": 41.5061, "lon": -81.6995, "elevation_ft":  581, "roof": "outdoors", "surface": "grass",     "pass_axis_bearing_deg": 110.0, "timezone": "America/New_York"},
    {"stadium_id": "PIT00", "stadium_name": "Acrisure Stadium",        "team": "PIT", "city": "Pittsburgh",    "state": "PA", "lat": 40.4468, "lon": -80.0158, "elevation_ft":  724, "roof": "outdoors", "surface": "grass",     "pass_axis_bearing_deg":  30.0, "timezone": "America/New_York"},

    # AFC South
    {"stadium_id": "HOU00", "stadium_name": "NRG Stadium",             "team": "HOU", "city": "Houston",       "state": "TX", "lat": 29.6847, "lon": -95.4107, "elevation_ft":   50, "roof": "closed",   "surface": "grass",     "pass_axis_bearing_deg":   0.0, "timezone": "America/Chicago"},
    {"stadium_id": "IND00", "stadium_name": "Lucas Oil Stadium",       "team": "IND", "city": "Indianapolis",  "state": "IN", "lat": 39.7601, "lon": -86.1639, "elevation_ft":  715, "roof": "closed",   "surface": "fieldturf", "pass_axis_bearing_deg":   0.0, "timezone": "America/New_York"},
    {"stadium_id": "JAX00", "stadium_name": "EverBank Stadium",        "team": "JAX", "city": "Jacksonville",  "state": "FL", "lat": 30.3239, "lon": -81.6373, "elevation_ft":   16, "roof": "outdoors", "surface": "grass",     "pass_axis_bearing_deg": 105.0, "timezone": "America/New_York"},
    {"stadium_id": "NAS00", "stadium_name": "Nissan Stadium",          "team": "TEN", "city": "Nashville",     "state": "TN", "lat": 36.1665, "lon": -86.7713, "elevation_ft":  394, "roof": "outdoors", "surface": "grass",     "pass_axis_bearing_deg":  20.0, "timezone": "America/Chicago"},

    # AFC West
    {"stadium_id": "DEN00", "stadium_name": "Empower Field at Mile High","team":"DEN","city": "Denver",        "state": "CO", "lat": 39.7439, "lon":-105.0201, "elevation_ft": 5280, "roof": "outdoors", "surface": "grass",     "pass_axis_bearing_deg":   0.0, "timezone": "America/Denver"},
    {"stadium_id": "KAN00", "stadium_name": "GEHA Field at Arrowhead", "team": "KC",  "city": "Kansas City",   "state": "MO", "lat": 39.0489, "lon": -94.4839, "elevation_ft":  889, "roof": "outdoors", "surface": "grass",     "pass_axis_bearing_deg":  20.0, "timezone": "America/Chicago"},
    {"stadium_id": "LAS00", "stadium_name": "Allegiant Stadium",       "team": "LV",  "city": "Paradise",      "state": "NV", "lat": 36.0908, "lon":-115.1830, "elevation_ft": 2030, "roof": "closed",   "surface": "grass",     "pass_axis_bearing_deg":   0.0, "timezone": "America/Los_Angeles"},
    {"stadium_id": "LAX00", "stadium_name": "SoFi Stadium",            "team": "LAC", "city": "Inglewood",     "state": "CA", "lat": 33.9534, "lon":-118.3387, "elevation_ft":   95, "roof": "closed",   "surface": "fieldturf", "pass_axis_bearing_deg":   0.0, "timezone": "America/Los_Angeles"},

    # NFC East
    {"stadium_id": "DAL00", "stadium_name": "AT&T Stadium",            "team": "DAL", "city": "Arlington",     "state": "TX", "lat": 32.7473, "lon": -97.0945, "elevation_ft":  551, "roof": "closed",   "surface": "fieldturf", "pass_axis_bearing_deg":   0.0, "timezone": "America/Chicago"},
    {"stadium_id": "EAS00_NYG", "stadium_name": "MetLife Stadium",     "team": "NYG", "city": "East Rutherford","state": "NJ","lat": 40.8135, "lon": -74.0744, "elevation_ft":    7, "roof": "outdoors", "surface": "fieldturf", "pass_axis_bearing_deg":  35.0, "timezone": "America/New_York"},
    {"stadium_id": "PHI00", "stadium_name": "Lincoln Financial Field", "team": "PHI", "city": "Philadelphia",  "state": "PA", "lat": 39.9008, "lon": -75.1675, "elevation_ft":   39, "roof": "outdoors", "surface": "grass",     "pass_axis_bearing_deg":  35.0, "timezone": "America/New_York"},
    {"stadium_id": "WAS00", "stadium_name": "Northwest Stadium",       "team": "WAS", "city": "Landover",      "state": "MD", "lat": 38.9077, "lon": -76.8645, "elevation_ft":  226, "roof": "outdoors", "surface": "grass",     "pass_axis_bearing_deg":  10.0, "timezone": "America/New_York"},

    # NFC North
    {"stadium_id": "CHI00", "stadium_name": "Soldier Field",           "team": "CHI", "city": "Chicago",       "state": "IL", "lat": 41.8623, "lon": -87.6167, "elevation_ft":  604, "roof": "outdoors", "surface": "grass",     "pass_axis_bearing_deg":   0.0, "timezone": "America/Chicago"},
    {"stadium_id": "DET00", "stadium_name": "Ford Field",              "team": "DET", "city": "Detroit",       "state": "MI", "lat": 42.3400, "lon": -83.0456, "elevation_ft":  636, "roof": "closed",   "surface": "fieldturf", "pass_axis_bearing_deg":   0.0, "timezone": "America/Detroit"},
    {"stadium_id": "GNB00", "stadium_name": "Lambeau Field",           "team": "GB",  "city": "Green Bay",     "state": "WI", "lat": 44.5013, "lon": -88.0622, "elevation_ft":  640, "roof": "outdoors", "surface": "grass",     "pass_axis_bearing_deg":  20.0, "timezone": "America/Chicago"},
    {"stadium_id": "MIN00", "stadium_name": "U.S. Bank Stadium",       "team": "MIN", "city": "Minneapolis",   "state": "MN", "lat": 44.9737, "lon": -93.2581, "elevation_ft":  841, "roof": "closed",   "surface": "fieldturf", "pass_axis_bearing_deg":   0.0, "timezone": "America/Chicago"},

    # NFC South
    {"stadium_id": "ATL00", "stadium_name": "Mercedes-Benz Stadium",   "team": "ATL", "city": "Atlanta",       "state": "GA", "lat": 33.7553, "lon": -84.4006, "elevation_ft": 1050, "roof": "closed",   "surface": "fieldturf", "pass_axis_bearing_deg":   0.0, "timezone": "America/New_York"},
    {"stadium_id": "CHA00", "stadium_name": "Bank of America Stadium", "team": "CAR", "city": "Charlotte",     "state": "NC", "lat": 35.2258, "lon": -80.8528, "elevation_ft":  748, "roof": "outdoors", "surface": "fieldturf", "pass_axis_bearing_deg":  25.0, "timezone": "America/New_York"},
    {"stadium_id": "NOR00", "stadium_name": "Caesars Superdome",       "team": "NO",  "city": "New Orleans",   "state": "LA", "lat": 29.9511, "lon": -90.0812, "elevation_ft":    3, "roof": "closed",   "surface": "fieldturf", "pass_axis_bearing_deg":   0.0, "timezone": "America/Chicago"},
    {"stadium_id": "TAM00", "stadium_name": "Raymond James Stadium",   "team": "TB",  "city": "Tampa",         "state": "FL", "lat": 27.9759, "lon": -82.5033, "elevation_ft":   16, "roof": "outdoors", "surface": "grass",     "pass_axis_bearing_deg":  30.0, "timezone": "America/New_York"},

    # NFC West
    {"stadium_id": "PHO00", "stadium_name": "State Farm Stadium",      "team": "ARI", "city": "Glendale",      "state": "AZ", "lat": 33.5276, "lon":-112.2626, "elevation_ft": 1148, "roof": "closed",   "surface": "grass",     "pass_axis_bearing_deg":   0.0, "timezone": "America/Phoenix"},
    {"stadium_id": "LAX00_LA", "stadium_name": "SoFi Stadium",         "team": "LA",  "city": "Inglewood",     "state": "CA", "lat": 33.9534, "lon":-118.3387, "elevation_ft":   95, "roof": "closed",   "surface": "fieldturf", "pass_axis_bearing_deg":   0.0, "timezone": "America/Los_Angeles"},
    {"stadium_id": "SFO00", "stadium_name": "Levi's Stadium",          "team": "SF",  "city": "Santa Clara",   "state": "CA", "lat": 37.4032, "lon":-121.9698, "elevation_ft":   16, "roof": "outdoors", "surface": "grass",     "pass_axis_bearing_deg":  60.0, "timezone": "America/Los_Angeles"},
    {"stadium_id": "SEA00", "stadium_name": "Lumen Field",             "team": "SEA", "city": "Seattle",       "state": "WA", "lat": 47.5952, "lon":-122.3316, "elevation_ft":    7, "roof": "outdoors", "surface": "fieldturf", "pass_axis_bearing_deg":  10.0, "timezone": "America/Los_Angeles"},

    # ----- PFR-coded synonyms (nflverse uses these IDs for the same venues) -----
    {"stadium_id": "NYC01", "stadium_name": "MetLife Stadium",         "team": "NYJ", "city": "East Rutherford","state": "NJ","lat": 40.8135, "lon": -74.0744, "elevation_ft":    7, "roof": "outdoors", "surface": "fieldturf", "pass_axis_bearing_deg":  35.0, "timezone": "America/New_York"},
    {"stadium_id": "SFO01", "stadium_name": "Levi's Stadium",          "team": "SF",  "city": "Santa Clara",   "state": "CA", "lat": 37.4032, "lon":-121.9698, "elevation_ft":   16, "roof": "outdoors", "surface": "grass",     "pass_axis_bearing_deg":  60.0, "timezone": "America/Los_Angeles"},
    {"stadium_id": "CAR00", "stadium_name": "Bank of America Stadium", "team": "CAR", "city": "Charlotte",     "state": "NC", "lat": 35.2258, "lon": -80.8528, "elevation_ft":  748, "roof": "outdoors", "surface": "fieldturf", "pass_axis_bearing_deg":  25.0, "timezone": "America/New_York"},
    {"stadium_id": "BOS00", "stadium_name": "Gillette Stadium",        "team": "NE",  "city": "Foxborough",    "state": "MA", "lat": 42.0909, "lon": -71.2643, "elevation_ft":  235, "roof": "outdoors", "surface": "fieldturf", "pass_axis_bearing_deg":  20.0, "timezone": "America/New_York"},
    {"stadium_id": "CHI98", "stadium_name": "Soldier Field",           "team": "CHI", "city": "Chicago",       "state": "IL", "lat": 41.8623, "lon": -87.6167, "elevation_ft":  604, "roof": "outdoors", "surface": "grass",     "pass_axis_bearing_deg":   0.0, "timezone": "America/Chicago"},
    {"stadium_id": "ATL01", "stadium_name": "Mercedes-Benz Stadium",   "team": "ATL", "city": "Atlanta",       "state": "GA", "lat": 33.7553, "lon": -84.4006, "elevation_ft": 1050, "roof": "closed",   "surface": "fieldturf", "pass_axis_bearing_deg":   0.0, "timezone": "America/New_York"},

    # ----- International venues (NFL plays a handful of games abroad each year) -----
    {"stadium_id": "LON00", "stadium_name": "Wembley Stadium",         "team": None,  "city": "London",        "state": "UK", "lat": 51.5560, "lon":   -0.2796, "elevation_ft":  151, "roof": "outdoors", "surface": "grass",     "pass_axis_bearing_deg":   0.0, "timezone": "Europe/London"},
    {"stadium_id": "LON01", "stadium_name": "Twickenham Stadium",      "team": None,  "city": "London",        "state": "UK", "lat": 51.4561, "lon":   -0.3414, "elevation_ft":   33, "roof": "outdoors", "surface": "grass",     "pass_axis_bearing_deg":   0.0, "timezone": "Europe/London"},
    {"stadium_id": "LON02", "stadium_name": "Tottenham Hotspur Stadium","team": None, "city": "London",        "state": "UK", "lat": 51.6043, "lon":   -0.0664, "elevation_ft":  171, "roof": "outdoors", "surface": "grass",     "pass_axis_bearing_deg":   0.0, "timezone": "Europe/London"},
    {"stadium_id": "MEX00", "stadium_name": "Estadio Azteca",          "team": None,  "city": "Mexico City",   "state": "MX", "lat": 19.3029, "lon":  -99.1505, "elevation_ft": 7382, "roof": "outdoors", "surface": "grass",     "pass_axis_bearing_deg":   0.0, "timezone": "America/Mexico_City"},
    {"stadium_id": "GER00", "stadium_name": "Allianz Arena",           "team": None,  "city": "Munich",        "state": "DE", "lat": 48.2188, "lon":   11.6247, "elevation_ft": 1631, "roof": "outdoors", "surface": "grass",     "pass_axis_bearing_deg":   0.0, "timezone": "Europe/Berlin"},
    {"stadium_id": "GER01", "stadium_name": "Deutsche Bank Park",      "team": None,  "city": "Frankfurt",     "state": "DE", "lat": 50.0686, "lon":    8.6450, "elevation_ft":  371, "roof": "outdoors", "surface": "grass",     "pass_axis_bearing_deg":   0.0, "timezone": "Europe/Berlin"},
    {"stadium_id": "SAO00", "stadium_name": "Arena Corinthians",       "team": None,  "city": "São Paulo",     "state": "BR", "lat":-23.5453, "lon":  -46.4742, "elevation_ft": 2493, "roof": "outdoors", "surface": "grass",     "pass_axis_bearing_deg":   0.0, "timezone": "America/Sao_Paulo"},
]


def ensure_venues() -> int:
    """Idempotently load the venues table; returns row count written."""
    df = pd.DataFrame(_VENUES)
    n = upsert_dataframe(df, "venues", key_columns=["stadium_id"])
    log.info("venues.seeded", rows=n)
    return n

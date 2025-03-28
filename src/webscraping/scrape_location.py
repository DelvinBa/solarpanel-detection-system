import requests

API_KEY = 'l7b26b9b8b79a045559cff48f83ad86324'


def get_coordinates(verblijfsobject_id):
    """
    Fetches coordinates (EPSG:28992) for a given verblijfsobject ID from the Kadaster BAG API.
    Returns only the first two coordinates [x, y] (ignoring the z value).
    """
    url = f"https://api.bag.kadaster.nl/lvbag/individuelebevragingen/v2/verblijfsobjecten/{verblijfsobject_id}"
    headers = {
        'Accept': 'application/hal+json',
        'Accept-Crs': 'epsg:28992',
        'X-Api-Key': API_KEY
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        try:
            data = response.json()
            coords = data["verblijfsobject"]["geometrie"]["punt"]["coordinates"]
            # Return only x and y (first two elements)
            return coords[:2]
        except KeyError as e:
            print("Coordinates not found in the response.")
            return None
    else:
        print(f"Request failed: {response.status_code}")
        return None


# For testing when run standalone:
if __name__ == "__main__":
    import pandas as pd
    csv_path = "data/interim/pid_vid_GM0153.csv"
    df_loaded = pd.read_csv(csv_path, dtype={"vid": str})
    print(df_loaded.head())
    first_vid = df_loaded.loc[0, "vid"]
    print(f"\nTesting first VID: {first_vid}")
    coords = get_coordinates(first_vid)
    if coords:
        print("✅ Coordinates fetched:", coords)
    else:
        print("❌ Failed to fetch coordinates.")

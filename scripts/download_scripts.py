import requests

def download_pq(url, filename):
    with requests.get(url, stream = True) as response:
        response.raise_for_status()
        with open(filename, 'wb') as f:
            for c in response.iter_content(chunk_size=8192):
                f.write(c)

year = 2019
start_month = 2
end_month = 7

filepath = "data/landing"

for month in range(start_month, end_month + 1):
    url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet"
    filename = f"{filepath}/yellow_tripdata_{year}-{month:02d}.parquet"
    download_pq(url, filename)
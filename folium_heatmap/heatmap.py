import pandas as pd
import os
import folium
from folium.plugins import HeatMap

df = pd.read_csv('test_data.csv')

# Parameters
min_price = 100000
max_amount = 4000000

hmap = folium.Map(location=[40.677, -73.985], zoom_start=14, tiles = "Stamen Terrain")

hm_data = HeatMap( list(zip(df["Lat"], df["Lng"], df["ClosingPrice"])),
                   min_opacity=0.1,
                   max_val=max_amount,
                   radius=20, blur=38,
                   max_zoom=1,
                 )

hmap.add_child(hm_data)

hmap.save(os.path.join('heatmap.html'))

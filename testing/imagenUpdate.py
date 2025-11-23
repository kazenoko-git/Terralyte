import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from imagen import Imagen
lat = 12.856632651034612
lon = 77.66337318359953
img = Imagen(provider="esri").getStitchedTiles(lat, lon, 18, radius=1)
print(img.size)

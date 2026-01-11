import os
import requests
import numpy as np
import rasterio
from rasterio.windows import from_bounds
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

# --- CONFIGURATION ---
PLOT_EXTENT = [-85.0, -75.0, 31.0, 37.5] # Min Lon, Max Lon, Min Lat, Max Lat
OUTPUT_FILE = "vhi_test_map.png"
VAR_TYPE = "VH.VHI" 

def get_current_year_week():
    today = datetime.utcnow()
    year, week, _ = today.isocalendar()
    return year, week

def download_latest_vhi():
    # NOAA STAR Server
    base_url = "https://www.star.nesdis.noaa.gov/data/pub0018/VHPdata4users/data/Blended_VH_4km/geo_TIFF/"
    year, week = get_current_year_week()
    
    # Try last 6 weeks
    for i in range(6):
        curr_week = week - i
        curr_year = year
        if curr_week <= 0:
            curr_week += 52
            curr_year -= 1
            
        # File name format: VHP.G04.C07.j01.P{Year}{Week}.VH.VHI.tif
        fname = f"VHP.G04.C07.j01.P{curr_year}{curr_week:03d}.{VAR_TYPE}.tif"
        url = base_url + fname
        
        print(f"Checking: {url}")
        try:
            r = requests.get(url, stream=True, timeout=15)
            if r.status_code == 200:
                print(f"Success! Found data for Year {curr_year} Week {curr_week}")
                with open("temp_vhi.tif", 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                return "temp_vhi.tif", f"{curr_year} Week {curr_week}"
        except Exception as e:
            print(f"Connection error: {e}")
            
    print("Could not find any VHI data.")
    return None, None

def plot_vhi(tif_path, date_str):
    with rasterio.open(tif_path) as src:
        print(f"File Bounds: {src.bounds}")
        
        # 1. Use from_bounds to safely calculate the slice
        # PLOT_EXTENT is [MinLon, MaxLon, MinLat, MaxLat]
        # from_bounds expects (left, bottom, right, top)
        # So we pass: (MinLon, MinLat, MaxLon, MaxLat)
        window = from_bounds(PLOT_EXTENT[0], PLOT_EXTENT[2], 
                             PLOT_EXTENT[1], PLOT_EXTENT[3], 
                             transform=src.transform)
        
        # Round the window to ensure we grab whole pixels
        window = window.round_offsets().round_shape()
        
        print(f"Calculated Window: {window}")

        # 2. Read data
        data = src.read(1, window=window)
        
        if data.size == 0:
            print("Error: Slicing resulted in empty data.")
            return

        # 3. Handle Coordinates
        # We need to generate the lat/lon arrays for the *sliced* window
        win_transform = src.window_transform(window)
        height, width = data.shape
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        
        # transform.xy returns coordinates for the center of the pixels
        xs, ys = rasterio.transform.xy(win_transform, rows, cols, offset='center')
        lons = np.array(xs)
        lats = np.array(ys)
        
        # 4. Mask NoData / Invalid Values
        # VHI is 0-100. Values < 0 are usually space/nodata.
        data = np.where(data < 0, np.nan, data)

    # Plotting
    print(f"Plotting Data Shape: {data.shape}")
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-80, central_latitude=34))
    ax.set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.STATES, linewidth=1.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    
    # shading='auto' handles the coordinate dimensions automatically
    mesh = ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree(), 
                         cmap='RdYlGn', vmin=0, vmax=100, shading='auto')
    
    plt.colorbar(mesh, label="Vegetation Health Index (VHI)", orientation='horizontal', pad=0.05, shrink=0.8)
    plt.title(f"NOAA STAR Vegetation Health (VIIRS)\nLatest Available: {date_str}")
    
    plt.savefig(OUTPUT_FILE, bbox_inches='tight')
    print(f"Saved test map to {OUTPUT_FILE}")

if __name__ == "__main__":
    tif, date_str = download_latest_vhi()
    if tif:
        plot_vhi(tif, date_str)
        if os.path.exists(tif):
            os.remove(tif)

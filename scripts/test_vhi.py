import os
import sys
import requests
import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

# --- CONFIGURATION ---
PLOT_EXTENT = [-85.0, -75.0, 31.0, 37.5] # MinLon, MaxLon, MinLat, MaxLat
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
            # Short connect timeout (10s), longer read timeout (60s) for large files
            r = requests.get(url, stream=True, timeout=(10, 60))
            if r.status_code == 200:
                print(f"Success! Found data for Year {curr_year} Week {curr_week}")
                print("Starting download (this may take a moment)...")
                
                with open("temp_vhi.tif", 'wb') as f:
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=1024*1024): # 1MB chunks
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                print(f"Download Complete. Size: {downloaded / (1024*1024):.2f} MB")
                return "temp_vhi.tif", f"{curr_year} Week {curr_week}"
        except Exception as e:
            print(f"Connection error/timeout: {e}")
            
    print("Could not find any VHI data.")
    return None, None

def plot_vhi(tif_path, date_str):
    print("Opening GeoTIFF...")
    with rasterio.open(tif_path) as src:
        print(f"File Bounds: {src.bounds}")
        
        # 1. Use from_bounds to calculate the slice
        # PLOT_EXTENT is [MinLon, MaxLon, MinLat, MaxLat] -> (left, bottom, right, top)
        # Note: rasterio expects (left, bottom, right, top)
        window = from_bounds(PLOT_EXTENT[0], PLOT_EXTENT[2], 
                             PLOT_EXTENT[1], PLOT_EXTENT[3], 
                             transform=src.transform)
        
        # 2. FIX DEPRECATION: Manually round window to integers
        # round_offsets() helps align to pixel grid, then we cast to int
        window = window.round_offsets(op='round')
        window = Window(col_off=window.col_off, row_off=window.row_off, 
                        width=max(1, round(window.width)), 
                        height=max(1, round(window.height)))
        
        print(f"Calculated Window: {window}")

        # 3. Read data
        print("Reading data subset...")
        data = src.read(1, window=window)
        
        if data.size == 0:
            print("Error: Slicing resulted in empty data.")
            return

        # 4. Handle Coordinates
        print("Calculating coordinates...")
        win_transform = src.window_transform(window)
        height, width = data.shape
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(win_transform, rows, cols, offset='center')
        lons = np.array(xs)
        lats = np.array(ys)
        
        # Mask NoData / Invalid Values (VHI uses < 0 for invalid)
        data = np.where(data < 0, np.nan, data)

    # Plotting
    print(f"Plotting Data Shape: {data.shape}")
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-80, central_latitude=34))
    ax.set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.STATES, linewidth=1.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    
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
    else:
        sys.exit(1) # Fail the action if no data found

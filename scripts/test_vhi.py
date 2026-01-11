import os
import requests
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta

# --- CONFIGURATION ---
PLOT_EXTENT = [-85.0, -75.0, 31.0, 37.5]
OUTPUT_FILE = "vhi_test_map.png"
# VHI (Vegetation Health Index) is often better for fire than pure NDVI
# because it includes thermal stress. 0 = Dead/Dry, 100 = Lush/Wet.
VAR_TYPE = "VH.VHI" # Options: "VH.VHI" (Health), "SM.SMN" (Smoothed NDVI)

def get_current_year_week():
    """Returns (Year, WeekNumber) for 'today'."""
    today = datetime.utcnow()
    # ISO Calendar returns (Year, Week, Day)
    year, week, _ = today.isocalendar()
    return year, week

def download_latest_vhi():
    """Loops backwards from current week to find the latest available NOAA STAR file."""
    base_url = "https://www.star.nesdis.noaa.gov/data/pub0018/VHPdata4users/data/Blended_VH_4km/geo_TIFF/"
    
    year, week = get_current_year_week()
    
    # Try the last 6 weeks (sometimes data lags by 1-2 weeks)
    for i in range(6):
        # Handle year rollover if we go back past week 1
        curr_week = week - i
        curr_year = year
        if curr_week <= 0:
            curr_week += 52
            curr_year -= 1
            
        # File Pattern: VHP.G04.C07.j01.P{Year}{Week:03d}.VH.VHI.tif
        # j01 = NOAA-20 Satellite (Current Operational)
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
            
    print("Could not find any VHI data in the last 6 weeks.")
    return None, None

def plot_vhi(tif_path, date_str):
    with rasterio.open(tif_path) as src:
        # Reproject or Slice?
        # NOAA STAR 4km is usually PlateCarree (Lat/Lon) but covers the globe.
        # We need to slice it to our domain to avoid massive memory usage.
        
        # 1. Calculate Window for our PLOT_EXTENT
        # Transform coords to pixels
        py, px = src.index(PLOT_EXTENT[2], PLOT_EXTENT[0]) # Lat (Max), Lon (Min) (Top-Left)
        py2, px2 = src.index(PLOT_EXTENT[3], PLOT_EXTENT[1]) # Lat (Min), Lon (Max) (Bottom-Right)
        
        # Handle the fact that lat index decreases as lat increases (usually)
        r_min, r_max = min(py, py2), max(py, py2)
        c_min, c_max = min(px, px2), max(px, px2)
        
        # Add padding
        pad = 50
        window = rasterio.windows.Window(c_min - pad, r_min - pad, 
                                       (c_max - c_min) + 2*pad, 
                                       (r_max - r_min) + 2*pad)
        
        data = src.read(1, window=window)
        transform = src.window_transform(window)
        
        # NoData handling (VHI uses -9999 or similar often, but usually 0-100 valid)
        data = np.where(data < 0, np.nan, data)

        # Create Coordinates for Plotting
        height, width = data.shape
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
        lons = np.array(xs)
        lats = np.array(ys)

    # Plotting
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-80, central_latitude=34))
    ax.set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())
    
    ax.add_feature(cfeature.STATES, linewidth=1.5)
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    
    # VHI Color Scheme: Red (0) -> Yellow (50) -> Green (100)
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
        # Cleanup
        os.remove(tif)

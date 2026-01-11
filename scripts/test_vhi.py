import sys
import os

print("--- Script Initializing ---", flush=True)

try:
    print("Importing requests...", flush=True)
    import requests
    print("Importing numpy...", flush=True)
    import numpy as np
    
    print("Importing rasterio...", flush=True)
    import rasterio
    from rasterio.windows import from_bounds, Window
    
    print("Importing matplotlib (Headless Mode)...", flush=True)
    import matplotlib
    matplotlib.use('Agg') # Force non-interactive backend
    import matplotlib.pyplot as plt
    
    print("Importing cartopy...", flush=True)
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    print("Imports Complete.", flush=True)
    
except Exception as e:
    print(f"CRITICAL IMPORT ERROR: {e}", flush=True)
    sys.exit(1)

from datetime import datetime

# --- CONFIGURATION ---
PLOT_EXTENT = [-85.0, -75.0, 31.0, 37.5]
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
    satellites = ['j01', 'npp']
    
    for i in range(6):
        curr_week = week - i
        curr_year = year
        if curr_week <= 0:
            curr_week += 52
            curr_year -= 1
            
        for sat in satellites:
            fname = f"VHP.G04.C07.{sat}.P{curr_year}{curr_week:03d}.{VAR_TYPE}.tif"
            url = base_url + fname
            print(f"Checking: {url}", flush=True)
            
            try:
                r = requests.get(url, stream=True, timeout=(10, 60))
                if r.status_code == 200:
                    print(f"Success! Found data for Year {curr_year} Week {curr_week} ({sat})", flush=True)
                    print("Starting download...", flush=True)
                    with open("temp_vhi.tif", 'wb') as f:
                        for chunk in r.iter_content(chunk_size=1024*1024):
                            if chunk: f.write(chunk)
                    print("Download Complete.", flush=True)
                    return "temp_vhi.tif", f"{curr_year} Week {curr_week}"
            except Exception as e:
                print(f"  > Error: {e}", flush=True)
            
    print("Could not find any VHI data.", flush=True)
    return None, None

def plot_vhi(tif_path, date_str):
    print("Opening GeoTIFF...", flush=True)
    try:
        with rasterio.open(tif_path) as src:
            print(f"File Bounds: {src.bounds}", flush=True)
            
            # Calculate Slice
            window = from_bounds(PLOT_EXTENT[0], PLOT_EXTENT[2], 
                                 PLOT_EXTENT[1], PLOT_EXTENT[3], 
                                 transform=src.transform)
            
            # Round Window
            window = window.round_offsets(op='round')
            window = Window(col_off=window.col_off, row_off=window.row_off, 
                            width=max(1, int(window.width)), 
                            height=max(1, int(window.height)))
            
            print(f"Calculated Window: {window}", flush=True)
            
            # Read Data
            data = src.read(1, window=window)
            
            if data.size == 0:
                print("Error: Slicing resulted in empty data.", flush=True)
                return

            # Mask Invalid Values (VHI uses < 0 for NoData)
            data = np.where(data < 0, np.nan, data)
            
            # CRITICAL CHANGE: Get exact extent of the slice for imshow
            # Returns (left, bottom, right, top)
            win_bounds = src.window_bounds(window)
            # Imshow expects [left, right, bottom, top]
            extent = [win_bounds[0], win_bounds[2], win_bounds[1], win_bounds[3]]
            print(f"Slice Extent: {extent}", flush=True)

        print("Initializing Plot...", flush=True)
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-80, central_latitude=34))
        ax.set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())
        
        print("Adding Map Features...", flush=True)
        # Wrap features in try/except to prevent network timeouts from killing the script
        try:
            ax.add_feature(cfeature.STATES, linewidth=1.5, edgecolor='black')
            ax.add_feature(cfeature.COASTLINE, linewidth=1, edgecolor='black')
        except Exception as e:
            print(f"Warning: Could not download map features: {e}", flush=True)
        
        print("Rendering Data (imshow)...", flush=True)
        # Use imshow instead of pcolormesh for stability
        ax.imshow(data, transform=ccrs.PlateCarree(), extent=extent, 
                  origin='upper', cmap='RdYlGn', vmin=0, vmax=100)
        
        plt.colorbar(label="Vegetation Health Index (VHI)", orientation='horizontal', pad=0.05, shrink=0.8)
        plt.title(f"NOAA STAR Vegetation Health (VIIRS)\nLatest Available: {date_str}")
        
        print("Saving Image...", flush=True)
        plt.savefig(OUTPUT_FILE, bbox_inches='tight')
        print(f"Saved test map to {OUTPUT_FILE}", flush=True)
    
    except Exception as e:
        print(f"Plotting Failed: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    tif, date_str = download_latest_vhi()
    if tif:
        plot_vhi(tif, date_str)
        if os.path.exists(tif): os.remove(tif)
    else:
        sys.exit(1)

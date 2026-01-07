import os
import sys
import numpy as np
import xarray as xr
import rasterio
import folium
import requests
from datetime import datetime, timedelta
import matplotlib.colors as mcolors

# --- CONFIGURATION ---
FUEL_PATH = 'data/fuel_SE_final.tif' 
OUTPUT_HTML = 'public/index.html'

# [cite_start]MOE Lookup based on Anderson (1982) & NOAA TM-205 [cite: 256, 290, 397]
MOE_LOOKUP = {
    1: 12, 2: 15, 3: 25, 4: 20, 5: 20, 6: 25, 7: 40,
    8: 30, 9: 25, 10: 25, 11: 15, 12: 20, 13: 25
}

def calculate_emc(T_degF, RH_percent):
    """
    Calculates 1-hr Equilibrium Moisture Content (EMC).
    Uses standard Simard (1968) equations found in fire behavior systems.
    """
    h = np.clip(RH_percent, 0, 100)
    t = T_degF
    
    cond1 = h < 10
    cond2 = (h >= 10) & (h < 50)
    cond3 = h >= 50
    
    emc = np.zeros_like(h)
    
    emc[cond1] = 0.03229 + 0.281073*h[cond1] - 0.000578*h[cond1]*t[cond1]
    emc[cond2] = 2.22749 + 0.160107*h[cond2] - 0.01478*t[cond2]
    emc[cond3] = 21.0606 + 0.005565*(h[cond3]**2) - 0.00035*h[cond3]*t[cond3] - 0.483199*h[cond3]
    
    return np.clip(emc, 0, 35)

def sample_fuel_grid(fuel_path, target_lats, target_lons):
    """
    Probes the static Fuel TIFF at the weather grid coordinates.
    """
    print(f"Sampling fuel model from {fuel_path}...")
    with rasterio.open(fuel_path) as src:
        # Flatten weather grid to list of (lon, lat)
        coords = zip(target_lons.ravel(), target_lats.ravel())
        sampled_values = src.sample(coords)
        fuel_flat = np.fromiter((val[0] for val in sampled_values), dtype=np.uint8)
        
    return fuel_flat.reshape(target_lats.shape)

def download_file(date_str, run, fhr):
    """
    Downloads HREF Mean product for a specific forecast hour.
    """
    # HREF Mean contains the Ensemble Mean of T and RH
    base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/prod/href.{date_str}/ensprod"
    filename = f"href.t{run}z.conus.mean.f{fhr:02d}.grib2"
    url = f"{base_url}/{filename}"
    
    print(f"Downloading {filename}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    
    try:
        with requests.get(url, stream=True, timeout=120, headers=headers) as r:
            if r.status_code == 404:
                print(f"404 Error: File not found {url}")
                return None
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return filename
    except Exception as e:
        print(f"Download failed: {e}")
        return None

def main():
    print("--- Starting Fire Weather Recovery Map (Direct Download) ---")
    
    # 1. DETERMINE RUN TIME
    # HREF runs available approx 3-4 hours after cycle time
    now = datetime.utcnow()
    # Simple logic: If it's past 14:00 UTC, try 12Z. Else try 00Z.
    if now.hour >= 14:
        run_cycle = "12"
    else:
        run_cycle = "00"
    
    # Date string YYYYMMDD
    date_str = now.strftime("%Y%m%d")
    
    # Fallback: If 12Z isn't on server yet (NOMADS can be slow), check previous day 12Z or 00Z?
    # For simplicity, we assume the run exists. If 404, we could add retry logic for previous cycle.
    
    print(f"Targeting HREF Run: {date_str} {run_cycle}Z")

    # 2. INITIALIZE MAP
    m = folium.Map(location=[34, -84], zoom_start=6, tiles='CartoDB dark_matter')
    folium.LayerControl().add_to(m)
    
    fuel_grid_cached = None
    moe_grid_cached = None

    # 3. LOOP FORECAST HOURS (1 to 18)
    # We download one file, process it, add to map, then delete it.
    for fhr in range(1, 19):
        grib_file = download_file(date_str, run_cycle, fhr)
        
        if not grib_file:
            print(f"Skipping hour f{fhr:02d} (Download failed)")
            continue

        try:
            # Open GRIB2 with xarray + cfgrib
            # We filter for T2M and RH2M to speed up reading
            ds = xr.open_dataset(
                grib_file, 
                engine='cfgrib', 
                filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2}
            )
            
            # Extract Data
            # Note: GRIB var names often differ. HREF Mean usually uses:
            # t2m = Temperature
            # r2 = Relative Humidity
            t_k = ds['t2m'].values
            rh = ds['r2'].values
            lats = ds.latitude.values
            lons = ds.longitude.values
            valid_time = ds.valid_time.values
            
            # --- ONE TIME SETUP (Fuel Grid) ---
            # We only sample the fuel grid once, the first time we get valid coordinates
            if fuel_grid_cached is None:
                fuel_grid_cached = sample_fuel_grid(FUEL_PATH, lats, lons)
                
                # Build MOE Grid
                moe_grid_cached = np.zeros_like(fuel_grid_cached, dtype=float)
                for fid, moe_val in MOE_LOOKUP.items():
                    moe_grid_cached[fuel_grid_cached == fid] = moe_val
                
                # Mask non-fuel
                moe_grid_cached[moe_grid_cached == 0] = 999 
                moe_grid_cached[fuel_grid_cached > 13] = 999
            
            # --- CALCULATE RECOVERY ---
            t_f = (t_k - 273.15) * 9/5 + 32
            fm_grid = calculate_emc(t_f, rh)
            recovery_grid = (fm_grid / moe_grid_cached) * 100
            
            # --- ADD TO MAP ---
            time_str = str(valid_time).split('T')[1][:5] + "Z"
            fg = folium.FeatureGroup(name=f"Hour {time_str} (+{fhr})", show=False)
            
            # Subsample for web performance (Skip every 15 points)
            stride = 15
            count = 0
            
            for i in range(0, lats.shape[0], stride):
                for j in range(0, lats.shape[1], stride):
                    val = recovery_grid[i, j]
                    lat = lats[i, j]
                    lon = lons[i, j]

                    # Domain Filter (Southeast Box)
                    if not (24 <= lat <= 38 and -90 <= lon <= -75):
                        continue
                        
                    # Skip invalid/urban
                    if val <= 0.1 or val > 300: 
                        continue

                    # Colors
                    if val < 50:
                        color, rating = '#d32f2f', 'POOR'   # Red
                    elif val < 70:
                        color, rating = '#ff9800', 'FAIR'   # Orange
                    elif val < 95:
                        color, rating = '#4caf50', 'GOOD'   # Green
                    else:
                        color, rating = '#2196f3', 'EXCELLENT' # Blue
                    
                    folium.Circle(
                        location=[lat, lon],
                        radius=2500,
                        color=color,
                        fill=True,
                        fill_opacity=0.6,
                        weight=0,
                        tooltip=f"<b>Time:</b> {time_str}<br><b>Rating:</b> {rating}<br><b>Recovery:</b> {val:.0f}%"
                    ).add_to(fg)
                    count += 1
            
            fg.add_to(m)
            print(f"Processed f{fhr:02d}: Added {count} points.")
            
            ds.close() # Close file handle

        except Exception as e:
            print(f"Error processing f{fhr:02d}: {e}")
        
        finally:
            # CLEANUP: Delete the huge GRIB file
            if os.path.exists(grib_file):
                os.remove(grib_file)
                print(f"Deleted {grib_file}")

    # 4. SAVE OUTPUT
    os.makedirs('public', exist_ok=True)
    m.save(OUTPUT_HTML)
    print("Map generation complete.")

if __name__ == "__main__":
    main()

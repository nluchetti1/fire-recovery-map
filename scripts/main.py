import os
import sys
import numpy as np
import xarray as xr
import rasterio
from herbie import Herbie
import folium
from datetime import datetime
import matplotlib.colors as mcolors

# --- CONFIGURATION ---
FUEL_PATH = 'data/fuel_SE_final.tif' 
OUTPUT_HTML = 'public/index.html'

# MOE Lookup based on Anderson (1982) & NOAA TM-205
# Key: Fuel Model (1-13), Value: Moisture of Extinction (%) [cite: 307]
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
    Returns a grid of Fuel Models matching the shape of the weather data.
    """
    with rasterio.open(fuel_path) as src:
        # Flatten the weather grid arrays to a list of (x, y) points
        # Note: Rasterio sample expects (x, y) -> (lon, lat)
        coords = zip(target_lons.ravel(), target_lats.ravel())
        
        # Sample the TIFF (this is very fast)
        sampled_values = src.sample(coords)
        
        # Extract values (generator to array)
        fuel_flat = np.fromiter((val[0] for val in sampled_values), dtype=np.uint8)
        
    # Reshape back to the original weather grid shape
    return fuel_flat.reshape(target_lats.shape)

def main():
    print("--- Starting Fire Weather Recovery Map ---")
    
    # 1. SETUP & WEATHER DOWNLOAD
    now = datetime.utcnow()
    # If before 14Z, use 00Z run; else 12Z run (simplified logic)
    run_hour = 0 if now.hour < 14 else 12
    dt = datetime(now.year, now.month, now.day, run_hour, 0)
    print(f"Targeting HREF Run: {dt}")

    # Initialize Herbie (Download Forecast Hours 1-18)
    H = Herbie(dt, model='href', product='sfc', fxx=range(1, 19))

    try:
        # Download Temp and RH (grib2 format)
        ds_t = H.xarray("TMP:2 m above ground", remove_grib=True)
        ds_rh = H.xarray("RH:2 m above ground", remove_grib=True)
    except Exception as e:
        print(f"Weather download failed: {e}")
        sys.exit(1)

    # 2. PREPARE FUEL DATA
    print("Matching Fuel Data to Weather Grid...")
    lats = ds_t.latitude.values
    lons = ds_t.longitude.values
    
    # Create the Fuel Model Grid (this aligns your TIFF to the HREF grid)
    fuel_grid = sample_fuel_grid(FUEL_PATH, lats, lons)
    
    # Create MOE (Moisture of Extinction) Grid
    moe_grid = np.zeros_like(fuel_grid, dtype=float)
    for fid, moe_val in MOE_LOOKUP.items():
        moe_grid[fuel_grid == fid] = moe_val
    
    # Mask invalid areas (Urban/Water/Snow are often 90-99 in Landfire)
    # Set them to infinity so the recovery ratio becomes 0
    moe_grid[moe_grid == 0] = 999 
    moe_grid[fuel_grid > 13] = 999

    # 3. BUILD MAP
    m = folium.Map(location=[34, -84], zoom_start=6, tiles='CartoDB dark_matter')
    
    # Define Colors for Ratings (NOAA TM-205 Page 10) [cite: 345]
    # Poor (<50%), Fair (51-70%), Good (71-95%), Excellent (>95%)
    # Colors: Red, Orange, LightGreen, Blue
    
    # Loop through each forecast hour
    for step_idx in range(ds_t.step.size):
        # Get Weather Data for this hour
        t_k = ds_t['t2m'].isel(step=step_idx).values
        rh = ds_rh['r2'].isel(step=step_idx).values
        valid_time = ds_t.valid_time.isel(step=step_idx).values
        time_str = str(valid_time).split('T')[1][:5] + "Z"
        
        # Convert T to Fahrenheit
        t_f = (t_k - 273.15) * 9/5 + 32
        
        # Calculate Fuel Moisture
        fm_grid = calculate_emc(t_f, rh)
        
        # Calculate Recovery Ratio (%)
        recovery_grid = (fm_grid / moe_grid) * 100
        
        # --- Visualization (Subsampling for Performance) ---
        fg = folium.FeatureGroup(name=f"Hour {time_str}", show=False)
        
        # Skip every 15 points to keep the web map fast
        stride = 15
        
        for i in range(0, lats.shape[0], stride):
            for j in range(0, lats.shape[1], stride):
                val = recovery_grid[i, j]
                
                # Skip invalid data (urban/water)
                if val <= 0.1: continue
                
                # Assign Rating & Color [cite: 345]
                if val < 50:
                    color, rating = '#d32f2f', 'POOR'  # Red
                elif val < 70:
                    color, rating = '#ff9800', 'FAIR'  # Orange
                elif val < 95:
                    color, rating = '#4caf50', 'GOOD'  # Green
                else:
                    color, rating = '#2196f3', 'EXCELLENT' # Blue

                # Add Dot
                folium.Circle(
                    location=[lats[i,j], lons[i,j]],
                    radius=2500, # 2.5km dot
                    color=color,
                    fill=True,
                    fill_opacity=0.5,
                    weight=0,
                    tooltip=f"<b>Time:</b> {time_str}<br><b>Rating:</b> {rating}<br><b>Recovery:</b> {val:.0f}%"
                ).add_to(fg)
                
        fg.add_to(m)

    folium.LayerControl().add_to(m)
    
    os.makedirs('public', exist_ok=True)
    m.save(OUTPUT_HTML)
    print("Map generated successfully in public/index.html")

if __name__ == "__main__":
    main()

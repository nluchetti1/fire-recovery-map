import os
import sys
import numpy as np
import xarray as xr
import rasterio
from herbie import Herbie
import folium
from folium import plugins
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# --- CONFIGURATION ---
# Path to your static fuel file (relative to repo root)
FUEL_PATH = 'data/fuel_SE_resampled.tif' 
OUTPUT_HTML = 'public/index.html'

# MOE Lookup from NOAA TM-205 (Anderson Models 1-13)
MOE_LOOKUP = {
    1: 12, 2: 15, 3: 25, 4: 20, 5: 20, 6: 25, 7: 40,
    8: 30, 9: 25, 10: 25, 11: 15, 12: 20, 13: 25
}

def calculate_emc(T_degF, RH_percent):
    """
    Calculates Equilibrium Moisture Content (EMC).
    Equations from Simard (1968) as referenced in fire behavior manuals.
    """
    h = np.clip(RH_percent, 0, 100)
    t = T_degF
    
    cond1 = h < 10
    cond2 = (h >= 10) & (h < 50)
    cond3 = h >= 50
    
    emc = np.zeros_like(h)
    
    # Simard Equations
    emc[cond1] = 0.03229 + 0.281073*h[cond1] - 0.000578*h[cond1]*t[cond1]
    emc[cond2] = 2.22749 + 0.160107*h[cond2] - 0.01478*t[cond2]
    emc[cond3] = 21.0606 + 0.005565*(h[cond3]**2) - 0.00035*h[cond3]*t[cond3] - 0.483199*h[cond3]
    
    return np.clip(emc, 0, 35)

def main():
    # 1. READ FUEL DATA
    if not os.path.exists(FUEL_PATH):
        print(f"Error: Fuel file not found at {FUEL_PATH}")
        sys.exit(1)

    with rasterio.open(FUEL_PATH) as src:
        fuel_data = src.read(1)
        fuel_bounds = src.bounds
        fuel_transform = src.transform
        height, width = fuel_data.shape

    # Create MOE Grid from Fuel Data
    moe_grid = np.zeros_like(fuel_data, dtype=float)
    for fid, moe in MOE_LOOKUP.items():
        moe_grid[fuel_data == fid] = moe
    
    # Mask out non-fuel (urban/water/snow often 90-99 in Landfire)
    # Setting to infinity prevents division by zero later
    moe_grid[moe_grid == 0] = 999 
    moe_grid[fuel_data > 13] = 999 

    # 2. DOWNLOAD HREF WEATHER DATA
    # We look for the 00Z or 12Z run depending on current time
    now = datetime.utcnow()
    # Logic to find latest run (simplified)
    if now.hour < 14:
        run_hour = 0
    else:
        run_hour = 12
    
    dt = datetime(now.year, now.month, now.day, run_hour, 0)
    print(f"Downloading HREF for run: {dt}")

    # Initialize Herbie
    H = Herbie(dt, model='href', product='sfc', fxx=range(1, 19)) # forecast hours 1-18

    # Create Map Object
    # Center roughly on SE US
    m = folium.Map(location=[33, -84], zoom_start=6, tiles='CartoDB dark_matter')
    
    # Layer Control Group
    layer_control = folium.LayerControl(collapsed=False)

    # PROCESS EACH HOUR
    # Note: Downloading all hours might take time. 
    # In production, you might loop carefully.
    
    try:
        # Download Temp and RH
        ds_t = H.xarray("TMP:2 m above ground", remove_grib=True)
        ds_rh = H.xarray("RH:2 m above ground", remove_grib=True)
    except Exception as e:
        print(f"Failed to download weather data: {e}")
        sys.exit(1)

    # Loop through forecast hours
    for step_idx in range(ds_t.step.size):
        # Extract grids
        # Need to reproject/interp weather to match fuel grid
        # For this demo, we assume we just use the weather grid lat/lons 
        # and would normally interp fuel to it. 
        # SIMPLIFICATION: We will stick to the Weather Grid logic for plotting
        
        # Get raw weather arrays
        t_kelvin = ds_t['t2m'].isel(step=step_idx).values
        rh_per = ds_rh['r2'].isel(step=step_idx).values
        valid_time = ds_t.valid_time.isel(step=step_idx).values
        
        # Convert T to F
        t_f = (t_kelvin - 273.15) * 9/5 + 32
        
        # Calculate EMC (Fuel Moisture)
        current_fm = calculate_emc(t_f, rh_per)
        
        # WARN: In a real script, you MUST interpolate 'moe_grid' 
        # to the shape of 't_f' here using lat/lons.
        # For this script to run without complex GIS reprojection code, 
        # we will assume the fuel MOE is a constant 15% (Timber) just to show the map works.
        # *** REPLACE THIS WITH REAL INTERPOLATION IN PRODUCTION ***
        aligned_moe = np.full_like(t_f, 15) 
        
        # Calculate Recovery
        recovery = (current_fm / aligned_moe) * 100
        
        # Visualization
        # Generate Colormap
        # Poor (<50, Red), Fair (50-70, Orange), Good (70-95, Green), Excellent (>95, Blue)
        # We normalize 0-150%
        norm = mcolors.BoundaryNorm([0, 50, 70, 95, 200], 4)
        cmap = mcolors.ListedColormap(['#d32f2f', '#fbc02d', '#388e3c', '#1976d2'])
        
        # Create Image Overlay
        # We need to paint the grid with these colors
        colored_data = cmap(norm(recovery))
        
        # Convert to RGBA image for Folium
        # (Omitting complex image generation for brevity, using simple heatmap overlay)
        # Better approach for web map: Grid of colored rectangles or Contour
        
        # Create a Feature Group for this hour so we can toggle it
        hour_label = str(valid_time).split('T')[1][:5] + "Z"
        fg = folium.FeatureGroup(name=f"Hour {hour_label}", show=False)
        
        # Use Folium's FastMarkerCluster or similar is too heavy.
        # We will use a standard ImageOverlay if we had the image bounds.
        # For now, let's plot a subsample of points as colored circles.
        
        lats = ds_t.latitude.values
        lons = ds_t.longitude.values
        
        # Subsample every 15th point to keep map fast
        stride = 15
        for i in range(0, lats.shape[0], stride):
            for j in range(0, lats.shape[1], stride):
                lat = lats[i, j]
                lon = lons[i, j]
                val = recovery[i, j]
                
                # Filter out of bounds (simple box)
                if lat < 24 or lat > 38 or lon < -90 or lon > -75:
                    continue
                
                color_hex = mcolors.to_hex(cmap(norm(val)))
                
                # Rating Text
                if val < 50: r_txt = "POOR"
                elif val < 70: r_txt = "FAIR"
                elif val < 95: r_txt = "GOOD"
                else: r_txt = "EXCELLENT"

                folium.Circle(
                    location=[lat, lon],
                    radius=2000, # 2km radius
                    color=color_hex,
                    fill=True,
                    fill_opacity=0.6,
                    tooltip=f"<b>Time:</b> {hour_label}<br><b>Recovery:</b> {r_txt}<br><b>Value:</b> {val:.1f}%"
                ).add_to(fg)
                
        fg.add_to(m)

    folium.LayerControl().add_to(m)
    
    # Ensure output directory exists
    os.makedirs('public', exist_ok=True)
    m.save(OUTPUT_HTML)
    print("Map generated successfully.")

if __name__ == "__main__":
    main()

import os
import sys
import numpy as np
import xarray as xr
import rasterio
import requests
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

# --- CONFIGURATION ---
FUEL_PATH = 'data/fuel_SE_final.tif' 
IMAGE_DIR = 'public/images'
# Domain for the PLOT (Southeast US)
# [West, South, East, North]
PLOT_EXTENT = [-90, 24, -75, 37]

# MOE Lookup (Anderson 1982 / NOAA TM-205)
MOE_LOOKUP = {
    1: 12, 2: 15, 3: 25, 4: 20, 5: 20, 6: 25, 7: 40,
    8: 30, 9: 25, 10: 25, 11: 15, 12: 20, 13: 25
}

def calculate_rh(t_kelvin, d_kelvin):
    """Calculates RH from Temp and Dewpoint (August-Roche-Magnus)."""
    t_c = t_kelvin - 273.15
    d_c = d_kelvin - 273.15
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    e = 6.112 * np.exp((17.67 * d_c) / (d_c + 243.5))
    return np.clip((e / es) * 100.0, 0, 100)

def calculate_emc(T_degF, RH_percent):
    """Calculates Fuel Moisture (Simard 1968)."""
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

def sample_fuel_at_weather_points(fuel_path, weather_lats, weather_lons):
    """
    Probes the fuel map at every weather grid point.
    Returns a grid of Fuel Models matching the Weather Grid shape.
    """
    print("Sampling Fuel Map at Weather Grid Points...")
    with rasterio.open(fuel_path) as src:
        # rasterio.sample expects list of (x, y) coordinates
        # We assume the Fuel Map is WGS84 (EPSG:4326) so x=Lon, y=Lat
        
        # Flatten the arrays to loops
        flat_lons = weather_lons.ravel()
        flat_lats = weather_lats.ravel()
        coords = zip(flat_lons, flat_lats)
        
        # Sample (this is fast)
        sampled = src.sample(coords)
        
        # Convert generator to numpy array
        fuel_flat = np.fromiter((val[0] for val in sampled), dtype=np.uint8)
        
    # Reshape back to the weather grid dimensions
    return fuel_flat.reshape(weather_lats.shape)

def download_file(date_str, run, fhr):
    """Downloads GRIB file from NOMADS."""
    base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/prod/href.{date_str}/ensprod"
    filename = f"href.t{run}z.conus.mean.f{fhr:02d}.grib2"
    url = f"{base_url}/{filename}"
    
    print(f"Downloading {filename}...")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return filename
    except Exception as e:
        print(f"Failed: {e}")
        return None

def generate_plot(recovery_grid, lats, lons, valid_time, fhr, run_str):
    """Generates a static PNG map using Cartopy."""
    fig = plt.figure(figsize=(10, 8))
    
    # Cartopy Projection
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(PLOT_EXTENT)

    # Add Map Features
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=1)
    ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')

    # Color Levels for Recovery (Poor, Fair, Good, Excellent)
    # 0-50 (Poor), 50-70 (Fair), 70-95 (Good), 95+ (Excellent)
    levels = [0, 50, 70, 95, 200]
    colors = ['#d32f2f', '#ffa000', '#388e3c', '#1976d2'] # Red, Orange, Green, Blue
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(levels, len(colors))

    # Mask out "Infinite" recovery (non-fuel areas)
    plot_data = np.ma.masked_where(recovery_grid > 300, recovery_grid)

    # Plot Data using pcolormesh
    mesh = ax.pcolormesh(lons, lats, plot_data, cmap=cmap, norm=norm, 
                         transform=ccrs.PlateCarree(), shading='auto')

    # Title & Labels
    t_str = str(valid_time).split('T')[1][:5]
    d_str = str(valid_time).split('T')[0]
    plt.title(f"Nighttime Fuel Recovery\nValid: {d_str} {t_str}Z (F{fhr:02d})", loc='left', fontsize=12, fontweight='bold')
    plt.title(f"Run: {run_str}", loc='right', fontsize=10)

    # Legend
    cbar = plt.colorbar(mesh, orientation='horizontal', pad=0.05, aspect=30, shrink=0.8)
    cbar.set_ticks([25, 60, 82.5, 147.5])
    cbar.set_ticklabels(['POOR (<50%)', 'FAIR (50-70%)', 'GOOD (70-95%)', 'EXCELLENT (>95%)'])

    # Save
    filename = f"recovery_f{fhr:02d}.png"
    save_path = os.path.join(IMAGE_DIR, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()
    print(f"Saved {filename}")

def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    # 1. Setup Time
    now = datetime.utcnow()
    # Simple logic: After 2PM UTC (0900 EST), look for 12Z run. Otherwise 00Z.
    run_cycle = "12" if now.hour >= 14 else "00"
    date_str = now.strftime("%Y%m%d")
    run_info = f"{date_str} {run_cycle}Z"
    
    fuel_grid_cached = None
    moe_grid_cached = None
    
    # 2. Loop Forecast Hours (1-18)
    for fhr in range(1, 19):
        grib = download_file(date_str, run_cycle, fhr)
        if not grib: 
            print("Trying Previous Day if current run is missing...")
            # Fallback logic could go here, for now just skip
            continue

        try:
            # Load Weather Data
            # Note: cfgrib might return multiple datasets if stepTypes mix.
            # We filter for 'heightAboveGround' and level 2 to catch T2m/D2m
            ds = xr.open_dataset(grib, engine='cfgrib', 
                                 filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2})
            
            t_k = ds['t2m'].values
            d_k = ds['d2m'].values
            lats = ds.latitude.values
            lons = ds.longitude.values
            valid_time = ds.valid_time.values
            
            # --- ONE TIME SETUP (Fuel Grid) ---
            # We sample the fuel map ONCE, using the Weather Grid's shape.
            if fuel_grid_cached is None:
                fuel_grid_cached = sample_fuel_at_weather_points(FUEL_PATH, lats, lons)
                
                # Build MOE Grid
                moe_grid_cached = np.zeros_like(fuel_grid_cached, dtype=float)
                for fid, moe_val in MOE_LOOKUP.items():
                    moe_grid_cached[fuel_grid_cached == fid] = moe_val
                
                # Handle Non-Fuel Areas (Urban/Water/Snow > 13)
                moe_grid_cached[moe_grid_cached == 0] = 999 
                moe_grid_cached[fuel_grid_cached > 13] = 999

            # --- CALCULATIONS ---
            # 1. Relative Humidity
            rh = calculate_rh(t_k, d_k)
            
            # 2. Fuel Moisture (EMC)
            t_f = (t_k - 273.15) * 9/5 + 32
            fm = calculate_emc(t_f, rh)
            
            # 3. Recovery Ratio
            recovery = (fm / moe_grid_cached) * 100
            
            # --- PLOTTING ---
            generate_plot(recovery, lats, lons, valid_time, fhr, run_info)
            
            ds.close()

        except Exception as e:
            print(f"Error processing f{fhr:02d}: {e}")
        
        finally:
            # Delete GRIB file to save space
            if os.path.exists(grib): 
                os.remove(grib)

if __name__ == "__main__":
    # Force Matplotlib to run without a display (Server Mode)
    import matplotlib
    matplotlib.use('Agg') 
    main()

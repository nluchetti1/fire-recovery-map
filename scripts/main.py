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

# DOMAIN: Southeast US [West, South, East, North]
# This clips the data AND sets the map view
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
    """
    print("Sampling Fuel Map at Weather Grid Points...")
    with rasterio.open(fuel_path) as src:
        # Flatten arrays for sampling
        flat_lons = weather_lons.ravel()
        flat_lats = weather_lats.ravel()
        coords = zip(flat_lons, flat_lats)
        
        # Sample (returns generator)
        sampled = src.sample(coords)
        
        # Convert to numpy array
        fuel_flat = np.fromiter((val[0] for val in sampled), dtype=np.uint8)
        
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
    """Generates a static PNG map zoomed into the SE US."""
    fig = plt.figure(figsize=(10, 8))
    
    # Use PlateCarree (standard lat/lon)
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # FORCE THE EXTENT (Zoom in)
    ax.set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())

    # Add Map Features
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=1)
    ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')

    # Color Levels: Poor (<50), Fair (50-70), Good (70-95), Excellent (>95)
    levels = [0, 50, 70, 95, 200]
    colors = ['#d32f2f', '#ffa000', '#388e3c', '#1976d2'] # Red, Orange, Green, Blue
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(levels, len(colors))

    # Mask invalid data (where recovery is 0 or crazy high)
    plot_data = np.ma.masked_where((recovery_grid < 1) | (recovery_grid > 300), recovery_grid)

    # Plot
    mesh = ax.pcolormesh(lons, lats, plot_data, cmap=cmap, norm=norm, 
                         transform=ccrs.PlateCarree(), shading='auto')

    # Labels
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
    run_cycle = "12" if now.hour >= 14 else "00"
    date_str = now.strftime("%Y%m%d")
    run_info = f"{date_str} {run_cycle}Z"
    
    fuel_grid_subset = None
    moe_grid_subset = None
    
    # 2. Loop Forecast Hours
    for fhr in range(1, 19):
        grib = download_file(date_str, run_cycle, fhr)
        if not grib: continue

        try:
            # Load Weather Data
            ds = xr.open_dataset(grib, engine='cfgrib', 
                                 filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2})
            
            # --- CRITICAL FIX: SUBSET DATA TO DOMAIN ---
            # This makes plotting faster AND fixes the zoom issue
            # We add a buffer of 1 degree to ensure full coverage
            min_lon = PLOT_EXTENT[0] + 360 # Convert -90 to 270 for slicing GRIB
            max_lon = PLOT_EXTENT[2] + 360
            min_lat = PLOT_EXTENT[1]
            max_lat = PLOT_EXTENT[3]
            
            # Slice the dataset (much faster than processing full CONUS)
            ds_sub = ds.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))
            
            # Extract Vars
            t_k = ds_sub['t2m'].values
            d_k = ds_sub['d2m'].values
            lats = ds_sub.latitude.values
            
            # FIX LONGITUDE: Convert 270 -> -90
            lons_raw = ds_sub.longitude.values
            lons = np.where(lons_raw > 180, lons_raw - 360, lons_raw)
            
            valid_time = ds_sub.valid_time.values

            # --- ONE TIME SETUP (Fuel Grid) ---
            if fuel_grid_subset is None:
                # Sample using the NEW subset coordinates
                fuel_grid_subset = sample_fuel_at_weather_points(FUEL_PATH, lats, lons)
                
                moe_grid_subset = np.zeros_like(fuel_grid_subset, dtype=float)
                for fid, moe_val in MOE_LOOKUP.items():
                    moe_grid_subset[fuel_grid_subset == fid] = moe_val
                
                moe_grid_subset[moe_grid_subset == 0] = 999 
                moe_grid_subset[fuel_grid_subset > 13] = 999

            # --- CALCULATION & PLOTTING ---
            rh = calculate_rh(t_k, d_k)
            t_f = (t_k - 273.15) * 9/5 + 32
            fm = calculate_emc(t_f, rh)
            recovery = (fm / moe_grid_subset) * 100
            
            generate_plot(recovery, lats, lons, valid_time, fhr, run_info)
            
            ds.close()

        except Exception as e:
            print(f"Error f{fhr:02d}: {e}")
        finally:
            if os.path.exists(grib): os.remove(grib)

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    main()

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
FUEL_PATH = 'data/fuel_SE_fbfm40.tif' 
IMAGE_DIR = 'public/images'

# DOMAIN: Tight zoom on the Carolinas/GA/VA
# [West, East, South, North]
PLOT_EXTENT = [-85.0, -75.0, 31.0, 37.5] 

# MOE Lookup for Scott & Burgan 40 (FBFM40)
# derived from Scott & Burgan (2005) Standard Fire Behavior Fuel Models
MOE_LOOKUP = {
    # Grass (GR)
    101: 15, 102: 15, 103: 30, 104: 15, 105: 40, 106: 40, 107: 15, 108: 30, 109: 40,
    # Grass-Shrub (GS)
    121: 15, 122: 15, 123: 40, 124: 40,
    # Shrub (SH)
    141: 15, 142: 15, 143: 40, 144: 30, 145: 15, 146: 30, 147: 15, 148: 15, 149: 40,
    # Timber-Understory (TU)
    161: 20, 162: 30, 163: 30, 164: 12, 165: 25,
    # Timber Litter (TL)
    181: 30, 182: 30, 183: 20, 184: 25, 185: 25, 186: 25, 187: 25, 188: 35, 189: 25,
    # Slash (SB)
    201: 25, 202: 25, 203: 25, 204: 25
}

def get_domain_slice(ds, extent):
    """
    Finds x/y indices that bound the requested Lat/Lon extent.
    This replaces manual guessing.
    """
    lats = ds.latitude.values
    lons = ds.longitude.values
    
    # Fix Longitude (0-360 -> -180 to 180) for comparison
    lons = np.where(lons > 180, lons - 360, lons)
    
    # Create a boolean mask for the domain (pad by 1 degree)
    mask = (
        (lons >= extent[0] - 1.0) & (lons <= extent[1] + 1.0) &
        (lats >= extent[2] - 1.0) & (lats <= extent[3] + 1.0)
    )
    
    # Find indices where mask is True
    rows, cols = np.where(mask)
    
    if len(rows) == 0:
        print("Warning: Domain extent is outside the data grid!")
        return slice(None), slice(None)

    # Add a buffer (padding) to ensure we cover the edges
    pad = 5
    y_min, y_max = max(0, rows.min()-pad), min(lats.shape[0], rows.max()+pad)
    x_min, x_max = max(0, cols.min()-pad), min(lats.shape[1], cols.max()+pad)
    
    return slice(y_min, y_max), slice(x_min, x_max)

def calculate_rh(t_kelvin, d_kelvin):
    """Calculates RH from Temp and Dewpoint."""
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
    """Probes the fuel map at every weather grid point."""
    print("Sampling Fuel Map at Weather Grid Points...")
    with rasterio.open(fuel_path) as src:
        # Flatten arrays for sampling
        flat_lons = weather_lons.ravel()
        flat_lats = weather_lats.ravel()
        coords = zip(flat_lons, flat_lats)
        
        # Sample (returns generator)
        sampled = src.sample(coords)
        
        # Convert to numpy array
        fuel_flat = np.fromiter((val[0] for val in sampled), dtype=np.uint16) # uint16 for FBFM40 codes
        
    return fuel_flat.reshape(weather_lats.shape)

def download_file(date_str, run, fhr):
    """Downloads GRIB file."""
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
    """Generates PNG map using Lambert Conformal (Standard US Weather View)."""
    fig = plt.figure(figsize=(10, 8))
    
    # Standard NWS-style Projection (Centered roughly on SE US)
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-80, central_latitude=34))
    
    # Force the Zoom to the configured extent
    ax.set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())

    # Map Features
    ax.add_feature(cfeature.COASTLINE, linewidth=1.5, zorder=10)
    ax.add_feature(cfeature.BORDERS, linewidth=1.5, zorder=10)
    ax.add_feature(cfeature.STATES, linewidth=1.0, edgecolor='black', zorder=10)
    ax.add_feature(cfeature.OCEAN, facecolor='#cceeff')
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0')

    # [cite_start]Color Levels [cite: 332-340]
    levels = [0, 50, 70, 95, 200]
    colors = ['#d32f2f', '#ffa000', '#388e3c', '#1976d2'] 
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(levels, len(colors))

    # Mask invalid values so they appear transparent (showing the LAND color)
    # This hides the "Red Background" and "White Points"
    plot_data = np.ma.masked_invalid(recovery_grid)

    # Plot
    mesh = ax.pcolormesh(lons, lats, plot_data, cmap=cmap, norm=norm, 
                         transform=ccrs.PlateCarree(), shading='auto', zorder=5)

    # Titles
    t_str = str(valid_time).split('T')[1][:5]
    d_str = str(valid_time).split('T')[0]
    plt.title(f"Nighttime Fuel Recovery (FBFM40)\nValid: {d_str} {t_str}Z (F{fhr:02d})", loc='left', fontsize=12, fontweight='bold')
    plt.title(f"Run: {run_str}", loc='right', fontsize=10)

    # Legend
    cbar = plt.colorbar(mesh, orientation='horizontal', pad=0.05, aspect=35, shrink=0.8)
    cbar.set_ticks([25, 60, 82.5, 147.5])
    cbar.set_ticklabels(['POOR', 'FAIR', 'GOOD', 'EXCELLENT'])

    filename = f"recovery_f{fhr:02d}.png"
    save_path = os.path.join(IMAGE_DIR, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()
    print(f"Saved {filename}")

def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    now = datetime.utcnow()
    run_cycle = "12" if now.hour >= 14 else "00"
    date_str = now.strftime("%Y%m%d")
    run_info = f"{date_str} {run_cycle}Z"
    
    fuel_grid_subset = None
    moe_grid_subset = None
    valid_fuel_mask = None
    y_slice, x_slice = None, None

    # Loop Forecast Hours 1-18
    for fhr in range(1, 19):
        grib = download_file(date_str, run_cycle, fhr)
        if not grib: continue

        try:
            # 1. Load FULL CONUS grid
            ds = xr.open_dataset(grib, engine='cfgrib', 
                                 filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2})
            
            # 2. Calculate Slicing Indices (First time only)
            if y_slice is None:
                print("Calculating domain indices...")
                y_slice, x_slice = get_domain_slice(ds, PLOT_EXTENT)
            
            # 3. Clip Data to Domain
            ds_sub = ds.isel(y=y_slice, x=x_slice)
            
            t_k = ds_sub['t2m'].values
            d_k = ds_sub['d2m'].values
            lats = ds_sub.latitude.values
            
            # Fix Longitude (0-360 -> -180 to 180)
            lons_raw = ds_sub.longitude.values
            lons = np.where(lons_raw > 180, lons_raw - 360, lons_raw)
            
            valid_time = ds_sub.valid_time.values

            # 4. Sample Fuel (One time setup)
            if fuel_grid_subset is None:
                fuel_grid_subset = sample_fuel_at_weather_points(FUEL_PATH, lats, lons)
                
                # Create MOE Grid
                moe_grid_subset = np.zeros_like(fuel_grid_subset, dtype=float)
                for fid, moe_val in MOE_LOOKUP.items():
                    moe_grid_subset[fuel_grid_subset == fid] = moe_val
                
                # Create Valid Mask for FBFM40 (Codes 101-204)
                valid_fuel_mask = (fuel_grid_subset >= 101) & (fuel_grid_subset <= 204)
                
                # Set invalid MOE to avoid division by zero (or bogus results)
                moe_grid_subset[~valid_fuel_mask] = 999 

            # 5. Calc & Plot
            rh = calculate_rh(t_k, d_k)
            t_f = (t_k - 273.15) * 9/5 + 32
            fm = calculate_emc(t_f, rh)
            
            recovery = (fm / moe_grid_subset) * 100
            
            # Mask out non-fuel areas (Ocean/Urban/NoData)
            recovery = np.where(valid_fuel_mask, recovery, np.nan)
            
            generate_plot(recovery, lats, lons, valid_time, fhr, run_info)
            
            ds.close()

        except Exception as e:
            print(f"Error f{fhr:02d}: {e}")
        finally:
            if os.path.exists(grib): os.remove(grib)

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg') # Run without display
    main()

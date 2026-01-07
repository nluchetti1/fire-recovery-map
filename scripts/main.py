import os
import sys
import numpy as np
import xarray as xr
import rasterio
import requests
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
from rasterio.enums import Resampling

# --- CONFIGURATION ---
FUEL_PATH = 'data/fuel_SE_final.tif' 
IMAGE_DIR = 'public/images'
domain = [-90, 24, -75, 37] # [West, South, East, North]

# [cite_start]MOE Lookup (Anderson 1982 / NOAA TM-205) [cite: 234, 306]
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
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(domain)

    # Add Map Features
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=1)
    ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='gray')

    # [cite_start]Color Levels for Recovery (Poor, Fair, Good, Excellent) [cite: 330]
    levels = [0, 50, 70, 95, 200]
    colors = ['#d32f2f', '#ffa000', '#388e3c', '#1976d2'] # Red, Orange, Green, Blue
    cmap = matplotlib.colors.ListedColormap(colors)
    norm = matplotlib.colors.BoundaryNorm(levels, len(colors))

    # Plot Data
    mesh = ax.pcolormesh(lons, lats, recovery_grid, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading='auto')

    # Title & Labels
    t_str = str(valid_time).split('T')[1][:5]
    d_str = str(valid_time).split('T')[0]
    plt.title(f"Fire Fuel Recovery (Nighttime)\nValid: {d_str} {t_str}Z (F{fhr:02d})", loc='left', fontsize=12, fontweight='bold')
    plt.title(f"HREF Run: {run_str}", loc='right', fontsize=10)

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
    
    # 2. Load Fuel Data (Resample ON THE FLY if needed, but file must be small)
    print("Loading Fuel Data...")
    with rasterio.open(FUEL_PATH) as src:
        # We read the fuel data. We will interp weather TO this grid for plotting.
        # Ideally, we read a window matching our domain to save RAM.
        window = rasterio.windows.from_bounds(*domain, transform=src.transform)
        fuel_data = src.read(1, window=window)
        fuel_transform = src.window_transform(window)
        height, width = fuel_data.shape
        
        # Create coordinate grids for the fuel map
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(fuel_transform, rows, cols, offset='center')
        lons_fuel = np.array(xs)
        lats_fuel = np.array(ys)

    # Build MOE Grid
    moe_grid = np.zeros_like(fuel_data, dtype=float)
    for fid, moe_val in MOE_LOOKUP.items():
        moe_grid[fuel_data == fid] = moe_val
    moe_grid[moe_grid == 0] = 999 

    # 3. Loop Forecast Hours
    for fhr in range(1, 19):
        grib = download_file(date_str, run_cycle, fhr)
        if not grib: continue

        try:
            ds = xr.open_dataset(grib, engine='cfgrib', filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2})
            
            # Interpolate Weather to Fuel Grid
            # (We use the fuel grid as the master for the plot so the fuel lines are sharp)
            t_k = ds['t2m'].interp(latitude=xr.DataArray(lats_fuel, dims=("y", "x")), 
                                   longitude=xr.DataArray(lons_fuel, dims=("y", "x")), 
                                   method="nearest").values
            d_k = ds['d2m'].interp(latitude=xr.DataArray(lats_fuel, dims=("y", "x")), 
                                   longitude=xr.DataArray(lons_fuel, dims=("y", "x")), 
                                   method="nearest").values
            
            # Calculate
            rh = calculate_rh(t_k, d_k)
            t_f = (t_k - 273.15) * 9/5 + 32
            fm = calculate_emc(t_f, rh)
            recovery = (fm / moe_grid) * 100
            
            # Plot
            generate_plot(recovery, lats_fuel, lons_fuel, ds.valid_time.values, fhr, run_info)
            
            ds.close()
        except Exception as e:
            print(f"Error f{fhr:02d}: {e}")
        finally:
            if os.path.exists(grib): os.remove(grib)

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg') # Run without display
    main()

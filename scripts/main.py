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
from metpy.plots import USCOUNTIES
from datetime import datetime, timedelta

# --- CONFIGURATION ---
FUEL_PATH = 'data/fuel_SE_fbfm40.tif' 
IMAGE_DIR = 'public/images'
PLOT_EXTENT = [-85.0, -75.0, 31.0, 37.5] 

# MOE Lookup for Scott & Burgan 40 (FBFM40)
MOE_LOOKUP = {
    101: 15, 102: 15, 103: 30, 104: 15, 105: 40, 106: 40, 107: 15, 108: 30, 109: 40,
    121: 15, 122: 15, 123: 40, 124: 40,
    141: 15, 142: 15, 143: 40, 144: 30, 145: 15, 146: 30, 147: 15, 148: 15, 149: 40,
    161: 20, 162: 30, 163: 30, 164: 12, 165: 25,
    181: 30, 182: 30, 183: 20, 184: 25, 185: 25, 186: 25, 187: 25, 188: 35, 189: 25,
    201: 25, 202: 25, 203: 25, 204: 25
}

def get_domain_slice(ds, extent):
    """Finds x/y indices for the extent."""
    lats = ds.latitude.values
    lons = ds.longitude.values
    lons = np.where(lons > 180, lons - 360, lons)
    
    mask = (
        (lons >= extent[0] - 1.0) & (lons <= extent[1] + 1.0) &
        (lats >= extent[2] - 1.0) & (lats <= extent[3] + 1.0)
    )
    
    rows, cols = np.where(mask)
    if len(rows) == 0: return slice(None), slice(None)

    pad = 5
    y_min, y_max = max(0, rows.min()-pad), min(lats.shape[0], rows.max()+pad)
    x_min, x_max = max(0, cols.min()-pad), min(lats.shape[1], cols.max()+pad)
    
    return slice(y_min, y_max), slice(x_min, x_max)

def calculate_rh(t_kelvin, d_kelvin):
    t_c = t_kelvin - 273.15
    d_c = d_kelvin - 273.15
    es = 6.112 * np.exp((17.67 * t_c) / (t_c + 243.5))
    e = 6.112 * np.exp((17.67 * d_c) / (d_c + 243.5))
    return np.clip((e / es) * 100.0, 0, 100)

def calculate_emc(T_degF, RH_percent):
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

def prepare_fuel_grid(fuel_path, lats, lons):
    """Loads and resamples the fuel grid to match weather coordinates."""
    # print("Sampling Fuel Map...") 
    # Commented out print to reduce log spam, enable if debugging
    with rasterio.open(fuel_path) as src:
        flat_lons = lons.ravel()
        flat_lats = lats.ravel()
        coords = zip(flat_lons, flat_lats)
        sampled = src.sample(coords)
        fuel_flat = np.fromiter((val[0] for val in sampled), dtype=np.uint16)
        
    fuel_grid = fuel_flat.reshape(lats.shape)
    
    moe_grid = np.zeros_like(fuel_grid, dtype=float)
    for fid, moe_val in MOE_LOOKUP.items():
        moe_grid[fuel_grid == fid] = moe_val
        
    valid_mask = (fuel_grid >= 101) & (fuel_grid <= 204)
    moe_grid[~valid_mask] = 999 
    
    return moe_grid, valid_mask

def download_file(url, local_filename):
    print(f"Downloading {local_filename}...")
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filename
    except Exception as e:
        print(f"Failed to download {local_filename}: {e}")
        return None

def generate_recovery_map(t_k, d_k, moe_grid, valid_mask):
    rh = calculate_rh(t_k, d_k)
    t_f = (t_k - 273.15) * 9/5 + 32
    fm = calculate_emc(t_f, rh)
    recovery = (fm / moe_grid) * 100
    return np.where(valid_mask, recovery, np.nan)

def plot_verification(f_rec, f_lats, f_lons, o_rec, o_lats, o_lons, valid_time):
    """
    Plots Forecast (Left) vs Observed (Right).
    Accepts separate lat/lon arrays for each since grids differ.
    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 8), 
                           subplot_kw={'projection': ccrs.LambertConformal(central_longitude=-80, central_latitude=34)})
    
    # Common Plot Settings
    levels = [0, 50, 70, 95, 200]
    colors = ['#d32f2f', '#ffa000', '#388e3c', '#1976d2'] 
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(levels, len(colors))
    
    # 1. Forecast Plot
    ax[0].set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())
    ax[0].add_feature(cfeature.STATES, linewidth=1.5)
    ax[0].add_feature(USCOUNTIES.with_scale('5m'), linewidth=0.5, alpha=0.5)
    ax[0].set_title(f"HREF Forecast (09Z)\nValid: {valid_time} UTC", fontweight='bold')
    mesh = ax[0].pcolormesh(f_lons, f_lats, f_rec, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading='auto')
    
    # 2. Observed Plot
    ax[1].set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())
    ax[1].add_feature(cfeature.STATES, linewidth=1.5)
    ax[1].add_feature(USCOUNTIES.with_scale('5m'), linewidth=0.5, alpha=0.5)
    ax[1].set_title(f"RTMA Observed (09Z)\nValid: {valid_time} UTC", fontweight='bold')
    ax[1].pcolormesh(o_lons, o_lats, o_rec, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading='auto')

    # Legend
    cbar = plt.colorbar(mesh, ax=ax.ravel().tolist(), orientation='horizontal', pad=0.05, aspect=50, shrink=0.6)
    cbar.set_ticks([25, 60, 82.5, 147.5])
    cbar.set_ticklabels(['POOR', 'FAIR', 'GOOD', 'EXCELLENT'])
    
    save_path = os.path.join(IMAGE_DIR, "verification_09z.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print("Saved Verification Plot")

def generate_main_plot(recovery_grid, lats, lons, valid_time, fhr, run_str):
    """Standard single-panel plot for the forecast loop."""
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=-80, central_latitude=34))
    ax.set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.STATES, linewidth=1.5, edgecolor='black', zorder=10)
    try:
        ax.add_feature(USCOUNTIES.with_scale('5m'), linewidth=0.8, edgecolor='black', zorder=11, alpha=0.6)
    except: pass
    ax.add_feature(cfeature.OCEAN, facecolor='#cceeff')
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0')

    levels = [0, 50, 70, 95, 200]
    colors = ['#d32f2f', '#ffa000', '#388e3c', '#1976d2'] 
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(levels, len(colors))
    plot_data = np.ma.masked_invalid(recovery_grid)

    mesh = ax.pcolormesh(lons, lats, plot_data, cmap=cmap, norm=norm, 
                         transform=ccrs.PlateCarree(), shading='auto', zorder=5)

    t_str = str(valid_time).split('T')[1][:5]
    d_str = str(valid_time).split('T')[0]
    plt.title(f"Nighttime Fuel Recovery\nValid: {d_str} {t_str}Z (F{fhr:02d})", loc='left', fontsize=12, fontweight='bold')
    plt.title(f"Run: {run_str}", loc='right', fontsize=10)
    cbar = plt.colorbar(mesh, orientation='horizontal', pad=0.05, aspect=35, shrink=0.8)
    cbar.set_ticks([25, 60, 82.5, 147.5])
    cbar.set_ticklabels(['POOR', 'FAIR', 'GOOD', 'EXCELLENT'])

    filename = f"recovery_f{fhr:02d}.png"
    save_path = os.path.join(IMAGE_DIR, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()
    print(f"Saved {filename}")

def run_verification_logic(moe_grid, valid_mask, h_lats, h_lons, y_sl, x_sl):
    """
    Runs the verification.
    Arguments:
      - moe_grid/valid_mask: Pre-calculated fuel mask for HREF grid
      - h_lats/h_lons: Coordinates for HREF grid
      - y_sl, x_sl: Slice indices to crop the grids efficiently
    """
    print("--- Starting Verification ---")
    today = datetime.utcnow().date()
    today_str = today.strftime("%Y%m%d")
    
    # Target: 09Z this morning
    fhr = 9
    
    href_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/prod/href.{today_str}/ensprod/href.t00z.conus.mean.f{fhr:02d}.grib2"
    href_file = download_file(href_url, "verif_href.grib2")
    
    rtma_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/rtma/prod/rtma2p5.{today_str}/rtma2p5.t09z.2dvaranl_ndfd.grb2_wexp"
    rtma_file = download_file(rtma_url, "verif_rtma.grib2")
    
    if not href_file or not rtma_file:
        print("Skipping verification (Files not available)")
        return

    try:
        # --- PROCESS HREF (Forecast) ---
        ds_href = xr.open_dataset(href_file, engine='cfgrib', 
                                  filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2})
        
        # SLICE using the indices passed from Main Loop (Robust)
        ds_href_sub = ds_href.isel(y=y_sl, x=x_sl)
        rec_href = generate_recovery_map(ds_href_sub['t2m'].values, ds_href_sub['d2m'].values, moe_grid, valid_mask)
        
        # --- PROCESS RTMA (Observed) ---
        ds_rtma = xr.open_dataset(rtma_file, engine='cfgrib')
        
        # RTMA has a DIFFERENT grid. We must compute a new slice and fuel mask.
        r_ysl, r_xsl = get_domain_slice(ds_rtma, PLOT_EXTENT)
        ds_rtma_sub = ds_rtma.isel(y=r_ysl, x=r_xsl)
        
        # Get RTMA Coordinates
        r_lats = ds_rtma_sub.latitude.values
        r_lons = ds_rtma_sub.longitude.values
        r_lons = np.where(r_lons > 180, r_lons - 360, r_lons)
        
        # Generate Fuel Mask specifically for RTMA points
        r_moe, r_mask = prepare_fuel_grid(FUEL_PATH, r_lats, r_lons)
        
        # Note: RTMA variable names can vary. Usually 't2m'/'d2m' or '2t'/'2d'.
        # xarray/cfgrib usually standardizes to t2m/d2m but check keys if it fails.
        t_var = 't2m' if 't2m' in ds_rtma_sub else '2t'
        d_var = 'd2m' if 'd2m' in ds_rtma_sub else '2d'
        
        rec_rtma = generate_recovery_map(ds_rtma_sub[t_var].values, ds_rtma_sub[d_var].values, r_moe, r_mask)
        
        # Plot with separate coordinates
        plot_verification(rec_href, h_lats, h_lons, rec_rtma, r_lats, r_lons, f"{today_str} 09:00")
        
        ds_href.close()
        ds_rtma.close()
        
    except Exception as e:
        print(f"Verification Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists("verif_href.grib2"): os.remove("verif_href.grib2")
        if os.path.exists("verif_rtma.grib2"): os.remove("verif_rtma.grib2")

def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    now = datetime.utcnow()
    run_cycle = "12" if now.hour >= 14 else "00"
    date_str = now.strftime("%Y%m%d")
    run_info = f"{date_str} {run_cycle}Z"
    
    # Store these to pass to verification later
    global_lats, global_lons = None, None
    global_moe, global_mask = None, None
    y_slice, x_slice = None, None # Keep indices for slicing verification HREF

    # --- 48 HOUR FORECAST LOOP ---
    for fhr in range(1, 49):
        base_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/prod/href.{date_str}/ensprod"
        filename = f"href.t{run_cycle}z.conus.mean.f{fhr:02d}.grib2"
        full_url = f"{base_url}/{filename}"
        
        grib = download_file(full_url, filename)
        if not grib: continue

        try:
            ds = xr.open_dataset(grib, engine='cfgrib', 
                                 filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2})
            
            if global_lats is None:
                y_slice, x_slice = get_domain_slice(ds, PLOT_EXTENT)
                ds_sub = ds.isel(y=y_slice, x=x_slice)
                
                global_lats = ds_sub.latitude.values
                lons_raw = ds_sub.longitude.values
                global_lons = np.where(lons_raw > 180, lons_raw - 360, lons_raw)
                
                global_moe, global_mask = prepare_fuel_grid(FUEL_PATH, global_lats, global_lons)
            else:
                ds_sub = ds.isel(y=y_slice, x=x_slice)

            recovery = generate_recovery_map(ds_sub['t2m'].values, ds_sub['d2m'].values, global_moe, global_mask)
            
            generate_main_plot(recovery, global_lats, global_lons, ds_sub.valid_time.values, fhr, run_info)
            ds.close()

        except Exception as e:
            print(f"Error f{fhr:02d}: {e}")
        finally:
            if os.path.exists(filename): os.remove(filename)

    # --- RUN VERIFICATION (After Forecast is done) ---
    if global_lats is not None:
        run_verification_logic(global_moe, global_mask, global_lats, global_lons, y_slice, x_slice)

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    main()

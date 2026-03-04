import warnings
warnings.filterwarnings("ignore")

import os
import shutil
import requests
import zipfile
from datetime import datetime, timedelta, timezone

import numpy as np
import xarray as xr
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.plots import USCOUNTIES

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
    print(f"Downloading from: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        with requests.get(url, stream=True, timeout=60, headers=headers) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return local_filename
    except Exception as e:
        print(f" -> Failed to download {local_filename}: {e}")
        return None

def generate_recovery_map(t_k, d_k, moe_grid, valid_mask):
    rh = calculate_rh(t_k, d_k)
    t_f = (t_k - 273.15) * 9/5 + 32
    fm = calculate_emc(t_f, rh)
    recovery = (fm / moe_grid) * 100
    return np.where(valid_mask, recovery, np.nan)

def generate_recovery_map_from_rh(t_k, rh, moe_grid, valid_mask):
    t_f = (t_k - 273.15) * 9/5 + 32
    fm = calculate_emc(t_f, rh)
    recovery = (fm / moe_grid) * 100
    return np.where(valid_mask, recovery, np.nan)

def plot_verification(f_rec, f_lats, f_lons, o_rec, o_lats, o_lons, valid_time, save_name, hour_str):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8), 
                           subplot_kw={'projection': ccrs.LambertConformal(central_longitude=-80, central_latitude=34)})
    
    levels = [0, 50, 70, 95, 200]
    colors = ['#d32f2f', '#ffa000', '#388e3c', '#1976d2'] 
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(levels, len(colors))
    
    ax[0].set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())
    ax[0].add_feature(cfeature.STATES, linewidth=1.5)
    ax[0].add_feature(USCOUNTIES.with_scale('5m'), linewidth=0.5, alpha=0.5)
    ax[0].set_title(f"HREF Forecast ({hour_str})\nValid: {valid_time} UTC", fontweight='bold')
    mesh = ax[0].pcolormesh(f_lons, f_lats, f_rec, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading='auto')
    
    ax[1].set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())
    ax[1].add_feature(cfeature.STATES, linewidth=1.5)
    ax[1].add_feature(USCOUNTIES.with_scale('5m'), linewidth=0.5, alpha=0.5)
    ax[1].set_title(f"RTMA Observed ({hour_str})\nValid: {valid_time} UTC", fontweight='bold')
    ax[1].pcolormesh(o_lons, o_lats, o_rec, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading='auto')

    cbar = plt.colorbar(mesh, ax=ax.ravel().tolist(), orientation='horizontal', pad=0.05, aspect=50, shrink=0.6)
    cbar.set_ticks([25, 60, 82.5, 147.5])
    cbar.set_ticklabels(['POOR', 'FAIR', 'GOOD', 'EXCELLENT'])
    
    save_path = os.path.join(IMAGE_DIR, save_name)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved Verification Plot: {save_name}")

def generate_main_plot(recovery_grid, lats, lons, valid_time, fhr, run_str, model="HREF"):
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
    plt.title(f"{model} Nighttime Fuel Recovery\nValid: {d_str} {t_str}Z (F{fhr:02d})", loc='left', fontsize=12, fontweight='bold')
    plt.title(f"Run: {run_str}", loc='right', fontsize=10)
    cbar = plt.colorbar(mesh, orientation='horizontal', pad=0.05, aspect=35, shrink=0.8)
    cbar.set_ticks([25, 60, 82.5, 147.5])
    cbar.set_ticklabels(['POOR', 'FAIR', 'GOOD', 'EXCELLENT'])

    filename = f"{model.lower()}_recovery_f{fhr:02d}.png"
    save_path = os.path.join(IMAGE_DIR, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()
    print(f"Saved {filename}")

def generate_ntr_plot(recovery_grid, lats, lons, valid_time, fhr, run_str, model="HREF"):
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
    plt.title(f"{model} 3-Hour Trailing Avg Recovery\nValid: {d_str} {t_str}Z (F{fhr:02d})", loc='left', fontsize=12, fontweight='bold')
    plt.title(f"Run: {run_str}", loc='right', fontsize=10)
    cbar = plt.colorbar(mesh, orientation='horizontal', pad=0.05, aspect=35, shrink=0.8)
    cbar.set_ticks([25, 60, 82.5, 147.5])
    cbar.set_ticklabels(['POOR', 'FAIR', 'GOOD', 'EXCELLENT'])

    filename = f"ntr_{model.lower()}_f{fhr:02d}.png"
    save_path = os.path.join(IMAGE_DIR, filename)
    plt.savefig(save_path, bbox_inches='tight', dpi=100)
    plt.close()
    print(f"Saved {filename}")

def run_verification_logic(moe_grid, valid_mask, h_lats, h_lons, y_sl, x_sl):
    print("\n--- Starting Verification Suite (01Z - 12Z) ---")
    today = datetime.utcnow().date()
    today_str = today.strftime("%Y%m%d")
    
    verif_files = []
    
    for v_hour in range(1, 13):
        fhr = v_hour
        hour_str = f"{v_hour:02d}Z"
        
        href_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/href/prod/href.{today_str}/ensprod/href.t00z.conus.mean.f{fhr:02d}.grib2"
        href_file = download_file(href_url, f"verif_href_{hour_str}.grib2")
        
        rtma_url = f"https://nomads.ncep.noaa.gov/pub/data/nccf/com/rtma/prod/rtma2p5.{today_str}/rtma2p5.t{hour_str.lower()}.2dvaranl_ndfd.grb2_wexp"
        rtma_file = download_file(rtma_url, f"verif_rtma_{hour_str}.grib2")
        
        if not href_file or not rtma_file:
            continue

        try:
            ds_href = xr.open_dataset(href_file, engine='cfgrib', filter_by_keys={'typeOfLevel': 'heightAboveGround', 'level': 2})
            ds_href_sub = ds_href.isel(y=y_sl, x=x_sl)
            rec_href = generate_recovery_map(ds_href_sub['t2m'].values, ds_href_sub['d2m'].values, moe_grid, valid_mask)
            
            ds_rtma = xr.open_dataset(rtma_file, engine='cfgrib')
            r_ysl, r_xsl = get_domain_slice(ds_rtma, PLOT_EXTENT)
            ds_rtma_sub = ds_rtma.isel(y=r_ysl, x=r_xsl)
            
            r_lats = ds_rtma_sub.latitude.values
            r_lons = ds_rtma_sub.longitude.values
            r_lons = np.where(r_lons > 180, r_lons - 360, r_lons)
            r_moe, r_mask = prepare_fuel_grid(FUEL_PATH, r_lats, r_lons)
            
            t_var = 't2m' if 't2m' in ds_rtma_sub else '2t'
            d_var = 'd2m' if 'd2m' in ds_rtma_sub else '2d'
            rec_rtma = generate_recovery_map(ds_rtma_sub[t_var].values, ds_rtma_sub[d_var].values, r_moe, r_mask)
            
            save_name = f"verification_{hour_str.lower()}.png"
            plot_verification(rec_href, h_lats, h_lons, rec_rtma, r_lats, r_lons, f"{today_str} {hour_str[:2]}:00", save_name, hour_str)
            verif_files.append(os.path.join(IMAGE_DIR, save_name))
            
            if v_hour == 9:
                shutil.copy(os.path.join(IMAGE_DIR, save_name), os.path.join(IMAGE_DIR, "verification_09z.png"))
            
            ds_href.close()
            ds_rtma.close()
            
        except Exception as e:
            print(f"Verification Failed for {hour_str}: {e}")
        finally:
            if os.path.exists(href_file): os.remove(href_file)
            if os.path.exists(rtma_file): os.remove(rtma_file)

    zip_path = os.path.join(IMAGE_DIR, 'verification_suite.zip')
    try:
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for file in verif_files:
                if os.path.exists(file):
                    zipf.write(file, os.path.basename(file))
    except Exception as e:
        pass

def preserve_verification():
    url = "https://nluchetti1.github.io/fire-recovery-map/images/verification_09z.png"
    save_path = os.path.join(IMAGE_DIR, "verification_09z.png")
    try:
        r = requests.get(url)
        if r.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(r.content)
    except Exception:
        pass

def main():
    os.makedirs(IMAGE_DIR, exist_ok=True)
    
    now = datetime.utcnow()
    run_cycle = "12" if now.hour >= 14 else "00"
    date_str = now.strftime("%Y%m%d")
    run_info = f"{date_str} {run_cycle}Z"
    
    global_lats, global_lons = None, None
    global_moe, global_mask = None, None
    y_slice, x_slice = None, None 
    
    # Track the trailing 3 hours for NTR averages
    href_trailing = []
    ndfd_trailing = []

    # --- 1. HREF 48 HOUR FORECAST LOOP ---
    print("--- Processing HREF ---")
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
            
            # Plot standard hour map
            generate_main_plot(recovery, global_lats, global_lons, ds_sub.valid_time.values, fhr, run_info, model="HREF")
            
            # Trailing 3-Hour Average Logic
            href_trailing.append(recovery)
            if len(href_trailing) > 3:
                href_trailing.pop(0) # Keep only the last 3 hours
            
            avg_recovery = np.nanmean(href_trailing, axis=0)
            generate_ntr_plot(avg_recovery, global_lats, global_lons, ds_sub.valid_time.values, fhr, run_info, model="HREF")
                
            ds.close()

        except Exception as e:
            print(f"Error f{fhr:02d}: {e}")
        finally:
            if os.path.exists(filename): os.remove(filename)

    # --- 2. NDFD FORECAST LOOP ---
    print("\n--- Processing NDFD ---")
    ndfd_temp_url = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.temp.bin"
    ndfd_rh_url = "https://tgftp.nws.noaa.gov/SL.us008001/ST.opnl/DF.gr2/DC.ndfd/AR.conus/VP.001-003/ds.rhm.bin"
    
    temp_file = download_file(ndfd_temp_url, "ndfd_temp.grib2")
    rh_file = download_file(ndfd_rh_url, "ndfd_rh.grib2")

    if temp_file and rh_file:
        try:
            ds_t = xr.open_dataset(temp_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': '2t'}})
            ds_rh = xr.open_dataset(rh_file, engine='cfgrib', backend_kwargs={'filter_by_keys': {'shortName': '2r'}})

            n_ysl, n_xsl = get_domain_slice(ds_t, PLOT_EXTENT)
            ds_t_sub = ds_t.isel(y=n_ysl, x=n_xsl)
            ds_rh_sub = ds_rh.isel(y=n_ysl, x=n_xsl)
            
            n_lats = ds_t_sub.latitude.values
            n_lons = ds_t_sub.longitude.values
            n_lons = np.where(n_lons > 180, n_lons - 360, n_lons)
            n_moe, n_mask = prepare_fuel_grid(FUEL_PATH, n_lats, n_lons)

            try:
                ndfd_time_np = ds_t_sub.time.values
                ts = (ndfd_time_np - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
                ndfd_init_dt = datetime.utcfromtimestamp(ts)
                ndfd_run_info = ndfd_init_dt.strftime("%Y%m%d %HZ")
            except Exception:
                ndfd_run_info = "Operational Run"

            valid_times_t = np.atleast_1d(ds_t_sub.valid_time.values)
            valid_times_rh = np.atleast_1d(ds_rh_sub.valid_time.values)
            common_times = np.intersect1d(valid_times_t, valid_times_rh)

            fhr = 1
            for v_time in np.sort(common_times):
                if fhr > 48: break 
                
                t_idx = np.where(valid_times_t == v_time)[0][0]
                rh_idx = np.where(valid_times_rh == v_time)[0][0]
                
                t_step = ds_t_sub.isel(step=t_idx)
                rh_step = ds_rh_sub.isel(step=rh_idx)
                
                t_data = t_step['t2m'].values if 't2m' in t_step.data_vars else t_step['2t'].values
                rh_data = rh_step['r2'].values if 'r2' in rh_step.data_vars else rh_step['2r'].values
                
                recovery = generate_recovery_map_from_rh(t_data, rh_data, n_moe, n_mask)
                
                # Plot standard hour map
                generate_main_plot(recovery, n_lats, n_lons, v_time, fhr, ndfd_run_info, model="NDFD")
                
                # Trailing 3-Hour Average Logic
                ndfd_trailing.append(recovery)
                if len(ndfd_trailing) > 3:
                    ndfd_trailing.pop(0)
                
                avg_recovery = np.nanmean(ndfd_trailing, axis=0)
                generate_ntr_plot(avg_recovery, n_lats, n_lons, v_time, fhr, ndfd_run_info, model="NDFD")
                    
                fhr += 1
                
            ds_t.close()
            ds_rh.close()
        except Exception as e:
            print(f"Error processing NDFD: {e}")
        finally:
            if os.path.exists(temp_file): os.remove(temp_file)
            if os.path.exists(rh_file): os.remove(rh_file)

    if now.hour >= 13:
        if global_lats is not None:
            run_verification_logic(global_moe, global_mask, global_lats, global_lons, y_slice, x_slice)
    else:
        preserve_verification()

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    main()

# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import subprocess as sp
import xarray as xr
import os
import copernicusmarine
copernicusmarine.login()

def download(year_st, year_end, lon, lat, ofo):
    # Set parameters
    data_request = {
       "dataset_id" : "cmems_SST_MED_SST_L4_REP_OBSERVATIONS_010_021",
       "longitude" : lon, 
       "latitude" : lat,
       "time" : [f"{year_st}-01-01", f"{year_end}-12-31"],
       "variables" : ['analysed_sst', 'analysis_error']
     }
     	
    
    # Set start date for the download loop
    start = datetime.strptime(data_request["time"][0], "%Y-%m-%d")
    end = datetime.strptime(data_request["time"][1], "%Y-%m-%d")
	
    # Load xarray dataset
    while start <= end:
        filename = f"SST_L4_{start.year}_raw.nc"
        output = os.path.join(ofo, filename)
        processed_output = output.replace("_raw.nc", ".nc")

        # Check if the processed file already exists
        if os.path.exists(processed_output):
            print(f"File {processed_output} already exists. Skipping download...")
        else:
            print(f"Downloading data for: {start.year}")
            start_datetime = start.strftime("%Y-%m-%d")
            end_datetime = (start + relativedelta(years=1) - timedelta(days=1)).strftime("%Y-%m-%d")

            copernicusmarine.subset(
                dataset_id = data_request["dataset_id"],
                minimum_longitude = data_request["longitude"][0],
                maximum_longitude = data_request["longitude"][1],
                minimum_latitude = data_request["latitude"][0],
                maximum_latitude = data_request["latitude"][1],
                start_datetime = start_datetime,
                end_datetime = end_datetime,
                variables = data_request["variables"],
                output_filename = output,
                force_download = True
            )
            
            # Open and process the dataset
            with xr.open_dataset(output) as SST_L4:
                print(f"Dataset dimensions before renaming: {SST_L4.dims}")
                # Rename dimensions
                SST_L4 = SST_L4.rename({'latitude': 'lat', 'longitude': 'lon'})
                # Transpose dimensions
                SST_L4 = SST_L4.transpose('time', 'lat', 'lon')


                lat_0 = 42.843059
                lon_0 = -0.179951
                SST_L4 = SST_L4.where((SST_L4.lat < lat_0) | (SST_L4.lon > lon_0))
                # Save processed file

                SST_L4.to_netcdf(processed_output)
                print(f"Processed data saved to: {processed_output}")
                
        # Remove the raw downloaded file
        os.remove(output)
        # SST_L4 = xr.open_mfdataset(output)
        # print(SST_L4.dims)
        # #Rename the dimensions for easier future handling
        # SST_L4 = SST_L4.rename({
        # 'latitude': 'lat',
        # 'longitude': 'lon'
        # }) 
        # #Change dimension order
        # SST_L4 = SST_L4.transpose('time', 'lat', 'lon') 
        # os.remove(output)
        # SST_L4.to_netcdf(output)
    	# set the next start time for the next year
        start += relativedelta(years=1)

    

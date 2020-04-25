from ftplib import FTP
import os
import h5py as h5py
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import multiprocessing as mp

os.environ['pmmuser'] = 'efergo9@gmail.com'
os.environ['pmmpass'] = 'efergo9@gmail.com'

class GPM:

    def __init__(self,local_filename, bounds = None):
        self.local_filename = local_filename
        dataset = h5py.File(local_filename, 'r')
        precip = dataset['Grid/precipitationCal'][:]
        precip = np.transpose(precip)
        precip = precip.squeeze()
        self.precip = precip[::-1]
        self.theLats= dataset['Grid/lat'][:][::-1]
        self.theLons = dataset['Grid/lon'][:]
        if bounds is not None:
            self.set_crop(bounds)

    def get_grid(self):
        #masked_precip = np.ma.masked_where(precip < 0,precip)
        return self.precip

    def plot(self):
        masked_precip = np.ma.masked_where(self.precip < 0,self.precip)
        plt.imshow(masked_precip)

    def set_crop(self, bounds):
        west, south, east, north = bounds
        self.west = np.floor(west*10)/10
        self.east = np.ceil(east*10)/10
        self.south = np.floor(south*10)/10
        self.north = np.ceil(north*10)/10
        self.lonmin_ind = np.where(self.theLons > self.west)[0].min()
        self.lonmax_ind = np.where(self.theLons < self.east)[0].max()
        self.latmin_ind = np.where(self.theLats > self.south)[0].max()
        self.latmax_ind = np.where(self.theLats < self.north)[0].min()
        # print('lonmin: ' + str(self.theLons[self.lonmin_ind]))
        # print('lonmax: ' + str(self.theLons[self.lonmax_ind]))
        # print('latmin: ' + str(self.theLats[self.latmin_ind]))
        # print('latmax: ' + str(self.theLats[self.latmax_ind]))

    def get_bounds(self):
        return (self.west,self.south,self.east,self.north)

    def get_bounds_transform(self):
        return (self.west,self.south,self.east,self.north,self.lonmax_ind-self.lonmin_ind+1,self.latmin_ind-self.latmax_ind+1)

    def get_crop(self):
        return self.precip[self.latmax_ind:self.latmin_ind+1,self.lonmin_ind:self.lonmax_ind+1]

    def plot_crop(self):
        crop = self.get_crop()
        masked_crop = np.ma.masked_where(crop < 0,crop)
        plt.imshow(masked_crop)

    def coordinates(self, bounds = None):
        if bounds is not None:
            return (self.theLats[self.latmax_ind:self.latmin_ind+1],self.theLons[self.lonmin_ind:self.lonmax_ind+1])
        else:
            return (self.theLats,self.theLons)

    def save_cropped_tif(self):
        local_filename = self.local_filename.split('/')
        local_filename[1] = 'tif'
        directory = '/'.join(local_filename[0:3])
        local_filename = '/'.join(local_filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if not os.path.isfile(local_filename + '-masked.tif'):
            data = self.get_crop().astype('float32')
            transform = rasterio.transform.from_bounds(*self.get_bounds_transform())
            writer = rasterio.open( local_filename + '-masked.tif', 'w', driver='GTiff',
                                        height = data.shape[0], width = data.shape[1],
                                        count=1, dtype=str(data.dtype),
                                        crs='epsg:4326',
                                        transform=transform,
                                        nodata=-9999.9
                                )

            writer.write(data, 1)
            writer.close()


def pmm_ftp(date,filename):

    directory = 'data_pmm/raw/' + date.replace('/','-')
    local_filename = os.path.join(directory, filename)

    if not os.path.exists(directory):
        os.makedirs(directory)

    ftp = FTP('arthurhou.pps.eosdis.nasa.gov', user = os.environ['pmmuser'], passwd=os.environ['pmmpass'])
    ftp.cwd('gpmdata/'+ date +'/imerg')
    if True: #not os.path.isfile(local_filename):
        with open(local_filename, 'wb') as f:
            ftp.retrbinary('RETR ' + filename, f.write)
    ftp.quit()

def pmm_ftp_batch(date):
    directory = 'data_pmm/raw/' + date.replace('/','-')
    if not os.path.exists(directory):
        os.makedirs(directory)

        # add force mode
        # get credentials at https://pps.gsfc.nasa.gov/register.html
        ftp = FTP('arthurhou.pps.eosdis.nasa.gov', user = os.environ['pmmuser'], passwd=os.environ['pmmpass'])
        ftp.cwd('gpmdata/'+ date +'/imerg')
        filenames = ftp.nlst()
        filenames.sort()

        for filename in filenames:
            local_filename = os.path.join(directory, filename)
            if not os.path.isfile(local_filename):
                print('Downloading: ' + local_filename)
                with open(local_filename, 'wb') as f:
                    ftp.retrbinary('RETR ' + filename, f.write)
        ftp.quit()

def get_hdf_list(dates):
    if isinstance(dates,str):
        dates = [dates]
    local_filenames = []

    for date in dates:
        directory = 'data_pmm/raw/' + date.replace('/','-')
        files = os.listdir(directory)
        files = [i for i in files if '3B-HHR' in i]
        files = [i for i in files if '.tif' not in i]
        files.sort()
        local_filenames.extend([os.path.join(directory, file) for file in files ])

    return local_filenames

def get_hdf_list(dates):
    if isinstance(dates,str):
        dates = [dates]
    local_filenames = []

    for date in dates:
        directory = 'data_pmm/raw/' + date.replace('/','-')
        files = os.listdir(directory)
        files = [i for i in files if '3B-HHR' in i]
        files = [i for i in files if '.tif' not in i]
        files.sort()
        local_filenames.extend([os.path.join(directory, file) for file in files ])

    return local_filenames

def get_tif_list(dates):
    if isinstance(dates,str):
        dates = [dates]
    local_filenames = []

    for date in dates:
        directory = 'data_pmm/tif/' + date.replace('/','-')
        files = os.listdir(directory)
        files = [i for i in files if '3B-HHR' in i]
        files = [i for i in files if '-masked.tif' in i]
        files.sort()
        local_filenames.extend([os.path.join(directory, file) for file in files ])
    return local_filenames

def get_resampled_tif_list(dates):
    if isinstance(dates,str):
        dates = [dates]
    local_filenames = []

    for date in dates:
        directory = 'data_pmm/tif/' + date.replace('/','-')
        files = os.listdir(directory)
        files = [i for i in files if '3B-HHR' in i]
        files = [i for i in files if '-masked-resampled.tif' in i]
        files = [i for i in files if '.cpg' not in i]
        files = [i for i in files if '.dbf' not in i]
        files = [i for i in files if '.prj' not in i]
        files = [i for i in files if '.shp' not in i]
        files = [i for i in files if '.shx' not in i]
        files.sort()
        local_filenames.extend( [os.path.join(directory, file) for file in files ])
    return local_filenames

def resample(filename,source_to_match):
    from osgeo import gdal, gdalconst
    # Source
    src_filename = filename
    src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()

    # We want a section of source that matches this:
    match_filename = source_to_match
    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize

    # Output / destination
    dst_filename = filename[:-4] + '-resampled.tif'
    dst = gdal.GetDriverByName('GTiff').Create(dst_filename, wide, high, 1, gdalconst.GDT_Float32)
    dst.SetGeoTransform( match_geotrans )
    dst.SetProjection( match_proj)

    # Do the work
    gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_NearestNeighbour)

    del dst # Flush

def rasterstat(areas,filename):
    #print(mp.current_process())
    if not os.path.exists(filename[:-4] + '.csv'):
        import pandas as pd
        #for filename in tiflist:
        print('\tWorking on: \t' + filename)
        data = rasterio.open(filename).read(1)
        avg_set = {}
        for row in areas:
            pixels = row[1]
            avg_set[row[0]] = calc_avg(pixels,data)
        pd.DataFrame.from_dict(avg_set,orient='index')\
            .rename_axis('Reach_ID')\
            .rename(columns={0:'rain'})\
            .to_csv(filename[:-4]+'.csv')
    else:
        print('\t' + filename[:-4] + '.csv' + ' already exists')

def calc_avg(pixels,grid):
    nr_pixels = len(pixels)
    summ = 0
    for x,y in pixels:
        summ += max(grid[x,y],0)
    avg = summ/nr_pixels
    return avg

def download_hydrosheds_data(url):
    import urllib
    import zipfile
    directory = 'data_hydrosheds'
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = url.split('/')[-1]
    local_filename = os.path.join(directory,filename)
    new_directory = local_filename[0:-4]

    if not os.path.exists(new_directory):
        if not os.path.isfile(local_filename):
            urllib.request.urlretrieve(url,local_filename)
        zip_ref = zipfile.ZipFile(local_filename, 'r')
        zip_ref.extractall(new_directory)
        zip_ref.close()


def get_reach_coords():
    """Get reach coordinates in both latlon and rain grid indexes
    Parameters:
        - None
    Returns:
        - reaches: pandas.DataFrame with columns=['Reach_ID', 'Lat_Lon',
        'Avg_pixel', 'Grid_coords']. 'Lat_Lon' is a (lat,lon) tuple with the
        latitude and longitude of the geometric center of the reach; 'Avg_pixel'
        is a (h, w) tuple where h and w are the coordinatesof the geometric
        center of the reach in TIF image pixels; 'Grid_coords' is a (i,j) tuple
        where i and j are the rain grid coordinates of the geometric center of the reach """
    # We have pixels of every reach. Now we are going to get the center of mass pixel
    # for every reach, and translate this to latitude and longitude
    import pandas as pd
    import globals

    bounds = globals.bounds(2)

    areas = pd.read_pickle('data_gloric/areas_gloric_pixel.pkl')['pixels'].to_numpy()
    avg_pixels = []
    for row in areas:
        nr_pixels = len(row)
        avg_height = 0
        avg_width = 0
        for pixel in row:
            avg_height += pixel[0]
            avg_width += pixel[1]
        avg_pixels.append((avg_height/nr_pixels, avg_width/nr_pixels))
    reaches = pd.read_pickle('data_gloric/areas_gloric_pixel.pkl')[['Reach_ID','pixels']]
    reaches['Avg_pixel'] = avg_pixels
    # pixels_height and pixels_width in pixels of tif image (use any of the masked tifs)
    pixels_height, pixels_width = rasterio.open('data_pmm/tif/2013-06-16/3B-HHR.MS.MRG.3IMERG'
        '.20130616-S000000-E002959.0000.V06B.HDF5-masked-resampled.tif').read(1).shape

    def pixel_to_coords(ij):
        """Convert pixel to latitude and longitude"""
        i, j = ij[0], ij[1]
        return (bounds[1]+i*(bounds[3]-bounds[1])/pixels_width,
                bounds[0]+j*(bounds[2]-bounds[0])/pixels_height)

    rain_grid = GPM('data_pmm/raw/2013-06-16/3B-HHR.MS.MRG.3IMERG.'
        '20130616-S000000-E002959.0000.V06B.HDF5', bounds).get_crop()

    def pixels_to_rain_grid_index(ij):
        """Converts avalanche pixels form TIF image into rain_grid index.
        Parameters:
            - avalanche_pixels: (h, w) tuple where h is the height coordinate
            and w the width coordinate for the avalanche geometric center in pixels
            of the TIF image."""
        i, j = ij
        # (rg_height,rg_width) is the shape of the rain_grid
        rg_height, rg_width = len(rain_grid), len(rain_grid[0])
        avalanche_i = round(i*(rg_height/pixels_height))
        avalanche_j = round(j*(rg_width/pixels_width))
        return avalanche_i, avalanche_j

    reaches['Lat_Lon'] = [pixel_to_coords(ij=ij) for ij in reaches.loc[:, 'Avg_pixel']]
    reaches['Grid_coords'] = [pixels_to_rain_grid_index(ij=ij) for ij in reaches.loc[:, 'Avg_pixel']]
    reaches = reaches.drop(columns=[ 'Reach_ID', 'pixels'] )

    return reaches


def get_histogram_bins_centers(bins):
    """
    Parameters: bins' limits
    Returns:    bins' centers
    """
    bins_centers = []
    for i in range(len(bins)-1):
        bin_min = bins[i]
        bin_max = bins[i+1]
        bins_centers.append(( bin_max+bin_min ) / 2)
    return bins_centers

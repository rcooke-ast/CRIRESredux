from reduce_base import ReduceBase
import numpy as np


def main():
    # Initialise the reduce class
    prep = False
    cals = False
    makediff = False
    step = 1  # 0: basis, 1: wavecal prelim + prepALIS, 2: combspec rebin
    thisred = Reduce(prefix="wray15199", match_name="WRAY 15-199", data_folder="Raw/",
                     use_diff=True,
                     step_listfiles=prep, step_make_combinations=prep,
                     step_pattern=False,  # Generate an image of the detector pattern
                     step_makedarkfit=False, step_makedarkframe=cals,  # Make a dark image
                     step_makeflat=cals,  # Make a flatfield image
                     step_makearc=cals,  # Make an arc image
                     step_makediff=makediff, step_subbg=False,  # Make difference and sum images
                     step_makecuts=False,  # Make difference and sum images
                     step_trace=False, step_extract=False, step_basis=(step==0),#step1,
                     ext_sky=False,  # Trace the spectrum and extract
                     step_wavecal_prelim=(step==1),  # Calculate a preliminary wavelength calibration solution
                     step_prepALIS=(step==1),
                     # Once the data are reduced, prepare a series of files to be used to fit the wavelength solution with ALIS
                     step_combspec=False, step_combspec_rebin=(step==2),
                     # First get the corrected data from ALIS, and then combine all exposures with this step.
                     step_wavecal_sky=False, step_comb_sky=False,
                     # Wavelength calibrate all sky spectra and then combine
                     step_sample_NumExpCombine=False)  # Combine a different number of exposures to estimate how S/N depends on the number of exposures combined.
    thisred.makePaths(redux_path="/Users/rcooke/Work/Research/BBN/helium34/Absorption/2023_CRIRES_Survey/WRAY_15-199/2024-04-07/")
    thisred._plotit = False
    thisred._comb_set = -1
    thisred.run()


class Reduce(ReduceBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Change some of the default parameters
        self._nbasis = 5  # Number of basis functions to use for the continuum
        self._numcomp = 1
        self._scalevariance = [10830.0, 10832.5]  # Scale the variance to match the measured variance in these regions
        self._scale_errors = True  # Scale the errors by 10x in regions with low flux. This is only used for fitting the wavelength solution with ALIS. The errors are scaled back to their extraction values during the combination.
        self._use_dark = True

    def get_science_frames(self):
        """
        This was auto-generated with step_make_combinations()
        """
#                A=3.0  (spec=0  DIT=240.0  NDIT=1)
        return  [[["CRIRE.2024-04-08T00:50:32.835.fits"], ["CRIRE.2024-04-08T01:34:25.122.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:41:12.657.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:10:12.944.fits",    # B=4.5
                                                           "CRIRE.2024-04-08T01:03:44.699.fits",    # B=4.5
                                                           "CRIRE.2024-04-08T02:32:25.863.fits",    # B=4.0
                                                           "CRIRE.2024-04-08T01:25:41.555.fits",    # B=4.0
                                                           "CRIRE.2024-04-08T01:17:02.277.fits",    # B=3.5
                                                           "CRIRE.2024-04-08T02:23:44.257.fits",    # B=3.5
                                                           "CRIRE.2024-04-08T00:54:50.246.fits",    # B=3.0
                                                           "CRIRE.2024-04-08T02:00:42.504.fits",    # B=3.0
                                                           "CRIRE.2024-04-08T01:43:05.014.fits",    # B=1.5
                                                           "CRIRE.2024-04-08T02:49:52.153.fits"]],  # B=1.5
#                B=3.0  (spec=1  DIT=240.0  NDIT=1)
                 [["CRIRE.2024-04-08T00:54:50.246.fits"], ["CRIRE.2024-04-08T01:38:52.247.fits",    # A=1.5
                                                           "CRIRE.2024-04-08T02:45:39.461.fits",    # A=1.5
                                                           "CRIRE.2024-04-08T01:56:12.578.fits",    # A=3.0
                                                           "CRIRE.2024-04-08T00:50:32.835.fits",    # A=3.0
                                                           "CRIRE.2024-04-08T02:19:25.102.fits",    # A=3.5
                                                           "CRIRE.2024-04-08T01:12:45.661.fits",    # A=3.5
                                                           "CRIRE.2024-04-08T01:21:23.261.fits",    # A=4.0
                                                           "CRIRE.2024-04-08T02:28:06.949.fits",    # A=4.0
                                                           "CRIRE.2024-04-08T00:59:26.640.fits",    # A=4.5
                                                           "CRIRE.2024-04-08T02:05:43.118.fits",    # A=4.5
                                                           "CRIRE.2024-04-08T01:30:07.765.fits",    # A=6.0
                                                           "CRIRE.2024-04-08T02:36:53.553.fits"]],  # A=6.0
#                A=4.5  (spec=2  DIT=240.0  NDIT=1)
                 [["CRIRE.2024-04-08T00:59:26.640.fits"], ["CRIRE.2024-04-08T01:34:25.122.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:41:12.657.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:10:12.944.fits",    # B=4.5
                                                           "CRIRE.2024-04-08T01:03:44.699.fits",    # B=4.5
                                                           "CRIRE.2024-04-08T02:32:25.863.fits",    # B=4.0
                                                           "CRIRE.2024-04-08T01:25:41.555.fits",    # B=4.0
                                                           "CRIRE.2024-04-08T01:17:02.277.fits",    # B=3.5
                                                           "CRIRE.2024-04-08T02:23:44.257.fits",    # B=3.5
                                                           "CRIRE.2024-04-08T00:54:50.246.fits",    # B=3.0
                                                           "CRIRE.2024-04-08T02:00:42.504.fits",    # B=3.0
                                                           "CRIRE.2024-04-08T01:43:05.014.fits",    # B=1.5
                                                           "CRIRE.2024-04-08T02:49:52.153.fits"]],  # B=1.5
#                B=4.5  (spec=3  DIT=240.0  NDIT=1)
                 [["CRIRE.2024-04-08T01:03:44.699.fits"], ["CRIRE.2024-04-08T01:38:52.247.fits",    # A=1.5
                                                           "CRIRE.2024-04-08T02:45:39.461.fits",    # A=1.5
                                                           "CRIRE.2024-04-08T01:56:12.578.fits",    # A=3.0
                                                           "CRIRE.2024-04-08T00:50:32.835.fits",    # A=3.0
                                                           "CRIRE.2024-04-08T02:19:25.102.fits",    # A=3.5
                                                           "CRIRE.2024-04-08T01:12:45.661.fits",    # A=3.5
                                                           "CRIRE.2024-04-08T01:21:23.261.fits",    # A=4.0
                                                           "CRIRE.2024-04-08T02:28:06.949.fits",    # A=4.0
                                                           "CRIRE.2024-04-08T00:59:26.640.fits",    # A=4.5
                                                           "CRIRE.2024-04-08T02:05:43.118.fits",    # A=4.5
                                                           "CRIRE.2024-04-08T01:30:07.765.fits",    # A=6.0
                                                           "CRIRE.2024-04-08T02:36:53.553.fits"]],  # A=6.0
#                A=3.5  (spec=4  DIT=240.0  NDIT=1)
                 [["CRIRE.2024-04-08T01:12:45.661.fits"], ["CRIRE.2024-04-08T01:34:25.122.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:41:12.657.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:10:12.944.fits",    # B=4.5
                                                           "CRIRE.2024-04-08T01:03:44.699.fits",    # B=4.5
                                                           "CRIRE.2024-04-08T02:32:25.863.fits",    # B=4.0
                                                           "CRIRE.2024-04-08T01:25:41.555.fits",    # B=4.0
                                                           "CRIRE.2024-04-08T01:17:02.277.fits",    # B=3.5
                                                           "CRIRE.2024-04-08T02:23:44.257.fits",    # B=3.5
                                                           "CRIRE.2024-04-08T00:54:50.246.fits",    # B=3.0
                                                           "CRIRE.2024-04-08T02:00:42.504.fits",    # B=3.0
                                                           "CRIRE.2024-04-08T01:43:05.014.fits",    # B=1.5
                                                           "CRIRE.2024-04-08T02:49:52.153.fits"]],  # B=1.5
#                B=3.5  (spec=5  DIT=240.0  NDIT=1)
                 [["CRIRE.2024-04-08T01:17:02.277.fits"], ["CRIRE.2024-04-08T01:38:52.247.fits",    # A=1.5
                                                           "CRIRE.2024-04-08T02:45:39.461.fits",    # A=1.5
                                                           "CRIRE.2024-04-08T01:56:12.578.fits",    # A=3.0
                                                           "CRIRE.2024-04-08T00:50:32.835.fits",    # A=3.0
                                                           "CRIRE.2024-04-08T02:19:25.102.fits",    # A=3.5
                                                           "CRIRE.2024-04-08T01:12:45.661.fits",    # A=3.5
                                                           "CRIRE.2024-04-08T01:21:23.261.fits",    # A=4.0
                                                           "CRIRE.2024-04-08T02:28:06.949.fits",    # A=4.0
                                                           "CRIRE.2024-04-08T00:59:26.640.fits",    # A=4.5
                                                           "CRIRE.2024-04-08T02:05:43.118.fits",    # A=4.5
                                                           "CRIRE.2024-04-08T01:30:07.765.fits",    # A=6.0
                                                           "CRIRE.2024-04-08T02:36:53.553.fits"]],  # A=6.0
#                A=4.0  (spec=6  DIT=240.0  NDIT=1)
                 [["CRIRE.2024-04-08T01:21:23.261.fits"], ["CRIRE.2024-04-08T01:34:25.122.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:41:12.657.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:10:12.944.fits",    # B=4.5
                                                           "CRIRE.2024-04-08T01:03:44.699.fits",    # B=4.5
                                                           "CRIRE.2024-04-08T02:32:25.863.fits",    # B=4.0
                                                           "CRIRE.2024-04-08T01:25:41.555.fits",    # B=4.0
                                                           "CRIRE.2024-04-08T01:17:02.277.fits",    # B=3.5
                                                           "CRIRE.2024-04-08T02:23:44.257.fits",    # B=3.5
                                                           "CRIRE.2024-04-08T00:54:50.246.fits",    # B=3.0
                                                           "CRIRE.2024-04-08T02:00:42.504.fits",    # B=3.0
                                                           "CRIRE.2024-04-08T01:43:05.014.fits",    # B=1.5
                                                           "CRIRE.2024-04-08T02:49:52.153.fits"]],  # B=1.5
#                B=4.0  (spec=7  DIT=240.0  NDIT=1)
                 [["CRIRE.2024-04-08T01:25:41.555.fits"], ["CRIRE.2024-04-08T01:38:52.247.fits",    # A=1.5
                                                           "CRIRE.2024-04-08T02:45:39.461.fits",    # A=1.5
                                                           "CRIRE.2024-04-08T01:56:12.578.fits",    # A=3.0
                                                           "CRIRE.2024-04-08T00:50:32.835.fits",    # A=3.0
                                                           "CRIRE.2024-04-08T02:19:25.102.fits",    # A=3.5
                                                           "CRIRE.2024-04-08T01:12:45.661.fits",    # A=3.5
                                                           "CRIRE.2024-04-08T01:21:23.261.fits",    # A=4.0
                                                           "CRIRE.2024-04-08T02:28:06.949.fits",    # A=4.0
                                                           "CRIRE.2024-04-08T00:59:26.640.fits",    # A=4.5
                                                           "CRIRE.2024-04-08T02:05:43.118.fits",    # A=4.5
                                                           "CRIRE.2024-04-08T01:30:07.765.fits",    # A=6.0
                                                           "CRIRE.2024-04-08T02:36:53.553.fits"]],  # A=6.0
#                A=6.0  (spec=8  DIT=240.0  NDIT=1)
                 [["CRIRE.2024-04-08T01:30:07.765.fits"], ["CRIRE.2024-04-08T01:34:25.122.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:41:12.657.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:10:12.944.fits",    # B=4.5
                                                           "CRIRE.2024-04-08T01:03:44.699.fits",    # B=4.5
                                                           "CRIRE.2024-04-08T01:25:41.555.fits",    # B=4.0
                                                           "CRIRE.2024-04-08T02:32:25.863.fits",    # B=4.0
                                                           "CRIRE.2024-04-08T02:23:44.257.fits",    # B=3.5
                                                           "CRIRE.2024-04-08T01:17:02.277.fits",    # B=3.5
                                                           "CRIRE.2024-04-08T02:00:42.504.fits",    # B=3.0
                                                           "CRIRE.2024-04-08T00:54:50.246.fits",    # B=3.0
                                                           "CRIRE.2024-04-08T02:49:52.153.fits",    # B=1.5
                                                           "CRIRE.2024-04-08T01:43:05.014.fits",    # B=1.5
                                                           "CRIRE.2024-04-08T01:38:52.247.fits",    # A=1.5
                                                           "CRIRE.2024-04-08T02:45:39.461.fits"]],  # A=1.5
#                B=6.0  (spec=9  DIT=240.0  NDIT=1)
                 [["CRIRE.2024-04-08T01:34:25.122.fits"], ["CRIRE.2024-04-08T01:43:05.014.fits",    # B=1.5
                                                           "CRIRE.2024-04-08T02:49:52.153.fits",    # B=1.5
                                                           "CRIRE.2024-04-08T02:45:39.461.fits",    # A=1.5
                                                           "CRIRE.2024-04-08T01:38:52.247.fits",    # A=1.5
                                                           "CRIRE.2024-04-08T00:50:32.835.fits",    # A=3.0
                                                           "CRIRE.2024-04-08T01:56:12.578.fits",    # A=3.0
                                                           "CRIRE.2024-04-08T01:12:45.661.fits",    # A=3.5
                                                           "CRIRE.2024-04-08T02:19:25.102.fits",    # A=3.5
                                                           "CRIRE.2024-04-08T01:21:23.261.fits",    # A=4.0
                                                           "CRIRE.2024-04-08T02:28:06.949.fits",    # A=4.0
                                                           "CRIRE.2024-04-08T02:05:43.118.fits",    # A=4.5
                                                           "CRIRE.2024-04-08T00:59:26.640.fits",    # A=4.5
                                                           "CRIRE.2024-04-08T01:30:07.765.fits",    # A=6.0
                                                           "CRIRE.2024-04-08T02:36:53.553.fits"]],  # A=6.0
#                A=1.5  (spec=10  DIT=240.0  NDIT=1)
                 [["CRIRE.2024-04-08T01:38:52.247.fits"], ["CRIRE.2024-04-08T01:34:25.122.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:41:12.657.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:10:12.944.fits",    # B=4.5
                                                           "CRIRE.2024-04-08T01:03:44.699.fits",    # B=4.5
                                                           "CRIRE.2024-04-08T02:32:25.863.fits",    # B=4.0
                                                           "CRIRE.2024-04-08T01:25:41.555.fits",    # B=4.0
                                                           "CRIRE.2024-04-08T01:17:02.277.fits",    # B=3.5
                                                           "CRIRE.2024-04-08T02:23:44.257.fits",    # B=3.5
                                                           "CRIRE.2024-04-08T00:54:50.246.fits",    # B=3.0
                                                           "CRIRE.2024-04-08T02:00:42.504.fits",    # B=3.0
                                                           "CRIRE.2024-04-08T01:30:07.765.fits",    # A=6.0
                                                           "CRIRE.2024-04-08T02:36:53.553.fits"]],  # A=6.0
#                B=1.5  (spec=11  DIT=240.0  NDIT=1)
                 [["CRIRE.2024-04-08T01:43:05.014.fits"], ["CRIRE.2024-04-08T01:34:25.122.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:41:12.657.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T01:56:12.578.fits",    # A=3.0
                                                           "CRIRE.2024-04-08T00:50:32.835.fits",    # A=3.0
                                                           "CRIRE.2024-04-08T02:19:25.102.fits",    # A=3.5
                                                           "CRIRE.2024-04-08T01:12:45.661.fits",    # A=3.5
                                                           "CRIRE.2024-04-08T01:21:23.261.fits",    # A=4.0
                                                           "CRIRE.2024-04-08T02:28:06.949.fits",    # A=4.0
                                                           "CRIRE.2024-04-08T00:59:26.640.fits",    # A=4.5
                                                           "CRIRE.2024-04-08T02:05:43.118.fits",    # A=4.5
                                                           "CRIRE.2024-04-08T01:30:07.765.fits",    # A=6.0
                                                           "CRIRE.2024-04-08T02:36:53.553.fits"]],  # A=6.0
#                A=3.0  (spec=12  DIT=240.0  NDIT=1)
                 [["CRIRE.2024-04-08T01:56:12.578.fits"], ["CRIRE.2024-04-08T01:34:25.122.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:41:12.657.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:10:12.944.fits",    # B=4.5
                                                           "CRIRE.2024-04-08T01:03:44.699.fits",    # B=4.5
                                                           "CRIRE.2024-04-08T02:32:25.863.fits",    # B=4.0
                                                           "CRIRE.2024-04-08T01:25:41.555.fits",    # B=4.0
                                                           "CRIRE.2024-04-08T01:17:02.277.fits",    # B=3.5
                                                           "CRIRE.2024-04-08T02:23:44.257.fits",    # B=3.5
                                                           "CRIRE.2024-04-08T00:54:50.246.fits",    # B=3.0
                                                           "CRIRE.2024-04-08T02:00:42.504.fits",    # B=3.0
                                                           "CRIRE.2024-04-08T01:43:05.014.fits",    # B=1.5
                                                           "CRIRE.2024-04-08T02:49:52.153.fits"]],  # B=1.5
#                B=3.0  (spec=13  DIT=240.0  NDIT=1)
                 [["CRIRE.2024-04-08T02:00:42.504.fits"], ["CRIRE.2024-04-08T01:38:52.247.fits",    # A=1.5
                                                           "CRIRE.2024-04-08T02:45:39.461.fits",    # A=1.5
                                                           "CRIRE.2024-04-08T01:56:12.578.fits",    # A=3.0
                                                           "CRIRE.2024-04-08T00:50:32.835.fits",    # A=3.0
                                                           "CRIRE.2024-04-08T02:19:25.102.fits",    # A=3.5
                                                           "CRIRE.2024-04-08T01:12:45.661.fits",    # A=3.5
                                                           "CRIRE.2024-04-08T01:21:23.261.fits",    # A=4.0
                                                           "CRIRE.2024-04-08T02:28:06.949.fits",    # A=4.0
                                                           "CRIRE.2024-04-08T00:59:26.640.fits",    # A=4.5
                                                           "CRIRE.2024-04-08T02:05:43.118.fits",    # A=4.5
                                                           "CRIRE.2024-04-08T01:30:07.765.fits",    # A=6.0
                                                           "CRIRE.2024-04-08T02:36:53.553.fits"]],  # A=6.0
#                A=4.5  (spec=14  DIT=240.0  NDIT=1)
                 [["CRIRE.2024-04-08T02:05:43.118.fits"], ["CRIRE.2024-04-08T01:34:25.122.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:41:12.657.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:10:12.944.fits",    # B=4.5
                                                           "CRIRE.2024-04-08T01:03:44.699.fits",    # B=4.5
                                                           "CRIRE.2024-04-08T02:32:25.863.fits",    # B=4.0
                                                           "CRIRE.2024-04-08T01:25:41.555.fits",    # B=4.0
                                                           "CRIRE.2024-04-08T01:17:02.277.fits",    # B=3.5
                                                           "CRIRE.2024-04-08T02:23:44.257.fits",    # B=3.5
                                                           "CRIRE.2024-04-08T00:54:50.246.fits",    # B=3.0
                                                           "CRIRE.2024-04-08T02:00:42.504.fits",    # B=3.0
                                                           "CRIRE.2024-04-08T01:43:05.014.fits",    # B=1.5
                                                           "CRIRE.2024-04-08T02:49:52.153.fits"]],  # B=1.5
#                B=4.5  (spec=15  DIT=240.0  NDIT=1)
                 [["CRIRE.2024-04-08T02:10:12.944.fits"], ["CRIRE.2024-04-08T01:38:52.247.fits",    # A=1.5
                                                           "CRIRE.2024-04-08T02:45:39.461.fits",    # A=1.5
                                                           "CRIRE.2024-04-08T01:56:12.578.fits",    # A=3.0
                                                           "CRIRE.2024-04-08T00:50:32.835.fits",    # A=3.0
                                                           "CRIRE.2024-04-08T02:19:25.102.fits",    # A=3.5
                                                           "CRIRE.2024-04-08T01:12:45.661.fits",    # A=3.5
                                                           "CRIRE.2024-04-08T01:21:23.261.fits",    # A=4.0
                                                           "CRIRE.2024-04-08T02:28:06.949.fits",    # A=4.0
                                                           "CRIRE.2024-04-08T00:59:26.640.fits",    # A=4.5
                                                           "CRIRE.2024-04-08T02:05:43.118.fits",    # A=4.5
                                                           "CRIRE.2024-04-08T01:30:07.765.fits",    # A=6.0
                                                           "CRIRE.2024-04-08T02:36:53.553.fits"]],  # A=6.0
#                A=3.5  (spec=16  DIT=240.0  NDIT=1)
                 [["CRIRE.2024-04-08T02:19:25.102.fits"], ["CRIRE.2024-04-08T01:34:25.122.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:41:12.657.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:10:12.944.fits",    # B=4.5
                                                           "CRIRE.2024-04-08T01:03:44.699.fits",    # B=4.5
                                                           "CRIRE.2024-04-08T02:32:25.863.fits",    # B=4.0
                                                           "CRIRE.2024-04-08T01:25:41.555.fits",    # B=4.0
                                                           "CRIRE.2024-04-08T01:17:02.277.fits",    # B=3.5
                                                           "CRIRE.2024-04-08T02:23:44.257.fits",    # B=3.5
                                                           "CRIRE.2024-04-08T00:54:50.246.fits",    # B=3.0
                                                           "CRIRE.2024-04-08T02:00:42.504.fits",    # B=3.0
                                                           "CRIRE.2024-04-08T01:43:05.014.fits",    # B=1.5
                                                           "CRIRE.2024-04-08T02:49:52.153.fits"]],  # B=1.5
#                B=3.5  (spec=17  DIT=240.0  NDIT=1)
                 [["CRIRE.2024-04-08T02:23:44.257.fits"], ["CRIRE.2024-04-08T01:38:52.247.fits",    # A=1.5
                                                           "CRIRE.2024-04-08T02:45:39.461.fits",    # A=1.5
                                                           "CRIRE.2024-04-08T01:56:12.578.fits",    # A=3.0
                                                           "CRIRE.2024-04-08T00:50:32.835.fits",    # A=3.0
                                                           "CRIRE.2024-04-08T02:19:25.102.fits",    # A=3.5
                                                           "CRIRE.2024-04-08T01:12:45.661.fits",    # A=3.5
                                                           "CRIRE.2024-04-08T01:21:23.261.fits",    # A=4.0
                                                           "CRIRE.2024-04-08T02:28:06.949.fits",    # A=4.0
                                                           "CRIRE.2024-04-08T00:59:26.640.fits",    # A=4.5
                                                           "CRIRE.2024-04-08T02:05:43.118.fits",    # A=4.5
                                                           "CRIRE.2024-04-08T01:30:07.765.fits",    # A=6.0
                                                           "CRIRE.2024-04-08T02:36:53.553.fits"]],  # A=6.0
#                A=4.0  (spec=18  DIT=240.0  NDIT=1)
                 [["CRIRE.2024-04-08T02:28:06.949.fits"], ["CRIRE.2024-04-08T01:34:25.122.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:41:12.657.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:10:12.944.fits",    # B=4.5
                                                           "CRIRE.2024-04-08T01:03:44.699.fits",    # B=4.5
                                                           "CRIRE.2024-04-08T02:32:25.863.fits",    # B=4.0
                                                           "CRIRE.2024-04-08T01:25:41.555.fits",    # B=4.0
                                                           "CRIRE.2024-04-08T01:17:02.277.fits",    # B=3.5
                                                           "CRIRE.2024-04-08T02:23:44.257.fits",    # B=3.5
                                                           "CRIRE.2024-04-08T00:54:50.246.fits",    # B=3.0
                                                           "CRIRE.2024-04-08T02:00:42.504.fits",    # B=3.0
                                                           "CRIRE.2024-04-08T01:43:05.014.fits",    # B=1.5
                                                           "CRIRE.2024-04-08T02:49:52.153.fits"]],  # B=1.5
#                B=4.0  (spec=19  DIT=240.0  NDIT=1)
                 [["CRIRE.2024-04-08T02:32:25.863.fits"], ["CRIRE.2024-04-08T01:38:52.247.fits",    # A=1.5
                                                           "CRIRE.2024-04-08T02:45:39.461.fits",    # A=1.5
                                                           "CRIRE.2024-04-08T01:56:12.578.fits",    # A=3.0
                                                           "CRIRE.2024-04-08T00:50:32.835.fits",    # A=3.0
                                                           "CRIRE.2024-04-08T02:19:25.102.fits",    # A=3.5
                                                           "CRIRE.2024-04-08T01:12:45.661.fits",    # A=3.5
                                                           "CRIRE.2024-04-08T01:21:23.261.fits",    # A=4.0
                                                           "CRIRE.2024-04-08T02:28:06.949.fits",    # A=4.0
                                                           "CRIRE.2024-04-08T00:59:26.640.fits",    # A=4.5
                                                           "CRIRE.2024-04-08T02:05:43.118.fits",    # A=4.5
                                                           "CRIRE.2024-04-08T01:30:07.765.fits",    # A=6.0
                                                           "CRIRE.2024-04-08T02:36:53.553.fits"]],  # A=6.0
#                A=6.0  (spec=20  DIT=240.0  NDIT=1)
                 [["CRIRE.2024-04-08T02:36:53.553.fits"], ["CRIRE.2024-04-08T01:34:25.122.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:41:12.657.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:10:12.944.fits",    # B=4.5
                                                           "CRIRE.2024-04-08T01:03:44.699.fits",    # B=4.5
                                                           "CRIRE.2024-04-08T01:25:41.555.fits",    # B=4.0
                                                           "CRIRE.2024-04-08T02:32:25.863.fits",    # B=4.0
                                                           "CRIRE.2024-04-08T02:23:44.257.fits",    # B=3.5
                                                           "CRIRE.2024-04-08T01:17:02.277.fits",    # B=3.5
                                                           "CRIRE.2024-04-08T02:00:42.504.fits",    # B=3.0
                                                           "CRIRE.2024-04-08T00:54:50.246.fits",    # B=3.0
                                                           "CRIRE.2024-04-08T02:49:52.153.fits",    # B=1.5
                                                           "CRIRE.2024-04-08T01:43:05.014.fits",    # B=1.5
                                                           "CRIRE.2024-04-08T01:38:52.247.fits",    # A=1.5
                                                           "CRIRE.2024-04-08T02:45:39.461.fits"]],  # A=1.5
#                B=6.0  (spec=21  DIT=240.0  NDIT=1)
                 [["CRIRE.2024-04-08T02:41:12.657.fits"], ["CRIRE.2024-04-08T01:43:05.014.fits",    # B=1.5
                                                           "CRIRE.2024-04-08T02:49:52.153.fits",    # B=1.5
                                                           "CRIRE.2024-04-08T02:45:39.461.fits",    # A=1.5
                                                           "CRIRE.2024-04-08T01:38:52.247.fits",    # A=1.5
                                                           "CRIRE.2024-04-08T00:50:32.835.fits",    # A=3.0
                                                           "CRIRE.2024-04-08T01:56:12.578.fits",    # A=3.0
                                                           "CRIRE.2024-04-08T01:12:45.661.fits",    # A=3.5
                                                           "CRIRE.2024-04-08T02:19:25.102.fits",    # A=3.5
                                                           "CRIRE.2024-04-08T01:21:23.261.fits",    # A=4.0
                                                           "CRIRE.2024-04-08T02:28:06.949.fits",    # A=4.0
                                                           "CRIRE.2024-04-08T02:05:43.118.fits",    # A=4.5
                                                           "CRIRE.2024-04-08T00:59:26.640.fits",    # A=4.5
                                                           "CRIRE.2024-04-08T01:30:07.765.fits",    # A=6.0
                                                           "CRIRE.2024-04-08T02:36:53.553.fits"]],  # A=6.0
#                A=1.5  (spec=22  DIT=240.0  NDIT=1)
                 [["CRIRE.2024-04-08T02:45:39.461.fits"], ["CRIRE.2024-04-08T01:34:25.122.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:41:12.657.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:10:12.944.fits",    # B=4.5
                                                           "CRIRE.2024-04-08T01:03:44.699.fits",    # B=4.5
                                                           "CRIRE.2024-04-08T02:32:25.863.fits",    # B=4.0
                                                           "CRIRE.2024-04-08T01:25:41.555.fits",    # B=4.0
                                                           "CRIRE.2024-04-08T01:17:02.277.fits",    # B=3.5
                                                           "CRIRE.2024-04-08T02:23:44.257.fits",    # B=3.5
                                                           "CRIRE.2024-04-08T00:54:50.246.fits",    # B=3.0
                                                           "CRIRE.2024-04-08T02:00:42.504.fits",    # B=3.0
                                                           "CRIRE.2024-04-08T01:30:07.765.fits",    # A=6.0
                                                           "CRIRE.2024-04-08T02:36:53.553.fits"]],  # A=6.0
#                B=1.5  (spec=23  DIT=240.0  NDIT=1)
                 [["CRIRE.2024-04-08T02:49:52.153.fits"], ["CRIRE.2024-04-08T01:34:25.122.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T02:41:12.657.fits",    # B=6.0
                                                           "CRIRE.2024-04-08T01:56:12.578.fits",    # A=3.0
                                                           "CRIRE.2024-04-08T00:50:32.835.fits",    # A=3.0
                                                           "CRIRE.2024-04-08T02:19:25.102.fits",    # A=3.5
                                                           "CRIRE.2024-04-08T01:12:45.661.fits",    # A=3.5
                                                           "CRIRE.2024-04-08T01:21:23.261.fits",    # A=4.0
                                                           "CRIRE.2024-04-08T02:28:06.949.fits",    # A=4.0
                                                           "CRIRE.2024-04-08T00:59:26.640.fits",    # A=4.5
                                                           "CRIRE.2024-04-08T02:05:43.118.fits",    # A=4.5
                                                           "CRIRE.2024-04-08T01:30:07.765.fits",    # A=6.0
                                                           "CRIRE.2024-04-08T02:36:53.553.fits"]]]  # A=6.0

    # def is_frame_in_set(self, frnum, comb_set):
    #     if comb_set < 0:
    #         return True
    #     frame_in_set = False
    #     if comb_set == 0:
    #         if frnum in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
    #             frame_in_set = True
    #     elif comb_set == 1:
    #         if frnum in [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
    #             frame_in_set = True
    #     return frame_in_set

    # def is_frame_masked(self, frnum):
    #     masked = False
    #     if frnum in [25, 27]:
    #         masked = True
    #     return masked

    def get_scale(self, idx):
        scale = 1
        if idx in []:
            exptime = self.get_exptime(idx)[1]
            scale = self.get_exptime(0)[1] / exptime
        print("Scale = ", scale)
        return scale

    def get_flat_frames(self):
        return ["CRIRE.2024-04-08T12:15:21.884.fits",
                "CRIRE.2024-04-08T12:15:42.253.fits",
                "CRIRE.2024-04-08T12:16:02.621.fits",
                "CRIRE.2024-04-08T12:16:22.997.fits",
                "CRIRE.2024-04-08T12:16:43.368.fits"]

    def get_dark_frames(self):
        """
        """
        # Group dark files with different exposure times
        return [["CRIRE.2024-04-08T12:36:33.362.fits","CRIRE.2024-04-08T12:36:09.987.fits","CRIRE.2024-04-08T12:35:46.629.fits"],#5s
                ["CRIRE.2024-04-08T12:34:56.109.fits", "CRIRE.2024-04-08T12:34:05.606.fits", "CRIRE.2024-04-08T12:33:15.073.fits"],#45s
                ["CRIRE.2024-04-08T12:31:09.530.fits","CRIRE.2024-04-08T12:29:04.000.fits","CRIRE.2024-04-08T12:26:58.466.fits"]]#120s

    def get_arc_frames(self):
        return ["CRIRE.2024-04-08T12:20:58.243.fits"]

    def get_exptime(self, idx):
        ndit = self.get_ndit(idx)
        if idx in []:
            etim = 300  # This is the DIT
        else:
            etim = 240  # This is the DIT
        exptime = etim * ndit
        return exptime, etim

    def get_ndit(self, idx):
        if idx in []:
            return 9  # This is the NDIT
        else:
            return 1  # This is the NDIT

    def get_objprof_limits(self, full=True):
        """
        Set the spectral regions to calculate the object profile. If full=True, then a more extended region is used.
        These values are relevant for tet01 Ori A, during the 2022 observations

        To determine these values, open up the two frames with the biggest difference in nod (e.g. +/- 6.5") in ds9,
        and hover the cursor over the middle of the strongest He I* absorption line. The inner left limit is the pixel
        number at the middle of the profile minus 90 pixels, and the right limit is the pixel number at the middle of
        the profile plus 45 pixels. The outer limits need to be large enough to be able to model the full object
        profile in 2D.
        """
        if full:
            # All of the object profile
            return [850.0, 1700.0], [1840.0, 2000.0]
        else:
            # Part of the object profile
            return [850.0, 1700.0], [1840.0, 2000.0]

    def print_SNregions(self, arr):
        """ Print the S/N in certain regions of the spectrum
        These values are relevant for tet01 Ori A, during the 2022 observations
        """
        print("(box) S/N = ", np.mean(arr[1400:1448]) / np.std(arr[1400:1448]))
        print("(box) S/N ab = ", np.mean(arr[1706:1726]) / np.std(arr[1706:1726]))

    def get_SNregions_fit(self, flux):
        """ Print the S/N in certain regions of the spectrum
        These values are relevant for tet01 Ori A, during the 2022 observations
        """
        # xfit = np.arange(1580, 1605)
        xfit = np.arange(1455, 1525)
        ww = (xfit,)
        modl = np.polyval(np.polyfit(xfit, flux[ww], 2), xfit)
        SN_spec = 1.0 / np.std(flux[ww] / modl)
        # xfit = np.arange(1706, 1726)
        xfit = np.arange(1750, 1770)
        ww = (xfit,)
        modl = np.polyval(np.polyfit(xfit, flux[ww], 2), xfit)
        SN_abs = 1.0 / np.std(flux[ww] / modl)
        return SN_spec, SN_abs


if __name__ == '__main__':
    main()

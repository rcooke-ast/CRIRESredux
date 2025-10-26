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
    thisred.makePaths(redux_path="/Users/rcooke/Work/Research/BBN/helium34/Absorption/2023_CRIRES_Survey/WRAY_15-199/2024-03-17/")
    thisred._plotit = False
    thisred._comb_set = -1
    thisred.run()


class Reduce(ReduceBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Change some of the default parameters
        self._nbasis = 5  # Number of basis functions to use for the continuum
        self._numcomp = 1
        self._scalevariance = [10827.0, 10829.5]  # Scale the variance to match the measured variance in these regions
        self._scale_errors = True  # Scale the errors by 10x in regions with low flux. This is only used for fitting the wavelength solution with ALIS. The errors are scaled back to their extraction values during the combination.

    def get_science_frames(self):
        """
        This was auto-generated with step_make_combinations()
        """
#                A=3.0  (spec=0  DIT=240.0  NDIT=1)
        return [[["CRIRE.2024-03-18T00:32:34.760.fits"], ["CRIRE.2024-03-18T01:48:12.579.fits",    # B=6.5
                                                       "CRIRE.2024-03-18T01:14:55.055.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T03:15:03.876.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T02:05:48.434.fits",    # B=5.5
                                                       "CRIRE.2024-03-18T02:23:37.553.fits",    # B=5.0
                                                       "CRIRE.2024-03-18T02:46:37.244.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T00:45:56.962.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T01:05:46.101.fits",    # B=4.0
                                                       "CRIRE.2024-03-18T02:55:31.439.fits",    # B=3.5
                                                       "CRIRE.2024-03-18T02:37:30.740.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T00:36:57.185.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T02:14:45.556.fits",    # B=2.5
                                                       "CRIRE.2024-03-18T01:56:57.278.fits",    # B=2.0
                                                       "CRIRE.2024-03-18T01:23:34.739.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T03:23:45.842.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T01:39:24.759.fits"]],  # B=1.0
#                B=3.0  (spec=1  DIT=240.0  NDIT=1)
            [["CRIRE.2024-03-18T00:36:57.185.fits"], ["CRIRE.2024-03-18T01:35:10.883.fits",    # A=1.0
                                                       "CRIRE.2024-03-18T01:19:21.386.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T03:19:33.158.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T01:52:44.305.fits",    # A=2.0
                                                       "CRIRE.2024-03-18T02:10:27.640.fits",    # A=2.5
                                                       "CRIRE.2024-03-18T02:33:12.480.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T00:32:34.760.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T02:51:07.489.fits",    # A=3.5
                                                       "CRIRE.2024-03-18T01:01:17.559.fits",    # A=4.0
                                                       "CRIRE.2024-03-18T02:42:08.339.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T00:41:32.102.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T02:19:12.721.fits",    # A=5.0
                                                       "CRIRE.2024-03-18T02:01:23.763.fits",    # A=5.5
                                                       "CRIRE.2024-03-18T01:10:32.765.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T03:10:44.465.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T01:43:49.758.fits"]],  # A=6.5
    #                A=4.5  (spec=2  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T00:41:32.102.fits"], ["CRIRE.2024-03-18T01:48:12.579.fits",    # B=6.5
                                                       "CRIRE.2024-03-18T01:14:55.055.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T03:15:03.876.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T02:05:48.434.fits",    # B=5.5
                                                       "CRIRE.2024-03-18T02:23:37.553.fits",    # B=5.0
                                                       "CRIRE.2024-03-18T02:46:37.244.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T00:45:56.962.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T01:05:46.101.fits",    # B=4.0
                                                       "CRIRE.2024-03-18T02:55:31.439.fits",    # B=3.5
                                                       "CRIRE.2024-03-18T02:37:30.740.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T00:36:57.185.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T02:14:45.556.fits",    # B=2.5
                                                       "CRIRE.2024-03-18T01:56:57.278.fits",    # B=2.0
                                                       "CRIRE.2024-03-18T01:23:34.739.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T03:23:45.842.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T01:39:24.759.fits"]],  # B=1.0
    #                B=4.5  (spec=3  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T00:45:56.962.fits"], ["CRIRE.2024-03-18T01:35:10.883.fits",    # A=1.0
                                                       "CRIRE.2024-03-18T01:19:21.386.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T03:19:33.158.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T01:52:44.305.fits",    # A=2.0
                                                       "CRIRE.2024-03-18T02:10:27.640.fits",    # A=2.5
                                                       "CRIRE.2024-03-18T02:33:12.480.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T00:32:34.760.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T02:51:07.489.fits",    # A=3.5
                                                       "CRIRE.2024-03-18T01:01:17.559.fits",    # A=4.0
                                                       "CRIRE.2024-03-18T02:42:08.339.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T00:41:32.102.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T02:19:12.721.fits",    # A=5.0
                                                       "CRIRE.2024-03-18T02:01:23.763.fits",    # A=5.5
                                                       "CRIRE.2024-03-18T01:10:32.765.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T03:10:44.465.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T01:43:49.758.fits"]],  # A=6.5
    #                A=3.5  (spec=4  DIT=300.0  NDIT=1)
             [["CRIRE.2024-03-18T00:50:31.816.fits"], ["CRIRE.2024-03-18T03:05:11.621.fits",    # B=4.0
                                                       "CRIRE.2024-03-18T00:55:48.898.fits"]],  # B=3.5
    #                B=3.5  (spec=5  DIT=300.0  NDIT=1)
             [["CRIRE.2024-03-18T00:55:48.898.fits"], ["CRIRE.2024-03-18T00:50:31.816.fits",    # A=3.5
                                                       "CRIRE.2024-03-18T02:59:54.844.fits"]],  # A=4.0
    #                A=4.0  (spec=6  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T01:01:17.559.fits"], ["CRIRE.2024-03-18T01:48:12.579.fits",    # B=6.5
                                                       "CRIRE.2024-03-18T01:14:55.055.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T03:15:03.876.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T02:05:48.434.fits",    # B=5.5
                                                       "CRIRE.2024-03-18T02:23:37.553.fits",    # B=5.0
                                                       "CRIRE.2024-03-18T02:46:37.244.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T00:45:56.962.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T01:05:46.101.fits",    # B=4.0
                                                       "CRIRE.2024-03-18T02:55:31.439.fits",    # B=3.5
                                                       "CRIRE.2024-03-18T02:37:30.740.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T00:36:57.185.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T02:14:45.556.fits",    # B=2.5
                                                       "CRIRE.2024-03-18T01:56:57.278.fits",    # B=2.0
                                                       "CRIRE.2024-03-18T01:23:34.739.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T03:23:45.842.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T01:39:24.759.fits"]],  # B=1.0
    #                B=4.0  (spec=7  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T01:05:46.101.fits"], ["CRIRE.2024-03-18T01:35:10.883.fits",    # A=1.0
                                                       "CRIRE.2024-03-18T01:19:21.386.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T03:19:33.158.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T01:52:44.305.fits",    # A=2.0
                                                       "CRIRE.2024-03-18T02:10:27.640.fits",    # A=2.5
                                                       "CRIRE.2024-03-18T02:33:12.480.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T00:32:34.760.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T02:51:07.489.fits",    # A=3.5
                                                       "CRIRE.2024-03-18T01:01:17.559.fits",    # A=4.0
                                                       "CRIRE.2024-03-18T02:42:08.339.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T00:41:32.102.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T02:19:12.721.fits",    # A=5.0
                                                       "CRIRE.2024-03-18T02:01:23.763.fits",    # A=5.5
                                                       "CRIRE.2024-03-18T01:10:32.765.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T03:10:44.465.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T01:43:49.758.fits"]],  # A=6.5
    #                A=6.0  (spec=8  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T01:10:32.765.fits"], ["CRIRE.2024-03-18T01:48:12.579.fits",    # B=6.5
                                                       "CRIRE.2024-03-18T01:14:55.055.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T03:15:03.876.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T02:05:48.434.fits",    # B=5.5
                                                       "CRIRE.2024-03-18T02:23:37.553.fits",    # B=5.0
                                                       "CRIRE.2024-03-18T02:46:37.244.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T00:45:56.962.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T01:05:46.101.fits",    # B=4.0
                                                       "CRIRE.2024-03-18T02:55:31.439.fits",    # B=3.5
                                                       "CRIRE.2024-03-18T00:36:57.185.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T02:37:30.740.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T02:14:45.556.fits",    # B=2.5
                                                       "CRIRE.2024-03-18T01:56:57.278.fits",    # B=2.0
                                                       "CRIRE.2024-03-18T03:23:45.842.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T01:23:34.739.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T01:39:24.759.fits",    # B=1.0
                                                       "CRIRE.2024-03-18T01:35:10.883.fits",    # A=1.0
                                                       "CRIRE.2024-03-18T01:19:21.386.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T03:19:33.158.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T01:52:44.305.fits"]],  # A=2.0
    #                B=6.0  (spec=9  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T01:14:55.055.fits"], ["CRIRE.2024-03-18T01:56:57.278.fits",    # B=2.0
                                                       "CRIRE.2024-03-18T01:23:34.739.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T03:23:45.842.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T01:39:24.759.fits",    # B=1.0
                                                       "CRIRE.2024-03-18T01:35:10.883.fits",    # A=1.0
                                                       "CRIRE.2024-03-18T01:19:21.386.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T03:19:33.158.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T01:52:44.305.fits",    # A=2.0
                                                       "CRIRE.2024-03-18T02:10:27.640.fits",    # A=2.5
                                                       "CRIRE.2024-03-18T00:32:34.760.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T02:33:12.480.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T02:51:07.489.fits",    # A=3.5
                                                       "CRIRE.2024-03-18T01:01:17.559.fits",    # A=4.0
                                                       "CRIRE.2024-03-18T02:42:08.339.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T00:41:32.102.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T02:19:12.721.fits",    # A=5.0
                                                       "CRIRE.2024-03-18T02:01:23.763.fits",    # A=5.5
                                                       "CRIRE.2024-03-18T01:10:32.765.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T03:10:44.465.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T01:43:49.758.fits"]],  # A=6.5
    #                A=1.5  (spec=10  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T01:19:21.386.fits"], ["CRIRE.2024-03-18T01:48:12.579.fits",    # B=6.5
                                                       "CRIRE.2024-03-18T01:14:55.055.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T03:15:03.876.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T02:05:48.434.fits",    # B=5.5
                                                       "CRIRE.2024-03-18T02:23:37.553.fits",    # B=5.0
                                                       "CRIRE.2024-03-18T02:46:37.244.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T00:45:56.962.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T01:05:46.101.fits",    # B=4.0
                                                       "CRIRE.2024-03-18T02:55:31.439.fits",    # B=3.5
                                                       "CRIRE.2024-03-18T02:37:30.740.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T00:36:57.185.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T02:14:45.556.fits",    # B=2.5
                                                       "CRIRE.2024-03-18T02:01:23.763.fits",    # A=5.5
                                                       "CRIRE.2024-03-18T01:10:32.765.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T03:10:44.465.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T01:43:49.758.fits"]],  # A=6.5
    #                B=1.5  (spec=11  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T01:23:34.739.fits"], ["CRIRE.2024-03-18T01:48:12.579.fits",    # B=6.5
                                                       "CRIRE.2024-03-18T01:14:55.055.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T03:15:03.876.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T02:05:48.434.fits",    # B=5.5
                                                       "CRIRE.2024-03-18T02:10:27.640.fits",    # A=2.5
                                                       "CRIRE.2024-03-18T02:33:12.480.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T00:32:34.760.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T02:51:07.489.fits",    # A=3.5
                                                       "CRIRE.2024-03-18T01:01:17.559.fits",    # A=4.0
                                                       "CRIRE.2024-03-18T02:42:08.339.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T00:41:32.102.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T02:19:12.721.fits",    # A=5.0
                                                       "CRIRE.2024-03-18T02:01:23.763.fits",    # A=5.5
                                                       "CRIRE.2024-03-18T01:10:32.765.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T03:10:44.465.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T01:43:49.758.fits"]],  # A=6.5
    #                A=1.0  (spec=12  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T01:35:10.883.fits"], ["CRIRE.2024-03-18T01:48:12.579.fits",    # B=6.5
                                                       "CRIRE.2024-03-18T01:14:55.055.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T03:15:03.876.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T02:05:48.434.fits",    # B=5.5
                                                       "CRIRE.2024-03-18T02:23:37.553.fits",    # B=5.0
                                                       "CRIRE.2024-03-18T02:46:37.244.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T00:45:56.962.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T01:05:46.101.fits",    # B=4.0
                                                       "CRIRE.2024-03-18T02:55:31.439.fits",    # B=3.5
                                                       "CRIRE.2024-03-18T02:37:30.740.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T00:36:57.185.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T02:19:12.721.fits",    # A=5.0
                                                       "CRIRE.2024-03-18T02:01:23.763.fits",    # A=5.5
                                                       "CRIRE.2024-03-18T01:10:32.765.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T03:10:44.465.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T01:43:49.758.fits"]],  # A=6.5
    #                B=1.0  (spec=13  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T01:39:24.759.fits"], ["CRIRE.2024-03-18T01:48:12.579.fits",    # B=6.5
                                                       "CRIRE.2024-03-18T01:14:55.055.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T03:15:03.876.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T02:05:48.434.fits",    # B=5.5
                                                       "CRIRE.2024-03-18T02:23:37.553.fits",    # B=5.0
                                                       "CRIRE.2024-03-18T02:33:12.480.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T00:32:34.760.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T02:51:07.489.fits",    # A=3.5
                                                       "CRIRE.2024-03-18T01:01:17.559.fits",    # A=4.0
                                                       "CRIRE.2024-03-18T02:42:08.339.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T00:41:32.102.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T02:19:12.721.fits",    # A=5.0
                                                       "CRIRE.2024-03-18T02:01:23.763.fits",    # A=5.5
                                                       "CRIRE.2024-03-18T01:10:32.765.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T03:10:44.465.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T01:43:49.758.fits"]],  # A=6.5
    #                A=6.5  (spec=14  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T01:43:49.758.fits"], ["CRIRE.2024-03-18T01:48:12.579.fits",    # B=6.5
                                                       "CRIRE.2024-03-18T01:14:55.055.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T03:15:03.876.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T02:05:48.434.fits",    # B=5.5
                                                       "CRIRE.2024-03-18T02:23:37.553.fits",    # B=5.0
                                                       "CRIRE.2024-03-18T00:45:56.962.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T02:46:37.244.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T01:05:46.101.fits",    # B=4.0
                                                       "CRIRE.2024-03-18T02:55:31.439.fits",    # B=3.5
                                                       "CRIRE.2024-03-18T02:37:30.740.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T00:36:57.185.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T02:14:45.556.fits",    # B=2.5
                                                       "CRIRE.2024-03-18T01:56:57.278.fits",    # B=2.0
                                                       "CRIRE.2024-03-18T03:23:45.842.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T01:23:34.739.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T01:39:24.759.fits",    # B=1.0
                                                       "CRIRE.2024-03-18T01:35:10.883.fits",    # A=1.0
                                                       "CRIRE.2024-03-18T03:19:33.158.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T01:19:21.386.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T01:52:44.305.fits",    # A=2.0
                                                       "CRIRE.2024-03-18T02:10:27.640.fits"]],  # A=2.5
    #                B=6.5  (spec=15  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T01:48:12.579.fits"], ["CRIRE.2024-03-18T02:14:45.556.fits",    # B=2.5
                                                       "CRIRE.2024-03-18T01:56:57.278.fits",    # B=2.0
                                                       "CRIRE.2024-03-18T01:23:34.739.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T03:23:45.842.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T01:39:24.759.fits",    # B=1.0
                                                       "CRIRE.2024-03-18T01:35:10.883.fits",    # A=1.0
                                                       "CRIRE.2024-03-18T03:19:33.158.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T01:19:21.386.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T01:52:44.305.fits",    # A=2.0
                                                       "CRIRE.2024-03-18T02:10:27.640.fits",    # A=2.5
                                                       "CRIRE.2024-03-18T02:33:12.480.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T00:32:34.760.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T02:51:07.489.fits",    # A=3.5
                                                       "CRIRE.2024-03-18T01:01:17.559.fits",    # A=4.0
                                                       "CRIRE.2024-03-18T02:42:08.339.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T00:41:32.102.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T02:19:12.721.fits",    # A=5.0
                                                       "CRIRE.2024-03-18T02:01:23.763.fits",    # A=5.5
                                                       "CRIRE.2024-03-18T03:10:44.465.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T01:10:32.765.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T01:43:49.758.fits"]],  # A=6.5
    #                A=2.0  (spec=16  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T01:52:44.305.fits"], ["CRIRE.2024-03-18T01:48:12.579.fits",    # B=6.5
                                                       "CRIRE.2024-03-18T01:14:55.055.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T03:15:03.876.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T02:05:48.434.fits",    # B=5.5
                                                       "CRIRE.2024-03-18T02:23:37.553.fits",    # B=5.0
                                                       "CRIRE.2024-03-18T02:46:37.244.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T00:45:56.962.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T01:05:46.101.fits",    # B=4.0
                                                       "CRIRE.2024-03-18T02:55:31.439.fits",    # B=3.5
                                                       "CRIRE.2024-03-18T02:37:30.740.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T00:36:57.185.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T02:14:45.556.fits",    # B=2.5
                                                       "CRIRE.2024-03-18T01:56:57.278.fits",    # B=2.0
                                                       "CRIRE.2024-03-18T01:10:32.765.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T03:10:44.465.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T01:43:49.758.fits"]],  # A=6.5
    #                B=2.0  (spec=17  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T01:56:57.278.fits"], ["CRIRE.2024-03-18T01:48:12.579.fits",    # B=6.5
                                                       "CRIRE.2024-03-18T01:14:55.055.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T03:15:03.876.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T01:52:44.305.fits",    # A=2.0
                                                       "CRIRE.2024-03-18T02:10:27.640.fits",    # A=2.5
                                                       "CRIRE.2024-03-18T02:33:12.480.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T00:32:34.760.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T02:51:07.489.fits",    # A=3.5
                                                       "CRIRE.2024-03-18T01:01:17.559.fits",    # A=4.0
                                                       "CRIRE.2024-03-18T02:42:08.339.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T00:41:32.102.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T02:19:12.721.fits",    # A=5.0
                                                       "CRIRE.2024-03-18T02:01:23.763.fits",    # A=5.5
                                                       "CRIRE.2024-03-18T01:10:32.765.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T03:10:44.465.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T01:43:49.758.fits"]],  # A=6.5
    #                A=5.5  (spec=18  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T02:01:23.763.fits"], ["CRIRE.2024-03-18T01:48:12.579.fits",    # B=6.5
                                                       "CRIRE.2024-03-18T01:14:55.055.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T03:15:03.876.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T02:05:48.434.fits",    # B=5.5
                                                       "CRIRE.2024-03-18T02:23:37.553.fits",    # B=5.0
                                                       "CRIRE.2024-03-18T02:46:37.244.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T00:45:56.962.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T01:05:46.101.fits",    # B=4.0
                                                       "CRIRE.2024-03-18T02:55:31.439.fits",    # B=3.5
                                                       "CRIRE.2024-03-18T00:36:57.185.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T02:37:30.740.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T02:14:45.556.fits",    # B=2.5
                                                       "CRIRE.2024-03-18T01:56:57.278.fits",    # B=2.0
                                                       "CRIRE.2024-03-18T01:23:34.739.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T03:23:45.842.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T01:39:24.759.fits",    # B=1.0
                                                       "CRIRE.2024-03-18T01:35:10.883.fits",    # A=1.0
                                                       "CRIRE.2024-03-18T01:19:21.386.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T03:19:33.158.fits"]],  # A=1.5
    #                B=5.5  (spec=19  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T02:05:48.434.fits"], ["CRIRE.2024-03-18T01:23:34.739.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T03:23:45.842.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T01:39:24.759.fits",    # B=1.0
                                                       "CRIRE.2024-03-18T01:35:10.883.fits",    # A=1.0
                                                       "CRIRE.2024-03-18T03:19:33.158.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T01:19:21.386.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T01:52:44.305.fits",    # A=2.0
                                                       "CRIRE.2024-03-18T02:10:27.640.fits",    # A=2.5
                                                       "CRIRE.2024-03-18T00:32:34.760.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T02:33:12.480.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T02:51:07.489.fits",    # A=3.5
                                                       "CRIRE.2024-03-18T01:01:17.559.fits",    # A=4.0
                                                       "CRIRE.2024-03-18T02:42:08.339.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T00:41:32.102.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T02:19:12.721.fits",    # A=5.0
                                                       "CRIRE.2024-03-18T02:01:23.763.fits",    # A=5.5
                                                       "CRIRE.2024-03-18T03:10:44.465.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T01:10:32.765.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T01:43:49.758.fits"]],  # A=6.5
    #                A=2.5  (spec=20  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T02:10:27.640.fits"], ["CRIRE.2024-03-18T01:48:12.579.fits",    # B=6.5
                                                       "CRIRE.2024-03-18T01:14:55.055.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T03:15:03.876.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T02:05:48.434.fits",    # B=5.5
                                                       "CRIRE.2024-03-18T02:23:37.553.fits",    # B=5.0
                                                       "CRIRE.2024-03-18T02:46:37.244.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T00:45:56.962.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T01:05:46.101.fits",    # B=4.0
                                                       "CRIRE.2024-03-18T02:55:31.439.fits",    # B=3.5
                                                       "CRIRE.2024-03-18T02:37:30.740.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T00:36:57.185.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T02:14:45.556.fits",    # B=2.5
                                                       "CRIRE.2024-03-18T01:56:57.278.fits",    # B=2.0
                                                       "CRIRE.2024-03-18T01:23:34.739.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T03:23:45.842.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T01:43:49.758.fits"]],  # A=6.5
    #                B=2.5  (spec=21  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T02:14:45.556.fits"], ["CRIRE.2024-03-18T01:48:12.579.fits",    # B=6.5
                                                       "CRIRE.2024-03-18T01:19:21.386.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T03:19:33.158.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T01:52:44.305.fits",    # A=2.0
                                                       "CRIRE.2024-03-18T02:10:27.640.fits",    # A=2.5
                                                       "CRIRE.2024-03-18T02:33:12.480.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T00:32:34.760.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T02:51:07.489.fits",    # A=3.5
                                                       "CRIRE.2024-03-18T01:01:17.559.fits",    # A=4.0
                                                       "CRIRE.2024-03-18T02:42:08.339.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T00:41:32.102.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T02:19:12.721.fits",    # A=5.0
                                                       "CRIRE.2024-03-18T02:01:23.763.fits",    # A=5.5
                                                       "CRIRE.2024-03-18T01:10:32.765.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T03:10:44.465.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T01:43:49.758.fits"]],  # A=6.5
    #                A=5.0  (spec=22  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T02:19:12.721.fits"], ["CRIRE.2024-03-18T01:48:12.579.fits",    # B=6.5
                                                       "CRIRE.2024-03-18T01:14:55.055.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T03:15:03.876.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T02:05:48.434.fits",    # B=5.5
                                                       "CRIRE.2024-03-18T02:23:37.553.fits",    # B=5.0
                                                       "CRIRE.2024-03-18T02:46:37.244.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T00:45:56.962.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T01:05:46.101.fits",    # B=4.0
                                                       "CRIRE.2024-03-18T02:55:31.439.fits",    # B=3.5
                                                       "CRIRE.2024-03-18T02:37:30.740.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T00:36:57.185.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T02:14:45.556.fits",    # B=2.5
                                                       "CRIRE.2024-03-18T01:56:57.278.fits",    # B=2.0
                                                       "CRIRE.2024-03-18T01:23:34.739.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T03:23:45.842.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T01:39:24.759.fits",    # B=1.0
                                                       "CRIRE.2024-03-18T01:35:10.883.fits"]],  # A=1.0
    #                B=5.0  (spec=23  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T02:23:37.553.fits"], ["CRIRE.2024-03-18T01:39:24.759.fits",    # B=1.0
                                                       "CRIRE.2024-03-18T01:35:10.883.fits",    # A=1.0
                                                       "CRIRE.2024-03-18T01:19:21.386.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T03:19:33.158.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T01:52:44.305.fits",    # A=2.0
                                                       "CRIRE.2024-03-18T02:10:27.640.fits",    # A=2.5
                                                       "CRIRE.2024-03-18T00:32:34.760.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T02:33:12.480.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T02:51:07.489.fits",    # A=3.5
                                                       "CRIRE.2024-03-18T01:01:17.559.fits",    # A=4.0
                                                       "CRIRE.2024-03-18T00:41:32.102.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T02:42:08.339.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T02:19:12.721.fits",    # A=5.0
                                                       "CRIRE.2024-03-18T02:01:23.763.fits",    # A=5.5
                                                       "CRIRE.2024-03-18T03:10:44.465.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T01:10:32.765.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T01:43:49.758.fits"]],  # A=6.5
    #                A=3.0  (spec=24  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T02:33:12.480.fits"], ["CRIRE.2024-03-18T01:48:12.579.fits",    # B=6.5
                                                       "CRIRE.2024-03-18T01:14:55.055.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T03:15:03.876.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T02:05:48.434.fits",    # B=5.5
                                                       "CRIRE.2024-03-18T02:23:37.553.fits",    # B=5.0
                                                       "CRIRE.2024-03-18T02:46:37.244.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T00:45:56.962.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T01:05:46.101.fits",    # B=4.0
                                                       "CRIRE.2024-03-18T02:55:31.439.fits",    # B=3.5
                                                       "CRIRE.2024-03-18T02:37:30.740.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T00:36:57.185.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T02:14:45.556.fits",    # B=2.5
                                                       "CRIRE.2024-03-18T01:56:57.278.fits",    # B=2.0
                                                       "CRIRE.2024-03-18T01:23:34.739.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T03:23:45.842.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T01:39:24.759.fits"]],  # B=1.0
    #                B=3.0  (spec=25  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T02:37:30.740.fits"], ["CRIRE.2024-03-18T01:35:10.883.fits",    # A=1.0
                                                       "CRIRE.2024-03-18T01:19:21.386.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T03:19:33.158.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T01:52:44.305.fits",    # A=2.0
                                                       "CRIRE.2024-03-18T02:10:27.640.fits",    # A=2.5
                                                       "CRIRE.2024-03-18T02:33:12.480.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T00:32:34.760.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T02:51:07.489.fits",    # A=3.5
                                                       "CRIRE.2024-03-18T01:01:17.559.fits",    # A=4.0
                                                       "CRIRE.2024-03-18T02:42:08.339.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T00:41:32.102.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T02:19:12.721.fits",    # A=5.0
                                                       "CRIRE.2024-03-18T02:01:23.763.fits",    # A=5.5
                                                       "CRIRE.2024-03-18T01:10:32.765.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T03:10:44.465.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T01:43:49.758.fits"]],  # A=6.5
    #                A=4.5  (spec=26  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T02:42:08.339.fits"], ["CRIRE.2024-03-18T01:48:12.579.fits",    # B=6.5
                                                       "CRIRE.2024-03-18T01:14:55.055.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T03:15:03.876.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T02:05:48.434.fits",    # B=5.5
                                                       "CRIRE.2024-03-18T02:23:37.553.fits",    # B=5.0
                                                       "CRIRE.2024-03-18T02:46:37.244.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T00:45:56.962.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T01:05:46.101.fits",    # B=4.0
                                                       "CRIRE.2024-03-18T02:55:31.439.fits",    # B=3.5
                                                       "CRIRE.2024-03-18T02:37:30.740.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T00:36:57.185.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T02:14:45.556.fits",    # B=2.5
                                                       "CRIRE.2024-03-18T01:56:57.278.fits",    # B=2.0
                                                       "CRIRE.2024-03-18T01:23:34.739.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T03:23:45.842.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T01:39:24.759.fits"]],  # B=1.0
    #                B=4.5  (spec=27  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T02:46:37.244.fits"], ["CRIRE.2024-03-18T01:35:10.883.fits",    # A=1.0
                                                       "CRIRE.2024-03-18T01:19:21.386.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T03:19:33.158.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T01:52:44.305.fits",    # A=2.0
                                                       "CRIRE.2024-03-18T02:10:27.640.fits",    # A=2.5
                                                       "CRIRE.2024-03-18T02:33:12.480.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T00:32:34.760.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T02:51:07.489.fits",    # A=3.5
                                                       "CRIRE.2024-03-18T01:01:17.559.fits",    # A=4.0
                                                       "CRIRE.2024-03-18T02:42:08.339.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T00:41:32.102.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T02:19:12.721.fits",    # A=5.0
                                                       "CRIRE.2024-03-18T02:01:23.763.fits",    # A=5.5
                                                       "CRIRE.2024-03-18T01:10:32.765.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T03:10:44.465.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T01:43:49.758.fits"]],  # A=6.5
    #                A=3.5  (spec=28  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T02:51:07.489.fits"], ["CRIRE.2024-03-18T01:48:12.579.fits",    # B=6.5
                                                       "CRIRE.2024-03-18T01:14:55.055.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T03:15:03.876.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T02:05:48.434.fits",    # B=5.5
                                                       "CRIRE.2024-03-18T02:23:37.553.fits",    # B=5.0
                                                       "CRIRE.2024-03-18T02:46:37.244.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T00:45:56.962.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T01:05:46.101.fits",    # B=4.0
                                                       "CRIRE.2024-03-18T02:55:31.439.fits",    # B=3.5
                                                       "CRIRE.2024-03-18T02:37:30.740.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T00:36:57.185.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T02:14:45.556.fits",    # B=2.5
                                                       "CRIRE.2024-03-18T01:56:57.278.fits",    # B=2.0
                                                       "CRIRE.2024-03-18T01:23:34.739.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T03:23:45.842.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T01:39:24.759.fits"]],  # B=1.0
    #                B=3.5  (spec=29  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T02:55:31.439.fits"], ["CRIRE.2024-03-18T01:35:10.883.fits",    # A=1.0
                                                       "CRIRE.2024-03-18T01:19:21.386.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T03:19:33.158.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T01:52:44.305.fits",    # A=2.0
                                                       "CRIRE.2024-03-18T02:10:27.640.fits",    # A=2.5
                                                       "CRIRE.2024-03-18T02:33:12.480.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T00:32:34.760.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T02:51:07.489.fits",    # A=3.5
                                                       "CRIRE.2024-03-18T01:01:17.559.fits",    # A=4.0
                                                       "CRIRE.2024-03-18T02:42:08.339.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T00:41:32.102.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T02:19:12.721.fits",    # A=5.0
                                                       "CRIRE.2024-03-18T02:01:23.763.fits",    # A=5.5
                                                       "CRIRE.2024-03-18T01:10:32.765.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T03:10:44.465.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T01:43:49.758.fits"]],  # A=6.5
    #                A=4.0  (spec=30  DIT=300.0  NDIT=1)
             [["CRIRE.2024-03-18T02:59:54.844.fits"], ["CRIRE.2024-03-18T03:05:11.621.fits",    # B=4.0
                                                       "CRIRE.2024-03-18T00:55:48.898.fits"]],  # B=3.5
    #                B=4.0  (spec=31  DIT=300.0  NDIT=1)
             [["CRIRE.2024-03-18T03:05:11.621.fits"], ["CRIRE.2024-03-18T00:50:31.816.fits",    # A=3.5
                                                       "CRIRE.2024-03-18T02:59:54.844.fits"]],  # A=4.0
    #                A=6.0  (spec=32  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T03:10:44.465.fits"], ["CRIRE.2024-03-18T01:48:12.579.fits",    # B=6.5
                                                       "CRIRE.2024-03-18T01:14:55.055.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T03:15:03.876.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T02:05:48.434.fits",    # B=5.5
                                                       "CRIRE.2024-03-18T02:23:37.553.fits",    # B=5.0
                                                       "CRIRE.2024-03-18T02:46:37.244.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T00:45:56.962.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T01:05:46.101.fits",    # B=4.0
                                                       "CRIRE.2024-03-18T02:55:31.439.fits",    # B=3.5
                                                       "CRIRE.2024-03-18T00:36:57.185.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T02:37:30.740.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T02:14:45.556.fits",    # B=2.5
                                                       "CRIRE.2024-03-18T01:56:57.278.fits",    # B=2.0
                                                       "CRIRE.2024-03-18T03:23:45.842.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T01:23:34.739.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T01:39:24.759.fits",    # B=1.0
                                                       "CRIRE.2024-03-18T01:35:10.883.fits",    # A=1.0
                                                       "CRIRE.2024-03-18T01:19:21.386.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T03:19:33.158.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T01:52:44.305.fits"]],  # A=2.0
    #                B=6.0  (spec=33  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T03:15:03.876.fits"], ["CRIRE.2024-03-18T01:56:57.278.fits",    # B=2.0
                                                       "CRIRE.2024-03-18T01:23:34.739.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T03:23:45.842.fits",    # B=1.5
                                                       "CRIRE.2024-03-18T01:39:24.759.fits",    # B=1.0
                                                       "CRIRE.2024-03-18T01:35:10.883.fits",    # A=1.0
                                                       "CRIRE.2024-03-18T01:19:21.386.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T03:19:33.158.fits",    # A=1.5
                                                       "CRIRE.2024-03-18T01:52:44.305.fits",    # A=2.0
                                                       "CRIRE.2024-03-18T02:10:27.640.fits",    # A=2.5
                                                       "CRIRE.2024-03-18T00:32:34.760.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T02:33:12.480.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T02:51:07.489.fits",    # A=3.5
                                                       "CRIRE.2024-03-18T01:01:17.559.fits",    # A=4.0
                                                       "CRIRE.2024-03-18T02:42:08.339.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T00:41:32.102.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T02:19:12.721.fits",    # A=5.0
                                                       "CRIRE.2024-03-18T02:01:23.763.fits",    # A=5.5
                                                       "CRIRE.2024-03-18T01:10:32.765.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T03:10:44.465.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T01:43:49.758.fits"]],  # A=6.5
    #                A=1.5  (spec=34  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T03:19:33.158.fits"], ["CRIRE.2024-03-18T01:48:12.579.fits",    # B=6.5
                                                       "CRIRE.2024-03-18T01:14:55.055.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T03:15:03.876.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T02:05:48.434.fits",    # B=5.5
                                                       "CRIRE.2024-03-18T02:23:37.553.fits",    # B=5.0
                                                       "CRIRE.2024-03-18T02:46:37.244.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T00:45:56.962.fits",    # B=4.5
                                                       "CRIRE.2024-03-18T01:05:46.101.fits",    # B=4.0
                                                       "CRIRE.2024-03-18T02:55:31.439.fits",    # B=3.5
                                                       "CRIRE.2024-03-18T02:37:30.740.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T00:36:57.185.fits",    # B=3.0
                                                       "CRIRE.2024-03-18T02:14:45.556.fits",    # B=2.5
                                                       "CRIRE.2024-03-18T02:01:23.763.fits",    # A=5.5
                                                       "CRIRE.2024-03-18T01:10:32.765.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T03:10:44.465.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T01:43:49.758.fits"]],  # A=6.5
    #                B=1.5  (spec=35  DIT=240.0  NDIT=1)
             [["CRIRE.2024-03-18T03:23:45.842.fits"], ["CRIRE.2024-03-18T01:48:12.579.fits",    # B=6.5
                                                       "CRIRE.2024-03-18T01:14:55.055.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T03:15:03.876.fits",    # B=6.0
                                                       "CRIRE.2024-03-18T02:05:48.434.fits",    # B=5.5
                                                       "CRIRE.2024-03-18T02:10:27.640.fits",    # A=2.5
                                                       "CRIRE.2024-03-18T02:33:12.480.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T00:32:34.760.fits",    # A=3.0
                                                       "CRIRE.2024-03-18T02:51:07.489.fits",    # A=3.5
                                                       "CRIRE.2024-03-18T01:01:17.559.fits",    # A=4.0
                                                       "CRIRE.2024-03-18T02:42:08.339.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T00:41:32.102.fits",    # A=4.5
                                                       "CRIRE.2024-03-18T02:19:12.721.fits",    # A=5.0
                                                       "CRIRE.2024-03-18T02:01:23.763.fits",    # A=5.5
                                                       "CRIRE.2024-03-18T01:10:32.765.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T03:10:44.465.fits",    # A=6.0
                                                       "CRIRE.2024-03-18T01:43:49.758.fits"]]]  # A=6.5

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
        return ["CRIRE.2024-03-18T10:04:53.567.fits",
                "CRIRE.2024-03-18T10:05:13.930.fits",
                "CRIRE.2024-03-18T10:05:34.290.fits",
                "CRIRE.2024-03-18T10:05:54.654.fits",
                "CRIRE.2024-03-18T10:06:15.018.fits"]

    def get_dark_frames(self):
        """
        """
        # Group dark files with different exposure times
        return [["CRIRE.2024-03-18T10:25:47.023.fits","CRIRE.2024-03-18T10:26:10.363.fits","CRIRE.2024-03-18T10:26:33.713.fits"],#5s
                ["CRIRE.2024-03-18T10:23:15.495.fits", "CRIRE.2024-03-18T10:24:05.993.fits", "CRIRE.2024-03-18T10:24:56.504.fits"],#45s
                ["CRIRE.2024-03-18T10:16:58.961.fits","CRIRE.2024-03-18T10:19:04.478.fits","CRIRE.2024-03-18T10:21:09.984.fits"]]#120s

    def get_arc_frames(self):
        return ["CRIRE.2024-03-18T10:10:58.837.fits"]

    def get_exptime(self, idx):
        ndit = self.get_ndit(idx)
        if idx in [4,5,30,31]:
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

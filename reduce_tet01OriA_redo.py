from reduce_base import ReduceBase
import numpy as np


def main():
    # Initialise the reduce class
    cals = False
    makediff = False
    step = 2
    thisred = Reduce(prefix="tet01OriA", match_name="tet01 Ori A", data_folder="Raw/",
                     use_diff=True,
                     step_listfiles=False, step_make_combinations=False,
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
    thisred.makePaths(redux_path="/Users/rcooke/Work/Research/BBN/helium34/Absorption/2023_CRIRES_Survey/tet01oriA/")
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
        #                A=1.0  (spec=0  DIT=10.0  NDIT=20)
        return [[["CRIRE.2022-10-24T06:00:36.335.fits"], ["CRIRE.2022-10-24T06:13:09.282.fits",  # B=6.5
                                                  "CRIRE.2022-10-26T07:37:01.870.fits",  # B=6.0
                                                  "CRIRE.2022-10-24T06:30:39.815.fits",  # B=5.5
                                                  "CRIRE.2022-10-24T06:52:30.465.fits",  # B=5.0
                                                  "CRIRE.2022-10-26T07:56:25.958.fits",  # B=4.5
                                                  "CRIRE.2022-10-26T08:14:33.857.fits",  # B=4.0
                                                  "CRIRE.2022-10-26T08:05:42.277.fits",  # B=3.5
                                                  "CRIRE.2022-10-26T07:45:45.398.fits",  # B=3.0
                                                  "CRIRE.2022-10-24T06:48:26.383.fits",  # A=5.0
                                                  "CRIRE.2022-10-24T06:26:34.004.fits",  # A=5.5
                                                  "CRIRE.2022-10-26T07:32:50.776.fits",  # A=6.0
                                                  "CRIRE.2022-10-24T06:09:01.470.fits"]],  # A=6.5
        #                B=1.0  (spec=1  DIT=10.0  NDIT=20)
        [["CRIRE.2022-10-24T06:04:35.716.fits"], ["CRIRE.2022-10-24T06:13:09.282.fits",  # B=6.5
                                                  "CRIRE.2022-10-26T07:37:01.870.fits",  # B=6.0
                                                  "CRIRE.2022-10-24T06:30:39.815.fits",  # B=5.5
                                                  "CRIRE.2022-10-24T06:52:30.465.fits",  # B=5.0
                                                  "CRIRE.2022-10-26T07:41:37.963.fits",  # A=3.0
                                                  "CRIRE.2022-10-26T08:01:34.447.fits",  # A=3.5
                                                  "CRIRE.2022-10-26T08:10:26.674.fits",  # A=4.0
                                                  "CRIRE.2022-10-26T07:52:14.741.fits",  # A=4.5
                                                  "CRIRE.2022-10-24T06:48:26.383.fits",  # A=5.0
                                                  "CRIRE.2022-10-24T06:26:34.004.fits",  # A=5.5
                                                  "CRIRE.2022-10-26T07:32:50.776.fits",  # A=6.0
                                                  "CRIRE.2022-10-24T06:09:01.470.fits"]],  # A=6.5
        #                A=6.5  (spec=2  DIT=10.0  NDIT=20)
        [["CRIRE.2022-10-24T06:09:01.470.fits"], ["CRIRE.2022-10-24T06:13:09.282.fits",  # B=6.5
                                                  "CRIRE.2022-10-26T07:37:01.870.fits",  # B=6.0
                                                  "CRIRE.2022-10-24T06:30:39.815.fits",  # B=5.5
                                                  "CRIRE.2022-10-24T06:52:30.465.fits",  # B=5.0
                                                  "CRIRE.2022-10-26T07:56:25.958.fits",  # B=4.5
                                                  "CRIRE.2022-10-26T08:14:33.857.fits",  # B=4.0
                                                  "CRIRE.2022-10-26T08:05:42.277.fits",  # B=3.5
                                                  "CRIRE.2022-10-26T07:45:45.398.fits",  # B=3.0
                                                  "CRIRE.2022-10-24T06:43:55.087.fits",  # B=2.5
                                                  "CRIRE.2022-10-24T06:22:02.926.fits",  # B=2.0
                                                  "CRIRE.2022-10-26T07:23:06.300.fits",  # B=1.5
                                                  "CRIRE.2022-10-24T06:04:35.716.fits",  # B=1.0
                                                  "CRIRE.2022-10-24T06:00:36.335.fits",  # A=1.0
                                                  "CRIRE.2022-10-26T07:19:05.099.fits",  # A=1.5
                                                  "CRIRE.2022-10-24T06:17:44.032.fits",  # A=2.0
                                                  "CRIRE.2022-10-24T06:39:29.485.fits"]],  # A=2.5
        #                B=6.5  (spec=3  DIT=10.0  NDIT=20)
        [["CRIRE.2022-10-24T06:13:09.282.fits"], ["CRIRE.2022-10-24T06:43:55.087.fits",  # B=2.5
                                                  "CRIRE.2022-10-24T06:22:02.926.fits",  # B=2.0
                                                  "CRIRE.2022-10-26T07:23:06.300.fits",  # B=1.5
                                                  "CRIRE.2022-10-24T06:04:35.716.fits",  # B=1.0
                                                  "CRIRE.2022-10-24T06:00:36.335.fits",  # A=1.0
                                                  "CRIRE.2022-10-26T07:19:05.099.fits",  # A=1.5
                                                  "CRIRE.2022-10-24T06:17:44.032.fits",  # A=2.0
                                                  "CRIRE.2022-10-24T06:39:29.485.fits",  # A=2.5
                                                  "CRIRE.2022-10-26T07:41:37.963.fits",  # A=3.0
                                                  "CRIRE.2022-10-26T08:01:34.447.fits",  # A=3.5
                                                  "CRIRE.2022-10-26T08:10:26.674.fits",  # A=4.0
                                                  "CRIRE.2022-10-26T07:52:14.741.fits",  # A=4.5
                                                  "CRIRE.2022-10-24T06:48:26.383.fits",  # A=5.0
                                                  "CRIRE.2022-10-24T06:26:34.004.fits",  # A=5.5
                                                  "CRIRE.2022-10-26T07:32:50.776.fits",  # A=6.0
                                                  "CRIRE.2022-10-24T06:09:01.470.fits"]],  # A=6.5
        #                A=2.0  (spec=4  DIT=10.0  NDIT=20)
        [["CRIRE.2022-10-24T06:17:44.032.fits"], ["CRIRE.2022-10-24T06:13:09.282.fits",  # B=6.5
                                                  "CRIRE.2022-10-26T07:37:01.870.fits",  # B=6.0
                                                  "CRIRE.2022-10-24T06:30:39.815.fits",  # B=5.5
                                                  "CRIRE.2022-10-24T06:52:30.465.fits",  # B=5.0
                                                  "CRIRE.2022-10-26T07:56:25.958.fits",  # B=4.5
                                                  "CRIRE.2022-10-26T08:14:33.857.fits",  # B=4.0
                                                  "CRIRE.2022-10-26T08:05:42.277.fits",  # B=3.5
                                                  "CRIRE.2022-10-26T07:45:45.398.fits",  # B=3.0
                                                  "CRIRE.2022-10-24T06:43:55.087.fits",  # B=2.5
                                                  "CRIRE.2022-10-24T06:22:02.926.fits",  # B=2.0
                                                  "CRIRE.2022-10-26T07:32:50.776.fits",  # A=6.0
                                                  "CRIRE.2022-10-24T06:09:01.470.fits"]],  # A=6.5
        #                B=2.0  (spec=5  DIT=10.0  NDIT=20)
        [["CRIRE.2022-10-24T06:22:02.926.fits"], ["CRIRE.2022-10-24T06:13:09.282.fits",  # B=6.5
                                                  "CRIRE.2022-10-26T07:37:01.870.fits",  # B=6.0
                                                  "CRIRE.2022-10-24T06:17:44.032.fits",  # A=2.0
                                                  "CRIRE.2022-10-24T06:39:29.485.fits",  # A=2.5
                                                  "CRIRE.2022-10-26T07:41:37.963.fits",  # A=3.0
                                                  "CRIRE.2022-10-26T08:01:34.447.fits",  # A=3.5
                                                  "CRIRE.2022-10-26T08:10:26.674.fits",  # A=4.0
                                                  "CRIRE.2022-10-26T07:52:14.741.fits",  # A=4.5
                                                  "CRIRE.2022-10-24T06:48:26.383.fits",  # A=5.0
                                                  "CRIRE.2022-10-24T06:26:34.004.fits",  # A=5.5
                                                  "CRIRE.2022-10-26T07:32:50.776.fits",  # A=6.0
                                                  "CRIRE.2022-10-24T06:09:01.470.fits"]],  # A=6.5
        #                A=5.5  (spec=6  DIT=10.0  NDIT=20)
        [["CRIRE.2022-10-24T06:26:34.004.fits"], ["CRIRE.2022-10-24T06:13:09.282.fits",  # B=6.5
                                                  "CRIRE.2022-10-26T07:37:01.870.fits",  # B=6.0
                                                  "CRIRE.2022-10-24T06:30:39.815.fits",  # B=5.5
                                                  "CRIRE.2022-10-24T06:52:30.465.fits",  # B=5.0
                                                  "CRIRE.2022-10-26T07:56:25.958.fits",  # B=4.5
                                                  "CRIRE.2022-10-26T08:14:33.857.fits",  # B=4.0
                                                  "CRIRE.2022-10-26T08:05:42.277.fits",  # B=3.5
                                                  "CRIRE.2022-10-26T07:45:45.398.fits",  # B=3.0
                                                  "CRIRE.2022-10-24T06:43:55.087.fits",  # B=2.5
                                                  "CRIRE.2022-10-24T06:22:02.926.fits",  # B=2.0
                                                  "CRIRE.2022-10-26T07:23:06.300.fits",  # B=1.5
                                                  "CRIRE.2022-10-24T06:04:35.716.fits",  # B=1.0
                                                  "CRIRE.2022-10-24T06:00:36.335.fits",  # A=1.0
                                                  "CRIRE.2022-10-26T07:19:05.099.fits"]],  # A=1.5
        #                B=5.5  (spec=7  DIT=10.0  NDIT=20)
        [["CRIRE.2022-10-24T06:30:39.815.fits"], ["CRIRE.2022-10-26T07:23:06.300.fits",  # B=1.5
                                                  "CRIRE.2022-10-24T06:04:35.716.fits",  # B=1.0
                                                  "CRIRE.2022-10-24T06:00:36.335.fits",  # A=1.0
                                                  "CRIRE.2022-10-26T07:19:05.099.fits",  # A=1.5
                                                  "CRIRE.2022-10-24T06:17:44.032.fits",  # A=2.0
                                                  "CRIRE.2022-10-24T06:39:29.485.fits",  # A=2.5
                                                  "CRIRE.2022-10-26T07:41:37.963.fits",  # A=3.0
                                                  "CRIRE.2022-10-26T08:01:34.447.fits",  # A=3.5
                                                  "CRIRE.2022-10-26T08:10:26.674.fits",  # A=4.0
                                                  "CRIRE.2022-10-26T07:52:14.741.fits",  # A=4.5
                                                  "CRIRE.2022-10-24T06:48:26.383.fits",  # A=5.0
                                                  "CRIRE.2022-10-24T06:26:34.004.fits",  # A=5.5
                                                  "CRIRE.2022-10-26T07:32:50.776.fits",  # A=6.0
                                                  "CRIRE.2022-10-24T06:09:01.470.fits"]],  # A=6.5
        #                A=2.5  (spec=8  DIT=10.0  NDIT=20)
        [["CRIRE.2022-10-24T06:39:29.485.fits"], ["CRIRE.2022-10-24T06:13:09.282.fits",  # B=6.5
                                                  "CRIRE.2022-10-26T07:37:01.870.fits",  # B=6.0
                                                  "CRIRE.2022-10-24T06:30:39.815.fits",  # B=5.5
                                                  "CRIRE.2022-10-24T06:52:30.465.fits",  # B=5.0
                                                  "CRIRE.2022-10-26T07:56:25.958.fits",  # B=4.5
                                                  "CRIRE.2022-10-26T08:14:33.857.fits",  # B=4.0
                                                  "CRIRE.2022-10-26T08:05:42.277.fits",  # B=3.5
                                                  "CRIRE.2022-10-26T07:45:45.398.fits",  # B=3.0
                                                  "CRIRE.2022-10-24T06:43:55.087.fits",  # B=2.5
                                                  "CRIRE.2022-10-24T06:22:02.926.fits",  # B=2.0
                                                  "CRIRE.2022-10-26T07:23:06.300.fits",  # B=1.5
                                                  "CRIRE.2022-10-24T06:09:01.470.fits"]],  # A=6.5
        #                B=2.5  (spec=9  DIT=10.0  NDIT=20)
        [["CRIRE.2022-10-24T06:43:55.087.fits"], ["CRIRE.2022-10-24T06:13:09.282.fits",  # B=6.5
                                                  "CRIRE.2022-10-26T07:19:05.099.fits",  # A=1.5
                                                  "CRIRE.2022-10-24T06:17:44.032.fits",  # A=2.0
                                                  "CRIRE.2022-10-24T06:39:29.485.fits",  # A=2.5
                                                  "CRIRE.2022-10-26T07:41:37.963.fits",  # A=3.0
                                                  "CRIRE.2022-10-26T08:01:34.447.fits",  # A=3.5
                                                  "CRIRE.2022-10-26T08:10:26.674.fits",  # A=4.0
                                                  "CRIRE.2022-10-26T07:52:14.741.fits",  # A=4.5
                                                  "CRIRE.2022-10-24T06:48:26.383.fits",  # A=5.0
                                                  "CRIRE.2022-10-24T06:26:34.004.fits",  # A=5.5
                                                  "CRIRE.2022-10-26T07:32:50.776.fits",  # A=6.0
                                                  "CRIRE.2022-10-24T06:09:01.470.fits"]],  # A=6.5
        #                A=5.0  (spec=10  DIT=10.0  NDIT=20)
        [["CRIRE.2022-10-24T06:48:26.383.fits"], ["CRIRE.2022-10-24T06:13:09.282.fits",  # B=6.5
                                                  "CRIRE.2022-10-26T07:37:01.870.fits",  # B=6.0
                                                  "CRIRE.2022-10-24T06:30:39.815.fits",  # B=5.5
                                                  "CRIRE.2022-10-24T06:52:30.465.fits",  # B=5.0
                                                  "CRIRE.2022-10-26T07:56:25.958.fits",  # B=4.5
                                                  "CRIRE.2022-10-26T08:14:33.857.fits",  # B=4.0
                                                  "CRIRE.2022-10-26T08:05:42.277.fits",  # B=3.5
                                                  "CRIRE.2022-10-26T07:45:45.398.fits",  # B=3.0
                                                  "CRIRE.2022-10-24T06:43:55.087.fits",  # B=2.5
                                                  "CRIRE.2022-10-24T06:22:02.926.fits",  # B=2.0
                                                  "CRIRE.2022-10-26T07:23:06.300.fits",  # B=1.5
                                                  "CRIRE.2022-10-24T06:04:35.716.fits",  # B=1.0
                                                  "CRIRE.2022-10-24T06:00:36.335.fits"]],  # A=1.0
        #                B=5.0  (spec=11  DIT=10.0  NDIT=20)
        [["CRIRE.2022-10-24T06:52:30.465.fits"], ["CRIRE.2022-10-24T06:04:35.716.fits",  # B=1.0
                                                  "CRIRE.2022-10-24T06:00:36.335.fits",  # A=1.0
                                                  "CRIRE.2022-10-26T07:19:05.099.fits",  # A=1.5
                                                  "CRIRE.2022-10-24T06:17:44.032.fits",  # A=2.0
                                                  "CRIRE.2022-10-24T06:39:29.485.fits",  # A=2.5
                                                  "CRIRE.2022-10-26T07:41:37.963.fits",  # A=3.0
                                                  "CRIRE.2022-10-26T08:01:34.447.fits",  # A=3.5
                                                  "CRIRE.2022-10-26T08:10:26.674.fits",  # A=4.0
                                                  "CRIRE.2022-10-26T07:52:14.741.fits",  # A=4.5
                                                  "CRIRE.2022-10-24T06:48:26.383.fits",  # A=5.0
                                                  "CRIRE.2022-10-24T06:26:34.004.fits",  # A=5.5
                                                  "CRIRE.2022-10-26T07:32:50.776.fits",  # A=6.0
                                                  "CRIRE.2022-10-24T06:09:01.470.fits"]],  # A=6.5
            #                A=1.5  (spec=13  DIT=10.0  NDIT=20)
            [["CRIRE.2022-10-26T07:19:05.099.fits"], ["CRIRE.2022-10-24T06:13:09.282.fits",  # B=6.5
                                                      "CRIRE.2022-10-26T07:37:01.870.fits",  # B=6.0
                                                      "CRIRE.2022-10-24T06:30:39.815.fits",  # B=5.5
                                                      "CRIRE.2022-10-24T06:52:30.465.fits",  # B=5.0
                                                      "CRIRE.2022-10-26T07:56:25.958.fits",  # B=4.5
                                                      "CRIRE.2022-10-26T08:14:33.857.fits",  # B=4.0
                                                      "CRIRE.2022-10-26T08:05:42.277.fits",  # B=3.5
                                                      "CRIRE.2022-10-26T07:45:45.398.fits",  # B=3.0
                                                      "CRIRE.2022-10-24T06:43:55.087.fits",  # B=2.5
                                                      "CRIRE.2022-10-24T06:26:34.004.fits",  # A=5.5
                                                      "CRIRE.2022-10-26T07:32:50.776.fits",  # A=6.0
                                                      "CRIRE.2022-10-24T06:09:01.470.fits"]],  # A=6.5
            #                B=1.5  (spec=14  DIT=10.0  NDIT=20)
            [["CRIRE.2022-10-26T07:23:06.300.fits"], ["CRIRE.2022-10-24T06:13:09.282.fits",  # B=6.5
                                                      "CRIRE.2022-10-26T07:37:01.870.fits",  # B=6.0
                                                      "CRIRE.2022-10-24T06:30:39.815.fits",  # B=5.5
                                                      "CRIRE.2022-10-24T06:39:29.485.fits",  # A=2.5
                                                      "CRIRE.2022-10-26T07:41:37.963.fits",  # A=3.0
                                                      "CRIRE.2022-10-26T08:01:34.447.fits",  # A=3.5
                                                      "CRIRE.2022-10-26T08:10:26.674.fits",  # A=4.0
                                                      "CRIRE.2022-10-26T07:52:14.741.fits",  # A=4.5
                                                      "CRIRE.2022-10-24T06:48:26.383.fits",  # A=5.0
                                                      "CRIRE.2022-10-24T06:26:34.004.fits",  # A=5.5
                                                      "CRIRE.2022-10-26T07:32:50.776.fits",  # A=6.0
                                                      "CRIRE.2022-10-24T06:09:01.470.fits"]],  # A=6.5
            #                A=6.0  (spec=15  DIT=10.0  NDIT=20)
            [["CRIRE.2022-10-26T07:32:50.776.fits"], ["CRIRE.2022-10-24T06:13:09.282.fits",  # B=6.5
                                                      "CRIRE.2022-10-26T07:37:01.870.fits",  # B=6.0
                                                      "CRIRE.2022-10-24T06:30:39.815.fits",  # B=5.5
                                                      "CRIRE.2022-10-24T06:52:30.465.fits",  # B=5.0
                                                      "CRIRE.2022-10-26T07:56:25.958.fits",  # B=4.5
                                                      "CRIRE.2022-10-26T08:14:33.857.fits",  # B=4.0
                                                      "CRIRE.2022-10-26T08:05:42.277.fits",  # B=3.5
                                                      "CRIRE.2022-10-26T07:45:45.398.fits",  # B=3.0
                                                      "CRIRE.2022-10-24T06:43:55.087.fits",  # B=2.5
                                                      "CRIRE.2022-10-24T06:22:02.926.fits",  # B=2.0
                                                      "CRIRE.2022-10-26T07:23:06.300.fits",  # B=1.5
                                                      "CRIRE.2022-10-24T06:04:35.716.fits",  # B=1.0
                                                      "CRIRE.2022-10-24T06:00:36.335.fits",  # A=1.0
                                                      "CRIRE.2022-10-26T07:19:05.099.fits",  # A=1.5
                                                      "CRIRE.2022-10-24T06:17:44.032.fits"]],  # A=2.0
            #                B=6.0  (spec=16  DIT=10.0  NDIT=20)
            [["CRIRE.2022-10-26T07:37:01.870.fits"], ["CRIRE.2022-10-24T06:22:02.926.fits",  # B=2.0
                                                      "CRIRE.2022-10-26T07:23:06.300.fits",  # B=1.5
                                                      "CRIRE.2022-10-24T06:04:35.716.fits",  # B=1.0
                                                      "CRIRE.2022-10-24T06:00:36.335.fits",  # A=1.0
                                                      "CRIRE.2022-10-26T07:19:05.099.fits",  # A=1.5
                                                      "CRIRE.2022-10-24T06:17:44.032.fits",  # A=2.0
                                                      "CRIRE.2022-10-24T06:39:29.485.fits",  # A=2.5
                                                      "CRIRE.2022-10-26T07:41:37.963.fits",  # A=3.0
                                                      "CRIRE.2022-10-26T08:01:34.447.fits",  # A=3.5
                                                      "CRIRE.2022-10-26T08:10:26.674.fits",  # A=4.0
                                                      "CRIRE.2022-10-26T07:52:14.741.fits",  # A=4.5
                                                      "CRIRE.2022-10-24T06:48:26.383.fits",  # A=5.0
                                                      "CRIRE.2022-10-24T06:26:34.004.fits",  # A=5.5
                                                      "CRIRE.2022-10-26T07:32:50.776.fits",  # A=6.0
                                                      "CRIRE.2022-10-24T06:09:01.470.fits"]],  # A=6.5
            #                A=3.0  (spec=17  DIT=10.0  NDIT=20)
            [["CRIRE.2022-10-26T07:41:37.963.fits"], ["CRIRE.2022-10-24T06:13:09.282.fits",  # B=6.5
                                                      "CRIRE.2022-10-26T07:37:01.870.fits",  # B=6.0
                                                      "CRIRE.2022-10-24T06:30:39.815.fits",  # B=5.5
                                                      "CRIRE.2022-10-24T06:52:30.465.fits",  # B=5.0
                                                      "CRIRE.2022-10-26T07:56:25.958.fits",  # B=4.5
                                                      "CRIRE.2022-10-26T08:14:33.857.fits",  # B=4.0
                                                      "CRIRE.2022-10-26T08:05:42.277.fits",  # B=3.5
                                                      "CRIRE.2022-10-26T07:45:45.398.fits",  # B=3.0
                                                      "CRIRE.2022-10-24T06:43:55.087.fits",  # B=2.5
                                                      "CRIRE.2022-10-24T06:22:02.926.fits",  # B=2.0
                                                      "CRIRE.2022-10-26T07:23:06.300.fits",  # B=1.5
                                                      "CRIRE.2022-10-24T06:04:35.716.fits"]],  # B=1.0
            #                B=3.0  (spec=18  DIT=10.0  NDIT=20)
            [["CRIRE.2022-10-26T07:45:45.398.fits"], ["CRIRE.2022-10-24T06:00:36.335.fits",  # A=1.0
                                                      "CRIRE.2022-10-26T07:19:05.099.fits",  # A=1.5
                                                      "CRIRE.2022-10-24T06:17:44.032.fits",  # A=2.0
                                                      "CRIRE.2022-10-24T06:39:29.485.fits",  # A=2.5
                                                      "CRIRE.2022-10-26T07:41:37.963.fits",  # A=3.0
                                                      "CRIRE.2022-10-26T08:01:34.447.fits",  # A=3.5
                                                      "CRIRE.2022-10-26T08:10:26.674.fits",  # A=4.0
                                                      "CRIRE.2022-10-26T07:52:14.741.fits",  # A=4.5
                                                      "CRIRE.2022-10-24T06:48:26.383.fits",  # A=5.0
                                                      "CRIRE.2022-10-24T06:26:34.004.fits",  # A=5.5
                                                      "CRIRE.2022-10-26T07:32:50.776.fits",  # A=6.0
                                                      "CRIRE.2022-10-24T06:09:01.470.fits"]],  # A=6.5
            #                A=4.5  (spec=19  DIT=10.0  NDIT=20)
            [["CRIRE.2022-10-26T07:52:14.741.fits"], ["CRIRE.2022-10-24T06:13:09.282.fits",  # B=6.5
                                                      "CRIRE.2022-10-26T07:37:01.870.fits",  # B=6.0
                                                      "CRIRE.2022-10-24T06:30:39.815.fits",  # B=5.5
                                                      "CRIRE.2022-10-24T06:52:30.465.fits",  # B=5.0
                                                      "CRIRE.2022-10-26T07:56:25.958.fits",  # B=4.5
                                                      "CRIRE.2022-10-26T08:14:33.857.fits",  # B=4.0
                                                      "CRIRE.2022-10-26T08:05:42.277.fits",  # B=3.5
                                                      "CRIRE.2022-10-26T07:45:45.398.fits",  # B=3.0
                                                      "CRIRE.2022-10-24T06:43:55.087.fits",  # B=2.5
                                                      "CRIRE.2022-10-24T06:22:02.926.fits",  # B=2.0
                                                      "CRIRE.2022-10-26T07:23:06.300.fits",  # B=1.5
                                                      "CRIRE.2022-10-24T06:04:35.716.fits"]],  # B=1.0
            #                B=4.5  (spec=20  DIT=10.0  NDIT=20)
            [["CRIRE.2022-10-26T07:56:25.958.fits"], ["CRIRE.2022-10-24T06:00:36.335.fits",  # A=1.0
                                                      "CRIRE.2022-10-26T07:19:05.099.fits",  # A=1.5
                                                      "CRIRE.2022-10-24T06:17:44.032.fits",  # A=2.0
                                                      "CRIRE.2022-10-24T06:39:29.485.fits",  # A=2.5
                                                      "CRIRE.2022-10-26T07:41:37.963.fits",  # A=3.0
                                                      "CRIRE.2022-10-26T08:01:34.447.fits",  # A=3.5
                                                      "CRIRE.2022-10-26T08:10:26.674.fits",  # A=4.0
                                                      "CRIRE.2022-10-26T07:52:14.741.fits",  # A=4.5
                                                      "CRIRE.2022-10-24T06:48:26.383.fits",  # A=5.0
                                                      "CRIRE.2022-10-24T06:26:34.004.fits",  # A=5.5
                                                      "CRIRE.2022-10-26T07:32:50.776.fits",  # A=6.0
                                                      "CRIRE.2022-10-24T06:09:01.470.fits"]],  # A=6.5
            #                A=3.5  (spec=21  DIT=10.0  NDIT=20)
            [["CRIRE.2022-10-26T08:01:34.447.fits"], ["CRIRE.2022-10-24T06:13:09.282.fits",  # B=6.5
                                                      "CRIRE.2022-10-26T07:37:01.870.fits",  # B=6.0
                                                      "CRIRE.2022-10-24T06:30:39.815.fits",  # B=5.5
                                                      "CRIRE.2022-10-24T06:52:30.465.fits",  # B=5.0
                                                      "CRIRE.2022-10-26T07:56:25.958.fits",  # B=4.5
                                                      "CRIRE.2022-10-26T08:14:33.857.fits",  # B=4.0
                                                      "CRIRE.2022-10-26T08:05:42.277.fits",  # B=3.5
                                                      "CRIRE.2022-10-26T07:45:45.398.fits",  # B=3.0
                                                      "CRIRE.2022-10-24T06:43:55.087.fits",  # B=2.5
                                                      "CRIRE.2022-10-24T06:22:02.926.fits",  # B=2.0
                                                      "CRIRE.2022-10-26T07:23:06.300.fits",  # B=1.5
                                                      "CRIRE.2022-10-24T06:04:35.716.fits"]],  # B=1.0
            #                B=3.5  (spec=22  DIT=10.0  NDIT=20)
            [["CRIRE.2022-10-26T08:05:42.277.fits"], ["CRIRE.2022-10-24T06:00:36.335.fits",  # A=1.0
                                                      "CRIRE.2022-10-26T07:19:05.099.fits",  # A=1.5
                                                      "CRIRE.2022-10-24T06:17:44.032.fits",  # A=2.0
                                                      "CRIRE.2022-10-24T06:39:29.485.fits",  # A=2.5
                                                      "CRIRE.2022-10-26T07:41:37.963.fits",  # A=3.0
                                                      "CRIRE.2022-10-26T08:01:34.447.fits",  # A=3.5
                                                      "CRIRE.2022-10-26T08:10:26.674.fits",  # A=4.0
                                                      "CRIRE.2022-10-26T07:52:14.741.fits",  # A=4.5
                                                      "CRIRE.2022-10-24T06:48:26.383.fits",  # A=5.0
                                                      "CRIRE.2022-10-24T06:26:34.004.fits",  # A=5.5
                                                      "CRIRE.2022-10-26T07:32:50.776.fits",  # A=6.0
                                                      "CRIRE.2022-10-24T06:09:01.470.fits"]],  # A=6.5
            #                A=4.0  (spec=23  DIT=10.0  NDIT=20)
            [["CRIRE.2022-10-26T08:10:26.674.fits"], ["CRIRE.2022-10-24T06:13:09.282.fits",  # B=6.5
                                                      "CRIRE.2022-10-26T07:37:01.870.fits",  # B=6.0
                                                      "CRIRE.2022-10-24T06:30:39.815.fits",  # B=5.5
                                                      "CRIRE.2022-10-24T06:52:30.465.fits",  # B=5.0
                                                      "CRIRE.2022-10-26T07:56:25.958.fits",  # B=4.5
                                                      "CRIRE.2022-10-26T08:14:33.857.fits",  # B=4.0
                                                      "CRIRE.2022-10-26T08:05:42.277.fits",  # B=3.5
                                                      "CRIRE.2022-10-26T07:45:45.398.fits",  # B=3.0
                                                      "CRIRE.2022-10-24T06:43:55.087.fits",  # B=2.5
                                                      "CRIRE.2022-10-24T06:22:02.926.fits",  # B=2.0
                                                      "CRIRE.2022-10-26T07:23:06.300.fits",  # B=1.5
                                                      "CRIRE.2022-10-24T06:04:35.716.fits"]],  # B=1.0
            #                B=4.0  (spec=24  DIT=10.0  NDIT=20)
            [["CRIRE.2022-10-26T08:14:33.857.fits"], ["CRIRE.2022-10-24T06:00:36.335.fits",  # A=1.0
                                                      "CRIRE.2022-10-26T07:19:05.099.fits",  # A=1.5
                                                      "CRIRE.2022-10-24T06:17:44.032.fits",  # A=2.0
                                                      "CRIRE.2022-10-24T06:39:29.485.fits",  # A=2.5
                                                      "CRIRE.2022-10-26T07:41:37.963.fits",  # A=3.0
                                                      "CRIRE.2022-10-26T08:01:34.447.fits",  # A=3.5
                                                      "CRIRE.2022-10-26T08:10:26.674.fits",  # A=4.0
                                                      "CRIRE.2022-10-26T07:52:14.741.fits",  # A=4.5
                                                      "CRIRE.2022-10-24T06:48:26.383.fits",  # A=5.0
                                                      "CRIRE.2022-10-24T06:26:34.004.fits",  # A=5.5
                                                      "CRIRE.2022-10-26T07:32:50.776.fits",  # A=6.0
                                                      "CRIRE.2022-10-24T06:09:01.470.fits"]]]  # A=6.5

    def get_science_frames_OLD(self):
        """
        This was auto-generated with step_make_combinations()
        """
#                A=1.0
        return  [[["CRIRE.2022-10-24T06:00:36.335.fits"], ["CRIRE.2022-10-24T07:12:44.231.fits"]],  # tet02OriA
#                B=1.0
                 [["CRIRE.2022-10-24T06:04:35.716.fits"], ["CRIRE.2022-10-24T07:10:18.640.fits"]],  # A=6.5
#                A=6.5
                 [["CRIRE.2022-10-24T06:09:01.470.fits"], ["CRIRE.2022-10-24T07:07:40.240.fits"]],  # B=1.0
#                B=6.5
                 [["CRIRE.2022-10-24T06:13:09.282.fits"], ["CRIRE.2022-10-24T07:05:20.323.fits"]],    # A=1.0
#                A=2.0
                 [["CRIRE.2022-10-24T06:17:44.032.fits"], ["CRIRE.2022-10-24T07:23:17.089.fits"]],  # A=6.5
#                B=2.0
                 [["CRIRE.2022-10-24T06:22:02.926.fits"], ["CRIRE.2022-10-24T07:20:48.548.fits"]],  # A=6.5
#                A=5.5
                 [["CRIRE.2022-10-24T06:26:34.004.fits"], ["CRIRE.2022-10-24T07:18:05.776.fits"]],  # A=1.5
#                B=5.5
                 [["CRIRE.2022-10-24T06:30:39.815.fits"], ["CRIRE.2022-10-24T07:15:40.165.fits"]],  # A=6.5
#                A=2.5
                 [["CRIRE.2022-10-24T06:39:29.485.fits"], ["CRIRE.2022-10-24T07:33:44.524.fits"]],  # A=6.5
#                B=2.5
                 [["CRIRE.2022-10-24T06:43:55.087.fits"], ["CRIRE.2022-10-24T07:31:18.386.fits"]],  # A=6.5
#                A=5.0
                 [["CRIRE.2022-10-24T06:48:26.383.fits"], ["CRIRE.2022-10-24T07:28:39.373.fits"]],  # A=1.0
#                B=5.0
                 [["CRIRE.2022-10-24T06:52:30.465.fits"], ["CRIRE.2022-10-24T07:26:13.455.fits"]],  # A=6.5
#                A=1.5
                 [["CRIRE.2022-10-26T07:19:05.099.fits"], ["CRIRE.2022-10-24T07:12:44.231.fits"]],  # A=6.5
#                B=1.5
                 [["CRIRE.2022-10-26T07:23:06.300.fits"], ["CRIRE.2022-10-24T07:10:18.640.fits"]],  # A=6.5
#                A=6.0
                 [["CRIRE.2022-10-26T07:32:50.776.fits"], ["CRIRE.2022-10-24T07:07:40.240.fits"]],  # A=2.0
#                B=6.0
                 [["CRIRE.2022-10-26T07:37:01.870.fits"], ["CRIRE.2022-10-24T07:05:20.323.fits"]],  # A=6.5
#                A=3.0
                 [["CRIRE.2022-10-26T07:41:37.963.fits"], ["CRIRE.2022-10-24T07:44:17.504.fits"]],  # B=1.0
#                B=3.0
                 [["CRIRE.2022-10-26T07:45:45.398.fits"], ["CRIRE.2022-10-24T07:41:51.559.fits"]],  # A=6.5
#                A=4.5
                 [["CRIRE.2022-10-26T07:52:14.741.fits"], ["CRIRE.2022-10-24T07:55:03.311.fits"]],  # B=1.0
#                B=4.5
                 [["CRIRE.2022-10-26T07:56:25.958.fits"], ["CRIRE.2022-10-24T07:36:33.491.fits"]],  # A=6.5
#                A=3.5
                 [["CRIRE.2022-10-26T08:01:34.447.fits"], ["CRIRE.2022-10-24T07:55:03.311.fits"]],  # B=1.0
#                B=3.5
                 [["CRIRE.2022-10-26T08:05:42.277.fits"], ["CRIRE.2022-10-24T07:52:37.385.fits"]],  # A=6.5
#                A=4.0
                 [["CRIRE.2022-10-26T08:10:26.674.fits"], ["CRIRE.2022-10-24T07:49:45.773.fits"]],  # B=1.0
#                B=4.0
                 [["CRIRE.2022-10-26T08:14:33.857.fits"], ["CRIRE.2022-10-24T07:41:51.559.fits"]]]  # A=6.5

    def is_frame_in_set(self, frnum, comb_set):
        if comb_set < 0:
            return True
        frame_in_set = False
        if comb_set == 0:
            if frnum in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
                frame_in_set = True
        elif comb_set == 1:
            if frnum in [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
                frame_in_set = True
        return frame_in_set

    def is_frame_masked(self, frnum):
        masked = False
        if frnum in [25, 27]:
            masked = True
        return masked

    def get_scale(self, idx):
        scale = 1
        if idx in []:
            exptime = self.get_exptime(idx)[1]
            scale = self.get_exptime(0)[1] / exptime
        print("Scale = ", scale)
        return scale

    def get_flat_frames(self):
        return ["CRIRE.2022-10-24T12:16:57.426.fits",
                "CRIRE.2022-10-24T12:19:47.093.fits",
                "CRIRE.2022-10-24T12:22:36.765.fits",
                "CRIRE.2022-10-24T12:25:26.444.fits",
                "CRIRE.2022-10-24T12:28:16.078.fits",
                "CRIRE.2022-10-24T12:31:05.749.fits",
                "CRIRE.2022-10-24T12:33:55.416.fits",
                "CRIRE.2022-10-24T12:36:45.086.fits",
                "CRIRE.2022-10-24T12:39:34.764.fits",
                "CRIRE.2022-10-24T12:42:24.432.fits",
                "CRIRE.2022-10-24T12:45:14.094.fits",
                "CRIRE.2022-10-24T12:48:03.767.fits",
                "CRIRE.2022-10-24T12:50:53.443.fits",
                "CRIRE.2022-10-24T12:53:43.108.fits",
                "CRIRE.2022-10-24T12:56:32.779.fits",
                "CRIRE.2022-10-24T12:59:22.447.fits",
                "CRIRE.2022-10-24T13:02:12.114.fits",
                "CRIRE.2022-10-24T13:05:01.782.fits",
                "CRIRE.2022-10-24T13:07:51.458.fits",
                "CRIRE.2022-10-24T13:10:41.130.fits",
                "CRIRE.2022-10-24T13:13:30.806.fits",
                "CRIRE.2022-10-24T13:16:20.477.fits",
                "CRIRE.2022-10-24T13:19:10.142.fits",
                "CRIRE.2022-10-24T13:21:59.815.fits",
                "CRIRE.2022-10-24T13:24:49.481.fits"]

    def get_dark_frames(self):
        # Group dark files with different exposure times
        return [["CRIRE.2022-10-22T10:50:21.494.fits","CRIRE.2022-10-22T10:50:44.941.fits","CRIRE.2022-10-22T10:51:08.426.fits"],#5s
                ["CRIRE.2022-10-22T15:50:32.090.fits", "CRIRE.2022-10-22T15:51:10.599.fits", "CRIRE.2022-10-22T15:51:49.070.fits"],#10s
                ["CRIRE.2022-10-26T11:03:01.530.fits","CRIRE.2022-10-26T11:03:31.004.fits","CRIRE.2022-10-26T11:04:00.460.fits"],#120s
                ["CRIRE.2022-10-22T10:41:32.754.fits", "CRIRE.2022-10-22T10:43:38.383.fits", "CRIRE.2022-10-22T10:45:43.983.fits"]]#7s
        # self._dark_files = [["CRIRE.2022-10-26T11:03:01.530.fits","CRIRE.2022-10-26T11:03:31.004.fits","CRIRE.2022-10-26T11:04:00.460.fits"]]#7s
        #              ["CRIRE.2022-10-23T09:56:07.724.fits",CRIRE.2022-10-23T09:56:46.200.fits DARK 10.0
        #               ["CRIRE.2022-10-22T10:47:49.622.fits","CRIRE.2022-10-22T10:48:40.271.fits","CRIRE.2022-10-22T10:49:30.864.fits"],#45s
        # return [["CRIRE.2022-10-22T10:41:32.754.fits", "CRIRE.2022-10-22T10:43:38.383.fits", "CRIRE.2022-10-22T10:45:43.983.fits"]]  # 120s
        # self._dark_files = [["CRIRE.2022-10-22T09:53:35.283.fits",
        #               "CRIRE.2022-10-22T09:52:56.793.fits",
        #               "CRIRE.2022-10-22T09:54:13.782.fits",
        #               "CRIRE.2022-10-22T10:13:26.766.fits",
        #               "CRIRE.2022-10-22T15:50:32.090.fits",
        #               "CRIRE.2022-10-22T15:51:10.599.fits",
        #               "CRIRE.2022-10-22T10:14:43.746.fits",
        #               "CRIRE.2022-10-22T10:14:05.264.fits",
        #               "CRIRE.2022-10-23T09:36:16.909.fits",
        #               "CRIRE.2022-10-22T15:51:49.070.fits",
        #               "CRIRE.2022-10-23T09:36:55.391.fits",
        #               "CRIRE.2022-10-23T09:37:33.865.fits",
        #               "CRIRE.2022-10-23T09:57:24.705.fits",
        #               "CRIRE.2022-10-23T09:56:46.200.fits",
        #               "CRIRE.2022-10-23T09:56:07.724.fits"]] #10s

    def get_arc_frames(self):
        return ["CRIRE.2022-10-22T10:30:27.878.fits"]

    def get_exptime(self, idx):
        ndit = self.get_ndit(idx)
        if idx in []:
            etim = 7  # This is the DIT
        else:
            etim = 10  # This is the DIT
        exptime = etim * ndit
        return exptime, etim

    def get_ndit(self, idx):
        if idx in []:
            return 9  # This is the NDIT
        else:
            return 20  # This is the NDIT

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
            return [850.0, 1574.0], [1707.0, 2000.0]
        else:
            # Part of the object profile
            return [850.0, 1574.0], [1707.0, 2000.0]

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

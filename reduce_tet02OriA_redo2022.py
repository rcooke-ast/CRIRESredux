from reduce_base import ReduceBase
import numpy as np


def main():
    # Initialise the reduce class
    step = 0
    thisred = Reduce(prefix="tet02OriA", match_name="tet02 Ori A", data_folder="Raw/",
                     use_diff=True,
                     step_listfiles=False, step_make_combinations=False,
                     step_pattern=False,  # Generate an image of the detector pattern
                     step_makedarkfit=False, step_makedarkframe=False,  # Make a dark image
                     step_makeflat=False,  # Make a flatfield image
                     step_makearc=False,  # Make an arc image
                     step_makediff=False,  # Make difference and sum images
                     step_makecuts=False,  # Make difference and sum images
                     step_trace=False, step_extract=False, step_basis=(step==0),
                     ext_sky=False,  # Trace the spectrum and extract
                     step_wavecal_prelim=(step==1),  # Calculate a preliminary wavelength calibration solution
                     step_prepALIS=(step==1),
                     # Once the data are reduced, prepare a series of files to be used to fit the wavelength solution with ALIS
                     step_combspec=False, step_combspec_rebin=(step==2),
                     # First get the corrected data from ALIS, and then combine all exposures with this step.
                     step_wavecal_sky=False, step_comb_sky=False,
                     # Wavelength calibrate all sky spectra and then combine
                     step_sample_NumExpCombine=False)  # Combine a different number of exposures to estimate how S/N depends on the number of exposures combined.
    thisred.makePaths(redux_path="/Users/rcooke/Work/Research/BBN/helium34/Absorption/2023_CRIRES_Survey/tet02oriA/2022/")
    thisred._plotit = False
    thisred._comb_set = -1
    thisred.run()


class Reduce(ReduceBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Change some of the default parameters
        self._nbasis = 3  # Number of basis functions to use for the continuum
        self._numcomp = 1
        self._scalevariance = [10827.0, 10829.5]  # Scale the variance to match the measured variance in these regions
        self._scale_errors = True  # Scale the errors by 10x in regions with low flux. This is only used for fitting the wavelength solution with ALIS. The errors are scaled back to their extraction values during the combination.

    def get_science_frames(self):
        #                A=1.0  (spec=0  DIT=5.0  NDIT=20)
        return [[["CRIRE.2022-10-24T07:05:20.323.fits"], ["CRIRE.2022-10-24T07:12:44.231.fits"]],  # B=6.5
        #                B=1.0  (spec=1  DIT=5.0  NDIT=20)
        [["CRIRE.2022-10-24T07:07:40.240.fits"], ["CRIRE.2022-10-24T07:10:18.640.fits"]],  # A=6.5
        #                A=6.5  (spec=2  DIT=5.0  NDIT=20)
        [["CRIRE.2022-10-24T07:10:18.640.fits"], ["CRIRE.2022-10-24T07:07:40.240.fits"]],  # B=1.0
        #                B=6.5  (spec=3  DIT=5.0  NDIT=20)
        [["CRIRE.2022-10-24T07:12:44.231.fits"], ["CRIRE.2022-10-24T07:05:20.323.fits"]],  # A=1.0
        #                A=2.0  (spec=4  DIT=5.0  NDIT=20)
        [["CRIRE.2022-10-24T07:15:40.165.fits"], ["CRIRE.2022-10-24T07:23:17.089.fits"]],  # B=5.5
        #                B=2.0  (spec=5  DIT=5.0  NDIT=20)
        [["CRIRE.2022-10-24T07:18:05.776.fits"], ["CRIRE.2022-10-24T07:20:48.548.fits"]],  # A=5.5
        #                A=5.5  (spec=6  DIT=5.0  NDIT=20)
        [["CRIRE.2022-10-24T07:20:48.548.fits"], ["CRIRE.2022-10-24T07:18:05.776.fits"]],  # B=2.0
        #                B=5.5  (spec=7  DIT=5.0  NDIT=20)
        [["CRIRE.2022-10-24T07:23:17.089.fits"], ["CRIRE.2022-10-24T07:15:40.165.fits"]],  # A=2.0
        #                A=2.5  (spec=8  DIT=5.0  NDIT=20)
        [["CRIRE.2022-10-24T07:26:13.455.fits"], ["CRIRE.2022-10-24T07:33:44.524.fits"]],  # B=5.0
        #                B=2.5  (spec=9  DIT=5.0  NDIT=20)
        [["CRIRE.2022-10-24T07:28:39.373.fits"], ["CRIRE.2022-10-24T07:31:18.386.fits"]],  # A=5.0
        #                A=5.0  (spec=10  DIT=5.0  NDIT=20)
        [["CRIRE.2022-10-24T07:31:18.386.fits"], ["CRIRE.2022-10-24T07:28:39.373.fits"]],  # B=2.5
        #                B=5.0  (spec=11  DIT=5.0  NDIT=20)
        [["CRIRE.2022-10-24T07:33:44.524.fits"], ["CRIRE.2022-10-24T07:26:13.455.fits"]],  # A=2.5
        #                A=3.0  (spec=12  DIT=5.0  NDIT=20)
        [["CRIRE.2022-10-24T07:36:33.491.fits"], ["CRIRE.2022-10-24T07:44:17.504.fits"]],  # B=4.5
        #                B=3.0  (spec=13  DIT=5.0  NDIT=20)
        [["CRIRE.2022-10-24T07:38:59.311.fits"], ["CRIRE.2022-10-24T07:41:51.559.fits"]],  # A=4.5
        #                A=4.5  (spec=14  DIT=5.0  NDIT=20)
        [["CRIRE.2022-10-24T07:41:51.559.fits"], ["CRIRE.2022-10-24T07:55:03.311.fits"]],  # B=4.0
        #                B=4.5  (spec=15  DIT=5.0  NDIT=20)
        [["CRIRE.2022-10-24T07:44:17.504.fits"], ["CRIRE.2022-10-24T07:36:33.491.fits"]],  # A=3.0
        #                A=3.5  (spec=16  DIT=5.0  NDIT=20)
        [["CRIRE.2022-10-24T07:47:20.982.fits"], ["CRIRE.2022-10-24T07:55:03.311.fits"]],  # B=4.0
        #                B=3.5  (spec=17  DIT=5.0  NDIT=20)
        [["CRIRE.2022-10-24T07:49:45.773.fits"], ["CRIRE.2022-10-24T07:52:37.385.fits"]],  # A=4.0
        #                A=4.0  (spec=18  DIT=5.0  NDIT=20)
        [["CRIRE.2022-10-24T07:52:37.385.fits"], ["CRIRE.2022-10-24T07:49:45.773.fits"]],  # B=3.5
        #                B=4.0  (spec=19  DIT=5.0  NDIT=20)
        [["CRIRE.2022-10-24T07:55:03.311.fits"], ["CRIRE.2022-10-24T07:41:51.559.fits"]],  # A=4.5
        #                A=6.0  (spec=20  DIT=15.0  NDIT=3)
        [["CRIRE.2022-10-24T07:58:20.868.fits"], ["CRIRE.2022-10-24T07:59:33.417.fits"]],  # B=6.0
        #                B=6.0  (spec=21  DIT=15.0  NDIT=3)
        [["CRIRE.2022-10-24T07:59:33.417.fits"], ["CRIRE.2022-10-24T07:58:20.868.fits"]]]  # A=6.0

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
        #return [["CRIRE.2022-08-10T11:17:33.973.fits", "CRIRE.2022-08-10T11:18:27.477.fits", "CRIRE.2022-08-10T11:19:20.965.fits", "CRIRE.2022-08-10T12:34:13.900.fits", "CRIRE.2022-08-10T12:35:07.381.fits", "CRIRE.2022-08-10T12:36:00.860.fits"]]
        return [["CRIRE.2022-10-22T10:50:21.494.fits","CRIRE.2022-10-22T10:50:44.941.fits","CRIRE.2022-10-22T10:51:08.426.fits"],#5s
                ["CRIRE.2022-10-22T15:50:32.090.fits", "CRIRE.2022-10-22T15:51:10.599.fits", "CRIRE.2022-10-22T15:51:49.070.fits"],#10s
                ["CRIRE.2022-10-26T11:03:01.530.fits","CRIRE.2022-10-26T11:03:31.004.fits","CRIRE.2022-10-26T11:04:00.460.fits"],#120s
                ["CRIRE.2022-10-22T10:41:32.754.fits", "CRIRE.2022-10-22T10:43:38.383.fits", "CRIRE.2022-10-22T10:45:43.983.fits"]]#7s

    def get_arc_frames(self):
        return ["CRIRE.2022-10-22T10:30:27.878.fits"]

    def get_exptime(self, idx):
        ndit = self.get_ndit(idx)
        if idx in [20, 21]:
            etim = 15  # This is the DIT
        else:
            etim = 5  # This is the DIT
        exptime = etim * ndit
        return exptime, etim

    def get_ndit(self, idx):
        if idx == [20, 21]:
            return 3  # This is the NDIT
        else:
            return 20  # This is the NDIT

    def get_objprof_limits(self, full=True):
        """
        Set the spectral regions to calculate the object profile. If full=True, then a more extended region is used.
        These values are relevant for HD319718

        To determine these values, open up the two frames with the biggest difference in nod (e.g. +/- 6.5") in ds9,
        and hover the cursor over the middle of the strongest He I* absorption line. The inner left limit is the pixel
        number at the middle of the profile minus 90 pixels, and the right limit is the pixel number at the middle of
        the profile plus 45 pixels. The outer limits need to be large enough to be able to model the full object
        profile in 2D.
        """
        if full:
            # All of the object profile
            return [850.0, 1560.0], [1705.0, 2000.0]
        else:
            # Part of the object profile
            return [850.0, 1560.0], [1705.0, 2000.0]
            # return [1410.0, 1700.0], [1820.0, 1940.0]

    def print_SNregions(self, arr):
        """ Print the S/N in certain regions of the spectrum
        These values are relevant for HD319718
        """
        print("(box) S/N = ", np.mean(arr[1455:1525]) / np.std(arr[1455:1525]))
        print("(box) S/N ab = ", np.mean(arr[1750:1770]) / np.std(arr[1750:1770]))

    def get_SNregions_fit(self, flux):
        """ Print the S/N in certain regions of the spectrum
        These values are relevant for HD319718
        """
        xfit = np.arange(1455, 1525)
        ww = (xfit,)
        modl = np.polyval(np.polyfit(xfit, flux[ww], 2), xfit)
        SN_spec = 1.0 / np.std(flux[ww] / modl)
        xfit = np.arange(1750, 1770)
        ww = (xfit,)
        modl = np.polyval(np.polyfit(xfit, flux[ww], 2), xfit)
        SN_abs = 1.0 / np.std(flux[ww] / modl)
        return SN_spec, SN_abs

if __name__ == '__main__':
    main()

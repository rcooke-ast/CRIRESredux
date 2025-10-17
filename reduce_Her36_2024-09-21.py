from reduce_base import ReduceBase
import numpy as np


def main():
    # Initialise the reduce class
    step = 2
    thisred = Reduce(prefix="her36", match_name="Her 36", data_folder="Raw/",
                     use_diff=True,
                     step_listfiles=False, step_make_combinations=False,
                     step_pattern=False,  # Generate an image of the detector pattern
                     step_makedarkfit=False, step_makedarkframe=False,  # Make a dark image
                     step_makeflat=False,  # Make a flatfield image
                     step_makearc=False,  # Make an arc image
                     step_makediff=False,  # Make difference and sum images
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
    thisred.makePaths(redux_path="/Users/rcooke/Work/Research/BBN/helium34/Absorption/2023_CRIRES_Survey/Her36/2024-09-21/")
    thisred._plotit = False
    thisred._comb_set = -1
    thisred.run()


class Reduce(ReduceBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Change some of the default parameters
        self._nbasis = 3
        self._numcomp = 1
        self._scalevariance = [10828.0, 10829.3]  # Scale the variance to match the measured variance in these regions
        self._scale_errors = True  # Scale the errors by 10x in regions with low flux. This is only used for fitting the wavelength solution with ALIS. The errors are scaled back to their extraction values during the combination.

    def get_science_frames(self):
        """
        """
        #                A=1.0  (spec=12  DIT=180.0  NDIT=1)
        return [
        [["CRIRE.2024-09-22T00:54:03.383.fits"], ["CRIRE.2024-09-22T01:07:15.258.fits"]],  # B=6.5
        #                B=1.0  (spec=13  DIT=180.0  NDIT=1)
        [["CRIRE.2024-09-22T00:58:17.840.fits"], ["CRIRE.2024-09-22T01:02:44.218.fits"]],  # A=6.5
        #                A=6.5  (spec=14  DIT=180.0  NDIT=1)
        [["CRIRE.2024-09-22T01:02:44.218.fits"], ["CRIRE.2024-09-22T00:58:17.840.fits"]],  # B=1.0
        #                B=6.5  (spec=15  DIT=180.0  NDIT=1)
        [["CRIRE.2024-09-22T01:07:15.258.fits"], ["CRIRE.2024-09-22T00:54:03.383.fits"]],  # A=1.0
        #                A=2.0  (spec=16  DIT=180.0  NDIT=1)
        #######################
        [["CRIRE.2024-09-22T01:11:58.708.fits"], ["CRIRE.2024-09-22T01:16:14.343.fits"]],  # B=5.5
        #                B=2.0  (spec=17  DIT=180.0  NDIT=1)
        [["CRIRE.2024-09-22T01:25:17.945.fits"], ["CRIRE.2024-09-22T01:20:45.546.fits"]],  # A=5.5
        #                A=5.5  (spec=18  DIT=180.0  NDIT=1)
        [["CRIRE.2024-09-22T01:20:45.546.fits"], ["CRIRE.2024-09-22T01:25:17.945.fits"]],  # B=2.0
        #                B=5.5  (spec=19  DIT=180.0  NDIT=1)
        [["CRIRE.2024-09-22T01:16:14.343.fits"], ["CRIRE.2024-09-22T01:11:58.708.fits"]],  # A=2.0
        #######################
        #                A=2.5  (spec=20  DIT=180.0  NDIT=1)
        [["CRIRE.2024-09-22T01:29:49.498.fits"], ["CRIRE.2024-09-22T01:43:04.639.fits"]],  # B=5.0
        #                B=2.5  (spec=21  DIT=180.0  NDIT=1)
        [["CRIRE.2024-09-22T01:34:16.888.fits"], ["CRIRE.2024-09-22T01:38:44.479.fits"]],  # A=5.0
        #                A=5.0  (spec=22  DIT=180.0  NDIT=1)
        [["CRIRE.2024-09-22T01:38:44.479.fits"], ["CRIRE.2024-09-22T01:34:16.888.fits"]],  # B=2.5
        #                B=5.0  (spec=23  DIT=180.0  NDIT=1)
        [["CRIRE.2024-09-22T01:43:04.639.fits"], ["CRIRE.2024-09-22T01:29:49.498.fits"]]]  # A=2.5

    def get_flat_frames(self):
        return ["CRIRE.2024-09-22T10:49:48.683.fits",
                "CRIRE.2024-09-22T10:50:09.057.fits",
                "CRIRE.2024-09-22T10:50:29.430.fits",
                "CRIRE.2024-09-22T10:50:49.804.fits",
                "CRIRE.2024-09-22T10:51:10.178.fits"]

    def get_dark_frames(self):
        # Group dark files with different exposure times
        #return [["CRIRE.2022-08-10T11:17:33.973.fits", "CRIRE.2022-08-10T11:18:27.477.fits", "CRIRE.2022-08-10T11:19:20.965.fits", "CRIRE.2022-08-10T12:34:13.900.fits", "CRIRE.2022-08-10T12:35:07.381.fits", "CRIRE.2022-08-10T12:36:00.860.fits"]]
        return [["CRIRE.2024-09-22T11:12:46.083.fits","CRIRE.2024-09-22T11:13:09.438.fits","CRIRE.2024-09-22T11:13:32.787.fits"],#5s
                ["CRIRE.2024-09-22T11:11:55.545.fits","CRIRE.2024-09-22T11:11:05.036.fits","CRIRE.2024-09-22T11:10:14.540.fits"],#45s
                ["CRIRE.2024-09-22T11:08:09.014.fits","CRIRE.2024-09-22T11:06:03.492.fits","CRIRE.2024-09-22T11:03:58.028.fits"]]#120s

    def get_arc_frames(self):
        return ["CRIRE.2024-09-22T10:55:25.366.fits"] # This should be FPET

    def get_trace(self, idx):
        return self._diff_name.format(0)

    def get_exptime(self, idx):
        ndit = self.get_ndit(idx)
        etim = 240.0
        # if idx in [20, 21]:
        #     etim = 15  # This is the DIT
        # else:
        #     etim = 180  # This is the DIT
        exptime = etim * ndit
        return exptime, etim

    def get_ndit(self, idx):
        return 1
        # if idx == [20, 21]:
        #     return 3  # This is the NDIT
        # else:
        #     return 20  # This is the NDIT

    def get_objprof_limits(self, full=True):
        """
        Set the spectral regions to calculate the object profile. If full=True, then a more extended region is used.
        These values are relevant for Her 36

        To determine these values, open up the two frames with the biggest difference in nod (e.g. +/- 6.5") in ds9,
        and hover the cursor over the middle of the strongest He I* absorption line. The inner left limit is the pixel
        number at the middle of the profile minus 90 pixels, and the right limit is the pixel number at the middle of
        the profile plus 45 pixels. The outer limits need to be large enough to be able to model the full object
        profile in 2D.
        """
        if full:
            # All of the object profile
            return [850.0, 1630.0], [1750.0, 2000.0]
        else:
            # Part of the object profile
            return [850.0, 1630.0], [1750.0, 2000.0]
            # return [1410.0, 1700.0], [1820.0, 1940.0]

    def print_SNregions(self, arr):
        """ Print the S/N in certain regions of the spectrum
        These values are relevant for HD319718
        """
        print("(box) S/N = ", np.mean(arr[1550:1605]) / np.std(arr[1550:1605]))
        print("(box) S/N ab = ", np.mean(arr[1750:1770]) / np.std(arr[1750:1770]))

    def get_SNregions_fit(self, flux):
        """ Print the S/N in certain regions of the spectrum
        These values are relevant for Her 36
        """
        xfit = np.arange(1550, 1605)
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

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
    thisred.makePaths(redux_path="/Users/rcooke/Work/Research/BBN/helium34/Absorption/2023_CRIRES_Survey/WRAY_15-199/2024-04-20/")
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
        self._use_dark = True

    def get_science_frames(self):
        """
        This was auto-generated with step_make_combinations()
        """
        #                A=1.0  (spec=0  DIT=240.0  NDIT=1)
        return [[["CRIRE.2024-04-21T00:14:13.696.fits"], ["CRIRE.2024-04-21T00:22:58.496.fits"]],  # A=6.5
        #                B=1.0  (spec=1  DIT=240.0  NDIT=1)
        [["CRIRE.2024-04-21T00:18:25.978.fits"], ["CRIRE.2024-04-21T00:22:58.496.fits"]],  # A=6.5
        #                A=6.5  (spec=2  DIT=240.0  NDIT=1)
        [["CRIRE.2024-04-21T00:22:58.496.fits"], ["CRIRE.2024-04-21T00:18:25.978.fits",  # B=1.0
                                                  "CRIRE.2024-04-21T00:14:13.696.fits"]]]  # A=1.0

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
        return ["CRIRE.2024-04-20T16:24:56.536.fits",
                "CRIRE.2024-04-20T16:25:16.905.fits",
                "CRIRE.2024-04-20T16:25:37.276.fits",
                "CRIRE.2024-04-20T16:25:57.641.fits",
                "CRIRE.2024-04-20T16:26:18.004.fits"]

    def get_dark_frames(self):
        """
        """
        # Group dark files with different exposure times
        return [["CRIRE.2024-04-20T16:45:20.199.fits","CRIRE.2024-04-20T16:45:43.574.fits","CRIRE.2024-04-20T16:46:06.891.fits"],#5s
                ["CRIRE.2024-04-20T16:36:32.146.fits","CRIRE.2024-04-20T16:38:37.680.fits","CRIRE.2024-04-20T16:40:43.193.fits"]]#120s

    def get_arc_frames(self):
        return ["CRIRE.2024-04-20T16:30:31.929.fits"]

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

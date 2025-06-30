from reduce_base import ReduceBase
import numpy as np


def main():
    # Initialise the reduce class
    step1 = True
    step2 = not step1
    # step1, step2 = False, False
    thisred = Reduce(prefix="hd319718", match_name="HD 319718", data_folder="Raw/",
                     use_diff=True,
                     step_listfiles=False, step_make_combinations=False,
                     step_pattern=False,  # Generate an image of the detector pattern
                     step_makedarkfit=False, step_makedarkframe=False,  # Make a dark image
                     step_makeflat=False,  # Make a flatfield image
                     step_makearc=False,  # Make an arc image
                     step_makediff=False,  # Make difference and sum images
                     step_makecuts=False,  # Make difference and sum images
                     step_trace=False, step_extract=False, step_basis=False,#step1,
                     ext_sky=False,  # Trace the spectrum and extract
                     step_wavecal_prelim=step1,  # Calculate a preliminary wavelength calibration solution
                     step_prepALIS=step1,
                     # Once the data are reduced, prepare a series of files to be used to fit the wavelength solution with ALIS
                     step_combspec=False, step_combspec_rebin=step2,
                     # First get the corrected data from ALIS, and then combine all exposures with this step.
                     step_wavecal_sky=False, step_comb_sky=False,
                     # Wavelength calibrate all sky spectra and then combine
                     step_sample_NumExpCombine=False)  # Combine a different number of exposures to estimate how S/N depends on the number of exposures combined.
    thisred.makePaths(redux_path="/Users/rcooke/Work/Research/BBN/helium34/Absorption/2023_CRIRES_Survey/HD319718/2024-08-24/")
    thisred._plotit = False
    thisred._comb_set = 0
    thisred.run()


class Reduce(ReduceBase):

    def get_science_frames(self):
        return [
            # Raw/CRIRE.2024-08-24T01:41:43.725.fits 1 A 3.0 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-24T01:46:09.043.fits 1 B 3.0 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-24T01:50:51.516.fits 1 A 4.5 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-24T01:55:16.872.fits 1 B 4.5 240.0 HD 319718 2.15
            #                A=3.0  (spec=12  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-24T01:41:43.725.fits"], ["CRIRE.2024-08-24T01:55:16.872.fits"]],  # B=4.5
            #                B=3.0  (spec=13  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-24T01:46:09.043.fits"], ["CRIRE.2024-08-24T01:50:51.516.fits"]],  # A=4.5
            #                A=4.5  (spec=14  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-24T01:50:51.516.fits"], ["CRIRE.2024-08-24T01:46:09.043.fits"]],  # B=3.0
            #                B=4.5  (spec=15  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-24T01:55:16.872.fits"], ["CRIRE.2024-08-24T01:41:43.725.fits"]],  # A=3.0
            # Raw/CRIRE.2024-08-24T01:59:49.598.fits 1 A 3.5 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-24T02:04:09.684.fits 1 B 3.5 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-24T02:08:35.187.fits 1 A 4.0 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-24T02:12:58.052.fits 1 B 4.0 240.0 HD 319718 2.15
            #                A=3.5  (spec=16  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-24T01:59:49.598.fits"], ["CRIRE.2024-08-24T02:12:58.052.fits"]],  # B=4.0
            #                B=3.5  (spec=17  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-24T02:04:09.684.fits"], ["CRIRE.2024-08-24T02:08:35.187.fits"]],  # A=4.0
            #                A=4.0  (spec=18  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-24T02:08:35.187.fits"], ["CRIRE.2024-08-24T02:04:09.684.fits"]],  # B=3,5
            #                B=4.0  (spec=19  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-24T02:12:58.052.fits"], ["CRIRE.2024-08-24T01:59:49.598.fits"]],  # A=3.5
            # Raw/CRIRE.2024-08-24T02:17:24.848.fits 1 A 1.5 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-24T02:21:42.887.fits 1 B 1.5 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-24T02:26:20.454.fits 1 A 6.0 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-24T02:30:46.047.fits 1 B 6.0 240.0 HD 319718 2.15
            #                A=1.5  (spec=20  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-24T02:17:24.848.fits"], ["CRIRE.2024-08-24T02:30:46.047.fits"]],  # B=6.0
            #                B=1.5  (spec=21  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-24T02:21:42.887.fits"], ["CRIRE.2024-08-24T02:26:20.454.fits"]],  # A=6.0
            #                A=6.0  (spec=22  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-24T02:26:20.454.fits"], ["CRIRE.2024-08-24T02:21:42.887.fits"]],  # B=1.5
            #                B=6.0  (spec=23  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-24T02:30:46.047.fits"], ["CRIRE.2024-08-24T02:17:24.848.fits"]]]  # A=1.5

    def get_flat_frames(self):
        return ["CRIRE.2024-08-24T13:05:26.590.fits",
                "CRIRE.2024-08-24T13:05:46.961.fits",
                "CRIRE.2024-08-24T13:06:07.331.fits",
                "CRIRE.2024-08-24T13:06:27.701.fits",
                "CRIRE.2024-08-24T13:06:48.071.fits"]

    def get_dark_frames(self):
        # Group dark files with different exposure times
        return [["CRIRE.2024-08-21T05:27:37.468.fits", "CRIRE.2024-08-21T05:28:00.820.fits", "CRIRE.2024-08-21T05:28:24.165.fits"],#5s
                ["CRIRE.2024-08-24T13:19:35.883.fits", "CRIRE.2024-08-24T13:21:41.395.fits", "CRIRE.2024-08-24T13:23:46.907.fits"]]#120s

    def get_arc_frames(self):
        return ["CRIRE.2024-08-24T13:11:03.426.fits"]

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
        These values are relevant for HD319718

        To determine these values, open up the two frames with the biggest difference in nod (e.g. +/- 6.5") in ds9,
        and hover the cursor over the middle of the strongest He I* absorption line. The inner left limit is the pixel
        number at the middle fo the profile minus 90 pixels, and the right limit is the pixel number at the middle of
        the profile plus 45 pixels. The outer limits need to be large enough to be able to model the full object
        profile in 2D.
        """
        if full:
            # All of the object profile
            return [850.0, 1600.0], [1755.0, 2000.0]
        else:
            # Part of the object profile
            return [850.0, 1600.0], [1755.0, 2000.0]
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

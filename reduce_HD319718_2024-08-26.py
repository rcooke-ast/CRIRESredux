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
    thisred.makePaths(redux_path="/Users/rcooke/Work/Research/BBN/helium34/Absorption/2023_CRIRES_Survey/HD319718/2024-08-26/")
    thisred._plotit = False
    thisred._comb_set = 0
    thisred.run()


class Reduce(ReduceBase):

    def get_science_frames(self):
        return [
            # Raw/CRIRE.2024-08-27T00:51:15.722.fits 1 A 1.0 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-27T00:55:32.006.fits 1 B 1.0 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-27T01:00:02.157.fits 1 A 6.5 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-27T01:04:38.353.fits 1 B 6.5 240.0 HD 319718 2.15
            #                A=1.0  (spec=12  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-27T00:51:15.722.fits"], ["CRIRE.2024-08-27T01:04:38.353.fits"]],  # B=6.5
            #                B=1.0  (spec=13  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-27T00:55:32.006.fits"], ["CRIRE.2024-08-27T01:00:02.157.fits"]],  # A=6.5
            #                A=6.5  (spec=14  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-27T01:00:02.157.fits"], ["CRIRE.2024-08-27T00:55:32.006.fits"]],  # B=1.0
            #                B=6.5  (spec=15  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-27T01:04:38.353.fits"], ["CRIRE.2024-08-27T00:51:15.722.fits"]],  # A=1.0
            # Raw/CRIRE.2024-08-27T01:09:15.019.fits 1 A 2.0 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-27T01:13:30.444.fits 1 B 2.0 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-27T01:18:18.750.fits 1 A 5.5 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-27T01:22:44.112.fits 1 B 5.5 240.0 HD 319718 2.15
            #                A=2.0  (spec=16  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-27T01:09:15.019.fits"], ["CRIRE.2024-08-27T01:22:44.112.fits"]],  # B=5.5
            #                B=2.0  (spec=17  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-27T01:13:30.444.fits"], ["CRIRE.2024-08-27T01:18:18.750.fits"]],  # A=5.5
            #                A=5.5  (spec=18  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-27T01:18:18.750.fits"], ["CRIRE.2024-08-27T01:13:30.444.fits"]],  # B=2.0
            #                B=5.5  (spec=19  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-27T01:22:44.112.fits"], ["CRIRE.2024-08-27T01:09:15.019.fits"]],  # A=2.0
            # Raw/CRIRE.2024-08-27T01:27:29.533.fits 1 A 2.5 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-27T01:31:51.569.fits 1 B 2.5 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-27T01:36:35.940.fits 1 A 5.0 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-27T01:41:05.947.fits 1 B 5.0 240.0 HD 319718 2.15
            #                A=2.5  (spec=20  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-27T01:27:29.533.fits"], ["CRIRE.2024-08-27T01:41:05.947.fits"]],  # B=5.0
            #                B=2.5  (spec=21  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-27T01:31:51.569.fits"], ["CRIRE.2024-08-27T01:36:35.940.fits"]],  # A=5.0
            #                A=5.0  (spec=22  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-27T01:36:35.940.fits"], ["CRIRE.2024-08-27T01:31:51.569.fits"]],  # B=2.5
            #                B=5.0  (spec=23  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-27T01:41:05.947.fits"], ["CRIRE.2024-08-27T01:27:29.533.fits"]],
            # Raw/CRIRE.2024-08-27T01:51:10.927.fits 1 A 3.0 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-27T01:55:31.915.fits 1 B 3.0 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-27T02:00:15.282.fits 1 A 4.5 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-27T02:04:40.909.fits 1 B 4.5 240.0 HD 319718 2.15
            #                A=1.0  (spec=12  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-27T01:51:10.927.fits"], ["CRIRE.2024-08-27T02:04:40.909.fits"]],  # B=6.5
            #                B=1.0  (spec=13  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-27T01:55:31.915.fits"], ["CRIRE.2024-08-27T02:00:15.282.fits"]],  # A=6.5
            #                A=6.5  (spec=14  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-27T02:00:15.282.fits"], ["CRIRE.2024-08-27T01:55:31.915.fits"]],  # B=1.0
            #                B=6.5  (spec=15  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-27T02:04:40.909.fits"], ["CRIRE.2024-08-27T01:51:10.927.fits"]],  # A=1.0
            # Raw/CRIRE.2024-08-27T02:09:11.391.fits 1 A 3.5 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-27T02:13:31.428.fits 1 B 3.5 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-27T02:17:54.742.fits 1 A 4.0 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-27T02:22:21.454.fits 1 B 4.0 240.0 HD 319718 2.15
            #                A=2.0  (spec=16  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-27T02:09:11.391.fits"], ["CRIRE.2024-08-27T02:22:21.454.fits"]],  # B=5.5
            #                B=2.0  (spec=17  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-27T02:13:31.428.fits"], ["CRIRE.2024-08-27T02:17:54.742.fits"]],  # A=5.5
            #                A=5.5  (spec=18  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-27T02:17:54.742.fits"], ["CRIRE.2024-08-27T02:13:31.428.fits"]],  # B=2.0
            #                B=5.5  (spec=19  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-27T02:22:21.454.fits"], ["CRIRE.2024-08-27T02:09:11.391.fits"]],  # A=2.0
            # Raw/CRIRE.2024-08-27T02:26:45.625.fits 1 A 1.5 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-27T02:30:59.794.fits 1 B 1.5 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-27T02:35:27.701.fits 1 A 6.0 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-08-27T02:39:53.604.fits 1 B 6.0 240.0 HD 319718 2.15
            #                A=2.5  (spec=20  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-27T02:26:45.625.fits"], ["CRIRE.2024-08-27T02:39:53.604.fits"]],  # B=5.0
            #                B=2.5  (spec=21  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-27T02:30:59.794.fits"], ["CRIRE.2024-08-27T02:35:27.701.fits"]],  # A=5.0
            #                A=5.0  (spec=22  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-27T02:35:27.701.fits"], ["CRIRE.2024-08-27T02:30:59.794.fits"]],  # B=2.5
            #                B=5.0  (spec=23  DIT=180.0  NDIT=1)
            [["CRIRE.2024-08-27T02:39:53.604.fits"], ["CRIRE.2024-08-27T02:26:45.625.fits"]]]  # A=2.5

    def get_flat_frames(self):
        return ["CRIRE.2024-08-25T13:59:10.675.fits",
                "CRIRE.2024-08-25T13:59:31.046.fits",
                "CRIRE.2024-08-25T13:59:51.412.fits",
                "CRIRE.2024-08-25T14:00:11.789.fits",
                "CRIRE.2024-08-25T14:00:32.161.fits"]

    def get_dark_frames(self):
        # Group dark files with different exposure times
        return [["CRIRE.2024-08-25T14:38:17.253.fits", "CRIRE.2024-08-25T14:38:40.584.fits", "CRIRE.2024-08-25T14:39:03.906.fits"],#5s
                ["CRIRE.2024-08-25T14:35:45.660.fits", "CRIRE.2024-08-25T14:36:36.216.fits", "CRIRE.2024-08-25T14:37:26.744.fits"],#45s
                ["CRIRE.2024-08-25T14:24:39.677.fits", "CRIRE.2024-08-25T14:26:45.200.fits", "CRIRE.2024-08-25T14:28:50.720.fits"]]#120s

    def get_arc_frames(self):
        return ["CRIRE.2024-08-25T14:06:46.545.fits"]

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

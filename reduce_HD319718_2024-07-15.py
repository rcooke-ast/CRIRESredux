from reduce_base import ReduceBase
import numpy as np


def main():
    # Initialise the reduce class
    step1 = False
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
    thisred.makePaths(redux_path="/Users/rcooke/Work/Research/BBN/helium34/Absorption/2023_CRIRES_Survey/HD319718/2024-07-15/")
    thisred._plotit = False
    thisred._comb_set = -1
    thisred.run()


class Reduce(ReduceBase):

    def get_science_frames(self):
        return [
            # Raw/CRIRE.2024-07-16T01:20:27.158.fits 1 A 1.0 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-07-16T01:24:47.364.fits 1 B 1.0 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-07-16T01:29:18.252.fits 1 A 6.5 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-07-16T01:33:43.531.fits 1 B 6.5 240.0 HD 319718 2.15
            #                A=1.0  (spec=12  DIT=180.0  NDIT=1)
            [["CRIRE.2024-07-16T01:20:27.158.fits"], ["CRIRE.2024-07-16T01:33:43.531.fits"]],  # B=6.5
            #                B=1.0  (spec=13  DIT=180.0  NDIT=1)
            [["CRIRE.2024-07-16T01:24:47.364.fits"], ["CRIRE.2024-07-16T01:29:18.252.fits"]],  # A=6.5
            #                A=6.5  (spec=14  DIT=180.0  NDIT=1)
            [["CRIRE.2024-07-16T01:29:18.252.fits"], ["CRIRE.2024-07-16T01:24:47.364.fits"]],  # B=1.0
            #                B=6.5  (spec=15  DIT=180.0  NDIT=1)
            [["CRIRE.2024-07-16T01:33:43.531.fits"], ["CRIRE.2024-07-16T01:20:27.158.fits"]],  # A=1.0
            # Raw/CRIRE.2024-07-16T01:38:20.569.fits 1 A 2.0 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-07-16T01:42:35.663.fits 1 B 2.0 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-07-16T01:47:24.092.fits 1 A 5.5 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-07-16T01:51:46.671.fits 1 B 5.5 240.0 HD 319718 2.15
            #                A=2.0  (spec=16  DIT=180.0  NDIT=1)
            [["CRIRE.2024-07-16T01:38:20.569.fits"], ["CRIRE.2024-07-16T01:51:46.671.fits"]],  # B=5.5
            #                B=2.0  (spec=17  DIT=180.0  NDIT=1)
            [["CRIRE.2024-07-16T01:42:35.663.fits"], ["CRIRE.2024-07-16T01:47:24.092.fits"]],  # A=5.5
            #                A=5.5  (spec=18  DIT=180.0  NDIT=1)
            [["CRIRE.2024-07-16T01:47:24.092.fits"], ["CRIRE.2024-07-16T01:42:35.663.fits"]],  # B=2.0
            #                B=5.5  (spec=19  DIT=180.0  NDIT=1)
            [["CRIRE.2024-07-16T01:51:46.671.fits"], ["CRIRE.2024-07-16T01:38:20.569.fits"]],  # A=2.0
            # Raw/CRIRE.2024-07-16T01:56:18.959.fits 1 A 2.5 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-07-16T02:00:43.402.fits 1 B 2.5 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-07-16T02:05:29.865.fits 1 A 5.0 240.0 HD 319718 2.15
            # Raw/CRIRE.2024-07-16T02:09:52.112.fits 1 B 5.0 240.0 HD 319718 2.15
            #                A=2.5  (spec=20  DIT=180.0  NDIT=1)
            [["CRIRE.2024-07-16T01:56:18.959.fits"], ["CRIRE.2024-07-16T02:09:52.112.fits"]],  # B=5.0
            #                B=2.5  (spec=21  DIT=180.0  NDIT=1)
            [["CRIRE.2024-07-16T02:00:43.402.fits"], ["CRIRE.2024-07-16T02:05:29.865.fits"]],  # A=5.0
            #                A=5.0  (spec=22  DIT=180.0  NDIT=1)
            [["CRIRE.2024-07-16T02:05:29.865.fits"], ["CRIRE.2024-07-16T02:00:43.402.fits"]],  # B=2.5
            #                B=5.0  (spec=23  DIT=180.0  NDIT=1)
            [["CRIRE.2024-07-16T02:09:52.112.fits"], ["CRIRE.2024-07-16T01:56:18.959.fits"]]]  # A=2.5

    def get_flat_frames(self):
        return ["CRIRE.2024-07-16T12:25:08.899.fits",
                "CRIRE.2024-07-16T12:25:29.264.fits",
                "CRIRE.2024-07-16T12:25:49.630.fits",
                "CRIRE.2024-07-16T12:26:09.997.fits",
                "CRIRE.2024-07-16T12:26:30.358.fits",
                "CRIRE.2024-07-16T13:15:22.547.fits",
                "CRIRE.2024-07-16T13:15:42.913.fits",
                "CRIRE.2024-07-16T13:16:03.280.fits",
                "CRIRE.2024-07-16T13:16:23.648.fits",
                "CRIRE.2024-07-16T13:16:44.011.fits"]

    def get_dark_frames(self):
        # Group dark files with different exposure times
        return [["CRIRE.2024-07-16T12:45:12.976.fits", "CRIRE.2024-07-16T12:45:36.320.fits", "CRIRE.2024-07-16T12:45:59.594.fits"],#5s
                ["CRIRE.2024-07-16T13:36:14.496.fits", "CRIRE.2024-07-16T13:37:04.980.fits", "CRIRE.2024-07-16T13:37:55.471.fits"],#45s
                ["CRIRE.2024-07-16T13:29:57.951.fits", "CRIRE.2024-07-16T13:32:03.457.fits", "CRIRE.2024-07-16T13:34:08.981.fits"]]#120s

    def get_arc_frames(self):
        return ["CRIRE.2024-07-16T13:21:25.630.fits"]

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
            # return [1450.0, 1510.0], [1725.0, 1830.0]
            # return [1450.0, 1610.0], [1725.0, 1870.0]
            return [850.0, 1590.0], [1735.0, 2000.0]
        else:
            # Part of the object profile
            # return [1450.0, 1510.0], [1725.0, 1830.0]
            # return [1480.0, 1610.0], [1725.0, 1840.0]
            return [850.0, 1590.0], [1735.0, 2000.0]
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

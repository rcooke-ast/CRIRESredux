from reduce_base import ReduceBase
import numpy as np


def main():
    # Initialise the reduce class
    step1 = True
    step2 = not step1
    thisred = Reduce(prefix="PDS241",
                     use_diff=True,
                     step_listfiles=False,
                     step_pattern=False,  # Generate an image of the detector pattern
                     step_makedarkfit=False, step_makedarkframe=False,  # Make a dark image
                     step_makeflat=False,  # Make a flatfield image
                     step_makearc=False,  # Make an arc image
                     step_makediff=False, step_subbg=False,  # Make difference and sum images
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
    thisred.makePaths(redux_path="/Users/rcooke/Work/Research/BBN/helium34/Absorption/2022_ESO_Survey/PDS241/CRIRES/")
    thisred.run()


class Reduce(ReduceBase):

    def get_science_frames(self):
        return [["CRIRE.2023-01-29T02:43:36.699.fits", "CRIRE.2023-01-29T02:47:48.547.fits"],  # A 1.0 B 1.0
                ["CRIRE.2023-01-29T02:51:51.199.fits", "CRIRE.2023-01-29T02:56:04.258.fits"],  # B 1.0 A 1.0
                ["CRIRE.2023-01-29T03:27:36.514.fits", "CRIRE.2023-01-29T03:00:42.213.fits"],  # B 6.5 A 6.5
                ["CRIRE.2023-01-29T03:31:39.291.fits", "CRIRE.2023-01-29T03:35:56.830.fits"],  # B 6.5 A 6.5
                ["CRIRE.2023-01-29T03:56:29.285.fits", "CRIRE.2023-01-29T03:40:28.416.fits"],  # B 4.5 A 3.0
                ["CRIRE.2023-01-29T03:45:46.666.fits", "CRIRE.2023-01-29T03:51:11.745.fits"]]  # B 3.0 A 4.5
        """
        CRIRE.2023-01-29T03:05:05.963.fits   B 6.5   LOW COUNTS
        CRIRE.2023-01-29T03:23:17.219.fits   A 6.5   LOW COUNTS
        CRIRE.2023-01-29T03:09:08.903.fits   B 6.5   ZERO COUNTS -- effectively the sky/dark
        CRIRE.2023-01-29T03:13:27.874.fits   A 6.5   LOW COUNTS
        """

    def get_flat_frames(self):
        return ["CRIRE.2023-01-29T14:36:21.954.fits",
                "CRIRE.2023-01-29T14:39:11.625.fits",
                "CRIRE.2023-01-29T14:42:01.300.fits",
                "CRIRE.2023-01-29T14:44:50.970.fits",
                "CRIRE.2023-01-29T14:47:40.643.fits",
                "CRIRE.2023-01-29T14:50:30.315.fits",
                "CRIRE.2023-01-29T14:53:19.988.fits",
                "CRIRE.2023-01-29T14:56:09.661.fits",
                "CRIRE.2023-01-29T14:58:59.337.fits",
                "CRIRE.2023-01-29T15:01:49.012.fits"]

    def get_dark_frames(self):
        return [["CRIRE.2023-01-31T19:21:41.698.fits", "CRIRE.2023-01-31T19:21:54.292.fits", "CRIRE.2023-01-31T19:22:06.862.fits"], #7s
                ["CRIRE.2023-01-22T09:21:04.857.fits", "CRIRE.2023-01-22T09:26:10.444.fits", "CRIRE.2023-01-22T09:31:16.061.fits"]]  # 300s

    def get_arc_frames(self):
        return ["CRIRE.2023-01-29T02:33:20.133.fits",
                "CRIRE.2023-01-29T02:37:01.799.fits",
                "CRIRE.2023-01-29T02:37:07.627.fits"]

    def get_exptime(self, idx):
        ndit = self.get_ndit(idx)
        if idx in [4, 5]:
            etim = 300  # This is the DIT
        else:
            etim = 240  # This is the DIT
        exptime = etim * ndit
        return exptime, etim

    def get_ndit(self, idx):
        return 1  # This is the NDIT

    def get_objprof_limits(self, full=True):
        """
        Set the spectral regions to calculate the object profile. If full=True, then a more extended region is used.
        These values are relevant for PDS 241, during the 2023 Jan observations
        """
        if full:
            # All of the object profile
            return [1400.0, 1720.0], [1800.0, 1950.0]
        else:
            # Part of the object profile
            return [1410.0, 1700.0], [1820.0, 1940.0]

    def print_SNregions(self, arr):
        """ Print the S/N in certain regions of the spectrum
        These values are relevant for PDS 241, during the 2023 Jan observations
        """
        print("(box) S/N = ", np.mean(arr[1400:1448]) / np.std(arr[1400:1448]))
        print("(box) S/N ab = ", np.mean(arr[1796:1826]) / np.std(arr[1796:1826]))

    def get_SNregions_fit(self, flux):
        """ Print the S/N in certain regions of the spectrum
        These values are relevant for PDS 241, during the 2023 Jan observations
        """
        xfit = np.arange(1580, 1605)
        ww = (xfit,)
        modl = np.polyval(np.polyfit(xfit, flux[ww], 2), xfit)
        SN_spec = 1.0 / np.std(flux[ww] / modl)
        xfit = np.arange(1796, 1826)
        ww = (xfit,)
        modl = np.polyval(np.polyfit(xfit, flux[ww], 2), xfit)
        SN_abs = 1.0 / np.std(flux[ww] / modl)
        return SN_spec, SN_abs


if __name__ == '__main__':
    main()

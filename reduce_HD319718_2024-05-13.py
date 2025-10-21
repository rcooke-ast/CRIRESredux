from reduce_base import ReduceBase
import numpy as np


def main():
    # Initialise the reduce class
    step = 0
    thisred = Reduce(prefix="hd319718", match_name="HD 319718", data_folder="Raw/",
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
    thisred.makePaths(redux_path="/Users/rcooke/Work/Research/BBN/helium34/Absorption/2023_CRIRES_Survey/HD319718/2024-05-13/")
    thisred._plotit = False
    thisred._comb_set = -1
    thisred.run()


class Reduce(ReduceBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Change some of the default parameters
        self._nbasis = 2
        self._numcomp = 2
        self._scalevariance = [10828.0, 10829.3]  # Scale the variance to match the measured variance in these regions
        self._scale_errors = True  # Scale the errors by 10x in regions with low flux. This is only used for fitting the wavelength solution with ALIS. The errors are scaled back to their extraction values during the combination.

    def get_science_frames(self):
        """
        CRIRE.2024-05-14T06:47:00.740.fits 1 A 1.0 180.0 HD 319718 2.15
        CRIRE.2024-05-14T06:50:14.266.fits 1 B 1.0 180.0 HD 319718 2.15
        CRIRE.2024-05-14T06:53:51.892.fits 1 A 6.5 180.0 HD 319718 2.15
        CRIRE.2024-05-14T06:57:18.699.fits 1 B 6.5 180.0 HD 319718 2.15
        """
        #                A=1.0  (spec=0  DIT=180.0  NDIT=1)
        return [
        [["CRIRE.2024-05-14T06:47:00.740.fits"], ["CRIRE.2024-05-14T06:57:18.699.fits"]],  # B=6.5
        #                B=1.0  (spec=1  DIT=180.0  NDIT=1)
        [["CRIRE.2024-05-14T06:50:14.266.fits"], ["CRIRE.2024-05-14T06:53:51.892.fits"]],  # A=6.5
        #                A=6.5  (spec=2  DIT=180.0  NDIT=1)
        [["CRIRE.2024-05-14T06:53:51.892.fits"], ["CRIRE.2024-05-14T06:50:14.266.fits"]],  # B=1.0
        #                B=6.5  (spec=3  DIT=180.0  NDIT=1)
        [["CRIRE.2024-05-14T06:57:18.699.fits"], ["CRIRE.2024-05-14T06:47:00.740.fits"]],  # A=1.0
        # CRIRE.2024-05-14T07:00:51.602.fits 1 A 2.0 180.0 HD 319718 2.15
        # CRIRE.2024-05-14T07:04:06.765.fits 1 B 2.0 180.0 HD 319718 2.15
        # CRIRE.2024-05-14T07:07:36.604.fits 1 A 5.5 180.0 HD 319718 2.15
        # CRIRE.2024-05-14T07:10:59.598.fits 1 B 5.5 180.0 HD 319718 2.15
        #                A=2.0  (spec=4  DIT=180.0  NDIT=1)
        [["CRIRE.2024-05-14T07:00:51.602.fits"], ["CRIRE.2024-05-14T07:10:59.598.fits"]],  # B=5.5
        #                B=2.0  (spec=5  DIT=180.0  NDIT=1)
        [["CRIRE.2024-05-14T07:04:06.765.fits"], ["CRIRE.2024-05-14T07:07:36.604.fits"]],  # A=5.5
        #                A=5.5  (spec=6  DIT=180.0  NDIT=1)
        [["CRIRE.2024-05-14T07:07:36.604.fits"], ["CRIRE.2024-05-14T07:04:06.765.fits"]],  # B=2.0
        #                B=5.5  (spec=7  DIT=180.0  NDIT=1)
        [["CRIRE.2024-05-14T07:10:59.598.fits"], ["CRIRE.2024-05-14T07:00:51.602.fits"]],  # A=2.0
        # CRIRE.2024-05-14T07:14:30.814.fits 1 A 2.5 180.0 HD 319718 2.15
        # CRIRE.2024-05-14T07:18:01.682.fits 1 B 2.5 180.0 HD 319718 2.15
        # CRIRE.2024-05-14T07:21:40.600.fits 1 A 5.0 180.0 HD 319718 2.15
        # CRIRE.2024-05-14T07:25:02.736.fits 1 B 5.0 180.0 HD 319718 2.15
        #                A=2.5  (spec=8  DIT=180.0  NDIT=1)
        [["CRIRE.2024-05-14T07:14:30.814.fits"], ["CRIRE.2024-05-14T07:25:02.736.fits"]],  # B=5.0
        #                B=2.5  (spec=9  DIT=180.0  NDIT=1)
        [["CRIRE.2024-05-14T07:18:01.682.fits"], ["CRIRE.2024-05-14T07:21:40.600.fits"]],  # A=5.0
        #                A=5.0  (spec=10  DIT=180.0  NDIT=1)
        [["CRIRE.2024-05-14T07:21:40.600.fits"], ["CRIRE.2024-05-14T07:18:01.682.fits"]],  # B=2.5
        #                B=5.0  (spec=11  DIT=180.0  NDIT=1)
        [["CRIRE.2024-05-14T07:25:02.736.fits"], ["CRIRE.2024-05-14T07:14:30.814.fits"]],  # A=2.5
        # CRIRE.2024-05-14T07:43:04.985.fits 1 A 1.0 180.0 HD 319718 2.15
        # CRIRE.2024-05-14T07:46:18.129.fits 1 B 1.0 180.0 HD 319718 2.15
        # CRIRE.2024-05-14T07:49:44.966.fits 1 A 6.5 180.0 HD 319718 2.15
        # CRIRE.2024-05-14T07:53:16.786.fits 1 B 6.5 180.0 HD 319718 2.15
        #                A=1.0  (spec=12  DIT=180.0  NDIT=1)
        [["CRIRE.2024-05-14T07:43:04.985.fits"], ["CRIRE.2024-05-14T07:53:16.786.fits"]],  # B=6.5
        #                B=1.0  (spec=13  DIT=180.0  NDIT=1)
        [["CRIRE.2024-05-14T07:46:18.129.fits"], ["CRIRE.2024-05-14T07:49:44.966.fits"]],  # A=6.5
        #                A=6.5  (spec=14  DIT=180.0  NDIT=1)
        [["CRIRE.2024-05-14T07:49:44.966.fits"], ["CRIRE.2024-05-14T07:46:18.129.fits"]],  # B=1.0
        #                B=6.5  (spec=15  DIT=180.0  NDIT=1)
        [["CRIRE.2024-05-14T07:53:16.786.fits"], ["CRIRE.2024-05-14T07:43:04.985.fits"]],  # A=1.0
        # CRIRE.2024-05-14T07:56:58.288.fits 1 A 2.0 180.0 HD 319718 2.15
        # CRIRE.2024-05-14T08:00:13.863.fits 1 B 2.0 180.0 HD 319718 2.15
        # CRIRE.2024-05-14T08:03:40.908.fits 1 A 5.5 180.0 HD 319718 2.15
        # CRIRE.2024-05-14T08:07:05.458.fits 1 B 5.5 180.0 HD 319718 2.15
        #                A=2.0  (spec=16  DIT=180.0  NDIT=1)
        [["CRIRE.2024-05-14T07:56:58.288.fits"], ["CRIRE.2024-05-14T08:07:05.458.fits"]],  # B=5.5
        #                B=2.0  (spec=17  DIT=180.0  NDIT=1)
        [["CRIRE.2024-05-14T08:00:13.863.fits"], ["CRIRE.2024-05-14T08:03:40.908.fits"]],  # A=5.5
        #                A=5.5  (spec=18  DIT=180.0  NDIT=1)
        [["CRIRE.2024-05-14T08:03:40.908.fits"], ["CRIRE.2024-05-14T08:00:13.863.fits"]],  # B=2.0
        #                B=5.5  (spec=19  DIT=180.0  NDIT=1)
        [["CRIRE.2024-05-14T08:07:05.458.fits"], ["CRIRE.2024-05-14T07:56:58.288.fits"]],  # A=2.0
        # CRIRE.2024-05-14T08:10:34.482.fits 1 A 2.5 180.0 HD 319718 2.15
        # CRIRE.2024-05-14T08:13:56.270.fits 1 B 2.5 180.0 HD 319718 2.15
        # CRIRE.2024-05-14T08:17:31.656.fits 1 A 5.0 180.0 HD 319718 2.15
        # CRIRE.2024-05-14T08:20:56.516.fits 1 B 5.0 180.0 HD 319718 2.15
        #                A=2.5  (spec=20  DIT=180.0  NDIT=1)
        [["CRIRE.2024-05-14T08:10:34.482.fits"], ["CRIRE.2024-05-14T08:20:56.516.fits"]],  # B=5.0
        #                B=2.5  (spec=21  DIT=180.0  NDIT=1)
        [["CRIRE.2024-05-14T08:13:56.270.fits"], ["CRIRE.2024-05-14T08:17:31.656.fits"]],  # A=5.0
        #                A=5.0  (spec=22  DIT=180.0  NDIT=1)
        [["CRIRE.2024-05-14T08:17:31.656.fits"], ["CRIRE.2024-05-14T08:13:56.270.fits"]],  # B=2.5
        #                B=5.0  (spec=23  DIT=180.0  NDIT=1)
        [["CRIRE.2024-05-14T08:20:56.516.fits"], ["CRIRE.2024-05-14T08:10:34.482.fits"]]]  # A=2.5

    def get_flat_frames(self):
        return ["CRIRE.2024-05-09T13:14:25.206.fits",
                "CRIRE.2024-05-09T13:14:45.576.fits",
                "CRIRE.2024-05-09T13:15:05.947.fits",
                "CRIRE.2024-05-09T13:15:26.315.fits",
                "CRIRE.2024-05-09T13:15:46.684.fits"]

    def get_dark_frames(self):
        # Group dark files with different exposure times
        #return [["CRIRE.2022-08-10T11:17:33.973.fits", "CRIRE.2022-08-10T11:18:27.477.fits", "CRIRE.2022-08-10T11:19:20.965.fits", "CRIRE.2022-08-10T12:34:13.900.fits", "CRIRE.2022-08-10T12:35:07.381.fits", "CRIRE.2022-08-10T12:36:00.860.fits"]]
        return [["CRIRE.2024-05-09T13:37:28.226.fits","CRIRE.2024-05-09T13:37:51.564.fits","CRIRE.2024-05-09T13:38:14.916.fits"],#5s
                ["CRIRE.2024-05-09T13:34:56.795.fits","CRIRE.2024-05-09T13:35:47.305.fits","CRIRE.2024-05-09T13:36:37.763.fits"],#45s
                ["CRIRE.2024-05-09T13:28:40.245.fits","CRIRE.2024-05-09T13:30:45.736.fits","CRIRE.2024-05-09T13:32:51.253.fits"]]#120s

    def get_arc_frames(self):
        return ["CRIRE.2024-05-14T09:39:19.309.fits"] # This should be FPET

    def get_trace(self, idx):
        return self._diff_name.format(0)

    def get_exptime(self, idx):
        ndit = self.get_ndit(idx)
        etim = 180.0
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

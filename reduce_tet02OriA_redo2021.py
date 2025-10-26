from reduce_base import ReduceBase


def main():
    # Initialise the reduce class
    step = 2
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
    thisred.makePaths(redux_path="/Users/rcooke/Work/Research/BBN/helium34/Absorption/2023_CRIRES_Survey/tet02oriA/2021/")
    thisred.run()


class Reduce(ReduceBase):

    def __init__(self, **kwargs):
        """
        Frame 07 has a wavy feature that is not ideal... Is this impacting the object profile?
        A similar problem with Frame 11 - even more severe.
        """
        super().__init__(**kwargs)
        # Change some of the default parameters
        self._nbasis = 3  # Number of basis functions to use for the continuum
        self._polyord = 4  # Order of polynomial to use to trace the spectrum
        self._numcomp = 1
        self._scalevariance = [10827.0, 10829.5]  # Scale the variance to match the measured variance in these regions
        self._scale_errors = True  # Scale the errors by 10x in regions with low flux. This is only used for fitting the wavelength solution with ALIS. The errors are scaled back to their extraction values during the combination.

    def get_science_frames(self):
        return [[["CRIRE.2021-09-18T08:07:41.423.fits"], ["CRIRE.2021-09-18T08:17:18.146.fits"]], # A 1.0 B 6.5
                [["CRIRE.2021-09-18T08:17:18.146.fits"], ["CRIRE.2021-09-18T08:07:41.423.fits"]],  # A 1.0 B 6.5
                [["CRIRE.2021-09-18T08:10:01.345.fits"], ["CRIRE.2021-09-18T08:14:55.309.fits"]], # B 1.0 A 6.5
                [["CRIRE.2021-09-18T08:14:55.309.fits"], ["CRIRE.2021-09-18T08:10:01.345.fits"]], # B 1.0 A 6.5
                [["CRIRE.2021-09-18T08:19:56.893.fits"], ["CRIRE.2021-09-18T08:27:23.018.fits"]], # B 5.5 A 2.0
                [["CRIRE.2021-09-18T08:27:23.018.fits"], ["CRIRE.2021-09-18T08:19:56.893.fits"]], # B 5.5 A 2.0
                [["CRIRE.2021-09-18T08:22:20.211.fits"], ["CRIRE.2021-09-18T08:24:58.720.fits"]], # A 5.5 B 2.0
                [["CRIRE.2021-09-18T08:24:58.720.fits"], ["CRIRE.2021-09-18T08:22:20.211.fits"]], # A 5.5 B 2.0
                [["CRIRE.2021-09-18T08:30:05.855.fits"], ["CRIRE.2021-09-18T08:37:31.481.fits"]], # B 5.0 A 2.5
                [["CRIRE.2021-09-18T08:37:31.481.fits"], ["CRIRE.2021-09-18T08:30:05.855.fits"]], # B 5.0 A 2.5
                [["CRIRE.2021-09-18T08:32:32.666.fits"], ["CRIRE.2021-09-18T08:35:07.606.fits"]], # A 5.0 B 2.5
                [["CRIRE.2021-09-18T08:35:07.606.fits"], ["CRIRE.2021-09-18T08:32:32.666.fits"]], # A 5.0 B 2.5
                [["CRIRE.2021-09-18T08:40:13.738.fits"], ["CRIRE.2021-09-18T08:47:43.634.fits"]], # B 4.5 A 3.0
                [["CRIRE.2021-09-18T08:47:43.634.fits"], ["CRIRE.2021-09-18T08:40:13.738.fits"]], # B 4.5 A 3.0
                [["CRIRE.2021-09-18T08:42:36.838.fits"], ["CRIRE.2021-09-18T08:45:19.056.fits"]], # A 4.5 B 3.0
                [["CRIRE.2021-09-18T08:45:19.056.fits"], ["CRIRE.2021-09-18T08:42:36.838.fits"]], # A 4.5 B 3.0
                [["CRIRE.2021-09-18T08:50:28.988.fits"], ["CRIRE.2021-09-18T08:58:03.460.fits"]], # B 4.0 A 3.5
                [["CRIRE.2021-09-18T08:58:03.460.fits"], ["CRIRE.2021-09-18T08:50:28.988.fits"]], # B 4.0 A 3.5
                [["CRIRE.2021-09-18T08:52:53.549.fits"], ["CRIRE.2021-09-18T08:55:39.570.fits"]], # A 4.0 B 3.5
                [["CRIRE.2021-09-18T08:55:39.570.fits"], ["CRIRE.2021-09-18T08:52:53.549.fits"]], # A 4.0 B 3.5
                [["CRIRE.2021-09-18T09:00:53.258.fits"], ["CRIRE.2021-09-18T09:01:28.810.fits"]], # B 6.0 A 6.0
                [["CRIRE.2021-09-18T09:01:28.810.fits"], ["CRIRE.2021-09-18T09:00:53.258.fits"]], # B 6.0 A 6.0
                [["CRIRE.2021-09-18T09:02:26.541.fits"], ["CRIRE.2021-09-18T09:02:47.160.fits"]], # B 6.0 A 6.0
                [["CRIRE.2021-09-18T09:02:47.160.fits"], ["CRIRE.2021-09-18T09:02:26.541.fits"]]]  # B 6.0 A 6.0

    def get_flat_frames(self):
        return ["CRIRE.2021-09-16T13:26:43.664.fits",
              "CRIRE.2021-09-16T13:27:12.447.fits",
              "CRIRE.2021-09-17T15:30:23.058.fits",
              "CRIRE.2021-09-18T12:13:56.888.fits",
              "CRIRE.2021-09-18T12:14:25.683.fits",
              "CRIRE.2021-09-16T13:26:58.054.fits",
              "CRIRE.2021-09-17T15:30:08.692.fits",
              "CRIRE.2021-09-17T15:30:37.465.fits",
              "CRIRE.2021-09-18T12:14:11.285.fits"]

    def get_dark_frames(self):
        # Group dark files with different exposure times
        return [["CRIRE.2021-09-18T10:00:40.505.fits","CRIRE.2021-09-18T10:01:03.878.fits","CRIRE.2021-09-18T10:01:27.193.fits"],#5s
                ["CRIRE.2021-09-18T10:21:23.649.fits", "CRIRE.2021-09-18T10:21:40.904.fits", "CRIRE.2021-09-18T10:21:58.226.fits"]]#3s

    def get_arc_frames(self):
        return ["CRIRE.2022-10-22T10:30:27.878.fits"]

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

    def get_exptime(self, idx):
        ndit = self.get_ndit(idx)
        etim = 5  # This is the DIT
        exptime = etim * ndit
        return exptime, etim

    def get_ndit(self, idx):
        if idx == 10:
            return 3  # This is the NDIT
        elif idx == 11:
            return 1
        else:
            return 20  # This is the NDIT


if __name__ == '__main__':
    main()

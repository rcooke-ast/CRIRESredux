from reduce_base import ReduceBase


def main():
    # Initialise the reduce class
    thisred = Reduce(prefix="tet01OriA",
                     use_diff=False,
                     step_listfiles=False,
                     step_pattern=False,  # Generate an image of the detector pattern
                     step_makedarkfit=False, step_makedarkframe=False,  # Make a dark image
                     step_makeflat=False,  # Make a flatfield image
                     step_makearc=False,  # Make an arc image
                     step_makediff=False,  # Make difference and sum images
                     step_makecuts=False,  # Make difference and sum images
                     step_trace=True, step_extract=True, step_skycoeffs=False, mean_skycoeff=False, step_basis=True,
                     ext_sky=False,  # Trace the spectrum and extract
                     step_wavecal_prelim=True,  # Calculate a preliminary wavelength calibration solution
                     step_prepALIS=False,
                     # Once the data are reduced, prepare a series of files to be used to fit the wavelength solution with ALIS
                     step_combspec=False, step_combspec_rebin=False,
                     # First get the corrected data from ALIS, and then combine all exposures with this step.
                     step_wavecal_sky=False, step_comb_sky=False,
                     # Wavelength calibrate all sky spectra and then combine
                     step_sample_NumExpCombine=False)  # Combine a different number of exposures to estimate how S/N depends on the number of exposures combined.
    thisred.makePaths(redux_path="/Users/rcooke/Work/Research/BBN/helium34/Absorption/2022_ESO_Survey/OrionNebula/CRIRES/")
    thisred.run()


class Reduce(ReduceBase):

    def get_science_frames(self):
        return [["CRIRE.2022-10-24T06:00:36.335.fits", "CRIRE.2022-10-24T06:13:09.282.fits"],  # A 1.0 B 6.5
                ["CRIRE.2022-10-24T06:04:35.716.fits", "CRIRE.2022-10-24T06:09:01.470.fits"],  # B 1 A 6.5
                ["CRIRE.2022-10-24T06:30:39.815.fits", "CRIRE.2022-10-24T06:17:44.032.fits"],  # B 5.5 A 2.0
                ["CRIRE.2022-10-24T06:22:02.926.fits", "CRIRE.2022-10-24T06:26:34.004.fits"],  # B 2.0 A 5.5
                ["CRIRE.2022-10-24T06:52:30.465.fits", "CRIRE.2022-10-24T06:39:29.485.fits"],  # B 5.0 A 2.5
                ["CRIRE.2022-10-24T06:43:55.087.fits", "CRIRE.2022-10-24T06:48:26.383.fits"],  # B 2.5 A 5.0
                ["CRIRE.2022-10-26T07:56:25.958.fits", "CRIRE.2022-10-26T07:41:37.963.fits"],  # B 4.5 A 3
                ["CRIRE.2022-10-26T07:45:45.398.fits", "CRIRE.2022-10-26T07:52:14.741.fits"],  # B 3 A 4.5
                ["CRIRE.2022-10-26T08:05:42.277.fits", "CRIRE.2022-10-26T08:10:26.674.fits"],  # B 3.5 A 4.0
                ["CRIRE.2022-10-26T08:14:33.857.fits", "CRIRE.2022-10-26T08:01:34.447.fits"],  # B 4.0 A 3.5
                ["CRIRE.2022-10-26T07:23:06.300.fits", "CRIRE.2022-10-26T07:32:50.776.fits"],  # B 1.5 A 6.0
                ["CRIRE.2022-10-26T07:37:01.870.fits", "CRIRE.2022-10-26T07:19:05.099.fits"],  # B 6.0 A 1.5
                ["CRIRE.2022-10-24T06:56:45.738.fits", "CRIRE.2022-10-26T08:19:06.223.fits"]]  # B 0.0 A 0.0  # WARNING!!! A=B=0 --> don't use diff!

        # self._matches = [["CRIRE.2022-10-24T06:17:44.032.fits", "CRIRE.2022-10-24T06:39:29.485.fits"], # A 2.0 A 2.5
        #            ["CRIRE.2022-10-24T06:17:44.032.fits", "CRIRE.2022-10-26T07:41:37.963.fits"]] # A 2.0 A 3.0

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
        # self._dark_files = [["CRIRE.2022-10-26T11:03:01.530.fits","CRIRE.2022-10-26T11:03:31.004.fits","CRIRE.2022-10-26T11:04:00.460.fits"]]#7s
        #              ["CRIRE.2022-10-23T09:56:07.724.fits",CRIRE.2022-10-23T09:56:46.200.fits DARK 10.0
        #               ["CRIRE.2022-10-22T10:47:49.622.fits","CRIRE.2022-10-22T10:48:40.271.fits","CRIRE.2022-10-22T10:49:30.864.fits"],#45s
        return [["CRIRE.2022-10-22T10:41:32.754.fits", "CRIRE.2022-10-22T10:43:38.383.fits", "CRIRE.2022-10-22T10:45:43.983.fits"]]  # 120s
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


if __name__ == '__main__':
    main()

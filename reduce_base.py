import os
import time
import glob
import pickle
import numpy as np
import astropy.io.fits as fits
import astropy.units as units
from astropy import stats
from matplotlib import pyplot as plt

from pypeit import utils
from pypeit.core import procimg, skysub, findobj_skymask
from pypeit.core.extract import fit_profile, extract_optimal, extract_boxcar
from pypeit.core.fitting import robust_fit
from pypeit import specobj

from linetools.spectra.xspectrum1d import XSpectrum1D
from scipy import signal
from scipy.signal import medfilt2d, correlate2d
from scipy import ndimage, interpolate
import scipy.optimize as opt
from IPython import embed
import reduce_utils as rwf
import copy
import mpfit


def myfunct(par, fjac=None, xmod=None, xmid=None, flux=None, error=None):
    model_spline = interpolate.CubicSpline(xmod, par)
    model = np.zeros(xmid.size)
    for pp in range(model.size):
        model[pp] = model_spline.integrate(xmid[pp] - 0.5, xmid[pp] + 0.5)
    # Non-negative status value means MPFIT should
    # continue, negative means stop the calculation.
    status = 0
    devs = (flux - model) / error
    return [status, devs]


def myfunct_pix(par, fjac=None, xval=None, flux=None, error=None, objspl=None):
    xss = par[2] * (xval + par[1])
    model = par[0] * objspl(xss) + par[3] + par[4] * xss
    # Non-negative status value means MPFIT should
    # continue, negative means stop the calculation.
    status = 0
    devs = (flux - model) / error
    return [status, devs]


def objprof_chisqfunc_model(par, spat, spec, tck, plotit=False):
    objprof = np.zeros(spat.size)
    for rr in range(spat.size):
        objprof[rr] = par[1]*interpolate.bisplev(par[0]*spat[rr] + par[2]*spat[rr]**2, spec[rr], tck)
        # objprof[rr] = par[1] * (1 + par[2]*spat[rr]/(spat[-1]-spat[0])) * interpolate.bisplev(spat[rr] * par[0], spec[rr], tck)
        # objprof[rr] = par[1] * interpolate.bisplev(spat[rr] * par[0], spec[rr], tck)
    if plotit:
        embed()
    return objprof


def objprof_chisqfunc(par, data, ivar, gpm, spat, spec, tck):
    objprof = objprof_chisqfunc_model(par, spat, spec, tck)
    diff = (data - objprof)*np.sqrt(ivar)
    return np.sum(diff[gpm] ** 2)


class ReduceBase:
    def __init__(self, prefix="targetname", match_name="", data_folder="Raw/", use_diff=False,
                 step_listfiles=False,
                 step_make_combinations=False,  # Generate all combinations of A-B differences (within some tolerance)
                 step_pattern=False,  # Generate an image of the detector pattern
                 step_makedarkfit=False, step_makedarkframe=False,  # Make a dark image
                 step_makeflat=False,  # Make a flatfield image
                 step_makearc=False,  # Make an arc image
                 step_makediff=False,  # Make difference and sum images
                 step_makecuts=False,  # Make difference and sum images
                 step_trace=False, step_extract=False, step_skycoeffs=False, mean_skycoeff=False, step_basis=False, step_subbg=False,
                 ext_sky=False,  # Trace the spectrum and extract
                 step_wavecal_prelim=True,  # Calculate a preliminary wavelength calibration solution
                 step_prepALIS=False,
                 # Once the data are reduced, prepare a series of files to be used to fit the wavelength solution with ALIS
                 step_combspec=False, step_combspec_rebin=False,
                 # First get the corrected data from ALIS, and then combine all exposures with this step.
                 step_wavecal_sky=False, step_comb_sky=False,  # Wavelength calibrate all sky spectra and then combine
                 step_sample_NumExpCombine=False):

        if use_diff and step_subbg:
            print("You cannot set use_diff=True and step_subbg=True")
            assert False

        self._prefix = prefix
        self._match_name = match_name
        self._data_folder = data_folder
        self._plotit = False
        self._specaxis = 0
        self._gain = 2.15  # This value comes from the header
        self._chip = 1  # self._chip can be 1, 2, or 3
        self._slice = np.index_exp[310:600, 0:2048]
        self._polyord = 2#5  # Polynomial order used to trace the spectra
        self._nods = ['A', 'B']
        self._velstep = 1.5  # Sample the FWHM by ~2.5 pixels
        self._maskval = -99999999  # Masked value for combining data
        self._sigcut = 3.0  # Rejection level when combining data
        self._comb_set = -1
        self._nbasis = 15
        self._maxnbasis = 5
        self._subpix_spec = 10  # Number of subpixels to use in the spectral direction
        self._subpix_spat = 10  # Number of subpixels to use in the spatial direction
        self._numcomp = 1  # Number of components to use when fitting the absorption lines to get a preliminary wavelength solution
        self._scalevariance = [10827.0, 10829.5]  # Scale the variance to match the measured variance in these regions
        self._scale_errors = False
        self._use_dark = False

        self.makePaths()

        # Set the reduction flags
        self._use_diff = use_diff
        self._step_listfiles = step_listfiles
        self._step_make_combinations = step_make_combinations
        self._step_pattern = step_pattern
        self._step_makedarkfit = step_makedarkfit
        self._step_makedarkframe = step_makedarkframe
        self._step_makeflat = step_makeflat
        self._step_makearc = step_makearc
        self._step_makediff = step_makediff
        self._step_makecuts = step_makecuts
        self._step_trace = step_trace
        self._step_extract = step_extract
        self._step_skycoeffs = step_skycoeffs
        self._mean_skycoeff = mean_skycoeff
        self._step_basis = step_basis
        self._step_subbg = True#step_subbg
        self._ext_sky = ext_sky
        self._step_wavecal_prelim = step_wavecal_prelim
        self._step_prepALIS = step_prepALIS
        self._step_combspec = step_combspec
        self._step_combspec_rebin = step_combspec_rebin
        self._step_wavecal_sky = step_wavecal_sky
        self._step_comb_sky = step_comb_sky
        self._step_sample_NumExpCombine = step_sample_NumExpCombine

        # Make these a bit simpler
        if self._step_basis:
            self._step_trace = True
            self._step_extract = True
        if self._step_combspec_rebin:
            self._step_combspec = True

        self._matches = self.get_science_frames()

        self._numframes = len(self._matches)
        self._numspec = len(self._matches) * len(self._nods)

        self._flat_files = self.get_flat_frames()
        self._dark_files = self.get_dark_frames()
        self._arc_files = self.get_arc_frames()

    def makePaths(self, redux_path=""):
        self._redux_path = redux_path
        self._cals_folder = "redux_"+self._prefix+"/calibrations/"
        self._proc_folder = "redux_"+self._prefix+"/processed/"
        self._alt_folder = "redux_"+self._prefix+"/alternative/"
        self._datapath = self._redux_path + self._data_folder
        self._calspath = self._redux_path + self._cals_folder
        self._procpath = self._redux_path + self._proc_folder
        self._altpath = self._redux_path + self._alt_folder

        # Check if paths exist, if not, make them
        if not os.path.exists("redux_"+self._prefix):
            os.mkdir("redux_"+self._prefix)
        if not os.path.exists(self._calspath):
            os.mkdir(self._calspath)
        if not os.path.exists(self._procpath):
            os.mkdir(self._procpath)
        self._chip_str = "chip{0:d}.fits".format(self._chip)
        self._pattern_name = self._calspath + "pattern_" + self._chip_str
        self._masterflat_name = self._calspath + "masterflat_" + self._chip_str
        self._masterdark_name = self._calspath + "masterdark_" + self._chip_str
        self._masterarc_name = self._calspath + "masterarc_" + self._chip_str
        self._diff_name = self._procpath + "diff_FR{0:02d}_" + self._chip_str
        self._sumd_name = self._procpath + "sumd_FR{0:02d}_" + self._chip_str
        self._maxd_name = self._procpath + "maxd_FR{0:02d}_" + self._chip_str
        self._bgem_name = self._procpath + "bgem_FR{0:02d}_" + self._chip_str
        self._cut_name = self._procpath + "cuts_FR{0:02d}_" + self._chip_str

    def is_frame_masked(self, frnum):
        masked = False
        if frnum in []:
            masked = True
        return masked

    def is_frame_in_set(self, frnum, comb_set):
        # Just assume all frames are OK
        if comb_set < 0:
            return True
        frame_in_set = False
        if comb_set == 0:
            if frnum in []:
                frame_in_set = True
        elif comb_set == 1:
            if frnum in []:
                frame_in_set = True
        return frame_in_set

    def get_science_frames(self):
        return None

    def get_dark_frames(self):
        return None

    def get_flat_frames(self):
        return None

    def get_arc_frames(self):
        return None

    def get_trace(self, idx):
        return self._diff_name.format(idx)

    def get_exptime(self, idx):
        return 1.0

    def get_ndit(self, idx):
        return 1

    def get_scale(self, idx):
        return 1.0

    def get_objprof_limits(self, full=True):
        """
        Set the spectral regions to calculate the object profile. If full=True, then a more extended region is used.
        These values are relevant for tet01 Ori A, during the 2022 observations
        """
        if full:
            # All of the object profile
            return [1400.0, 1620.0], [1690.0, 1950.0]
        else:
            # Part of the object profile
            return [1410.0, 1600.0], [1720.0, 1940.0]

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
        xfit = np.arange(1580, 1605)
        ww = (xfit,)
        modl = np.polyval(np.polyfit(xfit, flux[ww], 2), xfit)
        SN_spec = 1.0 / np.std(flux[ww] / modl)
        xfit = np.arange(1706, 1726)
        ww = (xfit,)
        modl = np.polyval(np.polyfit(xfit, flux[ww], 2), xfit)
        SN_abs = 1.0 / np.std(flux[ww] / modl)
        return SN_spec, SN_abs

    def run(self):
        if self._step_listfiles: self.step_listfiles()
        if self._step_make_combinations: self.step_make_combinations()
        if self._step_pattern: self.step_pattern()
        if self._step_makedarkfit: self.step_makedarkfit()
        if self._step_makedarkframe: self.step_makedarkframe()
        if self._step_makeflat: self.step_makeflat()
        if self._step_makearc: self.step_makearc()
        if self._step_makediff: self.step_makediff()
        if self._step_makecuts: self.step_makecuts()
        if self._step_trace: self.step_trace()
        if self._step_wavecal_prelim: self.step_wavecal_prelim()
        if self._step_prepALIS: self.step_prepALIS()
        if self._step_combspec: self.step_combspec()
        if self._step_wavecal_sky: self.step_wavecal_sky()
        if self._step_comb_sky: self.step_comb_sky()
        if self._step_sample_NumExpCombine: self.step_sample_NumExpCombine()

    def comb_prep(self, use_corrected=False, sky=False):
        print("Using comb_set = {0:d} for comb_prep".format(self._comb_set))
        raw_specs = []
        minwv = 9999999999999
        maxwv = -minwv
        if use_corrected:
            # usePath = self._altpath + "alt_"
            # if self._use_diff: usePath = self._procpath
            usePath = self._procpath
            numspec = 2 if self._prefix == "hd319718" else 1
            # print(self._numframes * len(self._nods), self._numframes, len(self._nods))
            for ff in range(self._numframes*numspec):# * len(self._nods)):
                if self.is_frame_masked(ff) or not self.is_frame_in_set(ff, self._comb_set):
                    # embed()
                    # assert False
                    print("FRAME {0:d} is masked!!".format(ff))
                    continue
                if sky:
                    outname = usePath + "spec1d_{0:02d}_{1:s}_sky_wzcorr.dat".format(ff // 2, self._nods[ff % 2])
                    opt_wave, opt_cnts, opt_cerr = np.loadtxt(outname, usecols=(0, 1, 2), unpack=True)
                else:
                    outname = usePath + self._prefix+"_ALIS_spec{0:02d}_wzcorr.dat".format(ff)
                    opt_wave, opt_inwave, opt_cnts, opt_cerr = np.loadtxt(outname, usecols=(0, 1, 2, 3), unpack=True)
                if self._scale_errors:
                    print("SCALING ERRORS OF PIXELS WITH LOW FLUX BY 0.1x, to compensate for scaling done in wavelength fitting")
                    # Open up the mask file to see which pixels were scaled
                    maskname = usePath + self._prefix+"_spec{0:02d}_wave.dat.scale".format(ff)
                    maskwave, mask = np.loadtxt(maskname, unpack=True)
                    idxmsk = np.where(np.in1d(maskwave, opt_inwave))
                    ww = np.where(mask[idxmsk] == 1)
                    opt_cerr[ww] *= 0.1
                raw_specs.append(XSpectrum1D.from_tuple((opt_wave, opt_cnts, opt_cerr), verbose=False))
                if np.min(opt_wave) < minwv:
                    minwv = np.min(opt_wave)
                if np.max(opt_wave) > maxwv:
                    maxwv = np.max(opt_wave)
            # If you want to combine multiple different days into a single spectrum, then set False to True and update extraPath.
            if self._prefix == "her36":
                print("\n\n\nWARNING :: you need to know what you're doing to combine here -- currently combining data on 21 & 29 Sept 2024!")
                embed()
                for ff in range(self._numframes):
                    extraPath = "/Users/rcooke/Work/Research/BBN/helium34/Absorption/2023_CRIRES_Survey/Her36/2024-09-29/redux_her36/processed/her36"
                    outname = extraPath + "_ALIS_spec{0:02d}_wzcorr.dat".format(ff)
                    opt_wave, opt_cnts, opt_cerr = np.loadtxt(outname, usecols=(0, 2, 3), unpack=True)
                    raw_specs.append(XSpectrum1D.from_tuple((opt_wave, opt_cnts, opt_cerr), verbose=False))
                    if np.min(opt_wave) < minwv:
                        minwv = np.min(opt_wave)
                    if np.max(opt_wave) > maxwv:
                        maxwv = np.max(opt_wave)
        else:
            # usePath = self._altpath
            # if self._use_diff: usePath = self._procpath
            usePath = self._procpath
            numspec = 2 if self._prefix == "hd319718" else 1
            for ff in range(self._numframes):
                for nn in range(numspec):#, nod in enumerate(self._nods):
                    this_ff = numspec*ff + nn
                    if self.is_frame_masked(ff):
                        print("FRAME {0:d} is masked!!".format(this_ff))
                        continue
                    #outname = usePath + "spec1d_wave_{0:02d}_{1:s}.dat".format(ff, nod)
                    outname = usePath + self._prefix + "_spec{0:02d}_wave.dat".format(this_ff)
                    #box_wave, box_cnts, box_cerr, opt_wave, opt_cnts, opt_cerr = np.loadtxt(outname, unpack=True)
                    # box_wave, box_cnts, box_cerr, box_sky = np.loadtxt(outname, unpack=True)
                    box_wave, box_cnts, box_cerr = np.loadtxt(outname, unpack=True)
                    raw_specs.append(XSpectrum1D.from_tuple((box_wave, box_cnts, box_cerr), verbose=False))
                    if np.min(box_wave) < minwv:
                        minwv = np.min(box_wave)
                    if np.max(box_wave) > maxwv:
                        maxwv = np.max(box_wave)
        # Generate the final wavelength array
        npix = np.log10(maxwv / minwv) / np.log10(1.0 + self._velstep / 299792.458)
        npix = int(npix)
        out_wave = minwv * (1.0 + self._velstep / 299792.458) ** np.arange(npix)
        return out_wave, raw_specs

    def comb_reject(self, out_wave, raw_specs, use_corrected=False):
        digits = []
        wave_bins = out_wave.copy()
        nspec = len(raw_specs)
        # Organise the pixels so we know each pixel that goes into a histogram bin.
        maxnumpix = 0
        for sp in range(nspec):
            dig = np.digitize(raw_specs[sp].wavelength.value, bins=wave_bins)
            digits.append(dig.copy())
            if raw_specs[sp].wavelength.size > maxnumpix:
                maxnumpix = raw_specs[sp].wavelength.size
        # Now, for each wave bin, reject some pixels
        bpm = np.ones((nspec, maxnumpix), dtype=bool)
        # Construct some convenience arrays the same shape as the BPM
        raw_wav = np.zeros(bpm.shape)
        raw_flx = np.zeros(bpm.shape)
        raw_err = np.zeros(bpm.shape)
        for sp in range(nspec):
            bpm[sp, :raw_specs[sp].wavelength.size] = 0
            raw_wav[sp, :raw_specs[sp].wavelength.size] = raw_specs[sp].wavelength.value
            if use_corrected:
                raw_flx[sp, :raw_specs[sp].wavelength.size] = raw_specs[sp].flux
                raw_err[sp, :raw_specs[sp].wavelength.size] = raw_specs[sp].sig
            else:
                medval = 1.0#np.median(raw_specs[sp].flux[raw_specs[sp].flux != 0.0])
                raw_flx[sp, :raw_specs[sp].wavelength.size] = raw_specs[sp].flux / medval
                raw_err[sp, :raw_specs[sp].wavelength.size] = raw_specs[sp].sig / medval
        # Mask bad pixels
        for pp in range(wave_bins.size - 1):
            pixf = np.array([])
            pixe = np.array([])
            midx, midy = np.array([], dtype=int), np.array([], dtype=int)
            for sp in range(nspec):
                ww = np.where((raw_wav[sp, :] >= wave_bins[pp]) & (raw_wav[sp, :] < wave_bins[pp + 1]))
                pixf = np.append(pixf, raw_flx[sp, ww[0]])
                pixe = np.append(pixe, raw_err[sp, ww[0]])
                midx = np.append(midx, sp * np.ones(ww[0].size, dtype=int))
                midy = np.append(midy, ww[0].copy())
            # Now iterate to find any pixels that should be masked
            prevsz = 0
            while True:
                gpm = np.where(bpm[(midx, midy)] == False)
                if gpm[0].size == 0: break
                medv = np.median(pixf[gpm])
                madv = 1.4826 * np.median(np.abs(pixf[gpm] - medv))
                devs = np.where(np.abs((pixf - medv) / np.sqrt(madv ** 2 + pixe ** 2)) > self._sigcut)
                if devs[0].size == prevsz:
                    break
                else:
                    prevsz = devs[0].size
                # Update the BPM
                bpm[(midx[devs], midy[devs])] = True
        return raw_wav, raw_flx, raw_err, bpm

    def comb_rebin(self, out_wave, raw_specs, sky=False, save=True):
        """
        This should only be used after the individual exposures have been processed with ALIS first
        """
        # usePath = self._altpath
        # if self._use_diff: usePath = self._procpath
        usePath = self._procpath
        npix, nspec = out_wave.size, len(raw_specs)
        new_specs = []
        out_flux = self._maskval * np.ones((npix, nspec))
        out_flue = self._maskval * np.ones((npix, nspec))
        for sp in range(nspec):
            new_specs.append(raw_specs[sp].rebin(out_wave * units.AA, do_sig=True, grow_bad_sig=True))
            gpm = new_specs[sp].sig != 0.0
            out_flux[gpm, sp] = new_specs[sp].flux[gpm]
            out_flue[gpm, sp] = new_specs[sp].sig[gpm]
        # Calculate a reference spectrum
        flx_ma = np.ma.array(out_flux, mask=out_flux == self._maskval, fill_value=0.0)
        ref_spec = np.ma.median(flx_ma, axis=1)
        ref_spec_mad = 1.4826 * np.ma.median(np.abs(flx_ma - ref_spec.reshape(ref_spec.size, 1)), axis=1)
        # Compute and apply the scaling to apply to all spectra, relative to the reference
        if self._plotit:
            for sp in range(nspec):
                plt.plot(out_wave, out_flux[:, sp] * np.median(ref_spec / out_flux[:, sp]), 'k-', drawstyle='steps-mid')
            plt.show()
        # Determine which pixels to reject/include in the final combination
        devs = (out_flux - ref_spec.reshape(ref_spec.size, 1)) / out_flue
        #devs = (out_flux - ref_spec.reshape(ref_spec.size, 1)) / ref_spec_mad.reshape(ref_spec.size, 1)#out_flue
        #    devs = (out_flux-ref_spec.reshape(ref_spec.size, 1))/np.ma.sqrt(out_flue**2 + ref_spec_mad.reshape(ref_spec.size, 1)**2)
        mskdev = np.ma.abs(devs) < self._sigcut
        # Make a new array
        new_mask = np.logical_not(mskdev.data & np.logical_not(flx_ma.mask))
        final_flux = np.ma.array(flx_ma.data, mask=new_mask, fill_value=0.0)
        final_flue = np.ma.array(out_flue, mask=new_mask, fill_value=0.0)
        # Compute the final weighted spectrum
        ivar = utils.inverse(final_flue ** 2)
        final_spec = np.ma.average(final_flux, weights=ivar, axis=1)
        variance = np.ma.average((final_flux - final_spec[:, np.newaxis]) ** 2, weights=ivar, axis=1)
        final_spec_err = np.sqrt(variance)
        # Calculate the excess variance
        spec, specerr = final_spec.data, final_spec_err.data
        specerr_new = self.scale_variance(out_wave, spec, specerr)
        if self._plotit or True:
            for sp in range(nspec):
                plt.plot(out_wave, out_flux[:, sp], 'k-', drawstyle='steps-mid')
                ww = new_mask[:, sp]
                plt.plot(out_wave[ww], out_flux[ww, sp], 'bx', drawstyle='steps-mid')
            plt.plot(out_wave, spec, 'r-', drawstyle='steps-mid')
            plt.plot(out_wave, specerr_new, 'g', drawstyle='steps-mid')
            plt.ylim(-0.2, 1.2)
            plt.show()
        # Save the final spectrum
        print("Saving output spectrum...")
        if save:
            if False:
                fitr = np.zeros(out_wave.size)
                if sky:
                    out_specname = usePath + self._prefix+"_HeI10833_scaleErr_fitr_wzcorr_comb_rebin_sky.dat"
                else:
                    out_specname = usePath + self._prefix+"_HeI10833_scaleErr_fitr_wzcorr_comb_rebin.dat"
                    fitr[np.where(
                        ((out_wave > 10827.0) & (out_wave < 10832.64)) | (
                                    (out_wave > 10833.16) & (out_wave < 10839)))] = 1
                np.savetxt(out_specname, np.transpose((out_wave, spec, specerr_new, fitr)))
            else:
                if sky:
                    out_specname = usePath + self._prefix+"_HeI10833_scaleErr_wzcorr_comb_rebin_sky.dat"
                else:
                    out_specname = usePath + self._prefix+"_HeI10833_scaleErr_wzcorr_comb_rebin_spec{0:d}.dat".format(self._comb_set)
                np.savetxt(out_specname, np.transpose((out_wave, spec, specerr_new)))
                plt.plot(out_wave, spec, 'k-', drawstyle='steps-mid')
                plt.show()
            print("File written: {0:s}".format(out_specname))
        return out_wave, spec, specerr_new, specerr, final_flux

    def comb_rebin_pixel(self, raw_specs, outfile=None):
        """
        This should only be used to combine spectra where the only difference
        is the frame that was used in the differencing (i.e. all the same A
        spectra but with multiple different B spectra)
        """
        out_wave = raw_specs[0].wavelength.value
        # If there's just one spectrum, no need to average
        if len(raw_specs) == 1:
            if outfile is not None:
                final_spec, final_spec_err = raw_specs[0].flux, raw_specs[0].sig
                np.savetxt(outfile, np.transpose((out_wave, final_spec, final_spec_err)))
                print("File written: {0:s}".format(outfile))
            return
        npix, nspec = raw_specs[0].flux.size, len(raw_specs)
        out_flux = self._maskval * np.ones((npix, nspec))
        out_flue = self._maskval * np.ones((npix, nspec))
        for sp in range(nspec):
            gpm = raw_specs[sp].sig != 0.0
            out_flux[gpm, sp] = raw_specs[sp].flux[gpm]
            out_flue[gpm, sp] = raw_specs[sp].sig[gpm]
        # Calculate a reference spectrum
        flx_ma = np.ma.array(out_flux, mask=out_flux==self._maskval, fill_value=0.0)
        ref_spec = np.ma.median(flx_ma, axis=1)
        ref_spec_mad = 1.4826 * np.ma.median(np.abs(flx_ma - ref_spec.reshape(ref_spec.size, 1)), axis=1)
        # Determine which pixels to reject/include in the final combination
        devs = (out_flux - ref_spec.reshape(ref_spec.size, 1)) / out_flue
        #devs = (out_flux - ref_spec.reshape(ref_spec.size, 1)) / ref_spec_mad.reshape(ref_spec.size, 1)#out_flue
        #    devs = (out_flux-ref_spec.reshape(ref_spec.size, 1))/np.ma.sqrt(out_flue**2 + ref_spec_mad.reshape(ref_spec.size, 1)**2)
        mskdev = np.ma.abs(devs) < self._sigcut
        # Make a new array
        new_mask = np.logical_not(mskdev.data & np.logical_not(flx_ma.mask))
        final_flux = np.ma.array(flx_ma.data, mask=new_mask, fill_value=0.0)
        final_flue = np.ma.array(out_flue, mask=new_mask, fill_value=0.0)
        # Compute the final weighted spectrum
        ivar = utils.inverse(final_flue ** 2)
        final_spec = np.ma.average(final_flux, weights=ivar, axis=1)
        # Are you sure this following line is correct?
        print("WARNING :: Might want to check the errors on the line below")
        variance = np.ma.average((final_flux - final_spec[:, np.newaxis]) ** 2, weights=ivar, axis=1)
        final_spec_err = np.sqrt(variance)
        if True:
            for sp in range(nspec):
                plt.plot(out_flux[:, sp], 'k-', drawstyle='steps-mid')
                ww = new_mask[:, sp]
                plt.plot(out_flux[ww, sp], 'bx', drawstyle='steps-mid')
            plt.plot(final_spec, 'r-', drawstyle='steps-mid')
            plt.plot(final_spec_err, 'g', drawstyle='steps-mid')
            plt.show()
            embed()
        # Save the final spectrum
        print("Saving output spectrum...")
        if outfile is not None:
            np.savetxt(outfile, np.transpose((out_wave, final_spec, final_spec_err)))
            print("File written: {0:s}".format(outfile))
        return

    def scale_variance(self, out_wave, spec, specerr, getSNR=False):
        wc = np.where((out_wave >= self._scalevariance[0]) & (out_wave <= self._scalevariance[1]))
        # wc = np.where((out_wave >= 10827.0) & (out_wave <= 10829.5))
        # wc = np.where((out_wave >= 10828.25) & (out_wave <= 10829.25))
        # wc = np.where((out_wave >= 10828.0) & (out_wave <= 10830.0))
        # wc = np.where((out_wave >= 10836.4) & (out_wave <= 10837.2))
        mcf = np.polyfit(out_wave[wc], spec[wc], 2)
        modcont = np.polyval(mcf, out_wave[wc])
        plt.plot(out_wave, spec, 'k-', drawstyle='steps')
        plt.plot(out_wave[wc], spec[wc], 'g-', drawstyle='steps')
        plt.plot(out_wave[wc], modcont, 'r-')
        plt.ylim(0.8, 1.2)
        plt.show()
        sig_meas = np.std(spec[wc] - modcont)
        sig_calc = np.mean(specerr[wc])
        scalefact = sig_meas / sig_calc
        specerr_new = scalefact * specerr
        # print("Noise is underestimated by a factor of {0:f}".format(np.median(specerr_new * utils.inverse(specerr))))
        print("Noise is underestimated by a factor of {0:f}".format(scalefact))
        print("New S/N = {0:f}".format(np.median(modcont / specerr_new[wc])))
        if getSNR:
            return np.median(1 / specerr[wc]), np.median(1 / specerr_new[wc])
        else:
            return specerr_new

    def excess_variance(self, out_wave, spec, specerr):
        wc = np.where((out_wave >= 10827) & (out_wave <= 10830))
        mcf = np.polyfit(out_wave[wc], spec[wc], 2)
        modcont = np.polyval(mcf, out_wave[wc])
        sig_meas = np.std(spec[wc] - modcont)
        sig_calc = np.mean(specerr[wc])
        excess_var = sig_meas ** 2 - sig_calc ** 2
        if excess_var < 0.0: excess_var = 0.0
        specerr_new = np.sqrt(excess_var + specerr ** 2)
        print("Excess, measured, calculated", excess_var, sig_meas ** 2, sig_calc ** 2)
        print("Noise is underestimated by a factor of {0:f}".format(np.median(specerr_new / specerr)))
        print("New S/N = {0:f}".format(np.median(modcont / specerr_new[wc])))
        return specerr_new

    def comb_spectrum(self, wave_bins, raw_wav, raw_flx, raw_err, bpm, spec_use, get_specerr_orig=False):
        ww = np.where((bpm == False) & (spec_use) & (raw_err != 0.0))
        ivar = 1.0 / raw_err[ww] ** 2
        spec, _ = np.histogram(raw_wav[ww], bins=wave_bins, weights=raw_flx[ww] * ivar)
        norm, _ = np.histogram(raw_wav[ww], bins=wave_bins, weights=ivar)
        normfact = (norm != 0) / (norm + (norm == 0))
        spec *= normfact
        specerr = np.sqrt(normfact)
        # Calculate the excess noise factor
        out_wave = 0.5 * (wave_bins[1:] + wave_bins[:-1])
        specerr_new = self.excess_variance(out_wave, spec, specerr)
        if get_specerr_orig:
            return out_wave, spec, specerr, specerr_new
        else:
            return out_wave, spec, specerr_new

    def get_darkname(self, basename, tim):
        return basename.replace(".fits", f"_{tim}s.fits")

    def trace_tilt(self, objtrc, trcnum=50, plotit=False, objfrm=None):
        """
        trcnum = the number of spatial pixels to trace either side of the object trace (objtrc)
        """
        msarc = fits.open(self._masterarc_name)[0].data.T
        if objfrm is None or self._redux_path=="/Users/rcooke/Work/Research/BBN/helium34/Absorption/2023_CRIRES_Survey/HD319718/2024-05-13/":
            medfilt = medfilt2d(msarc, kernel_size=(1, 7))
        else:
            medfilt = medfilt2d(objfrm, kernel_size=(1, 7))
        # Find the peak near the trace
        # 1679, 145
        nfit = 0
        for ff in range(-nfit, nfit + 1):
            #        idx = np.arange(1679-17+35*ff,1679+17+35*ff)
            if objfrm is None:
                idx = np.arange(1644 - 17 + 35 * ff, 1644 + 17 + 35 * ff)
                trcnum_use = trcnum
            else:
                idx = np.arange(1699 - 17, 1699 + 17)
                trcnum_use = trcnum-2
            amax = idx[np.argmax(medfilt[(idx, np.round(objtrc).astype(int)[idx])])]
            xpos = int(np.round(objtrc[amax]))
            allcen = np.zeros(1 + 2 * trcnum_use)
            # First trace one way
            this_amax = amax
            for ss in range(0, trcnum_use + 1):
                idx = np.arange(this_amax - 5, this_amax + 5)
                thisspec = medfilt[(idx, xpos + ss)]
                if np.all(thisspec==0.0): continue
                coeff = np.polyfit(idx, thisspec, 2)
                newmax = -0.5 * coeff[1] / coeff[0]
                allcen[trcnum_use + ss] = newmax
                this_amax = int(np.round(newmax))
            # Now trace the other way
            this_amax = amax
            for ss in range(0, trcnum_use):
                idx = np.arange(this_amax - 5, this_amax + 5)
                thisspec = medfilt[(idx, xpos - ss - 1)]
                if np.all(thisspec == 0.0): continue
                coeff = np.polyfit(idx, thisspec, 2)
                newmax = -0.5 * coeff[1] / coeff[0]
                allcen[trcnum_use - ss - 1] = newmax
                this_amax = int(np.round(newmax))
            # Now perform a fit to the tilt
            xdat = np.arange(-trcnum_use, +trcnum_use + 1)
            coeff = np.polyfit(xdat, allcen, 2)
            model = np.polyval(coeff, xdat)
            modcen = np.polyval(coeff, 0)
            coeff = np.polyfit(xdat, modcen - allcen, 2)
            if plotit:
                plt.plot(xdat, allcen, 'b')
                plt.plot(xdat, model, 'r-')
                plt.plot(xdat, allcen-model+modcen, 'g--')
        if plotit:
            plt.show()
        # Generate a tilt image
        spatimg = np.arange(msarc.shape[1])[None, :].repeat(msarc.shape[0], axis=0)-objtrc[:, None]
        specimg = np.arange(msarc.shape[0])[:, None].repeat(msarc.shape[1], axis=1)
        tiltdev = np.polyval(coeff, spatimg)
        tiltimg = specimg + tiltdev
        if objfrm is not None and self._redux_path=="/Users/rcooke/Work/Research/BBN/helium34/Absorption/2023_CRIRES_Survey/HD319718/2024-05-13/":
            # Histogram the data to get object profile
            nsamples = 10
            flx, bin = np.histogram(spatimg.flatten(), bins=np.arange(-20, 20, 1/nsamples), weights=objfrm.flatten())
            midbin = 0.5 * (bin[1:] + bin[:-1])
            # Filter the data to make it smoother
            win = signal.windows.hann(nsamples)
            win /= np.sum(win)
            flxfilt = signal.convolve(flx, win, mode='same')
            # Find the peaks
            peaks = signal.find_peaks(flxfilt, height=0.5 * np.max(flxfilt), distance=5 * nsamples)[0]
            if peaks.size != 2:
                print("Warning: Found {} peaks in the spatial profile".format(peaks.size))
                embed()
                assert False
            peak_pos = midbin[peaks]
            # Polynomial fit near the peaks to get a more accurate position
            peak_pos_fit = np.zeros_like(peak_pos)
            # Find the peak that's closest to zero and call it the primary one
            wpk = np.argmin(np.abs(peak_pos_fit))
            peak_pos_fit = peak_pos_fit[[wpk, 1 - wpk]]
            for pp in range(len(peak_pos)):
                pidx = np.where(np.abs(midbin - peak_pos[pp]) < 2/nsamples)[0]
                if len(pidx) < 3: continue
                coeffp = np.polyfit(midbin[pidx], flxfilt[pidx], 2)
                peak_pos_fit[pp] = -0.5 * coeffp[1] / coeffp[0]
            if plotit:
                plt.plot(midbin, flxfilt, 'r-')
                for pp in peak_pos_fit: plt.axvline(pp, color='b')
                plt.show()
            # Extract a spectrum at each peak position
            wave_spec, flux_spec = [], []
            for pp in peak_pos_fit:
                # Create a dummy PypeIt SpecObj to perform a boxcar extraction
                specobj_dict = dict(SLITID=999, DET='DET01', OBJTYPE='unknown', PYPELINE='MultiSlit')
                thisobj = specobj.SpecObj(**specobj_dict)
                # Set the trace position and boxcar properties
                thisobj.BOX_R_PIX = 1.0
                thisobj.TRACE_SPAT = objtrc + pp
                ivar = np.ones_like(objfrm)
                mask = np.ones_like(objfrm, dtype=bool)
                waveimg = tiltimg
                skyimg = np.zeros_like(objfrm)
                extract_boxcar(objfrm, ivar, mask, waveimg, skyimg, thisobj)
                if plotit: plt.plot(thisobj.BOX_WAVE, thisobj.BOX_COUNTS)
                wave_spec.append(thisobj.BOX_WAVE.copy())
                flux_spec.append(thisobj.BOX_COUNTS.copy())
            if plotit: plt.show()
            # Cross-correlate the two spectra near the He I* line
            # Find the best position to do the cross-correlation by identifying a Gaussian shape in the signal
            # Generate a Gaussian kernel
            gausskern = np.exp(-0.5 * ((np.arange(100) - 50 - 17.5) / 9)**2) + np.exp(-0.5 * ((np.arange(100) - 50 + 17.5) / 6)**2)
            ccfbst = signal.correlate(np.median(flux_spec[0])-flux_spec[0], gausskern, mode='full')
            lags = signal.correlation_lags(len(gausskern), len(flux_spec[0]))-50
            lagbst = lags[np.argmax(ccfbst)]
            if lagbst < 0:
                pos = flux_spec[0].size + lagbst
            else:
                pos = lagbst
            wid = 20
            ccf = signal.correlate(flux_spec[0][pos-wid:pos+wid], flux_spec[1][pos-wid:pos+wid], mode='full')
            lag = np.arange(-len(flux_spec[0][pos-wid:pos+wid])+1, len(flux_spec[0][pos-wid:pos+wid]))
            # Fit a quadratic to the peak
            amax = np.argmax(ccf)
            idx = np.arange(amax-5, amax+5)
            coeffarc = np.polyfit(lag[idx], ccf[idx], 2)
            newmax = -0.5 * coeffarc[1] / coeffarc[0]
            print("Spectral shift between two objects: {:.3f} pixels".format(newmax))
            if plotit:
                plt.plot(lag, ccf, 'b-')
                plt.axvline(newmax, color='r')
                plt.show()
            # Apply the shift to the tilt image
            if peak_pos_fit[1] > 0:
                ww = np.where(spatimg > np.mean(peak_pos_fit))
            else:
                ww = np.where(spatimg < np.mean(peak_pos_fit))
            tiltimg[ww] += newmax
            # Now re-extract to double check the shifting has worked as expected
            for pp in peak_pos_fit:
                # Create a dummy PypeIt SpecObj to perform a boxcar extraction
                specobj_dict = dict(SLITID=999, DET='DET01', OBJTYPE='unknown', PYPELINE='MultiSlit')
                thisobj = specobj.SpecObj(**specobj_dict)
                # Set the trace position and boxcar properties
                thisobj.BOX_R_PIX = 1.0
                thisobj.TRACE_SPAT = objtrc + pp
                ivar = np.ones_like(objfrm)
                mask = np.ones_like(objfrm, dtype=bool)
                waveimg = tiltimg
                skyimg = np.zeros_like(objfrm)
                extract_boxcar(objfrm, ivar, mask, waveimg, skyimg, thisobj)
                if plotit: plt.plot(thisobj.BOX_WAVE, thisobj.BOX_COUNTS)
            if plotit: plt.show()
        if plotit:
            plt.subplot(121)
            plt.imshow(msarc, origin='lower', aspect='auto')
            plt.ylim(1644 - 17, 1644 + 17)
            plt.subplot(122)
            plt.imshow(tiltimg, origin='lower', aspect='auto')
            plt.ylim(1644 - 17, 1644 + 17)
            plt.show()
        if False:
            # Pretty sure this does nothing interesting... have tried changing it before
            tiltimgarc = specimg + np.polyval(coeffarc, spatimg)
            tmp = np.where(np.abs((spatimg - objtrc[:,None]).flatten())<20)
            plt.subplot(211)
            plt.scatter(tiltimg.flatten()[tmp], objfrm.flatten()[tmp], c=spatimg.flatten()[tmp], s=0.1)
            plt.xlim(1650, 1750)
            plt.ylim(0, 10000)
            plt.subplot(212)
            plt.scatter(tiltimgarc.flatten()[tmp], objfrm.flatten()[tmp], c=spatimg.flatten()[tmp], s=0.1)
            plt.xlim(1650, 1750)
            plt.ylim(0, 10000)
            plt.show()
        return tiltimg, tiltdev

    def step_listfiles(self):
        filelist = "redux_"+self._prefix+"/files.list"
        if os.path.exists(filelist):
            files = open(filelist).readlines()
        else:
            files = glob.glob(self._data_folder+"CRIRE*.fits")
            files.sort()
        printfail = []
        print("Searching in:\n"+self._datapath)
        for ff in range(len(files)):
            fil = fits.open(self._datapath + files[ff].lstrip(self._data_folder).strip("\n"))
            try:
                print(files[ff].strip("\n"), fil[0].header['HIERARCH ESO DET NDIT'],
                      fil[0].header['HIERARCH ESO SEQ NODPOS'], fil[0].header['HIERARCH ESO SEQ NODTHROW'],
                      fil[0].header['EXPTIME'], fil[0].header['OBJECT'],
                      fil[1].header['HIERARCH ESO DET CHIP GAIN'])
            except:
                printfail.append(files[ff].strip("\n") + " " + fil[0].header['OBJECT'] + " " + str(fil[0].header['EXPTIME']))
                continue
        print("\n{0:d} files failed to be listed correctly. Here are some details:")
        for ff in range(len(printfail)):
            print(printfail[ff])

    def step_make_combinations(self, tolerance=4.0):
        print("ENTERING ::  step_make_combinations()")
        filelist = "redux_"+self._prefix+"/files.list"
        if os.path.exists(filelist):
            files = open(filelist).readlines()
        else:
            files = glob.glob(self._data_folder+"CRIRE*.fits")
            files.sort()
        # Get all of the information
        fname = []
        xpos = []
        dit = []
        ndit = []
        for ff in range(len(files)):
            fil = fits.open(self._datapath + files[ff].lstrip(self._data_folder).strip("\n"))
            try:
                if fil[0].header['OBJECT'] != self._match_name:
                    # print("DOESN'T MATCH!!!", fil[0].header['OBJECT'], self._match_name)
                    continue
                if fil[0].header['HIERARCH ESO SEQ NODPOS'] == 'A':
                    xpos.append(fil[0].header['HIERARCH ESO SEQ NODTHROW'])
                else:
                    xpos.append(-fil[0].header['HIERARCH ESO SEQ NODTHROW'])
                dit.append(fil[0].header['EXPTIME'])
                ndit.append(fil[0].header['HIERARCH ESO DET NDIT'])
                fname.append(files[ff].strip("\n").split("/")[-1])
            except:
                print("Didn't work!")
                continue
        # Print out every combination
        xpos = np.array(xpos)
        dit = np.array(dit)
        ncomb = len(fname)
        ntotal = 0
        for cc in range(ncomb):
            wt = np.where((np.abs(xpos-xpos[cc]) >= tolerance) &
                          (dit == dit[cc]))[0]
            ww = wt[np.argsort(xpos[(wt,)])]
            nodstr = "A"
            if xpos[cc] < 0: nodstr = "B"
            prtstr  = "#                {0:s}={1:.1f}  (spec={2:d}  DIT={3:.1f}  NDIT={4:d})\n".format(nodstr, abs(xpos[cc]), cc, dit[cc], ndit[cc])
            prtstr += "                 [[\"{0:s}\"], [".format(fname[cc])
            for mm in range(ww.size):
                if mm != 0: prtstr += "                                                           "
                nodstr = "A"
                if xpos[ww[mm]] < 0: nodstr = "B"
                extstr = "\"{0:s}\",    # {1:s}={2:.1f}\n".format(fname[ww[mm]], nodstr, abs(xpos[ww[mm]]))
                if mm == ww.size-1:
                    extstr = extstr.replace("\n", "").replace(",  ", "]],")
                prtstr += extstr
                ntotal += 1
            print(prtstr)
                # [["CRIRE.2022-10-24T06:00:36.335.fits"], ["CRIRE.2022-10-24T06:09:01.470.fits",   # A=6.5
                #                                           "CRIRE.2022-10-26T07:32:50.776.fits",   # A=6.0
                #                                           "CRIRE.2022-10-24T06:26:34.004.fits",   # A=5.5
                #                                           "CRIRE.2022-10-24T06:48:26.383.fits",   # A=5.0
                #                                           "CRIRE.2022-10-26T07:52:14.741.fits",   # A=4.5
                #                                           "CRIRE.2022-10-26T08:10:26.674.fits",   # A=4.0
                #                                           "CRIRE.2022-10-26T08:01:34.447.fits",   # A=3.5
                #                                           "CRIRE.2022-10-26T07:41:37.963.fits"]]
        print("Total number of combinations = {0:d}".format(ntotal))

    def step_pattern(self):
        print("Making detector pattern image")
        fil = fits.open(self._datapath + self._matches[0][0])
        rawdata = fil[self._chip].data * self._gain
        medvec = np.median(rawdata, axis=0)
        medframe = medvec.reshape((1, medvec.size)).repeat(rawdata.shape[0], axis=0)
        hdu = fits.PrimaryHDU(medframe[self._slice])
        hdu.writeto(self._pattern_name, overwrite=True)
        print("File written: {0:s}".format(self._pattern_name))

    def step_makedarkfit(self):
        print("Making dark image")
        # Now generate the flat field
        sigclip = 10.0
        rawdata = np.zeros((self._slice[0].shape + (len(self._dark_files), len(self._dark_files[0]),)))
        exptime = np.zeros(len(self._dark_files))
        for gg in range(len(self._dark_files)):
            for ff in range(len(self._dark_files[gg])):
                fil = fits.open(self._datapath + self._dark_files[gg][ff].strip("\n"))
                print(self._dark_files[gg][ff].strip("\n"), fil[0].header['HIERARCH ESO DET NDIT'],
                      fil[0].header['EXPTIME'],
                      fil[0].header['OBJECT'], fil[1].header['HIERARCH ESO DET CHIP GAIN'])
                rawdata[:, :, gg, ff] = fil[1].data[self._slice]
                if ff == 0: exptime[gg] = fil[0].header['EXPTIME']
                assert (exptime[gg] == fil[0].header['EXPTIME'])
        # Sigma clip
        bpm = np.zeros(rawdata.shape, dtype=bool)
        iternum, prev = 0, 0
        while True:
            mskarr = np.ma.array(rawdata, mask=bpm, fill_value=0.0)
            med = np.ma.median(rawdata, axis=3)
            mad = 1.4826 * np.ma.median(np.abs(mskarr - med[:, :, :, np.newaxis]), axis=3)
            ww = np.where(np.abs(mskarr - med[:, :, :, np.newaxis]) / mad[:, :, :, np.newaxis] > sigclip)
            if ww[0].size == prev:
                break
            elif iternum > 30:
                break
            else:
                prev = ww[0].size
                bpm[ww] = True
                iternum += 1
                print("ITERATION", iternum, prev)
        # Take an average of the dark frames in each group
        msdarkarr = np.ma.mean(mskarr, axis=3).data
        msdark = np.zeros(self._slice[0].shape + (2,))
        # Fit a linear function to each pixel so that we have dark counts per second
        print("Fitting the master dark frame")
        for xx in range(msdark.shape[0]):
            for yy in range(msdark.shape[1]):
                coeff = np.polyfit(exptime, msdarkarr[xx, yy, :], 1)
                msdark[xx, yy, :] = coeff  # np.polyval(coeff, np.array([0.0,1.0]))
                # msdark[xx,yy,1] -= msdark[xx,yy,0] # Need to subtract the constant offset
        hdu = fits.PrimaryHDU(msdark)
        hdu.writeto(self._masterdark_name, overwrite=True)
        print("File written: {0:s}".format(self._masterdark_name))

    def step_makedarkframe(self):
        print("Making dark image")
        # Now generate the dark frame
        sigclip = 10.0
        nslice = (self._slice[0].stop - self._slice[0].start, self._slice[1].stop - self._slice[1].start)
        rawdata = np.zeros((nslice + (len(self._dark_files), len(self._dark_files[0]),)))
        exptime = np.zeros(len(self._dark_files))
        for gg in range(len(self._dark_files)):
            for ff in range(len(self._dark_files[gg])):
                fil = fits.open(self._datapath + self._dark_files[gg][ff].strip("\n"))
                print(self._dark_files[gg][ff].strip("\n"), fil[0].header['HIERARCH ESO DET NDIT'],
                      fil[0].header['EXPTIME'],
                      fil[0].header['OBJECT'], fil[1].header['HIERARCH ESO DET CHIP GAIN'])
                rawdata[:, :, gg, ff] = fil[1].data[self._slice]
                if ff == 0: exptime[gg] = fil[0].header['EXPTIME']
                assert (exptime[gg] == fil[0].header['EXPTIME'])
        # Sigma clip
        bpm = np.zeros(rawdata.shape, dtype=bool)
        iternum, prev = 0, 0
        while True:
            mskarr = np.ma.array(rawdata, mask=bpm, fill_value=0.0)
            med = np.ma.median(rawdata, axis=3)
            mad = 1.4826 * np.ma.median(np.abs(mskarr - med[:, :, :, np.newaxis]), axis=3)
            ww = np.where(np.abs(mskarr - med[:, :, :, np.newaxis]) / mad[:, :, :, np.newaxis] > sigclip)
            if ww[0].size == prev:
                break
            elif iternum > 30:
                break
            else:
                prev = ww[0].size
                bpm[ww] = True
                iternum += 1
                print("ITERATION", iternum, prev)
        # Take an average of the dark frames in each group
        msdarkarr = np.ma.mean(mskarr, axis=3).data
        for gg in range(len(exptime)):
            hdu = fits.PrimaryHDU(msdarkarr[:, :, gg])
            newdarkname = self.get_darkname(self._masterdark_name, int(exptime[gg]))
            hdu.writeto(newdarkname, overwrite=True)
            print("File written: {0:s}".format(newdarkname))

    def step_makeflat(self):
        print("Making flatfield image")
        # Now generate the flat field
        sigclip = 10.0
        nslice = (self._slice[0].stop - self._slice[0].start, self._slice[1].stop - self._slice[1].start)
        rawdata = np.zeros((nslice + (len(self._flat_files),)))
        for ff in range(len(self._flat_files)):
            fil = fits.open(self._datapath + self._flat_files[ff].strip("\n"))
            print(self._flat_files[ff].strip("\n"), fil[0].header['HIERARCH ESO DET NDIT'], fil[0].header['EXPTIME'],
                  fil[0].header['OBJECT'], fil[1].header['HIERARCH ESO DET CHIP GAIN'])
            # Load the dark frame
            msdark = fits.open(self.get_darkname(self._masterdark_name, int(fil[0].header['EXPTIME'])))[0].data
            rawdata[:, :, ff] = fil[1].data[self._slice] - msdark
        # Sigma clip
        bpm = np.zeros(rawdata.shape, dtype=bool)
        iternum, prev = 0, 0
        while True:
            mskarr = np.ma.array(rawdata, mask=bpm, fill_value=0.0)
            med = np.ma.median(rawdata, axis=2)
            mad = 1.4826 * np.ma.median(np.abs(mskarr - med[:, :, np.newaxis]), axis=2)
            ww = np.where(np.abs(mskarr - med[:, :, np.newaxis]) / mad[:, :, np.newaxis] > sigclip)
            if ww[0].size == prev:
                break
            elif iternum > 30:
                break
            else:
                prev = ww[0].size
                bpm[ww] = True
                iternum += 1
                print("ITERATION", iternum, prev)
        msflat = np.ma.mean(mskarr, axis=2)
        normval = np.median(msflat.data[150:170, 1675:1695])
        hdu = fits.PrimaryHDU(msflat.data / normval)
        hdu.writeto(self._masterflat_name, overwrite=True)
        print("File written: {0:s}".format(self._masterflat_name))

    def step_makearc(self):
        sigclip = 10.0
        nslice = (self._slice[0].stop - self._slice[0].start, self._slice[1].stop - self._slice[1].start)
        rawdata = np.zeros((nslice + (len(self._arc_files),)))
        msflat = fits.open(self._masterflat_name)[0].data
        for ff in range(len(self._arc_files)):
            fil = fits.open(self._datapath + self._arc_files[ff].strip("\n"))
            print(self._arc_files[ff].strip("\n"), fil[0].header['HIERARCH ESO DET NDIT'], fil[0].header['EXPTIME'],
                  fil[0].header['OBJECT'], fil[1].header['HIERARCH ESO DET CHIP GAIN'])
            try:
                msdark = fits.open(self.get_darkname(self._masterdark_name, int(fil[0].header['EXPTIME'])))[0].data
            except FileNotFoundError:
                print("    WARNING :: No matching dark found, proceeding without dark subtraction")
                msdark = 0.0
            rawdata[:, :, ff] = fil[1].data[self._slice] - msdark
        # Sigma clip
        bpm = np.zeros(rawdata.shape, dtype=bool)
        iternum, prev = 0, 0
        while True:
            mskarr = np.ma.array(rawdata, mask=bpm, fill_value=0.0)
            med = np.ma.median(rawdata, axis=2)
            mad = 1.4826 * np.ma.median(np.abs(mskarr - med[:, :, np.newaxis]), axis=2)
            ww = np.where(np.abs(mskarr - med[:, :, np.newaxis]) / mad[:, :, np.newaxis] > sigclip)
            if ww[0].size == prev:
                break
            elif iternum > 30:
                break
            else:
                prev = ww[0].size
                bpm[ww] = True
                iternum += 1
                print("ITERATION", iternum, prev)
        msarc = np.ma.mean(mskarr, axis=2)
        hdu = fits.PrimaryHDU(msarc.data)
        hdu.writeto(self._masterarc_name, overwrite=True)
        print("File written: {0:s}".format(self._masterarc_name))

    def step_makediff(self):
        # Load the flat frame
        msflat = fits.open(self._masterflat_name)[0].data
        # Make difference images
        for mm in range(self._numframes):
            fil_a = fits.open(self._datapath + self._matches[mm][0][0])
            img_a = fil_a[self._chip].data
            # embed()
            # assert False
            if self._use_dark:
                print("USING DARK FRAME INSTEAD OF AVERAGED NOD FRAMES")
                try:
                    img_b = np.zeros_like(img_a)
                    img_b[self._slice] = fits.open(self.get_darkname(self._masterdark_name, int(fil_a[0].header['EXPTIME'])))[0].data
                except FileNotFoundError:
                    print("    WARNING :: No matching dark found, proceeding without dark subtraction")
            else:
                # Take an average of all the matched frames to be subtracted off the main science frame
                for ff in range(len(self._matches[mm][1])):
                    fil_b = fits.open(self._datapath + self._matches[mm][1][ff])
                    if ff == 0:
                        img_b_arr = fil_b[self._chip].data[:,:,np.newaxis]
                    else:
                        img_b_arr = np.concatenate((img_b_arr, fil_b[self._chip].data[:,:,np.newaxis]), axis=2)
                # average to remove some ofo the higher values
                img_b, _, _ = stats.sigma_clipped_stats(img_b_arr, sigma=3.0, maxiters=5, axis=2, stdfunc='mad_std')
            # img_b = np.mean(img_b_arr, axis=2)
            # if fil_a[0].header['HIERARCH ESO SEQ NODPOS'].strip() == 'A':
            #     print("Found A", mm)
            #     img_a = fil_a[self._chip].data
            #     img_b = fil_b[self._chip].data
            # else:
            #     print("Switch", fil_a[0].header['HIERARCH ESO SEQ NODPOS'].strip(), mm)
            #     img_b = fil_a[self._chip].data
            #     img_a = fil_b[self._chip].data
            if self._step_subbg and False:
                bgem1 = self._bgem_name.format(2 * mm)
                bgem2 = self._bgem_name.format(2 * mm + 1)
                img_a -= fits.open(bgem1)[0].data / self._gain
                img_b -= fits.open(bgem2)[0].data / self._gain
                print("step subbg")
                embed()
            scale = self.get_scale(mm)
            img_b *= scale
            ndit = self.get_ndit(mm)
            # Take the difference
            diff = (img_a - img_b) * ndit
            sumd = (img_a + img_b) * ndit
            maxd = np.max(np.dstack((img_a, img_b)), axis=2) * ndit
            # Save the output
            outname = self._diff_name.format(mm)
            hdu = fits.PrimaryHDU(diff[self._slice])# / msflat)
            hdu.writeto(outname, overwrite=True)
            print("File written: {0:s}".format(outname))
            # Summed image
            outname = self._sumd_name.format(mm)
            hdu = fits.PrimaryHDU(sumd[self._slice])# / msflat)
            hdu.writeto(outname, overwrite=True)
            print("File written: {0:s}".format(outname))
            # Max image
            outname = self._maxd_name.format(mm)
            hdu = fits.PrimaryHDU(maxd[self._slice])# / msflat)
            hdu.writeto(outname, overwrite=True)
            print("File written: {0:s}".format(outname))

    def save_bgemission(self, frame, idx):
        outname = self._bgem_name.format(idx)
        hdu = fits.PrimaryHDU(frame)
        hdu.writeto(outname, overwrite=True)
        print("File written: {0:s}".format(outname))

    def step_makecuts(self):
        # Make cut outs of the order of interest
        for mm in range(self._numframes):
            # Load the files
            fil_a = fits.open(self._datapath + self._matches[mm][0][0])
            img_a = fil_a[self._chip].data[self._slice]
            ndit = self.get_ndit(mm)
            # Take the difference
            cutA = img_a * ndit
            # Save the output
            outname = self._procpath + "cut_" + self._matches[mm][0][0]
            hdu = fits.PrimaryHDU(cutA)
            hdu.writeto(outname, overwrite=True)
            print("File written: {0:s}".format(outname))

    def basis_fitter(self, x, y, van, w=None, rcond=None, full=False, debug=False):
        order = van.shape[1]
        deg = order - 1
        # set up the least squares matrices in transposed form
        lhs = van.T
        rhs = y.T
        if w is not None:
            w = np.asarray(w) + 0.0
            if w.ndim != 1:
                raise TypeError("expected 1D vector for w")
            if len(x) != len(w):
                raise TypeError("expected x and w to have same length")
            # apply weights. Don't use inplace operations as they
            # can cause problems with NA.
            lhs = lhs * w
            rhs = rhs * w

        # set rcond
        if rcond is None:
            rcond = len(x) * np.finfo(x.dtype).eps

        # Determine the norms of the design matrix columns.
        if issubclass(lhs.dtype.type, np.complexfloating):
            scl = np.sqrt((np.square(lhs.real) + np.square(lhs.imag)).sum(1))
        else:
            scl = np.sqrt(np.square(lhs).sum(1))
        scl[scl == 0] = 1

        # Solve the least squares problem.
        c, resids, rank, s = np.linalg.lstsq(lhs.T / scl, rhs.T, rcond)
        c = (c.T / scl).T

        if debug:
            print("debugging...")
            embed()
        try:
            Vbase = np.linalg.inv(np.dot(van.T*w, van))
        except np.linalg.LinAlgError:
            # The fit didn't work...
            Vbase = np.zeros((lhs.shape[0], lhs.shape[0]))
        # Expand c to include non-fitted coefficients which are set to zero
        #     if deg.ndim > 0:
        #         if c.ndim == 2:
        #             cc = np.zeros((lmax+1, c.shape[1]), dtype=c.dtype)
        #         else:
        #             cc = np.zeros(lmax+1, dtype=c.dtype)
        #         cc[deg] = c
        #         c = cc

        # warn on rank reduction
        if rank != order and not full:
            pass#print("WARNING :: The fit may be poorly conditioned")

        return c, Vbase

    # def fit_object_profile(self, spat, flux, ivar, spacing=0.5):
    #     def fit_func_objprof(x, pars):
    #         interpolate.CubicSpline(x, y)
    #         return a * np.exp(-b * x) + c
    #
    #     xmin, xmax = np.min(spat) - 0.5, np.max(spat) + 0.5
    #     nsample = int(np.ceil((xmax - xmin) / spacing))
    #     xspl = np.linspace(xmin, xmax, nsample)
    #
    #     return interpolate.CubicSpline(xspl, yspl)

    def get_gpm(self, frame):
        gpm_img = np.ones(frame.shape, dtype=bool)
        # Identify salt and pepper pixels with a median filter
        ii, nmask, nnew = 0, 0, -1
        frame_med = frame.copy()
        while (nnew != 0):
            medfilt = medfilt2d(frame_med, kernel_size=(7, 1))
            madfilt = 1.4826 * medfilt2d(np.abs(frame_med - medfilt), kernel_size=(7, 1))
            wbad = np.where((gpm_img) & (np.abs((frame_med - medfilt) * utils.inverse(madfilt)) > 5))
            frame_med = medfilt
            gpm_img[wbad] = False
            nnew = wbad[0].size
            nmask += nnew
            ii += 1
            print(f"Iteration {ii} :: Number of new bad pixels = {nnew}... total number of masked pixels = {nmask}")
        return gpm_img

    def smoothing_spline(self, wave, flux, ivar, gpm, profile, plotit=False, lam=1.0E-5):
        """
        This input variables are all 2D images
        """
        wgud = np.where(gpm & (ivar > 0.0) & (profile > 0.0))
        splprf = profile[wgud]
        splwav = wave[wgud]
        splflx = flux[wgud] * utils.inverse(splprf)
        splwgt = ivar[wgud] * (splprf ** 2)
        wavsrt = np.argsort(splwav, kind='mergesort')
        # Remove duplicate wavelength points for the spline
        wdiff = np.where(np.diff(splwav[wavsrt]) != 0.0)[0]
        spl = interpolate.make_smoothing_spline(splwav[wavsrt[wdiff]], splflx[wavsrt[wdiff]], w=splwgt[wavsrt[wdiff]], lam=lam)
        if plotit:
            splmod = spl(splwav[wavsrt])
            plt.plot(splwav[wavsrt], splflx[wavsrt], 'k.', alpha=0.2)
            plt.plot(splwav[wavsrt], splmod, 'r-')
            plt.ylim(0.0, np.max(splmod))
            plt.xlim(splwav[np.argmin(splmod)] - 100, splwav[np.argmin(splmod)] + 100)
            plt.show()
        return spl

    def object_profile(self, allflux, allivar, allflux_nrm, allivar_nrm, allspecimg, allspatimg, gpm_img, inmaxspatl, inmaxspatr, objtrc, full=False, plotscat=False, inspec=None):
        maxspatl = inmaxspatl + 1
        maxspatr = inmaxspatr + 1
        limfl, limfr = self.get_objprof_limits(full=True)
        limpl, limpr = self.get_objprof_limits(full=False)
        evpix = (allspecimg > limfl[0]) & (allspecimg < limfr[1]) & (allspatimg > -inmaxspatl) & (allspatimg < inmaxspatr)
        # Perform the b-spline fit
        """
        iopt,kx,ky,m=          -1     ,      3     ,      3   ,    54010
        nxest,nyest,nmax=         167      ,   167      ,   167
        lwrk1,lwrk2,kwrk=    30936256 ,   17376779  ,     79610
        xb,xe,yb,ye=   1400.0091915489554   ,     1949.9988359224126     ,  -49.999699324369431     ,   49.997673302888870
        eps,s =   9.9999999999999998E-017 ,  53681.336037874549
        nx, ny = tx.size, ty.size
        u = nxest - kx - 1
        v = nyest - ky - 1
        km = max(kx, ky) + 1
        ne = max(nxest, nyest)
        bx, by = kx*v + ky + 1, ky*u + kx + 1
        b1, b2 = bx, bx + v - ky
        if bx > by:
            b1, b2 = by, by + u - kx

        [- 1 <= iopt <= 1,
        1 <= kx,
        ky <= 5,
        m >= (kx + 1) * (ky + 1),
        nxest >= 2 * kx + 2,
        nyest >= 2 * ky + 2,
        0 < eps < 1,
        nmax >= nxest,
        nmax >= nyest,
        lwrk1 >= u * v * (2 + b1 + b2) + 2 * (u + v + km * (m + ne) + ne - kx - ky) + b2 + 1
        kwrk >= m + (nxest - 2 * kx - 1) * (nyest - 2 * ky - 1),
        np.all((xb <= allspecimg[fitpix]) & (allspecimg[fitpix] <= xe)),
        np.all((yb <= allspatimg[fitpix]) & (allspatimg[fitpix] <= ye))]
        if iopt == -1:
            print(2 * kx + 2 <= nx <= nxest)
            print(2 * ky + 2 <= ny <= nyest)
        xb < tx(kx + 2) < tx(kx + 3) < ... < tx(nx - kx - 1) < xe
        yb < ty(ky + 2) < ty(ky + 3) < ... < ty(ny - ky - 1) < ye
        if iopt >= 0: s >= 0
        w(i) > 0, i = 1, ..., m
        """
        tsty = np.array([2, 2, 2, 2])  # Number of times to iterate, and the order of the profile for each iteration.
        ev_spec, ev_spat = allspecimg[evpix], allspatimg[evpix]
        idxs = np.where(evpix)
        gpm_img_new = gpm_img.copy() & (allivar_nrm != 0)
        # Generate a subpixel sampling of allspatimg
        subpixels = 10  # This is the subpixel sampling
        allspatimg_sub = ((np.arange(allflux.shape[1]*subpixels)[:, np.newaxis].repeat(allflux.shape[0], axis=1).T + 0.5) / subpixels) - objtrc[np.newaxis,:].T
        allspecimg_sub = np.repeat(allspecimg, subpixels, axis=1)  # This is a bit of a hack, but it should work pretty well when the tilts aren't too severe
        evpix_sub = np.repeat(evpix, subpixels, axis=1)  # This is the subpixels corresponding to the pixels where the profile is being evaluated  #(allspecimg_sub > limfl[0]) & (allspecimg_sub < limfr[1]) & (allspatimg_sub > -inmaxspatl) & (allspatimg_sub < inmaxspatr)
        ev_spec_sub, ev_spat_sub = allspecimg_sub[evpix_sub], allspatimg_sub[evpix_sub]
        idxs_sub = np.where(evpix_sub)
        for tt in range(tsty.size):
            # This seems to work OK for tet02OriA_2021
            if full:
                print("Using full object profile")
                fitpix = (gpm_img_new) & (((allspecimg > limfr[0]) & (allspecimg < limfr[1])) | ((allspecimg > limfl[0]) & (allspecimg < limfl[1]))) & (allspatimg > -maxspatl) & (allspatimg < maxspatr)
            else:
                print("Using part object profile")
                fitpix = (gpm_img_new) & (((allspecimg > limpr[0]) & (allspecimg < limpr[1])) | ((allspecimg > limpl[0]) & (allspecimg < limpl[1]))) & (allspatimg > -maxspatl) & (allspatimg < maxspatr)
#             if full:
#                 fitpix = (gpm_img_new) & (allspecimg > 1400) & (allspecimg < 1950) & (allspatimg > -maxspatl) & (allspatimg < maxspatr)
# #                fitpix = (gpm_img_new) & (((allspecimg > 1700) & (allspecimg < 1950)) | ((allspecimg > 1400) & (allspecimg < 1610))) & (allspatimg > -maxspatl) & (allspatimg < maxspatr)
#             else:
#                 fitpix = (gpm_img_new) & (((allspecimg > 1690) & (allspecimg < 1940)) | ((allspecimg > 1410) & (allspecimg < 1610))) & (allspatimg > -maxspatl) & (allspatimg < maxspatr)
            # Make ty
            ty = np.linspace(limfl[0], limfr[1], tsty[tt])
            ty = np.append(np.ones(3) * ty[0], np.append(ty, ty[-1] * np.ones(3)))
            # Make tx
            nxest = int(3+np.sqrt(np.sum(fitpix)/10)) - 6
            if False:
                mdreg = np.arange(-4.0, 4.1, 0.15)
                lreg = np.arange(np.min(allspatimg[fitpix]), -5.0, 1.0)
                rreg = np.linspace(5.0, np.max(allspatimg[fitpix]), lreg.size)
                tx = np.append(lreg, mdreg)
                tx = np.append(tx, rreg)
            else:
                tx = np.linspace(np.min(allspatimg[fitpix]), np.max(allspatimg[fitpix]), nxest)
            # Pad the ticks with repeated starting points
            tx = np.append(np.ones(3) * tx[0], np.append(tx, tx[-1] * np.ones(3)))
            try:
                tck = interpolate.bisplrep(allspatimg[fitpix], allspecimg[fitpix], allflux_nrm[fitpix], w=allivar_nrm[fitpix], task=-1, tx=tx, ty=ty)
            except:
                print("bisplrep failure")
                embed()
                assert False
            outImage = np.zeros_like(allflux_nrm)
            for ii in range(ev_spec.size):
                outImage[idxs[0][ii], idxs[1][ii]] = interpolate.bisplev(ev_spat[ii], ev_spec[ii], tck)
            # Reject deviant pixels
            tst = (allflux_nrm - outImage) * np.sqrt(allivar_nrm)
            bpix = np.where(gpm_img_new & (np.abs(tst) > 20))
            gpm_img_new[bpix] = False
            print("New bad pixels in object profile :: ", bpix[0].size)
            if bpix[0].size == 0:
                break
        # plt.subplot(121)
        # plt.imshow(outImage, origin='lower', aspect=0.3, interpolation='nearest')
        # plt.subplot(122)
        # plt.imshow(fitpix, origin='lower', aspect=0.3, interpolation='nearest')
        # plt.show()
        # embed()
        # Normalise
        print("Generating subpixellated profile")
        outImage_sub = np.zeros_like(allspatimg_sub)
        for ii in range(ev_spec_sub.size):
            outImage_sub[idxs_sub[0][ii], idxs_sub[1][ii]] = interpolate.bisplev(ev_spat_sub[ii], ev_spec_sub[ii], tck) / subpixels
        minv = np.min(outImage_sub[outImage_sub!=0.0])
        outImage[outImage!=0] -= minv*subpixels
        outImage_sub[outImage_sub!=0] -= minv
        norm = utils.inverse(np.sum(outImage, axis=1)[:, None])
        normsub = utils.inverse(np.sum(outImage_sub, axis=1)[:, None])
        # mednrm = np.median(norm[norm != 0.0])
        # norm /= mednrm
        # outImage *= mednrm
        outImage *= np.median(normsub[normsub != 0.0])
        idxf = np.where(fitpix)
        idxt = fitpix & (np.abs((allflux_nrm - (outImage * utils.inverse(normsub))) * np.sqrt(allivar_nrm)) > 2.5)
        if plotscat:
            plt.subplot(211)
            plt.scatter(allspatimg[idxf], (allflux_nrm[idxf] - (outImage * utils.inverse(normsub))[idxf]) * np.sqrt(allivar_nrm[idxf]), c=allspecimg[idxf], s=0.2)
            plt.ylim(-15, 15)
            plt.subplot(212)
            plt.scatter(allspatimg[idxs], (allflux_nrm[idxs] - (outImage * utils.inverse(normsub))[idxs]) * np.sqrt(allivar_nrm[idxs]), c=allspecimg[idxs], s=0.2)
            plt.ylim(-15, 15)
            plt.show()
        if False:
            plt.subplot(211)
            plt.scatter(allspatimg[idxs], outImage[idxs], c=allspecimg[idxs], s=0.1)
            plt.subplot(212)
            plt.scatter(allspatimg[idxs], outImage[idxs], c=allspecimg[idxs], s=0.1)
            plt.scatter(allspatimg[idxs], allflux_nrm[idxs], c=allspecimg[idxs], s=0.1)
            plt.show()
            plt.scatter(allspatimg[idxs], (allflux_nrm[idxs] - (outImage * utils.inverse(normsub))[idxs]) * np.sqrt(allivar_nrm[idxs]), c=allspecimg[idxs], s=0.1)
            plt.ylim(-5,5)
            plt.show()
            plt.hist((allflux_nrm[idxs] - (outImage * utils.inverse(normsub))[idxs]) * np.sqrt(allivar_nrm[idxs]), bin=np.linspace(-5, 5, 100))
            embed()
        # If inspec is provided, adjust the object profile image
        outImageAdjust = outImage.copy()
        opdata = np.zeros(outImageAdjust.shape, dtype=bool)
        if inspec is not None:
            # embed()
            # assert False
            if False:
                # Generate a smoothing spline
                spl = self.smoothing_spline(allspecimg, allflux, allivar, gpm_img_new, outImage, lam=1.0E-7)
                # Calculate the gradient
                xspl = np.linspace(0.0, allspecimg.shape[0]-1, allspecimg.shape[0]*10)
                yspl = spl(xspl)
                outspecimg = spl(allspecimg)
                dyspl = np.gradient(yspl, xspl)
                gradimg = np.interp(allspecimg, xspl, dyspl)
                modimg = np.interp(allspecimg, xspl, yspl)
            else:
                # Use the inspec directly
                xspl = np.arange(allspecimg.shape[0])
                modimg = np.interp(allspecimg, xspl, inspec)
                gradimg = np.gradient(modimg, xspl, axis=0)/np.median(inspec)
            ww = np.where((np.abs(gradimg) > 0.05) & (allflux*np.sqrt(allivar) > 15) & (gpm_img_new) & (outImage>0.0))
            maskout = np.zeros_like(allflux, dtype=bool)
            maskout[ww] = True
            if False:
                plt.subplot(151)
                plt.imshow(allflux, vmin=0, vmax=20000)
                plt.subplot(152)
                plt.imshow(modimg, vmin=0, vmax=200000)
                plt.subplot(153)
                plt.imshow(gradimg, vmin=-0.1, vmax=0.1)
                # plt.imshow(allflux*np.sqrt(allivar), vmin=0, vmax=20)
                plt.subplot(154)
                plt.imshow(gpm_img_new)
                plt.subplot(155)
                plt.imshow(maskout)
                plt.show()
            sumflux = np.sum(allflux*maskout, axis=1)
            nrmflux = np.sum(outImage*maskout, axis=1)
            outImageAdjust[ww] = allflux[ww]
            for ii in range(outImageAdjust.shape[0]):
                if nrmflux[ii] > 0:
                    widx = np.where(maskout[ii, :])[0]
                    outImageAdjust[ii, widx] *= nrmflux[ii] * utils.inverse(sumflux[ii])
            print("Adjusted object profile with input spectrum")
            opdata = outImageAdjust != outImage
            if plotscat:
                plt.subplot(141)
                plt.imshow(outImageAdjust, origin='lower', vmin=0.0, vmax=0.2, aspect=0.3, interpolation='nearest')
                plt.subplot(142)
                plt.imshow(outImage, origin='lower', vmin=0.0, vmax=0.2, aspect=0.3, interpolation='nearest')
                plt.subplot(143)
                plt.imshow(outImageAdjust * utils.inverse(outImage), origin='lower', vmin=0.9, vmax=1.1, aspect=0.3, interpolation='nearest')
                plt.subplot(144)
                plt.imshow(allflux, origin='lower', vmin=0.0, vmax=200000, aspect=0.3, interpolation='nearest')
                plt.show()

                wnz = np.where(outImage != 0.0)
                wslice = np.index_exp[wnz[0].min():wnz[0].max(), wnz[1].min():wnz[1].max()]
                rr = 828
                plt.subplot(211)
                plt.plot(allflux[wslice][rr, :], 'k-')
                plt.plot(outImageAdjust[wslice][rr, :] * np.max(allflux[wslice][rr, :]) / np.max(outImageAdjust[wslice][rr, :]), 'r-')
                plt.subplot(212)
                plt.plot((allflux[wslice][rr, :]) / (outImageAdjust[wslice][rr, :] * np.max(allflux[wslice][rr, :]) / np.max(outImageAdjust[wslice][rr, :])), 'k-')
                plt.show()

            # if True:
            if False:
                # Generate a smoothing spline
                spl = self.smoothing_spline(allspecimg, allflux, allivar, gpm_img_new, outImage)
                # Prepare the plots
                wnz = np.where(outImage != 0.0)
                # outspecimg = np.interp(allspecimg, np.arange(inspec.size), inspec)
                outspecimg = spl(allspecimg)

                tmpfit = allflux * utils.inverse(outImage * outspecimg)
                tmpwgt = allivar * (outImage * outspecimg) ** 2
                tmpmod = np.ones_like(tmpfit)
                # Perform a row by row fit the residual object profile
                for ii in range(allflux.shape[0]):
                    wtmp = np.where((tmpwgt[ii, :] > 0) & (np.isfinite(tmpfit[ii, :])) & (tmpfit[ii, :]>0.5) & (tmpfit[ii, :]<2) & (np.abs(allspatimg[ii,:])<4))[0]
                    if wtmp.size > 3:
                        robfit = fitting.robust_fit(allspatimg[ii, wtmp], tmpfit[ii, wtmp], 2, upper=2, lower=2)
                        tmpmod[ii, :] = robfit.eval(allspatimg[ii, :])
                        # pfit = np.polyfit(allspecimg[ii, wtmp], tmpfit[ii, wtmp], 2)#, w=tmpwgt[ii, wtmp])
                        # tmpmod[ii, :] = np.polyval(pfit, allspecimg[ii, :])
                        if ii in np.arange(1675, 1695):
                            plt.plot(allspatimg[ii, :], tmpfit[ii, :], 'k-')
                            plt.plot(allspatimg[ii, :], tmpmod[ii, :], 'r-')
                            plt.show()
                    else:
                        tmpmod[ii, :] = 1.0
                # plt.plot(tmpfit[1000,:]); plt.show()
                # plt.imshow(tmpmod, vmin=0.5, vmax=1.5)
                # plt.show()
                # plt.imshow(outImageAdjust * tmpmod, vmin=0.0, vmax=0.01)
                # plt.show()
                plt.subplot(131)
                plt.imshow(tmpfit, vmin=0, vmax=2)
                plt.subplot(132)
                plt.imshow(tmpmod, vmin=0.5, vmax=1.5)
                plt.subplot(133)
                plt.imshow(tmpfit * utils.inverse(tmpmod), vmin=0, vmax=2)
                plt.show()

            if False:
                objprofscale = outspecimg * utils.inverse(np.repeat(spl(np.arange(allspecimg.shape[0])[:,None]), allspecimg.shape[1], axis=1))
                modmax = np.max(np.abs(outspecimg[wnz]))
                wwsig = np.where((np.abs(outspecimg) < 0.01 * modmax) | (objprofscale <= 0.0))
                objprofscale[wwsig] = 1.0
                plt.subplot(121)
                plt.imshow(objprofscale, vmin=0, vmax=2)
                plt.subplot(122)
                plt.imshow(allflux, vmin=0, vmax=2E5)
                plt.show()
                # outImageAdjust = model*outspecimg#spec_optimal_flx.reshape((spec_optimal_flx.size, 1))
                outImageAdjust = outImage*objprofscale#spec_optimal_flx.reshape((spec_optimal_flx.size, 1))
                # modmax = np.max(model)
                wslice = np.index_exp[wnz[0].min():wnz[0].max(), wnz[1].min():wnz[1].max()]

                # Difference images:
                tmp_dat = allflux[wslice]
                tmpivar = allivar[wslice]
                tmp_mod = outImage[wslice]
                tmp_modb = outImageAdjust[wslice]
                tmp_gpm = gpm_img_new[wslice]
                tmp_allspat = allspatimg.copy()
                tmp_spat = tmp_allspat[wslice].copy()
                tmp_spec = allspecimg[wslice]
                tmp_allspat[np.logical_not(gpm_img_new)] = 100  # An Arbitrary large number >> 1
                trc_pix = (np.arange(tmp_dat.shape[0]), np.argmin(np.abs(tmp_allspat[wslice]), axis=1),)
                normb = tmp_dat[trc_pix] * utils.inverse(tmp_mod[trc_pix])
                diff = tmp_dat - tmp_mod * normb[:,None]
                diffb = tmp_dat - tmp_modb * normb[:,None]

                # plt.subplot(131)
                # plt.imshow(diff, vmin=-0.03 * modmax, vmax=0.03 * modmax)
                # plt.subplot(132)
                # plt.imshow(diffb, vmin=-0.03 * modmax, vmax=0.03 * modmax)
                # plt.subplot(133)
                # plt.imshow(outImageAdjust[wslice], vmin=0, vmax=np.max(outImageAdjust))
                # plt.show()

                # For each spectral row, calculate the spatial scale factor needed to reproduce a better object profile.
                newProfile = tmp_mod.copy()
                print("Fitting spatial scale factors to each row...")
                bst_scale = np.zeros((diff.shape[0],3))
                cut = 2
                for rr in range(diff.shape[0]):
                    if rr % int(np.ceil(0.25*diff.shape[0])) == 0:
                        print("{:d}% complete".format(int(100*(rr+1)/diff.shape[0])))
                    # Check if this row has an issue
                    wthis = np.where(np.abs(tmp_spat[rr,:])<5.0)[0]
                    chisq = np.sum(np.sort(diff[rr,wthis]**2 * (tmpivar[rr,wthis]))[cut:-cut])/(diff.shape[1]-2*cut)
                    if chisq <= 0.5:
                        continue
                    # Otherwise, let's do the fitting
                    bounds = ((0.75, 1.25), (0.01, 1.0E10), (-1, 1))
                    x0 = np.array([1.0, np.median(tmp_dat[rr,:]*utils.inverse(tmp_mod[rr,:])), 1.0])
                    ivarsend = np.ones(tmp_spat.shape[1]) # tmpivar[rr,:]  The tmpivar creates issues with the fitting, so just use uniform weights.
                    result = opt.minimize(objprof_chisqfunc, x0, args=(tmp_dat[rr,:], ivarsend, tmp_gpm[rr,:], tmp_spat[rr,:], tmp_spec[rr,:], tck), bounds=bounds)
                    if not result.success:
                        print("Fit failed for row {:d}...".format(rr))
                        bst_scale[rr, :] = np.array([1.0, 1.0, 1.0])
                    else:
                        bst_scale[rr,:] = result.x
                        newProfile[rr, :] = objprof_chisqfunc_model(result.x, tmp_spat[rr,:], tmp_spec[rr,:], tck, plotit=False)#(rr==869))
                        # Check for linear dependence
                        if True:
                            xlinfit = tmp_spat[rr,:]
                            normfit = utils.inverse(newProfile[rr,:]) * (np.max(newProfile[rr, :])/np.max(tmp_dat[rr, :]))
                            ylinfit = tmp_dat[rr,:] * normfit
                            errfit = np.ones(ylinfit.size)#np.sqrt(tmpivar[rr,:]) * utils.inverse(normfit)
                            gpmfit = tmp_gpm[rr,:] & (np.abs(xlinfit) < 10.0)
                            if np.sum(gpmfit) > 5:
                                robfit = fitting.robust_fit(xlinfit[gpmfit], ylinfit[gpmfit], 2, upper=2, lower=2)
                                newProfile[rr, :] *= robfit.eval(xlinfit)
                                # coeffs = np.polyfit(xlinfit[gpmfit], ylinfit[gpmfit], 2, w=errfit[gpmfit])
                                # newProfile[rr, :] *= np.polyval(coeffs, xlinfit)
                        # plt.plot(xlinfit, ylinfit, 'k-')
                        # plt.plot(xlinfit, np.polyval(coeffs, xlinfit), 'r-')
                        # plt.show()
                        # plt.plot((allflux[wslice][rr, :]) / (
                        #             outImageAdjust[wslice][rr, :] * np.max(allflux[wslice][rr, :]) / np.max(
                        #         outImageAdjust[wslice][rr, :])), 'k-')

                        # Now calculate the normalisation
                        fine_spat = np.linspace(tmp_spat[rr,:].min(), tmp_spat[rr,:].max(), tmp_spat.shape[1]*100)
                        fine_spec = np.interp(fine_spat, tmp_spat[rr,:], tmp_spec[rr,:])
                        fine_prof = objprof_chisqfunc_model(result.x, fine_spat, fine_spec, tck, plotit=False)
                        fint = np.trapz(fine_prof, x=fine_spat)
                        newProfile[rr, :] *= utils.inverse(fint)
            if False:
                devs = np.zeros(diff.shape[0])
                for rr in range(diff.shape[0]):
                    wthis = np.where(np.abs(tmp_spat[rr,:])<5.0)[0]
                    devs[rr] = stats.sigma_clipped_stats((diff[rr,wthis]*np.sqrt(tmpivar[rr,wthis])), sigma=2.0, maxiters=5)[0]#np.median(np.abs(diff[rr,wthis]))
                robavg, robmed, robstd = stats.sigma_clipped_stats(bst_scale[:,0], sigma=3.0, maxiters=5)
                ww = np.where((np.abs(bst_scale[:,0] - robmed) > 3.0*robstd) & (inspec[wslice[0]]>0.01*np.median(inspec)))[0]
                xarr = np.arange(bst_scale.shape[0])
                plt.plot(xarr, inspec[wslice[0]], 'k-')
                plt.plot(xarr[ww], inspec[wslice[0]][ww], 'ro')
                plt.show()
            # tmp_mod[ww, :] = newProfile[ww, :]
            # outImageAdjust[wslice] = newProfile
            if False:
                embed()
                assert False
                newNorm = tmp_dat[trc_pix] * utils.inverse(newProfile[trc_pix])
                newDiff = tmp_dat - newProfile * newNorm[:, None]
                plt.subplot(131)
                plt.imshow(diff, vmin=-0.03*modmax, vmax=0.03*modmax)
                plt.subplot(132)
                plt.imshow(newDiff, vmin=-0.03*modmax, vmax=0.03*modmax)
                plt.subplot(133)
                plt.imshow(outImageAdjust[wslice], vmin=0, vmax=np.max(outImageAdjust))
                plt.show()

                rr = 828
                plt.subplot(211)
                plt.plot(allflux[wslice][rr, :], 'k-')
                plt.plot(outImageAdjust[wslice][rr, :] * np.max(allflux[wslice][rr, :]) / np.max(outImageAdjust[wslice][rr, :]), 'r-')
                plt.subplot(212)
                plt.plot((allflux[wslice][rr, :])/(outImageAdjust[wslice][rr, :] * np.max(allflux[wslice][rr, :]) / np.max(outImageAdjust[wslice][rr, :])), 'k-')
                plt.show()

                plt.plot(outImage[wslice][rr, :], 'k-')
                plt.plot(outImageAdjust[wslice][rr, :], 'r-')
                plt.show()

                plt.subplot(211)
                plt.plot(bst_scale[:,0], bst_scale[:,1], 'kx', alpha=0.1)
                plt.subplot(212)
                plt.plot(bst_scale[:,0], bst_scale[:,2], 'kx', alpha=0.1)
                plt.show()

                plt.subplot(131)
                plt.imshow(tmp_dat[829-100:829+100,:], vmin=0, vmax=modmax)
                plt.subplot(132)
                plt.imshow(diff[829 - 100:829 + 100, :], vmin=-0.03*modmax, vmax=0.03*modmax)
                plt.subplot(133)
                rat = tmp_dat*utils.inverse(model[wslice]/modmax)
                plt.imshow(rat[829 - 100:829 + 100, :], vmin=0.9*modmax, vmax=1.1*modmax)
                plt.show()
                plt.plot(tmp_dat[829-100:829+100,:].flatten(), diff[829 - 100:829 + 100, :].flatten(), 'bx')
                plt.show()

        return outImageAdjust, gpm_img_new, opdata, norm, normsub

    def mean_bg(self, inImage, inMask, thisboxpix, allspecimg, allspatimg, nwindow_left, nwindow_right, idx):
        limfl, limfr = self.get_objprof_limits(full=True)
        out_wave, raw_specs = self.comb_prep(use_corrected=False)
        raw_wav, raw_flx, raw_err, bpm = self.comb_reject(out_wave, raw_specs, use_corrected=False)
        thiswave = raw_wav[idx, :]
        tmpnameAz = self._procpath + self._prefix + "_ALIS_spec{0:02d}_wzcorr.dat".format(idx)
        out_waveAz, inwaveAz = np.loadtxt(tmpnameAz, unpack=True, usecols=(0, 1))
        wA = np.where(np.in1d(thiswave, inwaveAz))
        coeff = np.polyfit(thisboxpix[wA], out_waveAz, 1)
        waveimg = np.polyval(coeff, allspecimg)
        ref_exptime, _ = self.get_exptime(idx // 2)
        # Make a new background image
        bgImage = np.zeros((allspecimg.shape[0], allspecimg.shape[1], self._numspec))
        print("Making background image")
        for sp in range(self._numspec):
            this_exptime, _ = self.get_exptime(sp // 2)
            # convert pixel to wavelength for this spectrum
            thiswave = raw_wav[sp, :]
            tmpnameAz = self._procpath + self._prefix + "_ALIS_spec{0:02d}_wzcorr.dat".format(sp)
            out_waveAz, inwaveAz = np.loadtxt(tmpnameAz, unpack=True, usecols=(0, 1))
            wA = np.where(np.in1d(thiswave, inwaveAz))
            coeff = np.polyfit(out_waveAz, thisboxpix[wA], 1)
            this_allspecimg = np.polyval(coeff, waveimg)
            # Load the bspline and the parameters needed for the fit
            with open(self._procpath + 'bgfitted_{0:02d}.knots'.format(sp), 'rb') as knots_file:
                knots = pickle.load(knots_file)
            evpix = (this_allspecimg > limfl[0]) & (this_allspecimg < limfr[1]) & (allspatimg > -nwindow_left) & (
                        allspatimg < nwindow_right)
            ev_spec, ev_spat = this_allspecimg[evpix], allspatimg[evpix]
            idxs = np.where(evpix)
            for ii in range(ev_spec.size):
                bgImage[idxs[0][ii], idxs[1][ii], sp] = (ref_exptime / this_exptime) * interpolate.bisplev(ev_spec[ii], ev_spat[ii], knots)
        outImage = np.mean(bgImage, axis=2)
        stdImage = np.std(bgImage, axis=2)
        # Do some masking
        outgpm = inMask & (np.abs(inImage - outImage)*utils.inverse(stdImage) < 10.0)
        # plt.subplot(121)
        # plt.imshow(inImage - outImage, vmin=-100, vmax=100)
        # plt.subplot(122)
        # plt.imshow(inImage - tmp, vmin=-100, vmax=100)
        # plt.show()
        return outImage, outgpm

    def iterate_bgfit(self, HIIresid, gpm_img, allspecimg, allspatimg, maxspatl, maxspatr, idx, trace_gpm, plotit=False):
        limfl, limfr = self.get_objprof_limits(full=True)
        evpix = (allspecimg > limfl[0]) & (allspecimg < limfr[1]) & (allspatimg > -maxspatl) & (allspatimg < maxspatr)
        # Perform the b-spline fit
        """
        iopt,kx,ky,m=          -1     ,      3     ,      3   ,    54010
        nxest,nyest,nmax=         167      ,   167      ,   167
        lwrk1,lwrk2,kwrk=    30936256 ,   17376779  ,     79610
        xb,xe,yb,ye=   1400.0091915489554   ,     1949.9988359224126     ,  -49.999699324369431     ,   49.997673302888870
        eps,s =   9.9999999999999998E-017 ,  53681.336037874549
        nx, ny = tx.size, ty.size
        u = nxest - kx - 1
        v = nyest - ky - 1
        km = max(kx, ky) + 1
        ne = max(nxest, nyest)
        bx, by = kx*v + ky + 1, ky*u + kx + 1
        b1, b2 = bx, bx + v - ky
        if bx > by:
            b1, b2 = by, by + u - kx

        [- 1 <= iopt <= 1,
        1 <= kx,
        ky <= 5,
        m >= (kx + 1) * (ky + 1),
        nxest >= 2 * kx + 2,
        nyest >= 2 * ky + 2,
        0 < eps < 1,
        nmax >= nxest,
        nmax >= nyest,
        lwrk1 >= u * v * (2 + b1 + b2) + 2 * (u + v + km * (m + ne) + ne - kx - ky) + b2 + 1
        kwrk >= m + (nxest - 2 * kx - 1) * (nyest - 2 * ky - 1),
        np.all((xb <= allspecimg[fitpix]) & (allspecimg[fitpix] <= xe)),
        np.all((yb <= allspatimg[fitpix]) & (allspatimg[fitpix] <= ye))]
        if iopt == -1:
            print(2 * kx + 2 <= nx <= nxest)
            print(2 * ky + 2 <= ny <= nyest)
        xb < tx(kx + 2) < tx(kx + 3) < ... < tx(nx - kx - 1) < xe
        yb < ty(ky + 2) < ty(ky + 3) < ... < ty(ny - ky - 1) < ye
        if iopt >= 0: s >= 0
        w(i) > 0, i = 1, ..., m
        """
        tsty = np.array([3, 4])
        ev_spec, ev_spat = allspecimg[evpix], allspatimg[evpix]
        idxs = np.where(evpix)
        gpm_img_new = gpm_img.copy()
        for tt in range(tsty.size):
            fitpix = gpm_img_new & evpix & trace_gpm
            # Make ty
            ty = np.linspace(-maxspatl, maxspatr, tsty[tt])
            ty = np.append(np.ones(3) * ty[0], np.append(ty, ty[-1] * np.ones(3)))
            # Make tx
            nxest = int(3 + np.sqrt(np.sum(fitpix) / 2)) - 6
            if True:
                mdreg = np.arange(1613.0, 1730.0 - 0.9, 2.5)
                # wtmp = (mdreg > 1695) & (mdreg < 1705)
                # mdreg = np.sort(np.append(mdreg, 0.5*(mdreg[wtmp][1:]+mdreg[wtmp][:-1])))
                loreg = np.linspace(np.min(allspecimg[fitpix]), 1612.0, (nxest - mdreg.size) // 2)
                hireg = np.linspace(1730.0, np.max(allspecimg[fitpix]), (nxest - mdreg.size) // 2)
                tx = np.append(loreg, mdreg)
                tx = np.append(tx, hireg)
            else:
                tx = np.linspace(np.min(allspecimg[fitpix]), np.max(allspecimg[fitpix]), nxest//2)
            # Pad the ticks with repeated starting points
            tx = np.append(np.ones(3) * tx[0], np.append(tx, tx[-1] * np.ones(3)))
            try:
                tck = interpolate.bisplrep(allspecimg[fitpix], allspatimg[fitpix], HIIresid[fitpix], task=-1, tx=tx, ty=ty)
            except:
                print("ticks failed...")
                embed()
                assert (False)
            outImage = np.zeros_like(HIIresid)
            for ii in range(ev_spec.size):
                outImage[idxs[0][ii], idxs[1][ii]] = interpolate.bisplev(ev_spec[ii], ev_spat[ii], tck)
            resids = outImage - HIIresid
            medfilt = medfilt2d(resids, kernel_size=(7, 1))
            madfilt = 1.4826 * medfilt2d(np.abs(resids - medfilt), kernel_size=(7, 1))
            wbad = np.where((gpm_img_new) & (trace_gpm) & (np.abs((resids - medfilt) * utils.inverse(madfilt)) > 10))
            print("Number of new masked pixels = ", wbad[0].size)
            gpm_img_new[wbad] = False
        # Save the final version of the knots
        with open(self._procpath+'bgfitted_{0:02d}.knots'.format(idx), 'wb') as pickle_file:
            pickle.dump(tck, pickle_file)
        if plotit:
            #slice = np.meshgrid(np.arange(outImage.shape[0]), np.arange(outImage.shape[1]), indexing='ij')
            slice = np.meshgrid(np.arange(limfl[0], limfr[1]), np.arange(outImage.shape[1]), indexing='ij')
            plt.subplot(131)
            plt.imshow(outImage[slice], vmin=0, vmax=10000)
            plt.subplot(132)
            plt.imshow(HIIresid[slice], vmin=0, vmax=10000)
            plt.subplot(133)
            plt.imshow(outImage[slice] - HIIresid[slice], vmin=-300, vmax=300)
            plt.show()
            print("resids plotted")
            embed()
        return outImage, gpm_img_new
        # from pypeit import flatfield
        # from pypeit.spectrographs.util import load_spectrograph
        # spectrograph = load_spectrograph("vlt_xshooter_nir")
        # par = spectrograph.config_specific_par(None).to_config()
        # pixelFlatField = flatfield.FlatField(fitimage, spectrograph,
        #                                      par['flatfield'], slits, wavetilts, wv_calib)

    def iterate_objfit(self, frame, ivar, gpm_img, spec, opspl, maxspatl, maxspatr, nbasis, numpixfit=10):
        outfluxb = np.zeros(frame.shape[0])
        outfluxb_err = np.zeros(frame.shape[0])
        outfluxbox = np.zeros(frame.shape[0])
        outfluxbox_err = np.zeros(frame.shape[0])
        HIIflux = np.zeros(frame.shape)
        model = np.zeros(frame.shape)
        modelstar = np.zeros(frame.shape)
        idealSN = np.zeros(frame.shape[0])
        coeffs = np.zeros((frame.shape[0], nbasis))
        maxxloc = max(maxspatl, maxspatr)
        for ss in range(frame.shape[0]):
            xdat = np.arange(frame.shape[1]) - spec.TRACE_SPAT[ss]
            xfit = (xdat + maxxloc) / maxxloc - 1
            yfit = frame[ss, :]
            wfit = ivar[ss, :]  # DONT CHANGE THIS WITHOUT CHANGING OUTFLUXBOX_ERR BELOW!!!
            gd = np.where((xdat > -maxspatl) & (xdat < maxspatr) & (gpm_img[ss, :]))  # Only include pixels that are defined within the spatial profile domain
            gdev = np.where((xdat > -maxspatl) & (xdat < maxspatr))  # Only include pixels that are defined within the spatial profile domain
            # Construct the vandermonde matrix
            vander = np.ones((xfit[gd].size, nbasis))
            vander[:, 0] = opspl[ss, :][gd]# opspl(xdat[gd])
            vander[:, 1:] = np.polynomial.legendre.legvander(xfit[gd], nbasis - 2)
            # Vander for the evaluation
            vanderev = np.ones((xfit[gdev].size, nbasis))
            vanderev[:, 0] = opspl[ss, :][gdev]# opspl(xdat[gd])
            vanderev[:, 1:] = np.polynomial.legendre.legvander(xfit[gdev], nbasis - 2)
            try:
                c, cov = self.basis_fitter(xfit[gd], yfit[gd], vander.copy(), w=wfit[gd], debug=False)  # ss==1000)
            except np.linalg.LinAlgError:
                print("LinAlgError on basis fit... Possibly the object profile is wrong, or the data are bad")
                embed()
                assert False
                c = np.zeros(nbasis)
                cov = np.zeros((nbasis, nbasis)) + 1e20
            coeffs[ss, :] = c.copy()
            ccont = c.copy()
            ccont[1] = 0
            gdc = np.where((np.abs(xfit) <= 1.0))  # Only include pixels that are defined within the spatial profile domain
            vanderc = np.ones((xfit[gdc].size, nbasis))
            vanderc[:, 0] = opspl[ss, :][gdc]#opspl(xdat[gdc])
            vanderc[:, 1:] = np.polynomial.legendre.legvander(xfit[gdc], nbasis - 2)  # The rest of the basis are the odd Legendre polynomials
            HIIflux[ss, gdc[0]] = frame[ss, gdc[0]] - np.dot(vanderc, c * np.append(1, np.zeros(nbasis - 1)))
            model[ss, gdev[0]] = np.dot(vanderev, c)
            modelstar[ss, gdev[0]] = np.dot(vanderev, c * np.append(1, np.zeros(nbasis - 1)))
            if False:
                plt.plot(xfit[gd], yfit[gd], 'k-', drawstyle='steps-mid')
                plt.plot(xfit[gd], np.dot(vander, c), 'r-')
                ccont = c.copy()
                ccont[0] = 0
                plt.plot(xfit[gdc], np.dot(vanderc, ccont), 'b--')
                plt.show()
            outfluxb[ss] = c[0]
            outfluxb_err[ss] = np.sqrt(cov[0, 0])
            #outfluxbox[ss] = np.sum(yfit[gd] - HIIflux[ss, gd]) / np.sum(vander[:, 0])
            outfluxbox[ss] = np.sum(yfit[gd]) / np.sum(vander[:, 0])
            outfluxbox_err[ss] = np.sqrt(np.sum(utils.inverse(wfit[gd]))) / np.sum(vander[:, 0])
            idealSN[ss] = np.sum(yfit[gd]) / np.sqrt(np.sum(1 / wfit[gd]))
        if False:
            outPath = self._altpath
            outA = outPath + "spec1d_{0:02d}.dat".format(idx)
            np.savetxt(outA, np.column_stack((outwave, outfluxbox, outfluxbox_err)))
            plt.subplot(311);
            plt.plot(outfluxb, 'b-', drawstyle='steps-mid');
            plt.plot(outfluxbox, 'r-', drawstyle='steps-mid');
            plt.subplot(312);
            plt.plot(outfluxbox, drawstyle='steps-mid');
            plt.subplot(313);
            plt.plot(idealSN, drawstyle='steps-mid');
            plt.show()
            plt.subplot(131);
            plt.imshow(HIIflux, aspect=0.5, vmin=-100, vmax=100);
            plt.subplot(132);
            plt.imshow(extfrm_use, aspect=0.5, vmin=-100, vmax=100);
            plt.subplot(133);
            plt.imshow(gpm_img * (extfrm_use - model) * np.sqrt(ivar_use), aspect=0.5, vmin=-1, vmax=1);
            plt.show()
            print("(opt) S/N = ", np.mean(outfluxb[1338:1448]) / np.std(outfluxb[1338:1448]))
            print("(box) S/N = ", np.mean(outfluxbox[1338:1448]) / np.std(outfluxbox[1338:1448]))
            print("(opt) S/N ab = ", np.mean(outfluxb[1706:1726]) / np.std(outfluxb[1706:1726]))
            print("(box) S/N ab = ", np.mean(outfluxbox[1706:1726]) / np.std(outfluxbox[1706:1726]))

            tst = gpm_img * (extfrm_use - model) * np.sqrt(ivar_use)
            gpmtst = np.where((allgpm) & (allspec > 1550) & (allspec < 1750) & (np.abs(allspat) < 10))
            plttst = tst.flatten()[gpmtst]
            mu = np.mean(plttst)
            sig = np.sqrt(np.mean((plttst - mu) ** 2))
            print(mu, sig)
            plt.hist(plttst, bins=np.linspace(-5, 5, 20))
            plt.show()
        HIIflux = model - modelstar
        if False:
            # Use the residual image to fit the background
            allspecimg = np.tile(np.arange(frame.shape[0]), (frame.shape[1],1)).T
            profimg = opspl.copy()
            spl = self.smoothing_spline(allspecimg, HIIflux, ivar, gpm_img, profimg)
            # outspecimg = spl(allspecimg)
            model = np.ones_like(profimg) * spl(np.arange(allspecimg.shape[0]))[:, None]
            bgfitted = (HIIflux - model)  # * gpm_extwin
            # Fit a smoothing spline to each column of the bgfitted image
            minspat, maxspat = np.min(allspatimg[gpm_extwin]), np.max(allspatimg[gpm_extwin])
            spatbins = np.linspace(minspat, maxspat, int(maxspat - minspat) // 3)
            for jj in range(spatbins.size - 1):
                thisextwin = (allspatimg >= spatbins[jj]) & (allspatimg < spatbins[jj + 1]) & gpm_extwin
                # print(jj+1, spatbins.size-1, np.sum(thisextwin))
                if np.sum(thisextwin) < 10: continue
                splbg = self.smoothing_spline(allspecimg, bgfitted, np.ones(allspecimg.shape), thisextwin,
                                              np.ones(allspecimg.shape), lam=1.0E-2)
                bgfitted[thisextwin] = splbg(allspecimg[thisextwin])
            bgfitted = ndimage.median_filter(bgfitted, size=(5, 5), mode='nearest')
            bgfitted = ndimage.gaussian_filter(bgfitted, sigma=1.0, mode='nearest')
        return HIIflux, outfluxbox, outfluxbox_err, outfluxb, outfluxb_err

    def iterate_objfit_chisq(self, frame, ivar, gpm_img, spec, opspl, maxspatl, maxspatr):
        """
        Gradually increase the value of nbasis until the reduced chi-squared drops below 1.
        """
        print("ENTERING  ::  iterate_objfit_chisq")
        outfluxb = np.zeros(frame.shape[0])
        outfluxb_err = np.zeros(frame.shape[0])
        outfluxbox = np.zeros(frame.shape[0])
        outfluxbox_err = np.zeros(frame.shape[0])
        HIIflux = np.zeros(frame.shape)
        model = np.zeros(frame.shape)
        modelstar = np.zeros(frame.shape)
        idealSN = np.zeros(frame.shape[0])
        use_nbasis = np.zeros(frame.shape[0])
        maxxloc = max(maxspatl, maxspatr)
        for ss in range(frame.shape[0]):
            nbasis = 2
            breaktime = False
            while True:
                xdat = np.arange(frame.shape[1]) - spec.TRACE_SPAT[ss]
                xfit = (xdat + maxxloc) / maxxloc - 1
                yfit = frame[ss, :]
                wfit = ivar[ss, :]  # DONT CHANGE THIS WITHOUT CHANGING OUTFLUXBOX_ERR BELOW!!!
                gd = np.where((xdat > -maxspatl) & (xdat < maxspatr) & (gpm_img[ss, :]))  # Only include pixels that are defined within the spatial profile domain
                # Construct the vandermonde matrix
                vander = np.ones((xfit[gd].size, nbasis))
                vander[:, 0] = opspl[ss, :][gd]# opspl(xdat[gd])
                vander[:, 1:] = np.polynomial.legendre.legvander(xfit[gd], nbasis - 2)
                try:
                    c, cov = self.basis_fitter(xfit[gd], yfit[gd], vander.copy(), w=wfit[gd], debug=False)  # ss==1000)
                except np.linalg.LinAlgError:
                    c = np.zeros(nbasis)
                    cov = np.zeros((nbasis, nbasis)) + 1e20
                ccont = c.copy()
                ccont[1] = 0
                gdc = np.where((np.abs(xfit) <= 1.0))  # Only include pixels that are defined within the spatial profile domain
                vanderc = np.ones((xfit[gdc].size, nbasis))
                vanderc[:, 0] = opspl[ss, :][gdc]#opspl(xdat[gdc])
                vanderc[:, 1:] = np.polynomial.legendre.legvander(xfit[gdc], nbasis - 2)  # The rest of the basis are the odd Legendre polynomials
                HIIflux[ss, gdc[0]] = frame[ss, gdc[0]] - np.dot(vanderc, c * np.append(1, np.zeros(nbasis - 1)))
                model[ss, gd[0]] = np.dot(vander, c)
                modelstar[ss, gd[0]] = np.dot(vander, c * np.append(1, np.zeros(nbasis - 1)))
                if False:
                    plt.plot(xfit[gd], yfit[gd], 'k-', drawstyle='steps-mid')
                    plt.plot(xfit[gd], np.dot(vander, c), 'r-')
                    ccont = c.copy()
                    ccont[0] = 0
                    plt.plot(xfit[gdc], np.dot(vanderc, ccont), 'b--')
                    plt.show()
                outfluxb[ss] = c[0]
                outfluxb_err[ss] = np.sqrt(cov[0, 0])
                #outfluxbox[ss] = np.sum(yfit[gd] - HIIflux[ss, gd]) / np.sum(vander[:, 0])
                outfluxbox[ss] = np.sum(yfit[gd]) / np.sum(vander[:, 0])
                outfluxbox_err[ss] = np.sqrt(np.sum(utils.inverse(wfit[gd]))) / np.sum(vander[:, 0])
                idealSN[ss] = np.sum(yfit[gd]) / np.sqrt(np.sum(1 / wfit[gd]))
                # Test if this spectral element has converged
                dof = gd[0].size - nbasis
                redchisq = np.sum( (yfit[gd]-model[ss, gd[0]])**2 * wfit[gd] ) / dof
                redchisq_med = gd[0].size * np.median((yfit[gd] - model[ss, gd[0]]) ** 2 * wfit[gd]) / dof
                if np.isnan(redchisq_med): redchisq_med = 0.0

                # if ss==1711:
                #     if nbasis == 2: plt.plot(xfit[gd], yfit[gd], 'k-', drawstyle='steps-mid')
                #     plt.plot(xfit[gd], model[ss, gd[0]])
                # print(ss, nbasis, redchisq, redchisq_med, dof)
                if breaktime:
                    break
                if redchisq_med <= 1.0:
                    if nbasis == 2 or gd[0].size < nbasis or dof < 10:
                        use_nbasis[ss] = nbasis
                        break
                    else:
                        use_nbasis[ss] = nbasis
                        break
                        # use_nbasis[ss] = nbasis-1
                        # nbasis -= 1 # Avoid overfitting... redo the fit one last time
                        # breaktime = True
                elif nbasis == self._maxnbasis:
                    use_nbasis[ss] = nbasis
                    break
                else:
                    nbasis += 1
            # plt.show()
            # if ss == 1711:
            #     embed()
        # embed()
        # plt.plot(use_nbasis)
        # plt.show()
        return HIIflux, outfluxbox, outfluxbox_err, outfluxb, outfluxb_err

    def objprof2D(self, allspecimg, allimg, extfrm_use, gpm_img, maxspatl, maxspatr):
        tsty = np.array([3, 7, 21])
        ev_spec, ev_spat = allspecimg[evpix], allspatimg[evpix]
        idxs = np.where(evpix)
        gpm_img_new = gpm_img.copy()
        for tt in range(tsty.size):
            fitpix = gpm_img_new & evpix
            # Make ty
            ty = np.linspace(-maxspatl, maxspatr, tsty[tt])
            ty = np.append(np.ones(3) * ty[0], np.append(ty, ty[-1] * np.ones(3)))
            # Make tx
            nxest = int(3+np.sqrt(np.sum(fitpix)/2)) - 6
            mdreg = np.arange(1613.0, 1730.0 - 0.9, 2.5)
            # wtmp = (mdreg > 1695) & (mdreg < 1705)
            # mdreg = np.sort(np.append(mdreg, 0.5*(mdreg[wtmp][1:]+mdreg[wtmp][:-1])))
            loreg = np.linspace(1400.00919155, 1612.0, (nxest - mdreg.size) // 2)
            hireg = np.linspace(1730.0, 1949.99883592, (nxest - mdreg.size) // 2)
            tx = np.append(loreg, mdreg)
            tx = np.append(tx, hireg)
            # Pad the ticks with repeated starting points
            tx = np.append(np.ones(3) * tx[0], np.append(tx, tx[-1] * np.ones(3)))
            try:
                tck = interpolate.bisplrep(allspecimg[fitpix], allspatimg[fitpix], HIIresid[fitpix], task=-1, tx=tx, ty=ty)
            except:
                print("ticks failed in objprof2d")
                embed()
                assert (False)

    def basis_fit(self, extfrm_use, ivar_use, tilts, waveimg, spatimg, spec, idx, extfrm_use_nrm, ivar_use_nrm, full_bg, edges=None, fullprof=False, plot_resid=False, inspec=None, trccen=None):
        # print("BIG ERROR!!! DELETE THIS RETURN STATEMENT")
        # print("BIG ERROR!!! DELETE THIS RETURN STATEMENT")
        # print("BIG ERROR!!! DELETE THIS RETURN STATEMENT")
        # print("BIG ERROR!!! DELETE THIS RETURN STATEMENT")
        # print("BIG ERROR!!! DELETE THIS RETURN STATEMENT")
        # print("BIG ERROR!!! DELETE THIS RETURN STATEMENT")
        # print("BIG ERROR!!! DELETE THIS RETURN STATEMENT")
        # if idx==0: return
        if edges is None:
            print("Error, edges must be a two element list")
            assert (False)
        ivar_out = ivar_use.copy()
        msflat = fits.open(self._masterflat_name)[0].data.T
        onslit = msflat > 0.1
        onslit[:, :32] = False
        onslit[:, 268:] = False
        sigrej = 3
        nbasis = self._nbasis  # 25
        binsize = 0.1
        nwindow = 20  # +/- 30 pixels is about the maximum window that can be used around the object trace when the nod is +/-6.5 arcseconds from the slit centre
        nspec, nspat = extfrm_use.shape
        # Set the window edges
        ledge, redge = edges
        nwindow_left = int(min(np.min(spec.TRACE_SPAT.flatten() - ledge), nwindow))
        nwindow_right = int(min(np.min(redge-spec.TRACE_SPAT.flatten()), nwindow))
        numtrc = 1
        if trccen is not None:
            if len(trccen) == 2:
                numtrc = 2
                midtrc = 0.5 * (trccen[0].TRACE_SPAT + trccen[1].TRACE_SPAT)
                tsttrc = np.median(spec.TRACE_SPAT.flatten() - midtrc)
                if tsttrc < 0:
                    nwindow_right = -tsttrc
                else:
                    nwindow_left = tsttrc
            else:
                print("I'm not ready for this functionality yet!")
                assert(False)
        print("Left window edge = {0:f}, Right window edge = {1:f}".format(float(nwindow_left), float(nwindow_right)))
        # Trace the spectral tilt
        # allspecimg = np.arange(extfrm_use.shape[0])[:, None].repeat(extfrm_use.shape[1], axis=1)
        objframe = extfrm_use if self._redux_path == "/Users/rcooke/Work/Research/BBN/helium34/Absorption/2023_CRIRES_Survey/HD319718/2024-05-13/" else None
        # allspecimg, allspecimg_dev = self.trace_tilt(spec.TRACE_SPAT.flatten(), trcnum=min(nwindow_left, nwindow_right), plotit=False, objfrm=objframe)
        allspecimg = np.arange(extfrm_use.shape[0])[:, None].repeat(extfrm_use.shape[1], axis=1)
        allspecimg_dev = np.zeros_like(allspecimg)
        allspatimg = (spatimg - spec.TRACE_SPAT[np.newaxis,:].T)
        allspec = allspecimg.flatten()
        allspat = allspatimg.flatten()
        allflux = extfrm_use.flatten()
        allivar = ivar_use.flatten()
        # bins = np.arange(-binsize / 2 - nwindow_left, nwindow_right + binsize, binsize)
        # inds = np.digitize(allspat, bins)
        gpm_img = onslit.copy()
        if numtrc == 1:
            gpm_extwin = (allspatimg > np.max(-nwindow_left-3,0)) & (allspatimg < nwindow_right+3) & (ivar_use > 0) & gpm_img & (allspecimg>2) & (allspecimg<nspec-3)
        elif numtrc == 2:
            gpm_extwin = (allspatimg >= np.max(-nwindow_left,0)) & (allspatimg < nwindow_right) & (ivar_use > 0) & gpm_img & (allspecimg>2) & (allspecimg<nspec-3)
        else:
            print("I'm not ready for this functionality yet!")
            assert(False)
        # Identify salt and pepper pixels with a median filter
        ii, nmask, nnew = 0, 0, -1
        extfrm_use_med = extfrm_use.copy()
        while (nnew != 0):
            medfilt = medfilt2d(extfrm_use_med, kernel_size=(7, 1))
            madfilt = 1.4826 * medfilt2d(np.abs(extfrm_use_med - medfilt), kernel_size=(7, 1))
            wbad = np.where((gpm_img) & (np.abs((extfrm_use_med - medfilt) * utils.inverse(madfilt)) > 10))
            extfrm_use_med = medfilt
            gpm_img[wbad] = False
            nnew = wbad[0].size
            nmask += nnew
            ii += 1
            print(f"Iteration {ii} :: Number of new bad pixels = {nnew}... total number of masked pixels = {nmask}")
        # plt.subplot(141);plt.imshow(gpm_img, aspect=0.5);plt.subplot(142);plt.imshow(extfrm_use, aspect=0.5, vmin=-1000, vmax=1000);plt.subplot(143);plt.imshow(extfrm_use_med, aspect=0.5, vmin=-1000, vmax=1000);plt.subplot(144);plt.imshow(medfilt, aspect=0.5, vmin=-1000, vmax=1000);plt.show()
        if False:
            allgpm = gpm_img.flatten()
            prof = np.zeros(bins.size - 1)
            # Mask based on object profile
            ii, nmask = 0, -1
            while (nmask != 0):
                # Go through and mask pixels
                nmask = 0
                for bb in range(bins.size):
                    thisbin = (inds == bb + 1) & (allgpm)
                    ww = np.where(thisbin)
                    med = np.median(allflux[ww])
                    mad = 1.4826 * np.median(np.abs(med - allflux[ww]))
                    wbad = np.where(thisbin & (np.abs((allflux - med) / mad) > sigrej))
                    allgpm[wbad] = False
                    nmask += wbad[0].size
                ii += 1
                print(f"Iteration {ii} :: Number of new bad pixels = {nmask}")
        # Calculate the object profile (used for the first basis vector)
        # if True:
        #     # test if the tilts worked
        #     gpm = np.where((gpm_img.flatten()) & (allspat > -nwindow) & (allspat < nwindow))
        #     cnts, _ = np.histogram(allspec[gpm], bins=np.arange(2048), weights=extfrm_use_med.flatten()[gpm])
        #     norm, _ = np.histogram(allspec[gpm], bins=np.arange(2048))
        #     cnts *= utils.inverse(norm)
        #     plt.plot(cnts)
        #     plt.show()
        opimg, opgpm, opdat, opnrm, opnrmsub = self.object_profile(extfrm_use, ivar_use, extfrm_use_nrm, ivar_use_nrm, allspecimg, allspatimg, gpm_extwin, nwindow_left, nwindow_right, spec.TRACE_SPAT, full=fullprof, inspec=inspec)
        # extfrm_use *= opnrm
        # xloc = 0.5 * (bins[1:] + bins[:-1])
        if False:
            limpl, limpr = self.get_objprof_limits(full=False)
            tmp = (allgpm) & ((allspec > limpr[1]) & (allspec < limpr[1])) | ((allspec > limpl[0]) & (allspec < limpl[1]))
            gpm = np.where(tmp)
            # First estimate of object profile
            cnts, _ = np.histogram(allspat[gpm], bins=bins, weights=extfrm_use_med.flatten()[gpm])
            norm, _ = np.histogram(allspat[gpm], bins=bins)
            cnts *= utils.inverse(norm)
            # Calculate the step width (used for the second basis vector)
            #     cnts /= (2*np.sum(cnts))  # Factor of 2 is because we only consider half the profile here.
            nrmcnts = np.sum(cnts)
            cnts /= nrmcnts
            # Interpolate the object profile so that it can be used at each wavelength
            #     xspl = np.append(-xloc[1:][::-1], xloc)
            #     yspl = np.append(cnts[1:][::-1], cnts)
            opspl = interpolate.CubicSpline(xloc, 1.0E6 * cnts)
        # plt.scatter(allspat[tmp], extfrm_use.flatten()[tmp], c=allspec[tmp], s=1)
        # plt.xlim(-20, 20)
        # plt.ylim(0, 30000)
        # plt.show()
        #
        # plt.scatter(allspec, extfrm_use.flatten(), c=allspat, s=1)
        # plt.xlim(1650, 1750)
        # plt.ylim(0, 30000)
        # plt.show()

        # tbins = interpolate.CubicSpline(50 * np.cumsum(cnts), xloc)(np.arange(50))
        # op_bpm = np.logical_not(onslit).flatten()
        # xop, yop = allspat.copy(), extfrm_use.flatten()
        # while True:
        #     this = np.where(np.logical_not(op_bpm) & (xop>-nwindow_left) & (xop<nwindow_right))
        #     asrt = np.argsort(xop[this])
        #     xmn, xmx = np.min(xop[this]), np.max(xop[this])
        #     nval = 1+int((xmx-xmn)*10)
        #     bs = 0.5*(xmx-xmn)/nval
        #     tbins = np.linspace(xmn+bs, xmx-bs, nval)
        #     spl = interpolate.splrep(xop[this][asrt], yop[this][asrt], task=-1, t=tbins)
        #     splthis = interpolate.splev(xop, spl)
        #     tst = np.abs(yop-splthis)
        #     mad = 1.4826*np.median(tst)
        #     numbad = np.where((tst > 5.0*mad) & np.logical_not(op_bpm))
        #     if numbad[0].size == 0:
        #         break
        #     else:
        #         print("Found {0:d} new bad pixels".format(numbad[0].size))
        #         op_bpm[numbad] = True
        # plt.plot(xloc, cnts/np.max(cnts), 'b-')
        # plt.plot(tbins, interpolate.splev(tbins, spl)/np.max(interpolate.splev(tbins, spl)), 'r-')
        # plt.show()
        #
        # opmodel = np.zeros(extfrm_use.shape, dtype=float)
        # gpm_img_new = tmp.reshape(extfrm_use.shape)
        # skymodel, objmodel, ivarmodel, extractmask = skysub.local_skysub_extract(
        #     extfrm_use.astype(float), ivar_use, tilts, allspecimg,
        #     np.zeros(extfrm_use.shape), gpm_img_new, ledge, redge,
        #     spec, ingpm=gpm_img_new,
        #     spat_pix=None,
        #     model_full_slit=False,
        #     sigrej=5.0,
        #     model_noise=False,  # base_var=basevar,
        #     bsp=0.1,
        #     std=False,
        #     adderr=0.0002,
        #     force_gauss=False,
        #     sn_gauss=4,
        #     show_profile=self._plotit,
        #     use_2dmodel_mask=False,
        #     no_local_sky=False)
        # opmodel[gpm_img_new] = objmodel

        # Now perform the fit
        #slice = np.meshgrid(np.arange(1600, 1750), np.arange(extfrm_use.shape[1]), indexing='ij')
        trace_mask = np.abs(allspatimg) > 3.0
        numiter = 1# if not self._use_diff else 1
        mean_bg = False
        testing, subtesting = False, False  # Need to (1) True, True, then set the best test value; (2) False, True, then set the best sub test value; (3) False, False, once both test and subtests have been done
        if testing:
            # Use this option to learn what the optimal number of pixels is
            tst = np.arange(1, 15, dtype=float)
        else:
            # Insert the optimal value into the array below
            tst = np.array([1.0], dtype=float)
        SN_spec, SN_abs = np.zeros(tst.size), np.zeros(tst.size)
        for tt in range(tst.size):
            #trace_mask = np.abs(allspatimg) > tst[tt]
            bgfitted = np.zeros(extfrm_use.shape)
            gpm_img_new = gpm_img.copy() & gpm_extwin# & opgpm
            #gpm_img_tmp = gpm_img.copy()
            for ii in range(numiter):
                if ii == 0: this_nbasis = nbasis
                elif ii <= 2: this_nbasis = nbasis//2
                else: this_nbasis = this_nbasis = nbasis//3
                if self._use_diff and False:
                    # Obtain an estimate of the background level
                    bgspec = np.median(extfrm_use, axis=1)
                    # Apply a median filter to the background spectrum
                    bgfilt = signal.medfilt(bgspec, 25)
                    bgfitted = np.tile(bgfilt, (extfrm_use.shape[1], 1)).T
                    # bgfitted = np.zeros(extfrm_use.shape)
                else:
                    # HIIresid, outfluxbox, outfluxbox_err = self.iterate_objfit(extfrm_use-bgfitted, ivar_use, gpm_img_new, spec, opspl, xloc, nbasis, numpixfit=tst[tt])
                    HIIresid, outfluxbox, outfluxbox_err, outfluxopt, outfluxopt_err = self.iterate_objfit(extfrm_use - bgfitted, ivar_use,
                                                                               gpm_img_new, spec, opimg, nwindow_left,
                                                                               nwindow_right, this_nbasis,
                                                                               numpixfit=tst[tt])
                    limfl, limfr = self.get_objprof_limits(full=True)
                    HIIresid[:int(limfl[0])+2, :] = 0
                    HIIresid[int(limfr[-1])-2:, :] = 0
                    # plt.imshow(HIIresid, vmin=0, vmax=100)
                    # plt.show()
                    # embed()
                    smooth = False
                    if smooth:
                        HIIresid = ndimage.median_filter(HIIresid, size=(7,7), mode='nearest')
                        HIIresid = ndimage.gaussian_filter(HIIresid, sigma=3.0, mode='nearest', axes=0)
                    if False:
                        plt.subplot(131)
                        plt.imshow(HIIresid, origin='lower', aspect='auto', vmin=-200, vmax=200, interpolation='nearest')
                        plt.subplot(132)
                        plt.imshow(HIIresidb, origin='lower', aspect='auto', vmin=-200, vmax=200, interpolation='nearest')
                        plt.subplot(133)
                        plt.imshow(HIIresidc, origin='lower', aspect='auto', vmin=-200, vmax=200, interpolation='nearest')
                        plt.show()
                    # Redo trace to be constant emission velocity
                    # objfrm = None if self._use_diff else HIIresid + bgfitted
                    objfrm = extfrm_use if self._redux_path == "/Users/rcooke/Work/Research/BBN/helium34/Absorption/2023_CRIRES_Survey/HD319718/2024-05-13/" else None
                    # allspecimg, allspecimg_dev = self.trace_tilt(spec.TRACE_SPAT.flatten(), trcnum=int(min(nwindow_left, nwindow_right)), plotit=False, objfrm=objfrm)
                    allspecimg, allspecimg_dev = self.trace_tilt(spec.TRACE_SPAT.flatten(), trcnum=int(max(nwindow_left, nwindow_right)), plotit=False, objfrm=None)
                    allspec = allspecimg.flatten()
                    # Fit the background emission
                    if mean_bg:
                        spec.BOX_R_PIX = 5.0
                        extract_boxcar(extfrm_use, ivar_use, gpm_img_new, allspecimg, bgfitted, spec)
                        thisboxpix = spec.BOX_WAVE.flatten()
                        bgfitted, gpm_img_new = self.mean_bg(HIIresid + bgfitted, gpm_img_new, thisboxpix, allspecimg, allspatimg, nwindow_left, nwindow_right, idx)
                    elif False:
                        # Use the residual image to fit the background
                        profimg = opimg.copy()
                        spl = self.smoothing_spline(allspecimg, extfrm_use-bgfitted, ivar_use, gpm_extwin, profimg)
                        # outspecimg = spl(allspecimg)
                        model = profimg * spl(np.arange(allspecimg.shape[0]))[:, None]
                        bgfitted = (extfrm_use - model)# * gpm_extwin
                        # Fit a smoothing spline to each column of the bgfitted image
                        minspat, maxspat = np.min(allspatimg[gpm_extwin]), np.max(allspatimg[gpm_extwin])
                        spatbins = np.linspace(minspat, maxspat, int(maxspat - minspat)//3)
                        for jj in range(spatbins.size - 1):
                            thisextwin = (allspatimg >= spatbins[jj]) & (allspatimg < spatbins[jj+1]) & gpm_extwin
                            # print(jj+1, spatbins.size-1, np.sum(thisextwin))
                            if np.sum(thisextwin) < 10: continue
                            splbg = self.smoothing_spline(allspecimg, bgfitted, np.ones(allspecimg.shape), thisextwin, np.ones(allspecimg.shape), lam=1.0E-2)
                            bgfitted[thisextwin] = splbg(allspecimg[thisextwin])
                        bgfitted = ndimage.median_filter(bgfitted, size=(5,5), mode='nearest')
                        bgfitted = ndimage.gaussian_filter(bgfitted, sigma=1.0, mode='nearest')
                        # if ii==numiter-1:
                        #     embed()
                        #     plt.imshow(bgfitted, origin='lower', aspect='auto', vmin=-200, vmax=200, interpolation='nearest')
                        #     plt.show()
                    else:
                        # bgfitted, gpm_img_new = self.iterate_bgfit(HIIresid+bgfitted, gpm_img_new, allspecimg, allspatimg, nwindow_left, nwindow_right, idx, trace_mask, plotit=False)#(ii==numiter-1))
                        bgfitted += HIIresid
                        # Convolve the bgfitted image with a Gaussian kernal along the spectral axis
                        smooth=False
                        if smooth:
                            bgfitted = ndimage.median_filter(bgfitted, size=(3, 3), mode='nearest')
                            bgfitted = ndimage.gaussian_filter(bgfitted, sigma=5.0, mode='nearest', axes=0)
                    # Should we save the background emission (only do this if the background emission has not been subtracted)?
                    if not self._step_subbg:
                        self.save_bgemission(bgfitted, idx)
                    # Redo object trace after removing the emission component
                    if self._prefix == "hd319718":
                        # Obtain two traces
                        trc_pos = self.trace_binary(extfrm_use - bgfitted, trc_pos_tmp=trccen)
                        # Find which of these two traces is closest to the original trace
                        dist1 = np.abs(trc_pos[0].TRACE_SPAT - spec.TRACE_SPAT)
                        dist2 = np.abs(trc_pos[1].TRACE_SPAT - spec.TRACE_SPAT)
                        spec = trc_pos[0] if np.mean(dist1) < np.mean(dist2) else trc_pos[1]
                    else:
                        # Mask out the regions with absorption features
                        obj_thismask = gpm_img_new.copy()
                        limsA, limsB = self.get_objprof_limits()
                        obj_thismask[int(limsA[1])+20:int(limsB[0])-20] = 0
                        spec = findobj_skymask.objs_in_slit(
                            extfrm_use - bgfitted, ivar_use, obj_thismask, ledge, redge,
                            ncoeff=self._polyord, boxcar_rad=3.0,
                            show_fits=self._plotit, nperslit=1)[0]
                    # Plot the new tracing results
                    if self._prefix == "hd319718" and plot_resid:
                        thisframe = extfrm_use - bgfitted
                        specpix = np.arange(thisframe.shape[self._specaxis])
                        plt.imshow(thisframe.T, vmin=-200, vmax=5000, origin='lower', aspect=thisframe.shape[self._specaxis]/thisframe.shape[1-self._specaxis])
                        plt.plot(specpix, spec.TRACE_SPAT, 'k--')
                        plt.show()

                    allspatimg = (spatimg - spec.TRACE_SPAT[np.newaxis,:].T)
                    allspat = allspatimg.flatten()
                    if False:
                        tmp = np.where(gpm_img_new.flatten() & (np.abs(allspat)<20))
                        plt.scatter(allspec[tmp], (HIIresid).flatten()[tmp], c=allspat[tmp], s=0.1)
                        plt.xlim(1650, 1750)
                        plt.ylim(0, 10000)
                        plt.show()
                        embed()
                        assert(False)
                    self.print_SNregions(outfluxbox)
                # plt.plot(outfluxbox)
                # xplot = np.arange(1338,1448)
                # modl = np.polyval(np.polyfit(xplot,spec_boxcar_flx[1338:1448],1), xplot)
                # SN_tmpA = np.mean(spec_boxcar_flx[1338:1448]) / np.std(spec_boxcar_flx[1338:1448]-modl)
                # xplot = np.arange(1706,1726)
                # modl = np.polyval(np.polyfit(xplot, spec_boxcar_flx[1706:1726],1), xplot)
                # SN_tmpB = np.mean(spec_boxcar_flx[1706:1726]) / np.std(spec_boxcar_flx[1706:1726]-modl)
                # print("(box) S/N = ", SNtmpA)
                # print("(box) S/N ab = ", SNtmpB)
            # One last iteration to get the boxcar extraction
            # HIIresid, outfluxbox, outfluxbox_err = self.iterate_objfit(extfrm_use - bgfitted, ivar_use, gpm_img_new, spec,
            #                                                            opspl, xloc, nbasis)

            # An error such as:
            # ValueError: Invalid combination of row and col input shapes.
            # in pypeit.core.moment.moment1d (when doing boxcar extraction) can be fixed by changing len(_row) to _row.ndim
            #profile_img = opspl(allspatimg)
            profile_img = opimg.copy()
            if subtesting:
                # Use this option to learn what the optimal number of pixels is
                subtst = np.arange(1, 15, dtype=float)
            else:
                # Insert the optimal value into the array below
                subtst = np.array([5.0])
            SN_spectmp, SN_abstmp = np.zeros(subtst.size), np.zeros(subtst.size)
            for br in range(subtst.size):
                spec.BOX_R_PIX = subtst[br]
                # print("about to extract")
                # embed()
                # assert False
                extract_boxcar(profile_img, ivar_use, gpm_img_new, allspecimg, np.zeros_like(profile_img), spec)
                boxwght = spec.BOX_COUNTS.copy().flatten()
                extract_boxcar(np.ones_like(profile_img), ivar_use, gpm_img_new, allspecimg, np.zeros_like(profile_img), spec)
                boxskywght = spec.BOX_COUNTS.copy().flatten()
                extract_boxcar(extfrm_use, ivar_use, gpm_img_new, allspecimg, bgfitted, spec)
                extract_optimal(extfrm_use, ivar_use, gpm_img_new, allspecimg, bgfitted, gpm_img_new, profile_img, spec, min_frac_use=0.05, base_var=None, count_scale=None, noise_floor=None)
                if False:
                    thisboxpix = spec.BOX_WAVE.flatten()
                    all_boxcar_waves = np.zeros((boxwght.size, self._numspec))
                    all_boxcar_specs = np.zeros((boxwght.size, self._numspec))
                    all_boxcar_sigma = np.zeros((boxwght.size, self._numspec))
                    all_boxcar_bgrnd = np.zeros((boxwght.size, self._numspec))
                    out_wave, raw_specs = self.comb_prep(use_corrected=False)
                    raw_wav, raw_flx, raw_err, bpm = self.comb_reject(out_wave, raw_specs, use_corrected=False)
                    thiswave = raw_wav[idx, :]
                    tmpnameAz = self._procpath + self._prefix + "_ALIS_spec{0:02d}_wzcorr.dat".format(idx)
                    out_waveAz, inwaveAz = np.loadtxt(tmpnameAz, unpack=True, usecols=(0, 1))
                    wA = np.where(np.in1d(thiswave, inwaveAz))
                    coeff = np.polyfit(thisboxpix[wA], out_waveAz, 1)
                    waveimg = np.polyval(coeff, allspecimg)
                    ref_exptime, _ = self.get_exptime(idx//2)
                    for sp in range(self._numspec):
                        this_exptime, _ = self.get_exptime(sp//2)
                        # convert pixel to wavelength for this spectrum
                        thiswave = raw_wav[sp, :]
                        tmpnameAz = self._procpath + self._prefix + "_ALIS_spec{0:02d}_wzcorr.dat".format(sp)
                        out_waveAz, inwaveAz = np.loadtxt(tmpnameAz, unpack=True, usecols=(0, 1))
                        wA = np.where(np.in1d(thiswave, inwaveAz))
                        coeff = np.polyfit(out_waveAz, thisboxpix[wA], 1)
                        this_allspecimg = np.polyval(coeff, waveimg)
                        # Load the bspline and the parameters needed for the fit
                        with open(self._procpath+'bgfitted_{0:02d}.knots'.format(sp), 'rb') as knots_file:
                            knots = pickle.load(knots_file)
                        evpix = (this_allspecimg > 1400.0) & (this_allspecimg < 1950) & (allspatimg > -nwindow_left) & (allspatimg < nwindow_right)
                        ev_spec, ev_spat = this_allspecimg[evpix], allspatimg[evpix]
                        idxs = np.where(evpix)
                        # Make a new background image
                        bgImage = np.zeros_like(HIIresid)
                        for ii in range(ev_spec.size):
                            bgImage[idxs[0][ii], idxs[1][ii]] = (ref_exptime/this_exptime)*interpolate.bisplev(ev_spec[ii], ev_spat[ii], knots)
                        extract_boxcar(extfrm_use, ivar_use, gpm_img_new, allspecimg, bgImage, spec)
                        all_boxcar_bgrnd[:, sp] = spec.BOX_COUNTS_SKY.flatten() * utils.inverse(boxskywght)
                        all_boxcar_specs[:, sp] = spec.BOX_COUNTS.flatten() * utils.inverse(boxwght)
                        all_boxcar_sigma[:, sp] = spec.BOX_COUNTS_SIG.flatten() * utils.inverse(boxwght)
                        all_boxcar_waves[:, sp] = spec.BOX_WAVE.flatten()
                    if False:
                        avg_wave = np.mean(all_boxcar_waves, axis=1)
                        avg_flux = np.mean(all_boxcar_specs, axis=1)
                        avg_disp = np.std(all_boxcar_specs, axis=1)
                        plt.fill_between(avg_wave, avg_flux-avg_disp, avg_flux+avg_disp, color='k', alpha=0.5)
                        for sp in range(self._numspec):
                            plt.plot(all_boxcar_waves[:, sp], all_boxcar_specs[:, sp], 'b-', drawstyle='steps-mid')
                        plt.plot(all_boxcar_waves[:, idx], all_boxcar_specs[:, idx], 'r-', drawstyle='steps-mid')
                        plt.plot(avg_wave, avg_flux, 'k-', drawstyle='steps-mid')
                        plt.show()
                    spec_boxcar_wav = all_boxcar_waves[:, idx]
                    spec_boxcar_flx = np.median(all_boxcar_specs, axis=1)
                    spec_boxcar_sig = all_boxcar_sigma[:, idx]
                    spec_boxcar_sky = np.median(all_boxcar_bgrnd, axis=1)
                else:
                    spec_boxcar_sky = spec.BOX_COUNTS_SKY.flatten() * utils.inverse(boxskywght)
                    spec_boxcar_flx = spec.BOX_COUNTS.flatten() * utils.inverse(boxwght)
                    spec_boxcar_sig = spec.BOX_COUNTS_SIG.flatten() * utils.inverse(boxwght)
                    spec_boxcar_wav = spec.BOX_WAVE.flatten()
                    spec_optimal_sky = spec.OPT_COUNTS_SKY.flatten()
                    spec_optimal_flx = spec.OPT_COUNTS.flatten()
                    spec_optimal_sig = spec.OPT_COUNTS_SIG.flatten()
                    spec_optimal_wav = spec.OPT_WAVE.flatten()
                # plt.plot(spec_boxcar_wav, spec_boxcar_flx, 'r-', drawstyle='steps-mid')
                # plt.plot(spec_optimal_wav, spec_optimal_flx, 'b-', drawstyle='steps-mid')
                # plt.show()
                SN_spectmp[br], SN_abstmp[br] = self.get_SNregions_fit(spec_boxcar_flx)
#            plt.show()
            SN_spec[tt] = np.max(SN_spectmp)
            SN_abs[tt] = np.max(SN_abstmp)
            this_SN_spec, this_SN_abs = self.get_SNregions_fit(spec_boxcar_flx)
            print("(box) S/N = ", this_SN_spec)
            print("(box) S/N ab = ", this_SN_abs)
        if testing or subtesting:
            if testing:
                plt.subplot(211)
                plt.plot(tst, SN_spec)
                plt.subplot(212)
                plt.plot(tst, SN_abs)
                plt.show()
            elif subtesting:
                plt.subplot(211)
                plt.plot(subtst, SN_spectmp)
                plt.subplot(212)
                plt.plot(subtst, SN_abstmp)
                plt.show()
            plt.plot(spec.BOX_WAVE.flatten(), spec_boxcar_flx, 'k-', drawstyle='steps-mid')
            plt.plot(spec.BOX_WAVE.flatten(), spec_boxcar_sig, 'r-', drawstyle='steps-mid')
            plt.plot(spec.BOX_WAVE.flatten(), spec_boxcar_sky, 'b-', drawstyle='steps-mid')
            plt.show()
        # Now fit background and object at the same time
        # COMMENTED OUT BECAUSE THESE VARIABLES ARE NOT USED FOR ANYTHING
        # HIIresid, outfluxbox, outfluxbox_err, outfluxopt, outfluxopt_err = self.iterate_objfit_chisq(extfrm_use-bgfitted, ivar_use, gpm_img_new, spec,
        #                                                                                              profile_img, nwindow_left, nwindow_right)

        # Calculate a new ivar based on the residuals
        # Make a smoothing spline through the data
        ivar_out = np.copy(ivar_use)
        spl = self.smoothing_spline(allspecimg, extfrm_use, ivar_use, gpm_img_new, profile_img)
        wnz = np.where(profile_img != 0.0)
        wslice = np.index_exp[wnz[0].min():wnz[0].max(), wnz[1].min():wnz[1].max()]
        outspecimg = spl(allspecimg)
        model = profile_img * spl(np.arange(allspecimg.shape[0]))[:, None]
        # model = profile_img * outspecimg
        if False:
            resid = (extfrm_use[wslice] - bgfitted[wslice] - model[wslice]) * np.sqrt(ivar_use[wslice])
            rsd_med, rsd_mad = np.zeros(resid.shape[1]), np.zeros(resid.shape[1])
            spat_bins = np.linspace(np.min(allspatimg[wnz]), np.max(allspatimg[wnz]), resid.shape[1]+1)
            rsdmask = np.zeros(ivar_out.shape, dtype=bool)
            for ii in range(2):
                scale_img = np.ones(ivar_out.shape)
                # Do two iterations of this. The second iteration just calculates for pixels that are significantly deviant
                for i in range(resid.shape[1]):
                    spatloc = (allspatimg[wslice] >= spat_bins[i]) & (allspatimg[wslice] < spat_bins[i+1])
                    wspat = np.where(spatloc)[0]
                    if wspat.size > 10:
                        rsd_med[i] = np.median(resid[wspat])
                        rsd_mad[i] = 1.4826 * np.median(np.abs(resid[wspat] - rsd_med[i]))
                    # Find which pixels are significantly discrepant, and inflate the variance there
                    if ii == 0:
                        wbad = np.where((np.abs(resid - rsd_med[i]) > 2.0 * rsd_mad[i]) & spatloc & (profile_img[wslice] != 0.0) & gpm_img_new[wslice])
                    else:
                        wbad = np.where(rsdmask[wslice] & spatloc & (profile_img[wslice] != 0.0) & gpm_img_new[wslice])
                    scalefactor = np.minimum(1, (0.5*rsd_mad[i]/resid[wbad])**2)
                    scale_img[wslice][wbad] = scalefactor
                scale_img = np.maximum(scale_img, 0.01)  # Don't reduce by more than 100x
                # Calculate the residual mask to use for the second iteration
                rsdmask = ndimage.binary_dilation(ndimage.binary_erosion(scale_img!=1), iterations=4).astype(bool)
            ivar_out *= scale_img
        elif True:
            # Scale the errors by sqrt(2) based on where the object profile is set to the data values, so that it encompasses uncertainty in the object profile, too..
            # plt.imshow(opdat)
            # plt.show()
            ivar_out[opdat] *= 0.5
        # plt.subplot(121)
        # plt.imshow(scale_img)
        # plt.subplot(122)
        # plt.imshow(scale_img)
        # plt.show()
        # Extract again
        if True:
            print("Final extraction with updated variance")
            normfact = opnrmsub.flatten()/np.median(opnrmsub.flatten()[opnrmsub.flatten()>0])
            normfact_box = opnrmsub.flatten() * utils.inverse(opnrm.flatten())
            extract_boxcar(extfrm_use, ivar_out, gpm_img_new, allspecimg, bgfitted, spec)
            extract_optimal(extfrm_use, ivar_out, gpm_img_new, allspecimg, bgfitted, gpm_img_new, profile_img, spec, min_frac_use=0.05, base_var=None, count_scale=None, noise_floor=None)
            spec_boxcar_sky = spec.BOX_COUNTS_SKY.flatten() * utils.inverse(boxskywght) * normfact_box
            spec_boxcar_flx = spec.BOX_COUNTS.flatten() * utils.inverse(boxwght) * normfact_box
            spec_boxcar_sig = spec.BOX_COUNTS_SIG.flatten() * utils.inverse(boxwght) * normfact_box
            spec_boxcar_wav = spec.BOX_WAVE.flatten()
            spec_optimal_sky = spec.OPT_COUNTS_SKY.flatten()
            spec_optimal_flx = spec.OPT_COUNTS.flatten() * normfact
            spec_optimal_sig = spec.OPT_COUNTS_SIG.flatten() * normfact
            spec_optimal_wav = spec.OPT_WAVE.flatten()

        # Plot the residual images
        if plot_resid:
            # embed()
            # assert False
            # outspecimgb = np.interp(allspecimg, np.arange(spec_optimal_flx.size), spec_optimal_flx)
            modelb = profile_img*spec_optimal_flx.reshape((spec_optimal_flx.size, 1))
            # model = profile_img*spec_optimal_flx.reshape((spec_optimal_flx.size, 1))
            modmax = np.max(model)

            if False:
                # Difference images:
                tmp_dat = extfrm_use[wslice]
                tmp_mod = profile_img[wslice]
                norm = np.max(tmp_dat, axis=1) / np.max(tmp_mod, axis=1)
                diff = tmp_dat - tmp_mod * norm[:,None]
                plt.subplot(131)
                plt.imshow(tmp_dat[829-100:829+100,:], vmin=0, vmax=modmax)
                plt.subplot(132)
                plt.imshow(diff[829 - 100:829 + 100, :], vmin=-0.03*modmax, vmax=0.03*modmax)
                plt.subplot(133)
                rat = tmp_dat*utils.inverse(model[wslice]/modmax)
                plt.imshow(rat[829 - 100:829 + 100, :], vmin=0.9*modmax, vmax=1.1*modmax)
                plt.show()
                plt.plot(tmp_dat[829-100:829+100,:].flatten(), diff[829 - 100:829 + 100, :].flatten(), 'bx')
                plt.show()

                # Update the profile image
                profile_img_new = profile_img.copy() * (outspecimg * utils.inverse(spec_optimal_flx[:, None]))
                # Plot to check
                idchks = [800,828]
                idchks = [880, 881, 882, 883, 884, 885, 886, 887]
                for idchk in idchks:
                    tmp_modb = profile_img_new[wslice]
                    plt.subplot(221)
                    norm = np.max(tmp_dat[idchk, :]) / np.max(tmp_mod[idchk, :])
                    plt.plot(tmp_dat[idchk, :], 'b-')
                    plt.plot(tmp_mod[idchk, :]*norm, 'r-')
                    plt.plot(tmp_modb[idchk, :]*norm, 'g-')
                    plt.subplot(223)
                    plt.plot(tmp_dat[idchk, :] - tmp_mod[idchk, :]*norm, 'r-')
                    plt.plot(tmp_dat[idchk, :] - tmp_modb[idchk, :]*norm, 'g-')
                    idchk += 5
                    norm = np.max(tmp_dat[idchk, :]) / np.max(tmp_mod[idchk, :])
                    plt.subplot(222)
                    plt.plot(tmp_dat[idchk, :], 'b-')
                    plt.plot(tmp_mod[idchk, :] * norm, 'r-')
                    plt.plot(tmp_modb[idchk, :] * norm, 'g-')
                    plt.subplot(224)
                    plt.plot(tmp_dat[idchk, :] - tmp_mod[idchk, :]*norm, 'r-')
                    plt.plot(tmp_dat[idchk, :] - tmp_modb[idchk, :]*norm, 'g-')
                    plt.show()

            plt.subplot(181)
            plt.imshow(extfrm_use[wslice], origin='lower', cmap='gray', aspect=0.3, vmin=-3*np.median(full_bg[wslice]), vmax=30*np.median(full_bg[wslice]))
            plt.subplot(182)
            plt.imshow(profile_img[wslice], origin='lower', cmap='gray', aspect=0.3, vmin=0, vmax=np.max(profile_img))
            plt.subplot(183)
            plt.imshow(model[wslice], origin='lower', cmap='gray', aspect=0.3, vmin=-3*np.median(full_bg[wslice]), vmax=30*np.median(full_bg[wslice]))
            plt.subplot(184)
            plt.imshow(full_bg[wslice], origin='lower', cmap='gray', aspect=0.3, vmin=-3*np.median(full_bg[wslice]), vmax=3*np.median(full_bg[wslice]))
            plt.subplot(185)
            madbg = 1.4826*np.median(np.abs(np.median(bgfitted[wslice])-bgfitted[wslice]))
            plt.imshow(bgfitted[wslice], origin='lower', cmap='gray', aspect=0.3, vmin=-5*madbg, vmax=5*madbg)
            plt.subplot(186)
            plt.imshow((extfrm_use[wslice] - bgfitted[wslice] - model[wslice])*np.sqrt(ivar_use[wslice]), origin='lower', cmap='gray', aspect=0.3, vmin=-3, vmax=3)
            plt.subplot(187)
            plt.imshow((extfrm_use[wslice] - bgfitted[wslice] - modelb[wslice])*np.sqrt(ivar_use[wslice]), origin='lower', cmap='gray', aspect=0.3, vmin=-3, vmax=3)
            plt.subplot(188)
            plt.imshow((extfrm_use[wslice] - bgfitted[wslice] - model[wslice])*np.sqrt(ivar_out[wslice]), origin='lower', cmap='gray', aspect=0.3, vmin=-3, vmax=3)
            plt.show()

            plt.clf()
            wsliceb = np.index_exp[1650-125:1650+125, wnz[1].min():wnz[1].max()]
            plt.subplot(231)
            plt.imshow((extfrm_use[wsliceb] - bgfitted[wsliceb] - model[wsliceb])*np.sqrt(ivar_use[wsliceb]), origin='lower', cmap='gray', aspect=0.3, vmin=-3, vmax=3)
            plt.subplot(234)
            plt.imshow((extfrm_use[wsliceb] - bgfitted[wsliceb] - modelb[wsliceb])*np.sqrt(ivar_use[wsliceb]), origin='lower', cmap='gray', aspect=0.3, vmin=-3, vmax=3)
            plt.subplot(132)
            plt.imshow((extfrm_use[wslice] - bgfitted[wslice] - model[wslice])*np.sqrt(ivar_use[wslice]), origin='lower', cmap='gray', aspect=0.2, vmin=-3, vmax=3)
            plt.subplot(133)
            plt.imshow((extfrm_use[wslice] - bgfitted[wslice] - modelb[wslice])*np.sqrt(ivar_use[wslice]), origin='lower', cmap='gray', aspect=0.2, vmin=-3, vmax=3)
            plt.show()
            # embed()
            # assert False
            if False:
                rr = 828
                plt.plot(extfrm_use[wslice][rr,:], 'k-')
                plt.plot(bgfitted[wslice][rr,:], 'b-')
                plt.plot(model[wslice][rr,:], 'g-')
                plt.plot(bgfitted[wslice][rr,:]+model[wslice][rr,:], 'r-')
                plt.show()

            plt.subplot(211)
            plt.plot(np.arange(spec_optimal_flx.size), spec_optimal_flx, drawstyle='steps-mid', label='optimal')
            plt.plot(spec.BOX_WAVE.flatten(), spec_boxcar_flx, 'k-', drawstyle='steps-mid', label='boxcar')
            plt.legend()
            plt.subplot(212)
            plt.plot(np.arange(spec_optimal_flx.size), spec_optimal_flx-spec_boxcar_flx, color='k', drawstyle='steps-mid', label='optimal-boxcar')
            plt.plot(np.arange(spec_optimal_flx.size), spec_optimal_sig, color='r', drawstyle='steps-mid', label='optimal sigma')
            plt.plot(np.arange(spec_optimal_flx.size), spec_boxcar_sig, color='b', drawstyle='steps-mid', label='boxcar sigma')
            plt.legend()
            plt.show()

        # plt.plot(spec_boxcar_flx, 'k-', drawstyle='steps-mid')
        # plt.show()
        # Save the extracted spectrum
        # skytxt = ""
        # if self._ext_sky: skytxt = "_sky"
        # outPath = self._altpath
        # if self._use_diff: outPath = self._procpath
        #outPath = self._procpath
        #outA = outPath + "spec1d_{0:02d}.dat".format(idx)
        # np.savetxt(outA, np.column_stack((spec_boxcar_wav, spec_boxcar_flx, spec_boxcar_sig, spec_boxcar_sky)))
        # np.savetxt(outA, np.column_stack((spec_optimal_wav, spec_optimal_flx, spec_optimal_sig, spec_optimal_sky)))
        #np.savetxt(outA, np.column_stack((spec_optimal_wav, outfluxopt, outfluxopt_err, spec_optimal_sky)))
        if False:
            embed()
            # Now do the extraction
            skymodel, objmodel, ivarmodel, extractmask = skysub.local_skysub_extract(
                extfrm_use.astype(float), ivar_use, tilts, allspecimg,
                bgfitted, gpm_img_new, ledge, redge,
                spec, ingpm=gpm_img_new,
                spat_pix=None,
                model_full_slit=False,
                sigrej=5.0,
                model_noise=False,  # base_var=basevar,
                bsp=0.1,
                std=False,
                adderr=0.0002,
                force_gauss=False,
                sn_gauss=4,
                show_profile=self._plotit,
                use_2dmodel_mask=False,
                no_local_sky=False)
            plt.subplot(211)
            plt.plot(spec['BOX_WAVE'][0, :], spec['BOX_COUNTS'][0, :] * utils.inverse(boxwght), 'k-', drawstyle='steps-mid')
            plt.plot(spec['BOX_WAVE'][0, :], spec['BOX_COUNTS_SKY'][0, :], 'b-', drawstyle='steps-mid')
            plt.subplot(212)
            plt.plot(spec['OPT_WAVE'][0, :], spec['OPT_COUNTS'][0, :], 'r-', drawstyle='steps-mid')
            plt.plot(spec['OPT_WAVE'][0, :], spec['OPT_COUNTS_SKY'][0, :], 'g-', drawstyle='steps-mid')
            plt.show()
        # xspec1d = XSpectrum1D.from_tuple((spec_optimal_wav, outfluxopt, outfluxopt_err), verbose=False)  # This one accounts for background
        # xspec1d = XSpectrum1D.from_tuple((spec_optimal_wav, spec_optimal_flx, spec_optimal_sig), verbose=False)  # This one assumes no background
        #
        # Use the boxcar sig, since it basically agrees with the optimal sig, but accounts for the differences
        # when the flux gradient is large, and the optimal profile fails to capture the correct flux in these regions.
        # xspec1d = XSpectrum1D.from_tuple((spec_optimal_wav, spec_optimal_flx, spec_boxcar_sig), verbose=False)  # This one assumes no background
        if False:
            # Plot the spectra to see if the scaling has been correctly applied
            embed()
            # nrmfact = opnrmsub.flatten() * utils.inverse(opnrm.flatten())
            nrmfact = np.ones(spec_optimal_wav.size)
            # nrmfact = opnrm.flatten()# * utils.inverse(boxwght.flatten())
            plt.plot(spec_optimal_wav, spec_optimal_flx, 'k-', drawstyle='steps-mid')
            plt.plot(spec_boxcar_wav, spec_boxcar_flx*nrmfact, 'r-', drawstyle='steps-mid')
            plt.plot(spec_optimal_wav, nrmfact*np.median(spec_optimal_flx)/np.median(nrmfact), 'b-', drawstyle='steps-mid')
            plt.show()
        xspec1d = XSpectrum1D.from_tuple((spec_optimal_wav, spec_optimal_flx, spec_optimal_sig), verbose=False)  # This one assumes no background
        return spec_optimal_flx, outspecimg, bgfitted, xspec1d, ivar_out
        # return spec_optimal_flx, bgfitted, xspec1d, ivar_out

    def trace_binary(self, frame, trc_pos_tmp=None):
        """
        Trace two objects in a frame assuming there are two peaks per row
        """
        ppos_fit = np.zeros((frame.shape[self._specaxis], 2))
        for rc in range(frame.shape[self._specaxis]):
            pks, pkshgt = signal.find_peaks(frame[rc, :], height=100)
            # Check there are two peaks
            if len(pks) < 2:
                continue
            peakind = np.argsort(pkshgt['peak_heights'])[-2:]
            peakpos = pks[peakind]
            # Of these, sort by the peak position
            peakpos = np.sort(peakpos)
            # Fit a low order polynomial around the peaks
            spatarr = np.arange(frame.shape[1 - self._specaxis])
            for pp in range(2):
                fitind = np.where((spatarr >= peakpos[pp] - 2.0) & (spatarr <= peakpos[pp] + 2.0))[0]
                pcoeff = np.polyfit(spatarr[fitind], frame[rc, fitind], 2)
                ppos_fit[rc, pp] = -0.5 * pcoeff[1] / pcoeff[0]
        # Make two new traces, fit with a polynomial order
        trc_pos = trc_pos_tmp
        for tt in range(2):
            # median filter to get rid of bad points
            ppos_fit[:, tt] = ndimage.median_filter(ppos_fit[:, tt], size=11)
            # fit with a robust polynomial
            result = robust_fit(np.arange(frame.shape[self._specaxis]), ppos_fit[:, tt], self._polyord,
                                in_gpm=ppos_fit[:, tt] != 0, maxdev=5.0)
            trc_pos[tt].TRACE_SPAT = result.eval(np.arange(frame.shape[self._specaxis]))
        return trc_pos

    def step_trace(self):
        print("entering :: step_trace()")
        st_time = time.time()
        # Obtain a mask of the bad pixels
        fil = fits.open(self._datapath + self._matches[0][0][0])
        frm = fil[1].data[self._slice].T
        frm_filt = ndimage.median_filter(frm, size=(7, 1))
        bpm = np.abs(frm - frm_filt) > 1000
        # Perform the object trace and extraction
        all_traces = []
        for ff in range(self._numframes):
            # if ff not in [18]:  # 18 is wavy, 6 has strange features
            #     continue
            raw_specs = []
            nmtch = 1#len(self._matches[ff][1])
            for xx in range(nmtch):
                # Load the trace frame
                trname = self.get_trace(ff)
                trframe = fits.open(trname)[0].data.T.astype(float)
                trframe *= self._gain
                if self._use_diff:
                    if True:
                        difnstr = self._diff_name.format(ff)
                        print("extracting", difnstr)
                        frame = fits.open(difnstr)[0].data.T.astype(float)
                        frame *= self._gain
                        frame2 = -frame.copy()
                        framesum = fits.open(self._maxd_name.format(ff))[0].data.T
                        framesum *= self._gain
                        # Calculate the readnoise
                        rnfrm = fits.open(difnstr)[0].data.astype(float) * self._gain
                        statpix = np.append(rnfrm[:4, :].flatten(), rnfrm[-4:, :].flatten())
                        ronoise = 1.4826 * np.median(np.abs(statpix - np.median(statpix)))
                        ronoise2 = ronoise
                        print("RON  mean, std = ", np.mean(statpix), np.std(statpix))
                        print("RON  median, 1.4826*MAD = ", np.median(statpix), ronoise)
                    else:
                        # This was used before, but it's got cut prefix not diff prefix... why is that?
                        fn_frtmp = self._procpath + "cut_" + self._matches[ff][0][0]
                        fn_subfr = self._procpath + "cut_" + self._matches[ff][1][xx]
                        frtmp = fits.open(fn_frtmp)[0].data.T.astype(float)
                        subfr = fits.open(fn_subfr)[0].data.T.astype(float)
                        print("extracting combination ::")
                        print(fn_frtmp)
                        print(fn_subfr)
                        print("-----")
                        frame = frtmp - subfr
                        # ttttrjc = fits.PrimaryHDU(frame)
                        # ttttrjc.writeto("testing.fits")
                        frame *= self._gain
                        framesum = np.max(np.dstack((frtmp, subfr)), axis=2)
                        # Calculate the readnoise
                        rnfrm = frame.copy().T
                        statpix = np.append(rnfrm[:4, :].flatten(), rnfrm[-4:, :].flatten())
                        ronoise = 1.4826 * np.median(np.abs(statpix - np.median(statpix)))
                        ronoise2 = ronoise
                        print("RON  mean, std = ", np.mean(statpix), np.std(statpix))
                        print("RON  median, 1.4826*MAD = ", np.median(statpix), ronoise)
                else:
                    assert(False)  # Not implemented yet
                    frm1 = self._cut_name.format(2 * ff)
                    frm2 = self._cut_name.format(2 * ff + 1)
                    print("Reducing... " + frm1 + " and " + frm2)
                    frame = fits.open(frm1)[0].data.T.astype(float)
                    frame2 = fits.open(frm2)[0].data.T.astype(float)
                    frame *= self._gain
                    frame2 *= self._gain
                    framesum = frame.copy()
                    # Calculate the readnoise
                    rnfrm = fits.open(frm1)[0].data * self._gain
                    statpix = np.append(rnfrm[:4, :].flatten(), rnfrm[-4:, :].flatten())
                    ronoise = 1.4826 * np.median(np.abs(statpix - np.median(statpix)))
                    # print("RON1  mean, std = ", np.mean(statpix), np.std(statpix))
                    print("RON1  median, 1.4826*MAD = ", np.median(statpix), ronoise)
                    rnfrm = fits.open(frm2)[0].data * self._gain
                    statpix = np.append(rnfrm[:4, :].flatten(), rnfrm[-4:, :].flatten())
                    ronoise2 = 1.4826 * np.median(np.abs(statpix - np.median(statpix)))
                    # print("RON2  mean, std = ", np.mean(statpix), np.std(statpix))
                    print("RON2  median, 1.4826*MAD = ", np.median(statpix), ronoise2)

                # Prepare an inverse variance image
                datasec_img = np.ones_like(frame)
                rn2img = procimg.rn2_frame(datasec_img, ronoise)
                darkcurr = 0.03
                exptime, etim = self.get_exptime(ff)
                msflat = fits.open(self._masterflat_name)[0].data.T
                basevar = procimg.base_variance(rn2img, darkcurr=darkcurr, exptime=exptime)
                if self._use_diff:
                    frame_for_ivar = framesum
                else:
                    msdark = fits.open(self.get_darkname(self._masterdark_name, etim))[0].data.T
                    frame_for_ivar = ((frame / self._gain) + msdark) * self._gain
                rawvarframe = procimg.variance_model(basevar, frame_for_ivar)
                # Ivar
                ivar = utils.inverse(rawvarframe)
                # Flatfield the data (note, this needs to be done after calculating the RO noise)
                frame /= msflat
                trframe /= msflat
                # Set up some arrays
                waveimg = np.arange(frame.shape[0])[:, np.newaxis].repeat(frame.shape[1], axis=1)
                spatimg = np.arange(frame.shape[1])[:, np.newaxis].repeat(frame.shape[0], axis=1).T
                tilts = waveimg / (frame.shape[0] - 1)
                global_sky = np.zeros(frame.shape)
                # Set the masks
                thismask = np.ones(frame.shape, dtype=bool) & np.logical_not(bpm)
                ingpm = thismask.copy()
                # Find an estimate of the slit edges
                cen = 99 - 35.0 * np.linspace(0, 1, frame.shape[self._specaxis]) + 56 + 13
                # trc_edg, _ = findobj_skymask.objs_in_slit(
                #         frame, thismask,
                #         np.zeros(frame.shape[self._specaxis]), np.ones(frame.shape[self._specaxis])*frame.shape[1-self._specaxis],
                #         has_negative=True, ncoeff=self._polyord,
                #         show_fits=plotit, nperslit=1)
                # if ff>=10:
                #     if len(trc_edg) == 0:
                #         cen = 99 - 35.0*np.linspace(0,1,frame.shape[self._specaxis]) + 56
                #     else:
                #         cen = trc_edg[0].TRACE_SPAT + 56
                # else:
                #     cen = trc_edg[0].TRACE_SPAT + 35
                ledge = cen - 85
                redge = cen + 85
                boxcar_rad = 3.0#10.0
                # plt.imshow(frame, vmin=-1000, vmax=1000)
                # plt.plot(ledge, np.arange(ledge.size), 'r-')
                # plt.plot(redge, np.arange(redge.size), 'r-')
                # plt.show()
                # frame_filt = ndimage.median_filter(frm, size=(7, 1))
                # frame_filt = ndimage.gaussian_filter(frame_filt, sigma=1.0)
                numtrc = 1
                if self._prefix == "hd319718":
                    numtrc = 2
                # Mask out the regions with absorption features
                limsA, limsB = self.get_objprof_limits()
                thismask[int(limsA[1]):int(limsB[0])] = 0
                # Now perform the trace
                trc_pos_tmp = findobj_skymask.objs_in_slit(
                    trframe, ivar, thismask, ledge, redge,
                    ncoeff=self._polyord, boxcar_rad=boxcar_rad,
                    show_fits=self._plotit, nperslit=numtrc)
                if trname != difnstr and numtrc == 1:
                    # Redfine the edges based on the expected trace of the object
                    new_ledge = trc_pos_tmp[0].TRACE_SPAT - (np.median(trc_pos_tmp[0].TRACE_SPAT - ledge))
                    new_redge = trc_pos_tmp[0].TRACE_SPAT - (np.median(trc_pos_tmp[0].TRACE_SPAT - redge))
                    trc_pos = findobj_skymask.objs_in_slit(
                        frame, ivar, thismask, new_ledge, new_redge,
                        ncoeff=self._polyord, boxcar_rad=boxcar_rad,
                        show_fits=self._plotit, nperslit=numtrc, std_trace=trc_pos_tmp[0].TRACE_SPAT)
                elif trname != difnstr:
                    # Redfine the edges based on the expected trace of the object
                    new_ledge = trc_pos_tmp[0].TRACE_SPAT - (np.median(trc_pos_tmp[0].TRACE_SPAT - ledge))
                    new_redge = trc_pos_tmp[0].TRACE_SPAT - (np.median(trc_pos_tmp[0].TRACE_SPAT - redge))
                    trc_pos_fid = findobj_skymask.objs_in_slit(
                        frame, ivar, thismask, new_ledge, new_redge,
                        ncoeff=self._polyord, boxcar_rad=boxcar_rad,
                        show_fits=self._plotit, nperslit=1, std_trace=trc_pos_tmp[0].TRACE_SPAT)
                    if len(trc_pos_fid) != numtrc:
                        # This code was written with HD 319718 in mind
                        if self._prefix != "hd319718":
                            raise ValueError("This code is only for HD319718 -- Number of traces do not match!")
                        if False:
                            # OLD METHOD
                            # Register the frame according to the trace position
                            relspat = spatimg - trc_pos_fid[0].TRACE_SPAT[np.newaxis, :].T
                            # median filter the frames to reduce noise
                            frame_filt = ndimage.median_filter(frame, size=(5, 1))
                            # frame_filt = ndimage.gaussian_filter(frame_filt, sigma=1.0)
                            # Compute a histogram of the spatial positions
                            whist = np.where((np.abs(relspat) < 20) & (np.abs(waveimg-frame.shape[0]//2) < 200))
                            bins = np.arange(-20, 20, 0.1)
                            histprof, bin_edges = np.histogram(relspat[whist], bins=bins, weights=frame_filt[whist])
                            normprof, bin_edges = np.histogram(relspat[whist], bins=bins)
                            histprof *= utils.inverse(normprof)
                            # Smooth the profile
                            histprof = ndimage.gaussian_filter1d(histprof, sigma=2.0)
                            # Find the second trace position
                            pks, pkshgt = signal.find_peaks(histprof, height=100)
                            # Check there are two peaks
                            if len(pks) < 2:
                                raise ValueError("Could not find second trace for HD319718!")
                            peakind = np.argsort(pkshgt['peak_heights'])[-2:]
                            peakpos = pks[peakind]
                            # Fit a low order polynomial around the peaks
                            ppos_fit = np.zeros(2)
                            for pp in range(2):
                                fitind = np.where((bin_edges[:-1] >= bin_edges[peakpos[pp]]-2.0) &
                                                  (bin_edges[1:] <= bin_edges[peakpos[pp]]+2.0))[0]
                                pcoeff = np.polyfit(0.5*(bin_edges[fitind]+bin_edges[fitind+1]),
                                                    histprof[fitind], 2)
                                ppos_fit[pp] = -0.5 * pcoeff[1]/pcoeff[0]
                            # Make two new traces, offset by these values
                            trc_pos = trc_pos_tmp
                            trc_pos[0].TRACE_SPAT = trc_pos_fid[0].TRACE_SPAT + ppos_fit[0]
                            trc_pos[1].TRACE_SPAT = trc_pos_fid[0].TRACE_SPAT + ppos_fit[1]
                        else:
                            # NEW METHOD -- recalculate the trace of two objects simultaneously
                            # For each for in the data, find two peaks in the spatial profile
                            trc_pos = self.trace_binary(frame, trc_pos_tmp=trc_pos_tmp)
                    else:
                        trc_pos = trc_pos_fid
                else:
                    trc_pos = trc_pos_tmp
                if self._plotit or numtrc == 2:
                    spec = np.arange(frame.shape[self._specaxis])
                    if self._prefix == "wray15199":
                        plt.imshow(frame.T, vmin=-2, vmax=50, origin='lower', aspect=frame.shape[self._specaxis]/frame.shape[1-self._specaxis])
                    else:
                        plt.imshow(frame.T, vmin=-200, vmax=5000, origin='lower', aspect=frame.shape[self._specaxis]/frame.shape[1-self._specaxis])
                    for sss in range(len(trc_pos)):
                        plt.plot(spec, trc_pos[sss].TRACE_SPAT, 'k--')
                        plt.plot(spec, trc_pos[sss].TRACE_SPAT + boxcar_rad, 'b-')
                        plt.plot(spec, trc_pos[sss].TRACE_SPAT - boxcar_rad, 'r-')
                        # plt.plot(spec, trc_neg[0].TRACE_SPAT + boxcar_rad, 'b-')
                        # plt.plot(spec, trc_neg[0].TRACE_SPAT - boxcar_rad, 'r-')
                    plt.plot(spec, trc_pos[0].TRACE_SPAT - 40, 'g-')
                    plt.plot(spec, trc_pos[0].TRACE_SPAT - 100, 'g-')

                    # plt.plot(spec, trc_posb[0].TRACE_SPAT + boxcar_rad, 'b--')
                    # plt.plot(spec, trc_posb[0].TRACE_SPAT - boxcar_rad, 'r--')
                    # plt.plot(spec, trc_posb[0].TRACE_SPAT - 40, 'g--')
                    # plt.plot(spec, trc_posb[0].TRACE_SPAT - 100, 'g--')
                    plt.show()

                all_traces.append(trc_pos)
                # all_traces.append(trc_neg)
                if self._step_extract:
                    # Optimal Extraction
                    skymodel = np.zeros_like(frame)
                    objmodel = np.zeros_like(frame)
                    ivarmodel = np.zeros_like(frame)
                    extractmask = np.zeros_like(frame)
                    if self._prefix == "hd319718":
                        trcs = [trc_pos[0], trc_pos[1]]#, trc_neg[0]]
                    else:
                        trcs = [trc_pos[0]]#, trc_neg[0]]
                    # store traces for this frame, to be sent to basis fit.
                    if numtrc == 2:
                        trccen = trcs
                    elif numtrc == 1:
                        trccen = None
                    else:
                        trccen = None
                        print("Not ready for this number of traces!")
                        assert (False)
                    # Loop over traces
                    isstd = False  # True
                    if self._ext_sky: isstd = False
                    print("There are {} traces to extract".format(len(trcs)))
                    for tt in range(len(trcs)):
                        # if ff == 5 and tt == 1: pass
                        # else: continue
                        if self._ext_sky:
                            ee = 1 - tt
                        else:
                            ee = tt
                        # Loop over the correct frame and corresponding ivar image
                        # ivar_use = ndimage.median_filter(ivar, size=(7, 1))
                        ivar_use = ivar
                        extfrm_use = frame
                        # Are we doing basis fitting
                        if self._step_basis:
                            extfrm_use_nrm, ivar_use_nrm = extfrm_use.copy(), ivar_use.copy()
                            numiterfit = 15
                            # embed()
                            # assert False
                            bgspec = stats.sigma_clipped_stats(extfrm_use, axis=1)[1]
                            # bgspec = np.ma.median(extfrm_use, axis=1)
                            # bgspec = np.median(extfrm_use, axis=1)
                            # Apply a median filter to the background spectrum
                            bgfilt = signal.medfilt(bgspec, 25)
                            bgfitted = np.tile(bgfilt, (extfrm_use.shape[1], 1)).T
                            # Make a smoothing spline of the background pixels
                            if False:  # No improvement from just using median.
                                med = np.median(extfrm_use)
                                mad = 1.4826*np.median(np.abs(extfrm_use-bgfitted))
                                ssgpm = np.abs(extfrm_use-bgfitted) < 0.5*mad
                                ssgpm = ndimage.binary_erosion(ssgpm, iterations=2)
                                bgivar = np.ones_like(tilts)
                                splbg = self.smoothing_spline(tilts, extfrm_use-bgfitted, bgivar, ssgpm, np.ones_like(tilts), lam=1.0E-5)
                                bgimg = splbg(tilts)+bgfitted
                                plt.subplot(151)
                                plt.imshow(ssgpm)
                                plt.subplot(152)
                                plt.imshow(extfrm_use, vmin=med-mad, vmax=med+mad)
                                plt.subplot(153)
                                plt.imshow(bgimg, vmin=med-mad, vmax=med+mad)
                                plt.subplot(154)
                                plt.imshow(extfrm_use-bgimg, vmin=-mad, vmax=mad)
                                plt.subplot(155)
                                plt.imshow(extfrm_use-bgfitted, vmin=-mad, vmax=mad)
                                plt.show()
                                plt.hist((extfrm_use-bgfitted)[ssgpm], bins=100, range=(-mad, mad), color='b', alpha=0.5)
                                plt.hist((extfrm_use-bgimg)[ssgpm], bins=100, range=(-mad, mad), color='r', alpha=0.5)
                                plt.show()
                            objspec = None
                            ivar_send = ivar_use.copy()
                            bgfrac_adjust = 1.0  #0.1
                            for it in range(numiterfit):
                                if it == 0:
                                    bgfrac_adjust = 0.1
                                elif it >= 3:
                                    bgfrac_adjust = 1.0
                                objspec, objspec_img, bgfitted_new, xspec1d, ivar_send = self.basis_fit(extfrm_use.copy()-bgfitted, ivar_send, tilts, waveimg, spatimg, trcs[ee], 2 * ff + tt, extfrm_use_nrm, ivar_use_nrm, bgfitted, edges=[ledge, redge], fullprof=it!=0, plot_resid=(it==numiterfit-1), inspec=objspec, trccen=trccen)
                                # objspec, objspec_img, bgfitted_new, xspec1d, ivar_send = self.basis_fit(extfrm_use.copy()-bgfitted, ivar_send, tilts, waveimg, spatimg, trcs[ee], 2 * ff + tt, extfrm_use_nrm, ivar_use_nrm, bgfitted, edges=[ledge, redge], fullprof=it!=0, plot_resid=False, inspec=objspec, trccen=trccen)
                                if it <= 20:
                                    # For the low iterations, don't change the inverse variance
                                    ivar_send = np.copy(ivar_use)
                                extfrm_use_nrm, ivar_use_nrm = extfrm_use.copy()-bgfitted-bgfrac_adjust*bgfitted_new, ivar_send.copy()
                                extfrm_use_nrm *= utils.inverse(objspec_img)
                                ivar_use_nrm *= objspec_img**2
                                bgfitted += bgfrac_adjust*bgfitted_new
                            raw_specs.append(xspec1d)
                            # embed()
                            # plt.plot(xspec1d.wavelength, xspec1d.flux, 'k-')
                            # plt.plot(xspec1d.wavelength, xspec1d.sig, 'r-')
                            # plt.show()
                            continue
                        else:
                            # Identify salt and pepper pixels with a median filter
                            trcs[ee].BOX_R_PIX = 5.0
                            ii, nmask, nnew = 0, 0, -1
                            extfrm_use_med = extfrm_use.copy()
                            ingpm = thismask.copy()
                            while (nnew != 0):
                                medfilt = medfilt2d(extfrm_use_med, kernel_size=(7, 1))
                                madfilt = 1.4826 * medfilt2d(np.abs(extfrm_use_med - medfilt), kernel_size=(7, 1))
                                wbad = np.where(ingpm & (np.abs((extfrm_use_med - medfilt) * utils.inverse(madfilt)) > 10))
                                extfrm_use_med = medfilt
                                ingpm[wbad] = False
                                nnew = wbad[0].size
                                nmask += nnew
                                ii += 1
                                print(f"Iteration {ii} :: Number of new bad pixels = {nnew}... total number of masked pixels = {nmask}")
                            # embed()
                            # assert(False)
                            # Now do the extraction
                            skymodel[thismask], objmodel[thismask], ivarmodel[thismask], extractmask[
                                thismask] = skysub.local_skysub_extract(
                                extfrm_use, ivar_use, tilts, waveimg,
                                global_sky, thismask, ledge, redge,
                                trcs[ee], ingpm=ingpm,
                                spat_pix=None,
                                model_full_slit=False,
                                sigrej=5.0,
                                model_noise=False,  # base_var=basevar,
                                bsp=0.5,
                                std=isstd,
                                adderr=0.0002,
                                force_gauss=False,
                                sn_gauss=4,
                                show_profile=self._plotit,
                                use_2dmodel_mask=True,
                                no_local_sky=True)
                            spec_boxcar_flx = trcs[ee].BOX_COUNTS.flatten()
                            spec_boxcar_sig = trcs[ee].BOX_COUNTS_SIG.flatten()
                            spec_boxcar_wav = trcs[ee].BOX_WAVE.flatten()
                            spec_optimal_flx = trcs[ee].OPT_COUNTS.flatten()
                            spec_optimal_sig = trcs[ee].OPT_COUNTS_SIG.flatten()
                            spec_optimal_wav = trcs[ee].OPT_WAVE.flatten()
                            extract_boxcar(objmodel, ivar_use, ingpm, waveimg, np.zeros_like(profile_img), trcs[ee])
                            boxwght = spec.BOX_COUNTS.copy().flatten()
                    if not self._step_skycoeffs and not self._step_basis:
                        skytxt = ""
                        if self._ext_sky: skytxt = "_sky"
                        outPath = self._altpath
                        if self._use_diff: outPath = self._procpath
                        outA = outPath + "spec1d_{0:02d}_{1:s}{2:s}.dat".format(ff, self._nods[0], skytxt)
                        outB = outPath + "spec1d_{0:02d}_{1:s}{2:s}.dat".format(ff, self._nods[1], skytxt)
                        np.savetxt(outA, np.transpose((trc_pos['BOX_WAVE'][0, :], trc_pos['BOX_COUNTS'][0, :],
                                                       trc_pos['BOX_COUNTS_SIG'][0, :], trc_pos['OPT_WAVE'][0, :],
                                                       trc_pos['OPT_COUNTS'][0, :], trc_pos['OPT_COUNTS_SIG'][0, :])))
                        np.savetxt(outB, np.transpose((trc_neg['BOX_WAVE'][0, :], trc_neg['BOX_COUNTS'][0, :],
                                                       trc_neg['BOX_COUNTS_SIG'][0, :], trc_neg['OPT_WAVE'][0, :],
                                                       trc_neg['OPT_COUNTS'][0, :], trc_neg['OPT_COUNTS_SIG'][0, :])))
                        if self._plotit or True:
                            plt.subplot(211)
                            plt.plot(trc_pos['BOX_WAVE'][0, :], trc_pos['BOX_COUNTS'][0, :], 'k-', drawstyle='steps-mid')
                            plt.plot(trc_pos['BOX_WAVE'][0, :], trc_pos['BOX_COUNTS_SKY'][0, :], 'b-',
                                     drawstyle='steps-mid')
                            plt.subplot(212)
                            plt.plot(trc_pos['OPT_WAVE'][0, :], trc_pos['OPT_COUNTS'][0, :], 'r-', drawstyle='steps-mid')
                            plt.plot(trc_pos['OPT_WAVE'][0, :], trc_pos['OPT_COUNTS_SKY'][0, :], 'g-',
                                     drawstyle='steps-mid')
            # Now collect all of the extractions of this one frame to make a master file, and save it.
            print("TOTAL (inner loop) TIME = ", (time.time() - st_time) / 60.0, "mins")
            for tt in range(numtrc):
                out_specname = self._procpath + self._prefix+"_spec{0:02d}.dat".format(numtrc*ff+tt)
                self.comb_rebin_pixel([raw_specs[tt]], outfile=out_specname)
        print("TOTAL TIME = ", (time.time()-st_time)/60.0, "mins")

    def step_wavecal_prelim(self):
        usePath = self._procpath
        limpl, limpr = self.get_objprof_limits(full=False)
        rwf.wavecal_prelim(usePath, self._prefix, self._numframes, limpl[1]-20.0, limpr[0]+20.0, numcomp=self._numcomp, scale_errors=self._scale_errors)

    def step_prepALIS(self):
        out_wave, raw_specs = self.comb_prep(use_corrected=False)
        npix, nspec = out_wave.size, len(raw_specs)
        out_flux = self._maskval * np.ones((npix, nspec))
        out_flue = self._maskval * np.ones((npix, nspec))
        # Reject
        raw_wav, raw_flx, raw_err, bpm = self.comb_reject(out_wave, raw_specs, use_corrected=False)
        lminwv, lmaxwv = 10826.0, 10840.0
        fminwv, fmaxwv = 10827.0, 10839.0
        datlines, zerolines, strall = "", "", ""
        #usePath = self._altpath + "alt_"
        #if self._use_diff: usePath = self._procpath
        usePath = self._procpath
        snr = np.zeros(nspec)
        for sp in range(nspec):
            wave, flux, flue, fitr = raw_wav[sp, :], raw_flx[sp, :], raw_err[sp, :], 1 - bpm[sp, :]
            snr[sp] = np.median(flux*utils.inverse(flue))
            ww = np.where((wave > lminwv) & (wave < lmaxwv))
            wf = np.where((wave < fminwv) | (wave > fmaxwv))
            fitr[wf] = 0
            outname = usePath + self._prefix+"_ALIS_spec{0:02d}.dat".format(sp)
            np.savetxt(outname, np.transpose((wave[ww], flux[ww], flue[ww], fitr[ww])))
            print("File written: {0:s}".format(outname))
            datlines += "  {0:s}   specid=He{1:02d}    fitrange=columns   resolution=vfwhm(3.657crires)  shift=vshiftscale(0.0,1.0)  columns=[wave,flux,error,fitrange,continuum]  plotone=True   label=HeI_10830_{1:02d}\n".format(
                outname, sp)
            zerolines += "  constant 0.0 specid=He{0:02d}\n".format(sp)
            strall += "He{0:02d},".format(sp)
        # print(
        #     "\n\n\nHere is some informtion to run with ALIS to fix the wavelength scale. This must be done before you can proceeed to the next step:\n\n")
        self.make_fitallexp_ALIS(datlines, zerolines, strall[:-1], snr)
        # print(datlines)
        # print(zerolines)
        # print(strall)

    def make_fitallexp_ALIS(self, datlines, zerolines, strall, snr):
        # Create the ALIS input file for the wavelength calibration
        alis_file = self._procpath + self._prefix + "_fitallexp.mod"
        with open(alis_file, 'w') as f:
            f.write("# ALIS input file for wavelength calibration\n")
            f.write("#\n")
            f.write("run blind False\n")
            f.write("run ncpus 8\n")
            f.write("out fits True\n")
            f.write("#plot only True\n")
            f.write("plot dims 3x3\n")
            f.write("plot fits True\n")
            f.write("plot labels True\n")
            f.write("plot ticklabels True\n")
            f.write("plot ticks True\n")
            f.write("plot fitregions True\n")
            f.write("out wavecorr True\n")
            f.write("\n")
            f.write("# Data files:\n")
            f.write("data read\n")
            for line in datlines.splitlines():
                f.write(line+"\n")
            f.write("  HeI3188.dat         specid=1   fitrange=[3186.9,3189.9]   resolution=vfwhm(5.74uves)   columns=[wave,flux,error,continuum]  plotone=True   label=HeI_3188\n")
            f.write("  HeI3889.dat         specid=2   fitrange=[3887.9,3890.9]   resolution=vfwhm(5.74uves)  shift=vshift(0.0)   columns=[wave,flux,error,continuum]  plotone=True   label=HeI_3889\n")
            f.write("data end\n\n")
            f.write("model read\n")
            f.write(" lim voigt bturb [0.01,None]\n")
            f.write(" lim constant value [None,None]\n")
            f.write(" fix vfwhm value True\n")
            f.write(" emission\n")
            f.write("  legendre 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0  specid={0:s}\n".format(strall))
            f.write("  legendre 1.0 0.0 0.0 0.0 0.0 0.0 0.0  specid=1\n")
            f.write("  legendre 1.0 0.0 0.0 0.0 0.0 0.0 0.0  specid=2\n")
            f.write(" absorption\n")
            f.write("# Continuum scaling\n")
            refidx = np.argmax(snr)
            print("Reference spectrum = {0:d} (SNR={1:.2f})".format(refidx, snr[refidx]))
            print("The other spectra have S/N: {0:s}".format(", ".join(["{0:.2f}".format(s) for s in snr])))
            f.write("###  Continuum scaling should have one missing to avoid degeneracy!!!\n")
            for ll in range(len(strall.split(","))):
                if ll != refidx:
                    f.write("  legendre 1.0 0.0   specid=He{0:02d}  continuum=True\n".format(ll))
            f.write("# 4He I absorption\n")
            f.write("  voigt   ion=4He_I    13.5      -8.140716678115E-05ra  7.0da      0.0000000TEMP specid=1,2,{0:s}\n".format(strall))
            f.write("# Blends for sky absorption lines\n")
            f.write("#  voigt   ion=1Ly_a    10.4223493      7.912444979322      0.010000TELLURIC      0.0000000TEMP      damping=11.2439061damp          specid={0:s}\n".format(strall))
            f.write("#  voigt   ion=1Ly_a    12.0044660      7.912843032687      0.010000TELLURIC      0.0000000TEMP      damping=11.2439061damp          specid={0:s}\n".format(strall))
            f.write("#  voigt   ion=1Ly_a    11.5198892      7.914386416610      0.010000TELLURIC      0.0000000TEMP      damping=11.2439061damp          specid={0:s}\n".format(strall))
            f.write(" zerolevel\n")
            for line in zerolines.splitlines():
                f.write(line+"\n")
            f.write("model end\n\n")
        print("\n\n\nAn ALIS file has been prepared for you to run with ALIS to determine the wavelength scale. This must be done before you can proceeed to the next step:\n\n")
        print(alis_file)

    def step_wavecal_sky(self):
        # Start by loading and processing all of the target data
        out_wave, raw_specs = self.comb_prep(use_corrected=False)
        npix, nspec = out_wave.size, len(raw_specs)
        out_flux = self._maskval * np.ones((npix, nspec))
        out_flue = self._maskval * np.ones((npix, nspec))
        # Reject
        raw_wav, raw_flx, raw_err, bpm = self.comb_reject(out_wave, raw_specs, use_corrected=False)
        lminwv, lmaxwv = 10826.0, 10840.0
        fminwv, fmaxwv = 10827.0, 10839.0
        datlines, zerolines, strall = "", "", ""
        raw_specs = []
        for sp in range(nspec):
            wave, flux, flue, fitr = raw_wav[sp, :], raw_flx[sp, :], raw_err[sp, :], 1 - bpm[sp, :]
            ww = np.where((wave > lminwv) & (wave < lmaxwv))
            wf = np.where((wave < fminwv) | (wave > fmaxwv))
            skyname = self._procpath + "spec1d_{0:02d}.dat".format(sp)
            errspec, sky_counts = np.loadtxt(skyname, unpack=True, usecols=(2,3))  # errspec is a hack here... really should use the error of sky counts
            errspec[errspec==0.0] = np.median(errspec)
            # Load the old and corrected wavelength scale
            tmpnameAz = self._procpath + self._prefix+"_ALIS_spec{0:02d}_wzcorr.dat".format(sp)
            out_waveAz, inwaveAz, flux = np.loadtxt(tmpnameAz, unpack=True, usecols=(0, 1, 2))
            wA = np.where(np.in1d(wave, inwaveAz))
            np.savetxt(skyname.replace(".dat", "_sky_wzcorr.dat"),
                       np.transpose((out_waveAz, sky_counts[wA])))
            exptime, etim = self.get_exptime(sp//2)
            plt.plot(out_waveAz, sky_counts[wA]/exptime, 'k-', drawstyle='steps-mid')
            raw_specs.append(XSpectrum1D.from_tuple((out_waveAz, sky_counts[wA], errspec[wA]), verbose=False))
        get_wavename = self._procpath + self._prefix + "_HeI10833_scaleErr_wzcorr_comb_rebin.dat"
        out_wave = np.loadtxt(get_wavename, unpack=True, usecols=(0,))
        wav, flx, err, err_orig, final_flux = self.comb_rebin(out_wave, raw_specs, sky=True)
        print("in wavecal sky")
        embed()
        exptime, etim = self.get_exptime(0.0)
        plt.plot(wav, flx/exptime, 'r-', drawstyle='steps-mid')
        plt.show()

        #plt.plot(wav, err_orig, 'k-', drawstyle='steps-mid')
        plt.plot(wav, err_orig/flx, 'r-', drawstyle='steps-mid')
        plt.show()

    def step_combspec(self):
        out_wave, raw_specs = self.comb_prep(use_corrected=True)
        if self._step_combspec_rebin:
            self.comb_rebin(out_wave, raw_specs)
        else:
            print("ERROR :: This is not yet implemented/working very well...")
            embed()
            assert(False)
            wave_bins = out_wave.copy()
            npix, nspec = out_wave.size, len(raw_specs)
            out_flux = self._maskval * np.ones((npix, nspec))
            out_flue = self._maskval * np.ones((npix, nspec))
            # Reject
            raw_wav, raw_flx, raw_err, bpm = self.comb_reject(out_wave, raw_specs, use_corrected=True)
            # Find all good pixels and create the final histogram
            for ss in range(bpm.shape[0]):
                spec_use = np.ones(bpm.shape, dtype=bool)
                for mm in range(bpm.shape[0] - ss, bpm.shape[0]):
                    spec_use[mm, :] = False
                out_wave, spec, specerr = self.comb_spectrum(wave_bins, raw_wav, raw_flx, raw_err, bpm, spec_use)
                fitr = np.zeros(out_wave.size)
                fitr[np.where(((out_wave > 10827.0) & (out_wave < 10832.64)) | ((out_wave > 10833.16) & (out_wave < 10839)))] = 1
                np.savetxt(self._procpath + "tet02_OriA_HeI10833_scaleErr_wzcorr_fitr_comb{0:02d}.dat".format(bpm.shape[0] - ss), np.transpose((out_wave, spec, specerr, fitr)))
            # Save the final spectrum
            print("Saving output spectrum...")
            if False:
                fitr = np.zeros(out_wave.size)
                fitr[np.where(
                    ((out_wave > 10827.0) & (out_wave < 10832.64)) | ((out_wave > 10833.16) & (out_wave < 10839)))] = 1
                np.savetxt(self._procpath + "tet02_OriA_HeI10833_scaleErr_wzcorr_fitr.dat",
                           np.transpose((out_wave, spec, specerr_new, fitr)))
            else:
                print("ERROR... specerr_new does not exist")
                embed()
                # np.savetxt(self._procpath + "tet02_OriA_HeI10833_scaleErr_wzcorr_fitr.dat",
                #            np.transpose((out_wave, spec, specerr_new, fitr)))
            if plotit or True:
                for sp in range(nspec):
                    plt.plot(raw_wav[sp, :], raw_flx[sp, :], 'k-', drawstyle='steps-mid')
                    ww = np.where(bpm[sp, :])[0]
                    plt.plot(raw_wav[sp, ww], raw_flx[sp, ww], 'rx')
                plt.plot(out_wave, spec, 'g-', drawstyle='steps-mid')
                plt.plot(out_wave, specerr_new, 'r-', drawstyle='steps-mid')
                plt.show()

    def step_comb_sky(self):
        out_wave, raw_specs = self.comb_prep(use_corrected=True, sky=True)
        self.comb_rebin(out_wave, raw_specs, sky=True)

    def step_sample_NumExpCombine(self):
        nsample = 100
        print("sample_numexpcombine")
        embed()
        out_wave, raw_specs = self.comb_prep(use_corrected=True)
        nspec = len(raw_specs) - 4
        # Find all good pixels and create the final histogram
        snr_all, snr_all_adj = np.zeros(nspec), np.zeros(nspec)
        snr_all_err, snr_all_adj_err = np.zeros(nspec), np.zeros(nspec)
        for ss in range(nspec - 1):
            snr, snradj = np.zeros(nsample), np.zeros(nsample)
            for nn in range(nsample):
                ffs = np.arange(nspec)
                np.random.shuffle(ffs)
                raw_specs_samp = []
                for mm in range(ss, nspec):
                    raw_specs_samp.append(raw_specs[ffs[mm]])
                out_wave, spec, specerr, _, _ = self.comb_rebin(out_wave, raw_specs_samp, save=False)
                snr[nn], snradj[nn] = self.scale_variance(out_wave, spec, specerr, getSNR=True)
            ww = np.where(snr > 100)
            snr_all[nspec - ss - 1] = np.median(snr[ww])
            snr_all_err[nspec - ss - 1] = 1.4826 * np.median(np.abs(snr[ww] - np.median(snr[ww])))
            snr_all_adj[nspec - ss - 1] = np.mean(snradj)
            snr_all_adj_err[nspec - ss - 1] = np.std(snradj)
        # The case for 2 frames
        nsample = 10000
        snr, snradj = np.zeros(nsample), np.zeros(nsample)
        for nn in range(nsample):
            ffs = np.arange(nspec)
            np.random.shuffle(ffs)
            raw_specs_samp = []
            for mm in range(ss, nspec):
                raw_specs_samp.append(raw_specs[ffs[mm]])
            out_wave, spec, specerr, _, _ = self.comb_rebin(out_wave, raw_specs_samp, save=False)
            snr[nn], snradj[nn] = self.scale_variance(out_wave, spec, specerr, getSNR=True)
        ww = np.where(snr > 100)
        snr_all[1] = np.median(snr[ww])
        snr_all_err[1] = 1.4826 * np.median(np.abs(snr[ww] - np.median(snr[ww])))
        # The case for 1 frame
        snr, snradj = np.zeros(nspec), np.zeros(nspec)
        for ss in range(nspec):
            snr[ss], snradj[ss] = self.scale_variance(raw_specs[ss].wavelength.value, raw_specs[ss].flux.value,
                                                      raw_specs[ss].sig.value, getSNR=True)
        snr_all[0] = np.median(snr)
        snr_all_err[0] = 1.4826 * np.median(np.abs(snr - np.median(snr)))
        snr_all_adj[0] = np.mean(snradj)
        snr_all_adj_err[0] = np.std(snradj)
        np.savetxt("SNR_NumExpCombine.dat", np.transpose((snr_all, snr_all_err, snr_all_adj, snr_all_adj_err)))
        # Plot it up
        numexp = np.arange(nspec) + 1
        model = snr_all[-1] * np.sqrt(numexp) / np.sqrt(nspec)
        plt.plot(numexp, snr_all, 'b-')
        plt.plot(numexp, snr_all + snr_all_err, 'b--')
        plt.plot(numexp, snr_all - snr_all_err, 'b--')
        plt.plot(numexp, model, 'r-')
        # plt.plot(numexp, snr_all_adj+snr_all_adj_err, 'b-')
        # plt.plot(numexp, snr_all_adj-snr_all_adj_err, 'b-')
        plt.show()
        # wave_bins = out_wave.copy()
        # npix, nspec = out_wave.size, len(raw_specs)
        # out_flux = self._maskval*np.ones((npix, nspec))
        # out_flue = self._maskval*np.ones((npix, nspec))
        # # Reject
        # raw_wav, raw_flx, raw_err, bpm = comb_reject(out_wave, raw_specs, use_corrected=True)
        # nspec = bpm.shape[0]
        # # Find all good pixels and create the final histogram
        # snr_all, snr_all_adj = np.zeros(nspec), np.zeros(nspec)
        # snr_all_err, snr_all_adj_err = np.zeros(nspec), np.zeros(nspec)
        # for ss in range(nspec):
        #     print("UP TO HERE!", ss+1, nspec)
        #     snr, snradj = np.zeros(nsample), np.zeros(nsample)
        #     for nn in range(nsample):
        #         ffs = np.arange(nspec)
        #         np.random.shuffle(ffs)
        #         spec_use = np.ones(bpm.shape, dtype=bool)
        #         for mm in range(nspec-ss, nspec):
        #             spec_use[ffs[mm], :] = False
        #         out_wave, spec, specerr, specerr_new = comb_spectrum(wave_bins, raw_wav, raw_flx, raw_err, bpm, spec_use, get_specerr_orig=True)
        #         snr[nn], snradj[nn] = scale_variance(out_wave, spec, specerr, getSNR=True)
        #     snr_all[nspec-ss-1] = np.mean(snr)
        #     snr_all_err[nspec-ss-1] = np.std(snr)
        #     snr_all_adj[nspec-ss-1] = np.mean(snradj)
        #     snr_all_adj_err[nspec-ss-1] = np.std(snradj)
        # np.savetxt("SNR_NumExpCombine.dat", np.transpose((snr_all, snr_all_err, snr_all_adj, snr_all_adj_err)))

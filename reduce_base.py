import os
import numpy as np
import astropy.io.fits as fits
import astropy.units as units
from matplotlib import pyplot as plt
from pypeit import utils
from pypeit.core import procimg, skysub, findobj_skymask
from pypeit.core.extract import fit_profile, extract_optimal

from linetools.spectra.xspectrum1d import XSpectrum1D
from scipy import signal
from scipy.signal import medfilt2d, correlate2d
from scipy import ndimage, interpolate
from IPython import embed
import reduce_wavecal_fitting as rwf
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


class ReduceBase:
    def __init__(self, prefix="targetname", use_diff=False,
                 step_listfiles=False,
                 step_pattern=False,  # Generate an image of the detector pattern
                 step_makedarkfit=False, step_makedarkframe=False,  # Make a dark image
                 step_makeflat=False,  # Make a flatfield image
                 step_makearc=False,  # Make an arc image
                 step_makediff=False,  # Make difference and sum images
                 step_makecuts=False,  # Make difference and sum images
                 step_trace=False, step_extract=False, step_skycoeffs=False, mean_skycoeff=False, step_basis=False,
                 ext_sky=False,  # Trace the spectrum and extract
                 step_wavecal_prelim=True,  # Calculate a preliminary wavelength calibration solution
                 step_prepALIS=False,
                 # Once the data are reduced, prepare a series of files to be used to fit the wavelength solution with ALIS
                 step_combspec=False, step_combspec_rebin=False,
                 # First get the corrected data from ALIS, and then combine all exposures with this step.
                 step_wavecal_sky=False, step_comb_sky=False,  # Wavelength calibrate all sky spectra and then combine
                 step_sample_NumExpCombine=False):

        self._prefix = prefix
        self._plotit = False
        self._specaxis = 0
        self._gain = 2.15  # This value comes from the header
        self._chip = 1  # self._chip can be 1, 2, or 3
        self._slice = np.meshgrid(np.arange(310, 600), np.arange(2048), indexing='ij')
        self._polyord = 10  # Polynomial order used to trace the spectra
        self._nods = ['A', 'B']
        self._velstep = 1.5  # Sample the FWHM by ~2.5 pixels
        self._maskval = -99999999  # Masked value for combining data
        self._sigcut = 3.0  # Rejection level when combining data

        self.makePaths()

        # Set the reduction flags
        self._use_diff = use_diff
        self._step_listfiles = step_listfiles
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
        self._ext_sky = ext_sky
        self._step_wavecal_prelim = step_wavecal_prelim
        self._step_prepALIS = step_prepALIS
        self._step_combspec = step_combspec
        self._step_combspec_rebin = step_combspec_rebin
        self._step_wavecal_sky = step_wavecal_sky
        self._step_comb_sky = step_comb_sky
        self._step_sample_NumExpCombine = step_sample_NumExpCombine

        self._matches = self.get_science_frames()

        self._numframes = len(self._matches)

        self._flat_files = self.get_flat_frames()
        self._dark_files = self.get_dark_frames()
        self._arc_files = self.get_arc_frames()

    def makePaths(self, redux_path="",
                  data_folder="Raw/"):
        self._redux_path = redux_path
        self._data_folder = data_folder
        self._cals_folder = "redux_"+self._prefix+"/calibrations/"
        self._proc_folder = "redux_"+self._prefix+"/processed/"
        self._alt_folder = "redux_"+self._prefix+"/alternative/"
        self._datapath = self._redux_path + self._data_folder
        self._calspath = self._redux_path + self._cals_folder
        self._procpath = self._redux_path + self._proc_folder
        self._altpath = self._redux_path + self._alt_folder

        # Check if paths exist, if not, make them
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
        self._cut_name = self._procpath + "cuts_FR{0:02d}_" + self._chip_str

    def get_science_frames(self):
        return None

    def get_dark_frames(self):
        return None

    def get_flat_frames(self):
        return None

    def get_arc_frames(self):
        return None

    def run(self):
        if self._step_listfiles: self.step_listfiles()
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
        raw_specs = []
        minwv = 9999999999999
        maxwv = -minwv
        if use_corrected:
            usePath = self._altpath + "alt_"
            if self._use_diff: usePath = self._procpath
            for ff in range(self._numframes * len(self._nods)):
                #        for ff in range(self._numframes):
                if sky:
                    outname = usePath + "spec1d_{0:02d}_{1:s}_sky_wzcorr.dat".format(ff // 2, self._nods[ff % 2])
                    opt_wave, opt_cnts, opt_cerr = np.loadtxt(outname, usecols=(0, 1, 2), unpack=True)
                else:
                    outname = usePath + self.prefix+"_ALIS_spec{0:02d}_wzcorr.dat".format(ff)
                    opt_wave, opt_cnts, opt_cerr = np.loadtxt(outname, usecols=(0, 2, 3), unpack=True)
                raw_specs.append(XSpectrum1D.from_tuple((opt_wave, opt_cnts, opt_cerr), verbose=False))
                if np.min(opt_wave) < minwv:
                    minwv = np.min(opt_wave)
                if np.max(opt_wave) > maxwv:
                    maxwv = np.max(opt_wave)
        else:
            usePath = self._altpath
            if self._use_diff: usePath = self._procpath
            for ff in range(self._numframes):
                for nod in self._nods:
                    outname = usePath + "spec1d_wave_{0:02d}_{1:s}.dat".format(ff, nod)
                    box_wave, box_cnts, box_cerr, opt_wave, opt_cnts, opt_cerr = np.loadtxt(outname, unpack=True)
                    raw_specs.append(XSpectrum1D.from_tuple((opt_wave, opt_cnts, opt_cerr), verbose=False))
                    if np.min(opt_wave) < minwv:
                        minwv = np.min(opt_wave)
                    if np.max(opt_wave) > maxwv:
                        maxwv = np.max(opt_wave)
        # Generate the final wavelength array
        npix = np.log10(maxwv / minwv) / np.log10(1.0 + self._velstep / 299792.458)
        npix = np.int(npix)
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
        bpm = np.ones((nspec, maxnumpix), dtype=np.bool)
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
                raw_flx[sp, :raw_specs[sp].wavelength.size] = raw_specs[sp].flux / np.median(raw_specs[sp].flux)
                raw_err[sp, :raw_specs[sp].wavelength.size] = raw_specs[sp].sig / np.median(raw_specs[sp].flux)
        # Mask bad pixels
        for pp in range(wave_bins.size - 1):
            pixf = np.array([])
            pixe = np.array([])
            midx, midy = np.array([], dtype=np.int), np.array([], dtype=np.int)
            for sp in range(nspec):
                ww = np.where((raw_wav[sp, :] >= wave_bins[pp]) & (raw_wav[sp, :] < wave_bins[pp + 1]))
                pixf = np.append(pixf, raw_flx[sp, ww[0]])
                pixe = np.append(pixe, raw_err[sp, ww[0]])
                midx = np.append(midx, sp * np.ones(ww[0].size, dtype=np.int))
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
        usePath = self._altpath
        if self._use_diff: usePath = self._procpath
        npix, nspec = out_wave.size, len(raw_specs)
        new_specs = []
        out_flux = self._maskval * np.ones((npix, nspec))
        out_flue = self._maskval * np.ones((npix, nspec))
        for sp in range(nspec):
            new_specs.append(raw_specs[sp].rebin(out_wave * units.AA, do_sig=True))
            gpm = new_specs[0].sig != 0.0
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
        if self._plotit:
            for sp in range(nspec):
                plt.plot(out_wave, out_flux[:, sp], 'k-', drawstyle='steps-mid')
                ww = new_mask[:, sp]
                plt.plot(out_wave[ww], out_flux[ww, sp], 'bx', drawstyle='steps-mid')
            plt.plot(out_wave, spec, 'r-', drawstyle='steps-mid')
            plt.plot(out_wave, specerr_new, 'g', drawstyle='steps-mid')
            plt.show()
        # Save the final spectrum
        print("Saving output spectrum...")
        if save:
            if False:
                fitr = np.zeros(out_wave.size)
                if sky:
                    out_specname = usePath + "tet02_OriA_HeI10833_scaleErr_fitr_wzcorr_comb_rebin_sky.dat"
                else:
                    out_specname = usePath + "tet02_OriA_HeI10833_scaleErr_fitr_wzcorr_comb_rebin.dat"
                    fitr[np.where(
                        ((out_wave > 10827.0) & (out_wave < 10832.64)) | (
                                    (out_wave > 10833.16) & (out_wave < 10839)))] = 1
                np.savetxt(out_specname, np.transpose((out_wave, spec, specerr_new, fitr)))
            else:
                if sky:
                    out_specname = usePath + "tet02_OriA_HeI10833_scaleErr_wzcorr_comb_rebin_sky.dat"
                else:
                    out_specname = usePath + "tet02_OriA_HeI10833_scaleErr_wzcorr_comb_rebin.dat"
                np.savetxt(out_specname, np.transpose((out_wave, spec, specerr_new)))
            print("File written: {0:s}".format(out_specname))
        return out_wave, spec, specerr_new

    def scale_variance(self, out_wave, spec, specerr, getSNR=False):
        wc = np.where((out_wave >= 10827) & (out_wave <= 10830))
        mcf = np.polyfit(out_wave[wc], spec[wc], 2)
        modcont = np.polyval(mcf, out_wave[wc])
        sig_meas = np.std(spec[wc] - modcont)
        sig_calc = np.mean(specerr[wc])
        scalefact = sig_meas / sig_calc
        specerr_new = scalefact * specerr
        print("Noise is underestimated by a factor of {0:f}".format(np.median(specerr_new / specerr)))
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

    def trace_tilt(self, objtrc, trcnum=50, plotit=False):
        """
        trcnum = the number of spatial pixels to trace either side of the object trace (objtrc)
        """
        msarc = fits.open(self._masterarc_name)[0].data.T
        medfilt = medfilt2d(msarc, kernel_size=(1, 7))
        # Find the peak near the trace
        # 1679, 145
        nfit = 0
        for ff in range(-nfit, nfit + 1):
            #        idx = np.arange(1679-17+35*ff,1679+17+35*ff)
            idx = np.arange(1644 - 17 + 35 * ff, 1644 + 17 + 35 * ff)
            amax = idx[np.argmax(medfilt[(idx, np.round(objtrc).astype(int)[idx])])]
            xpos = int(np.round(objtrc[amax]))
            allcen = np.zeros(1 + 2 * trcnum)
            # First trace one way
            this_amax = amax
            for ss in range(0, trcnum + 1):
                idx = np.arange(this_amax - 5, this_amax + 5)
                thisspec = medfilt[(idx, xpos + ss)]
                coeff = np.polyfit(idx, thisspec, 2)
                newmax = -0.5 * coeff[1] / coeff[0]
                allcen[trcnum + ss] = newmax
                this_amax = int(np.round(newmax))
            # Now trace the other way
            this_amax = amax
            for ss in range(0, trcnum):
                idx = np.arange(this_amax - 5, this_amax + 5)
                thisspec = medfilt[(idx, xpos - ss - 1)]
                coeff = np.polyfit(idx, thisspec, 2)
                newmax = -0.5 * coeff[1] / coeff[0]
                allcen[trcnum - ss - 1] = newmax
                this_amax = int(np.round(newmax))
            # Now perform a fit to the tilt
            xdat = np.arange(xpos - trcnum, xpos + trcnum + 1)
            coeff = np.polyfit(xdat, allcen, 2)
            model = np.polyval(coeff, xdat)
            modcen = np.polyval(coeff, objtrc[amax])
            coeff = np.polyfit(xdat, modcen - allcen, 2)
            if plotit:
                #             plt.plot(xdat, allcen, 'b')
                #             plt.plot(xdat, model, 'r-')
                plt.plot(xdat, allcen - model)
        plt.show()
        # Generate a tilt image
        spatimg = np.arange(msarc.shape[1])[None, :].repeat(msarc.shape[0], axis=0)
        specimg = np.arange(msarc.shape[0])[:, None].repeat(msarc.shape[1], axis=1)
        tiltimg = specimg + np.polyval(coeff, spatimg)
        return tiltimg

    def step_listfiles(self):
        files = open("files.list").readlines()
        for ff in range(len(files)):
            fil = fits.open(self._datapath + files[ff].strip("\n"))
            try:
                print(files[ff].strip("\n"), fil[0].header['HIERARCH ESO DET NDIT'],
                      fil[0].header['HIERARCH ESO SEQ NODPOS'], fil[0].header['HIERARCH ESO SEQ NODTHROW'],
                      fil[0].header['EXPTIME'], fil[0].header['OBJECT'],
                      fil[1].header['HIERARCH ESO DET self._chip self._gain'])
            except:
                print(files[ff].strip("\n"), fil[0].header['OBJECT'], fil[0].header['EXPTIME'])
                continue

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
                      fil[0].header['OBJECT'], fil[1].header['HIERARCH ESO DET self._chip self._gain'])
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
        rawdata = np.zeros((self._slice[0].shape + (len(self._dark_files), len(self._dark_files[0]),)))
        exptime = np.zeros(len(self._dark_files))
        for gg in range(len(self._dark_files)):
            for ff in range(len(self._dark_files[gg])):
                fil = fits.open(self._datapath + self._dark_files[gg][ff].strip("\n"))
                print(self._dark_files[gg][ff].strip("\n"), fil[0].header['HIERARCH ESO DET NDIT'],
                      fil[0].header['EXPTIME'],
                      fil[0].header['OBJECT'], fil[1].header['HIERARCH ESO DET self._chip self._gain'])
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
        rawdata = np.zeros((self._slice[0].shape + (len(self._flat_files),)))
        for ff in range(len(self._flat_files)):
            fil = fits.open(self._datapath + self._flat_files[ff].strip("\n"))
            print(self._flat_files[ff].strip("\n"), fil[0].header['HIERARCH ESO DET NDIT'], fil[0].header['EXPTIME'],
                  fil[0].header['OBJECT'], fil[1].header['HIERARCH ESO DET self._chip self._gain'])
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
        rawdata = np.zeros((self._slice[0].shape + (len(self._arc_files),)))
        msflat = fits.open(self._masterflat_name)[0].data
        for ff in range(len(self._arc_files)):
            fil = fits.open(self._datapath + self._arc_files[ff].strip("\n"))
            print(self._arc_files[ff].strip("\n"), fil[0].header['HIERARCH ESO DET NDIT'], fil[0].header['EXPTIME'],
                  fil[0].header['OBJECT'], fil[1].header['HIERARCH ESO DET self._chip self._gain'])
            msdark = fits.open(self.get_darkname(self._masterdark_name, int(fil[0].header['EXPTIME'])))[0].data
            rawdata[:, :, ff] = fil[1].data[self._slice] - msdark
        # Sigma clip
        bpm = np.zeros(rawdata.shape, dtype=np.bool)
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
            fil_a = fits.open(self._datapath + self._matches[mm][0])
            fil_b = fits.open(self._datapath + self._matches[mm][1])
            # Double check which is img_a and which is img_b
            if fil_a[0].header['HIERARCH ESO SEQ NODPOS'].strip() == 'A':
                print("Found A", mm)
                img_a = fil_a[self._chip].data
                img_b = fil_b[self._chip].data
            else:
                print("Switch", fil_a[0].header['HIERARCH ESO SEQ NODPOS'].strip(), mm)
                img_b = fil_a[self._chip].data
                img_a = fil_b[self._chip].data
            ndit = 20
            if mm == 12: ndit = 9
            # Take the difference
            diff = (img_a - img_b) * ndit
            sumd = (img_a + img_b) * ndit
            # Save the output
            outname = self._diff_name.format(mm)
            hdu = fits.PrimaryHDU(diff[self._slice] / msflat)
            hdu.writeto(outname, overwrite=True)
            print("File written: {0:s}".format(outname))
            # Summed image
            outname = self._sumd_name.format(mm)
            hdu = fits.PrimaryHDU(sumd[self._slice] / msflat)
            hdu.writeto(outname, overwrite=True)
            print("File written: {0:s}".format(outname))

    def step_makecuts(self):
        # Load the flat frame
        # msflat = fits.open(self._masterflat_name)[0].data
        # Make cut outs of the order of interest
        for mm in range(self._numframes):
            # Load the files
            fil_a = fits.open(self._datapath + self._matches[mm][0])
            fil_b = fits.open(self._datapath + self._matches[mm][1])
            assert (fil_a[0].header['EXPTIME'] == fil_b[0].header[
                'EXPTIME'])  # Otherwise, would need to generate two different dark frames below
            # Generate the dark frame
            msdark = fits.open(self.get_darkname(self._masterdark_name, int(fil_a[0].header['EXPTIME'])))[0].data
            # Double check which is img_a and which is img_b
            if fil_a[0].header['HIERARCH ESO SEQ NODPOS'].strip() == 'A':
                print("Found A", mm)
                img_a = fil_a[self._chip].data[self._slice] - msdark
                img_b = fil_b[self._chip].data[self._slice] - msdark
            else:
                print("Switch", fil_a[0].header['HIERARCH ESO SEQ NODPOS'].strip(), mm)
                img_b = fil_a[self._chip].data[self._slice] - msdark
                img_a = fil_b[self._chip].data[self._slice] - msdark
            ndit = 20
            if mm == 12: ndit = 9
            # Take the difference
            cutA = img_a * ndit
            cutB = img_b * ndit
            # Save the output
            outname = self._cut_name.format(2 * mm)
            hdu = fits.PrimaryHDU(cutA)  # /msflat)
            hdu.writeto(outname, overwrite=True)
            print("File written: {0:s}".format(outname))
            # Summed image
            outname = self._cut_name.format(2 * mm + 1)
            hdu = fits.PrimaryHDU(cutB)  # /msflat)
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
            embed()
        try:
            Vbase = np.linalg.inv(np.dot(lhs, lhs.T))
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
            print("WARNING :: The fit may be poorly conditioned")

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

    def object_profile(self, allflux, allspat, allgpmtmp):
        allgpm = allgpmtmp.copy()
        sigrej = 3.0
        binsize = 0.1
        bins = np.arange(-binsize / 2 - 50.0, 50.0 + binsize, binsize)
        inds = np.digitize(allspat, bins)
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
        gpm = np.where(allgpm)
        # First estimate of object profile
        cnts, _ = np.histogram(allspat[gpm], bins=bins, weights=allflux[gpm])
        norm, _ = np.histogram(allspat[gpm], bins=bins)
        cnts *= utils.inverse(norm)
        # Calculate the step width (used for the second basis vector)
        xloc = 0.5 * (bins[1:] + bins[:-1])
        nrmcnts = np.sum(cnts)
        cnts /= nrmcnts
        #     opspl = interpolate.CubicSpline(xloc, cnts)
        return xloc, cnts

    def basis_fit(self, extfrm_use, ivar_use, tilts, waveimg, spatimg, spec, idx, skycoeffs=False):
        msflat = fits.open(self._masterflat_name)[0].data
        onslit = msflat > 0.1
        onslit[:, :31] = False
        onslit[:, 269:] = False
        sigrej = 3
        nbasis = 3  # 25
        binsize = 0.1
        bins = np.arange(-binsize / 2 - 50.0, 50.0 + binsize, binsize)
        nspec, nspat = extfrm_use.shape
        # Trace the spectral tilt
        if skycoeffs:
            allspecimg = np.arange(extfrm_use.shape[0])[:, None].repeat(extfrm_use.shape[1], axis=1)
        else:
            allspecimg = self.trace_tilt(spec.TRACE_SPAT.flatten(), trcnum=50, plotit=False)
        allspec = allspecimg.flatten()
        allspat = (spatimg - spec.TRACE_SPAT.T).flatten()
        allflux = extfrm_use.flatten()
        allivar = ivar_use.flatten()
        inds = np.digitize(allspat, bins)
        gpm_img = onslit.copy()
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
        if False:
            # test if the tilts worked
            gpm = np.where((gpm_img.flatten()) & (allspat > -50) & (allspat < 50))
            cnts, _ = np.histogram(allspec[gpm], bins=np.arange(2048), weights=extfrm_use_med.flatten()[gpm])
            norm, _ = np.histogram(allspec[gpm], bins=np.arange(2048))
            cnts *= utils.inverse(norm)
            plt.plot(cnts)
            plt.show()
        tmp = (allgpm) & ((allspec > 1800) & (allspec < 1950)) | ((allspec > 1400) & (allspec < 1550))
        gpm = np.where(tmp)
        # First estimate of object profile
        cnts, _ = np.histogram(allspat[gpm], bins=bins, weights=extfrm_use_med.flatten()[gpm])
        norm, _ = np.histogram(allspat[gpm], bins=bins)
        cnts *= utils.inverse(norm)
        # Calculate the step width (used for the second basis vector)
        xloc = 0.5 * (bins[1:] + bins[:-1])
        #     cnts /= (2*np.sum(cnts))  # Factor of 2 is because we only consider half the profile here.
        nrmcnts = np.sum(cnts)
        cnts /= nrmcnts
        # Interpolate the object profile so that it can be used at each wavelength
        #     xspl = np.append(-xloc[1:][::-1], xloc)
        #     yspl = np.append(cnts[1:][::-1], cnts)
        opspl = interpolate.CubicSpline(xloc, 1.0E6 * cnts)
        #        plt.plot(xloc, cnts)
        #        plt.show()
        if skycoeffs:
            try:
                inname = self._procpath + "skycoeffs_splinefit_k3_lags.npy"
                alllags = np.load(inname)
                # np.column_stack((spl1[0],spl1[1],spl2[0],spl2[1]))
                spl1 = (allspl[:, 0], allspl[:, 1], 3)
                spl2 = (allspl[:, 2], allspl[:, 3], 3)
                outwave = np.arange(nspec) + alllags[idx]
            except:
                outwave = np.arange(nspec)
            # Now perform the fit
            outfluxb = np.zeros(extfrm_use.shape[0])
            outfluxb_err = np.zeros(extfrm_use.shape[0])
            outfluxbox = np.zeros(extfrm_use.shape[0])
            outfluxbox_err = np.zeros(extfrm_use.shape[0])
            HIIflux = np.zeros(extfrm_use.shape)
            model = np.zeros(extfrm_use.shape)
            modelstar = np.zeros(extfrm_use.shape)
            idealSN = np.zeros(extfrm_use.shape[0])
            coeffs = np.zeros((extfrm_use.shape[0], nbasis))
            for ss in range(extfrm_use.shape[0]):
                xdat = np.arange(extfrm_use.shape[1]) - spec.TRACE_SPAT[0, ss]
                xfit = (xdat + np.max(xloc)) / np.max(xloc) - 1
                yfit = extfrm_use[ss, :]
                #             wfit = np.ones(yfit.size)
                #            wfit = np.abs(extfrm_use[ss,:])**0.25
                wfit = ivar_use[ss, :]  # DONT CHANGE THIS WITHOUT CHANGING OUTFLUXBOX_ERR BELOW!!!
                gd = np.where((np.abs(xfit) <= 10.0 / np.max(xloc)) & (
                    gpm_img[ss, :]))  # Only include pixels that are defined within the spatial profile domain
                # Construct the vandermonde matrix
                vander = np.ones((xfit[gd].size, nbasis))
                vander[:, 0] = opspl(xdat[gd])
                #             for pp in range(xfit[gd].size):
                #                 vander[pp, 0] = opspl.integrate(xdat[gd][pp]-0.5, xdat[gd][pp]+0.5) # Object profile
                # vander[:, 1] = basis2[ss,:][gd] # Even step function (orthogonal to object profile)
                # vander[:, 1] =  np.sign(xfit[gd])*2*(kbst+1) * xfit[gd]**3 - 3*(kbst+1) * xfit[gd]**2 + kbst
                # vander[:, 2:] = np.polynomial.legendre.legvander(xfit[gd], (nbasis-2)*2-1)[:,1::2]  # The rest of the basis are the odd Legendre polynomials
                vander[:, 1:] = np.polynomial.legendre.legvander(xfit[gd], nbasis - 2)
                c, cov = basis_fitter(xfit[gd], yfit[gd], vander.copy(), w=wfit[gd], debug=False)  # ss==1000)
                coeffs[ss, :] = c.copy()
                ccont = c.copy()
                ccont[0] = 0
                gdc = np.where(
                    (np.abs(xfit) <= 1.0))  # Only include pixels that are defined within the spatial profile domain
                vanderc = np.ones((xfit[gdc].size, nbasis))
                # for pp in range(xfit[gd].size): vanderc[pp, 0] = opspl.integrate(xdat[gd][pp]-0.5, xdat[gd][pp]+0.5) # Object profile
                # vanderc[:, 1] =  np.sign(xfit[gdc])*2*(kbst+1) * xfit[gdc]**3 - 3*(kbst+1) * xfit[gdc]**2 + kbst
                # vanderc[:, 2:] = np.polynomial.legendre.legvander(xfit[gdc], (nbasis-2)*2-1)[:,1::2]  # The rest of the basis are the odd Legendre polynomials
                vanderc[:, 1:] = np.polynomial.legendre.legvander(xfit[gdc],
                                                                  nbasis - 2)  # The rest of the basis are the odd Legendre polynomials
                HIIflux[ss, gdc[0]] = np.dot(vanderc, ccont)
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
                outfluxbox[ss] = np.sum(yfit[gd] - HIIflux[ss, gd]) / np.sum(vander[:, 0])
                outfluxbox_err[ss] = np.sqrt(np.sum(utils.inverse(wfit[gd]))) / np.sum(vander[:, 0])
                idealSN[ss] = np.sum(yfit[gd]) / np.sqrt(np.sum(1 / wfit[gd]))
            outPath = self._altpath
            outA = outPath + "spec1d_{0:02d}.dat".format(idx)
            np.savetxt(outA, np.column_stack((outwave, outfluxbox, outfluxbox_err)))
            if False:
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
        else:
            inname = self._procpath + "skycoeffs_splinefit_k3.npy"
            allspl = np.load(inname)
            inname = self._procpath + "skycoeffs_splinefit_k3_lags.npy"
            alllags = np.load(inname)
            # np.column_stack((spl1[0],spl1[1],spl2[0],spl2[1]))
            spl1 = (allspl[:, 0], allspl[:, 1], 3)
            spl2 = (allspl[:, 2], allspl[:, 3], 3)

            outwave = np.arange(nspec) + alllags[idx]
            basis1 = interpolate.splev(outwave, spl1)
            basis2 = interpolate.splev(outwave, spl2)
            # Loop over all spectral elements
            HIIflux = np.zeros(extfrm_use.shape)
            outfluxbox = np.zeros(extfrm_use.shape[0])
            outfluxbox_err = np.zeros(extfrm_use.shape[0])
            idealSN = np.zeros(extfrm_use.shape[0])
            for ss in range(extfrm_use.shape[0]):
                xdat = np.arange(extfrm_use.shape[1]) - spec.TRACE_SPAT[0, ss]
                xfit = (xdat + np.max(xloc)) / np.max(xloc) - 1
                yfit = extfrm_use[ss, :]
                wfit = ivar_use[ss, :]
                gdmask = (np.abs(xfit) <= 10.0 / np.max(xloc)) & (gpm_img[ss, :])
                gd = np.where(gdmask)  # Only include pixels that are defined within the spatial profile domain
                # Construct the vandermonde matrix
                vander = np.ones((xfit[gd].size, nbasis))
                vander[:, 0] = opspl(xdat[gd])
                vander[:, 1:] = np.polynomial.legendre.legvander(xfit[gd], nbasis - 2)
                ccont = np.array([0.0, basis1[ss], basis2[ss]])
                gdc = np.where(np.abs(xfit) <= 1.0)
                vanderc = np.ones((xfit[gdc].size, nbasis))
                vanderc[:, 1:] = np.polynomial.legendre.legvander(xfit[gdc],
                                                                  nbasis - 2)  # The rest of the basis are the odd Legendre polynomials
                HIIflux[ss, gdc[0]] = np.dot(vanderc, ccont)
                HIIfluxMod = np.dot(vander, ccont)
                #                 fullprof = 0.0
                #                 for ii in range(xdat.size):
                #                     fullprof += (1-(gdmask | (np.abs(xdat)>49.5))[ii]) * opspl.integrate(xdat[ii]-0.5, xdat[ii]+0.5)
                outfluxbox[ss] = np.sum(yfit[gd] - HIIfluxMod) / np.sum(vander[:, 0])
                outfluxbox_err[ss] = np.sqrt(np.sum(utils.inverse(wfit[gd]))) / np.sum(vander[:, 0])
            print("(box) S/N = ", np.mean(outfluxbox[1338:1448]) / np.std(outfluxbox[1338:1448]))
            print("(box) S/N ab = ", np.mean(outfluxbox[1706:1726]) / np.std(outfluxbox[1706:1726]))
            plt.subplot(311);
            plt.plot(outfluxbox, 'b-', drawstyle='steps-mid');
            plt.subplot(312);
            plt.plot(outfluxbox / outfluxbox_err, drawstyle='steps-mid');
            plt.subplot(313);
            plt.plot(idealSN, drawstyle='steps-mid');
            plt.show()
            #             plt.subplot(131);plt.imshow(HIIflux, aspect=0.5, vmin=-1000, vmax=1000); plt.subplot(132); plt.imshow(extfrm_use*gpm_img, aspect=0.5, vmin=-1000, vmax=1000); plt.subplot(133); plt.imshow(gpm_img*(extfrm_use-HIIflux), aspect=0.5, vmin=-1000, vmax=1000); plt.show()
            outPath = self._procpath
            outA = outPath + "spec1d_{0:02d}.dat".format(idx)
            np.savetxt(outA, np.column_stack((outwave, outfluxbox, outfluxbox_err)))

        # Save the sky coefficients or the spectrum
        if skycoeffs:
            outname = self._procpath + "skycoeffs_{0:02d}.npy".format(idx)
            np.save(outname, coeffs)
            return True
        else:
            return False
        return False

    def step_trace(self):
        # Obtain a mask of the bad pixels
        fil = fits.open(self._datapath + self._matches[0][0])
        frm = fil[1].data[self._slice].T
        frm_filt = ndimage.median_filter(frm, size=(7, 1))
        bpm = np.abs(frm - frm_filt) > 50
        # Perform the object trace and extraction
        all_traces = []
        for ff in range(self._numframes):
            if self._use_diff:
                difnstr = self._diff_name.format(ff)
                print("extracting", difnstr)
                frame = fits.open(difnstr)[0].data.T
                frame *= self._gain
                framesum = fits.open(self._sumd_name.format(ff))[0].data.T
                framesum *= self._gain
                # Calculate the readnoise
                rnfrm = fits.open(difnstr)[0].data * self._gain
                statpix = np.append(rnfrm[:4, :].flatten(), rnfrm[-4:, :].flatten())
                ronoise = 1.4826 * np.median(np.abs(statpix - np.median(statpix)))
                print("RON  mean, std = ", np.mean(statpix), np.std(statpix))
                print("RON  median, 1.4826*MAD = ", np.median(statpix), ronoise)
            else:
                frm1 = self._cut_name.format(2 * ff)
                frm2 = self._cut_name.format(2 * ff + 1)
                print("Reducing... " + frm1)
                frame = fits.open(frm1)[0].data.T
                frame2 = fits.open(frm2)[0].data.T
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
            etim = 10
            exptime = etim * 20.0
            if ff == 12:
                etim = 7
                exptime = etim * 9
            msflat = fits.open(self._masterflat_name)[0].data.T
            msdark = fits.open(self.get_darkname(self._masterdark_name, etim))[0].data.T
            basevar = procimg.base_variance(rn2img, darkcurr=darkcurr, exptime=exptime)
            frame_for_ivar = ((frame / self._gain) + msdark) * self._gain
            rawvarframe = procimg.variance_model(basevar, frame_for_ivar)
            # Ivar
            ivar = utils.inverse(rawvarframe)
            waveimg = np.arange(frame.shape[0])[:, np.newaxis].repeat(frame.shape[1], axis=1)
            spatimg = np.arange(frame.shape[1])[:, np.newaxis].repeat(frame.shape[0], axis=1).T
            tilts = waveimg / (frame.shape[0] - 1)
            global_sky = np.zeros(frame.shape)
            if not self._use_diff:
                rn2img2 = procimg.rn2_frame(datasec_img, ronoise2)
                basevar2 = procimg.base_variance(rn2img2, darkcurr=darkcurr, exptime=exptime)
                frame2_for_ivar = ((frame2 / self._gain) + msdark) * self._gain
                rawvarframe2 = procimg.variance_model(basevar2, frame2_for_ivar)
                # Ivar
                ivar2 = utils.inverse(rawvarframe2)
            # Flatfield the data (note, this needs to be done after calculating the RO noise)
            frame /= msflat
            frame2 /= msflat
            # Set the masks
            thismask = np.ones(frame.shape, dtype=np.bool) & np.logical_not(bpm)
            ingpm = thismask.copy()
            # Find an estimate of the slit edges
            cen = 99 - 35.0 * np.linspace(0, 1, frame.shape[self._specaxis]) + 56
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
            ledge = cen - 100
            redge = cen + 100
            boxcar_rad = 3.0
            trc_pos = findobj_skymask.objs_in_slit(
                frame, ivar, thismask, ledge, redge,
                ncoeff=self._polyord, boxcar_rad=boxcar_rad,
                show_fits=self._plotit, nperslit=1)
            if self._use_diff:
                trc_neg = findobj_skymask.objs_in_slit(
                    -frame, ivar, thismask, ledge, redge,
                    ncoeff=self._polyord, boxcar_rad=boxcar_rad,
                    show_fits=self._plotit, nperslit=1)
            else:
                trc_neg = findobj_skymask.objs_in_slit(
                    frame2, ivar, thismask, ledge, redge,
                    ncoeff=self._polyord, boxcar_rad=boxcar_rad,
                    show_fits=self._plotit, nperslit=1)
            if self._plotit:
                spec = np.arange(frame.shape[self._specaxis])
                plt.imshow(frame.T, vmin=-200, vmax=200, origin='lower')
                plt.plot(spec, trc_pos[0].TRACE_SPAT + boxcar_rad, 'b-')
                plt.plot(spec, trc_neg[0].TRACE_SPAT + boxcar_rad, 'b-')
                plt.plot(spec, trc_pos[0].TRACE_SPAT - boxcar_rad, 'r-')
                plt.plot(spec, trc_neg[0].TRACE_SPAT - boxcar_rad, 'r-')
                plt.plot(spec, trc_pos[0].TRACE_SPAT - 40, 'g-')
                plt.plot(spec, trc_pos[0].TRACE_SPAT - 100, 'g-')
                plt.show()

            all_traces.append(trc_pos)
            all_traces.append(trc_neg)
            if self._mean_skycoeff:
                continue
            elif self._step_extract:
                # Optimal Extraction
                skymodel = np.zeros_like(frame)
                objmodel = np.zeros_like(frame)
                ivarmodel = np.zeros_like(frame)
                extractmask = np.zeros_like(frame)
                trcs = [trc_pos, trc_neg]
                isstd = False  # True
                if self._ext_sky: isstd = False
                for tt in range(2):
                    if self._ext_sky:
                        ee = 1 - tt
                    else:
                        ee = tt
                    if self._use_diff:
                        scl = 1
                        if tt == 1: scl = -1
                        ivar_use = ivar
                        extfrm_use = scl * frame
                    else:
                        if tt == 0:
                            ivar_use = ivar
                            extfrm_use = frame
                        else:
                            ivar_use = ivar2
                            extfrm_use = frame2
                    # Are we doing basis fitting
                    if self._step_basis or self._step_skycoeffs:
                        retval = self.basis_fit(extfrm_use, ivar_use, tilts, waveimg, spatimg, trcs[ee], 2 * ff + tt,
                                                skycoeffs=self._step_skycoeffs)
                        if retval:
                            continue
                    else:
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
                            bsp=0.1,
                            std=isstd,
                            adderr=0.0002,
                            force_gauss=False,
                            sn_gauss=4,
                            show_profile=self._plotit,
                            use_2dmodel_mask=False,
                            no_local_sky=False)
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
        #        plt.show()
        if self._mean_skycoeff:
            embed()
            assert (False)
            # Load the sky coefficients
            nfiles = 2 * len(self._matches)
            nbasis = 3
            nspec, nspat = frame.shape
            all_coeffs = np.zeros((frame.shape[0], nbasis, nfiles))
            for ff in range(nfiles):
                inname = self._procpath + "skycoeffs_{0:02d}.npy".format(ff)
                all_coeffs[:, :, ff] = np.load(inname)
            nodpos = np.array(
                [1.0, -6.5, -1, 6.5, -5.5, 2.0, -2.0, 5.5, -5.0, 2.5, -2.5, 5.0, -4.5, 3, -3, 4.5, -3.5, 4.0, -4.0, 3.5,
                 -1.5, 6.0, -6.0, 1.5, -0.0, 0.0])
            #            for ss in range(1478,1479):
            #                plt.scatter(all_coeffs[ss,1,:], all_coeffs[ss,2,:], c=nodpos, cmap='rainbow')
            #            plt.show()
            allspecimg = trace_tilt(all_traces[0][0].TRACE_SPAT, trcnum=40, plotit=True)
            tiltspl = interpolate.RectBivariateSpline(np.arange(nspec), np.arange(nspat), allspecimg)
            if False:
                # Plot it up
                for ss in range(nfiles):
                    scl = 1
                    if ss >= nfiles - 2: scl = (20 * 10) / (7 * 9)  # Ratio of the exposure times
                    plt.subplot(211)
                    plt.plot(tiltspl.ev(np.arange(nspec), all_traces[ss][0].TRACE_SPAT), scl * all_coeffs[:, 1, ss])
                    plt.subplot(212)
                    plt.plot(tiltspl.ev(np.arange(nspec), all_traces[ss][0].TRACE_SPAT), scl * all_coeffs[:, 2, ss])
                plt.show()
            # Cross-correlate to find shifts
            nfit = 3
            lagvals = np.zeros(nfiles)
            for bb in range(1, nbasis):
                for ss in range(0, nfiles):
                    scl = 1
                    if ss >= nfiles - 2: scl = (20 * 10) / (7 * 9)  # Ratio of the exposure times
                    corr = signal.correlate(all_coeffs[:, bb, 0], scl * all_coeffs[:, bb, ss])
                    lags = signal.correlation_lags(all_coeffs.shape[0], all_coeffs.shape[0])
                    amax = np.argmax(corr)
                    coeff = np.polyfit(lags[amax - nfit:amax + nfit + 1], corr[amax - nfit:amax + nfit + 1], 2)
                    lagvals[ss] += -0.5 * coeff[1] / coeff[0]
            #                     plt.subplot(211)
            #                     plt.plot(np.arange(nspec)+lagvals[ss], scl*all_coeffs[:,1,ss])
            #                     plt.subplot(212)
            #                     plt.plot(np.arange(nspec)+lagvals[ss], scl*all_coeffs[:,2,ss])
            #             plt.show()
            lagvals *= 1 / (nbasis - 1)  # This takes the average of all bases
            # Combine all coefficients
            allx, ally1, ally2 = np.array([]), np.array([]), np.array([])
            for ss in range(nfiles):
                scl = 1
                if ss >= nfiles - 2: scl = (20 * 10) / (7 * 9)  # Ratio of the exposure times
                allx = np.append(allx, np.arange(nspec) + lagvals[ss])
                ally1 = np.append(ally1, scl * all_coeffs[:, 1, ss])
                ally2 = np.append(ally2, scl * all_coeffs[:, 2, ss])
            asrt = np.argsort(allx)
            spl1 = interpolate.splrep(allx[asrt], ally1[asrt], task=-1, t=np.arange(1, nspec, 3))
            spl2 = interpolate.splrep(allx[asrt], ally2[asrt], task=-1, t=np.arange(1, nspec, 3))
            for ss in range(nfiles):
                scl = 1
                if ss >= nfiles - 2: scl = (20 * 10) / (7 * 9)  # Ratio of the exposure times
                plt.subplot(211)
                plt.plot(np.arange(nspec) + lagvals[ss], scl * all_coeffs[:, 1, ss])
                plt.subplot(212)
                plt.plot(np.arange(nspec) + lagvals[ss], scl * all_coeffs[:, 2, ss])
            plt.subplot(211)
            plt.plot(allx[asrt], interpolate.splev(allx[asrt], spl1), 'r-')
            plt.subplot(212)
            plt.plot(allx[asrt], interpolate.splev(allx[asrt], spl2), 'r-')
            plt.show()
            outname = self._procpath + "skycoeffs_splinefit_k3.npy"
            np.save(outname, np.column_stack((spl1[0], spl1[1], spl2[0], spl2[1])))
            outname = self._procpath + "skycoeffs_splinefit_k3_tiltimg.npy"
            np.save(outname, allspecimg)
            outname = self._procpath + "skycoeffs_splinefit_k3_lags.npy"
            np.save(outname, lagvals)

    def step_wavecal_prelim(self):
        usePath = self._altpath
        # if self._use_diff: usePath = self._procpath
        rwf.wavecal_prelim(usePath)
        # rwf.wavecal_telluric(usePath)

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
        usePath = self._altpath + "alt_"
        if self._use_diff: usePath = self._procpath
        for sp in range(nspec):
            wave, flux, flue, fitr = raw_wav[sp, :], raw_flx[sp, :], raw_err[sp, :], 1 - bpm[sp, :]
            ww = np.where((wave > lminwv) & (wave < lmaxwv))
            wf = np.where((wave < fminwv) | (wave > fmaxwv))
            fitr[wf] = 0
            outname = usePath + "tet02OriA_ALIS_spec{0:02d}.dat".format(sp)
            np.savetxt(outname, np.transpose((wave[ww], flux[ww], flue[ww], fitr[ww])))
            print("File written: {0:s}".format(outname))
            datlines += "  {0:s}   specid=He{1:02d}    fitrange=columns   resolution=vfwhm(3.657crires)  shift=vshiftscale(0.0,1.0)  columns=[wave,flux,error,fitrange,continuum]  plotone=True   label=HeI_10830_{1:02d}\n".format(
                outname, sp)
            zerolines += "  constant 0.0 specid=He{0:02d}\n".format(sp)
            strall += "He{0:02d},".format(sp)
        print(
            "\n\n\nHere is some informtion to run with ALIS to fix the wavelength scale. This must be done before you can proceeed to the next step:\n\n")
        print(datlines)
        print(zerolines)
        print(strall)

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
        for sp in range(nspec):
            wave, flux, flue, fitr = raw_wav[sp, :], raw_flx[sp, :], raw_err[sp, :], 1 - bpm[sp, :]
            ww = np.where((wave > lminwv) & (wave < lmaxwv))
            wf = np.where((wave < fminwv) | (wave > fmaxwv))
            skyname = self._procpath + "spec1d_{0:02d}_{1:s}_sky.dat".format(sp // 2, self._nods[sp % 2])
            sky_counts, sky_error = np.loadtxt(skyname, unpack=True, usecols=(1, 2))
            # Load the old and corrected wavelength scale
            tmpnameAz = self._procpath + "tet02OriA_ALIS_spec{0:02d}_wzcorr.dat".format(sp)
            out_waveAz, inwaveAz, flux = np.loadtxt(tmpnameAz, unpack=True, usecols=(0, 1, 2))
            wA = np.where(np.in1d(wave, inwaveAz))
            np.savetxt(skyname.replace("_sky", "_sky_wzcorr"),
                       np.transpose((out_waveAz, sky_counts[wA], sky_error[wA])))
            plt.plot(out_waveAz, sky_counts[wA], 'k-', drawstyle='steps-mid')
        plt.show()

    def step_combspec(self):
        out_wave, raw_specs = self.comb_prep(use_corrected=True)
        if step_combspec_rebin:
            self.comb_rebin(out_wave, raw_specs)
        else:
            wave_bins = out_wave.copy()
            npix, nspec = out_wave.size, len(raw_specs)
            out_flux = self._maskval * np.ones((npix, nspec))
            out_flue = self._maskval * np.ones((npix, nspec))
            # Reject
            raw_wav, raw_flx, raw_err, bpm = self.comb_reject(out_wave, raw_specs, use_corrected=True)
            # Find all good pixels and create the final histogram
            for ss in range(bpm.shape[0]):
                spec_use = np.ones(bpm.shape, dtype=np.bool)
                for mm in range(bpm.shape[0] - ss, bpm.shape[0]):
                    spec_use[mm, :] = False
                out_wave, spec, specerr = self.comb_spectrum(wave_bins, raw_wav, raw_flx, raw_err, bpm, spec_use)
                fitr = np.zeros(out_wave.size)
                fitr[np.where(
                    ((out_wave > 10827.0) & (out_wave < 10832.64)) | ((out_wave > 10833.16) & (out_wave < 10839)))] = 1
                np.savetxt(
                    self._procpath + "tet02_OriA_HeI10833_scaleErr_wzcorr_fitr_comb{0:02d}.dat".format(
                        bpm.shape[0] - ss),
                    np.transpose((out_wave, spec, specerr, fitr)))
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
                out_wave, spec, specerr = self.comb_rebin(out_wave, raw_specs_samp, save=False)
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
            out_wave, spec, specerr = self.comb_rebin(out_wave, raw_specs_samp, save=False)
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
        #         spec_use = np.ones(bpm.shape, dtype=np.bool)
        #         for mm in range(nspec-ss, nspec):
        #             spec_use[ffs[mm], :] = False
        #         out_wave, spec, specerr, specerr_new = comb_spectrum(wave_bins, raw_wav, raw_flx, raw_err, bpm, spec_use, get_specerr_orig=True)
        #         snr[nn], snradj[nn] = scale_variance(out_wave, spec, specerr, getSNR=True)
        #     snr_all[nspec-ss-1] = np.mean(snr)
        #     snr_all_err[nspec-ss-1] = np.std(snr)
        #     snr_all_adj[nspec-ss-1] = np.mean(snradj)
        #     snr_all_adj_err[nspec-ss-1] = np.std(snradj)
        # np.savetxt("SNR_NumExpCombine.dat", np.transpose((snr_all, snr_all_err, snr_all_adj, snr_all_adj_err)))

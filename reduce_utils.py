# Perform some fits to each spectrum to determine an approximate wavelength solution
# Then go through and perform fits to the telluric absorption.

import numpy as np
import astropy.io.fits as fits
from matplotlib import pyplot as plt
from IPython import embed
from scipy.optimize import curve_fit
from scipy.special import wofz

wave_he = 10833.306444
crires_vfwhm = 299792.458/80000.0
polyord = 3
nods = ['A', 'B']

def calculate_wavelength(pixels, wscl, wcons):
    # 10833.306444 = the vacuum wavelength of the strongest line in the triplet
    # wcons is the constant offset or shift
    # wscl is the angstroms/pixel
    return  wscl*(pixels-1657.5) + wcons + wave_he

def voigt(wavein, logn, dopp, wcen, fval, gamma):
    wv = wcen * 1.0e-8
    cold = 10.0**logn
    zp1=1.0
    bl=dopp*wv/2.99792458E5
    a=gamma*wv*wv/(3.76730313461770655E11*bl)
    cns=wv*wv*fval/(bl*2.002134602291006E12)
    cne=cold*cns
    ww=(wavein*1.0e-8)/zp1
    v=wv*ww*((1.0/ww)-(1.0/wv))/bl
    tau = cne*wofz(v + 1j * a).real
    return np.exp(-1.0*tau)

def conv(x, y, cont0, vfwhm):
    """
    Define the functional form of the model
    --------------------------------------------------------
    x  : array of wavelengths
    y  : model flux array
    p  : array of parameters for this model
    --------------------------------------------------------
    """
    sigd = vfwhm / ( 2.99792458E5 * ( 2.0*np.sqrt(2.0*np.log(2.0)) ) )
    if np.size(sigd) == 1: cond = sigd > 0
    else: cond = np.size(np.where(sigd > 0.0)) >= 1
    if cond:
        ysize=y.size
        fsigd=6.0*sigd
        dwav = 0.5*(x[2:]-x[:-2])/x[1:-1]
        dwav = np.append(np.append(dwav[0],dwav),dwav[-1])
        if np.size(sigd) == 1:
            df= int(np.min([np.int(np.ceil(fsigd/dwav).max()), ysize//2 - 1]))
            yval = cont0*np.ones(2*df+1)
            yval[df:2*df+1] = (x[df:2*df+1]/x[df] - 1.0)/sigd
            yval[:df] = (x[:df]/x[df] - 1.0)/sigd
            gaus = np.exp(-0.5*yval*yval)
            size = ysize + gaus.size - 1
            fsize = 2 ** np.int(np.ceil(np.log2(size))) # Use this size for a more efficient computation
            conv = np.fft.fft(y, fsize)
            conv *= np.fft.fft(gaus/gaus.sum(), fsize)
            ret = np.fft.ifft(conv).real.copy()
            del conv
            return ret[df:df+ysize]
        elif np.size(sigd) == szflx:
            yb = y.copy()
            df=np.min([np.int(np.ceil(fsigd/dwav).max()), ysize//2 - 1])
            for i in range(szflx):
                if sigd[i] == 0.0:
                    yb[i] = y[i]
                    continue
                yval = cont0*np.ones(2*df+1)
                yval[df:2*df+1] = (x[df:2*df+1]/x[df] - 1.0)/sigd[i]
                yval[:df] = (x[:df]/x[df] - 1.0)/sigd[i]
                gaus = np.exp(-0.5*yval*yval)
                size = ysize + gaus.size - 1
                fsize = 2 ** np.int(np.ceil(np.log2(size))) # Use this size for a more efficient computation
                conv  = np.fft.fft(y, fsize)
                conv *= np.fft.fft(gaus/gaus.sum(), fsize)
                ret   = np.fft.ifft(conv).real.copy()
                yb[i] = ret[df:df+ysize][i]
            del conv
            return yb
        else:
            print("vfwhm and flux arrays have different sizes.")
    else: return y

def model(pixels, zerolev, cont0, cont1, cont2, wscl, wcons, logn, bval):
    # Calculate the wavelength solution
    wave = calculate_wavelength(pixels, wscl, wcons)
    # Calculate the continuum
    cont = cont0 + (cont1 * (wave-wave_he)) + (cont2 * (wave-wave_he)**2)
    # Calculate the absorption
    absmodel = np.ones(pixels.size)
    absmodel *= voigt(wave, logn, bval, 10832.057472, 5.9902e-02, 1.0216e+07)
    absmodel *= voigt(wave, logn, bval, 10833.216751, 1.7974e-01, 1.0216e+07)
    absmodel *= voigt(wave, logn, bval, 10833.306444, 2.9958e-01, 1.0216e+07)
    # Combine the model
    model = zerolev + cont*absmodel
    # return model
    # Convolve the model
    modconv = conv(wave, model, cont0+zerolev, crires_vfwhm)
    modconv[:5] = model[:5]
    modconv[-5:] = model[-5:]
    return modconv


def wavecal_prelim(procpath, numspec, mn_fit, mx_fit, basis=True):
    bvals = []
    for ff in range(numspec):
        for nn, nod in enumerate(nods):
            if basis:
                filn = "spec1d_{0:02d}.dat".format(2 * ff + nn)
                box_wave, box_cnts, box_cerr, box_sky = np.loadtxt(procpath+filn, unpack=True)
            else:
                filn = "spec1d_{0:02d}_{1:s}.dat".format(ff, nod)
                box_wave, box_cnts, box_cerr, box_wave, opt_cnts, opt_cerr = np.loadtxt(procpath + filn, unpack=True)
            for bo in range(2):
                if bo==1 and basis: continue

                # Grab the data
                if bo==0:
                    wfit = np.where((box_wave>mn_fit) & (box_wave<mx_fit))
                    pfit, ffit, efit = box_wave[wfit], box_cnts[wfit], box_cerr[wfit]
                else:
                    wfit = np.where((opt_wave>mn_fit) & (opt_wave<mx_fit))
                    pfit, ffit, efit = opt_wave[wfit], opt_cnts[wfit], opt_cerr[wfit]
                # Prepare the fitting
                zerolev = 0.0
                cold = 13.65
                bval = 6.7
                cont = [np.median(ffit), 0.0, 0.0]
                # tet01 Ori A:
                wpar = [1.3/35.0, 0.0]  # 1.3/35.0 is an estimate of the Angstroms/pixel and 0.0 means pixel 1657.5 = wavelength 10833.306444
                # PDS 241:
                #wpar = [1.3 / 35.0, -150.0 / 35.0]  # PDS 241 --> -150 means "this absorption occurs 150 pixels to the right of tet01 Ori A"
                if False:
                    # Use this code to check wpar values
                    mfit = model(np.arange(2000), *params)
                    plt.plot(np.arange(2000), mfit, 'b-')
                    mfit = model(pfit, *params)
                    plt.plot(pfit, mfit, 'r-')
                    plt.show()
                params = [zerolev, cont[0], cont[1], cont[2], wpar[0], wpar[1], cold, bval]
                # Perform the fit
                popt, pcov = curve_fit(model, pfit, ffit, p0=params, sigma=efit)
                # Plot the final result
                mfit = model(pfit, *popt)
                bvals.append(popt[-1])
                print(filn, bo)
                wvtmp = calculate_wavelength(pfit, popt[4], popt[5])
                vltmp = 299792.458*(wvtmp-10833.306444) / wvtmp
                cont = popt[1] + (popt[2] * (wvtmp - wave_he)) + (popt[3] * (wvtmp - wave_he) ** 2)
                #np.savetxt("PDS241_tmp.dat", np.column_stack((vltmp, ffit/cont)))
                plt.plot(vltmp, cont, 'c-')
                plt.plot(vltmp, ffit, 'k-', drawstyle='steps-mid')
                plt.plot(vltmp, mfit, 'r-')
                plt.axvline(36.6, color='m')
                plt.xlabel("Velocity relative to strongest He I* absorption")
                plt.ylabel("Flux")
                plt.show()
                # embed()
                # Apply the wavelength solution and subtract the zero-level
                if bo == 0:
                    box_wave = calculate_wavelength(box_wave, popt[4], popt[5])
                    #box_cnts -= popt[0]
                else:
                    opt_wave = calculate_wavelength(opt_wave, popt[4], popt[5])
                    #opt_cnts -= popt[0]
            # Output the files
            outfiln = filn.replace("spec1d", "spec1d_wave")
            nrm_val = np.median(box_cnts[wfit])
            if basis:
                np.savetxt(procpath + outfiln, np.transpose((box_wave, box_cnts/nrm_val, box_cerr/nrm_val, box_sky/nrm_val)))
            else:
                np.savetxt(procpath+outfiln, np.transpose((box_wave, box_cnts/nrm_val, box_cerr/nrm_val, opt_wave, opt_cnts/nrm_val, opt_cerr/nrm_val)))

def wavecal_telluric(procpath, numspec):
    tmp_data_lines = np.array([10777.25, 10807.53, 10803.45, 10814.61, 10835.96, 10837.86])
    tmp_vacm_lines = np.array([10775.83, 10806.63, 10802.54, 10813.75, 10835.08, 10836.95])
    vacm_lines = np.zeros(tmp_vacm_lines.size)
    wavsky, fluxsky = np.loadtxt("calibrations/skyabs_highres.dat", unpack=True)
    wavsky *= 10.0  # Convert from nm to A
    # Find the actual centroids of the vacuum lines:
    # plt.plot(wavsky, fluxsky, 'k-')
    for vv in range(tmp_vacm_lines.size):
        wtmp = np.argmin(np.abs(wavsky-tmp_vacm_lines[vv]))
        wmax = wtmp-10 + np.argmin(fluxsky[wtmp-10:wtmp+10])
        coeff = np.polyfit(wavsky[wmax-2:wmax+3] - wavsky[wmax], fluxsky[wmax-2:wmax+3], 2)
        vacm_lines[vv] = wavsky[wmax] - 0.5*coeff[1]/coeff[0]
        # model = np.polyval(coeff, wavsky[wmax-2:wmax+3] - wavsky[wmax])
        # plt.plot(wavsky[wmax-2:wmax+3], model, 'r-')
        # plt.plot([wavsky[wtmp]], [fluxsky[wtmp]], 'bx')
        # plt.plot([wavsky[wmax]], [fluxsky[wmax]], 'rx')
    #     plt.axvline(vacm_lines[vv], color='r')
    # plt.show()
    # Now fit the data
    for ff in range(numspec):
        for nod in nods:
            filn = "spec1d_wave_{0:02d}_{1:s}.dat".format(ff, nod)
            box_wave, box_cnts, box_cerr, opt_wave, opt_cnts, opt_cerr = np.loadtxt(procpath+filn, unpack=True)
            pixels = np.arange(box_wave.size)
            ################
            # Boxcar wavecal
            if False:
                data_lines = np.zeros(tmp_data_lines.size)
                # plt.plot(box_wave, box_cnts, 'k-')
                for vv in range(tmp_data_lines.size):
                    wtmp = np.argmin(np.abs(box_wave-tmp_data_lines[vv]))
                    wmax = wtmp-10 + np.argmin(box_cnts[wtmp-10:wtmp+10])
                    coeff = np.polyfit(pixels[wmax-3:wmax+4] - pixels[wmax], box_cnts[wmax-3:wmax+4], 2)
                    data_lines[vv] = pixels[wmax] - 0.5*coeff[1]/coeff[0]
                #     plt.axvline(data_lines[vv], color='r')
                # plt.show()
                # Fit the solution and update the wavelength calibration
                coeff = np.polyfit(data_lines, vacm_lines, polyord)
                new_box_wave = np.polyval(coeff, pixels)
            else:
                # Don't change the boxcar wavelength calibration
                new_box_wave = box_wave.copy()
            #################
            # Optimal wavecal
            data_lines = np.zeros(tmp_data_lines.size)
            plt.plot(opt_wave, opt_cnts, 'k-')
            mfilt = conv(opt_wave, opt_cnts, np.median(opt_cnts), crires_vfwhm)
            for vv in range(tmp_data_lines.size):
                wtmp = np.argmin(np.abs(opt_wave-tmp_data_lines[vv]))
                wmax = wtmp-10 + np.argmin(mfilt[wtmp-10:wtmp+10])
                coeff = np.polyfit(pixels[wmax-3:wmax+4] - pixels[wmax], mfilt[wmax-3:wmax+4], 2)
                data_lines[vv] = pixels[wmax] - 0.5*coeff[1]/coeff[0]
                model = np.polyval(coeff, pixels[wmax-3:wmax+4] - pixels[wmax])
                plt.plot(opt_wave[wmax-3:wmax+4], model, 'r-')
                plt.axvline(data_lines[vv], color='r')
            plt.show()
            # Fit the solution and update the wavelength calibration
            #embed()
            ww, polyfit = np.arange(data_lines.size), polyord
            if (ff == 10): ww, polyfit = np.array([0,2,3,4]), polyord-1
            coeff = np.polyfit(data_lines[ww], vacm_lines[ww], polyfit)
            Appix = np.polyval(coeff, np.array([pixels.size//2, pixels.size//2+1]))
            tstvals = (np.polyval(coeff, data_lines[ww]) - vacm_lines[ww])/(Appix[1]-Appix[0])
            print("Test mean, std (pixels) =", np.mean(tstvals), np.std(tstvals))
            # print(tstvals)
            new_opt_wave = np.polyval(coeff, pixels)
            # Save the output
            outfiln = filn.replace("spec1d_wave", "spec1d_waveTell")
            print("Saving {0:s}".format(outfiln))
            np.savetxt(procpath+outfiln, np.transpose((new_box_wave, box_cnts, box_cerr, new_opt_wave, opt_cnts, opt_cerr)))

# plt.show()
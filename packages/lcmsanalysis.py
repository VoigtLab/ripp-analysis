import lcms
import pandas as pd
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
import pickle
import scipy
from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.signal import find_peaks


def asym_peak(t, a0, a1, a2, a3, a4):
    """function takes a retention time, and five peak shape parameters:'
    #a0 = pars[0]  # peak area
    #a1 = pars[1]  # elution time
    #a2 = pars[2]  # width of gaussian
    #a3 = pars[3]  # exponential damping term
    #a4 - baseline adjustment
    returns the y-axis intensity at that retention time for those parameters.
    from Anal. Chem. 1994, 66, 1294-1301
    """
    term1 = (a0/(2*a3))
    term2 = np.exp((a2**2/(2*a3**2)) + (a1 - t)/a3)
    term3 = erf(((t-a1)/(np.sqrt(2.0)*a2)) - (a2/(np.sqrt(2.0)*a3))) + 1.0
    f = (term1 * term2 * term3) + a4
    return f

def asym_peak_residuals(pars, t, intensity):
    'from Anal. Chem. 1994, 66, 1294-1301'
    a0 = pars[0]  # peak area
    a1 = pars[1]  # elution time
    a2 = pars[2]  # width of gaussian
    a3 = pars[3]  # exponential damping term
    a4 = pars[4]
    f = (a0/2/a3*np.exp(a2**2/2.0/a3**2 + (a1 - t)/a3)
         *(erf((t-a1)/(np.sqrt(2.0)*a2) - a2/np.sqrt(2.0)/a3) + 1.0)) + a4
    return f - intensity

def integrate_mass(peak_info, mz_range = [400, 2500], rt_range = [1,5.5], plot=False):
    """Function takes a dictionary (peak_info) which includes metadata about a potential
    peak in an LC-MS file. Minimum data needed in dictionary:
    key 'lcd' - the lc-ms datafile object
    key 'mass' - the mass of interest
    key 'dm' - the delta around the mass
    
    Returns the same dictionary back, with new keyed information added to the dictionary:
    key 'msdf' -  the raw extracted compound chromatogram information
    key 'charges_at_peak' - the charge adducts observed of the compound
    key 'fit_succes' - the success of fitting the ECC with a skewed gaussian (0 or 1)
    All fitting parameters, keys: 'fit_area', 'fit_rt', 'fit_width', 'fit_skew', 'fit_baseline',
         'fit_area_err', 'fit_rt_err', 'fit_width_err', 'fit_skew_err', and 'fit_baseline_err'
    key 'observed_mass' - the observed mass
    key 'observed_mass_std' - the observed mass standard deviation
        *note that the observed mass and observed mass standard deviation are calculated from the
        m/z signals filtered from the input 'mass' and 'dm' and can therefore change depending on
        those input parameters (a larger input 'dm' can lead to a larger 'observed_mass_std' and 
        should be used accordingly
    key 'peak_msdf' - msdf, but filtered for retention times around the peak and with m/z's
        grouped by charge and m/z (with intensities summed), and mass backcalculated from each m/z
    key 'big_peaks' - the number of peaks with prominance >80% of the max range observed
    key 'little_peaks' - the number of peaks with prominance >40% of the max range observed
    """
    
    min_mz, max_mz = mz_range
    lcd = peak_info['lcd']
    mass = peak_info['mass']
    delta_mass = peak_info['dm']
    
    msdf = lcd.ecc(mass, delta_mass)
    
    ##################### ANALYSIS ###################
    if len(msdf) < 10:
        peak_info['charges_at_peak'] = pd.Series()
        print("NO PEAK: ", peak_info['peak_type'], " in ", peak_info['extract'])
        return peak_info
    
    #### GRAB THE MOST INTENSE RT FOR EACH CHARGE STATE #######
    #this ends up being just the most abundant row (mz, rt, int, charge) for each charge state
    high_intensities_per_charge = msdf.sort_values('int').groupby('charge', group_keys=False).nth(-1).reset_index()
    
    
    peak_time = high_intensities_per_charge['rt'].round(1).mode()[0]

    #
    charges_at_peak = high_intensities_per_charge[(high_intensities_per_charge['rt'] > peak_time - 0.2) &
                           (high_intensities_per_charge['rt'] < peak_time + 0.2)]['charge']
    
    peak_info['msdf'] = msdf
    peak_info['charges_at_peak'] = charges_at_peak
    
    #reset charges_at_peak variable to just be the count of charges, instead of a list of the charges
    charges_at_peak = charges_at_peak.count()    
    
    ###### FIT SKEWED GAUSSIAN ON RAW DATA ######
    #Group all EICS by retention time (sum the intensities) for fitting the chromatogram
    fitting = msdf.groupby(['rt'], group_keys=False).agg({'mz':'count', 'int':'sum'}).reset_index()
    
    #peak_info['fitting_df'] = fitting
    
    #plot chromatogram of compound
    #plt.figure(figsize=(10,5))
    sns.lineplot(data = fitting, x='rt', y='int')
    
    #this is just to get a rough estimate of the peak area
    fitting_integrate = fitting[(fitting['rt'] > peak_time-0.3) &
                                (fitting['rt'] < peak_time+0.3)]
    rough_area = np.trapz(y=fitting_integrate['int'], x=fitting_integrate['rt'])
    
    #Curvefitting!
    try:
        parguess = (max([rough_area, 101]), peak_time, 0.03, 0.1, fitting['int'].min())
        (peak_area, peak_rt, peak_width, peak_skew, peak_baseline), pcov = \
                        curve_fit(asym_peak, fitting['rt'], fitting['int'],
                                  p0=parguess,
                                  bounds = ((100, 1, 0.001, 0.01, 100),
                                            (1000000000, fitting['rt'].max()+0.1, 0.1, 10, 10000000)),
                                  method='trf', maxfev=1000000)

        area_error, rt_error, width_error, skew_error, baseline_error = np.sqrt(np.diag(pcov))
        peak_info['fit_success'] = 1
    except:
        print(parguess)
        print(peak_info)
        peak_info['fit_success'] = 0
        return peak_info
    
    if plot:
        x = np.linspace(1,5, 1000)
        y = asym_peak(x, peak_area, peak_rt, peak_width, peak_skew, peak_baseline)
        plt.plot(x,y)
    
    #### GET THE MOLECULAR WEIGHT OBSERVED
    peak_msdf = msdf[(msdf['rt'] > peak_rt - (2*peak_width)) & (msdf['rt'] < peak_rt + (2*peak_width))]
    if len(peak_msdf) < 10:
        print("NO PEAK: ", peak_info['peak_type'], " in ", peak_info['extract'])
        return peak_info
    peak_msdf = peak_msdf.groupby(['mz', 'charge'], group_keys=False).agg({'int': 'sum'}).reset_index()
    peak_msdf['mass'] = peak_msdf['mz'] * peak_msdf['charge'] - peak_msdf['charge']
    #For each charge, grab the largest m/z peaks per charge state (at least 5, at most 20), intermediate
    # values based on the number of m/z peaks observed
    submsdf = peak_msdf.groupby('charge', group_keys=False).apply(lambda x:
                                            x.nlargest(max([5, min([20, int(len(x)/10)])]), 'int'))
    if len(submsdf) < 3:
        print("NO PEAK: ", peak_info['peak_type'], " in ", peak_info['extract'])
        return peak_info
    peak_info['observed_mass'] = np.average(submsdf['mass'], weights=submsdf['int'])
    peak_info['observed_mass_std'] = np.sqrt(np.cov(submsdf['mass'], aweights=submsdf['int']))
    peak_info['peak_msdf'] = peak_msdf
    peak_info['fit_area'] = peak_area
    peak_info['fit_rt'] = peak_rt
    peak_info['fit_width'] = peak_width
    peak_info['fit_skew'] = peak_skew
    peak_info['fit_baseline'] = peak_baseline
    peak_info['fit_area_err'] = area_error
    peak_info['fit_rt_err'] = rt_error
    peak_info['fit_width_err'] = width_error
    peak_info['fit_skew_err'] = skew_error
    peak_info['fit_baseline_err'] = baseline_error
    
    ##### FIND PEAKS ON ROLLING WINDOW AVERAGED DATA ##########
    rolling_df = fitting.set_index(pd.TimedeltaIndex(fitting['rt'], unit='m')).rolling('1s').mean()
    x = list(rolling_df['rt'])
    y = list(rolling_df['int'])
    big_peak_inds,_ = find_peaks(y, prominence=(max(y) - min(y)) * 0.8)
    low_peak_inds, _ = find_peaks(y, prominence=(max(y) - min(y)) * 0.4)  
    if plot:
        plt.plot(rolling_df['rt'], rolling_df['int'])
        plt.scatter([x[i] for i in big_peak_inds], [y[i] for i in big_peak_inds], marker="o", color='red')
        plt.scatter([x[i] for i in low_peak_inds], [y[i] for i in low_peak_inds], marker="o", color='black')
    
    #Print out fit metrics
    metrics = ""
    for thing, val, err in zip(['area', 'time', 'width', 'skew', 'baseline'],
                               [peak_area, peak_rt, peak_width, peak_skew, peak_baseline],
                               [area_error, rt_error, width_error, skew_error, baseline_error]):
        metrics = metrics + "{} : {} +/- {}\n".format(thing, val, err)
    metrics = metrics + "charges at peak: {}".format(charges_at_peak)
    
    if plot:
        ax = plt.gca()
        plt.text(0.05, 0.95,metrics, horizontalalignment='left',
                 verticalalignment='top', transform=ax.transAxes)
        plt.title("{} - {}".format(peak_info['extract'], peak_info['peak_type']))
    
    peak_info['big_peaks'] = len(big_peak_inds)
    peak_info['little_peaks'] = len(low_peak_inds)
    
    return peak_info
    
def analyze_extract(sample, plots_directory = "", df_directory = "extract_dataframes"):
    """Function takes a dictionary that is generated from a row of the import excel file. At a
    minimum it needs:
    key 'ms' - the lcms machine used (value: 'qqq' or 'qtof')
    key 'mzxml_file' - the full path of the mzxml file
    key 'unmod_mass' - the mass of the unmodified peptide
    key 'mod_mass' - the mass of the ready-to-be-modified peptide (this is not the modified peptide mass!)
        most often this will match the unmod_mass, lasso peptides are an example where the
        peptide is cleaved and this mass would be the mass of the peptide after cleavage of the leader, but
        without cyclization.)
    key 'mass_shift' - expected mass shift of modification
    key 'mod_checks' - maximum number of modification mass shifts to check for
    key 'mod_num' - the number of expected mass shifts for the properly modified peptide
    key 'extract' - extract number for the sample
    """
    plots_directory = plots_directory.rstrip()
    ms_type = sample['ms']
    if ms_type == 'qqq':
        dm = 2
    elif ms_type == 'qtof':
        dm = 1
    else:
        dm = None
        print("No LCMS type given", ms_type)
        return None
    
    extract_number = sample['extract']

    if sample['mod_mass'] < 0 or not sample['mod_mass']:
        print("Row is not formatted properly.... No Mass: ", extract_number)
        return None
    if sample['sequence'] == "" or not sample['sequence'] or not type(sample['sequence']) == str:
        print(type(sample['sequence']))
        print("Row is not formatted properly.... No Sequence: ", extract_number)
        return None
    lcd = lcms.lcd(sample['mzxml_file'], min_int=1000)
    lcd.scan_filter()
    
    peak_info = dict(sample)
    peak_info['lcd'] = lcd
    peak_info['dm'] = dm
    peak_info['ms_type'] = ms_type
    
    ##################### GRABBING MASSES ######################
    unmod_mass = sample['unmod_mass']
    mod_mass = sample['mod_mass']
    mass_shift = sample['mod_shift']
    check_mods = int(sample['mod_checks'])
    expected_mods = int(sample['mod_num'])

    peak_info['mass_shift'] = mass_shift
    peak_info['check_mods'] = check_mods
    peak_info['expected_mods'] = expected_mods
    peak_info['sequence'] = sample['sequence'].strip().strip("*").strip("M").lstrip("M")
    
    expected_mod_mass = mod_mass + (expected_mods*mass_shift)
    
    #organized as 'type', mass, number of mods
    peaks = [('unmod', unmod_mass, 0),
             ('mod', expected_mod_mass, expected_mods)]
    
    
    if mod_mass != unmod_mass:
        peaks.append(('other', mod_mass, 0))
    else:
        for i in range(1, check_mods+1):
            if i == expected_mods:
                continue
            else:
                peaks.append(('other', mod_mass+(i*mass_shift), i))
    
    plot = True if plots_directory else False
    if plot:
        fig, axes = plt.subplots(nrows=2*len(peaks)+1,
                                 ncols=1, 
                                 figsize=(10.5,8+(8*len(peaks)*2)), 
                                 dpi=int(65536/(8+(8*len(peaks)*4))))
        plt.sca(axes[0])
        lcd.tic(plot=True)
        plt.title(extract_number)
    
    all_peaks = []
    for ind, peak in enumerate(peaks):
        if plot:
            plt.sca(axes[2*ind + 1])
        this_peak = dict(peak_info.items())
        this_peak['peak_type'] = peak[0]
        this_peak['mass'] = peak[1]
        this_peak['mod_num'] = peak[2]
        integrated_peak = integrate_mass(this_peak, mz_range = [500, 2500], rt_range = [1,6], plot=plot)
        
        if not integrated_peak:
            print('throwaway', extract_number)
            continue
        if 'fit_rt' in integrated_peak:
            if plot:
                plt.sca(axes[2*ind + 2])
            ip_rt = integrated_peak['fit_rt']
            _ = integrated_peak['lcd'].spectra(ip_rt-0.1, ip_rt+0.1,
                                               plot=plot, biggest=1000, mz_range=[500,2500])
        integrated_peak['lcd']
        all_peaks.append(integrated_peak)
        
    if plot:
        plt.tight_layout()
        plt.savefig('{}/{}.pdf'.format(plots_directory, extract_number))
        plt.close()
    
    
    if len(all_peaks) > 0:
        peaks_df = pd.DataFrame(all_peaks)
        with open("./{}/{}.pickle".format(df_directory, int(extract_number)), 'wb') as f:
                pickle.dump(peaks_df, f)
    
    #remove the big sized objects to not run out of ram
    for peak in all_peaks:
        if 'msdf' in peak:
            del peak['msdf']
        if 'lcd' in peak:
            del peak['lcd']
        if 'peak_msdf' in peak:
            del peak['peak_msdf']
    
    print("Extract: {}; {} of {} Peaks".format(extract_number, len([p for p in all_peaks if 'fit_rt' in p]), len(all_peaks)))
    return all_peaks

def check_continuity(charges):
    """takes a list of charges and returns the length of the longest continuous consecutive stretch
    """
    if not charges:
        return 0
    longest_streak = 1
    streak = 1
    previous = charges[0]
    for c in charges[1:]:
        if c == previous + 1:
            streak += 1
            if streak > longest_streak:
                longest_streak = streak
        else:
            streak = 0
        previous = c
    return longest_streak

def lcms_df_processor(data_df, min_charges=8, min_contiguous_charges=4, max_big_peaks=1,
                      max_little_peaks=3, min_qqq_area=10000, min_qtof_area=1000, group_peaks=True,
                      mod_other=True):
    data_df = data_df.fillna(0)
    
    data_df['contiguous_charges'] = data_df['charges_at_peak'].apply(lambda x: check_continuity(list(x)))
    data_df['measured_area'] = data_df['fit_area'].copy()
    
    remove_criteria = ((data_df['charges_at_peak'].apply(lambda x: x.count()) < min_charges) | 
                (data_df['contiguous_charges'] < min_contiguous_charges) |
                (data_df['big_peaks'] > max_big_peaks) |
                (data_df['little_peaks'] > max_little_peaks) |
                ((data_df['fit_area']<min_qtof_area) & (data_df['ms'] == 'qtof')) |
                ((data_df['fit_area']<min_qqq_area) & (data_df['ms'] == 'qqq')) |
                (data_df['fit_skew'] >1.5) |
                (data_df['fit_success'] == 0) |
                (data_df['fit_width'] > 0.0999999))

    data_df.loc[remove_criteria, "fit_area"] = 0.0
    
    if group_peaks:
        grouped_df = data_df.groupby(['extract', 'peak_type'], group_keys=False).agg(
            {'fit_area':'sum', 'fit_rt':'mean', 'media':'first', 'mod_plasmid':'first',
             'modification description':'first', 'pep_plasmid':'first', 'peptide description':'first',
             'sequence':'first', 'ms':'first'}).reset_index()
    else:
        grouped_df = data_df
    
    total_areas = grouped_df.groupby('extract', group_keys=False).agg({'fit_area':'sum'}).reset_index()
    
    grouped_df['total_area'] = grouped_df.apply(lambda x:
            float(total_areas[total_areas['extract'] == x['extract']]['fit_area']), axis=1)
    
    grouped_df['mod_area'] = grouped_df['fit_area']
    if mod_other:
        #This is specific to ThcoK/PadeK modifications. If the modification shift is 80 (phosphorylation) then
        # set the peak fraction of the modified peak for each extract to be modified + other modification states
        # (since any number of phosphorylations is considered phosphorylated)
        for extract in set(grouped_df['extract']):
            if grouped_df[grouped_df['extract'] == extract]['mod_plasmid'].isin([7137, 7138]).iloc[0]:
                ind = grouped_df[(grouped_df['extract'] == extract) & (grouped_df['peak_type'] == 'mod')].index
                mod_area = grouped_df[(grouped_df['extract'] == extract) & (grouped_df['peak_type'] == 'mod')]['mod_area']
                if len(mod_area):
                    mod_area = mod_area.iloc[0]
                else:
                    mod_area = 0
                other_area = grouped_df[(grouped_df['extract'] == extract) & (grouped_df['peak_type'] == 'other')]['mod_area']
                if len(other_area):
                    other_area = other_area.sum()
                else:
                    other_area = 0
                grouped_df.set_value(ind, 'mod_area', mod_area+other_area)
            
    grouped_df['mod_plasmid'] = grouped_df['mod_plasmid'].astype('int')
    
    grouped_df['peak_fraction'] = grouped_df['mod_area'] / grouped_df['total_area']
    
    return grouped_df.fillna(0)
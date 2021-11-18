import xmltodict
import struct
import base64
import codecs
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt



def unpack_format(data_struct):  # type: (dict) -> str
    fmt = ''
    if data_struct['@endian'] == 'little':
        fmt += '<'
    elif data_struct['@endian'] == 'big':
        fmt += '>'

    count = int(data_struct['@length'])

    if data_struct['@precision'] == '32':
        fmt += 'f'*count
    elif data_struct['@precision'] == '64':
        fmt += 'd'*count
    return fmt

def charge_list(molar_mass, dmass, mz_range=[500, 2000]):
    min_mz, max_mz = mz_range
    lowest_charge = int(molar_mass / max_mz)
    highest_charge = int(molar_mass / min_mz)
    charges = range(lowest_charge,highest_charge)
    chargemzs = [(molar_mass + charge) / charge for charge in charges]
    dms = [dmass / charge for charge in charges]
    return zip(charges, chargemzs, dms)

class lcd():
    """Python object for importing a single mass spec mzxml file and providing functions for analysis
    of the underlying dataframe that stores all the lcms data in long form
    """
    
    def __init__(self, filename, min_int = 1000):
        """function takes a filename and a minimum signal intensity to filter out noise"""
        with open(filename, 'rb') as f:
            tree = xmltodict.parse(f.read())
            self._spectra = self._read_data_xml(tree, min_int=min_int)
            self._df = pd.DataFrame(self._spectra, columns=['rt', 'mz', 'int'])
    
    def _read_data_xml(self, xml_tree, min_rt = 1.0, min_int = 1000):
        """internal function for reading an xml tree into a list of spectra at each timepoint"""
        #spectra variable will end up being a list of m/z,intensity tuples
        spectra = []
        i=0
        spec_count = len(xml_tree['mzData']['spectrumList']['spectrum'])

        for timepoint in xml_tree['mzData']['spectrumList']['spectrum']:
            i += 1
            # find RT
            RT = -1.0
            Polarity = 1.0
            for param in timepoint['spectrumDesc']['spectrumSettings']['spectrumInstrument']['cvParam']:
                if param['@name'] == 'TimeInMinutes':
                    RT = float(param['@value'])
                elif param['@name'] == 'Polarity':
                    if param['@value'] == 'Negative':
                        Polarity = -1.0
            #Error if no RT is found
            if RT < 0:
                raise ValueError('Error during XML parsing: No RT found for spectrum %s.' % timepoint['@id'])
            #Only store RTs after the threshold (i.e. skip any timepoints before the min_rt)
            elif RT < min_rt:
                continue
            else:
                pass

            # Read mz data - list of m/z's observed at this timepoint
            mz_struct = timepoint['mzArrayBinary']['data']
            if not '#text' in mz_struct:
                continue
            mz_string = base64.decodebytes(mz_struct['#text'].encode('ascii'))
            mz_format = unpack_format(mz_struct)
            mz_data = struct.unpack(mz_format, mz_string)

            # Read intensity data - list of intensities for each m/z observed at this timepoint
            ic_struct = timepoint['intenArrayBinary']['data']
            ic_string = base64.decodebytes(ic_struct['#text'].encode('ascii'))
            ic_data = struct.unpack(unpack_format(ic_struct), ic_string)

            # Add m/z and intensity to specturm
            for ion, intensity in zip(mz_data, ic_data):
                if intensity < min_int:
                    continue
                ion *= Polarity
                spectra.append((RT, ion, intensity))
        return spectra
    
    def scan_filter(self, rtwindow=0.1, mzwindow=300):
        """Removes all peaks under the median peak value observed in the window.
        """
        new_df = self._df.copy()
        scaler = 1 / rtwindow
        start_time_tenths = int(new_df['rt'].min()*scaler)
        end_time_tenths = int(new_df['rt'].max()*scaler)+1
        for time in range(start_time_tenths, end_time_tenths):
            startrt = (time*rtwindow) - (rtwindow/2)
            endrt = (time*rtwindow) + (rtwindow/2)
            for mz in range(500,3300,mzwindow):
                startmz = mz - (mzwindow/2)
                endmz = mz + (mzwindow/2)
                med_val = self._df[(self._df['rt'] >= startrt) 
                                  & (self._df['rt'] <= endrt)
                                  & (self._df['mz'] >= startmz)
                                  & (self._df['mz'] <= endmz)]['int'].median()
                mad_val = self._df[(self._df['rt'] >= startrt) 
                                  & (self._df['rt'] <= endrt)
                                  & (self._df['mz'] >= startmz)
                                  & (self._df['mz'] <= endmz)]['int'].mad()
                cutoff = med_val + mad_val
                remove = ((new_df['rt'] >= startrt) 
                          & (new_df['rt'] <= endrt) 
                          & (new_df['mz'] >= startmz)
                          & (new_df['mz'] <= endmz)
                          & (new_df['int'] <= cutoff))
                new_df = new_df[~remove]
        self._df = new_df
    
    def tic(self, plot=False):
        """extract a total ion chromatogram (sum of all ion intensities at each timepoint) and return the resulting
        dataframe in long form with retention time and intensity. Optionally plot the result using matplotlib.
        """
        #Group by retention time with sum of all data per retention time
        msdf = self._df.groupby(['rt']).sum().reset_index()
        if plot:
            sns.lineplot(data = msdf, x='rt', y='int')
        return msdf
    
    def eic(self, mz, dmz, rt_range=[0,10], plot=False):
        """extract an extracted ion chromatogram (a single m/z from each timepoint) and return the resulting
        dataframe in long form with retention time and intensity. Parameters are mass to be extracted and delta
        mass windiw to use, all ions with m/z of mass+/-dm will have their intensities summed and m/z's averaged.
        Optional parameters to plot the EIC (plot=True), restrict the rention_time window returned
        (rt_range=(start time, end time)), and return the integration of the peak (integrate=True). Function
        returns a tuple of the dataframe, and the integration (or None for integrate=False).
        """
        start_mz = mz - dmz
        end_mz = mz + dmz
        msdf = self._df[(self._df['mz'] >= start_mz) & (self._df['mz'] <= end_mz)]
        msdf = msdf[(msdf['rt'] >= rt_range[0]) & (msdf['rt'] <= rt_range[1])]
        if plot:
            sns.lineplot(data = msdf, x='rt', y='int')
        return msdf
    
    def ecc(self, mass, dm, mz_range=[500,3500], rt_range=[0,10]):
        """function takes a mass, and delta mass. Extracts EICs for every possible charge state and concatenates
        into a long form dataframe, with a 'charge', 'eic_mz', and 'eic_dmz' columns added to distinguish charge
        states.
        """
        ########### GRAB ALL THE EICS AND CONCAT INTO DATAFRAME ########
        msdfs = []
        for charge, mz, dmz in charge_list(mass, dm, mz_range=mz_range):
            eicdf = self.eic(mz, dmz, rt_range = rt_range, plot=False)
            eicdf['charge'] = charge
            eicdf['eic_mz'] = mz
            eicdf['eic_dmz'] = dmz
            msdfs.append(eicdf)

        #dataframe of all EIC masses/rts/intensities/charges
        msdf = pd.concat(msdfs, ignore_index=True)
        msdf = msdf.groupby(['mz', 'rt', 'charge'], group_keys=False).agg({'eic_mz':'first',
                                                                           'eic_dmz':'first',
                                                                           'int':'sum'}).reset_index()
        return msdf
    
    def spectra(self, start, end, plot=False, mz_range=[0,10000], biggest=None):
        msdf = self._df[(self._df['rt'] >= start) & 
                        (self._df['rt'] <= end)].groupby(['mz'], group_keys=False).sum().reset_index()
        msdf = msdf[(msdf['mz'] >= mz_range[0]) & (msdf['mz'] <= mz_range[1])]
        if biggest:
            msdf = msdf.nlargest(biggest, 'int')
        if plot:
            plt.bar(msdf['mz'], msdf['int'], width=0.1)
        return msdf
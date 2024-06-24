import pandas as pd


def generate_actions(reference_power, ders):
    # takes DataFrame of current DER statuses, determines on/off controls at current time
    # If reference power is NaN, return empty controls (OCHRE runs thermostat control instead)
    if pd.isna(reference_power):
        return {}, {}

    # Calculate SOC for each DER and sort by SOC
    e = ders['EBM Energy (kWh)']
    e_min = ders['EBM Min Energy (kWh)']
    e_max = ders['EBM Max Energy (kWh)']
    ders['SOC (-)'] = (e - e_min) / (e_max - e_min)
    ders = ders.sort_values('SOC (-)')

    # Turn on DERs with lowest SOC, up to the reference power
    power_sum = ders['EBM Max Power (kW)'].cumsum()
    flex_max_id = (power_sum - reference_power).abs().idxmin()  # get DER index closest to reference power
    ders['On'] = False
    ders.loc[:flex_max_id, 'On'] = True

    # Force DERs on or off if SOC is out of bounds
    ders['Forced On'] = ders['SOC (-)'] <= 0
    ders['Forced Off'] = ders['SOC (-)'] >= 1
    ders['On'] = ders['Forced On'] | (ders['On'] & ~ders['Forced Off'])
    ders['Flex'] = ~ders['Forced On'] & ~ders['Forced Off']

    # update results (used in co-simulation)
    totals = ders.sum()
    n_on = totals['On']
    n_forced_on = totals['Forced On']
    n_forced_off = totals['Forced Off']
    n_flex_on = n_on - n_forced_on
    n_flex = len(ders) - n_forced_on - n_forced_off

    results = {
        'DERs Forced On': n_forced_on,
        'DERs Forced Off': n_forced_off,
        'Flex DERs': n_flex,
        'Total DERs On': n_on,
        'Flex DERs On': n_flex_on,
        'Flex DERs Ratio On (-)': n_flex_on / n_flex if n_flex else 0,
        'Average SOC (-)': ders['SOC (-)'].mean(),
        'Average SOC, Clipped (-)': ders['SOC (-)'].clip(lower=0, upper=1).mean(),
        'Estimated Power (kW)': ders.loc[ders['On'], 'EBM Max Power (kW)'].sum(),
        'Reference Power (kW)': reference_power,
    }

    return ders['On'].to_dict(), results

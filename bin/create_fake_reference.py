import os
import sys
import numpy as np
import pandas as pd
import datetime as dt
from bin_hpc.create_Pbase_scenarios import create_wh_baseline_aggregate, create_hvac_baseline_aggregate


def make_scheduled_reference(e_min, e_max):
    start_time = e_min.index[0]
    time_res = e_min.index[1] - e_min.index[0]

    # specify time and SOC of fleet
    df_activations = pd.DataFrame({
        'Hour': [0, 1, 3.5, 4, 5, 8,
                9, 9.5, 13, 13.5, 15, 16, 16.5,
                20.5, 22, 22.25, 22.5, 23.5],
        'SOC': [0.5, 0.8, 0.4, 0.4, 1, 0.2,
                0.2, 0.7, 0, 0, 1, 0.5, 0.5,
                0.1, 0.5, 0.5, 0.9, 0.5],
    })

    # convert to time
    df_activations.index = [(start_time + dt.timedelta(hours=h)).time() for h in df_activations['Hour']]
    print(df_activations.head())

    # merge 
    df = pd.DataFrame({'zero': 0}, index=e_min.index)
    df['Time of Day'] = df.index.time
    df = df.join(df_activations['SOC'], on='Time of Day')
    print(df.head())

    # interpolate
    df = df.interpolate()

    # convert SOC to energy, and energy to power
    df['Energy'] = e_min * (1 - df['SOC']) + e_max * df['SOC']  # in kWh
    print(df.head())
    activations = df['Energy'].diff() / time_res.total_seconds() * 3600  # in kW

    return activations
    
    
# def make_random_reference_old(e_capacity, p_max, soc_init, magnitude=0.2):
#     # hour_interval = 4
#     # pct_increase = np.array([magnitude if n % (2 * hour_interval) < hour_interval else -magnitude for n in range(len(baseline))])
#     # reference = baseline * (1 + pct_increase)
    
#     # Use hour interval for consistent duration for up/down flexibility times
#     # up_times = (baseline.index.hour // hour_interval) % 2 == 1

#     # Use random duration times (always at least 1 hour)
#     # by_hour = baseline_power.resample(dt.timedelta(hours=1)).ffill()
#     # up_times = pd.Series(np.random.choice([True, False], len(by_hour)), index=by_hour.index)
#     # up_times = up_times.reindex(baseline_power.index).ffill()

#     # add random activation durations using time-inhomogenous markov chain
#     start_time = e_capacity.index[0]
#     end_time = e_capacity.index[-1]
#     time_res = e_capacity.index[1] - start_time
#     time_res_states = dt.timedelta(minutes=15)
#     n_states = 3  # activation up, down and off
#     # create distribution of activation duration, 20% at 15min, 15% at 30min, ... 5% at 4hr
#     duration_prop = [0.20, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0, 0.05, 0, 0.05, 0, 0, 0, 0.05]
#     n_periods = (end_time - start_time) // time_res_states
#     durations = np.random.choice(range(1, len(duration_prop) + 1), size=n_periods, p=duration_prop)
#     start_times = np.insert(durations, 0, 0).cumsum()  # cumulative sum of durations, start at time 0
#     start_times = start_time + start_times * time_res_states  # convert to datetime
#     state_deltas = np.random.choice(range(1, n_states), size=n_periods)
#     states = (np.insert(state_deltas, 0, 0).cumsum() % n_states) - 1  # convert states to [-1, 0, 1]
#     delta = pd.Series(states, index=start_times)
#     delta = delta.resample(time_res).loc[:end_time]
#     assert delta.index == baseline_power.index

#     # Option 1:
#     reference = baseline_power.copy()
#     reference.loc[up_times] *= 1 + magnitude
#     reference.loc[~up_times] *= 1 - magnitude

#     # Option 2:
#     # add or subtract fraction of average power to reference. Force non-negative values
#     # power_change = baseline.mean() * magnitude
#     # delta = pd.Series(-power_change, index=baseline.index)
#     # delta.loc[up_times] = power_change
#     # reference = baseline + delta.clip(lower=-baseline, upper=baseline)
    
#     return reference

def run_checks(forecast, reference):
    # enforce non-negativity - cumulative sum must be monotonically increasing
    if reference.min() < 0:
        cum_sum = reference.cumsum()
        cum_max = cum_sum.cummax()
        reference = cum_max.diff()
    assert reference.min() >= 0
 
    # enforce daily energy conservation
    daily_ratios = forecast.groupby(forecast.index.date).sum() / reference.groupby(reference.index.date).sum()
    assert daily_ratios.min() > 0.8
    assert daily_ratios.max() < 1.2
    daily_ratios = daily_ratios.reindex(forecast.index).ffill()
    reference *= daily_ratios
    assert (reference.groupby(reference.index.date).sum() - forecast.groupby(forecast.index.date).sum()).abs().max() < 0.1

    return reference

def make_random_reference(baseline):
    
    x=np.arange(0, 5.76, 0.02)
    y=list()
    directions = list()

    x1 = 1.5
    x2 = 3
    x3 = 4.5
    f1 = np.random.uniform(1, 3)
    f2 = np.random.uniform(1, 3)
    f3 = np.random.uniform(1, 3)
    f4 = np.random.uniform(1, 3)
    upper_tres = 0.5
    lower_tres = 0.5

    for i in range(0,len(x)):

        if x[i]< x1:
            c = np.cos(2 * np.pi * f1 * x[i])
        elif x[i]< x2:
            phi1 = 2 * np.pi * f1 * x1 - 2 * np.pi * f2 * x1
            c = np.cos(2 * np.pi * f2 * x[i] + phi1)
        elif x[i] < x3:
            phi2 = 2 * np.pi * f2 * x2 - 2 * np.pi * f3 * x2 + phi1
            c = np.cos(2 * np.pi * f3 * x[i] + phi2) 
        else:
            phi3 = 2 * np.pi * f3 * x3 - 2 * np.pi * f4 * x3 + phi2
            c = np.cos(2 * np.pi * f4 * x[i] + phi3) 
        y.append(c)
        if c >= upper_tres:
            directions.append(1)
        elif c <= lower_tres:
            directions.append(-1)
        else:
            directions.append(0) 
    directions = np.tile(directions, 7)
    activations = - np.multiply(baseline, np.multiply(np.random.beta(0.5, 0.5, \
        size=len(baseline)), directions))

    return activations


def main(scenario_name):
    
    YEAR = 2018
    if os.environ.get('NREL_CLUSTER') == 'kestrel':
        cosim_path = '/projects/performnrel/cosim_scenarios'
    SCENARIOS_FILE = os.path.join(cosim_path, 'scenarios.csv')
    REFERENCE_FILE = os.path.join(cosim_path, scenario_name, 'system_reference.csv')
    BASELINE_FILE = os.path.join(cosim_path, scenario_name, 'Baseline_results_5min.csv')
    # META_FILE = os.path.join(cosim_path, scenario_name, 'compiled', 'all_ochre_inputs.csv')

    # meta_data = pd.read_csv(META_FILE, index_col=0, parse_dates=True)
    # total_rated_power = np.sum(meta_data["Equipment.Water Heating.Capacity (W)"])/1000

    df_scenarios = pd.read_csv(SCENARIOS_FILE, index_col='Scenario Name')
    if scenario_name not in df_scenarios.index:
        raise Exception('Co-sim error, bad scenario name: ' + scenario_name)
    scenario_params = df_scenarios.loc[scenario_name]
    scenario_params = scenario_params[scenario_params.notna()].to_dict()
    
    start_month = scenario_params.get('Start Month', 1)
    start_day = scenario_params.get('Start Day', 1)
    start_time = dt.datetime(YEAR, start_month, start_day, 0, 0) 
    operating_day = start_time.replace(second=0)

    # Load baseline power file
    df = pd.read_csv(BASELINE_FILE, index_col='Time', parse_dates=True)
    der_type = 'HVAC Cooling' if 'hvac' in BASELINE_FILE else 'Water Heating'
    # baseline_power = df[f'{der_type} Electric Power (kW)']
    e_min = df[f'{der_type} EBM Min Energy (kWh)']
    e_max = df[f'{der_type} EBM Max Energy (kWh)']
    # p_max = df[f'{der_type} EBM Max Power (kW)']

    # Create forecast (reloads baseline file)
    scenario_path = os.path.join(cosim_path, scenario_name)
    if der_type == 'HVAC Cooling':
        p_base = create_hvac_baseline_aggregate(scenario_path)
        forecast = p_base.quantile(0.5, axis=1)
    else:
        p_base = create_wh_baseline_aggregate(scenario_path)
        forecast = p_base.mean(axis=1)
    time_res = df.index[1] - df.index[0]
    forecast = forecast.loc[operating_day: operating_day + dt.timedelta(days=7) - time_res]

    # create random reference
    activations = make_random_reference(forecast)
    reference = forecast.to_frame(name='kW') + activations.to_frame(name='kW')
    
    # create scheduled reference
    # activations = make_scheduled_reference(e_min, e_max)
    # reference = forecast.to_frame(name='kW') + activations.to_frame(name='kW')
    # reference = np.clip(reference, 0, total_rated_power)
    # reference = run_checks(forecast, reference)

    # Save reference to reference file
    print(f'Saving reference to {REFERENCE_FILE}')
    reference.to_csv(REFERENCE_FILE)


if __name__ == '__main__':

    scenario_name = sys.argv[1]
    if len(sys.argv) >= 3:
        reference_file = sys.argv[2]
    else:
        # use path from baseline file, called reference.csv
        reference_file = 'system_reference.csv'

    main(scenario_name)

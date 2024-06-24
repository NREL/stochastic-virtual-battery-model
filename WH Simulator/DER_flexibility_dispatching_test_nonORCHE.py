import os
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .Aggregator import WHAggregator
from .WaterHeaterModels import WaterHeater


def create_fleet(withdrawrate):

    np.random.seed(50)
    Parameters = pd.DataFrame(np.empty, index=range(Num_WH),
                              columns=['SetpointT', 'InitialT', 'AmbientT', 'InletT', 'UpperT', 'LowerT',
                                       'Efficiency', 'Volumn', 'RatePower'])
    Parameters['SetpointT'] = np.random.uniform(50, 50, size=Num_WH)
    Parameters['InitialT'] = np.random.uniform(48, 52, size=Num_WH) 
    Parameters['AmbientT'] = np.random.uniform(15, 15, size=Num_WH)
    Parameters['InletT'] = np.random.uniform(5.6, 5.6, size=Num_WH)
    Parameters['UpperT'] = Parameters['SetpointT'] + np.ones(Num_WH) * 2
    Parameters['LowerT'] = Parameters['SetpointT'] - np.ones(Num_WH) * 2
    Parameters['Efficiency'] = np.random.uniform(0.99, 1, size=Num_WH)
    Parameters['Volumn'] = np.random.uniform(190, 190, size=Num_WH)
    Parameters['RatePower'] = np.random.uniform(3.8, 3.8, size=Num_WH)

    fleet = {}
    IDs = []

    for num in range(Num_WH):
        dict_WH = {
            "SpecificHeatCapacity": 4.186,
            "InitialT": Parameters['InitialT'][num],
            "AmbientT": Parameters['AmbientT'][num],
            "InletT": Parameters['InletT'][num],
            "UpperT": Parameters['UpperT'][num],
            "LowerT": Parameters['LowerT'][num],
            "Efficiency": Parameters['Efficiency'][num],
            "WaterDensity": 0.99,
            "Volumn": Parameters['Volumn'][num],
            "WithdrawRate": withdrawrate[num],
            "TimeConstant": 150 * 60 * 60,
            "RatePower": Parameters['RatePower'][num],
            "ControlInterval": 20,
            "ID": num
        }
        WH = WaterHeater(dict_WH)
        fleet[num] = WH
        IDs.append(num)

    return fleet


def run_with_agg(fleet, aggregator, reference, duration):

    aggregator.receive_signal(reference)

    for WH in fleet.values():
        WH.reset(duration)

    for k in range(duration):

        SoCs = pd.DataFrame(np.empty(len(fleet.keys())), index=fleet.keys(), columns=['soc'])
        States = pd.DataFrame(np.empty(len(fleet.keys())), index=fleet.keys(), columns=['state'])
        for num in fleet.keys():
            SoCs.soc.loc[num] = fleet[num].sending_soc()
            States.state.loc[num] = fleet[num].sending_state()

        Actions = aggregator.generate_actions_sorting(k, SoCs, 3.8)
        # Actions = aggregator.generate_actions(k, States, 3.8)

        # call water heater models to implement the actions
        for WH in fleet.values():
            WH.singlestep_control(Actions.action[WH.ID], k)

    # Collecting the actual power consumption
    powers = []
    temperatures = []
    socs = []
    for num in range(Num_WH):
        powers.append([z * fleet[num].Prate for z in fleet[num].Z])
        temperatures.append(fleet[num].T)
        socs.append((fleet[num].T-fleet[num].Tlower)*100/(fleet[num].Tupper-fleet[num].Tlower))
    powers = np.array(powers)
    temperatures = np.array(temperatures)
    socs = np.array(socs)

    return powers, temperatures, socs


if __name__ == "__main__":

    week = "week2018_01_22"
    day = 1

    main_path = os.path.dirname(__file__)

    probability_file = os.path.join(main_path, 'Inputs', 'Probability_38_C1.csv')
    DAoffers_file = os.path.join(main_path, 'Outputs', 'DAFlexibilityOffers.csv')
    RToffers_file = os.path.join(main_path, 'Outputs', 'RTFlexibilityOffers.csv')
    DA_solution_file = os.path.join(main_path, 'Inputs', 'Market_DASolution' + week + '_bugfixed.csv')
    RT_solution_15min_file = os.path.join(main_path, 'Inputs', 'Market_RTSolution_15m' + week + '_bugfixed.csv')
    RT_solution_5min_file = os.path.join(main_path, 'Inputs', 'Market_RTSolution_5m' + week + '_bugfixed.csv')
    
    # timing parameters
    sim_resolution = dt.timedelta(seconds=20)
    RT_15_resolution = dt.timedelta(minutes=15)
    RT_5_resolution = dt.timedelta(minutes=5)
    DA_resolution = dt.timedelta(hours=1)

    start_time = dt.datetime(2013, 8, 12)
    duration = dt.timedelta(days=1)
    end_time = start_time + duration
    timestamps_20s = pd.date_range(start_time, end_time, freq=sim_resolution, closed='left')
    timestamps_15m = pd.date_range(start_time, end_time, freq=RT_15_resolution, closed='left')
    timestamps_5m = pd.date_range(start_time, end_time, freq=RT_5_resolution, closed='left')
    timestamps_1h = pd.date_range(start_time, end_time, freq=DA_resolution, closed='left')
    
    # water heater properties
    Num_WH = 222
    WD_Mag = 2
    duration_20s = 24 * 60 * 3
    rated_power = 3.8

    # importing real time actual water withdraw profile
    probabilities = pd.read_csv(probability_file, index_col=0, parse_dates=True)['Probability']
    probabilities = probabilities.resample(sim_resolution).pad().reindex(timestamps_20s)

    np.random.seed(100)
    withdraw_rates = []
    for idx in probabilities.index:
        p = probabilities.loc[idx]
        withdraw_rates.append(np.random.choice(np.arange(0, 2), p=[1 - p, p], size=Num_WH) * WD_Mag)
    withdraw_rates = np.array(withdraw_rates).T
    
    # create fleet and aggregator
    fleet = create_fleet(withdraw_rates)
    aggregator = WHAggregator(fleet.keys(), duration_20s)

    # loading pricing data
    DA_solution = pd.read_csv(DA_solution_file, parse_dates=True, \
        names=['flex up', 'flex down', 'prob_up', 'prob_down', 'strike up', 'strike down']).loc[day*24:(day+1)*24-1]
    DA_solution = DA_solution.set_index(timestamps_1h)
    RT_solution_15m = pd.read_csv(RT_solution_15min_file, parse_dates=True, \
        names=['15-min price']).loc[day*24*4:(day+1)*24*4-1]
    RT_solution_15m = RT_solution_15m.set_index(timestamps_15m)
    RT_solution_5m = pd.read_csv(RT_solution_5min_file, parse_dates=True, \
        names=['5-min price']).loc[day*24*12:(day+1)*24*12-1]
    RT_solution_5m = RT_solution_5m.set_index(timestamps_5m)

    DA_Flexup_Price = (DA_solution['flex up'] - \
        DA_solution['prob_up'] * DA_solution['strike up']).values
    DA_Flexdown_Price = (DA_solution['flex down'] + \
        DA_solution['prob_down'] * DA_solution['strike down']).values
    Strikeup_Price = DA_solution['strike up'].reindex(timestamps_15m).ffill().values
    Strikedown_Price = DA_solution['strike down'].reindex(timestamps_15m).ffill().values
    RTPD_Price = RT_solution_15m['15-min price'].values
    RT_Price = RT_solution_5m['5-min price'].resample('5T').mean().values

    # Generate a reference signal
    DAoffers = pd.read_csv(DAoffers_file, index_col=0).set_index(timestamps_1h)
    RToffers = pd.read_csv(RToffers_file, index_col=0).set_index(timestamps_15m)
    
    activation_up = pd.DataFrame(np.clip(np.sign(RT_solution_15m['15-min price'] 
        - DA_solution['strike up'].reindex(timestamps_15m).ffill()), 0, 1),  
        index=timestamps_15m, columns=['Flexup'])
    activation_down = pd.DataFrame(np.clip(np.sign( 
        DA_solution['strike down'].reindex(timestamps_15m).ffill() 
        - RT_solution_15m['15-min price']), 0, 1), 
        index=timestamps_15m, columns=['Flexdown'])
    
    reference_power = DAoffers['Pbase'].reindex(timestamps_15m).ffill() - \
        np.multiply(activation_up['Flexup'], RToffers['Pup']) + \
        np.multiply(activation_down['Flexdown'], RToffers['Pdown'])

    reference_power = (reference_power.reindex(timestamps_20s).ffill()).to_frame(name='Power')

    # run case with aggregator
    agg_power, agg_temp, agg_socs = run_with_agg(fleet, aggregator, reference_power, duration_20s)
    agg_power = pd.DataFrame(agg_power.sum(axis=0), index=reference_power.index, columns=['Power'])
    agg_temp = pd.DataFrame(agg_temp.T, index=reference_power.index, columns=range(Num_WH))

    agg_HSOC = []
    for WH in fleet.values():
        agg_HSOC.append((agg_temp[WH.ID] - WH.Tlower) * 100/(WH.Tupper - WH.Tlower))
    ave_agg_HSOC = np.mean(np.array(agg_HSOC), axis=0)
    plt.plot(ave_agg_HSOC)
    plt.show()

    # Plot the total power consumption and save to file
    power_data = pd.DataFrame({
        'Baseline Power (kW)': DAoffers['Pbase'].reindex(timestamps_20s).ffill().values,
        'Reference Power (kW)': reference_power.Power,
        'Aggregator Control Power (kW)': agg_power.Power,
    }, index=timestamps_20s).resample('15min').mean()
    ax = power_data.plot()
    ax.legend(loc="upper right")
    plt.show()

    # Plot the aggregated control temperatures
    ax = agg_temp.plot()
    plt.show()

    plt.plot(timestamps_20s, aggregator.Num_mustON, label='muston')
    plt.plot(timestamps_20s, np.ones(len(timestamps_20s))*Num_WH - aggregator.Num_mustOFF, label='mustoff')
    plt.plot(timestamps_20s, reference_power.Power/WD_Mag, label='target on')
    plt.legend()
    plt.show()

    # Calculating the settlements
    DAsettlement = np.sum(
        np.multiply(DAoffers['Pup'], DA_Flexup_Price) + 
        np.multiply(DAoffers['Pdown'], DA_Flexdown_Price))
    RTsettlement = np.sum(
        np.multiply(activation_up['Flexup'], \
        np.multiply(RTPD_Price, DAoffers['Pup'].reindex(timestamps_15m).ffill()) -
        np.multiply(RTPD_Price-Strikeup_Price, RToffers['Pup']))) * 15/60 + \
        np.sum(
        np.multiply(activation_down['Flexdown'], \
        np.multiply(RTPD_Price, DAoffers['Pdown'].reindex(timestamps_15m).ffill()) -
        np.multiply(RTPD_Price-Strikedown_Price, RToffers['Pdown']))) * 15/60 
    Deviation_Flexup = np.clip(np.multiply(
        activation_up['Flexup'], 
        np.abs(RToffers['Pup'] - (RToffers['Pbase'] - agg_power.resample('15T').mean().Power)) - 
        RToffers['Pup'] * 0.05), 0, 1000)
    Deviation_Flexdown = np.clip(np.multiply(
        activation_down['Flexdown'], 
        np.abs(RToffers['Pdown'] - (agg_power.resample('15T').mean().Power - RToffers['Pbase'])) - 
        RToffers['Pdown'] * 0.05), 0, 1000)
    RTpenalty = np.sum(np.multiply(RT_Price, Deviation_Flexup)) * 15/60 + \
                np.sum(np.multiply(RT_Price, Deviation_Flexdown)) * 15/60 

    print("DA settlement ", DAsettlement)
    print("RT settlement ", RTsettlement)
    print("RT penalty ", RTpenalty)
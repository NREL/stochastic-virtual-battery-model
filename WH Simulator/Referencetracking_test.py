from WaterHeaterModels import WaterHeater
from Aggregator import WHAggregator
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt


def reference_signal_alter_baseline(baseline, magnitude, interval, direction):
    plus = np.array([int(n % (2*interval) < interval) for n in range(len(baseline.Power))])
    minus = np.array([int(n % (2*interval) >= interval) for n in range(len(baseline.Power))])
    if direction == "+":
        reference = baseline.Power.values \
                    + np.multiply(magnitude * plus, baseline.Power.values) \
                    - np.multiply(magnitude * minus, baseline.Power.values)
        reference = pd.DataFrame(reference, index=baseline.index, columns=['Power'])
        sign = plus - minus
    else:
        reference = baseline.Power.values \
                    - np.multiply(magnitude * plus, baseline.Power) \
                    + np.multiply(magnitude * minus, baseline.Power)
        reference = pd.DataFrame(reference, index=baseline.index, columns=['Power'])
        sign = minus - plus
    return reference, sign

if __name__ == "__main__":

    duration_1h = 24
    duration_5min = duration_1h * 12 # 5min interval duration
    duration_20s = duration_1h * 60 * 3 # 20s interval duration
    daterange_5min = pd.date_range("00:00:00", "23:55:00", freq="5min") # 5min interval daterange
    daterange_20s = pd.date_range("00:00:00", "23:59:40", freq="20s") # 20s interval daterange
    daterange_1h = pd.date_range("00:00:00", "23:00:00", freq="1h") # 1h interval daterange
    Num_WH = 100
    WD_Mag = 2.5
    Power_rate = 4.8

    np.random.seed(50)
    # Loading the probability model and creating the correspondent water withdrawal profiles for individual water heaters
    Probability = pd.read_csv(os.getcwd() + "/Inputs/Probability.csv", index_col=0)

    withdrawrate = []
    for t in range(duration_5min):
        p = Probability.Probability[t]
        withdraw_5min = np.random.choice(np.arange(0, 2), p=[1 - p, p], size=Num_WH) * WD_Mag
        for n in range(5*3):
            withdrawrate.append(withdraw_5min)
    withdrawrate = np.array(withdrawrate).T

    # Creating the water heater fleet based on randomized device parameters
    Parameters = pd.DataFrame(np.empty, index=range(Num_WH),
                              columns=['InitialT', 'AmbientT', 'InletT', 'UpperT', 'LowerT',
                                       'Efficiency', 'Volumn', 'WithdrawRate', 'RatePower'])
    Parameters['InitialT'] = np.random.uniform(49, 49, size=Num_WH)
    Parameters['AmbientT'] = np.random.uniform(15, 18, size=Num_WH)
    Parameters['InletT'] = np.random.uniform(5.6, 5.6, size=Num_WH)
    Parameters['UpperT'] = np.random.randint(50, 52, size=Num_WH)
    Parameters['LowerT'] = Parameters['UpperT'].values - np.ones(len(Parameters['UpperT'])) * 3
    Parameters['Efficiency'] = np.random.uniform(0.99, 1, size=Num_WH)
    Parameters['Volumn'] = np.random.uniform(227, 227, size=Num_WH)
    Parameters['RatePower'] = np.random.uniform(Power_rate, Power_rate, size=Num_WH)

    WHfleet = {}
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
        WH.TS_baseline_run(duration_20s)
        WHfleet[num] = WH
        IDs.append(num)

    # Collecting the baseline power consumption
    Baseline_P = []
    for num in range(Num_WH):
        Baseline_P.append([z * WHfleet[num].Prate for z in WHfleet[num].Z])
    Baseline_P = pd.DataFrame(np.sum(np.array(Baseline_P), axis=0), index=daterange_20s, columns=['Power'])

    # Creating an aggregator instance
    WHAggregator = WHAggregator(IDs, duration_20s)

    # Generaing the reference signal
    Reference_P, Sign = reference_signal_alter_baseline(Baseline_P, 0.4, 3*60*8, "-")

    # Aggregator receives the reference signal from the market
    WHAggregator.receive_signal(Reference_P)

    # Reseting the water heater time-series records
    for WH in WHfleet.values():
        WH.reset(duration_20s)

    # Closed-loop control
    for k in range(duration_20s):

        # call water heater models to collect current states
        # States = pd.DataFrame(np.empty(len(IDs)), index=IDs, columns=['state'])
        # for num in WHfleet.keys():
        #     States.state.loc[num] = WHfleet[num].sending_state()
        SoCs = pd.DataFrame(np.empty(len(IDs)), index=IDs, columns=['soc'])
        for num in WHfleet.keys():
            SoCs.soc.loc[num] = WHfleet[num].sending_soc()

        # call the aggregator to generate actions based on the states
        # Actions = WHAggregator.generate_actions(k, States, Power_rate)
        Actions = WHAggregator.generate_actions_sorting(k, SoCs, Power_rate)

        # call water heater models to implement the actions
        for WH in WHfleet.values():
            WH.singlestep_control(Actions.action[WH.ID], k)

    # Collecting the actual power consumption
    Actual_P = []
    Actual_HSoC = []

    for num in range(Num_WH):
        Actual_P.append([z * WHfleet[num].Prate for z in WHfleet[num].Z])
        Actual_HSoC.append((WHfleet[num].T-WHfleet[num].Tlower)*100/(WHfleet[num].Tupper-WHfleet[num].Tlower))
    Actual_P = pd.DataFrame(np.sum(np.array(Actual_P), axis=0), index=daterange_20s, columns=['Power'])
    Actual_HSoC = pd.DataFrame(np.array(Actual_HSoC).T, index=daterange_20s, columns=IDs)

    # Plotting the power consumptions
    sns.set(rc={'figure.figsize': (10, 6)})
    sns.lineplot(x=range(duration_1h), y=Baseline_P.resample('60T').mean().Power, drawstyle='steps-pre', label='Baseline consumption (kW)')
    sns.lineplot(x=range(duration_1h), y=Actual_P.resample('60T').mean().Power, drawstyle='steps-pre', label='Actual consumption (kW)')
    sns.lineplot(x=range(duration_1h), y=Reference_P.resample('60T').mean().Power, drawstyle='steps-pre', label='Reference consumption (kW)')
    plt.legend(loc="upper right")
    plt.xlabel("Control interval (1hour)")
    plt.show()

    # Plotting the temperatures
    for ID in IDs:
        sns.lineplot(x=range(duration_1h), y=Actual_HSoC.resample('60T').mean()[ID], linewidth=1)
    plt.xlabel("Control interval (1hour)")
    plt.ylim(-50, 150)
    plt.show()

    total_deviation = np.sum(np.abs(Actual_P.resample('60T').mean().Power - Reference_P.resample('60T').mean().Power))
    print(total_deviation)

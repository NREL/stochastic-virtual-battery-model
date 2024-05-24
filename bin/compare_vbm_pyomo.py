import pandas as pd
import numpy as np
import os
import datetime as dt
import matplotlib.pyplot as plt
from pyomo.environ import *
from pyomo.opt import SolverFactory

import pyutilib.subprocess.GlobalData
pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False

main_path = os.path.join(os.path.dirname(__file__), '..', '..')
plt.rc('font', size=13)


def compare_actual_and_expected_energy_dynamics(start_time, end_time, scenario_name):

    timestamps_5m = pd.date_range(start_time, end_time, freq=dt.timedelta(minutes=5),
                                  inclusive='left')
    month = timestamps_5m[0].month
    day = timestamps_5m[0].day

    # loading the actual energy evolution
    Eactual = pd.read_parquet(os.path.join(main_path, "cosim_scenarios", scenario_name, "compiled",
                                            "WH_results_5min_summed.parquet"), engine='pyarrow')\
                                            .loc[timestamps_5m]['Water Heating EBM Energy (kWh)'].values

    # calculating the expected energy evolution when reference consumption and actual consumption are followed
    Penergy_5m = pd.read_csv(os.path.join(main_path, "cosim_scenarios", scenario_name,
                                          "baseline_power_scenarios.csv"), index_col=0,
                             parse_dates=True).loc[timestamps_5m].mean(axis=1).values
    Pbaseline_5m = pd.read_csv(os.path.join(main_path, "cosim_scenarios", scenario_name, 
                                            'Baseline_results_5min.csv'), index_col=0, 
                             parse_dates=True)['Water Heating Electric Power (kW)']\
                             .loc[timestamps_5m].values
    Pactual_5m = pd.read_parquet(os.path.join(main_path, "cosim_scenarios", scenario_name, "compiled",
                                            "WH_results_5min_summed.parquet"), engine='pyarrow')\
                                            .loc[timestamps_5m]['Water Heating Electric Power (kW)'].values
    Preference_5m = pd.read_csv(os.path.join(main_path, "cosim_scenarios", scenario_name,
                                             'aggregator_reference_' + str(month) + '_' + str(day) + '.csv'), index_col=0,
                                parse_dates=True).loc[timestamps_5m].values
    vbm = pd.read_csv(os.path.join(main_path, "cosim_scenarios", scenario_name, \
        "Baseline_results_5min.csv"), index_col=0, parse_dates=True).loc[timestamps_5m]
    Emax = vbm["Water Heating EBM Max Energy (kWh)"].values
    Emin = vbm["Water Heating EBM Min Energy (kWh)"].values
    Pmax = vbm["Water Heating EBM Max Power (kW)"].values

    Pvbm_energy = []
    Pvbm_baseline = []
    Pmuston = []
    Pmax_mustoff = []
    Eactual = pd.read_parquet(os.path.join(main_path, "cosim_scenarios", scenario_name, "compiled",
                                            "WH_results_5min_summed.parquet"), engine='pyarrow')\
                                            .loc[timestamps_5m]['Water Heating EBM Energy (kWh)'].values
    Einit = Eactual[0]
    for t in range(len(timestamps_5m)):
        if Preference_5m[t] >= Penergy_5m[t]:
            Pmuston.append(max(-0.37 * Einit + 6694.73, -0.02 * Einit + 469.54))
            Pmax_mustoff.append(Pmax[t] - max(0.10 * Einit - 1793.66, \
                6.72 * Einit - 125964.57, 35.26 * Einit - 666089.45))
        else:
            Pmuston.append(max(-0.72 * Einit + 13028.17, -0.04 * Einit + 770.96))
            Pmax_mustoff.append(Pmax[t] - max(0.03 * Einit - 478.36, \
                4.60 * Einit -84431.28, 17.51 * Einit - 666089.45))

        if Einit + (1/12) * (Preference_5m[t] - Penergy_5m[t]) > Emax[t]:
            P = 12 * (Emax[t] - Einit) + Penergy_5m[t]
        elif Einit + (1/12) * (Preference_5m[t] - Penergy_5m[t]) < Emin[t]:
            P = 12 * (Emin[t] - Einit) + Penergy_5m[t]
        else: 
            P = Preference_5m[t]
        
        if P < Pmuston[t]:
            P = Pmuston[t]
        elif P > Pmax_mustoff[t]:
            P = Pmax_mustoff[t]
        Pvbm_energy.append(P)
        Einit = Einit + (1/12) * (P - Penergy_5m[t])
    
    Einit = Eactual[0]
    for t in range(len(timestamps_5m)):
        if Preference_5m[t] >= Pbaseline_5m[t]:
            Pmuston.append(max(-0.37 * Einit + 6694.73, -0.02 * Einit + 469.54))
            Pmax_mustoff.append(Pmax[t] - max(0.10 * Einit - 1793.66, \
                6.72 * Einit - 125964.57, 35.26 * Einit - 666089.45))
        else:
            Pmuston.append(max(-0.72 * Einit + 13028.17, -0.04 * Einit + 770.96))
            Pmax_mustoff.append(Pmax[t] - max(0.03 * Einit - 478.36, \
                4.60 * Einit -84431.28, 17.51 * Einit - 666089.45))

        if Einit + (1/12) * (Preference_5m[t] - Pbaseline_5m[t]) > Emax[t]:
            P = 12 * (Emax[t] - Einit) + Pbaseline_5m[t]
        elif Einit + (1/12) * (Preference_5m[t] - Pbaseline_5m[t]) < Emin[t]:
            P = 12 * (Emin[t] - Einit) + Pbaseline_5m[t]
        else: 
            P = Preference_5m[t]
        
        if P < Pmuston[t]:
            P = Pmuston[t]
        elif P > Pmax_mustoff[t]:
            P = Pmax_mustoff[t]
        Pvbm_baseline.append(P)
        Einit = Einit + (1/12) * (P - Pactual_5m[t])

    Einit_actual = Eactual[0]
    Einit_vbm_energy = Eactual[0]
    Einit_vbm_baseline = Eactual[0]
    Expected_E_vbm_actual = []
    Expected_E_vbm_energy = []
    Expected_E_vbm_baseline = []
    for t in range(len(timestamps_5m)):
        E_vbm_actual = Einit_actual + (1/12) * (Pactual_5m[t] - Pbaseline_5m[t])
        E_vbm_energy = Einit_vbm_energy + (1/12) * (Pvbm_energy[t] - Penergy_5m[t])
        E_vbm_baseline = Einit_vbm_baseline + (1/12) * (Pvbm_baseline[t] - Pbaseline_5m[t])
        Einit_actual = E_vbm_actual
        Einit_vbm_energy = E_vbm_energy
        Einit_vbm_baseline = E_vbm_baseline
        Expected_E_vbm_actual.append(E_vbm_actual)
        Expected_E_vbm_energy.append(E_vbm_energy)   
        Expected_E_vbm_baseline.append(E_vbm_baseline)   
       
    Eenergy_5m = pd.read_csv(os.path.join(main_path, "cosim_scenarios", scenario_name,
                                          "Baseline_results_5min.csv"), index_col=0,
                             parse_dates=True)['Water Heating EBM Energy (kWh)'].loc[timestamps_5m]
    plt.plot(timestamps_5m, Eactual, label='Actual Energy State (kWh)', color='blue')
    plt.plot(timestamps_5m, Expected_E_vbm_energy, label='Estimated Energy State (kWh) based on Pvbm_energy and expected Pbaseline', color='orange')
    # plt.fill_between(timestamps_5m, np.array(Expected_E_vbm_energy).flatten() + (Eenergy_5m.values - Eenergy_5m.values[0]), \
    #     np.array(Expected_E_vbm_energy).flatten(), label="Gap introduced by constant energy state assumption", alpha=0.5, color='orange')
    plt.plot(timestamps_5m, Expected_E_vbm_baseline, label='Estimated Energy State (kWh) based on Pvbm_baseline and actual Pbaseline', color='green')
    plt.fill_between(timestamps_5m, np.array(Expected_E_vbm_baseline).flatten() + (Eenergy_5m.values - Eenergy_5m.values[0]), \
        np.array(Expected_E_vbm_baseline).flatten(), label="Gap introduced by constant energy state assumption", alpha=0.5, color='green')
    # plt.plot(timestamps_5m, Expected_E_vbm_actual, label='Estimated Energy State (kWh) based on Pactual and actual Pbaseline', color='red')
    # plt.fill_between(timestamps_5m, np.array(Expected_E_vbm_actual).flatten() + (Eenergy_5m.values - Eenergy_5m.values[0]), \
    #     np.array(Expected_E_vbm_actual).flatten(), label="Gap introduced by constant energy state assumption", alpha=0.5, color='red')
    plt.plot(timestamps_5m, Emax, linestyle="--", label='Emax (kWh)', color='red')
    plt.plot(timestamps_5m, Emin, linestyle="--", label='Emin (kWh)', color='purple')
    plt.xlabel('Time')
    plt.ylabel('Energy (kWh)')
    plt.ylim(17000, 19300)
    plt.legend()
    plt.show()

def compare_actual_versus_estimated_power_consumption(start_time, end_time, scenario_name):

    timestamps_5m = pd.date_range(start_time, end_time, freq=dt.timedelta(minutes=5),
                                  inclusive='left')
    timestamps_15m = pd.date_range(start_time, end_time, freq=dt.timedelta(minutes=15),
                                  inclusive='left')
    month = timestamps_5m[0].month
    day = timestamps_5m[0].day
                              
    vbm = pd.read_csv(os.path.join(main_path, "cosim_scenarios", scenario_name, \
        "Baseline_results_5min.csv"), index_col=0, parse_dates=True).loc[timestamps_5m]
    Emax = vbm["Water Heating EBM Max Energy (kWh)"].values
    Emin = vbm["Water Heating EBM Min Energy (kWh)"].values
    Pmax = vbm["Water Heating EBM Max Power (kW)"].values

    Pactual_5m = pd.read_parquet(os.path.join(main_path, "cosim_scenarios", scenario_name, "compiled",
                                            "WH_results_5min_summed.parquet"), engine='pyarrow')\
                                            .loc[timestamps_5m]['Water Heating Electric Power (kW)'].values
    
    Penergy_5m = pd.read_csv(os.path.join(main_path, "cosim_scenarios", scenario_name,
                                          "baseline_power_scenarios.csv"), index_col=0,
                             parse_dates=True).loc[timestamps_5m].mean(axis=1).values
    Preference_5m = pd.read_csv(os.path.join(main_path, "cosim_scenarios", \
        scenario_name, 'aggregator_reference_' + str(month) + '_' + str(day) + '.csv'),
        index_col=0, parse_dates=True).loc[timestamps_5m].values
    
    Pvbm = []
    Pmuston = []
    Pmax_mustoff = []
    Eactual = pd.read_parquet(os.path.join(main_path, "cosim_scenarios", scenario_name, "compiled",
                                            "WH_results_5min_summed.parquet"), engine='pyarrow')\
                                            .loc[timestamps_5m]['Water Heating EBM Energy (kWh)'].values
    Einit = Eactual[0]
    for t in range(len(timestamps_5m)):
        if Preference_5m[t] >= Penergy_5m[t]:
            Pmuston.append(max(-0.37 * Einit + 6694.73, -0.02 * Einit + 469.54))
            Pmax_mustoff.append(Pmax[t] - max(0.10 * Einit - 1793.66, \
                6.72 * Einit - 125964.57, 35.26 * Einit - 666089.45))
        else:
            Pmuston.append(max(-0.72 * Einit + 13028.17, -0.04 * Einit + 770.96))
            Pmax_mustoff.append(Pmax[t] - max(0.03 * Einit - 478.36, \
                4.60 * Einit -84431.28, 17.51 * Einit - 666089.45))

        if Einit + (1/12) * (Preference_5m[t] - Penergy_5m[t]) > Emax[t]:
            P = 12 * (Emax[t] - Einit) + Penergy_5m[t]
        elif Einit + (1/12) * (Preference_5m[t] - Penergy_5m[t]) < Emin[t]:
            P = 12 * (Emin[t] - Einit) + Penergy_5m[t]
        else: 
            P = Preference_5m[t]
        
        if P < Pmuston[t]:
            P = Pmuston[t]
        elif P > Pmax_mustoff[t]:
            P = Pmax_mustoff[t]
        Pvbm.append(P)
        Einit = Einit + (1/12) * (P - Penergy_5m[t])
    plt.plot(pd.DataFrame(Pactual_5m, index=timestamps_5m, columns=['kW']).resample('15T').mean()['kW'], label='Actual Power Consumption (kW)', color='blue')
    plt.plot(pd.DataFrame(Pvbm, index=timestamps_5m, columns=['kW']).resample('15T').mean()['kW'], label='Estimated Power Consumption (kW)', color='orange')
    # plt.fill_between(timestamps_15m, pd.DataFrame(Pvbm, index=timestamps_5m, columns=['kW']).resample('15T').mean()['kW'],\
    #      pd.DataFrame(Pactual_5m, index=timestamps_5m, columns=['kW']).resample('15T').mean()['kW'], color='orange', label='delivery risk', alpha=0.5)
    plt.plot(pd.DataFrame(Pmuston, index=timestamps_5m, columns=['kW']).resample('15T').mean()['kW'], label='Pmuston (kW)', linestyle='--', color='purple')
    plt.plot(pd.DataFrame(Pmax_mustoff, index=timestamps_5m, columns=['kW']).resample('15T').mean()['kW'], label='Prated - Pmustoff (kW)', linestyle='--', color='red')
    plt.xlabel('Time')
    plt.ylabel('Power consumption (kW)')
    plt.ylim(-100, 10000)
    plt.legend()
    plt.show()    

def compare_actual_and_expected_energy_dynamics_baseline(start_time, end_time, scenario_name, delta_Emax, delta_Emin, delta_Pmax, delta_Pmin, alpha_aggregate, eta_aggregate):

    timestamps_5m = pd.date_range(start_time, end_time, freq=dt.timedelta(minutes=5),
                                  inclusive='left')
    timestamps_15m = pd.date_range(start_time, end_time, freq=dt.timedelta(minutes=15),
                                  inclusive='left')
    month = timestamps_5m[0].month
    day = timestamps_5m[0].day
                              
    Pactual_5m = pd.read_parquet(os.path.join(main_path, "cosim_scenarios", scenario_name, "compiled",
                                            "WH_results_5min_summed.parquet"), engine='pyarrow')\
                                            .loc[timestamps_5m]['Water Heating Electric Power (kW)'].values
    Penergy_5m = pd.read_csv(os.path.join(main_path, "cosim_scenarios", scenario_name,
                                          "baseline_power_scenarios.csv"), index_col=0,
                             parse_dates=True).loc[timestamps_5m].mean(axis=1).values
    Preference_5m = pd.read_csv(os.path.join(main_path, "cosim_scenarios", \
        scenario_name, 'aggregator_reference_' + str(month) + '_' + str(day) + '.csv'),
        index_col=0, parse_dates=True).loc[timestamps_5m].values
    
    Eactual = pd.read_parquet(os.path.join(main_path, "cosim_scenarios", scenario_name, "compiled",
                                            "WH_results_5min_summed.parquet"), engine='pyarrow')\
                                            .loc[timestamps_5m]['Water Heating EBM Energy (kWh)'].values

    Pvbm_energy = []
    delta_Einit = 0
    for t in range(len(timestamps_5m)):

        if Preference_5m[t] - Penergy_5m[t] > delta_Pmax[t]:
            delta_P = delta_Pmax[t]
        elif Preference_5m[t] - Penergy_5m[t] < delta_Pmin[t]:
            delta_P = delta_Pmin[t]
        else:
            delta_P = Preference_5m[t] - Penergy_5m[t]

        if delta_Einit + (1/12) * (eta_aggregate * delta_P - alpha_aggregate * delta_Einit) > delta_Emax[t]:
            delta_P = (12 * (delta_Emax[t] - delta_Einit) +  alpha_aggregate * delta_Einit)/eta_aggregate
        elif delta_Einit + (1/12) * (eta_aggregate * delta_P - alpha_aggregate * delta_Einit) < delta_Emin[t]:
            delta_P = (12 * (delta_Emin[t] - delta_Einit) +  alpha_aggregate * delta_Einit)/eta_aggregate
        else:
            delta_P = delta_P

        delta_Einit = delta_Einit + (1/12) * (eta_aggregate * delta_P - alpha_aggregate * delta_Einit)
        Pvbm_energy.append(Penergy_5m[t] + delta_P)
    
    meta_data = pd.read_csv(os.path.join(main_path, "cosim_scenarios", scenario_name, 'compiled', \
        "all_ochre_inputs.csv"), index_col=0, parse_dates=True)
    C = meta_data["Equipment.Water Heating.Tank Volume (L)"] * 4186
    setpoint = meta_data["Equipment.Water Heating.Setpoint Temperature (C)"]
    ambient = np.array([20] * len(setpoint)) # ask michael where to find the ambient temperature
    # Ebaseline = np.array([np.sum(np.multiply(C, setpoint - ambient))] * len(timestamps_5m)) / (3.6e6)
    Ebaseline = np.array([Eactual[0]]* len(timestamps_5m))
    Emax = (delta_Emax + Ebaseline)
    Emin = (delta_Emin + Ebaseline) 

    delta_Einit_actual = 0
    delta_Einit_vbm_energy = 0
    Expected_delta_E_vbm_actual = []
    Expected_delta_E_vbm_energy = []
    for t in range(len(timestamps_5m)):
        delta_E_vbm_actual = delta_Einit_actual + (1/12) * (1 * (Pactual_5m[t] - Penergy_5m[t]) - 1 * delta_Einit_actual) 
        delta_E_vbm_energy = delta_Einit_vbm_energy + (1/12) * (1 * (Pvbm_energy[t] - Penergy_5m[t]) - 1 * delta_Einit_vbm_energy) 
        delta_Einit_actual = delta_E_vbm_actual
        delta_Einit_vbm_energy = delta_E_vbm_energy
        Expected_delta_E_vbm_actual.append(delta_E_vbm_actual)
        Expected_delta_E_vbm_energy.append(delta_E_vbm_energy)   
    Expected_E_vbm_actual = (np.array(Expected_delta_E_vbm_actual) + Ebaseline) 
    Expected_E_vbm_energy = (np.array(Expected_delta_E_vbm_energy) + Ebaseline) 

    # Eenergy_5m = pd.read_csv(os.path.join(main_path, "cosim_scenarios", scenario_name,
    #                                       "Baseline_results_5min.csv"), index_col=0,
    #                          parse_dates=True)['Water Heating EBM Energy (kWh)'].loc[timestamps_5m]
    
    plt.plot(timestamps_5m, Eactual, label='Actual Energy State (kWh)', color='blue')
    plt.plot(timestamps_5m, Expected_E_vbm_energy, label='Estimated Energy State (kWh) based on expected Pbaseline', color='orange')
    # plt.fill_between(timestamps_5m, np.array(Expected_E_vbm_energy).flatten() + (Eenergy_5m.values - Eenergy_5m.values[0]), \
    #     np.array(Expected_E_vbm_energy).flatten(), label="Gap introduced by constant energy state assumption", alpha=0.5, color='orange')
    plt.plot(timestamps_5m, Expected_E_vbm_actual, label='Estimated Energy State (kWh) based on actual Pbaseline', color='green')
    # plt.fill_between(timestamps_5m, np.array(Expected_E_vbm_actual).flatten() + (Eenergy_5m.values - Eenergy_5m.values[0]), \
    #     np.array(Expected_E_vbm_actual).flatten(), label="Gap introduced by constant energy state assumption", alpha=0.5, color='red')
    plt.plot(timestamps_5m, Emax, linestyle="--", label='Emax (kWh)', color='red')
    plt.plot(timestamps_5m, Emin, linestyle="--", label='Emin (kWh)', color='purple')
    plt.xlabel('Time')
    plt.ylabel('Energy (kWh)')
    plt.ylim(17000, 19300)
    plt.legend()
    plt.show()

def compare_actual_versus_estimated_power_consumption_baseline(start_time, end_time, scenario_name, delta_Emax, delta_Emin, delta_Pmax, delta_Pmin, alpha_aggregate, eta_aggregate):

    timestamps_5m = pd.date_range(start_time, end_time, freq=dt.timedelta(minutes=5),
                                  inclusive='left')
    timestamps_15m = pd.date_range(start_time, end_time, freq=dt.timedelta(minutes=15),
                                  inclusive='left')
    month = timestamps_5m[0].month
    day = timestamps_5m[0].day
                              
    Pactual_5m = pd.read_parquet(os.path.join(main_path, "cosim_scenarios", scenario_name, "compiled",
                                            "WH_results_5min_summed.parquet"), engine='pyarrow')\
                                            .loc[timestamps_5m]['Water Heating Electric Power (kW)'].values
    Penergy_5m = pd.read_csv(os.path.join(main_path, "cosim_scenarios", scenario_name,
                                          "baseline_power_scenarios.csv"), index_col=0,
                             parse_dates=True).loc[timestamps_5m].mean(axis=1).values
    Preference_5m = pd.read_csv(os.path.join(main_path, "cosim_scenarios", \
        scenario_name, 'aggregator_reference_' + str(month) + '_' + str(day) + '.csv'),
        index_col=0, parse_dates=True).loc[timestamps_5m].values
    
    Pvbm = []
    delta_Einit = 0
    for t in range(len(timestamps_5m)):

        if Preference_5m[t] - Penergy_5m[t] > delta_Pmax[t]:
            delta_P = delta_Pmax[t]
        elif Preference_5m[t] - Penergy_5m[t] < delta_Pmin[t]:
            delta_P = delta_Pmin[t]
        else:
            delta_P = Preference_5m[t] - Penergy_5m[t]

        if delta_Einit + (1/12) * (eta_aggregate * delta_P - alpha_aggregate * delta_Einit) > delta_Emax[t]:
            delta_P = (12 * (delta_Emax[t] - delta_Einit) +  alpha_aggregate * delta_Einit)/eta_aggregate
        elif delta_Einit + (1/12) * (eta_aggregate * delta_P - alpha_aggregate * delta_Einit) < delta_Emin[t]:
            delta_P = (12 * (delta_Emin[t] - delta_Einit) +  alpha_aggregate * delta_Einit)/eta_aggregate
        else:
            delta_P = delta_P

        delta_Einit = delta_Einit + (1/12) * (eta_aggregate * delta_P - alpha_aggregate * delta_Einit)
        Pvbm.append(Penergy_5m[t] + delta_P)
 
    plt.plot(pd.DataFrame(Pactual_5m, index=timestamps_5m, columns=['kW']).resample('15T').mean()['kW'], label='Actual Power Consumption (kW)', color='blue')
    plt.plot(pd.DataFrame(Pvbm, index=timestamps_5m, columns=['kW']).resample('15T').mean()['kW'], label='Estimated Power Consumption (kW)', color='orange')
    # plt.fill_between(timestamps_15m, 
    #      pd.DataFrame(Pvbm, index=timestamps_5m, columns=['kW']).resample('15T').mean()['kW'],\
    #      pd.DataFrame(Pactual_5m, index=timestamps_5m, columns=['kW']).resample('15T').mean()['kW'], color='orange', label='delivery risk', alpha=0.5)
    plt.plot(pd.DataFrame(delta_Pmax + Penergy_5m, index=timestamps_5m, columns=['kW']).resample('15T').mean()['kW'], label='Pmax (kW)', linestyle='--', color='red')
    plt.plot(pd.DataFrame(delta_Pmin + Penergy_5m, index=timestamps_5m, columns=['kW']).resample('15T').mean()['kW'], label='Pmin (kW)', linestyle='--', color='purple')
    plt.xlabel('Time')
    plt.ylabel('Power consumption (kW)')
    plt.ylim(-100, 10000)
    plt.legend()
    plt.show()    

    print("Delivery risk: ", 1/4 * np.sum(np.abs(pd.DataFrame(Pvbm, index=timestamps_5m, columns=['kW']).resample('15T').mean().kW - \
        pd.DataFrame(Pactual_5m, index=timestamps_5m, columns=['kW']).resample('15T').mean().kW)))

def calculate_baselineVBM_parameters(start_time, end_time, scenario_name):
    timestamps_5m = pd.date_range(start_time, end_time, freq=dt.timedelta(minutes=5),
                                  inclusive='left')
    meta_data = pd.read_csv(os.path.join(main_path, "cosim_scenarios", scenario_name, 'compiled', \
        "all_ochre_inputs.csv"), index_col=0, parse_dates=True)
    R = 1/meta_data["Equipment.Water Heating.UA (W/K)"]
    C = meta_data["Equipment.Water Heating.Tank Volume (L)"] * 4186
    Prate = meta_data["Equipment.Water Heating.Capacity (W)"]
    deadband = meta_data["Equipment.Water Heating.Deadband Temperature (C)"]
    eta = meta_data["Equipment.Water Heating.Efficiency (-)"]
    alpha = 1/(R*C)
    eta_aggregate = np.mean(eta)
    alpha_aggregate = np.mean(alpha)
    
    Pbaseline = pd.read_csv(os.path.join(main_path, "cosim_scenarios", scenario_name, \
        "baseline_power_scenarios_units.csv"), index_col=0, parse_dates=True).loc[timestamps_5m]

    temp = np.multiply(eta_aggregate * np.multiply(alpha, np.multiply(C, deadband/2)), 1/np.multiply(alpha+np.abs(alpha_aggregate-alpha), eta))
    beta_opt_E = temp/np.sum(temp) 
    Emax = pd.DataFrame([temp[0]/(3.6e6 * beta_opt_E[0])]*len(timestamps_5m), index=timestamps_5m, columns=['kWh'])
    Emin = - Emax
    Pmax = pd.DataFrame(index=timestamps_5m, columns=['kW'])
    Pmin = pd.DataFrame(index=timestamps_5m, columns=['kW'])
    for t in timestamps_5m:
        Pmin['kW'].loc[t] = - np.min(Pbaseline.loc[t]/beta_opt_E)
        Pmax['kW'].loc[t] = np.min((Prate/1000 - Pbaseline.loc[t])/beta_opt_E)
    return Emax['kWh'].values, Emin['kWh'].values, Pmax['kW'].values, Pmin['kW'].values, alpha_aggregate, eta_aggregate
    
def calculate_baselineVBM_parameters_geometric(start_time, end_time, scenario_name):
    timestamps_5m = pd.date_range(start_time, end_time, freq=dt.timedelta(minutes=5),
                                  inclusive='left')
    meta_data = pd.read_csv(os.path.join(main_path, "cosim_scenarios", scenario_name, 'compiled', \
        "all_ochre_inputs.csv"), index_col=0, parse_dates=True)
    r = 1/meta_data["Equipment.Water Heating.UA (W/K)"]
    c = meta_data["Equipment.Water Heating.Tank Volume (L)"] * 4186
    # setpoint = meta_data["Equipment.Water Heating.Setpoint Temperature (C)"]
    deadband = meta_data["Equipment.Water Heating.Deadband Temperature (C)"]
    eta = meta_data["Equipment.Water Heating.Efficiency (-)"]
    results_5m = pd.read_parquet(os.path.join(main_path, "cosim_scenarios", scenario_name, "compiled",
                                            'WH_results_5min.parquet'), engine='pyarrow')
    results_5m = results_5m.reset_index(level=[1])
    houses = results_5m['House'].unique()
    tempinit = pd.DataFrame(index=houses, columns=['C'])
    for house in houses:
        results_5m_filtered = results_5m[results_5m.House == house]
        tempinit.C[house] = results_5m_filtered['Hot Water Average Temperature (C)'].iloc[0]
    Prate = meta_data["Equipment.Water Heating.Capacity (W)"]
    Pbaseline = pd.read_csv(os.path.join(main_path, "cosim_scenarios", scenario_name, \
        "baseline_power_scenarios_units.csv"), index_col=0, parse_dates=True).loc[timestamps_5m].values
    u_upper = Prate.values * np.ones((len(timestamps_5m), 1)) - Pbaseline
    u_lower = Pbaseline
    x_upper = np.multiply(np.multiply(c, deadband/2), 1/eta)
    x_lower = np.multiply(np.multiply(c, deadband/2), 1/eta)

    r_0 = np.mean(r)
    c_0 = np.mean(c)
    # setpoint_0 = np.mean(setpoint)
    deadband_0 = np.mean(deadband)
    eta_0 = np.mean(eta)
    tempinit_0 = np.mean(tempinit.C)
    Prate_0 = np.mean(Prate)
    Pbaseline_0 = np.mean(Pbaseline, axis=1)
    u_upper_0 = Prate_0 - Pbaseline_0
    u_lower_0 = Pbaseline_0
    x_upper_0 = c_0 * (deadband_0/2) * (1/eta_0)
    x_lower_0 = c_0 * (deadband_0/2) * (1/eta_0)

    delta_t = 1/12
    Im = np.diag([1] * len(timestamps_5m))
    tempamb = 20

    A_0 = Im + np.diag([-(1-delta_t/(r_0 * c_0))] * (len(timestamps_5m) - 1), -1)
    B_0 = delta_t * Im
    C_0 = np.array([(1-delta_t/(r_0 * c_0)) * c_0 * (tempinit_0 - tempamb)/eta_0] + [0] * (len(timestamps_5m)-1)).T
    F_0 = np.vstack((Im, -Im, np.matmul(np.linalg.inv(A_0), B_0), -np.matmul(np.linalg.inv(A_0), B_0)))
    H_0 = np.hstack((u_upper_0, u_lower_0, x_upper_0 - np.matmul(np.linalg.inv(A_0), C_0), \
        x_lower_0 + np.matmul(np.linalg.inv(A_0), C_0))).reshape(4*len(timestamps_5m), 1)
    # F_0 = np.vstack((np.matmul(np.linalg.inv(A_0), B_0), -np.matmul(np.linalg.inv(A_0), B_0)))
    # H_0 = np.hstack((x_upper_0 - np.matmul(np.linalg.inv(A_0), C_0), \
    #     x_lower_0 + np.matmul(np.linalg.inv(A_0), C_0))).reshape(2*len(timestamps_5m), 1)
    coeff = []
    intercept = []
    for k in range(len(meta_data.index)):
        print('House: ', k)
        A_k = Im + np.diag([-(1-delta_t/(r[k] * c[k]))] * (len(timestamps_5m) - 1), -1)
        B_k = delta_t * Im
        C_k = np.array([(1-delta_t/(r[k] * c[k])) * c[k] * (tempinit.C.values[k] - tempamb)/eta[k]] + [0] * (len(timestamps_5m)-1)).T
        F_k = np.vstack((Im, -Im, np.matmul(np.linalg.inv(A_k), B_k), -np.matmul(np.linalg.inv(A_k), B_k)))
        H_k = np.hstack((u_upper[:, k], u_lower[:, k], x_upper.values[k] - np.matmul(np.linalg.inv(A_k), C_k), \
            x_lower.values[k] + np.matmul(np.linalg.inv(A_k), C_k))).reshape(4*len(timestamps_5m), 1)
        # F_k = np.vstack((Im, -Im, np.matmul(np.linalg.inv(A_k), B_k), -np.matmul(np.linalg.inv(A_k), B_k)))
        # H_k = np.hstack((x_upper.values[k] - np.matmul(np.linalg.inv(A_k), C_k), \
        #     x_lower.values[k] + np.matmul(np.linalg.inv(A_k), C_k))).reshape(2*len(timestamps_5m), 1)
        if k == 0:
            coeff_k, intercept_k, opt_model = optimizer_init(F_0, H_0, F_k, H_k, len(timestamps_5m))
        else:
            coeff_k, intercept_k = optimizer_update(F_0, H_0, F_k, H_k, opt_model, len(timestamps_5m))
        coeff.append(coeff_k)
        intercept.append(intercept_k)
    print(coeff)
    print(intercept)

def optimizer_init(F_0, H_0, F_k, H_k, period):

    model = ConcreteModel("VBM")
    print("Create the VBM")
    model.Idx_time = range(period)
    model.Idx_time_4 = range(4 * period)
    model.Idx_4 = range(4)
    F_0_dict = {}
    H_0_dict = {}
    F_k_dict = {}
    H_k_dict = {}
    for t_a in range(4 * period):
        for t_b in range(period):
            F_0_dict[(t_a, t_b)] = F_0[t_a, t_b]
            F_k_dict[(t_a, t_b)] = F_k[t_a, t_b]
    for t in range(4 * period):
        H_0_dict[t] = H_0[t]
        H_k_dict[t] = H_k[t]
    model.F_0 = Param(model.Idx_time_4, model.Idx_time, initialize=F_0_dict, mutable=True)
    model.H_0 = Param(model.Idx_time_4, initialize=H_0_dict, mutable=True) 
    model.F_k = Param(model.Idx_time_4, model.Idx_time, initialize=F_k_dict, mutable=True)
    model.H_k = Param(model.Idx_time_4, initialize=H_k_dict, mutable=True) 
    
    # define variable
    G_dict = {}
    for t_a in range(4 * period):
        for t_b in range(4 * period):
            G_dict[(t_a, t_b)] = 1/(4*period)
    model.G = Var(model.Idx_time_4, model.Idx_time_4, domain=NonNegativeReals)
    model.s = Var(domain=PosiveReals, initialize=1)
    model.r = Var(model.Idx_time, domain=Reals, initialize=np.array([0] * period))

    def obj_expression(model):
        return model.s

    model.OBJ = Objective(rule=obj_expression, sense=minimize)

    def Equality(model, T_a, T_b):
        return sum(model.G[T_a, t] * model.F_0[t, T_b] for t in model.Idx_time_4) == model.F_k[T_a, T_b]

    def Inequality(model, T):
        return sum(model.G[T, t] * model.H_0[t] for t in model.Idx_time_4) <= model.s * model.H_k[T] + sum( model.F_k[T, t] * model.r[t] for t in model.Idx_time)

    model.Equality = Constraint(model.Idx_time_4, model.Idx_time, rule=Equality)
    model.Inequality = Constraint(model.Idx_time_4, rule=Inequality)
    
    opt = SolverFactory("gurobi")
    result = opt.solve(model, tee=True)

    print(result)
    model.display()

    print("model solved")

    s = value(model.s)
    print('s:', s)

    r = []
    for t in model.Idx_time:
        r.append(value(model.r[t]))
    print('r:', r)

    coeff = 1/s
    intercept = -np.array(r)/s
    return coeff, intercept, model

def optimizer_update(F_0, H_0, F_k, H_k, model, period):
    F_0_dict = {}
    H_0_dict = {}
    F_k_dict = {}
    H_k_dict = {}
    for t_a in range(4 * period):
        for t_b in range(period):
            F_0_dict[(t_a, t_b)] = F_0[t_a, t_b]
            F_k_dict[(t_a, t_b)] = F_k[t_a, t_b]
    for t in range(4 * period):
        H_0_dict[t] = H_0[t]
        H_k_dict[t] = H_k[t]
    model.F_0.reconstruct(co1_S0_real)
    model.H_0.reconstruct(co1_S0_imag)  # first order coefficient of the linear Q0 equation
    model.F_k.reconstruct(co0_S0_real)  # intercept of the linear P0 equation
    model.H_k.reconstruct(co0_S0_imag) 
    opt = SolverFactory("gurobi")
    result = opt.solve(model, tee=True)

    print(result)
    model.display()

    print("model solved")

    s = value(model.s)
    r = []
    for t in model.Idx_time:
        r.append(value(model.r[t]))
    coeff = 1/s
    intercept = -np.array(r)/s
    
    return coeff, intercept
    
if __name__ == "__main__":

    # week = "apr_16th"
    # scenario_name = "wh_all_dao_apr16th_simple"
    # start_time = dt.datetime(2018, 4, 18)
    # end_time = dt.datetime(2018, 4, 19)

    week = "jun_4th"
    scenario_name = "wh_all_dao_jun4th_simple"
    start_time = dt.datetime(2018, 6, 5)
    end_time = dt.datetime(2018, 6, 6)

    compare_actual_and_expected_energy_dynamics(start_time, end_time, scenario_name)
    compare_actual_versus_estimated_power_consumption(start_time, end_time, scenario_name)
    
    # delta_Emax, delta_Emin, delta_Pmax, delta_Pmin, alpha_aggregate, eta_aggregate = calculate_baselineVBM_parameters(start_time, end_time, scenario_name)
    # compare_actual_and_expected_energy_dynamics_baseline(start_time, end_time, scenario_name, delta_Emax, delta_Emin, delta_Pmax, delta_Pmin, alpha_aggregate, eta_aggregate)
    # compare_actual_versus_estimated_power_consumption_baseline(start_time, end_time, scenario_name, delta_Emax, delta_Emin, delta_Pmax, delta_Pmin, alpha_aggregate, eta_aggregate)
    
    Emax, Emin, delta_Pmax, delta_Pmin = calculate_baselineVBM_parameters_geometric(start_time, end_time, scenario_name)

    pass

import os
import sys
import numpy as np
import pandas as pd
import datetime as dt
# from creating_Pbase_scenarios import create_wh_baseline_aggregate, create_hvac_baseline_aggregate
import pwlf
import matplotlib.pyplot as plt

main_path = os.path.dirname(__file__)

def create_virtual_battery_model(training_scenario, save_to_scenario):
    
    Pactual = pd.read_parquet(os.path.join(main_path, "..", "..", "cosim_scenarios", training_scenario, 'compiled', \
        'WH_results_5min_summed.parquet'),engine='pyarrow')["Water Heating Electric Power (kW)"].to_frame('kW')
  
    Pbaseline = pd.read_csv(os.path.join(main_path,"..", "..", "cosim_scenarios", training_scenario, 'Baseline_results_5min.csv'),
                           index_col=0, parse_dates=True)['Water Heating Electric Power (kW)']\
        .loc[Pactual.index].to_frame("kW")
    
    Eactual = pd.read_parquet(os.path.join(main_path, "..", "..", "cosim_scenarios", training_scenario, "compiled",
                                            "WH_results_5min_summed.parquet"), engine='pyarrow')\
                                            ['Water Heating EBM Energy (kWh)']
    
    results_5m = pd.read_parquet(os.path.join(main_path, "..", "..", "cosim_scenarios", training_scenario, "compiled",
                                            'WH_results_5min.parquet'), engine='pyarrow')
    results_5m = results_5m.reset_index(level=[1])
    
    houses_meta = pd.read_csv(os.path.join(main_path, "..", "..", "cosim_scenarios", training_scenario, "compiled",
                                            'all_ochre_inputs.csv'), index_col=0)
    houses = results_5m['House'].unique()
    soc_5m = pd.DataFrame(index=Eactual.index, columns=houses)
    Pmuston_5m = pd.DataFrame(index=Eactual.index, columns=['kW'])
    Pmustoff_5m = pd.DataFrame(index=Eactual.index, columns=['kW'])
    Prated_5m = pd.DataFrame(index=Eactual.index, columns=['kW'])
    for house in houses:
        results_5m_filtered = results_5m[results_5m.House == house]
        soc_5m[house] = (results_5m_filtered['Water Heating EBM Energy (kWh)'] - \
            results_5m_filtered['Water Heating EBM Min Energy (kWh)'])/(\
            results_5m_filtered['Water Heating EBM Max Energy (kWh)'] - \
            results_5m_filtered['Water Heating EBM Min Energy (kWh)'])
    print(soc_5m.describe())

    for t_5m in Eactual.index:
        houses_muston = soc_5m.loc[t_5m][soc_5m.loc[t_5m] <= 0].index
        houses_mustoff = soc_5m.loc[t_5m][soc_5m.loc[t_5m] >= 1].index    
        Pmuston_5m.loc[t_5m]['kW'] = np.sum(houses_meta.loc[houses_muston]['Equipment.Water Heating.Capacity (W)'])/1000
        Pmustoff_5m.loc[t_5m]['kW'] = np.sum(houses_meta.loc[houses_mustoff]['Equipment.Water Heating.Capacity (W)'])/1000
        Prated_5m.loc[t_5m]['kW'] = np.sum(houses_meta['Equipment.Water Heating.Capacity (W)']/1000)

    data_tosort = pd.DataFrame(np.array(\
        [Pmuston_5m.kW, Pmustoff_5m.kW, \
        Pactual.kW, \
        Pbaseline.kW, \
        Pactual.kW - Pbaseline.kW, \
        Prated_5m.kW,
        Eactual]).T, \
        index=Eactual.index, \
        columns=['Pmuston', 'Pmustoff', \
        'Pactual', 'Pbaseline', 'actual-baseline', 'Prated', 'energy'])
    data_sorted = data_tosort.sort_values(by=['energy'], ascending=True)
    data_filtered_up = data_sorted[data_sorted['actual-baseline'] < 0]
    data_filtered_down = data_sorted[data_sorted['actual-baseline'] >= 0]
    print(data_filtered_up.energy)
    print(data_filtered_up.Pmuston)
    print(data_filtered_up['actual-baseline'])
    model_on_up = pwlf.PiecewiseLinFit(data_filtered_up.energy.to_list(), data_filtered_up.Pmuston.to_list(), weights=np.abs(data_filtered_up['actual-baseline'].to_list()))
    model_on_up.fit(2)
    model_on_down = pwlf.PiecewiseLinFit(data_filtered_down.energy.to_list(), data_filtered_down.Pmuston.to_list(), weights=np.abs(data_filtered_down['actual-baseline'].to_list()))
    model_on_down.fit(2)
    model_off_up = pwlf.PiecewiseLinFit(data_filtered_up.energy.to_list(), data_filtered_up.Pmustoff.to_list(), weights=np.abs(data_filtered_up['actual-baseline'].to_list()))
    model_off_up.fit(3)
    model_off_down = pwlf.PiecewiseLinFit(data_filtered_down.energy.to_list(), data_filtered_down.Pmustoff.to_list(), weights=np.abs(data_filtered_down['actual-baseline'].to_list()))
    model_off_down.fit(3)
    
    fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    plt.scatter(data_sorted.Pbaseline, data_sorted.Pmuston, c=data_sorted['actual-baseline'], cmap='coolwarm') 
    plt.scatter(data_filtered_up.Pbaseline, model_on_up.predict(data_filtered_up.energy), c='blue')
    plt.scatter(data_filtered_down.Pbaseline, model_on_down.predict(data_filtered_down.energy), c='red')
    plt.xlabel('Pbaseline (kW)')
    plt.ylabel('Pmuston (kW)')
    # ax.set_zlabel('Pmuston (kW)')
    plt.savefig(os.path.join(main_path, "..", "..", "cosim_scenarios", save_to_scenario,
                                            "Muston_fig.png"))
    plt.clf()
    print("coefficients for on and up:", model_on_up.slopes)
    print("intercept for on and up:", model_on_up.intercepts)
    print("coefficients for on and down:", model_on_down.slopes)
    print("intercept for on and down:", model_on_down.intercepts)

    fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    plt.scatter(data_sorted.Pbaseline, data_sorted.Pmustoff, c=data_sorted['actual-baseline'], cmap='coolwarm') 
    plt.scatter(data_filtered_up.Pbaseline, model_off_up.predict(data_filtered_up.energy), c='blue')
    plt.scatter(data_filtered_down.Pbaseline, model_off_down.predict(data_filtered_down.energy), c='red')
    plt.xlabel('Pbaseline (kW)')
    plt.ylabel('Pmustoff (kW)')
    # ax.set_zlabel('Pmustoff (kW)')
    plt.savefig(os.path.join(main_path, "..", "..", "cosim_scenarios", save_to_scenario,
                                            "Mustoff_fig.png"))
    plt.clf()
    print("coefficients for off and up:", model_off_up.slopes)
    print("intercept for off and up:", model_off_up.intercepts)
    print("coefficients for off and down:", model_off_down.slopes)
    print("intercept for off and down:", model_off_down.intercepts)

    plt.scatter(data_sorted.energy, data_sorted.Pmuston/len(houses), c=data_sorted['actual-baseline'], cmap='coolwarm') 
    plt.scatter(data_filtered_up.energy, model_on_up.predict(data_filtered_up.energy)/len(houses), c='blue')
    plt.scatter(data_filtered_down.energy, model_on_down.predict(data_filtered_down.energy)/len(houses), c='red')
    plt.scatter(data_sorted.energy, (data_sorted.Prated - data_sorted.Pmustoff)/len(houses), c=data_sorted['actual-baseline'], cmap='coolwarm') 
    plt.scatter(data_filtered_up.energy, (data_filtered_up.Prated - model_off_up.predict(data_filtered_up.energy))/len(houses), c='blue')
    plt.scatter(data_filtered_down.energy, (data_filtered_down.Prated - model_off_down.predict(data_filtered_down.energy))/len(houses), c='red')
    plt.xlabel('Energy (kWh)')
    plt.ylabel('Pmustoff and Pmuston (kW)')
    plt.ylim(-0.1, 5.7)
    plt.savefig(os.path.join(main_path, "..", "..", "cosim_scenarios", save_to_scenario,
                                            "Power_constraint.png"))
    plt.clf()

    power_constraints_parameters = pd.DataFrame([[model_on_up.slopes[0], model_on_up.intercepts[0]],\
                                                 [model_on_up.slopes[1], model_on_up.intercepts[1]],\
                                                 [model_on_down.slopes[0], model_on_down.intercepts[0]],\
                                                 [model_on_down.slopes[1], model_on_down.intercepts[1]],\
                                                 [model_off_up.slopes[0], model_off_up.intercepts[0]],\
                                                 [model_off_up.slopes[1], model_off_up.intercepts[1]],\
                                                 [model_off_up.slopes[2], model_off_up.intercepts[2]],\
                                                 [model_off_down.slopes[0], model_off_down.intercepts[0]],\
                                                 [model_off_down.slopes[1], model_off_down.intercepts[1]],\
                                                 [model_off_down.slopes[2], model_off_down.intercepts[2]]], \
                                                index=['on_up_1', 'on_up_2', 'on_down_1', 'on_down_2',\
                                                       'off_up_1', 'off_up_2', 'off_up_3', \
                                                       'off_down_1', 'off_down_2', 'off_down_3'], \
                                                columns=['coefficient', 'intercept'])
    power_constraints_parameters.to_csv(os.path.join(main_path, "..", "..", "cosim_scenarios", save_to_scenario,
                                            "power_constraints_parameters.csv"))
                                            
if __name__ == '__main__':
    
    training_scenario = sys.argv[1]
    save_to_scenario = sys.argv[2]

    create_virtual_battery_model(training_scenario, save_to_scenario)
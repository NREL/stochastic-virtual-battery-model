import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


class WHAggregator(object):

    def __init__(self, IDs, duration):
        self.IDs = IDs
        self.Num_WH = len(self.IDs)
        self.duration = duration
        self.Num_mustON = [0] * self.duration
        self.Num_mustOFF = [0] * self.duration
        self.Num_Flex = [0] * self.duration
        self.HSoC_Flex = [0] * self.duration
        self.signal = None
        self.soc_a = None
        self.soc_b = None
        self.soc_c = None
        self.soc_i = None
        self.nmuston_a = None
        self.nmuston_b = None
        self.nmuston_c = None
        self.nmuston_d = None
        self.nmuston_i = None
        self.nmustoff_a = None
        self.nmustoff_b = None
        self.nmustoff_c = None
        self.nmustoff_d = None
        self.nmustoff_i = None

    def receive_signal(self, signal):
        self.signal = signal

    def reset(self):
        self.Num_mustON = [0] * self.duration
        self.Num_mustOFF = [0] * self.duration
        self.Num_Flex = [0] * self.duration
        self.HSoC_Flex = [0] * self.duration

    # def update_HSoC_Flex(self, k, States, Ts, Tuppers, Tlowers):
    #     SoCs_Flex = []
    #     for index in States[(States.state == "FLEX1") | (States.state == "FLEX2") | (States.state == "FLEX3") | \
    #                         (States.state == "FLEX4") | (States.state == "FLEX5")].index:
    #         SoCs_Flex.append((Ts.loc[index]-Tlowers.loc[index])/(Tuppers.loc[index]-Tlowers.loc[index]))
    #     self.HSoC_Flex[k] = np.mean(SoCs_Flex)

    def generate_actions(self, k, States, ratedPower):

        States['action'] = np.nan
        Num_ON = round(self.signal.Power[k]/ratedPower)
        Num_mustON = len(States[States.state == "MustON"])
        Num_mustOFF = len(States[States.state == "MustOFF"])
        Num_Flex = self.Num_WH - Num_mustON - Num_mustOFF
        self.Num_mustON[k] = Num_mustON
        self.Num_mustOFF[k] = Num_mustOFF
        self.Num_Flex[k] = Num_Flex
        States.loc[States.state == "MustON", 'action'] = 1
        States.loc[States.state == "MustOFF", 'action'] = 0

        if Num_ON > Num_mustON:
            if Num_ON - Num_mustON < Num_Flex:
                x = np.zeros(Num_Flex)
                for idx in random.sample(range(Num_Flex), Num_ON-Num_mustON):
                    x[idx] = 1
                States.loc[(States.state == "FLEX1") | (States.state == "FLEX2") | (States.state == "FLEX3") | \
                                   (States.state == "FLEX4")|(States.state == "FLEX5"), 'action'] = x
            else:
                States.loc[(States.state == "FLEX1")|(States.state == "FLEX2")|(States.state == "FLEX3")| \
                       (States.state == "FLEX4")|(States.state == "FLEX5"), 'action'] = 1
        else:
            States.loc[(States.state == "FLEX1")|(States.state == "FLEX2")|(States.state == "FLEX3")| \
                       (States.state == "FLEX4")|(States.state == "FLEX5"), 'action'] = 0

        Actions = States
        return Actions

    # def generate_actions_nonoptout(self, k, States, ratedPower):

    #     States['action'] = np.nan
    #     Num_ON = round(self.signal.Power[k]/ratedPower)
    #     Num_mustON = len(States[States.state == "MustON"])
    #     Num_mustOFF = len(States[States.state == "MustOFF"])
    #     Num_Flex = self.Num_WH - Num_mustON - Num_mustOFF
    #     self.Num_mustON[k] = Num_mustON
    #     self.Num_mustOFF[k] = Num_mustOFF
    #     self.Num_Flex[k] = Num_Flex

    #     if Num_mustON <= Num_ON <= Num_mustON + Num_Flex:
    #         States.loc[States.state == "MustON", 'action'] = 1
    #         States.loc[States.state == "MustOFF", 'action'] = 0
    #         x = np.zeros(Num_Flex)
    #         for idx in random.sample(range(Num_Flex), Num_ON - Num_mustON):
    #             x[idx] = 1
    #         States.loc[(States.state == "FLEX1") | (States.state == "FLEX2") | (States.state == "FLEX3") | \
    #                    (States.state == "FLEX4") | (States.state == "FLEX5"), 'action'] = x
    #     elif Num_mustON + Num_Flex < Num_ON:
    #         States.loc[States.state == "MustON", 'action'] = 1
    #         x = np.zeros(Num_mustOFF)
    #         for idx in random.sample(range(Num_mustOFF), Num_ON - Num_mustON - Num_Flex):
    #             x[idx] = 1
    #         States.loc[States.state == "MustOFF", 'action'] = x
    #         States.loc[(States.state == "FLEX1") | (States.state == "FLEX2") | (States.state == "FLEX3") | \
    #                    (States.state == "FLEX4") | (States.state == "FLEX5"), 'action'] = 1
    #     elif Num_ON < Num_mustON:
    #         x = np.ones(Num_mustON)
    #         for idx in random.sample(range(Num_mustON), Num_mustON - Num_ON):
    #             x[idx] = 0
    #         States.loc[States.state == "MustON", 'action'] = x
    #         States.loc[States.state == "MustOFF", 'action'] = 0
    #         States.loc[(States.state == "FLEX1") | (States.state == "FLEX2") | (States.state == "FLEX3") | \
    #                    (States.state == "FLEX4") | (States.state == "FLEX5"), 'action'] = 0

    #     Actions = States
    #     return Actions


    def generate_actions_sorting(self, k, SoCs, ratedPower):
        SoCs['action'] = np.zeros(len(SoCs.index))
        Num_ON = round(self.signal.Power[k] / ratedPower)
        self.Num_mustON[k] = len(SoCs[SoCs.soc <= 0])
        self.Num_mustOFF[k] = len(SoCs[SoCs.soc >= 100])
        SoCs_sorted = SoCs.sort_values('soc')

        if self.Num_mustON[k] > Num_ON:
            IDs_ON = SoCs_sorted.index[:self.Num_mustON[k]]
        elif Num_ON > len(SoCs.index) - self.Num_mustOFF[k]:
            IDs_ON = SoCs_sorted.index[:len(SoCs.index) - self.Num_mustOFF[k]]
        else:
            IDs_ON = SoCs_sorted.index[:Num_ON]
        for id in IDs_ON:
            SoCs.loc[id]['action'] = 1

        Actions = SoCs
        return Actions


    # def generate_actions_sorting_nonoptout(self, k, SoCs, ratedPower):
    #     SoCs['action'] = np.zeros(len(SoCs.index))
    #     Num_ON = round(self.signal.Power[k] / ratedPower)
    #     self.Num_mustON[k] = len(SoCs[SoCs.soc <= 0])
    #     self.Num_mustOFF[k] = len(SoCs[SoCs.soc >= 100])
    #     SoCs_sorted = SoCs.sort_values('soc')

    #     IDs_ON = SoCs_sorted.index[:Num_ON]
    #     for id in IDs_ON:
    #         SoCs.loc[id]['action'] = 1

    #     Actions = SoCs
    #     return Actions


    def virtual_battery_model_fitting(self, his_P, his_V, his_soc, his_nmuston, his_nmustoff, Interval):

        X_soc = np.array([
            his_soc.Percentage.resample(Interval).mean().values[:-1],
            his_P.Power.resample(Interval).mean().values[:-1],
            his_V.Liter.resample(Interval).mean().values[:-1],
        ]).T
        y_soc = his_soc.Percentage.resample(Interval).mean().values[1:]
        X_soc_train, X_soc_test, y_soc_train, y_soc_test = train_test_split(
            X_soc, y_soc, test_size=0.3, random_state=90)

        X_nmuston = np.array([
            his_nmuston.Count.resample(Interval).mean().values[:-1],
            his_soc.Percentage.resample(Interval).mean().values[:-1],
            his_P.Power.resample(Interval).mean().values[:-1],
            his_V.Liter.resample(Interval).mean().values[:-1]
        ]).T
        y_nmuston = his_nmuston.Count.resample(Interval).mean().values[1:]
        X_nmuston_train, X_nmuston_test, y_nmuston_train, y_nmuston_test = train_test_split(
            X_nmuston, y_nmuston, test_size=0.3, random_state=90)

        X_nmustoff = np.array([
            his_nmustoff.Count.resample(Interval).mean().values[:-1],
            his_soc.Percentage.resample(Interval).mean().values[:-1],
            his_P.Power.resample(Interval).mean().values[:-1],
            his_V.Liter.resample(Interval).mean().values[:-1]
        ]).T
        y_nmustoff = his_nmustoff.Count.resample(Interval).mean().values[1:]
        X_nmustoff_train, X_nmustoff_test, y_nmustoff_train, y_nmustoff_test = train_test_split(
            X_nmustoff, y_nmustoff, test_size=0.3, random_state=90)

        pipe_soc = make_pipeline(LinearRegression())
        pipe_nmuston = make_pipeline(LinearRegression())
        pipe_nmustoff = make_pipeline(LinearRegression())

        pipe_soc.fit(X_soc_train, y_soc_train)
        pipe_nmuston.fit(X_nmuston_train, y_nmuston_train)
        pipe_nmustoff.fit(X_nmustoff_train, y_nmustoff_train)

        print("Score for the soc model:", pipe_soc.score(X_soc_test, y_soc_test))
        print("Score for the nmuston model:", pipe_nmuston.score(X_nmuston_test, y_nmuston_test))
        print("Score for the nmustoff model:", pipe_nmustoff.score(X_nmustoff_test, y_nmustoff_test))

        [self.soc_a, self.soc_b, self.soc_c] = pipe_soc.steps[0][1].coef_
        self.soc_i = pipe_soc.steps[0][1].intercept_
        [self.nmuston_a, self.nmuston_b, self.nmuston_c, self.nmuston_d] = pipe_nmuston.steps[0][1].coef_
        self.nmuston_i = pipe_nmuston.steps[0][1].intercept_
        [self.nmustoff_a, self.nmustoff_b, self.nmustoff_c, self.nmustoff_d] = pipe_nmustoff.steps[0][1].coef_
        self.nmustoff_i = pipe_nmustoff.steps[0][1].intercept_

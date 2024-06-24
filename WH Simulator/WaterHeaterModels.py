import numpy as np

class WaterHeater(object):

    def __init__(self, dict_WH):

        self.C = dict_WH["SpecificHeatCapacity"]
        self.Tinit = dict_WH["InitialT"]
        self.Tambient = dict_WH["AmbientT"]
        self.Tinlet = dict_WH["InletT"]
        self.Tupper = dict_WH["UpperT"]
        self.Tlower = dict_WH["LowerT"]
        self.Eff = dict_WH["Efficiency"]
        self.WD = dict_WH["WaterDensity"]
        self.L = dict_WH["Volumn"]
        self.V = dict_WH["WithdrawRate"]
        self.tao = dict_WH["TimeConstant"]
        self.Prate = dict_WH["RatePower"]
        self.I = dict_WH["ControlInterval"]
        self.ID = dict_WH["ID"]

    def thermalModel(self, T_before, Z, V):

        u = self.Prate * Z / self.Eff
        d = self.C * self.WD * self.L * (T_before-self.Tambient) / self.tao
        w = self.C * self.WD * (V/60) * (T_before-self.Tinlet)

        T_after = T_before + self.I * (u-d-w) / (self.C * self.WD * self.L)

        return T_after

    def updateRecord(self, k, T, Z):
        self.T[k] = T
        self.Z[k] = Z
        self.S[k] = self.state_5(T)

    def TS_baseline_run(self, duration):

        self.T = [0] * duration
        self.Z = [0] * duration
        self.S = [0] * duration

        k = 0
        T = self.Tinit
        Z = 0
        while k < duration:
            if T >= self.Tupper:
                Z = 0
            if T <= self.Tlower:
                Z = 1
            self.updateRecord(k, T, Z)
            T = self.thermalModel(T, Z, self.V[k])
            k += 1

    def TS_dlc_run(self, duration, starttime, endtime):

        self.T = [0] * duration
        self.Z = [0] * duration
        self.S = [0] * duration

        k = 0
        T = self.Tinit
        Z = 0
        while k < duration:
            if k < starttime or k > endtime:
                if T >= self.Tupper:
                    Z = 0
                if T <= self.Tlower:
                    Z = 1
            else:
                Z = 0
            self.updateRecord(k, T, Z)
            T = self.thermalModel(T, Z, self.V[k])
            k += 1

    def reset(self, duration):

        self.T = [0] * duration
        self.Z = [0] * duration
        self.S = [0] * duration

        self.T_RT = self.Tinit
        self.k = 0
        self.S_RT = self.state_5(self.T_RT)

    def sending_state(self):
        return self.S_RT

    def sending_soc(self):
        return (self.T_RT-self.Tlower)*100/(self.Tupper-self.Tlower)

    def sending_T(self):
        return self.T_RT

    def singlestep_control(self, Z, k):
        self.updateRecord(k, self.T_RT, Z)
        self.T_RT = self.thermalModel(self.T_RT, Z, self.V[k])
        self.S_RT = self.state_5(self.T_RT)

    def singlestep_dlc(self, k):
        self.updateRecord(k, self.T_RT, 0)
        self.T_RT = self.thermalModel(self.T_RT, 0, self.V[k])
        self.S_RT = self.state_5(self.T_RT)

    def experience(self):

        experience_single = {'State': self.S,
                             'Action': self.Z,
                             'Disturbance': np.sign(self.V)}
        return experience_single

    # def state_5(self, T):
    #
    #     delta = self.Tupper - self.Tlower - 1
    #     if T >= self.Tupper - 0.5:
    #         S = "MustOFF"  # must off
    #     elif T < self.Tlower + 0.5:
    #         S = "MustON"  # must on
    #     elif (self.Tupper - 0.5 - delta / 5) <= T < self.Tupper - 0.5:
    #         S = "FLEX1"
    #     elif (self.Tupper - 0.5 - 2 * delta / 5) <= T < (self.Tupper - 0.5 - delta / 5):
    #         S = "FLEX2"
    #     elif (self.Tlower + 0.5 + 2 * delta / 5) <= T < (self.Tupper - 0.5 - 2 * delta / 5):
    #         S = "FLEX3"
    #     elif (self.Tlower + 0.5 + delta / 5) <= T < (self.Tlower + 0.5 + 2 * delta / 5):
    #         S = "FLEX4"
    #     elif self.Tlower + 0.5 <= T < (self.Tlower + 0.5 + delta / 5):
    #         S = "FLEX5"
    #
    #     return S

    def state_5(self, T):

        delta = self.Tupper - self.Tlower
        if T >= self.Tupper:
            S = "MustOFF"  # must off
        elif T <= self.Tlower:
            S = "MustON"  # must on
        elif (self.Tupper - delta / 5) <= T < self.Tupper:
            S = "FLEX1"
        elif (self.Tupper - 2 * delta / 5) <= T < (self.Tupper - delta / 5):
            S = "FLEX2"
        elif (self.Tlower + 2 * delta / 5) <= T < (self.Tupper - 2 * delta / 5):
            S = "FLEX3"
        elif (self.Tlower + delta / 5) <= T < (self.Tlower + 2 * delta / 5):
            S = "FLEX4"
        elif self.Tlower < T < (self.Tlower + delta / 5):
            S = "FLEX5"

        return S

    def state_3(self, T):

        delta = self.Tupper - self.Tlower
        if T >= self.Tupper:
            S = "MustOFF"  # must off
        elif T < self.Tlower:
            S = "MustON"  # must on
        elif (self.Tupper - delta / 3) <= T < self.Tupper:
            S = "FLEX1"
        elif (self.Tlower + delta / 3) <= T < (self.Tupper - delta / 3):
            S = "FLEX2"
        elif self.Tlower <= T < (self.Tlower + delta / 3):
            S = "FLEX3"

        return S

    def state_1(self, T):

        if T >= self.Tupper:
            S = "MustOFF"  # must off
        elif T < self.Tlower:
            S = "MustON"  # must on
        elif self.Tlower <= T < self.Tupper:
            S = "FLEX"

        return S


if __name__ == "__main__":
    pass
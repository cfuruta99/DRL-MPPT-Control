import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def smooth_data(data, window_size=15):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

class MPPTEnv(gym.Env):
    def __init__(self, num_modules=1):
        super(MPPTEnv, self).__init__()
        print("=== MPPTEnv Initialization Start ===")
        self.voltage_gradual = 0.0
        self.voltage_step = 0.0
        self.voltage_new = 0.0
        self.previous_voltage_gradual = self.voltage_gradual
        self.irradiance = 0.0
        self.temperature = 25.0
        self.max_voltage = 32.0
        self.min_voltage = 0.0
        self.env_step_count = 0
        self.num_modules = num_modules
        self.final_irradiance_600 = 600
        self.final_irradiance_1000 = 1000
        self.current_phase = "ramp_up"
        self.irradiance_log_gradual = []
        self.irradiance_log_step = []
        self.irradiance_log_new = []
        self.power_log = []
        self.voltage_log = []
        self.current_log = []
        self.reward_log_gradual = []
        self.reward_log_step = []
        self.reward_log_new = []
        self.temperature_log = []
        self.power_log_step = []
        self.voltage_log_step = []
        self.power_log_new = []
        self.voltage_log_new = []
        self.module_area = 1.6
        self.efficiency = 0.18

        # <<<=== TEMP EFFECT ===>>>
        # Temperature coefficient for power: -0.4 % per °C above 25°C (typical for c-Si)
        self.temp_coeff_power = -0.004  # per °C
        # <<<=== END ===>>>

        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([25.0, 0.0, 0.0]),
            high=np.array([self.max_voltage, self.final_irradiance_1000, 50.0]),
            dtype=np.float32
        )
        self.step_data = {1: [], 20: [], 100: [], 1001: [], 2000: []}
        self.previous_power_gradual = 0.0

        # Debug current working directory
        current_dir = os.getcwd()
        print("Current working directory:", current_dir)
        # CSV paths
        csv_path_gradual = "C:/Users/lab/Desktop/DRL MPPT/po_mppt_data_gradual.csv"
        csv_path_step = "C:/Users/lab/Desktop/DRL MPPT/po_mppt_data_step.csv"
        print(f"Attempting to load CSV from: {csv_path_gradual} and {csv_path_step}")

        # Load P&O data from both CSV files with fallback for old structure
        self.po_data_gradual = None
        self.po_data_step = None
        try:
            if not os.path.exists(csv_path_gradual):
                error_msg = f"File not found at {csv_path_gradual}"
                print(error_msg)
                raise FileNotFoundError(error_msg)
            if not os.path.exists(csv_path_step):
                error_msg = f"File not found at {csv_path_step}"
                print(error_msg)
                raise FileNotFoundError(error_msg)
            print("Files exist. Attempting to read CSV...")
            self.po_data_gradual = pd.read_csv(csv_path_gradual, encoding='utf-8')
            self.po_data_step = pd.read_csv(csv_path_step, encoding='utf-8')
            print("P&O data loaded successfully from:", csv_path_gradual, "and", csv_path_step)
            print("P&O data columns (gradual raw):", self.po_data_gradual.columns.tolist())
            print("P&O data columns (step raw):", self.po_data_step.columns.tolist())
            self.po_data_gradual.columns = [col.strip().lower() for col in self.po_data_gradual.columns]
            self.po_data_step.columns = [col.strip().lower() for col in self.po_data_step.columns]
            print("P&O data columns (gradual processed):", self.po_data_gradual.columns.tolist())
            print("P&O data columns (step processed):", self.po_data_step.columns.tolist())

            # Check for new or old column names
            if all(col in self.po_data_gradual.columns for col in ['power_po_gradual', 'voltage_po_gradual']):
                self.po_data_gradual = self.po_data_gradual[['power_po_gradual', 'voltage_po_gradual']]
            elif all(col in self.po_data_gradual.columns for col in ['power_po', 'voltage_po']):
                self.po_data_gradual.columns = ['power_po_gradual', 'voltage_po_gradual']
            else:
                error_msg = f"Missing required columns in {csv_path_gradual}. Found: {self.po_data_gradual.columns.tolist()}"
                print(error_msg)
                raise ValueError(error_msg)

            if all(col in self.po_data_step.columns for col in ['power_po_step', 'voltage_po_step']):
                self.po_data_step = self.po_data_step[['power_po_step', 'voltage_po_step']]
            elif all(col in self.po_data_step.columns for col in ['power_po', 'voltage_po']):
                self.po_data_step.columns = ['power_po_step', 'voltage_po_step']
            else:
                error_msg = f"Missing required columns in {csv_path_step}. Found: {self.po_data_step.columns.tolist()}"
                print(error_msg)
                raise ValueError(error_msg)

            print("Validating data types...")
            self.po_data_gradual = self.po_data_gradual.astype({'power_po_gradual': float, 'voltage_po_gradual': float})
            self.po_data_step = self.po_data_step.astype({'power_po_step': float, 'voltage_po_step': float})

            if self.po_data_gradual[['power_po_gradual', 'voltage_po_gradual']].isna().any().any() or \
               self.po_data_step[['power_po_step', 'voltage_po_step']].isna().any().any():
                error_msg = "NaN values found in power or voltage columns"
                print(error_msg)
                raise ValueError(error_msg)

            print("P&O data length (gradual):", len(self.po_data_gradual))
            print("P&O data length (step):", len(self.po_data_step))
            print("Sample P&O power_po_gradual values (first 5):", self.po_data_gradual['power_po_gradual'].head().tolist())
            print("Sample P&O voltage_po_gradual values (first 5):", self.po_data_gradual['voltage_po_gradual'].head().tolist())
            print("Sample P&O power_po_step values (first 5):", self.po_data_step['power_po_step'].head().tolist())
            print("Sample P&O voltage_po_step values (first 5):", self.po_data_step['voltage_po_step'].head().tolist())
            print("Sample P&O power_po_gradual values (last 5):", self.po_data_gradual['power_po_gradual'].tail().tolist())
            print("Sample P&O voltage_po_gradual values (last 5):", self.po_data_gradual['voltage_po_gradual'].tail().tolist())
            print("Sample P&O power_po_step values (last 5):", self.po_data_step['power_po_step'].tail().tolist())
            print("Sample P&O voltage_po_step values (last 5):", self.po_data_step['voltage_po_step'].tail().tolist())
        except FileNotFoundError as e:
            print(f"FileNotFoundError: {e}")
            raise
        except ValueError as ve:
            print(f"ValueError: {ve}")
            raise
        except pd.errors.ParserError as pe:
            print(f"ParserError: Failed to parse CSV file: {pe}")
            raise
        except Exception as e:
            print(f"Unexpected error loading po_mppt_data.csv: {str(e)}")
            raise

        if self.po_data_gradual is None or self.po_data_step is None:
            error_msg = "Failed to load po_mppt_data_gradual.csv or po_mppt_data_step.csv."
            print(error_msg)
            raise RuntimeError(error_msg)
        print("=== MPPTEnv Initialization End ===")

    # <<<=== TEMP EFFECT ===>>>
    def _temp_scaling_factor(self):
        """
        Returns power scaling factor:
        - 1.0 at 25°C
        - < 1.0 if T > 25°C
        - > 1.0 if T < 25°C (bonus for cooler operation)
        """
        return 1.0 + self.temp_coeff_power * (self.temperature - 25.0)
    # <<<=== END ===>>>

    def get_pv_curve(self, irradiance, model_type='drl'):
        """Generate the theoretical PV curve for a given irradiance and model type."""
        voc = self.get_open_circuit_voltage(irradiance)
        voltages = np.linspace(0, voc, 100)
        powers = []

        if model_type == 'po':
            vmp = 26.0
            imp = 285.0 / vmp
        else:
            vmp = 27.1
            imp = 288.0 / vmp

        for v in voltages:
            isc = self.get_short_circuit_current(irradiance)
            if v <= vmp:
                current = isc * (1 - (v / vmp) ** 2 * 0.05)
            else:
                current = imp * (vmp / v) ** 1.5
            current = np.clip(current, 0, isc)
            power = v * current * self.num_modules

            # <<<=== TEMP EFFECT ===>>>
            power *= self._temp_scaling_factor()
            # <<<=== END ===>>>

            if model_type == 'drl' and power > 288:
                power = 288
            powers.append(power)

        return voltages, powers

    def plot_irradiance(self):
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(self.irradiance_log_gradual)), self.irradiance_log_gradual, 'b-', label='Gradual Irradiance')
        plt.plot(range(len(self.irradiance_log_step)), self.irradiance_log_step, 'r--', label='Step Irradiance')
        plt.plot(range(len(self.irradiance_log_new)), self.irradiance_log_new, 'c-.', label='Reverse Ramp Irradiance')
        plt.xlabel("Time (s)")
        plt.ylabel("Irradiance (W/m²)")
        plt.title("Irradiance vs Time (Gradual vs Step vs Reverse Ramp)")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_power_vs_time(self):
        max_steps = 20000
        theoretical_power_gradual = [288 * (irr / 1000) for irr in self.irradiance_log_gradual]
        theoretical_power_step = [288 * (irr / 1000) for irr in self.irradiance_log_step]
        theoretical_power_new = [288 * (irr / 1000) for irr in self.irradiance_log_new]

        # Gradual Irradiance Graph
        plt.figure(figsize=(8, 6))
        plt.plot(range(max_steps), self.power_log[:max_steps], 'b-', label='DRL Power (Gradual)', alpha=0.8)
        plt.plot(range(max_steps), theoretical_power_gradual[:max_steps], 'm--', label='Theoretical Power (Gradual)', alpha=0.8)
        if self.po_data_gradual is not None and 'power_po_gradual' in self.po_data_gradual.columns and len(self.po_data_gradual) > 0:
            po_power_gradual = self.po_data_gradual['power_po_gradual'].values
            po_steps_gradual = min(len(po_power_gradual), max_steps)
            plt.plot(range(po_steps_gradual), po_power_gradual[:po_steps_gradual], 'r-.', label='P&O Power (Gradual)', alpha=0.8)
        plt.xlabel("Time (s)")
        plt.ylabel("Power Output (W)")
        plt.title("Power Output vs Time (Gradual Irradiance)")
        plt.grid()
        plt.legend()
        plt.ylim(0, 300)
        plt.show()

        # Step Irradiance Graph
        plt.figure(figsize=(8, 6))
        plt.plot(range(max_steps), self.power_log_step[:max_steps], 'g-', label='DRL Power (Step)', alpha=0.8)
        plt.plot(range(max_steps), theoretical_power_step[:max_steps], 'y--', label='Theoretical Power (Step)', alpha=0.8)
        if self.po_data_step is not None and 'power_po_step' in self.po_data_step.columns and len(self.po_data_step) > 0:
            po_power_step = self.po_data_step['power_po_step'].values
            po_steps_step = min(len(po_power_step), max_steps)
            plt.plot(range(po_steps_step), po_power_step[:po_steps_step], 'r-.', label='P&O Power (Step)', alpha=0.8)
        plt.xlabel("Time (s)")
        plt.ylabel("Power Output (W)")
        plt.title("Power Output vs Time (Step Irradiance)")
        plt.grid()
        plt.legend()
        plt.ylim(0, 300)
        plt.show()

        # Reverse Ramp Irradiance Graph
        plt.figure(figsize=(8, 6))
        plt.plot(range(max_steps), self.power_log_new[:max_steps], 'c-', label='DRL Power (Reverse Ramp)', alpha=0.8)
        plt.plot(range(max_steps), theoretical_power_new[:max_steps], 'k--', label='Theoretical Power (Reverse Ramp)', alpha=0.8)
        plt.xlabel("Time (s)")
        plt.ylabel("Power Output (W)")
        plt.title("Power Output vs Time (Reverse Ramp Irradiance)")
        plt.grid()
        plt.legend()
        plt.ylim(0, 300)
        plt.show()

    def plot_voltage_vs_time(self):
        theoretical_voltage_gradual = []
        theoretical_voltage_step = []
        theoretical_voltage_new = []
        for irr_gradual, irr_step, irr_new in zip(self.irradiance_log_gradual, self.irradiance_log_step, self.irradiance_log_new):
            voc_gradual = self.get_open_circuit_voltage(irr_gradual)
            voc_step = self.get_open_circuit_voltage(irr_step)
            voc_new = self.get_open_circuit_voltage(irr_new)
            theoretical_voltage_gradual.append(0.7 * voc_gradual)
            theoretical_voltage_step.append(0.7 * voc_step)
            theoretical_voltage_new.append(0.7 * voc_new)
        max_steps = min(len(self.voltage_log), len(self.voltage_log_step), len(self.voltage_log_new), 20000)

        # Gradual
        plt.figure(figsize=(8, 6))
        plt.plot(range(max_steps), self.voltage_log[:max_steps], 'b-', label='DRL Voltage (Gradual)', alpha=0.8)
        plt.plot(range(max_steps), theoretical_voltage_gradual[:max_steps], 'm--', label='Theoretical Voltage (Gradual)', alpha=0.8)
        if self.po_data_gradual is not None and 'voltage_po_gradual' in self.po_data_gradual.columns and len(self.po_data_gradual) > 0:
            po_voltage_gradual = self.po_data_gradual['voltage_po_gradual'].values
            po_steps_gradual = min(len(po_voltage_gradual), max_steps)
            plt.plot(range(po_steps_gradual), po_voltage_gradual[:po_steps_gradual], 'r-.', label='P&O Voltage (Gradual)', alpha=0.8)
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title("Voltage vs Time (Gradual Irradiance)")
        plt.grid()
        plt.legend()
        plt.show()

        # Step
        plt.figure(figsize=(8, 6))
        plt.plot(range(max_steps), self.voltage_log_step[:max_steps], 'g-', label='DRL Voltage (Step)', alpha=0.8)
        plt.plot(range(max_steps), theoretical_voltage_step[:max_steps], 'y--', label='Theoretical Voltage (Step)', alpha=0.8)
        if self.po_data_step is not None and 'voltage_po_step' in self.po_data_step.columns and len(self.po_data_step) > 0:
            po_voltage_step = self.po_data_step['voltage_po_step'].values
            po_steps_step = min(len(po_voltage_step), max_steps)
            plt.plot(range(po_steps_step), po_voltage_step[:po_steps_step], 'r-.', label='P&O Voltage (Step)', alpha=0.8)
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title("Voltage vs Time (Step Irradiance)")
        plt.grid()
        plt.legend()
        plt.show()

        # Reverse Ramp
        plt.figure(figsize=(8, 6))
        plt.plot(range(max_steps), self.voltage_log_new[:max_steps], 'c-', label='DRL Voltage (Reverse Ramp)', alpha=0.8)
        plt.plot(range(max_steps), theoretical_voltage_new[:max_steps], 'k--', label='Theoretical Voltage (Reverse Ramp)', alpha=0.8)
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title("Voltage vs Time (Reverse Ramp Irradiance)")
        plt.grid()
        plt.legend()
        plt.show()

    def plot_power_vs_voltage(self):
        # Gradual
        gradual_irradiance = self.irradiance_log_gradual[-1] if self.irradiance_log_gradual else 1000
        drl_voltages_gradual, drl_powers_gradual = self.get_pv_curve(irradiance=gradual_irradiance, model_type='drl')
        po_voltages_gradual, po_powers_gradual = self.get_pv_curve(irradiance=gradual_irradiance, model_type='po')
        plt.figure(figsize=(8, 6))
        plt.plot(drl_voltages_gradual, drl_powers_gradual, 'b-', label='DRL (Gradual Irradiance)', alpha=0.8)
        drl_max_power_gradual = max(drl_powers_gradual)
        drl_max_idx_gradual = drl_powers_gradual.index(drl_max_power_gradual)
        drl_max_voltage_gradual = drl_voltages_gradual[drl_max_idx_gradual]
        plt.plot(drl_max_voltage_gradual, drl_max_power_gradual, 'bo', label=f'DRL MPP (Gradual, {drl_max_power_gradual:.1f}W at {drl_max_voltage_gradual:.1f}V)')
        plt.plot(po_voltages_gradual, po_powers_gradual, 'r-', label='P&O (Gradual Irradiance)', alpha=0.8, linestyle='-.')
        po_max_power_gradual = max(po_powers_gradual)
        po_max_idx_gradual = po_powers_gradual.index(po_max_power_gradual)
        po_max_voltage_gradual = po_voltages_gradual[po_max_idx_gradual]
        plt.plot(po_max_voltage_gradual, po_max_power_gradual, 'ro', label=f'P&O MPP (Gradual, {po_max_power_gradual:.1f}W at {po_max_voltage_gradual:.1f}V)')
        plt.xlabel("Voltage (V)")
        plt.ylabel("Power Output (W)")
        plt.title("Power Output vs Voltage (Gradual Irradiance)")
        plt.grid()
        plt.legend()
        plt.show()

        # Step
        step_irradiance = self.irradiance_log_step[-1] if self.irradiance_log_step else 1000
        drl_voltages_step, drl_powers_step = self.get_pv_curve(irradiance=step_irradiance, model_type='drl')
        po_voltages_step, po_powers_step = self.get_pv_curve(irradiance=step_irradiance, model_type='po')
        plt.figure(figsize=(8, 6))
        plt.plot(drl_voltages_step, drl_powers_step, 'g-', label='DRL (Step Irradiance)', alpha=0.8)
        drl_max_power_step = max(drl_powers_step)
        drl_max_idx_step = drl_powers_step.index(drl_max_power_step)
        drl_max_voltage_step = drl_voltages_step[drl_max_idx_step]
        plt.plot(drl_max_voltage_step, drl_max_power_step, 'go', label=f'DRL MPP (Step, {drl_max_power_step:.1f}W at {drl_max_voltage_step:.1f}V)')
        plt.plot(po_voltages_step, po_powers_step, 'r-', label='P&O (Step Irradiance)', alpha=0.8, linestyle='-.')
        po_max_power_step = max(po_powers_step)
       
        po_max_idx_step = po_powers_step.index(po_max_power_step)
        po_max_voltage_step = po_voltages_step[po_max_idx_step]
        plt.plot(po_max_voltage_step, po_max_power_step, 'ro', label=f'P&O MPP (Step, {po_max_power_step:.1f}W at {po_max_voltage_step:.1f}V)')
        plt.xlabel("Voltage (V)")
        plt.ylabel("Power Output (W)")
        plt.title("Power Output vs Voltage (Step Irradiance)")
        plt.grid()
        plt.legend()
        plt.show()

        # Reverse Ramp
        new_irradiance = max(self.irradiance_log_new) if self.irradiance_log_new else 1000
        drl_voltages_new, drl_powers_new = self.get_pv_curve(irradiance=new_irradiance, model_type='drl')
        po_voltages_new, po_powers_new = self.get_pv_curve(irradiance=new_irradiance, model_type='po')
        plt.figure(figsize=(8, 6))
        plt.plot(drl_voltages_new, drl_powers_new, 'c-', label='DRL (Reverse Ramp Irradiance)', alpha=0.8)
        drl_max_power_new = max(drl_powers_new)
        drl_max_idx_new = drl_powers_new.index(drl_max_power_new)
        drl_max_voltage_new = drl_voltages_new[drl_max_idx_new]
        plt.plot(drl_max_voltage_new, drl_max_power_new, 'co', label=f'DRL MPP (Reverse Ramp, {drl_max_power_new:.1f}W at {drl_max_voltage_new:.1f}V)')
        plt.plot(po_voltages_new, po_powers_new, 'r-', label='P&O (Reverse Ramp Irradiance)', alpha=0.8, linestyle='-.')
        po_max_power_new = max(po_powers_new)
        po_max_idx_new = po_powers_new.index(po_max_power_new)
        po_max_voltage_new = po_voltages_new[po_max_idx_new]
        plt.plot(po_max_voltage_new, po_max_power_new, 'ro', label=f'P&O MPP (Reverse Ramp, {po_max_power_new:.1f}W at {po_max_voltage_new:.1f}V)')
        plt.xlabel("Voltage (V)")
        plt.ylabel("Power Output (W)")
        plt.title("Power Output vs Voltage (Reverse Ramp Irradiance)")
        plt.grid()
        plt.legend()
        plt.show()

    def plot_temperature_vs_time(self):
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(self.temperature_log)), self.temperature_log)
        plt.xlabel("Time (s)")
        plt.ylabel("Temperature (°C)")
        plt.title("Temperature Change vs Time")
        plt.grid()
        plt.show()

    def plot_reward_vs_time(self):
        max_steps = min(len(self.reward_log_gradual), len(self.reward_log_step), len(self.reward_log_new), 20000)
        plt.figure(figsize=(8, 6))
        plt.plot(range(max_steps), self.reward_log_gradual[:max_steps], 'b-', label='Reward (Gradual Irradiance)', alpha=0.8)
        plt.plot(range(max_steps), self.reward_log_step[:max_steps], 'g-', label='Reward (Step Irradiance)', alpha=0.8)
        plt.plot(range(max_steps), self.reward_log_new[:max_steps], 'c-.', label='Reward (Reverse Ramp Irradiance)', alpha=0.8)
        plt.xlabel("Time (s)")
        plt.ylabel("Reward")
        plt.title("Reward vs Time")
        plt.grid()
        plt.legend()
        plt.ylim(0, 350)
        plt.show()

    def get_open_circuit_voltage(self, irradiance):
        Voc_stc = 37.0
        beta = -0.002
        alpha = 0.05
        temperature_factor = 1 + beta * (self.temperature - 25)
        irradiance_factor = 1 + alpha * (irradiance / 1000)
        return Voc_stc * temperature_factor * irradiance_factor

    def get_short_circuit_current(self, irradiance):
        Isc_stc = 10.5
        isc_coefficient = 0.1
        return Isc_stc * (irradiance / 1000) * (1 + isc_coefficient * (irradiance / 1000))

    def get_reward(self, voltage, power, voltage_change, irradiance):
        oc_voltage = self.get_open_circuit_voltage(irradiance)
        mppt_voltage = 0.7 * oc_voltage
        max_power_theoretical = 288 * (irradiance / 1000) if irradiance > 0 else 0
        if irradiance <= 0:
            reward = 0.0
        else:
            power_reward = 300 * power / (max_power_theoretical + 1e-6)
            mpp_bonus = 30 if 0.95 * max_power_theoretical <= power <= 1.05 * max_power_theoretical else 0
            voltage_deviation = abs(voltage - mppt_voltage) / mppt_voltage
            voltage_penalty = -30 * voltage_deviation
            over_power_penalty = -50 * max(0, power - 1.05 * max_power_theoretical) / max_power_theoretical
            low_power_penalty = -1 if power < 0.5 * max_power_theoretical else 0
            voltage_change_penalty = -20 * abs(voltage_change) / mppt_voltage
            reward = power_reward + mpp_bonus + voltage_penalty + over_power_penalty + low_power_penalty + voltage_change_penalty
        return reward

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
        print(f"Environment resetting at step {self.env_step_count}")
        self.voltage_gradual = 25.0
        self.voltage_step = 25.0
        self.voltage_new = 25.0
        self.previous_voltage_gradual = self.voltage_gradual
        self.irradiance = 0.0
        self.env_step_count = 0
        self.current_phase = "ramp_up"
        self.irradiance_log_gradual = []
        self.irradiance_log_step = []
        self.irradiance_log_new = []
        self.power_log = []
        self.voltage_log = []
        self.current_log = []
        self.reward_log_gradual = []
        self.reward_log_step = []
        self.reward_log_new = []
        self.temperature_log = []
        self.power_log_step = []
        self.voltage_log_step = []
        self.power_log_new = []
        self.voltage_log_new = []
        self.step_data = {1: [], 20: [], 100: [], 1001: [], 2000: []}
        self.previous_power_gradual = 0.0
        obs = np.array([self.voltage_gradual, self.irradiance, self.temperature], dtype=np.float32)
        info = {}
        return obs, info

    def step(self, action):
        self.env_step_count += 1
        # Gradual profile
        if self.env_step_count <= 100:
            self.irradiance = (self.env_step_count / 100) * self.final_irradiance_1000
            self.current_phase = "ramp_up"
        elif 100 < self.env_step_count <= 8000:
            self.irradiance = self.final_irradiance_1000
            self.current_phase = "steady_high"
        elif 8000 < self.env_step_count <= 13000:
            self.irradiance = self.final_irradiance_600
            self.current_phase = "cloud_cover"
        elif 13000 < self.env_step_count <= 18000:
            self.irradiance = self.final_irradiance_600 + (
                (self.env_step_count - 13000) / 5000
            ) * (self.final_irradiance_1000 - self.final_irradiance_600)
            self.current_phase = "ramp_up_again"
        else:
            self.irradiance = self.final_irradiance_1000
            self.current_phase = "final_steady"

        # Step profile
        if self.env_step_count >= 13000:
            irradiance_step = self.final_irradiance_1000
        else:
            irradiance_step = self.irradiance
        self.irradiance_log_step.append(irradiance_step)

        # Reverse Ramp profile
        if self.env_step_count <= 100:
            self.irradiance_new = (self.env_step_count / 100) * self.final_irradiance_600
        elif 100 < self.env_step_count <= 8000:
            self.irradiance_new = self.final_irradiance_600
        elif 8000 < self.env_step_count <= 13000:
            self.irradiance_new = self.final_irradiance_1000
        elif 13000 < self.env_step_count <= 18000:
            self.irradiance_new = self.final_irradiance_1000 - (
                (self.env_step_count - 13000) / 5000
            ) * (self.final_irradiance_1000 - self.final_irradiance_600)
        else:
            self.irradiance_new = self.final_irradiance_600
        self.irradiance_log_new.append(self.irradiance_new)
        self.irradiance_log_gradual.append(self.irradiance)

        voltage_step = 0.05
        voltage_change = action[0] * voltage_step
        print(f"Step {self.env_step_count} | Action: {action[0]:.3f}")

        # Calculate power and voltage for gradual profile
        oc_voltage_gradual = self.get_open_circuit_voltage(self.irradiance)
        mppt_voltage_gradual = 0.7 * oc_voltage_gradual
        self.voltage_gradual += voltage_change
        self.voltage_gradual = np.clip(self.voltage_gradual, self.min_voltage, mppt_voltage_gradual)
        power_gradual, current_gradual, _ = self.calculate_power(self.voltage_gradual, self.irradiance)
        max_power_theoretical_gradual = 288 * (self.irradiance / 1000)
        power_gradual = min(power_gradual, max_power_theoretical_gradual)

        # Step profile
        oc_voltage_step = self.get_open_circuit_voltage(irradiance_step)
        mppt_voltage_step = 0.7 * oc_voltage_step
        self.voltage_step += voltage_change
        self.voltage_step = np.clip(self.voltage_step, self.min_voltage, mppt_voltage_step)
        power_step, _, _ = self.calculate_power(self.voltage_step, irradiance_step)
        max_power_step_theoretical = 288 * (irradiance_step / 1000)
        power_step = min(power_step, max_power_step_theoretical)

        # Reverse Ramp profile
        oc_voltage_new = self.get_open_circuit_voltage(self.irradiance_new)
        mppt_voltage_new = 0.7 * oc_voltage_new
        self.voltage_new += voltage_change
        self.voltage_new = np.clip(self.voltage_new, self.min_voltage, mppt_voltage_new)
        power_new, _, _ = self.calculate_power(self.voltage_new, self.irradiance_new)
        max_power_new_theoretical = 288 * (self.irradiance_new / 1000)
        power_new = min(power_new, max_power_new_theoretical)

        if self.env_step_count in self.step_data:
            self.step_data[self.env_step_count].append((self.voltage_gradual, current_gradual, power_gradual / self.num_modules))

        print(f"Step {self.env_step_count} | Irradiance (Gradual): {self.irradiance}, Voc (Gradual): {oc_voltage_gradual}, Isc (Gradual): {self.get_short_circuit_current(self.irradiance)}, Power (Gradual): {power_gradual}, Voltage (Gradual): {self.voltage_gradual}")
        print(f"Step {self.env_step_count} | Irradiance (Step): {irradiance_step}, Voc (Step): {oc_voltage_step}, Isc (Step): {self.get_short_circuit_current(irradiance_step)}, Power (Step): {power_step}, Voltage (Step): {self.voltage_step}")
        print(f"Step {self.env_step_count} | Irradiance (Reverse Ramp): {self.irradiance_new}, Voc (Reverse Ramp): {oc_voltage_new}, Isc (Reverse Ramp): {self.get_short_circuit_current(self.irradiance_new)}, Power (Reverse Ramp): {power_new}, Voltage (Reverse Ramp): {self.voltage_new}")

        reward_gradual = self.get_reward(self.voltage_gradual, power_gradual, voltage_change, self.irradiance)
        reward_step = self.get_reward(self.voltage_step, power_step, voltage_change, irradiance_step)
        reward_new = self.get_reward(self.voltage_new, power_new, voltage_change, self.irradiance_new)

        self.previous_voltage_gradual = self.voltage_gradual
        self.previous_power_gradual = power_gradual
        self.voltage_log.append(self.voltage_gradual)
        self.power_log.append(power_gradual)
        self.voltage_log_step.append(self.voltage_step)
        self.power_log_step.append(power_step)
        self.voltage_log_new.append(self.voltage_new)
        self.power_log_new.append(power_new)
        self.current_log.append(current_gradual)
        self.temperature_log.append(self.temperature)
        self.reward_log_gradual.append(reward_gradual)
        self.reward_log_step.append(reward_step)
        self.reward_log_new.append(reward_new)

        info = {"power_output": power_gradual, "phase": self.current_phase}
        terminated = False
        truncated = False
        obs = np.array([self.voltage_gradual, self.irradiance, self.temperature], dtype=np.float32)
        return obs, reward_gradual, terminated, truncated, info

    def calculate_power(self, voltage, irradiance):
        isc = self.get_short_circuit_current(irradiance)
        voc = self.get_open_circuit_voltage(irradiance)
        vmp = 0.7 * voc
        imp = 288 / vmp

        if voltage <= vmp:
            current = isc * (1 - (voltage / vmp) ** 2 * 0.05)
        else:
            current = imp * (vmp / voltage) ** 1.5
        current = np.clip(current, 0, isc)

        if voltage * current > 288:
            current = min(current, 288 / voltage)

        power_per_module = voltage * current

        # <<<=== TEMP EFFECT ===>>>
        power_per_module *= self._temp_scaling_factor()
        # <<<=== END ===>>>

        print(f"Voltage: {voltage}, Current: {current}, Power per module: {power_per_module}, Irradiance: {irradiance}")
        return power_per_module * self.num_modules, current, power_per_module

    def print_step_summary(self):
        print("\nStep Summary:")
        for step, data_list in self.step_data.items():
            if data_list:
                for voltage, current, power in data_list:
                    print(f"Step {step}: Voltage: {voltage:.2f}, Current: {current:.2f}, Power per module: {power:.2f}")

# ==============================================================================
# Compatibility wrapper for Stable-Baselines3
# ==============================================================================
class SB3CompatibleMPPTEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        self.env = MPPTEnv(*args, **kwargs)
    @property
    def observation_space(self):
        return self.env.observation_space
    @property
    def action_space(self):
        return self.env.action_space
    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return self.env.reset(seed=seed)
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info
    def render(self, mode='human'):
        return self.env.render(mode)
    def plot_irradiance(self):
        return self.env.plot_irradiance()
    def plot_power_vs_time(self):
        return self.env.plot_power_vs_time()
    def plot_voltage_vs_time(self):
        return self.env.plot_voltage_vs_time()
    def plot_power_vs_voltage(self):
        return self.env.plot_power_vs_voltage()
    def plot_temperature_vs_time(self):
        return self.env.plot_temperature_vs_time()
    def plot_reward_vs_time(self):
        return self.env.plot_reward_vs_time()
    def print_step_summary(self):
        return self.env.print_step_summary()

register(id="MPPTEnv-v0", entry_point="mppt_env:MPPTEnv")

def make_env():
    return SB3CompatibleMPPTEnv()

# ==============================================================================
# Main Simulation – ONLY TEMPERATURE TESTS (no 20 000-step run)
# ==============================================================================
print("=== Main Simulation Start ===")
env = SB3CompatibleMPPTEnv()
obs, info = env.reset()
print(f"Initial observation: {obs}, Info: {info}")

# --------------------------------------------------------------
# 1. Load or train the PPO model (must exist before the tests)
# --------------------------------------------------------------
try:
    model = PPO.load("trained_mppt_model", env=env)
    print("Loaded existing trained model.")
except Exception:
    print("No pre-trained model found – training a quick one for the demo.")
    policy_kwargs = dict(log_std_init=-0.5)
    model = PPO(
        "MlpPolicy",
        env,
        ent_coef=0.02,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log="./ppo_log",
        learning_rate=1e-4,
        n_steps=2048,
    )
    model.learn(total_timesteps=2000)      # short training, enough for the test

# --------------------------------------------------------------
# 2. TEST 1 – Hot panel (45 °C)
# --------------------------------------------------------------
print("\n=== TEST 1: Hot Panel (45 °C) ===")
env.env.temperature = 45.0
obs, _ = env.reset()
hot_powers = []
for _ in range(200):                     # short run, just to see the effect
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    hot_powers.append(info.get('power_output', 0.0))
print(f"Hot-panel max power (gradual profile): {max(hot_powers):.2f} W")

# --------------------------------------------------------------
# 3. TEST 2 – STC (25 °C)
# --------------------------------------------------------------
print("\n=== TEST 2: STC (25 °C) ===")
env.env.temperature = 25.0
obs, _ = env.reset()  
stc_powers = []
for _ in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    stc_powers.append(info.get('power_output', 0.0))
print(f"STC max power (gradual profile): {max(stc_powers):.2f} W")

# --------------------------------------------------------------
# Plot the two short runs for visual confirmation
# --------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(hot_powers, label="45 °C (hot)", alpha=0.8)
plt.plot(stc_powers, label="25 °C (STC)", alpha=0.8)
plt.xlabel("Step")
plt.ylabel("Power (W)")
plt.title("Temperature Effect – 200-step test")
plt.legend()
plt.grid(True)
plt.show()


print("\n=== Temperature-test demo finished ===")


obs, info = env.reset()
voltages, irradiances_gradual, irradiances_step, irradiances_new, temperatures, actual_powers, rewards_list = [], [], [], [], [], [], []
steps = 20000
for step in range(steps):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step}: Irradiance recorded (Gradual) -> {env.env.irradiance_log_gradual[-1]}")
    print(f"Step {step}: Irradiance recorded (Step) -> {env.env.irradiance_log_step[-1]}")
    print(f"Step {step}: Irradiance recorded (Reverse Ramp) -> {env.env.irradiance_log_new[-1]}")
    print(f"Step {step}: Voltage recorded -> {obs[0]}")
    print(f"Step {step}: Power output recorded -> {info.get('power_output', 0.0)}")
    print(f"Step output: obs={obs}, reward={reward}, terminated={terminated}, truncated={truncated}, info={info}")
    if terminated or truncated:
        obs, info = env.reset()
    voltages.append(obs[0])
    irradiances_gradual.append(env.env.irradiance_log_gradual[-1])
    irradiances_step.append(env.env.irradiance_log_step[-1])
    irradiances_new.append(env.env.irradiance_log_new[-1])
    temperatures.append(obs[2])
    actual_powers.append(info.get('power_output', 0.0))
    rewards_list.append(reward)
    print(f"Step {step}: Irradiance (Gradual) = {irradiances_gradual[-1]}, Irradiance (Step) = {irradiances_step[-1]}, Irradiance (Reverse Ramp) = {irradiances_new[-1]}")
    if step % 500 == 0:
        print(f"Step {step}: Irradiance Log (Last 10 values - Gradual): {env.env.irradiance_log_gradual[-10:]}")
        print(f"Step {step}: Irradiance Log (Last 10 values - Step): {env.env.irradiance_log_step[-10:]}")
        print(f"Step {step}: Irradiance Log (Last 10 values - Reverse Ramp): {env.env.irradiance_log_new[-10:]}")

print(f"Length of DRL power_log: {len(env.env.power_log)}")
print(f"Length of DRL voltage_log: {len(env.env.voltage_log)}")
print(f"Length of power_log_step: {len(env.env.power_log_step)}")
print(f"Length of voltage_log_step: {len(env.env.voltage_log_step)}")
print(f"Length of power_log_new: {len(env.env.power_log_new)}")
print(f"Length of voltage_log_new: {len(env.env.voltage_log_new)}")
print(f"Length of reward_log_gradual: {len(env.env.reward_log_gradual)}")
print(f"Length of reward_log_step: {len(env.env.reward_log_step)}")
print(f"Length of reward_log_new: {len(env.env.reward_log_new)}")
print(f"Sample DRL power_log values (first 5): {env.env.power_log[:5]}")
print(f"Sample DRL voltage_log values (first 5): {env.env.voltage_log[:5]}")
print(f"Sample DRL power_log_step values (first 5): {env.env.power_log_step[:5]}")
print(f"Sample DRL voltage_log_step values (first 5): {env.env.voltage_log_step[:5]}")
print(f"Sample DRL power_log_new values (first 5): {env.env.power_log_new[:5]}")
print(f"Sample DRL voltage_log_new values (first 5): {env.env.voltage_log_new[:5]}")
print(f"Sample DRL power_log values (last 5): {env.env.power_log[-5:]}")
print(f"Sample DRL voltage_log values (last 5): {env.env.voltage_log[-5:]}")
print(f"Sample DRL power_log_step values (last 5): {env.env.power_log_step[-5:]}")
print(f"Sample DRL voltage_log_step values (last 5): {env.env.voltage_log_step[-5:]}")
print(f"Sample DRL power_log_new values (last 5): {env.env.power_log_new[-5:]}")
print(f"Sample DRL voltage_log_new values (last 5): {env.env.voltage_log_new[-5:]}")

env.plot_irradiance()
env.plot_power_vs_time()
env.plot_voltage_vs_time()
env.plot_power_vs_voltage()
env.plot_temperature_vs_time()
env.plot_reward_vs_time()

model.save("trained_mppt_model")
print("Model training complete and saved.")

smoothed_power = smooth_data(actual_powers, window_size=15)
smoothed_irradiance_gradual = smooth_data(irradiances_gradual, window_size=15)
smoothed_irradiance_step = smooth_data(irradiances_step, window_size=15)
smoothed_irradiance_new = smooth_data(irradiances_new, window_size=15)
smoothed_rewards = smooth_data(rewards_list, window_size=15)

env.print_step_summary()
print("Final Irradiance Log (Gradual): ", env.env.irradiance_log_gradual)
print("Final Irradiance Log (Step): ", env.env.irradiance_log_step)
print("Final Irradiance Log (Reverse Ramp): ", env.env.irradiance_log_new)
print("=== Main Simulation End ===")
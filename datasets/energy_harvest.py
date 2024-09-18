import pandas as pd
import numpy as np
import scipy.signal as signal
import scipy

class EnergyHarvester():
    """
    characterize energy harvesting output from a piezoelectric energy harvester
        M. Gorlatova, J. Sarik, G. Grebla, M. Cong, I. Kymissis and G. Zussman, 
        "Movers and Shakers: Kinetic Energy Harvesting for the Internet of Things," 
        in IEEE Journal on Selected Areas in Communications, vol. 33, no. 8, pp. 
        1624-1639, Aug. 2015, doi: 10.1109/JSAC.2015.2391690.

    usage example:
        energy_params = {
            'proof_mass': 10**-3,
            'spring_const': 0.17,
            'spring_damp': 0.0055,
            'disp_max': 0.01,
            'efficiency': 0.25
        }
        harvester = EnergyHarvester(**energy_params)
        time, power = harvester.power(data)
        energy = harvester.energy(power, time)
    """

    def __init__(self,
                 proof_mass=10**-3,
                 spring_const=0.17,
                 spring_damp=0.0055,
                 disp_max=0.01,
                 efficiency=0.5) -> None:
        """
        proof_mass:
            mass of the proof mass in kg
        
        spring_const:
            spring constant in N/m

        spring_damp:
            damping constant in N/(m/s)

        disp_max:
            maximum displacement of the proof mass in m

        efficiency:
            fraction of energy actually harvested [0,1]
        """
        self.proof_mass = proof_mass
        self.spring_const = spring_const
        self.spring_damp = spring_damp
        self.disp_max = disp_max
        self.efficiency = efficiency

    def power(self, data : pd.DataFrame, 
              use_x=True, use_y=True, use_z=True) -> (np.ndarray, np.ndarray):
        """
        calculates power per unit time, units in Watts

        data:
            pandas dataframe with columns: time, x, y, z
            x, y, z units should be in m/s^2
            time units should be in seconds

        use_x, use_y, use_z:
            boolean values indicating whether or not to use the respective
            axis in the energy harvest calculation

        returns:
            time_out: numpy array of time values in seconds
            power_out: numpy array of power values in Watts
        """

        # validate input
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas dataframe")
        time = data.get('time', None)
        accx = data.get('x', None)
        accy = data.get('y', None)
        accz = data.get('z', None)
        if any([x is None for x in [time, accx, accy, accz]]):
            raise ValueError("data must have columns: time, x, y, z")
        
        # preprocess
        amag = np.sqrt(((accx**2) if use_x else 0) + 
                       ((accy**2) if use_y else 0) + 
                       ((accz**2) if use_z else 0))
        t_step = np.mean(np.diff(time))   # these should all be the same value
        fs = 1/t_step
        # generate filter (3rd order butterworth, 0.1Hz cutoff)
        # cutoff is specified as a fraction of the nyquist frequency (fs/2)
        iirb, iira = signal.butter(3, (2*0.1)/fs, 'highpass')
        filter_amag = signal.filtfilt(iirb, iira, amag)

        # calculate power
        tf = signal.TransferFunction(
                [1], 
                [1, 
                 self.spring_damp/self.proof_mass, 
                 self.spring_const/self.proof_mass])
        
        # calculate position of proof mass
        time_out, zpos, _ = signal.lsim(tf, filter_amag, time)
        zpos = np.clip(zpos.flatten(), -self.disp_max, self.disp_max)

        # calculate velocity of proof mass
        zvel = np.gradient(zpos, time)

        # calculate power: power = damping * velocity^2
        damp_power = self.spring_damp * (zvel**2)

        return time_out, damp_power

    def energy(self, time : np.ndarray, power : np.ndarray) -> np.ndarray:
        """
        calculates energy per unit time, units in Joules

        time:
            numpy array of time values in seconds

        power:
            numpy array of power values in Watts
        
        returns:
            energy: numpy array of energy values in Joules, same length as time and power
        """
        return scipy.integrate.cumulative_trapezoid(power, time, initial=0)*self.efficiency
    
    def generate_valid_mask(self, energy : np.ndarray, accel_samples : int) -> np.ndarray:
        """
        generates a mask of valid samples based on the energy output of the harvester
        mask elements are 1 if valid and NaN if invalid

        energy:
            numpy array of energy values in Joules

        accel_samples:
            number of accelerometer samples per packet

        returns:
            valid: numpy array of mask values, same length as energy
            threshold: energy threshold per packet in J            
        """
        thresh = self._energy_per_packet(accel_samples)
        valid = np.empty(len(energy))
        valid[:] = np.nan

        total = thresh
        for i, e in enumerate(energy):
            if (e > total):
                valid[i:i+accel_samples] = 1
                total += thresh

        return valid, thresh
    
    @staticmethod
    def get_data_sparsity(valid : np.ndarray) -> float:
        return np.mean(np.nan_to_num(valid, nan=0))

    def _energy_per_packet(self, samples : int) -> float:
        # returns in J
        # https://www.researchgate.net/publication/254038162_How_low_energy_is_Bluetooth_low_energy_Comparative_measurements_with_ZigBee802154
        # https://www.bosch-sensortec.com/products/motion-sensors/accelerometers/bma400/

        def tx_energy(bytes):
            num_packets = np.ceil(bytes / 240)
            return (15 * 1 + 24 * 1.4) * num_packets + 84 * 0.008 * bytes
        def acc_energy(n):
            return (6.3 * n) / 25 # 25 Hz sampling rate, 3.5uA, 1.8V
        
         # 6 bytes per acc sample
        if samples <= 16:
            # real profiling for 16 samples resulted in ~42uJ
            return np.float64(50e-6)
        else:
            return 10**-6 * (tx_energy(6 * samples) + acc_energy(samples))


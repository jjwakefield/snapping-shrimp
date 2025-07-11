import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.stats import levy_stable


def get_snap_rate_from_temp(temperature_celsius):
    """
    Calculates the seasonal snapping shrimp snap rate from water temperature.

    This function uses a linear regression model derived from the annual dataset
    in Bohnenstiehl et al. (2016), Figure 5b. It captures the strong
    seasonal trend of higher snap rates in warmer months and lower rates in
    colder months.

    Args:
        temperature_celsius (float): The water temperature in degrees Celsius.

    Returns:
        float: The estimated snap rate in snaps per minute.
    """
    # Coefficients derived from Bohnenstiehl et al. (2016), Fig 5b.
    SLOPE = 137.0
    INTERCEPT = -685.0

    snap_rate_per_minute = (SLOPE * temperature_celsius) + INTERCEPT
    return max(0, snap_rate_per_minute)


class BiologicalNoiseModel:
    """A base class for generating biological noise models."""

    def __init__(self, duration, fs, start_time_hours=0.0):
        self.duration = duration
        self.fs = fs
        # Set the absolute start time in seconds from the start of the day
        self.start_time_seconds = start_time_hours * 3600
        # The time vector for plotting is relative to the start of the snippet
        self.time_vector = np.arange(0, self.duration, 1 / self.fs)
        # The signal buffer for the snippet
        self.signal = np.zeros_like(self.time_vector)

    def generate(self):
        raise NotImplementedError("Each subclass must implement its own generate method.")

    def save_wav(self, filename):
        """
        Saves the pressure signal as a 16-bit WAV file by normalising
        to the signal's own peak value to guarantee no clipping.
        """
        print(f"Saving audio to {filename}...")

        peak_pressure = np.max(np.abs(self.signal))

        if peak_pressure == 0:
            print("Signal is silent. Nothing to save.")
            return

        # Normalise the signal by its own absolute maximum value
        normalised_signal = self.signal / peak_pressure

        audio_data = np.int16(normalised_signal * 32767)
        wavfile.write(filename, self.fs, audio_data)
        print("Save complete.")

    def plot(
        self, title_suffix="", xlim=None, include_title=False, save_path=None, show_plot=False
    ):
        """Plots the signal waveform in Pascals."""
        plt.style.use("seaborn-v0_8-paper")
        font_size_label, font_size_title = 12, 14
        fig, ax = plt.subplots(figsize=(8, 4))

        ax.plot(self.time_vector, self.signal, lw=1.0)
        ax.set_xlabel("Time (s)", fontsize=font_size_label)
        ax.set_ylabel("Pressure (Pa)", fontsize=font_size_label)
        if include_title:
            ax.set_title(
                f"{self.__class__.__name__}{title_suffix}", fontsize=font_size_title, weight="bold"
            )

        if xlim:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(0, self.duration)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="both", which="major", labelsize=font_size_label - 1)
        ax.grid(True, linestyle=":", alpha=0.7)
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if show_plot:
            plt.show()


class SnappingShrimp(BiologicalNoiseModel):
    """
    Generates snapping shrimp noise where the snap rate is seasonally dependent
    on water temperature and modulated by rhythmic cycles.
    """

    # MODIFIED: Added delay_duration parameter
    def __init__(
        self,
        duration,
        fs,
        start_time_hours,
        temperature_celsius,
        alpha,
        scale_pa,
        delay_duration=0.0006,  # Duration of the flat line before the snap
        onset_duration=0.0001,
        snap_duration=0.0014,
        onset_level=0.15,
        onset_freq=1500,
        snap_decay=2000,
        low_cutoff_hz=1000,
        high_cutoff_hz=20000,
        diurnal_amplitude=0.0,
        diurnal_phase_hours=0.0,
        tidal_amplitude=0.0,
        tidal_phase_hours=0.0,
    ):
        """Initialises the SnappingShrimp model."""
        super().__init__(duration, fs, start_time_hours)
        self.temperature_celsius = temperature_celsius
        self.base_lambda_rate = get_snap_rate_from_temp(temperature_celsius) / 60.0

        self.diurnal_amplitude, self.diurnal_phase = (
            diurnal_amplitude,
            diurnal_phase_hours * np.pi / 12.0,
        )
        self.tidal_amplitude, self.tidal_phase = (
            tidal_amplitude,
            tidal_phase_hours * np.pi / 6.21,
        )

        self.alpha, self.scale_pa = alpha, scale_pa

        # Store waveform generation parameters
        self.delay_duration = delay_duration  # Store new parameter
        self.onset_duration = onset_duration
        self.snap_duration = snap_duration
        self.onset_level = onset_level
        self.onset_freq = onset_freq
        self.snap_decay = snap_decay
        self.low_cutoff_hz = low_cutoff_hz
        self.high_cutoff_hz = high_cutoff_hz

        self.time_template, self.snap_template = self._create_snap_template()

    def _create_snap_template(self):
        """
        Creates a single, snap waveform template.

        The template is constructed in three parts: a silent delay, a rising
        onset wave, and the main snap impulse. The combined waveform is then
        bandpass filtered.

        Returns
        -------
        t_final : np.ndarray
            The time vector for the snap template in seconds.
        final_filtered_waveform : np.ndarray
            The generated snap waveform template.
        """
        # --- Part 1: Pre-onset Delay (Silence) ---
        delay_samples = int(self.delay_duration * self.fs)
        delay_padding = np.zeros(delay_samples)

        # --- Part 2: Rising Onset Wave ---
        onset_samples = int(self.onset_duration * self.fs)
        t_onset = np.linspace(0, self.onset_duration, onset_samples, endpoint=False)
        onset_sine = np.sin(2 * np.pi * self.onset_freq * t_onset)
        onset_ramp = np.linspace(0, 1, onset_samples)
        onset_wave = onset_sine * onset_ramp * self.onset_level

        # --- Part 3: Main Snap Impulse ---
        snap_samples = int(self.snap_duration * self.fs)
        t_snap = np.linspace(0, self.snap_duration, snap_samples, endpoint=False)
        snap_noise = np.random.uniform(-1, 1, size=len(t_snap))
        snap_envelope = np.exp(-self.snap_decay * t_snap)
        snap_impulse = snap_noise * snap_envelope

        # --- Combine all parts ---
        raw_waveform = np.concatenate([delay_padding, onset_wave, snap_impulse])

        # Part 4: Apply a bandpass filter
        b, a = signal.butter(
            N=4, Wn=[self.low_cutoff_hz, self.high_cutoff_hz], btype="bandpass", fs=self.fs
        )
        final_filtered_waveform = signal.filtfilt(b, a, raw_waveform)

        num_samples = len(final_filtered_waveform)
        t_final = np.arange(num_samples) / self.fs

        return t_final, final_filtered_waveform

    def _rate_function(self, t):
        """Calculates the time-varying snap rate lambda(t) in snaps/sec."""
        diurnal_rad = 2 * np.pi * t / (24 * 3600)
        diurnal_mod = self.diurnal_amplitude * np.sin(diurnal_rad + self.diurnal_phase)
        tidal_rad = 2 * np.pi * t / (12.42 * 3600)
        tidal_mod = self.tidal_amplitude * np.sin(tidal_rad + self.tidal_phase)
        rate = self.base_lambda_rate * (1 + diurnal_mod + tidal_mod)
        return np.maximum(0, rate)

    def generate(self):
        """
        Generates the full time series of snapping shrimp noise.

        This method uses a Non-Homogeneous Poisson Process (NHPP) via the
        thinning algorithm to generate snap arrival times. Each snap is then
        assigned an amplitude from an SαS distribution whose scale is
        dynamically adjusted based on the instantaneous snap rate.

        Returns
        -------
        np.ndarray
            The generated noise signal as a 1D NumPy array.
        """
        print(f"Generating {self.__class__.__name__} noise...")

        # --- Generate Snap Times using NHPP Thinning Algorithm ---
        # 1. Determine the maximum possible snap rate for the majorant function.
        lambda_max = self.base_lambda_rate * (1 + self.diurnal_amplitude + self.tidal_amplitude)
        if lambda_max <= 0:
            return

        # 2. Generate candidate snap times from a homogeneous Poisson process.
        num_candidates = int(self.duration * lambda_max * 1.1)  # Add 10% buffer
        if num_candidates == 0:
            return

        candidate_intervals = np.random.exponential(scale=1.0 / lambda_max, size=num_candidates)
        candidate_times = self.start_time_seconds + np.cumsum(candidate_intervals)
        end_time_seconds = self.start_time_seconds + self.duration
        candidate_times = candidate_times[candidate_times < end_time_seconds]

        if len(candidate_times) == 0:
            return

        # 3. "Thin" the candidates by accepting them with probability p(t) = lambda(t)/lambda_max.
        actual_rates = self._rate_function(candidate_times)
        random_checks = np.random.uniform(0, 1, size=len(candidate_times))
        accepted_mask = random_checks < (actual_rates / lambda_max)
        snap_times = candidate_times[accepted_mask]

        if len(snap_times) == 0:
            return

        # --- Generate Signal from Accepted Snap Times ---
        # Calculate dynamic amplitude scales based on the snap rate at each event time.
        rates_at_snap_times = self._rate_function(snap_times)
        modulation_factors = rates_at_snap_times / self.base_lambda_rate
        dynamic_scales_pa = self.scale_pa * modulation_factors

        # Generate amplitudes from the SαS distribution using the dynamic scales.
        amplitudes_pa = levy_stable.rvs(
            alpha=self.alpha, beta=0, scale=dynamic_scales_pa, size=len(snap_times)
        )

        # Place the scaled snap template at each accepted time in the final signal array.
        snap_len = len(self.snap_template)
        for i, absolute_time_point in enumerate(snap_times):
            relative_time = absolute_time_point - self.start_time_seconds
            start_index = int(round(relative_time * self.fs))
            # Ensure the template fits within the signal array bounds
            if start_index + snap_len < len(self.signal):
                self.signal[start_index : start_index + snap_len] += (
                    self.snap_template * amplitudes_pa[i]
                )

        print(
            f"Generation complete. Kept {len(snap_times)} of {len(candidate_times)} candidate snaps."
        )

        return self.signal


# --- Main Execution ---
if __name__ == "__main__":
    np.random.seed(42)

    # --- 1. Define Physical and Environmental Parameters ---
    TARGET_SL_DB = 190
    WATER_TEMP_C = 22
    P_REF_UPA = 1e-6
    SCALE_PRESSURE_PA = P_REF_UPA * (10 ** (TARGET_SL_DB / 20.0))

    # MODIFIED: Instantiation now uses the new waveform parameters
    shrimp_model = SnappingShrimp(
        # --- Simulation Window ---
        start_time_hours=18.0,
        duration=60,
        # --- Core Simulation Settings ---
        fs=60_000,
        temperature_celsius=WATER_TEMP_C,
        # --- Snap Amplitude Distribution Parameters ---
        alpha=1.5,
        scale_pa=SCALE_PRESSURE_PA,
        # --- New Individual Snap Waveform Parameters ---
        delay_duration=0.0006,
        onset_duration=0.0001,
        snap_duration=0.0014,  # Duration of the main impulse part
        onset_level=0.15,
        onset_freq=1500,
        snap_decay=2000,
        low_cutoff_hz=1500,
        high_cutoff_hz=20000,
        # --- Rhythmic Cycle Parameters ---
        diurnal_amplitude=0.25,
        diurnal_phase_hours=6,
    )

    print(f"Water Temperature: {shrimp_model.temperature_celsius}°C")
    print(f"Base Snap Rate: {shrimp_model.base_lambda_rate:.2f} snaps/s")

    # --- 2. Generate, Save, and Plot ---
    shrimp_model.generate()
    shrimp_model.save_wav("shrimp_simulation.wav")
    shrimp_model.plot(save_path="shrimp_simulation.svg", show_plot=False)

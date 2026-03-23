from peak import Peak
import numpy as np
import pywt
from scipy.signal import savgol_filter
import plotly.graph_objects as go
import matplotlib.pyplot as plt

class Signal:
    '''
    Represents a voltammogram signal with shared experimental parameters.
    '''
    # Class-level attribute (Static)
    # Shared by all instances of Signal
    sig_id: int = 1
    E: np.ndarray = np.array([])
    I_baseline = np.array([])

    def __init__(self, I: np.ndarray):
        '''
        Initializes a Signal instance and performs automated preprocessing.

        This constructor validates the input current dimensions against the shared 
        potential array (E). If a global baseline is available, it automatically 
        normalizes the signal. Finally, it applies a smoothing filter to the 
        current data.

        Args:
            I (np.ndarray): The raw current measurements (A) obtained from the 
                voltammogram. Must match the length of the static `Signal.E` array.

        Raises:
            ValueError: If the static potential array `Signal.E` has not been 
                initialized.
            ValueError: If the length of the provided current array `I` does 
                not match the length of the static potential array `Signal.E`.

        Note:
            The initialization sequence is: Validation -> Normalization (if baseline exists) -> Smoothing.    
        '''
        if Signal.E.size == 0:
            raise ValueError("The potential array has not been set")
        if I.size != Signal.E.size:
            raise ValueError(f"Current array size ({I.size}) does not match potential array size ({self.E.size}).")
        self.I = I
        self._ssa_components = None
        
        if Signal.I_baseline.size != 0:
            self._normalize_signal()
        self._smoth_signal()
        self.peak = Peak(Signal.E, self.I)
        Signal.sig_id += 1

    def _normalize_signal(self) -> None:
        '''
        Normalizez the signal, by subtracting the shared class-level baseline current from the instance current.

        This performs a point-by-point vector subtraction ($I_{normalized} = I_{raw} - I_{baseline}$).
        It requires that the static `I_baseline` attribute has been populated 
        prior to execution.

        Raises:
            ValueError: If `Signal.I_baseline` is empty or has not been set.
        
        Note:
            This method modifies `self.I` in-place.
        '''
        if Signal.I_baseline.size == 0:
            raise ValueError("The baseline current was not set.")
        self.I = self.I - Signal.I_baseline

    def _smoth_signal(self, win_length: int = 5, polyorder: int = 3) -> None:
        '''
        Applies a Savitzky-Golay filter to the current array to reduce noise.

        The Savitzky-Golay filter smooths data by fitting successive sub-sets 
        of adjacent data points with a low-degree polynomial using the 
        method of linear least squares.

        Args:
            win_length (int): The length of the filter window (number of coefficients). 
                Must be a positive odd integer. Defaults to 5.
            polyorder (int): The order of the polynomial used to fit the samples. 
                Must be less than `win_length`. Defaults to 3.
        Note:
            This method modifies `self.I` in-place.
        '''
        self.I = savgol_filter(self.I, window_length=win_length, polyorder=polyorder)

    def _apply_ssa(self, window_size: int = 10) -> np.ndarray:
        '''
        Applies Singular Spectrum Analysis (SSA) to the current signal.

        Constructs a trajectory matrix from the instance's current array (`self.I`) 
        and performs Singular Value Decomposition (SVD) to extract the principal 
        components (singular values).

        Args:
            window_size (int): The embedding window length. Must be less than or 
                equal to half the length of the signal. Defaults to 10.

        Raises:
            ValueError: If the current array is empty or the `window_size` exceeds 
                half the length of the signal.
        '''
        if self._ssa_components is not None:
            return self._ssa_components
    
        N = self.I.size
        if N == 0:
            raise ValueError("The current array is empty. Cannot perform SSA.")
        if window_size > N // 2:
            raise ValueError(f"Window size ({window_size}) cannot exceed half the signal length ({N // 2}).")

        # Create the trajectory matrix (L x K)
        L = window_size
        K = N - L + 1
        X = np.column_stack([self.I[i:i + L] for i in range(K)])

        # Perform SVD to get the singular values
        _, Sigma, _ = np.linalg.svd(X)
        
        self._ssa_components = Sigma
        return self._ssa_components

    @classmethod
    def set_common_potential_E(cls, E: np.ndarray) -> None:
        '''Sets the shared potential array for all future Signal instances.'''
        cls.E = E
    
    @classmethod
    def set_common_baseline_I(cls, I_baseline: np.ndarray) -> None:
        '''Sets the shared baseline current array for all future Signal instances.'''
        cls.I_baseline = I_baseline

    def get_peak_current_value(self) -> float:
        '''Returns the maximum current (Ip) of the identified peak.'''
        return self.peak.Ip
    
    def get_peak_potential_value(self) -> float:
        '''Returns the potential (Ep) at which the peak current occurs.'''
        return self.peak.Ep
    
    def get_peak_fwhm(self, is_plotting: bool = False) -> float:
        '''
        Calculates the Full Width at Half Maximum (FWHM) of the peak.

        The FWHM is determined by finding the potential width where the current 
        is at least 50% of the peak maximum (Ip). It searches within the refined 
        peak boundaries (start_idx to end_idx).

        Args:
            is_plotting (bool): If True, generates a Matplotlib figure showing 
                the peak, the 50% threshold line, and the intercepted FWHM points. 
                Defaults to False.

        Returns:
            float: The width of the peak in Volts (V).

        Raises:
            ValueError: If fewer than two data points exist above the 50% 
                threshold, making width calculation impossible.
        '''
        threshold = self.peak.Ip * 0.5
        idxs_above_threshold = np.where(self.I[self.peak.start_idx: self.peak.end_idx + 1] >= threshold)[0] + self.peak.start_idx

        if len(idxs_above_threshold) < 2:
            raise ValueError("Not enough points above threshold to calculate FWHM.")

        fwhm_left_idx = idxs_above_threshold[0]
        fwhm_right_idx = idxs_above_threshold[-1] + 1

        if is_plotting:
            plt.plot(Signal.E, self.I, label='Current vs Potential')
            plt.hlines(threshold, Signal.E[fwhm_left_idx], Signal.E[fwhm_right_idx], colors='r', linestyles='dashed', label='FWHM Threshold')
            plt.plot(Signal.E[fwhm_left_idx], threshold, 'go')
            plt.plot(Signal.E[fwhm_right_idx], threshold, 'go')
            plt.vlines([Signal.E[fwhm_left_idx], Signal.E[fwhm_right_idx]], 0, threshold, colors='r', linestyles='dashed', label='FWHM Points')
            plt.xlabel('Potential (V)')
            plt.ylabel('Current (A)')
            plt.title('Full Width at Half Maximum (FWHM)')
            plt.legend()
            plt.grid()
            plt.show()

        return Signal.E[fwhm_right_idx] - Signal.E[fwhm_left_idx]
    
    def get_peak_auc(self) -> float:
        '''
        Calculates the Area Under the Curve (AUC) for the peak region.

        Uses the trapezoidal rule to integrate the current (I) over the 
        potential (E) within the refined peak boundaries. 
        Note: The units of the result are [Amperes * Volts].

        Returns:
            float: The numerical integral of the peak.
        '''
        auc = np.trapezoid(self.I[self.peak.start_idx: self.peak.end_idx + 1], Signal.E[self.peak.start_idx: self.peak.end_idx + 1])
        # if is_plotting:
        #     pass
        return auc
    
    def get_pca1_comp(self) -> float:
        '''
        Retrieves the first principal component (PCA1) from the Singular Spectrum Analysis (SSA)
          of the signal.

        This component typically captures the most significant underlying trend or 
        variance within the voltammogram's current data.

        Returns:
            float: The magnitude of the first principal component.
        '''
        components = self._apply_ssa()
        return float(components[0])
    
    def get_first_derivative_max(self) -> float:
        '''
        Calculates the maximum value of the first derivative localized to the peak.

        Computes the numerical gradient of the current with respect to the 
        potential (dI/dE) specifically within the peak boundaries to avoid noise or 
        background signal contributions.

        Returns:
            float: The maximum slope of the signal in Amperes per Volt (A/V).
        '''
        I_peak = self.I[self.peak.start_idx : self.peak.end_idx + 1]
        E_peak = Signal.E[self.peak.start_idx : self.peak.end_idx + 1]

        if I_peak.size < 2:
            return 0.0
        
        dI_dE = np.gradient(I_peak, E_peak)
        return float(np.max(dI_dE))

    def get_second_derivative_min(self) -> float:
        '''
        Calculates the minimum value of the second derivative localized to the peak.

        Computes the second numerical gradient (d^2I/dE^2) of the current with 
        respect to the potential within the refined peak boundaries. This metric 
        identifies the point of sharpest downward concavity at the peak's apex.

        Returns:
            float: The minimum value of the second derivative (A/V^2).
        '''

        I_peak = self.I[self.peak.start_idx : self.peak.end_idx + 1]
        E_peak = Signal.E[self.peak.start_idx : self.peak.end_idx + 1]

        if I_peak.size < 3:
            return 0.0
        
        dI_dE = np.gradient(I_peak, E_peak)
        d2I_dE2 = np.gradient(dI_dE, E_peak)
        return float(np.min(d2I_dE2))
    
    def get_peak_sharpness(self) -> float:
        '''
        Calculates the sharpness of the identified peak.

        Sharpness is defined as the ratio of the peak's maximum current (Ip) 
        to its Full Width at Half Maximum (FWHM). This metric normalizes the 
        peak height against its width.

        Returns:
            float: The sharpness ratio (A/V). Returns 0.0 if the FWHM cannot 
                be calculated.
        '''
        try:
            fwhm = self.get_peak_fwhm()
            if fwhm == 0:
                return 0.0
            return self.peak.Ip / fwhm
        except ValueError:
            return 0.0

    def get_peak_compactness(self) -> float:
        '''
        Calculates the compactness (fill factor) of the peak.

        Compactness evaluates how tightly the peak's area is concentrated. 
        It is calculated as the ratio of the Area Under the Curve (AUC) to the 
        area of the peak's bounding box (Ip * peak base width). 

        Returns:
            float: A dimensionless ratio representing the fill factor. 
                A perfect triangle yields 0.5, while an ideal Gaussian 
                bell curve yields approximately 0.7.
        '''
        auc = self.get_peak_auc()
        base_width = np.abs(self.peak.E_end - self.peak.E_start)
        
        bounding_box_area = self.peak.Ip * base_width
        
        if bounding_box_area == 0:
            return 0.0
            
        return np.abs(auc / bounding_box_area)
    
    def get_current_variance(self) -> float:
        '''
        Calculates the statistical variance of the current signal.

        This metric computes the global variance of the entire current array 
        (`self.I`), measuring the dispersion of the current values around 
        their mean. 

        Returns:
            float: The variance of the current measurements in squared Amperes (A^2).
        '''
        return float(np.var(self.I))
    
    def get_peak_skewness(self) -> float:
        '''
        Calculates the statistical skewness of the peak shape.

        Treats the refined peak region as a frequency distribution where the 
        current (I) represents the weight at each potential (E). It computes 
        the third standardized moment to measure the horizontal asymmetry 
        of the bell curve.

        Returns:
            float: The skewness of the peak. A perfectly symmetric peak returns 
                0.0. A positive value indicates a tail extending towards more 
                positive potentials, while a negative value indicates a tail 
                towards more negative potentials. Returns 0.0 if the peak 
                area is zero or the standard deviation is zero.
        '''
        I_peak = self.I[self.peak.start_idx : self.peak.end_idx + 1]
        E_peak = Signal.E[self.peak.start_idx : self.peak.end_idx + 1]

        total_I = np.sum(I_peak)
        if total_I == 0:
            return 0.0

        # Calculate the weighted mean potential (the centroid of the peak)
        mu = np.sum(E_peak * I_peak) / total_I

        # Calculate the weighted standard deviation
        variance = np.sum(I_peak * (E_peak - mu)**2) / total_I
        if variance == 0:
            return 0.0
        sigma = np.sqrt(variance)

        # Calculate the 3rd standardized moment (Skewness)
        skewness = np.sum(I_peak * (E_peak - mu)**3) / (total_I * sigma**3)
        return float(skewness)

    def get_peak_kurtosis(self) -> float:
        '''
        Calculates the statistical excess kurtosis of the peak shape.

        Treats the refined peak region as a frequency distribution and computes 
        the fourth standardized moment. This metric measures the "tailedness" 
        or "peakedness" of the signal compared to a normal Gaussian distribution.
        The result is adjusted by subtracting 3 (Fisher's excess kurtosis) so 
        that a perfect normal distribution returns 0.0.

        Returns:
            float: The excess kurtosis of the peak. A positive value indicates a 
                sharper peak with heavier tails, while a negative value indicates 
                a flatter, broader peak. Returns 0.0 if the peak area or variance 
                is zero.
        '''
        I_peak = self.I[self.peak.start_idx : self.peak.end_idx + 1]
        E_peak = Signal.E[self.peak.start_idx : self.peak.end_idx + 1]

        total_I = np.sum(I_peak)
        if total_I == 0:
            return 0.0

        # Calculate the centroid (weighted mean)
        mu = np.sum(E_peak * I_peak) / total_I

        # Calculate the variance and standard deviation
        variance = np.sum(I_peak * (E_peak - mu)**2) / total_I
        if variance == 0:
            return 0.0
        sigma = np.sqrt(variance)

        # Calculate the 4th standardized moment (Pearson Kurtosis)
        pearson_kurtosis = np.sum(I_peak * (E_peak - mu)**4) / (total_I * sigma**4)

        excess_kurtosis = pearson_kurtosis - 3.0
        return float(excess_kurtosis)

    def get_tchebichef_curve_moments(self, order: int = 2) -> float:
        '''
        Calculates a specific order Chebyshev (Tchebichef) curve moment for the peak.

        This method normalizes the peak's potential domain to [-1, 1] and fits 
        the current data to a series of orthogonal Chebyshev polynomials of the 
        first kind. The resulting coefficients act as orthogonal shape moments.

        Args:
            order (int): The degree of the Chebyshev polynomial moment to extract. 
                Order 2 (default) captures the primary parabolic concavity of the peak.

        Returns:
            float: The coefficient (moment) of the requested Chebyshev order. 
                Returns 0.0 if the peak contains too few points to fit the curve.
        '''
        import numpy.polynomial.chebyshev as cheb

        I_peak = np.array(self.I[self.peak.start_idx : self.peak.end_idx + 1], dtype=float)
        E_peak = np.array(Signal.E[self.peak.start_idx : self.peak.end_idx + 1], dtype=float)
        
        if len(I_peak) < order + 1:
            return 0.0
            
        E_min, E_max = E_peak[0], E_peak[-1]
        if E_max == E_min:
            return 0.0
            
        # Map the physical potential (E) to Chebyshev's standard domain [-1, 1]
        E_normalized = 2.0 * (E_peak - E_min) / (E_max - E_min) - 1.0
        
        # Fit the normalized peak to a Chebyshev series
        # Return an array of coefficients [c_0, c_1, c_2, ... c_order]
        coeffs = cheb.chebfit(E_normalized, I_peak, deg=order)
    
        return float(coeffs[order])
    
    def get_left_slope(self) -> float:
        '''
        Calculates the slope of the leading (left) edge of the peak.

        The slope is determined by a first-order polynomial fit (linear regression) 
        on data points between a calculated threshold (alpha * Ip) and the peak 
        maximum (Ip).

        Returns:
            float: The slope of the linear fit ($dI/dE$).

        Raises:
            ValueError: If the window contains fewer than two points, 
                preventing a linear fit.
        '''
        alpha = 0.1
        threshold = alpha * self.peak.Ip
        threshold_idx = self.peak.start_idx + np.where(self.I[self.peak.start_idx:self.peak.Ip_idx] >= threshold)[0][0]

        I_left = np.array(self.I[threshold_idx:self.peak.Ip_idx], dtype=float)
        E_left = np.array(Signal.E[threshold_idx:self.peak.Ip_idx], dtype=float)

        if len(I_left) < 2 or len(E_left) < 2:
            raise ValueError("Not enough points to calculate left slope.")
        
        coefficients = np.polyfit(E_left, I_left, deg=1)
        return coefficients[0]

    def get_right_slope(self) -> float:
        '''
        Calculates the slope of the trailing (right) edge of the peak.

        The slope is determined by a first-order polynomial fit (linear regression) 
        on data points between the peak maximum (Ip) and a calculated 
        threshold (alpha * Ip) on the descent.

        Returns:
            float: The slope of the linear fit ($dI/dE$).

        Raises:
            ValueError: If the window contains fewer than two points.
        '''
        alpha = 0.1
        threshold = alpha * self.peak.Ip
        threshold_idx = self.peak.Ip_idx + np.where(self.I[self.peak.Ip_idx:self.peak.end_idx] >= threshold)[0][-1]

        I_right = np.array(self.I[self.peak.Ip_idx + 1:threshold_idx], dtype=float)
        E_right = np.array(Signal.E[self.peak.Ip_idx + 1:threshold_idx], dtype=float)

        if len(I_right) < 2 or len(E_right) < 2:
            raise ValueError("Not enough points to calculate left slope.")
        
        coefficients = np.polyfit(E_right, I_right, deg=1)
        return coefficients[0]
    
    def get_asymetry(self) -> float:
        '''
        Calculates the peak asymmetry factor based on the ratio of edge slopes.

        A perfectly symmetric peak will return a value near 1.0. Values 
        deviating from 1.0 indicate tailing or non-ideal electron 
        transfer kinetics.

        Returns:
            float: The absolute ratio of the left slope to the right slope.
        '''
        left_slope = self.get_left_slope()
        right_slope = self.get_right_slope()
        return np.abs(left_slope) / np.abs(right_slope)
    
    def get_mean_peak(self) -> float:
        '''
        Calculates the average current value within the peak boundaries.

        This metric computes the arithmetic mean of the current array,
        providing a measure of the overall reaction magnitude 

        Returns:
            float: The mean current of the peak in Amperes (A). Returns 0.0 if 
                the peak region is empty.
        '''
        I_peak = self.I[self.peak.start_idx : self.peak.end_idx + 1]
        
        if I_peak.size == 0:
            return 0.0
        return float(np.mean(I_peak))

    def get_signal_entropy(self, num_bins: int | str = 'fd') -> float:
        '''
        Calculates the Shannon entropy of the time-domain signal.

        This method uses a histogram to estimate the probability mass function 
        of current amplitudes. By default, it uses the Freedman-Diaconis ('fd') 
        rule to dynamically determine the optimal bin count based on signal 
        variance and sample size.

        Args:
            num_bins (int or str): The number of bins or a string specifying 
                an estimator (e.g., 'fd', 'sturges', 'scott'). Defaults to 'fd'.
        Returns:
            float: The Shannon entropy of the signal in bits. Returns 0.0 if 
                the signal is empty.
        '''
        if self.I.size == 0:
            return 0.0
        
        hist, _ = np.histogram(self.I, bins=num_bins)
        
        hist_nonzero = hist[hist > 0]
        
        # Normalize the counts to create a probability mass function (PMF)
        p = hist_nonzero / np.sum(hist_nonzero)
        
        entropy = -np.sum(p * np.log2(p))
        return float(entropy)

    def get_spectral_entropy(self) -> float:
        '''
        Calculates the spectral Shannon entropy of the signal.

        This method transforms the signal into the frequency domain via Fast Fourier Transform (FFT). 
        It normalizes the power spectrum (excluding the DC offset) into a 
        probability distribution to measure how the signal's energy is 
        spread across different frequencies.

        Returns:
            float: The spectral entropy in bits. A low value indicates a 
                highly structured signal dominated by a few frequencies, 
                while a high value indicates flat, white-noise-like behavior.
        '''
        if self.I.size == 0:
            return 0.0

        fft_components = np.fft.fft(self.I)
        power_spectrum = np.abs(fft_components[1:]) ** 2
        
        total_power = np.sum(power_spectrum)
        if total_power == 0:
            return 0.0
            
        # 2. Normalize to create a probability distribution
        p = power_spectrum / total_power
        p_nonzero = p[p > 0]
        
        spectral_entropy = -np.sum(p_nonzero * np.log2(p_nonzero))
        return float(spectral_entropy)
    
    def get_fft_power(self) -> float:
        '''
        Calculates the total spectral power of the signal's dynamic variations.

        This method applies a 1D Fast Fourier Transform (FFT) to the entire 
        current array. It computes the total power of the AC components by 
        summing the squared magnitudes of the frequencies, 
        excluding the DC component (0 Hz) to ignore baseline offset.

        Returns:
            float: The total spectral power of the signal's variations. Returns 
                0.0 if the signal is empty.
        '''
        if self.I.size == 0:
            return 0.0

        fft_components = np.fft.fft(self.I)
        
        power_spectrum = np.abs(fft_components[1:]) ** 2
        
        # Normalize by the length of the signal to get the true average power
        total_ac_power = np.sum(power_spectrum) / self.I.size
        return float(total_ac_power)
    
    def get_pca2_comp(self) -> float:
        '''
        Retrieves the second principal component (PCA2) from the Singular Spectrum Analysis (SSA) of the signal.

        This component generally isolates the second most significant variance, 
        often associated with secondary electrochemical behaviors or structural 
        features of the peak.

        Returns:
            float: The magnitude of the second principal component.
        '''
        components = self._apply_ssa()
        return float(components[1]) if len(components) > 1 else 0.0

    def get_pca3_comp(self) -> float:
        '''
        Retrieves the third principal component (PCA3) from the Singular Spectrum Analysis (SSA)
          of the signal.

        This component captures tertiary variance patterns, which may represent 
        minor structural details or structured noise in the voltammogram.

        Returns:
            float: The magnitude of the third principal component.
        '''
        components = self._apply_ssa()
        return float(components[2]) if len(components) > 2 else 0.0
    
    def get_wavelet_energy(self, scales: np.ndarray | None = None) -> float:
        '''
        Calculates the Continuous Wavelet Transform (CWT) energy of the signal.

        This method applies a Continuous Wavelet Transform using the mexh
        wavelet across a standard range of width scales. It computes the total wavelet energy 
        by summing the squared coefficients of the resulting time-scale matrix.
        
        Args:
            scales (np.ndarray, optional): The scales for the transform. 
                Defaults to np.arange(1, 31).

        Returns:
            float: The total energy of the wavelet transform. Returns 0.0 if 
                the signal is empty.
        '''
        if self.I.size == 0:
            return 0.0

        if scales is None:
            scales = np.arange(1, 31)
        
        coeffs, _ = pywt.cwt(self.I, scales, 'mexh')
        
        # Calculate the total wavelet energy
        wavelet_energy = np.linalg.norm(coeffs)**2
        return float(wavelet_energy)
    
    def pplot(self, start: int = 0, end: int = 230) -> None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=Signal.E[start:end], y=self.I[start:end], mode="lines", name="signal"))
        # plotting the area
        fig.add_trace(go.Scatter(
            x=Signal.E[self.peak.start_idx:self.peak.end_idx + 1],
            y=self.I[self.peak.start_idx:self.peak.end_idx + 1],
            mode="lines",
            name="signal",
            fill="tozeroy"
        ))
        # plotting starting of the peak
        fig.add_trace(go.Scatter(
            x=[self.peak.E_start],
            y=[self.I[self.peak.start_idx]],
            mode="lines+markers",
            name="left"
        ))
        # plotting ending of the peak
        fig.add_trace(go.Scatter(
            x=[self.peak.E_end],
            y=[self.I[self.peak.end_idx]],
            mode="lines+markers",
            name="right"
        ))
        # plotting the peak
        fig.add_trace(go.Scatter(
            x=[self.peak.Ep],
            y=[self.peak.Ip],
            mode="lines+markers",
            name="peak"
        ))
            
        fig.update_layout(
            # width=750,
            height=750,
            template="plotly_white"
        )
        fig.update_xaxes(
            showgrid=True,
            minor=dict(showgrid=True)
        )
        fig.update_yaxes(
            showgrid=True,
            minor=dict(showgrid=True)
        )
        fig.show()
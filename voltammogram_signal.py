from peak import Peak
import numpy as np
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

        I_left = np.array(self.I[threshold_idx:self.peak.Ip_idx])
        E_left = np.array(Signal.E[threshold_idx:self.peak.Ip_idx])

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

        I_right = np.array(self.I[self.peak.Ip_idx + 1:threshold_idx])
        E_right = np.array(Signal.E[self.peak.Ip_idx + 1:threshold_idx])

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
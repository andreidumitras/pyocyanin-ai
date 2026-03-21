import numpy as np

class Peak:
    def __init__(self, E: np.ndarray, I: np.ndarray, E_start: float = -0.55, E_end: float = -0.25):
        '''
        Initializes a Peak object by identifying the maximum intensity and defining 
        refined boundaries within a specified potential window.

        The method first locates the maximum current (Ip) and corresponding potential (Ep) 
        within the range [E_start, E_end]. It then performs a local search outward 
        from the peak index to find the nearest local minima or zero-crossings, 
        which define the true start and end of the peak.

        Args:
            E (np.ndarray): Array of potential values (V).
            I (np.ndarray): Array of current values (A).
            E_start (float): The lower bound of the initial search window. 
                Defaults to -0.55.
            E_end (float): The upper bound of the initial search window. 
                Defaults to -0.25.

        Attributes:
            Ip_idx (int): The index in the original arrays corresponding to the peak maximum.
            Ip (float): The current value at the peak maximum.
            Ep (float): The potential value at the peak maximum.
            E_start (float): The potential at the refined start of the peak (left local minimum).
            start_idx (int): The index of the refined start point.
            E_end (float): The potential at the refined end of the peak (right local minimum).
            end_idx (int): The index of the refined end point.
        '''
        peak_indexes = np.where((E >= E_start) & (E <= E_end))
        peak_start_idx = peak_indexes[0][0]
        peak_end_idx = peak_indexes[0][-1]

        self.Ip_idx = np.argmax(I[peak_start_idx:peak_end_idx]) + peak_start_idx
        self.Ip = I[self.Ip_idx]
        self.Ep = E[self.Ip_idx]

        left_idx = self.Ip_idx
        Imin_left = self.Ip
        i = left_idx
        while left_idx > peak_start_idx and Imin_left > 0:
            if I[i] < Imin_left:
                Imin_left = I[i]
                left_idx = i
            i -= 1
        self.E_start = E[left_idx]
        self.start_idx = left_idx

        right_idx = self.Ip_idx
        Imin_right = self.Ip
        i = right_idx
        while right_idx < peak_end_idx and Imin_right > 0:
            if I[i] < Imin_right:
                Imin_right = I[i]
                right_idx = i
            i += 1
        self.E_end = E[right_idx]
        self.end_idx = right_idx
        
    def __repr__(self) -> str:
        # Formal representation of the object
        return f"Peak(Ip = {self.Ip}, Ep = {self.Ep}, Ip_idx = {self.Ip_idx}, E_start = {self.E_start}, E_end = {self.E_end}, start_idx = {self.start_idx}, end_idx = {self.end_idx})"

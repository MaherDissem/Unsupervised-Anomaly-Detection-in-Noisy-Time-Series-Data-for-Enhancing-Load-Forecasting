import numpy as np


class SynthLoadAnomaly():

    def _anomaly_type1(self, target, indices, lengths, k=0):
        """
        Anomaly type 1 that drops the power time series values to a negative value potentially followed by zero values
        before adding the missed sum of power to the end of the anomaly.
        """
        for idx, length in zip(indices, lengths):
            if length <= 2:
                raise Exception("Type 1 power anomalies must be longer than 2.")
            else:
                # WARNING: This could lead to a overflow quite fast?
                energy_at_start = target[:idx].sum() + k
                energy_at_end = target[:idx + length].sum() + k
                target[idx] = -1 * energy_at_start          # replace first by negative peak
                target[idx + 1:idx + length - 1] = 0        # set other values to zero
                target[idx + length - 1] = energy_at_end    # replace last with sum of missing values + k
        return target


    def _anomaly_type2(self, target, indices, lengths, softstart=True):
        """
        Anomaly type 2 that drops the power time series values to potentially zero and adds the missed sum of power to
        the end of the anomaly.
        """
        for idx, length in zip(indices, lengths):
            if length <= 1:
                raise Exception("Type 2 power anomalies must be longer than 1.")
            else:
                if softstart:
                    r = np.random.rand()
                    energy_consumed = target[idx:idx + length].sum()
                    target[idx] = r * target[idx]
                    target[idx + 1:idx + length - 1] = 0
                    target[idx + length - 1] = energy_consumed - target[idx]
                else:
                    energy_consumed = target[idx:idx + length].sum()
                    target[idx:idx + length - 1] = 0
                    target[idx + length - 1] = energy_consumed
        return target


    def _anomaly_type3(self, target, indices, lengths,
                        is_extreme=False, range_r=(0.01, 3.99), k=0):
        """
        Anomaly type 3 that creates a negatives peak in the power time series.
        """
        for idx, length in zip(indices, lengths):
            if length > 1:
                raise Exception("Type 3 power anomalies can't be longer than 1.")
            else:
                if is_extreme:
                    energy_consumed = target[:idx].sum()
                    target[idx] = -1 * energy_consumed - k
                else:
                    r = np.random.uniform(*range_r)
                    target[idx] = -1 * r * target[idx - 1]
        return target


    def _anomaly_type4(self, target, indices, lengths,
                    is_extreme=False, range_r=(0.01, 3.99), k=0):
        """
        Anomaly type 4 that creates a positive peak in the power time series.
        """
        for idx, length in zip(indices, lengths):
            if length > 1:
                raise Exception("Type 4 power anomalies can't be longer than 1.")
            else:
                if is_extreme:
                    energy_consumed = target[:idx].sum()
                    target[idx] = energy_consumed - k
                else:
                    r = np.random.uniform(*range_r)
                    target[idx] = r * target[idx - 1]
        return target


    def inject_anomaly(self, sequence, anom_type=0, n_anom=1, minimum_length=10):
        sequence = sequence.copy()
        n = len(sequence)
        # TODO add handling multiple number of anomalies

        if anom_type==0:
            anom_type = np.random.randint(1, 4)

        if anom_type==1:
            position = np.random.randint(n//4, (len(sequence)-1)//2)
            remaining_length = len(sequence)-1-position
            length = np.random.randint(max(3, (remaining_length//3*2)), remaining_length)
            return self._anomaly_type1(sequence.copy(), [position], [length]) # TODO add returning exact anom position 
        
        if anom_type==2:
            position = np.random.randint(n//4, (len(sequence)-1)//2)
            remaining_length = len(sequence)-1-position
            length = np.random.randint(max(2, (remaining_length//3*2)), remaining_length)
            return self._anomaly_type2(sequence.copy(), [position], [length])
        
        if anom_type==3:
            position = np.random.randint(n//4, (len(sequence)-1)//3*2)
            return self._anomaly_type3(sequence, [position], [1])
        
        if anom_type==4:
            position = np.random.randint(n//4, (len(sequence)-1)//3*2)
            return self._anomaly_type4(sequence, position, [1])

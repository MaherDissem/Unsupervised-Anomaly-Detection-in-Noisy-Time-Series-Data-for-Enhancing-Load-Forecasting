import numpy as np


class SynthLoadAnomaly():

    def __init__(
            self, 
            prob_1=0.25, prob_2=0.25, prob_3=0.25, prob_4=0.25, prob_softstart=0.5, prob_extreme=0.5, anomaly_max_length=12, 
            seed=0
        ):
        
        self.prob_1 = prob_1
        self.prob_2 = prob_2
        self.prob_3 = prob_3
        self.prob_4 = prob_4
        self.prob_softstart = prob_softstart
        self.prob_extreme = prob_extreme
        self.anomaly_max_length = anomaly_max_length
        self.seed = seed
        np.random.seed(seed)


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


    def _inject_anomaly(self, sequence, anom_type=1):
        sequence = sequence.copy()
        n = len(sequence)
        
        if anom_type==1:
            position = np.random.randint(n//4, (len(sequence)-1)//2)
            length = np.random.randint(3, min(n-position-1, self.anomaly_max_length))
            anom_idx = range(position, position+length)
            anomalous_sequence = self._anomaly_type1(sequence.copy(), [position], [length])
            return anomalous_sequence, anom_idx
        
        elif anom_type==2:
            position = np.random.randint(n//4, (len(sequence)-1)//2)
            length = np.random.randint(2, min(n-position-1, self.anomaly_max_length))
            anom_idx = range(position, position+length)
            softstart = np.random.choice([True, False], p=[self.prob_softstart, 1-self.prob_softstart])
            anomalous_sequence = self._anomaly_type2(sequence.copy(), [position], [length], softstart)
            return anomalous_sequence, anom_idx
        
        elif anom_type==3:
            position = np.random.randint(n//4, (len(sequence)-1)//4*3)
            anom_idx = range(position, position+1)
            is_extreme = np.random.choice([True, False], p=[self.prob_extreme, 1-self.prob_extreme])
            anomalous_sequence = self._anomaly_type3(sequence.copy(), [position], [1], is_extreme)
            return anomalous_sequence, anom_idx
        
        elif anom_type==4:
            position = np.random.randint(n//4, (len(sequence)-1)//4*3)
            anom_idx = range(position, position+1)
            is_extreme = np.random.choice([True, False], p=[self.prob_extreme, 1-self.prob_extreme])
            anomalous_sequence = self._anomaly_type4(sequence.copy(), [position], [1], is_extreme)
            return anomalous_sequence, anom_idx

        else:
            raise NotImplementedError("Anomaly type not implemented")


    def inject_anomaly(self, sequence, n_anomalies=1):
        anom_types = np.random.choice([1, 2, 3, 4], n_anomalies, p=[self.prob_1, self.prob_2, self.prob_3, self.prob_4])
        anom_indices = []
        for i in range(n_anomalies):
            sequence, anom_idx = self._inject_anomaly(sequence, anom_types[i])
            anom_indices.extend(anom_idx)
        return sequence, anom_indices

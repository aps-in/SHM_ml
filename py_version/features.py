
import numpy as np

class Featurizer():
    '''
    The class contains methods to obtain staistical features required.
    
    '''
    
    def __init__(self, data, axis = 1):
        self.data = data
        self.axis = axis
    
    def mean(self):
        ans = np.mean(self.data, self.axis)
        return ans

    def median(self):
        ans = np.median(self.data, self.axis)
        return ans

    def min_value(self):
        ans = np.min(self.data, self.axis)
        return ans

    def max_value(self):
        ans = np.max(self.data, self.axis)
        return ans

    def peak_to_peak(self):
        ans = np.max(self.data, self.axis) - np.min(self.data, self.axis)
        return ans

    def variance(self):
        ans = np.var(self.data, self.axis)
        return ans

    def rms(self):
        ans = np.sqrt(np.mean(self.data ** 2, self.axis))
        return ans

    def abs_mean(self):
        ans = np.mean(np.absolute(self.data), self.axis)
        return ans

    def shapefactor(self):
        ans = self.rms() / self.abs_mean()
        return ans

    def impulsefactor(self):
        ans = np.max(np.absolute(self.data), self.axis) / self.abs_mean()
        return ans

    def crestfactor(self):
        ans = np.max(np.absolute(self.data), self.axis) / np.sqrt(np.mean(self.data ** 2, self.axis))
        return ans

    def clearancefactor(self):
        ans = np.max(np.absolute(self.data), self.axis)
        ans /= ((np.mean(np.sqrt(np.absolute(self.data)), self.axis)) ** 2)
        return ans

    def std(self):
        ans = np.std(self.data, self.axis)
        return ans

    def skew(self):
        ans = scipy.stats.skew(self.data, self.axis)
        return ans

    def kurtosis(self):
        ans = scipy.stats.kurtosis(self.data, self.axis)
        return ans

    def abslogmean(self):
        ans = np.mean(np.log(np.abs(self.data)+1e-12), self.axis)
        return ans

    def meanabsdev(self):
        if self.axis == 0:
            ans = np.mean(np.abs(self.data - np.mean(self.data, self.axis)), self.axis)
        else:
            ans = np.mean(
                np.abs(self.data - np.mean(self.data, self.axis).reshape(self.data.shape[0], 1)), self.axis)
        return ans

    def medianabsdev(self):
        if self.axis == 0:
            ans = np.median(np.abs(self.data - np.median(self.data, self.axis)), self.axis)
        else:
            ans = np.median(
                np.abs(self.data - np.median(self.data, self.axis).reshape(self.data.shape[0], 1)), self.axis)
        return ans

    def midrange(self):
        ans = (np.max(self.data, self.axis) + np.min(self.data, self.axis)) / 2
        return ans

    def coeff_var(self):
        ans = scipy.stats.variation(self.data, self.axis)
        return ans
    
    all_funcs = [mean, median, min_value, max_value, peak_to_peak, variance, \
                rms, abs_mean, shapefactor, impulsefactor,crestfactor, clearancefactor, \
                std, skew, kurtosis, abslogmean, meanabsdev, medianabsdev, midrange, coeff_var]
    
    features = ['mean', 'median', 'min_value', 'max_value', 'peak_to_peak', 'variance', \
                'rms', 'abs_mean', 'shapefactor', 'impulsefactor', 'crestfactor', 'clearancefactor', \
                'std', 'skew', 'kurtosis', 'abslogmean', 'meanabsdev', 'medianabsdev', 'midrange', 'coeff_var']


if __name__ == "__main__":
    data = np.load('data.npy', allow_pickle = True)
    num_sensors = 16
    featurized_data = []
    for sensor in range(num_sensors):
        sensor_feature = []
        f = Featurizer(data[:, sensor, :], axis = 1)
        for func in f.all_funcs:
            sensor_feature.append(func(f))
        featurized_data.append(np.array(sensor_feature).T)
    featurized_data = np.array(featurized_data)
    print('Shape of the featurized data: ', featurized_data.shape)
    num_datapoints = featurized_data.shape[1]
    final_data = []
    for i in range(num_datapoints):
        final_data.append(featurized_data[:, i, :].ravel())
    final_data = np.array(final_data)
    print('Final shape of the featurized data: ', final_data.shape)
    np.save('featurized_data.npy', final_data)
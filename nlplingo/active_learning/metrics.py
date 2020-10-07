import numpy as np

def best_vs_second_best(predictions):
    """Computes best vs second best metric
        :type predictions: numpy.nparray
        :rtype: numpy.nparray
    """
    pred_sorted_arg = np.argsort(-predictions, axis=1)

    best_vs_second_best_score = 1 - abs(
        predictions[range(predictions.shape[0]), pred_sorted_arg[:, 0]] -
        predictions[range(predictions.shape[0]), pred_sorted_arg[:, 1]]
    )

    return best_vs_second_best_score

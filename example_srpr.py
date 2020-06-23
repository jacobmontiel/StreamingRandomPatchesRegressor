import numpy as np

from skmultiflow.data import FileStream
from skmultiflow.trees import HoeffdingTreeRegressor
from skmultiflow.evaluation import EvaluatePrequential

from streaming_random_patches_regressor import StreamingRandomPatchesRegressor


###############################################################################
#                                    Options                                  #
###############################################################################
SEED = 123456
n_estimators = 3
aggregation_method = 'median'            # 'median', 'mean'
drift_detection_criteria= 'prediciton'   # 'error', 'prediction'
subspace_mode = "randompatches"          # "randomsubspaces", "resampling", "randompatches"
###############################################################################

stream = FileStream('datasets/cal_housing.csv')

SRPR = StreamingRandomPatchesRegressor(n_estimators=n_estimators,
                                       aggregation_method=aggregation_method,
                                       random_state=SEED)
HTR = HoeffdingTreeRegressor(random_state=SEED)  # , leaf_prediction='mean')

evaluator = EvaluatePrequential(pretrain_size=0,
                                show_plot=True,
                                metrics=['mean_square_error',
                                         'mean_absolute_error',
                                         'true_vs_predicted']
                                )

evaluator.evaluate(stream=stream, model=[SRPR, HTR], model_names=['SRP-Reg', 'HT-Reg'])

print(SRPR.get_info())
print('DONE')

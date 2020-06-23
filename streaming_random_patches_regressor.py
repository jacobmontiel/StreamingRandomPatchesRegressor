import numpy as np

from skmultiflow.core import RegressorMixin
from skmultiflow.trees import HoeffdingTreeRegressor
from skmultiflow.drift_detection import ADWIN
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
from skmultiflow.metrics import RegressionMeasurements
from skmultiflow.utils import get_dimensions, check_random_state

from skmultiflow.meta import StreamingRandomPatchesClassifier
from skmultiflow.meta.streaming_random_patches import StreamingRandomPatchesBaseLearner


class StreamingRandomPatchesRegressor(RegressorMixin, StreamingRandomPatchesClassifier):
    """
    Streaming Random Patches regressor.

    Parameters
    ----------
    base_estimator: BaseSKMObject or sklearn.BaseObject, default=HoeffdingTreeClassifier
            The base estimator.

    n_estimators: int
        Number of members in the ensemble.

    subspace_mode: str, (default='percentage')
        | Defines how 'm, defined by ``subspace_size``, is interpreted.
        | `'`M`` represents the total number of features.
        | This only applies to subspaces and random patches options.
        | 'm' - Specified value
        | 'sqrtM1' - ``sqrt(M)+1``
        | 'MsqrtM1' - ``M-(sqrt(M)+1)``
        | 'percentage' - Percentage

    subspace_size: int, (default=60)
        Number of features per subset for each classifier.
        Negative value means ``total_features - subspace_size``.

    training_method: str, (default='randompatches')
        | The training method to use.
        | 'randomsubspaces' - Random subspaces
        | 'resampling' - Resampling (bagging)
        | 'randompatches' - Random patches

    lam: float, (default=6.0)
        Lambda value for bagging.

    drift_detection_method: BaseDriftDetector, (default=ADWIN(delta=1e-5))
        Drift detection method.

    warning_detection_method: BaseDriftDetector, (default=ADWIN(delta=1e-4))
        Warning detection method.

    disable_weighted_vote: bool (default=True)
        If True, uses weighted voting. Only applies if aggregation_method='mean'

    disable_drift_detection: bool (default=False)
        If True, disables drift detection and background learner.

    disable_background_learner: bool (default=False)
        If True, disables background learner and trees are reset immediately if drift is detected.

    drift_detection_criteria: str (default='error')
        | The criteria used to track drifts.
        | 'error' - absolute error
        | 'prediction' - predicted target values

    aggregation_method: str (default='mean')
        | The method to use to aggregate predictions in the ensemble.
        | 'mean'
        | 'median'

    nominal_attributes: list, optional
        List of Nominal attributes. If emtpy, then assume that all attributes are numerical.

    random_state: int, RandomState instance or None, optional (default=None)
       If int, random_state is the seed used by the random number generator;
       If RandomState instance, random_state is the random number generator;
       If None, the random number generator is the RandomState instance used
       by `np.random`.

    Notes
    -----
    The Streaming Random Patches [1]_ ensemble method for regression trains
    each base learner on a  subset of  features  and  instances from  the
    original  data, namely  a  random  patch. This  strategy  to  enforce
    diverse base  models  is  similar  to  the  one  in  the  random forest,
    yet it  is  not  restricted to  using  decision  trees  as  base  learner.

    This method is an adaptation of [2]_ for regression.

    References
    ----------
    .. [1] Heitor Gomes, Jacob Montiel, Saulo Martiello Mastelini,
       Bernhard Pfahringer, and Albert Bifet.
       On Ensemble Techniques for Data Stream Regression.
       IJCNN'20. International Joint Conference on Neural Networks. 2020.

    .. [2] Heitor Murilo Gomes, Jesse Read, Albert Bifet.
       Streaming Random Patches for Evolving Data Stream Classification.
       IEEE International Conference on Data Mining (ICDM), 2019.

    """
    _MEAN = 'mean'
    _MEDIAN = 'median'
    _ERROR = 'error'
    _PREDICTION = 'prediction'

    def __init__(self, base_estimator=HoeffdingTreeRegressor(grace_period=50,
                                                             split_confidence=0.01,
                                                             random_state=1),
                 n_estimators: int = 100,
                 subspace_mode: str = "percentage",
                 subspace_size: int = 60,
                 training_method: str = "randompatches",
                 lam: float = 6.0,
                 drift_detection_method: BaseDriftDetector = ADWIN(delta=1e-5),
                 warning_detection_method: BaseDriftDetector = ADWIN(delta=1e-4),
                 disable_weighted_vote: bool = True,
                 disable_drift_detection: bool = False,
                 disable_background_learner: bool = False,
                 drift_detection_criteria='error',
                 aggregation_method='mean',
                 nominal_attributes=None,
                 random_state=None):

        super().__init__(base_estimator=base_estimator,
                         n_estimators=n_estimators,
                         subspace_mode=subspace_mode,
                         subspace_size=subspace_size,
                         training_method=training_method,
                         lam=lam,
                         drift_detection_method=drift_detection_method,
                         warning_detection_method=warning_detection_method,
                         disable_weighted_vote=disable_weighted_vote,
                         disable_drift_detection=disable_drift_detection,
                         disable_background_learner=disable_background_learner,
                         nominal_attributes=nominal_attributes,
                         random_state=random_state)
        self._base_performance_evaluator = RegressionMeasurements()
        self._base_learner_class = StreamingRandomPatchesRegressorBaseLearner

        if aggregation_method not in {self._MEAN, self._MEDIAN}:
            raise ValueError("Invalid aggregation_method: {}.\n"
                             "Valid options are: {}".format(aggregation_method,
                                                            {self._MEAN, self._MEDIAN}))
        self.aggregation_method = aggregation_method

        if drift_detection_criteria not in {self._ERROR, self._PREDICTION}:
            raise ValueError("Invalid drift_detection_criteria: {}.\n"
                             "Valid options are: {}".format(drift_detection_criteria,
                                                            {self._ERROR, self._PREDICTION}))
        self.drift_detection_criteria = drift_detection_criteria

    def partial_fit(self, X, y, sample_weight=None):
        n_rows, n_cols = get_dimensions(X)

        if sample_weight is None:
            sample_weight = np.ones(n_rows)

        for i in range(n_rows):
            self._partial_fit(np.asarray([X[i]]), np.asarray([y[i]]),
                              sample_weight=np.asarray([sample_weight[i]]))

    def _partial_fit(self, X, y, sample_weight=None):
        self._n_samples_seen += 1
        _, n_features = get_dimensions(X)

        if not self.ensemble:
            self._init_ensemble(n_features)

        for i in range(len(self.ensemble)):
            # Get prediction for instance
            y_pred = self.ensemble[i].predict(X)

            # Update performance evaluator
            self.ensemble[i].performance_evaluator.add_result(y[0], y_pred[0])

            # Train using random subspaces without resampling,
            # i.e. all instances are used for training.
            if self.training_method == self._TRAIN_RANDOM_SUBSPACES:
                self.ensemble[i].partial_fit(X=X, y=y, sample_weight=np.asarray([1.]),
                                             n_samples_seen=self._n_samples_seen,
                                             random_state=self._random_state)
            # Train using random patches or resampling,
            # thus we simulate online bagging with Poisson(lambda=...)
            else:
                k = self._random_state.poisson(lam=self.lam)
                if k > 0:
                    self.ensemble[i].partial_fit(X=X, y=y, sample_weight=np.asarray([k]),
                                                 n_samples_seen=self._n_samples_seen,
                                                 random_state=self._random_state)

    def predict(self, X):
        n_samples, n_features = get_dimensions(X)
        y_pred = []

        if self.ensemble is None:
            self._init_ensemble(n_features=n_features)
            return np.zeros(n_samples)

        for i in range(n_samples):
            y_pred.append(self._predict(np.asarray([X[i]])))
        return np.asarray(y_pred)

    def _predict(self, X):
        y_pred = np.zeros(self.n_estimators)
        weights = None if self.disable_weighted_vote else np.zeros(self.n_estimators)

        for i in range(len(self.ensemble)):
            y_pred[i] = self.ensemble[i].predict(X)
            if not self.disable_weighted_vote:
                weights[i] = self.ensemble[i].performance_evaluator.get_average_error()

        if self.aggregation_method == self._MEAN:
            return np.average(a=y_pred, weights=weights)
        else:  # self.aggregation_method == self._MEDIAN:
            return np.median(y_pred)

    def predict_proba(self, X):
        raise NotImplementedError('predict_proba is not implemented for this method.')


class StreamingRandomPatchesRegressorBaseLearner(StreamingRandomPatchesBaseLearner):
    _ERROR = 'error'
    _PREDICTION = 'prediction'

    def __init__(self,
                 idx_original,
                 base_estimator,
                 performance_evaluator,
                 created_on,
                 disable_background_learner,
                 disable_drift_detector,
                 drift_detection_method,
                 warning_detection_method,
                 drift_detection_criteria,
                 is_background_learner,
                 feature_indexes=None,
                 nominal_attributes=None,
                 random_state=None):
        super().__init__(idx_original=idx_original,
                         base_estimator=base_estimator,
                         performance_evaluator=performance_evaluator,
                         created_on=created_on,
                         disable_background_learner=disable_background_learner,
                         disable_drift_detector=disable_drift_detector,
                         drift_detection_method=drift_detection_method,
                         warning_detection_method=warning_detection_method,
                         drift_detection_criteria=drift_detection_criteria,
                         is_background_learner=is_background_learner,
                         feature_indexes=feature_indexes,
                         nominal_attributes=nominal_attributes)

        # Background learner
        self._background_learner: StreamingRandomPatchesRegressorBaseLearner = None
        self._background_learner_class = StreamingRandomPatchesRegressorBaseLearner

        # Rest only applies when using periodic pseudo drift detectors

        # Use the same random_state object of the meta learner
        self.random_state = random_state
        self._random_state = check_random_state(self.random_state)

        # Drift detection
        self.drift_detection_criteria = drift_detection_criteria
        # If the drift detection method is periodic-fixed,
        # then set the shift option based on the instance index
        if isinstance(self.drift_detection_method, PeriodicTrigger):
            if self.drift_detection_method.trigger_method == PeriodicTrigger._FIXED_TRIGGER:
                self.drift_detection_method.set_params(w=self.idx_original)
            if self.drift_detection_method.trigger_method == PeriodicTrigger._RANDOM_TRIGGER:
                self.drift_detection_method.set_params(random_state=
                                                       check_random_state(self.random_state))

        if isinstance(self.warning_detection_method, PeriodicTrigger):
            if self.warning_detection_method.trigger_method == PeriodicTrigger._FIXED_TRIGGER:
                self.warning_detection_method.set_params(w=self.idx_original)
            if self.warning_detection_method.trigger_method == PeriodicTrigger._RANDOM_TRIGGER:
                self.warning_detection_method.set_params(random_state=
                                                         check_random_state(self.random_state))

        # Only used when paired with periodic drift detectors
        self.disable_warning_detector = False

    def partial_fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray,
                    n_samples_seen: int, random_state: np.random):
        n_features_total = get_dimensions(X)[1]
        if self.feature_indexes is not None:
            # Select the subset of features to use
            X_subset = np.asarray([X[0][self.feature_indexes]])
            if self._set_nominal_attributes and hasattr(self.base_estimator, 'nominal_attributes'):
                self.base_estimator.nominal_attributes = \
                    self._remap_nominal_attributes(self.feature_indexes, self.nominal_attributes)
                self._set_nominal_attributes = False
        else:
            # Use all features
            X_subset = X

        self.base_estimator.partial_fit(X=X_subset, y=y, sample_weight=sample_weight)
        if self.drift_detection_criteria == self._ERROR:
            # Track absolute error
            drift_detector_input = np.abs(self.base_estimator.predict(X_subset)[0] - y)
        else:   # self.drift_detection_criteria == self._PREDICTION
            # Track predicted target values
            drift_detector_input = self.base_estimator.predict(X_subset)[0]
        #
        if self._background_learner:
            # Note: Pass the original instance X so features are correctly selected
            # at the beginning of partial_fit
            self._background_learner.partial_fit(X=X, y=y, sample_weight=sample_weight,
                                                 n_samples_seen=n_samples_seen,
                                                 random_state=random_state)

        if not self.disable_drift_detector and not self.is_background_learner:
            # Check for warnings only if the background learner is active
            if not self.disable_background_learner and not self.disable_warning_detector:
                # Update the warning detection method
                self.warning_detection_method.add_element(drift_detector_input)
                # Check if there was a change
                if self.warning_detection_method.detected_change():
                    self.n_warnings_detected += 1
                    self._trigger_warning(n_features=n_features_total,
                                          n_samples_seen=n_samples_seen,
                                          random_state=random_state)
                    if isinstance(self.warning_detection_method, PeriodicTrigger):
                        self.disable_warning_detector = True

            # ===== Drift detection =====
            # Update the drift detection method
            self.drift_detection_method.add_element(drift_detector_input)
            # Check if the was a change
            if self.drift_detection_method.detected_change():
                self.n_drifts_detected += 1
                # There was a change, reset the model
                self.reset(n_features=n_features_total, n_samples_seen=n_samples_seen,
                           random_state=random_state)
                if isinstance(self.warning_detection_method, PeriodicTrigger):
                    self.disable_warning_detector = False

    def predict(self, X):
        if self.feature_indexes is not None:
            # Select the subset of features to use
            X_subset = np.asarray([X[0][self.feature_indexes]])
        else:
            # Use all features
            X_subset = X

        return self.base_estimator.predict(X_subset)

    def predict_proba(self, X):
        raise NotImplementedError('predict_proba is not implemented for this method.')


class PeriodicTrigger(BaseDriftDetector):
    """ Generates pseudo drift detection signals.

    There are two approaches:

    - 'Fixed' where the drift signal is generated every t_0 samples. In this case,
      the 'w' parameter can be used to indicate a shift of size ``(w * 0.1 * t_0)``.
    - 'random' corresponds to a pseudo-random drift detection strategy.

    Parameters
    ----------
    trigger_method: str (default='fixed')
        | The trigger method to use
        | 'fixed'
        | 'random'

    t_0: int (default=300)
        Reference point

    w: int (default=0)
        | Auxiliary parameter.
        | If method is 'fixed', the adds a shift of size (w * 0.1 * t_0) to the trigger point

    random_state: int, RandomState instance or None, optional (default=None)
       If int, random_state is the seed used by the random number generator;
       If RandomState instance, random_state is the random number generator;
       If None, the random number generator is the RandomState instance used
       by `np.random`.

    """
    _FIXED_TRIGGER = 'fixed'
    _RANDOM_TRIGGER = 'random'

    def __init__(self,
                 trigger_method='fixed',
                 t_0=300,
                 w=0,
                 random_state=None):

        super().__init__()
        if trigger_method not in {self._FIXED_TRIGGER, self._RANDOM_TRIGGER}:
            raise ValueError("Invalid trigger_method: {}.\n"
                             "Valid options are: {}".format(trigger_method,
                                                            {self._FIXED_TRIGGER,
                                                             self._RANDOM_TRIGGER}))
        self.trigger_method = trigger_method
        self.t_0 = t_0
        self.w = w
        self.random_state = random_state
        self._random_state = check_random_state(self.random_state)

        self.data_points_seen = 0

    def add_element(self, input_value=None):
        self.data_points_seen += 1

        if self.trigger_method == self._FIXED_TRIGGER:
            self._fixed_trigger()

        else:   # self.trigger_method == self._RANDOM_TRIGGER
            self._random_trigger()

    def _fixed_trigger(self):
        if self.data_points_seen > (self.t_0 + (self.w * int(self.t_0 * .1))):
            self.in_concept_change = True
            self.data_points_seen = 0

    def _random_trigger(self):
        t = self.data_points_seen
        t_0 = self.t_0
        W = self.w
        threshold = 1 / (1 + np.exp(-4 * (t - t_0) / W))
        self.in_concept_change = self._random_state.rand() < threshold

    def reset(self):
        self.data_points_seen = 0

package hex.ensemble;

import hex.Model;
import hex.ModelBuilder;
import hex.ModelCategory;

import hex.grid.Grid;
import water.DKV;
import water.Job;
import water.Key;
import water.Scope;
import water.exceptions.H2OIllegalArgumentException;
import water.exceptions.H2OModelBuilderIllegalArgumentException;
import water.fvec.Frame;
import water.fvec.Vec;

import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.stream.Stream;

import water.util.ArrayUtils;
import water.util.Log;

/**
 * An ensemble of other models, created by <i>stacking</i> with the SuperLearner algorithm or a variation.
 */
public class StackedEnsemble extends ModelBuilder<StackedEnsembleModel,StackedEnsembleModel.StackedEnsembleParameters,StackedEnsembleModel.StackedEnsembleOutput> {
  StackedEnsembleDriver _driver;
  // The in-progress model being built
  protected StackedEnsembleModel _model;

  public StackedEnsemble(StackedEnsembleModel.StackedEnsembleParameters parms) {
    super(parms);
    init(false);
  }

  public StackedEnsemble(boolean startup_once) {
    super(new StackedEnsembleModel.StackedEnsembleParameters(), startup_once);
  }

  @Override
  public ModelCategory[] can_build() {
    return new ModelCategory[]{
            ModelCategory.Regression,
            ModelCategory.Binomial,
            ModelCategory.Multinomial
    };
  }

  @Override
  public BuilderVisibility builderVisibility() {
    return BuilderVisibility.Stable;
  }

  @Override
  public boolean isSupervised() {
    return true;
  }

  @Override
  protected StackedEnsembleDriver trainModelImpl() {
    return _driver = _parms._blending == null ? new StackedEnsembleCVStackingDriver() : new StackedEnsembleBlendingDriver();
  }

  @Override
  public boolean haveMojo() {
    return true;
  }

  @Override
  public void init(boolean expensive) {
    super.init(expensive);

    checkFoldColumnPresent(_parms._metalearner_fold_column, train(), valid(), _parms.blending());
    validateAndExpandBaseModels();
  }

  /**
   * Validates base models and if grid is provided instead of a model it gets expanded.
   */
  private void validateAndExpandBaseModels() {
    // H2O Flow initializes SE with no base_models
    if (_parms._base_models == null) return;

    List<Key> baseModels = new ArrayList<Key>();
    for (Key baseModelKey : _parms._base_models) {
      Object retrievedObject = DKV.getGet(baseModelKey);
      if (retrievedObject instanceof Model) {
        baseModels.add(baseModelKey);
      } else if (retrievedObject instanceof Grid) {
        Grid grid = (Grid) retrievedObject;
        Collections.addAll(baseModels, grid.getModelKeys());
      } else if (retrievedObject == null) {
        throw new IllegalArgumentException(String.format("Specified id \"%s\" does not exist.", baseModelKey));
      } else {
        throw new IllegalArgumentException(String.format("Unsupported type \"%s\" as a base model.", retrievedObject.getClass().toString()));
      }
    }
    _parms._base_models = baseModels.toArray(new Key[0]);
  }

  /**
   * Checks for presence of a fold column in given {@link Frame}s. Null fold column means no checks are done.
   *
   * @param foldColumnName Name of the fold column. Null means no fold column has been specified
   * @param frames         A list of frames to check the presence of fold column in
   */
  private static void checkFoldColumnPresent(final String foldColumnName, final Frame... frames) {
    if (foldColumnName == null) return; // Unspecified fold column implies no checks are needs on provided frames 

    for (Frame frame : frames) {
      if (frame == null) continue; // No frame provided, no checks required
      if (frame.vec(foldColumnName) == null) {
        throw new IllegalArgumentException(String.format("Specified fold column '%s' not found in one of the supplied data frames. Available column names are: %s",
                foldColumnName, Arrays.toString(frame.names())));
      }
    }
  }

  static void addModelPredictionsToLevelOneFrame(Model aModel, Frame aModelsPredictions, Frame levelOneFrame) {
    if (aModel._output.isBinomialClassifier()) {
      // GLM uses a different column name than the other algos
      Vec preds = aModelsPredictions.vec(2); // Predictions column names have been changed...
      synchronized (levelOneFrame) {
        levelOneFrame.add(aModel._key.toString(), preds);
      }
    } else if (aModel._output.isMultinomialClassifier()) { //Multinomial
      //Need to remove 'predict' column from multinomial since it contains outcome
      Frame probabilities = aModelsPredictions.subframe(ArrayUtils.remove(aModelsPredictions.names(), "predict"));
      probabilities.setNames(
              Stream.of(probabilities.names())
                      .map((name) -> aModel._key.toString().concat("/").concat(name))
                      .toArray(String[]::new)
      );
      synchronized (levelOneFrame) {
        levelOneFrame.add(probabilities);
      }
    } else if (aModel._output.isAutoencoder()) {
      throw new H2OIllegalArgumentException("Don't yet know how to stack autoencoders: " + aModel._key);
    } else if (!aModel._output.isSupervised()) {
      throw new H2OIllegalArgumentException("Don't yet know how to stack unsupervised models: " + aModel._key);
    } else {
      Vec preds = aModelsPredictions.vec("predict");
      synchronized (levelOneFrame) {
        levelOneFrame.add(aModel._key.toString(), preds);
      }
    }
  }

  private abstract class StackedEnsembleDriver extends Driver {

    /**
     * Prepare a "level one" frame for a given set of models, predictions-frames and actuals.  Used for preparing
     * training and validation frames for the metalearning step, and could also be used for bulk predictions for
     * a StackedEnsemble.
     */
    private Frame prepareLevelOneFrame(String levelOneKey, Model[] baseModels, Frame[] baseModelPredictions, Frame actuals) {
      if (null == baseModels) throw new H2OIllegalArgumentException("Base models array is null.");
      if (null == baseModelPredictions) throw new H2OIllegalArgumentException("Base model predictions array is null.");
      if (baseModels.length == 0) throw new H2OIllegalArgumentException("Base models array is empty.");
      if (baseModelPredictions.length == 0)
        throw new H2OIllegalArgumentException("Base model predictions array is empty.");
      if (baseModels.length != baseModelPredictions.length)
        throw new H2OIllegalArgumentException("Base models and prediction arrays are different lengths.");

      if (null == levelOneKey) levelOneKey = "levelone_" + _model._key.toString();
      Frame levelOneFrame = new Frame(Key.<Frame>make(levelOneKey));

      for (int i = 0; i < baseModels.length; i++) {
        Model baseModel = baseModels[i];
        Frame baseModelPreds = baseModelPredictions[i];

        if (null == baseModel) {
          Log.warn("Failed to find base model; skipping: " + baseModels[i]);
          continue;
        }
        if (null == baseModelPreds) {
          Log.warn("Failed to find base model " + baseModel + " predictions; skipping: " + baseModelPreds._key);
          continue;
        }
        StackedEnsemble.addModelPredictionsToLevelOneFrame(baseModel, baseModelPreds, levelOneFrame);
      }
      // Add metalearner_fold_column to level one frame if it exists
      if (_model._parms._metalearner_fold_column != null) {
        Vec foldColumn = actuals.vec(_model._parms._metalearner_fold_column);
        levelOneFrame.add(_model._parms._metalearner_fold_column, foldColumn);
      }
      // Add response column to level one frame
      Vec responseColumn = actuals.vec(_model.responseColumn);
      levelOneFrame.add(_model.responseColumn, responseColumn);

      // TODO: what if we're running multiple in parallel and have a name collision?

      Frame old = DKV.getGet(levelOneFrame._key);
      if (old != null && old instanceof Frame) {
        Frame oldFrame = (Frame) old;
        // Remove ALL the columns so we don't delete them in remove_impl.  Their
        // lifetime is controlled by their model.
        oldFrame.removeAll();
        oldFrame.write_lock(_job);
        oldFrame.update(_job);
        oldFrame.unlock(_job);
      }

      levelOneFrame.delete_and_lock(_job);
      levelOneFrame.unlock(_job);
      Log.info("Finished creating \"level one\" frame for stacking: " + levelOneFrame.toString());
      DKV.put(levelOneFrame);
      return levelOneFrame;
    }

    /**
     * Prepare a "level one" frame for a given set of models and actuals.
     * Used for preparing validation frames for the metalearning step, and could also be used for bulk predictions for a StackedEnsemble.
     */
    private Frame prepareLevelOneFrame(String levelOneKey, Key<Model>[] baseModelKeys, Frame actuals, boolean isTraining) {
      List<Model> baseModels = new ArrayList<>();
      List<Frame> baseModelPredictions = new ArrayList<>();

      for (Key<Model> k : baseModelKeys) {
        if (_model._output._metalearner == null || _model.isUsefulBaseModel(k)) {
          Model aModel = DKV.getGet(k);
          if (null == aModel)
            throw new H2OIllegalArgumentException("Failed to find base model: " + k);

          Frame predictions = getPredictionsForBaseModel(aModel, actuals, isTraining);
          baseModels.add(aModel);
          baseModelPredictions.add(predictions);
        }
      }
      boolean keepLevelOneFrame = isTraining && _parms._keep_levelone_frame;
      Frame levelOneFrame = prepareLevelOneFrame(levelOneKey, baseModels.toArray(new Model[0]), baseModelPredictions.toArray(new Frame[0]), actuals);
      if (keepLevelOneFrame) {
        levelOneFrame = levelOneFrame.deepCopy(levelOneFrame._key.toString());
        levelOneFrame.write_lock(_job);
        levelOneFrame.update(_job);
        levelOneFrame.unlock(_job);
        Scope.untrack(levelOneFrame.keysList());
      }
      return levelOneFrame;
    }

    protected Frame buildPredictionsForBaseModel(Model model, Frame frame) {
      Key<Frame> predsKey = buildPredsKey(model, frame);
      Frame preds = DKV.getGet(predsKey);
      if (preds == null) {
        preds =  model.score(frame, predsKey.toString(), null, false);  // no need for metrics here (leaks in client mode)
        Scope.untrack(preds.keysList());
      }
      if (_model._output._base_model_predictions_keys == null)
        _model._output._base_model_predictions_keys = new Key[0];

      if (!ArrayUtils.contains(_model._output._base_model_predictions_keys, predsKey)){
        _model._output._base_model_predictions_keys = ArrayUtils.append(_model._output._base_model_predictions_keys, predsKey);
      }
      //predictions are cleaned up by metalearner if necessary
      return preds;
    }

    protected abstract StackedEnsembleModel.StackingStrategy strategy();

    /**
     * @RETURN THE FRAME THAT IS USED TO COMPUTE THE PREDICTIONS FOR THE LEVEL-ONE TRAINING FRAME.
     */
    protected abstract Frame getActualTrainingFrame();

    protected abstract Frame getPredictionsForBaseModel(Model model, Frame actualsFrame, boolean isTrainingFrame);

    private Key<Frame> buildPredsKey(Key model_key, long model_checksum, Key frame_key, long frame_checksum) {
      return Key.make("preds_" + model_checksum + "_on_" + frame_checksum);
    }

    protected Key<Frame> buildPredsKey(Model model, Frame frame) {
      return frame == null || model == null ? null : buildPredsKey(model._key, model.checksum(), frame._key, frame.checksum());
    }

    public void computeImpl() {
      init(true);
      if (error_count() > 0) throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(StackedEnsemble.this);

      _model = new StackedEnsembleModel(dest(), _parms, new StackedEnsembleModel.StackedEnsembleOutput(StackedEnsemble.this));
      _model._output._stacking_strategy = strategy();
      try {
        _model.delete_and_lock(_job); // and clear & write-lock it (smashing any prior)
        _model.checkAndInheritModelProperties();
        _model.update(_job);
      } finally {
        _model.unlock(_job);
      }

      String levelOneTrainKey = "levelone_training_" + _model._key.toString();
      Frame levelOneTrainingFrame = prepareLevelOneFrame(levelOneTrainKey, _model._parms._base_models, getActualTrainingFrame(), true);
      Frame levelOneValidationFrame = null;
      if (_model._parms.valid() != null) {
        String levelOneValidKey = "levelone_validation_" + _model._key.toString();
        levelOneValidationFrame = prepareLevelOneFrame(levelOneValidKey, _model._parms._base_models, _model._parms.valid(), false);
      }

      Metalearner.Algorithm metalearnerAlgoSpec = _model._parms._metalearner_algorithm;
      Metalearner.Algorithm metalearnerAlgoImpl = Metalearners.getActualMetalearnerAlgo(metalearnerAlgoSpec);

      // Compute metalearner
      if(metalearnerAlgoImpl != null) {
        Key<Model> metalearnerKey = Key.<Model>make("metalearner_" + metalearnerAlgoSpec + "_" + _model._key);

        Job metalearnerJob = new Job<>(metalearnerKey, ModelBuilder.javaName(metalearnerAlgoImpl.toString()),
                "StackingEnsemble metalearner (" + metalearnerAlgoSpec + ")");
        //Check if metalearner_params are passed in
        boolean hasMetaLearnerParams = _model._parms._metalearner_parameters != null;
        long metalearnerSeed = _model._parms._seed;

        Metalearner metalearner = Metalearners.createInstance(metalearnerAlgoSpec.name());
        metalearner.init(
                levelOneTrainingFrame,
                levelOneValidationFrame,
                _model._parms._metalearner_parameters,
                _model,
                _job,
                metalearnerKey,
                metalearnerJob,
                _parms,
                hasMetaLearnerParams,
                metalearnerSeed
        );
        metalearner.compute();
      } else {
        throw new H2OIllegalArgumentException("Invalid `metalearner_algorithm`. Passed in " + metalearnerAlgoSpec +
                " but must be one of " + Arrays.toString(Metalearner.Algorithm.values()));
      }
    } // computeImpl
  }

  private class StackedEnsembleCVStackingDriver extends StackedEnsembleDriver {

    @Override
    protected StackedEnsembleModel.StackingStrategy strategy() {
      return StackedEnsembleModel.StackingStrategy.cross_validation;
    }

    @Override
    protected Frame getActualTrainingFrame() {
      return _model._parms.train();
    }

    @Override
    protected Frame getPredictionsForBaseModel(Model model, Frame actualsFrame, boolean isTraining) {
      Frame fr;
      if (isTraining) {
        // for training, retrieve predictions from cv holdout predictions frame as all base models are required to get built with keep_cross_validation_frame=true
        if (null == model._output._cross_validation_holdout_predictions_frame_id)
          throw new H2OIllegalArgumentException("Failed to find the xval predictions frame id. . .  Looks like keep_cross_validation_predictions wasn't set when building the models.");

        fr = DKV.getGet(model._output._cross_validation_holdout_predictions_frame_id);

        if (null == fr)
          throw new H2OIllegalArgumentException("Failed to find the xval predictions frame. . .  Looks like keep_cross_validation_predictions wasn't set when building the models, or the frame was deleted.");

      } else {
        fr = buildPredictionsForBaseModel(model, actualsFrame);
      }
      return fr;
    }

  }

  private class StackedEnsembleBlendingDriver extends StackedEnsembleDriver {

    @Override
    protected StackedEnsembleModel.StackingStrategy strategy() {
      return StackedEnsembleModel.StackingStrategy.blending;
    }

    @Override
    protected Frame getActualTrainingFrame() {
      return _model._parms.blending();
    }

    @Override
    protected Frame getPredictionsForBaseModel(Model model, Frame actualsFrame, boolean isTrainingFrame) {
      return buildPredictionsForBaseModel(model, actualsFrame);
    }
  }

}

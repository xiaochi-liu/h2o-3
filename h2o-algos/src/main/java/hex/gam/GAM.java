package hex.gam;

import hex.DataInfo;
import hex.ModelBuilder;
import hex.ModelCategory;
import hex.ModelMetrics;
import hex.gam.GAMModel.GAMParameters;
import hex.gam.MatrixFrameUtils.GamUtils;
import hex.gam.MatrixFrameUtils.GenerateGamMatrixOneColumn;
import hex.glm.GLM;
import hex.glm.GLMModel;
import hex.glm.GLMModel.GLMParameters;
import hex.quantile.Quantile;
import hex.quantile.QuantileModel;
import jsr166y.ForkJoinTask;
import jsr166y.RecursiveAction;
import water.DKV;
import water.Key;
import water.MemoryManager;
import water.Scope;
import water.exceptions.H2OModelBuilderIllegalArgumentException;
import water.fvec.Frame;
import water.fvec.Vec;
import water.util.ArrayUtils;
import water.util.Log;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static hex.gam.GAMModel.cleanUpInputFrame;
import static hex.gam.MatrixFrameUtils.GamUtils.AllocateType.*;
import static hex.gam.MatrixFrameUtils.GamUtils.*;
import static hex.glm.GLMModel.GLMParameters.Family.multinomial;
import static hex.glm.GLMModel.GLMParameters.Family.ordinal;


public class GAM extends ModelBuilder<GAMModel, GAMModel.GAMParameters, GAMModel.GAMModelOutput> {

  @Override
  public ModelCategory[] can_build() {
    return new ModelCategory[]{ModelCategory.Regression};
  }

  @Override
  public boolean isSupervised() {
    return true;
  }

  @Override
  public BuilderVisibility builderVisibility() {
    return BuilderVisibility.Experimental;
  }

  @Override
  public boolean havePojo() {
    return false;
  }

  @Override
  public boolean haveMojo() {
    return false;
  }

  public GAM(boolean startup_once) {
    super(new GAMModel.GAMParameters(), startup_once);
  }

  public GAM(GAMModel.GAMParameters parms) {
    super(parms);
    init(false);
  }

  public GAM(GAMModel.GAMParameters parms, Key<GAMModel> key) {
    super(parms, key);
    init(false);
  }
  
  double[][] _knots;

  /***
   * This method will look at the keys of knots stored in _parms._knot_ids and copy them over to double[][]
   * array.
   *
   * @return double[][] array containing the knots specified by users
   */
  public double[][] generateKnotsFromKeys() {
    int numGamCols = _parms._gam_columns.length;
    double[][] knots = new double[numGamCols][];
    boolean allNull = _parms._knot_ids ==null;

    for (int index=0; index < numGamCols; index++) {
      final Frame predictVec = new Frame(new String[]{_parms._gam_columns[index]}, new Vec[]{_parms.train().vec(_parms._gam_columns[index])});
      String tempKey = allNull?null:_parms._knot_ids[index];
      if (tempKey != null && (tempKey.length() > 0)) {  // read knots location from Frame given by user
        final Frame knotFrame = Scope.track((Frame)DKV.getGet(Key.make(tempKey)));
        double[][] knotContent = new double[(int)knotFrame.numRows()][1];
        final ArrayUtils.FrameToArray f2a = new ArrayUtils.FrameToArray(0,0, knotFrame.numRows(), knotContent);
        knotContent = f2a.doAll(knotFrame).getArray();
        knots[index] = new double[knotContent.length];
        final double[][] knotCTranspose = ArrayUtils.transpose(knotContent);
        System.arraycopy(knotCTranspose[0],0,knots[index], 0, knots[index].length);
        failVerifyKnots(knots[index]);
      } else {  // current column knotkey is null
        knots[index] = generateKnotsOneColumn(predictVec, _parms._num_knots[index]);
        failVerifyKnots(knots[index]);
      }
    }
    return knots;
  }
  
  // this function will check and make sure the knots location specified in knots are valid in the following sense:
  // 1. They do not contain NaN
  // 2. They are sorted in ascending order.
  public void failVerifyKnots(double[] knots) {
    for (int index = 0; index < knots.length; index++) {
      if (Double.isNaN(knots[index])) {
        error("knots formation", "default knots generation contain NaN.   Please specify it" +
                " manually");
        return;
      }
      if (index > 0 && knots[index - 1] >= knots[index]) { // knots must be increasing in order
        error("knots formation", String.format("knots not sorted in ascending order. Knots at index %d: %f." +
                "  Knots at index %d: %f", index-1, knots[index-1], index, knots[index]));
        return;
      }
    }
  }

  // This method will generate knot locations by choosing them from a uniform quantile distribution of that
  // chosen column.
  public double[] generateKnotsOneColumn(Frame gamFrame, int knotNum) {
    double[] knots = MemoryManager.malloc8d(knotNum);
    try {
      Scope.enter();
      Frame tempFrame = new Frame(gamFrame);  // make sure we have a frame key
      DKV.put(tempFrame);
      double[] prob = MemoryManager.malloc8d(knotNum);
      assert knotNum > 1;
      double stepProb = 1.0 / (knotNum - 1);
      for (int knotInd = 0; knotInd < knotNum; knotInd++)
        prob[knotInd] = knotInd * stepProb;
      QuantileModel.QuantileParameters parms = new QuantileModel.QuantileParameters();
      parms._train = tempFrame._key;
      parms._probs = prob;
      QuantileModel qModel = new Quantile(parms).trainModel().get();
      DKV.remove(tempFrame._key);
      Scope.track_generic(qModel);
      System.arraycopy(qModel._output._quantiles[0], 0, knots, 0, knotNum);
    } finally {
      Scope.exit();
    }
    return knots;
  }
  
  @Override
  public void init(boolean expensive) {
    super.init(expensive);
    if (expensive) {  // add custom check here
      if (error_count() > 0)
        throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(GAM.this);

      if (_parms._gam_columns == null)  // check _gam_columns contains valid columns
        error("_gam_columns", "must specify columns names to apply GAM to.  If you don't have any," +
                " use GLM.");
      else {  // check and make sure gam_columns column types are legal
        Frame dataset = _parms.train();
        List<String> cNames = Arrays.asList(dataset.names());
        for (int index = 0; index < _parms._gam_columns.length; index++) {
          String cname = _parms._gam_columns[index];
          if (!cNames.contains(cname))
            error("gam_columns", "column name: " + cname + " does not exist in your dataset.");
          if (dataset.vec(cname).isCategorical())
            error("gam_columns", "column " + cname + " is categorical and cannot be used as a gam " +
                    "column.");
          if (dataset.vec(cname).isBad() || dataset.vec(cname).isTime() || dataset.vec(cname).isUUID() || 
          dataset.vec(cname).isConst())
            error("gam_columns", String.format("Column '%s' of type '%s' cannot be used as GAM column. Column types " +
                    "BAD, TIME, CONSTANT and UUID cannot be used.", cname, dataset.vec(cname).get_type_str()));
          if (!dataset.vec(cname).isNumeric())
            error("gam_columns", "column " + cname + " is not numerical and cannot be used as a gam" +
                    " column.");
        }
      }
      if ((_parms._bs != null) && (_parms._gam_columns.length != _parms._bs.length))  // check length
        error("gam colum number","Number of gam columns implied from _bs and _gam_columns do not match.");
      if (_parms._bs == null) // default to bs type 0
        _parms._bs = new int[_parms._gam_columns.length];
      if (_parms._num_knots == null) {  // user did not specify any knot numbers, we will use default 10
        _parms._num_knots = new int[_parms._gam_columns.length];  // different columns may have different
        for (int index = 0; index < _parms._gam_columns.length; index++) {  // for zero value _num_knots, set to valid number
          if (_parms._num_knots[index] == 0) {
            long numRowMinusNACnt = _train.numRows() - _parms.train().vec(_parms._gam_columns[index]).naCnt();
            _parms._num_knots[index] = numRowMinusNACnt < 10 ? (int) numRowMinusNACnt : 10;
          }
        }
      }
      int cindex = 0;
      for (int numKnots : _parms._num_knots) {  // check to make sure numKnot is valid
        long eligibleRows = _train.numRows() - _parms.train().vec(_parms._gam_columns[cindex]).naCnt();
        if (numKnots > eligibleRows) {
          error("_num_knots", " number of knots specified in _num_knots: " + _parms._num_knots[cindex] + " exceed number " +
                  "of rows in training frame minus NA rows: " + eligibleRows + ".  Reduce _num_knots.");
        }
        cindex++;
      }
      if ((_parms._num_knots != null) && (_parms._num_knots.length != _parms._gam_columns.length))
        error("gam colum number","Number of gam columns implied from _num_knots and _gam_columns do not match.");
      if (_parms._knot_ids !=null) { // check knots location specification
        if (_parms._knot_ids.length != _parms._gam_columns.length)
          error("gam colum number", "Number of gam columns implied from _num_knots and _knot_ids do not" +
                  " match.");
      }
      _knots = generateKnotsFromKeys(); // generate knots and verify that they are given correctly
      if ( _parms._saveZMatrix && ((_train.numCols() - 1 + _parms._num_knots.length) < 2))
        error("_saveZMatrix", "can only be enabled if we number of predictors plus" +
                " Gam columns in gam_columns exceeds 2");
      if ((_parms._lambda_search || !_parms._intercept || _parms._lambda == null || _parms._lambda[0] > 0))
        _parms._use_all_factor_levels = true;
      if (_parms._link == null || _parms._link.equals(GLMParameters.Link.family_default))
        _parms._link = _parms._family.defaultLink;
    }
    
    if ((_parms._family == GLMParameters.Family.multinomial || _parms._family == GLMParameters.Family.binomial)
            && response().get_type() != Vec.T_CAT){
      error("_response_column", String.format("For given response family '%s', please provide a categorical" +
              " response column. Current response column type is '%s'.", _parms._family, response().get_type_str()));
    }
  }

  @Override
  protected boolean computePriorClassDistribution() {
    return (_parms._family== multinomial)||(_parms._family== ordinal);
  }

  @Override
  protected GAMDriver trainModelImpl() {
    return new GAMDriver();
  }

  @Override
  protected int nModelsInParallel(int folds) {
    return nModelsInParallel(folds, 2);
  }

  private class GAMDriver extends Driver {
    double[][][] _zTranspose; // store for each GAM predictor transpose(Z) matrix
    double[][][] _penalty_mat_center;  // store for each GAM predictor the penalty matrix
    double[][][] _penalty_mat;  // penalty matrix before centering
    public double[][][] _binvD; // store BinvD for each gam column specified for scoring
    public int[] _numKnots;  // store number of knots per gam column
    String[][] _gamColNames;  // store column names of GAM columns
    String[][] _gamColNamesCenter;  // gamColNames after de-centering is performed.
    Key<Frame>[] _gamFrameKeys;
    Key<Frame>[] _gamFrameKeysCenter;

    /***
     * This method will take the _train that contains the predictor columns and response columns only and add to it
     * the following:
     * 1. For each predictor included in gam_columns, expand it out to calculate the f(x) and attach to the frame.
     * 2. It will calculate the ztranspose that is used to center the gam columns.
     * 3. It will calculate a penalty matrix used to control the smoothness of GAM.
     *
     * @return
     */
    Frame adaptTrain() {
      int numGamFrame = _parms._gam_columns.length;
      _zTranspose = GamUtils.allocate3DArray(numGamFrame, _parms, firstOneLess);
      _penalty_mat = _parms._savePenaltyMat?GamUtils.allocate3DArray(numGamFrame, _parms, sameOrig):null;
      _penalty_mat_center = GamUtils.allocate3DArray(numGamFrame, _parms, bothOneLess);
      _binvD = GamUtils.allocate3DArray(numGamFrame, _parms, firstTwoLess);
      _numKnots = MemoryManager.malloc4(numGamFrame);
      _gamColNames = new String[numGamFrame][];
      _gamColNamesCenter = new String[numGamFrame][];
      _gamFrameKeys = new Key[numGamFrame];
      _gamFrameKeysCenter = new Key[numGamFrame];

      addGAM2Train();  // add GAM columns to training frame
      return buildGamFrame(numGamFrame, _gamFrameKeysCenter, _train, _parms._response_column); // add gam cols to _train
    }

    void addGAM2Train() {
      int numGamFrame = _parms._gam_columns.length;
      RecursiveAction[] generateGamColumn = new RecursiveAction[numGamFrame];
      for (int index = 0; index < numGamFrame; index++) {
        final Frame predictVec = new Frame(new String[]{_parms._gam_columns[index]}, new Vec[]{_parms.train().vec(_parms._gam_columns[index])});  // extract the vector to work on
        final int numKnots = _parms._num_knots[index];  // grab number of knots to generate
        final int numKnotsM1 = numKnots - 1;
        final int splineType = _parms._bs[index];
        final int frameIndex = index;
 //       final boolean nullKnots = _knots[frameIndex]==null;
        final String[] newColNames = new String[numKnots];
        for (int colIndex = 0; colIndex < numKnots; colIndex++) {
          newColNames[colIndex] = _parms._gam_columns[index] + "_" + splineType + "_" + colIndex;
        }
        _gamColNames[frameIndex] = new String[numKnots];
        _gamColNamesCenter[frameIndex] = new String[numKnotsM1];
        System.arraycopy(newColNames, 0, _gamColNames[frameIndex], 0, numKnots);
        generateGamColumn[frameIndex] = new RecursiveAction() {
          @Override
          protected void compute() {
            GenerateGamMatrixOneColumn genOneGamCol = new GenerateGamMatrixOneColumn(splineType, numKnots, 
                    _knots[frameIndex], predictVec,
                    _parms._standardize).doAll(numKnots, Vec.T_NUM, predictVec);
            if (_parms._savePenaltyMat)  // only save this for debugging
              GamUtils.copy2DArray(genOneGamCol._penaltyMat, _penalty_mat[frameIndex]); // copy penalty matrix
            // calculate z transpose
              Frame oneAugmentedColumnCenter = genOneGamCol.outputFrame(Key.make(), newColNames,
                      null);
              oneAugmentedColumnCenter = genOneGamCol.centralizeFrame(oneAugmentedColumnCenter,
                      predictVec.name(0) + "_" + splineType + "_center_", _parms);
              GamUtils.copy2DArray(genOneGamCol._ZTransp, _zTranspose[frameIndex]); // copy transpose(Z)
              double[][] transformedPenalty = ArrayUtils.multArrArr(ArrayUtils.multArrArr(genOneGamCol._ZTransp,
                      genOneGamCol._penaltyMat), ArrayUtils.transpose(genOneGamCol._ZTransp));  // transform penalty as zt*S*z
              GamUtils.copy2DArray(transformedPenalty, _penalty_mat_center[frameIndex]);
              _gamFrameKeysCenter[frameIndex] = oneAugmentedColumnCenter._key;
              DKV.put(oneAugmentedColumnCenter);
              System.arraycopy(oneAugmentedColumnCenter.names(), 0, _gamColNamesCenter[frameIndex], 0,
                      numKnotsM1);
            GamUtils.copy2DArray(genOneGamCol._bInvD, _binvD[frameIndex]);
            _numKnots[frameIndex] = genOneGamCol._numKnots;
          }
        };
      }
      ForkJoinTask.invokeAll(generateGamColumn);
    }
    
    void verifyGamTransformedFrame(Frame gamTransformed) {
      int numGamCols = _gamColNamesCenter.length;
      int numGamFrame = _parms._gam_columns.length;
      for (int findex = 0; findex < numGamFrame; findex++) {
        for (int index = 0; index < numGamCols; index++) {
          if (gamTransformed.vec(_gamColNamesCenter[findex][index]).isConst())
            error(_gamColNamesCenter[findex][index], "gam column transformation generated constant columns" +
                    " for " + _parms._gam_columns[findex]);
        }
      }
    }
    
    @Override
    public void computeImpl() {
      init(true);     //this can change the seed if it was set to -1
      if (error_count() > 0)   // if something goes wrong, let's throw a fit
        throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(GAM.this);

      Frame newTFrame = new Frame(rebalance(adaptTrain(), false, _result+".temporary.train"));  // get frames with correct predictors and spline functions
      verifyGamTransformedFrame(newTFrame);
      
      if (error_count() > 0)   // if something goes wrong during gam transformation, let's throw a fit again!
        throw H2OModelBuilderIllegalArgumentException.makeFromBuilder(GAM.this);
      
      if (valid() != null) {  // transform the validation frame if present
        _valid = cleanUpInputFrame(valid().clone(), valid().names(), _parms, _gamColNamesCenter, _binvD, _zTranspose, _knots,
                _numKnots);
        Scope.track(_valid);
        DKV.put(_valid);
      }
      DKV.put(newTFrame); // This one will cause deleted vectors if add to Scope.track
      _job.update(0, "Initializing model training");
      buildModel(newTFrame); // build gam model 
    }

    public final void buildModel(Frame newTFrame) {
      GAMModel model = null;
      DataInfo dinfo = null;
      try {
        _job.update(0, "Adding GAM columns to training dataset...");
        dinfo = new DataInfo(_train.clone(), _valid, 1, _parms._use_all_factor_levels 
                || _parms._lambda_search, _parms._standardize ? DataInfo.TransformType.STANDARDIZE : DataInfo.TransformType.NONE, DataInfo.TransformType.NONE,
                _parms.missingValuesHandling() == GLMParameters.MissingValuesHandling.Skip,
                _parms.missingValuesHandling() == GLMParameters.MissingValuesHandling.MeanImputation || _parms.missingValuesHandling() == GLMParameters.MissingValuesHandling.PlugValues,
                _parms.makeImputer(),
                false, hasWeightCol(), hasOffsetCol(), hasFoldCol(), _parms.interactionSpec());
        DKV.put(dinfo._key, dinfo);
        model = new GAMModel(dest(), _parms, new GAMModel.GAMModelOutput(GAM.this, dinfo._adaptedFrame, dinfo));
        model.delete_and_lock(_job);
        if (_parms._keep_gam_cols) {  // save gam column keys
          model._output._gamTransformedTrainCenter = newTFrame._key;

        }
        _job.update(1, "calling GLM to build GAM model...");
        GLMModel glmModel = buildGLMModel(_parms, newTFrame, _valid==null?null:_valid._key); // obtained GLM model
        Scope.track_generic(glmModel);
        _job.update(0, "Building out GAM model...");
        fillOutGAMModel(glmModel, model, dinfo); // build up GAM model
        model.update(_job);
        // build GAM Model Metrics
        _job.update(0, "Scoring training frame");
        scoreGenModelMetrics(model, train(), true); // score training dataset and generate model metrics
        if (valid() != null)
          scoreGenModelMetrics(model, valid(), false); // score validation dataset and generate model metrics
      } finally {
        List<Key<Vec>> keep = new ArrayList<>();
        if (model != null) {
          if (_parms._keep_gam_cols) {
            addFrameKeys2Keep(keep, newTFrame._key);
          }
          model.unlock(_job);
          Scope.untrack(keep);  // leave the vectors alone.
        }
        if (dinfo!=null)
          dinfo.remove();
      }
    }
    
    /**
     * This part will perform scoring and generate the model metrics for training data and validation data if 
     * provided by user.
     *      
     * @param model
     * @param scoreFrame
     * @param forTraining: true for training dataset and false for validation dataset
     */
    private void scoreGenModelMetrics(GAMModel model, Frame scoreFrame, boolean forTraining) {
      Frame scoringTrain = new Frame(scoreFrame);
      model.adaptTestForTrain(scoringTrain, true, true);
      Frame scoredResult = model.score(scoringTrain);
      scoredResult.delete();
      ModelMetrics mtrain = ModelMetrics.getFromDKV(model, scoringTrain);
      if (mtrain!=null) {
        if (forTraining)
          model._output._training_metrics = mtrain;
        else 
          model._output._validation_metrics = mtrain;
        Log.info("GAM[dest="+dest()+"]"+mtrain.toString());
      } else {
        Log.info("Model metrics is empty!");
      }
    }

    GLMModel buildGLMModel(GAMParameters parms, Frame trainData, Key<Frame> validKey) {
      GLMParameters glmParam = GamUtils.copyGAMParams2GLMParams(parms, trainData, validKey);  // copy parameter from GAM to GLM
      int numGamCols = _parms._gam_columns.length;
      for (int find = 0; find < numGamCols; find++) {
        if ((_parms._scale != null) && (_parms._scale[find] != 1.0))
          _penalty_mat_center[find] = ArrayUtils.mult(_penalty_mat_center[find], _parms._scale[find]);
      }

      GLMModel model = new GLM(glmParam, _penalty_mat_center,  _gamColNamesCenter).trainModel().get();
      return model;
    }
    
    void fillOutGAMModel(GLMModel glm, GAMModel model, DataInfo dinfo) {
      model._gamColNamesNoCentering = _gamColNames;  // copy over gam column names
      model._gamColNames = _gamColNamesCenter;
      model._output._zTranspose = _zTranspose;
      model._gamFrameKeysCenter = _gamFrameKeysCenter;
      model._nclass = _nclass;
      model._output._binvD = _binvD;
      model._output._knots = _knots;
      model._output._numKnots = _numKnots;
      if (_parms._keep_gam_cols)
        model._output._gam_transformed_center_key = model._output._gamTransformedTrainCenter.toString();
      if (_parms._savePenaltyMat) {
        model._output._penaltyMatrices_center = _penalty_mat_center;
        model._output._penaltyMatrices = _penalty_mat;
      }
      copyGLMCoeffs(glm, model, dinfo);  // copy over coefficient names and generate coefficients as beta = z*GLM_beta
      copyGLMtoGAMModel(model, glm);  // copy over fields from glm model to gam model
    }
    
    private void copyGLMtoGAMModel(GAMModel model, GLMModel glmModel) {
      model._output._glm_best_lamda_value = glmModel._output.bestSubmodel().lambda_value; // exposed best lambda used
      model._output._glm_training_metrics = glmModel._output._training_metrics;
      if (valid() != null)
        model._output._glm_validation_metrics = glmModel._output._validation_metrics;
      model._output._glm_scoring_history = model.copyTwoDimTable(glmModel._output._scoring_history);
      model._output._glm_model_summary = model.copyTwoDimTable(glmModel._output._model_summary);
      if (_parms._family == multinomial || _parms._family == ordinal) {
        model._output._coefficients_table = model.genCoefficientTableMultinomial(new String[]{"Coefficients",
                        "Standardized Coefficients"}, model._output._model_beta_multinomial,
                model._output._standardized_model_beta_multinomial, model._output._coefficient_names,"GAM Coefficients");
        model._output._coefficients_table_no_centering = model.genCoefficientTableMultinomial(new String[]{"coefficients " +
                        "no centering", "standardized coefficients no centering"}, 
                model._output._model_beta_multinomial_no_centering, model._output._standardized_model_beta_multinomial_no_centering, 
                model._output._coefficient_names_no_centering,"GAM Coefficients No Centering");
        model._output._standardized_coefficient_magnitudes = model.genCoefficientMagTableMultinomial(new String[]{"coefficients", "signs"},
                model._output._standardized_model_beta_multinomial, model._output._coefficient_names, "standardized coefficients magnitude");
      } else{
        model._output._coefficients_table = model.genCoefficientTable(new String[]{"coefficients", "standardized coefficients"}, model._output._model_beta,
                model._output._standardized_model_beta, model._output._coefficient_names, "GAM Coefficients");
        model._output._coefficients_table_no_centering = model.genCoefficientTable(new String[]{"coefficients no centering", 
                        "standardized coefficients no centering"}, model._output._model_beta_no_centering,
                model._output._standardized_model_beta_no_centering,
                model._output._coefficient_names_no_centering, 
                "GAM Coefficients No Centering");
        model._output._standardized_coefficient_magnitudes = model.genCoefficientMagTable(new String[]{"coefficients", "signs"}, 
                model._output._standardized_model_beta, model._output._coefficient_names, "standardized coefficients magnitude");
      }
      
      if (_parms._compute_p_values) {
        model._output._glm_zvalues = glmModel._output.zValues().clone();
        model._output._glm_pvalues = glmModel._output.pValues().clone();
        model._output._glm_stdErr = glmModel._output.stdErr().clone();
        model._output._glm_vcov = glmModel._output.vcov().clone();
      }
      model._output._glm_dispersion = glmModel._output.dispersion();
      model._nobs = glmModel._nobs;
      model._nullDOF = glmModel._nullDOF;
      model._ymu = new double[glmModel._ymu.length];
      model._rank = glmModel._output.bestSubmodel().rank();
      model._ymu = new double[glmModel._ymu.length];
      System.arraycopy(glmModel._ymu, 0, model._ymu, 0, glmModel._ymu.length);
    }
    
    void copyGLMCoeffs(GLMModel glm, GAMModel model, DataInfo dinfo) {
      int totCoefNumsNoCenter = dinfo.fullN()+1+_parms._gam_columns.length;
      model._output._coefficient_names_no_centering = new String[totCoefNumsNoCenter]; // copy coefficient names from GLM to GAM
      int gamNumStart = copyGLMCoeffNames2GAMCoeffNames(model, glm, dinfo);
      copyGLMCoeffs2GAMCoeffs(model, glm, dinfo, _parms._family, gamNumStart, _parms._standardize, _nclass); // obtain beta without centering
      // copy over GLM coefficients
      int glmCoeffLen = glm._output._coefficient_names.length;
      model._output._coefficient_names = new String[glmCoeffLen];
      System.arraycopy(glm._output._coefficient_names, 0, model._output._coefficient_names, 0,
              glmCoeffLen);
      if (_parms._family == multinomial || _parms._family == ordinal) {
        double[][] model_beta_multinomial = glm._output.get_global_beta_multinomial();
        double[][] standardized_model_beta_multinomial = glm._output.getNormBetaMultinomial();
        model._output._model_beta_multinomial = new double[_nclass][glmCoeffLen];
        model._output._standardized_model_beta_multinomial = new double[_nclass][glmCoeffLen];
        for (int classInd = 0; classInd < _nclass; classInd++) {
          System.arraycopy(model_beta_multinomial[classInd], 0, model._output._model_beta_multinomial[classInd],
                  0, glmCoeffLen);
          System.arraycopy(standardized_model_beta_multinomial[classInd], 0, 
                  model._output._standardized_model_beta_multinomial[classInd], 0, glmCoeffLen);
        }
      } else {
        model._output._model_beta = new double[glmCoeffLen];
        model._output._standardized_model_beta = new double[glmCoeffLen];
        System.arraycopy(glm._output.beta(), 0, model._output._model_beta, 0, glmCoeffLen);
        System.arraycopy(glm._output.getNormBeta(), 0, model._output._standardized_model_beta, 0, 
                glmCoeffLen);
      }
    }
  }
}

package hex.gam;

import hex.*;
import hex.deeplearning.DeepLearningModel;
import hex.gam.MatrixFrameUtils.AddGamColumns;
import hex.glm.GLM;
import hex.glm.GLMModel;
import hex.glm.GLMModel.GLMParameters.Family;
import hex.glm.GLMModel.GLMParameters.GLMType;
import hex.glm.GLMModel.GLMParameters.Link;
import hex.glm.GLMModel.GLMParameters.Solver;
import water.*;
import water.fvec.Chunk;
import water.fvec.Frame;
import water.fvec.NewChunk;
import water.fvec.Vec;
import water.udf.CFuncRef;
import water.util.ArrayUtils;
import water.util.FrameUtils;
import water.util.Log;
import water.util.TwoDimTable;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Comparator;

import static hex.gam.MatrixFrameUtils.GamUtils.equalColNames;
import static hex.gam.MatrixFrameUtils.GamUtils.sortCoeffMags;
import static hex.glm.GLMModel.GLMParameters.MissingValuesHandling;

public class GAMModel extends Model<GAMModel, GAMModel.GAMParameters, GAMModel.GAMModelOutput> {
  public String[][] _gamColNamesNoCentering; // store column names only for GAM columns
  public String[][] _gamColNames; // store column names only for GAM columns after decentering
  public Key<Frame>[] _gamFrameKeysCenter;
  public int _nclass; // 2 for binomial, > 2 for multinomial and ordinal
  public double[] _ymu;
  public long _nobs;
  public long _nullDOF;
  public int _rank;

  @Override public String[] makeScoringNames() {
    String[] names = super.makeScoringNames();
    if (_output._glm_vcov != null)
      names = ArrayUtils.append(names, "StdErr");
    return names;
  }
  
  @Override public ModelMetrics.MetricBuilder makeMetricBuilder(String[] domain) {
    if (domain==null && (_parms._family==Family.binomial || _parms._family==Family.quasibinomial || 
            _parms._family==Family.negativebinomial || _parms._family==Family.fractionalbinomial))
      domain = new String[]{"0","1"};
    GLMModel.GLMWeightsFun glmf = new GLMModel.GLMWeightsFun(_parms._family, _parms._link, _parms._tweedie_variance_power,
            _parms._tweedie_link_power, _parms._theta);
    return new MetricBuilderGAM(domain, _ymu, glmf, _rank, true, _parms._intercept, _nclass);
  }

  public GAMModel(Key<GAMModel> selfKey, GAMParameters parms, GAMModelOutput output) {
    super(selfKey, parms, output);
    assert(Arrays.equals(_key._kb, selfKey._kb));
  }

  public TwoDimTable copyTwoDimTable(TwoDimTable table) {
    String[] rowHeaders = table.getRowHeaders();
    String[] colTypes = table.getColTypes();
    int tableSize = rowHeaders.length;
    int colSize = colTypes.length;
    TwoDimTable tableCopy = new TwoDimTable("glm scoring history", "",
            rowHeaders, table.getColHeaders(), colTypes, table.getColFormats(),
            "names");
    for (int rowIndex = 0; rowIndex < tableSize; rowIndex++)  {
      for (int colIndex = 0; colIndex < colSize; colIndex++) {
        tableCopy.set(rowIndex, colIndex,table.get(rowIndex, colIndex));
      }
    }
    return tableCopy;
  }
  
  TwoDimTable genCoefficientTable(String[] colHeaders, double[] coefficients, double[] coefficientsStand,
                                  String[] coefficientNames, String tableHeader) {
    String[] colTypes = new String[]{ "double", "double"};
    String[] colFormat = new String[]{"%5f", "%5f"};
    int nCoeff = coefficients.length;
    String[] coeffNames = new String[nCoeff];
    System.arraycopy(coefficientNames, 0, coeffNames, 0, nCoeff);
    
    Log.info("genCoefficientMagTableMultinomial", String.format("coemffNames length: %d.  coefficients " +
            "length: %d, coeffSigns length: %d", coeffNames.length, coefficients.length, coefficientsStand.length));
    
    TwoDimTable table = new TwoDimTable(tableHeader, "", coeffNames, colHeaders, colTypes, colFormat,
            "names");
    fillUpCoeffs(coeffNames, coefficients, coefficientsStand, table, 0);
    return table;
  }

  TwoDimTable genCoefficientMagTableMultinomial(String[] colHeaders, double[][] coefficients,
                                     String[] coefficientNames, String tableHeader) {
    String[] colTypes = new String[]{ "double", "string"};
    String[] colFormat = new String[]{"%5f", ""};
    int nCoeff = coefficients[0].length;
    int nClass = coefficients.length;
    String[] coeffNames = new String[nCoeff - 1];
    String[] coeffNames2 = new String[coeffNames.length];
    double[] coeffMags = new double[coeffNames.length];
    double[] coeffMags2 = new double[coeffNames.length];
    String[] coeffSigns = new String[coeffNames.length];

    Log.info("genCoefficientMagTableMultinomial", String.format("coeffNames length: %d.  coeffMags " +
            "length: %d, coeffSigns length: %d", coeffNames.length, coeffMags.length, coeffSigns.length));
    
    int countIndex = 0;
    for (int index = 0; index < nCoeff; index++) {
      if (coefficientNames[index].equals("Intercept")) {
        for (int classInd = 0; classInd < nClass; classInd++) {
          coeffMags[countIndex] += Math.abs(coefficients[classInd][index]);
        }
        coeffNames[countIndex] = coefficientNames[index];
        coeffSigns[countIndex] = "POS";   // assign all signs to positive for multinomial
        countIndex++;
      }
    }
    // sort in descending order of the magnitudes
    Integer[] indices = new Integer[nCoeff-1];
    Log.info("genCoefficientMagTableMultinomial", String.format("index length: %d. ", indices.length));
    for (int index = 0; index < indices.length; index++)
      indices[index] = index;

    Arrays.sort(indices, new Comparator<Integer>() {
      @Override
      public int compare(Integer o1, Integer o2) {
        if(coeffMags[o1] < coeffMags[o2]) return +1;
        if(coeffMags[o1] > coeffMags[o2]) return -1;
        return 0;
      }
    });
    
    // reorder names and coeffMags with indices
    for (int index = 0; index < coeffMags.length; index++) {
      coeffMags2[index] = coeffMags[indices[index]];
      coeffNames2[index] = coeffNames[indices[index]];
    }
    
    Log.info("genCoefficientMagTableMultinomial", String.format("coeffNames2 length: %d.  coeffMags2 " +
            "length: %d, coeffSigns length: %d", coeffNames2.length, coeffMags2.length, coeffSigns.length));
    
    TwoDimTable table = new TwoDimTable(tableHeader, "Standardized Coefficient Magnitutes", coeffNames2, colHeaders, colTypes, colFormat,
            "names");
    fillUpCoeffsMag(coeffNames2, coeffMags2, coeffSigns, table, 0);
    return table;
  }

  TwoDimTable genCoefficientMagTable(String[] colHeaders, double[] coefficients,
                                  String[] coefficientNames, String tableHeader) {
    String[] colTypes = new String[]{ "double", "string"};
    String[] colFormat = new String[]{"%5f", ""};
    int nCoeff = coefficients.length;
    String[] coeffNames = new String[nCoeff-1];
    double[] coeffMags = new double[nCoeff-1]; // skip over intercepts
    String[] coeffSigns = new String[nCoeff-1];
    int countMagIndex = 0;
    for (int index = 0; index < nCoeff; index++) {
      if (!coefficientNames[index].equals("Intercept")) {
        coeffMags[countMagIndex] = Math.abs(coefficients[index]);
        coeffSigns[countMagIndex] = coefficients[index] > 0 ? "POS" : "NEG";
        coeffNames[countMagIndex++] = coefficientNames[index];
      }
    }
    Integer[] indices = sortCoeffMags(coeffMags.length, coeffMags); // sort magnitude indices in decreasing magnitude
    String[] names2 = new String[coeffNames.length];
    double[] mag2 = new double[coeffNames.length];
    String[] sign2 = new String[coeffNames.length];
    for (int i = 0; i < coeffNames.length; ++i) {
      names2[i] = coeffNames[indices[i]];
      mag2[i] = coeffMags[indices[i]];
      sign2[i] = coeffSigns[indices[i]];
    }
    Log.info("genCoefficientMagTableMultinomial", String.format("coeffNames length: %d.  coeffMags " +
            "length: %d, coeffSigns length: %d", coeffNames.length, coeffMags.length, coeffSigns.length));

    TwoDimTable table = new TwoDimTable(tableHeader, "", names2, colHeaders, colTypes, colFormat,
            "names");
    fillUpCoeffsMag(names2, mag2, sign2, table, 0);
    return table;
  }
  


  private void fillUpCoeffsMag(String[] names, double[] coeffMags, String[] coeffSigns, TwoDimTable tdt, int rowStart) {
    int arrLength = coeffMags.length+rowStart;
    int arrCounter=0;
    for (int i=rowStart; i<arrLength; i++) {
      tdt.set(i, 0, coeffMags[arrCounter]);
      tdt.set(i, 1, coeffSigns[arrCounter]);
      arrCounter++;
    }
  }
  
  TwoDimTable genCoefficientTableMultinomial(String[] colHeaders, double[][] coefficients, double[][] coefficients_stand, 
                                                     String[] coefficientNames, String tableHeader) {
    String[] colTypes = new String[]{"double", "double"};
    String[] colFormat = new String[]{"%5f", "%5f"};
    int nCoeff = coefficients[0].length;
    int nclass = coefficients.length;
    int totCoeff = nCoeff*nclass;
    String[] coeffNames = new String[totCoeff];

    int coeffCounter=0;
    for (int classInd=0; classInd < nclass; classInd++){
      for (int ind=0; ind < nCoeff; ind++) {
        coeffNames[coeffCounter++] = coefficientNames[ind]+"_class_"+classInd;
      }
    }
    TwoDimTable table = new TwoDimTable(tableHeader, "", coeffNames, colHeaders, colTypes, colFormat,
            "names");
    for (int classInd=0; classInd<nclass; classInd++)
      fillUpCoeffs(coeffNames, coefficients[classInd], coefficients_stand[classInd], table, classInd*nCoeff);
    return table;
  }

  private void fillUpCoeffs(String[] names, double[] coeffValues, double[] coeffValuesStand, TwoDimTable tdt, int rowStart) {
    int arrLength = coeffValues.length+rowStart;
    int arrCounter=0;
    for (int i=rowStart; i<arrLength; i++) {
      tdt.set(i, 0, coeffValues[arrCounter]);
      tdt.set(i, 1, coeffValuesStand[arrCounter]);
      arrCounter++;
    }
  }

  @SuppressWarnings("WeakerAccess")
  public static class GAMParameters extends Model.Parameters {
    // the following parameters will be passed to GLM algos
    public boolean _standardize = false; // pass to GLM algo
    public Family _family;
    public Link _link;
    public Solver _solver = Solver.AUTO;
    public double _tweedie_variance_power;
    public double _tweedie_link_power;
    public double _theta; // 1/k and is used by negative binomial distribution only
    public double _invTheta;
    public double [] _alpha;
    public double [] _lambda;
    public Serializable _missing_values_handling = MissingValuesHandling.MeanImputation;
    public double _prior = -1;
    public boolean _lambda_search = false;
    public int _nlambdas = -1;
    public boolean _non_negative = false;
    public boolean _exactLambdas = false;
    public double _lambda_min_ratio = -1; // special
    public boolean _use_all_factor_levels = false;
    public int _max_iterations = -1;
    public boolean _intercept = true;
    public double _beta_epsilon = 1e-4;
    public double _objective_epsilon = -1;
    public double _gradient_epsilon = -1;
    public double _obj_reg = -1;
    public boolean _compute_p_values = false;
    public boolean _remove_collinear_columns = false;
    public String[] _interactions=null;
    public StringPair[] _interaction_pairs=null;
    public boolean _early_stopping = true;
    public Key<Frame> _beta_constraints = null;
    public Key<Frame> _plug_values = null;
    // internal parameter, handle with care. GLM will stop when there is more than this number of active predictors (after strong rule screening)
    public int _max_active_predictors = -1;
    public boolean _stdOverride; // standardization override by beta constraints

    // the following parameters are for GAM
    public int[] _num_knots; // array storing number of knots per basis function
    public double[][] _knots;// store knots for each gam column specified in _gam_X
    public String[] _knot_ids;  // store frame keys that contain knots location for each gam column in gam_X;
    public String[] _gam_columns; // array storing which predictor columns are needed
   // public BSType _bs; // choose spline function for gam column
    public int[] _bs; // choose spline function for gam column, 0 = cr
    public double[] _scale;  // array storing scaling values to control wriggliness of fit
    public GLMType _glmType = GLMType.gam; // internal parameter
    public boolean _saveZMatrix = false;  // if asserted will save Z matrix
    public boolean _keep_gam_cols = false;  // if true will save the keys to gam Columns only
    public boolean _savePenaltyMat = false; // if true will save penalty matrices as tripple array

    public String algoName() { return "GAM"; }
    public String fullName() { return "General Additive Model"; }
    public String javaName() { return GAMModel.class.getName(); }

    @Override
    public long progressUnits() {
      return 1;
    }

    public long _seed = -1;

    public enum BSType {
      cr  // will support more in the future
    }

    public InteractionSpec interactionSpec() {
      return InteractionSpec.create(_interactions, _interaction_pairs);
    }
    
    public MissingValuesHandling missingValuesHandling() {
      if (_missing_values_handling instanceof MissingValuesHandling)
        return (MissingValuesHandling) _missing_values_handling;
      assert _missing_values_handling instanceof DeepLearningModel.DeepLearningParameters.MissingValuesHandling;
      switch ((DeepLearningModel.DeepLearningParameters.MissingValuesHandling) _missing_values_handling) {
        case MeanImputation:
          return MissingValuesHandling.MeanImputation;
        case Skip:
          return MissingValuesHandling.Skip;
        default:
          throw new IllegalStateException("Unsupported missing values handling value: " + _missing_values_handling);
      }
    }

    public DataInfo.Imputer makeImputer() {
      if (missingValuesHandling() == MissingValuesHandling.PlugValues) {
        if (_plug_values == null || _plug_values.get() == null) {
          throw new IllegalStateException("Plug values frame needs to be specified when Missing Value Handling = PlugValues.");
        }
        return new GLM.PlugValuesImputer(_plug_values.get());
      } else { // mean/mode imputation and skip (even skip needs an imputer right now! PUBDEV-6809)
        return new DataInfo.MeanImputer();
      }
    }

    public final static double linkInv(double x, Link link, double tweedie_link_power) {
      switch(link) {
//        case multinomial: // should not be used
        case identity:
          return x;
        case ologlog:
          return 1.0-Math.exp(-1.0*Math.exp(x));
        case ologit:
        case logit:
          return 1.0 / (Math.exp(-x) + 1.0);
        case log:
          return Math.exp(x);
        case inverse:
          double xx = (x < 0) ? Math.min(-1e-5, x) : Math.max(1e-5, x);
          return 1.0 / xx;
        case tweedie:
          return tweedie_link_power == 0
                  ?Math.max(2e-16,Math.exp(x))
                  :Math.pow(x, 1/ tweedie_link_power);
        default:
          throw new RuntimeException("unexpected link function  " + link.toString());
      }
    }
  }

  public static class GAMModelOutput extends Model.Output {
    public String[] _coefficient_names_no_centering;
    public String[] _coefficient_names;    
    public TwoDimTable _glm_model_summary;
    public ModelMetrics _glm_training_metrics;
    public ModelMetrics _glm_validation_metrics;
    public ModelMetrics _glm_cross_validation_metrics;
    public double _glm_dispersion;
    public double[] _glm_zvalues;
    public double[] _glm_pvalues;
    public double[][] _glm_vcov;
    public double[] _glm_stdErr;
    public double _glm_best_lamda_value;
    public TwoDimTable _glm_scoring_history;
    public TwoDimTable _coefficients_table;
    public TwoDimTable _coefficients_table_no_centering;
    public TwoDimTable _standardized_coefficient_magnitudes;
    public int _best_lambda_idx; // lambda which minimizes deviance on validation (if provided) or train (if not)
    public int _lambda_1se = -1; // lambda_best + sd(lambda); only applicable if running lambda search with nfold
    public int _selected_lambda_idx; // lambda which minimizes deviance on validation (if provided) or train (if not)
    public double[] _model_beta_no_centering; // coefficients generated during model training
    public double[] _standardized_model_beta_no_centering; // standardized coefficients generated during model training
    public double[] _model_beta; // coefficients generated during model training
    public double[] _standardized_model_beta; // standardized coefficients generated during model training
    public double[][] _model_beta_multinomial_no_centering;  // store multinomial coefficients during model training
    public double[][] _standardized_model_beta_multinomial_no_centering;  // store standardized multinomial coefficients during model training
    public double[][] _model_beta_multinomial;  // store multinomial coefficients during model training
    public double[][] _standardized_model_beta_multinomial;  // store standardized multinomial coefficients during model training
    private double[] _zvalues;
    private double _dispersion;
    private boolean _dispersionEstimated;
    public double[][][] _zTranspose; // Z matrix for de-centralization, can be null
    public double[][][] _penaltyMatrices_center; // stores t(Z)*t(D)*Binv*D*Z and can be null
    public double[][][] _penaltyMatrices;          // store t(D)*Binv*D and can be null
    public double[][][] _binvD; // store BinvD for each gam column specified for scoring
    public double[][] _knots; // store knots location for each gam column
    public int[] _numKnots;  // store number of knots per gam column
    public Key<Frame> _gamTransformedTrainCenter;  // contain key of predictors, all gam columns centered
    public DataInfo _dinfo;
    public String[] _responseDomains;
    public String _gam_transformed_center_key;
    final Family _family;
    
    public double dispersion(){ return _dispersion;}

    @Override
    public int nclasses() {
      if (_family == Family.multinomial || _family == Family.ordinal)
        return super.nclasses();
      if (Family.binomial == _family || Family.quasibinomial == _family
              || Family.fractionalbinomial == _family)
        return 2;
      return 1;
    }

    /** Names of levels for a categorical response column. */
    @Override
    public String[] classNames() {
      if (_family == Family.fractionalbinomial) {
        return new String[]{"0", "1"};
      } else 
        return super.classNames();
    }

    public GAMModelOutput(GAM b, Frame adaptr, DataInfo dinfo) {
      super(b, adaptr);
      _dinfo = dinfo;
      _domains = dinfo._adaptedFrame.domains(); // get domain of dataset predictors
      _responseDomains = dinfo._adaptedFrame.lastVec().domain();
      _family = b._parms._family;
    }

    @Override public ModelCategory getModelCategory() {
      switch (_family) {
        case binomial: return ModelCategory.Binomial;
        case multinomial: return ModelCategory.Multinomial;
        case ordinal: return ModelCategory.Ordinal;
        default: return ModelCategory.Regression;
      }
    }
  }

  /**
   * This method will massage the input training frame such that it can be used for scoring for a GAM model.
   *
   * @param test Testing Frame, updated in-place
   * @param expensive Try hard to adapt; this might involve the creation of
   *  whole Vecs and thus get expensive.  If {@code false}, then only adapt if
   *  no warnings and errors; otherwise just the messages are produced.
   *  Created Vecs have to be deleted by the caller (e.g. Scope.enter/exit).
   * @param computeMetrics
   * @return
   */
  @Override
  public String[] adaptTestForTrain(Frame test, boolean expensive, boolean computeMetrics) {
    // compare column names with test frame.  If equal, call adaptTestForTrain.  Otherwise, need to adapt it first
    String[] testNames = test.names();
    if (!equalColNames(testNames, _output._dinfo._adaptedFrame.names(), _parms._response_column)) {  // shallow check: column number, column names only
      Frame adptedF = cleanUpInputFrame(test, testNames); // column names here need to be in same sequence of dinfo._adaptedFrame
      int testNumCols = test.numCols();
      for (int index = 0; index < testNumCols; index++)
        test.remove(0);
      int adaptNumCols = adptedF.numCols();
      for (int index = 0; index < adaptNumCols; index++)
        test.add(adptedF.name(index), adptedF.vec(index));
      return super.adaptTestForTrain(test, expensive, computeMetrics);
    }
    return super.adaptTestForTrain(test, expensive, computeMetrics);
  }

  public Frame cleanUpInputFrame(Frame test, String[] testNames) {
    Frame adptedF = new Frame(Key.make(), test._names.clone(), test.vecs().clone()); // clone test dataset
    int numGamCols = _output._numKnots.length;
    Vec[] gamCols = new Vec[numGamCols];
    for (int vind=0; vind<numGamCols; vind++)
      gamCols[vind] = adptedF.vec(_parms._gam_columns[vind]).clone();
    Frame onlyGamCols = new Frame(_parms._gam_columns, gamCols);
    AddGamColumns genGamCols = new AddGamColumns(_output._binvD, _output._zTranspose, _output._knots, 
            _output._numKnots, _output._dinfo, onlyGamCols);
    genGamCols.doAll(genGamCols._gamCols2Add, Vec.T_NUM, onlyGamCols);
    String[] gamColsNames = new String[genGamCols._gamCols2Add];
    int offset = 0;
    for (int ind=0; ind<genGamCols._numGAMcols; ind++) {
      System.arraycopy(_gamColNames[ind], 0, gamColsNames, offset, _gamColNames[ind].length);
      offset+= _gamColNames[ind].length;
    }
    Frame oneAugmentedColumn = genGamCols.outputFrame(Key.make(), gamColsNames, null);
    if (_parms._ignored_columns != null) {  // remove ignored columns
      for (String iname:_parms._ignored_columns) {
        if (ArrayUtils.contains(testNames, iname)) {
          adptedF.remove(iname);
        }
      }
    }
    int numCols = adptedF.numCols();  // remove constant or bad frames.
    for (int vInd=0; vInd<numCols; vInd++) {
      Vec v = adptedF.vec(vInd);
      if ((_parms._ignore_const_cols &&  v.isConst()) || v.isBad())
        adptedF.remove(vInd);
    }
    Vec respV = null;
    if (ArrayUtils.contains(testNames, _parms._response_column))
      respV = adptedF.remove(_parms._response_column);
    adptedF.add(oneAugmentedColumn.names(), oneAugmentedColumn.removeAll());
    Scope.track(oneAugmentedColumn);
    
    if (respV != null)
      adptedF.add(_parms._response_column, respV);
    return adptedF;
  }

  @Override
  protected Frame predictScoreImpl(Frame fr, Frame adaptFrm, String destination_key, Job j, boolean computeMetrics,
                                   CFuncRef customMetricFunc) {
    String[] predictNames = makeScoringNames();
    String[][] domains = new String[predictNames.length][];
    GAMScore gs = makeScoringTask(adaptFrm, j, computeMetrics);
    gs.doAll(predictNames.length, Vec.T_NUM, gs._dinfo._adaptedFrame);
    if (gs._computeMetrics)
      gs._mb.makeModelMetrics(this, fr, adaptFrm, gs.outputFrame());
    domains[0] = gs._predDomains;
    return gs.outputFrame(Key.make(destination_key), predictNames, domains);  // place holder
  }
  
  private GAMScore makeScoringTask(Frame adaptFrm, Job j, boolean computeMetrics) {
    int responseId = adaptFrm.find(_output.responseName());
    if(responseId > -1 && adaptFrm.vec(responseId).isBad()) { // remove inserted invalid response
      adaptFrm = new Frame(adaptFrm.names(),adaptFrm.vecs());
      adaptFrm.remove(responseId);
    }
// Build up the names & domains.
    final boolean detectedComputeMetrics = computeMetrics && (adaptFrm.vec(_output.responseName()) != null && !adaptFrm.vec(_output.responseName()).isBad());
    String [] domain = _output.nclasses()<=1 ? null : (!detectedComputeMetrics ? _output._domains[_output._domains.length-1] : adaptFrm.lastVec().domain());
// Score the dataset, building the class distribution & predictions
    return new GAMScore(j, this, _output._dinfo.scoringInfo(_output._names,adaptFrm),domain,detectedComputeMetrics);
  }

  private class GAMScore extends MRTask<GAMScore> {
    private DataInfo _dinfo;
    private double[] _coeffs;
    private double[][] _coeffs_multinomial;
    private int _nclass;
    private boolean _computeMetrics;
    final Job _j;
    Family _family;
    private transient double[] _eta;  // store eta calculation
    private String[] _predDomains;
    final GAMModel _m;
    private final double _defaultThreshold;
    private int _lastClass;
    ModelMetrics.MetricBuilder _mb;
    final boolean _sparse;
    private transient double[][] _vcov;
    private transient double[] _tmp;


    private GAMScore(Job j, GAMModel m, DataInfo dinfo, String[] domain, boolean computeMetrics) {
      _j = j;
      _m = m;
      _computeMetrics = computeMetrics;
      _sparse = FrameUtils.sparseRatio(dinfo._adaptedFrame) < .5;
      _predDomains = domain;
      _m._parms = m._parms;
      _nclass = m._output.nclasses();
      if(_m._parms._family == GLMModel.GLMParameters.Family.multinomial ||
              _m._parms._family == GLMModel.GLMParameters.Family.ordinal){
        _coeffs = null;
        _coeffs_multinomial = m._output._model_beta_multinomial;
      } else {
        double [] beta = m._output._model_beta;
        int [] ids = new int[beta.length-1];
        int k = 0;
        for(int i = 0; i < beta.length-1; ++i){ // pick out beta that is not zero in ids
          if(beta[i] != 0) ids[k++] = i;
        }
        if(k < beta.length-1) {
          ids = Arrays.copyOf(ids,k);
          dinfo = dinfo.filterExpandedColumns(ids);
          double [] beta2 = MemoryManager.malloc8d(ids.length+1);
          int l = 0;
          for(int x:ids)
            beta2[l++] = beta[x];
          beta2[l] = beta[beta.length-1];
          beta = beta2;
        }
        _coeffs_multinomial = null;
        _coeffs = beta;
      }
      _dinfo = dinfo;
      _dinfo._valid = true; // marking dinfo as validation data set disables an assert on unseen levels (which should not happen in train)
      _defaultThreshold = m.defaultThreshold();
      _family = m._parms._family;
      _lastClass = _nclass-1;
    }

    @Override
    public void map(Chunk[]chks, NewChunk[] nc) {
      if (isCancelled() || _j != null && _j.stop_requested()) return;
      if (_family.equals(Family.ordinal)||_family.equals(Family.multinomial))
        _eta = MemoryManager.malloc8d(_nclass);
      _vcov = _m._output._glm_vcov;
      if (_vcov != null)
        _tmp = MemoryManager.malloc8d(_vcov.length);
      int numPredVals = _nclass<=1?1:_nclass+1; // number of predictor values expected.
      double[] predictVals = MemoryManager.malloc8d(numPredVals);
      float[] trueResponse = null;

      if (_computeMetrics) {
        _mb = _m.makeMetricBuilder(_predDomains);
        trueResponse = new float[1];
      }
      DataInfo.Row r = _dinfo.newDenseRow();
      int chkLen = chks[0]._len;
      for (int rid=0; rid<chkLen; rid++) {  // extract each row
        _dinfo.extractDenseRow(chks, rid, r);
        processRow(r, predictVals, nc, numPredVals);
        if (_computeMetrics && !r.response_bad) {
          trueResponse[0] = (float) r.response[0];
          _mb.perRow(predictVals, trueResponse, r.weight, r.offset, _m);
        }
      }
      if (_j != null) _j.update(1);
    }

    private void processRow(DataInfo.Row r, double[] ps, NewChunk[] preds, int ncols) {
      if (r.predictors_bad)
        Arrays.fill(ps, Double.NaN);  // output NaN with bad predictor entries
      else if (r.weight == 0)
        Arrays.fill(ps, 0.0); // zero weight entries got 0 too
      switch (_family) {
        case multinomial: ps = scoreMultinomialRow(r, r.offset, ps); break;
        case ordinal: ps = scoreOrdinalRow(r, r.offset, ps); break;
        default: ps = scoreRow(r, r.offset, ps); break;
      }
      for (int predCol=0; predCol < ncols; predCol++) { // write prediction to NewChunk
        preds[predCol].addNum(ps[predCol]);
      }
      if (_vcov != null) 
        preds[ncols].addNum(Math.sqrt(r.innerProduct(r.mtrxMul(_vcov, _tmp))));
    }

    public double[] scoreRow(DataInfo.Row r, double offset, double[] preds) {
      double mu = _m._parms.linkInv(r.innerProduct(_coeffs) + offset, _m._parms._link,
              _m._parms._tweedie_link_power);
      if (_m._parms._family == GLMModel.GLMParameters.Family.binomial ||
              _m._parms._family == GLMModel.GLMParameters.Family.quasibinomial ||
      _m._parms._family == Family.negativebinomial || _m._parms._family == Family.fractionalbinomial) { // threshold for prediction
        preds[0] = mu >= _defaultThreshold?1:0;
        preds[1] = 1.0 - mu; // class 0
        preds[2] = mu; // class 1
      } else
        preds[0] = mu;
      return preds;
    }

    public double[] scoreOrdinalRow(DataInfo.Row r, double offset, double[] preds) {
      final double[][] bm = _coeffs_multinomial;
      Arrays.fill(preds,0); // initialize to small number
      preds[0] = _lastClass;  // initialize to last class by default here
      double previousCDF = 0.0;
      for (int cInd = 0; cInd < _lastClass; cInd++) { // classify row and calculate PDF of each class
        double eta = r.innerProduct(bm[cInd]) + offset;
        double currCDF = 1.0 / (1 + Math.exp(-eta));
        preds[cInd + 1] = currCDF - previousCDF;
        previousCDF = currCDF;

        if (eta > 0) { // found the correct class
          preds[0] = cInd;
          break;
        }
      }
      for (int cInd = (int) preds[0] + 1; cInd < _lastClass; cInd++) {  // continue PDF calculation
        double currCDF = 1.0 / (1 + Math.exp(-r.innerProduct(bm[cInd]) + offset));
        preds[cInd + 1] = currCDF - previousCDF;
        previousCDF = currCDF;

      }
      preds[_nclass] = 1-previousCDF;
      return preds;
    }

    public double[] scoreMultinomialRow(DataInfo.Row r, double offset, double[] preds) {
      double[] eta = _eta;
      final double[][] bm = _coeffs_multinomial;
      double sumExp = 0;
      double maxRow = Double.NEGATIVE_INFINITY;
      for (int c = 0; c < bm.length; ++c) {
        eta[c] = r.innerProduct(bm[c]) + offset;
        if(eta[c] > maxRow)
          maxRow = eta[c];
      }
      for (int c = 0; c < bm.length; ++c)
        sumExp += eta[c] = Math.exp(eta[c]-maxRow); // intercept
      sumExp = 1.0 / sumExp;
      for (int c = 0; c < bm.length; ++c)
        preds[c + 1] = eta[c] * sumExp;
      preds[0] = ArrayUtils.maxIndex(eta);
      return preds;
    }
    
    @Override 
    public void reduce(GAMScore other) {
      if (_mb !=null)
        _mb.reduce(other._mb);
    }
    
    @Override
    protected void postGlobal() {
      if (_mb != null)
        _mb.postGlobal();
    }
  }

  @Override
  public double[] score0(double[] data, double[] preds) {
    throw new UnsupportedOperationException("GAMModel.score0 should never be called");
  }

  @Override
  protected Futures remove_impl(Futures fs, boolean cascade) {
    Keyed.remove(_output._gamTransformedTrainCenter, fs, true);
    super.remove_impl(fs, cascade);
    return fs;
  }

  @Override protected AutoBuffer writeAll_impl(AutoBuffer ab) {
    if (_output._gamTransformedTrainCenter!=null)
      ab.putKey(_output._gamTransformedTrainCenter);
    return super.writeAll_impl(ab);
  }

  @Override protected Keyed readAll_impl(AutoBuffer ab, Futures fs) {
    return super.readAll_impl(ab, fs);
  }
}

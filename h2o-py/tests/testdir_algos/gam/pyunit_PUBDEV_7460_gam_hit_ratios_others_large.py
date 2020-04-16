from __future__ import division
from __future__ import print_function
import sys
sys.path.insert(1, "../../../")
import h2o
from tests import pyunit_utils
from h2o.estimators.gam import H2OGeneralizedAdditiveEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator


# In this test, we check to make sure model metrics are calculated correctly for GAM multinomial.
def test_gam_model_predict():
    covtype_df = h2o.import_file(pyunit_utils.locate("bigdata/laptop/covtype/covtype.full.csv"))

    #split the data as described above
    train, valid, test = covtype_df.split_frame([0.7, 0.15], seed=1234)

    #Prepare predictors and response columns
    covtype_X = covtype_df.col_names[:-1]     #last column is Cover_Type, our desired response variable 
    covtype_y = covtype_df.col_names[-1]

    glm_multi = H2OGeneralizedLinearEstimator(family='multinomial', solver='L_BFGS', lambda_search=True)
    glm_multi.train(covtype_X, covtype_y, training_frame=train, validation_frame=valid)
    glm_multi.summary()
    gam_multi = H2OGeneralizedAdditiveEstimator(family='multinomial', solver='L_BFGS', lambda_search=True, 
                                                gam_columns=["Hillshade_Noon","Hillshade_9am"], scale = [0.1,0.1])
    gam_multi.train(covtype_X, covtype_y, training_frame=train, validation_frame=valid)
    

    glm_multi.hit_ratio_table(valid=True)
    glm_multi.confusion_matrix(valid)
    gam_multi.hit_ratio_table(valid=True)
    gam_multi.confusion_matrix(valid)
    


if __name__ == "__main__":
    pyunit_utils.standalone_test(test_gam_model_predict)
else:
    test_gam_model_predict()

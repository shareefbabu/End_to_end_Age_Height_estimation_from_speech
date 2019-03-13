# End_to_end_Age_Height_estimation_speech
To predict the speaker height and age without training the model use the python file in 
scripts/cnn_fstat_2output_HT_age_predict.py

The input the script is
   1. path to your 60 MFCC features.
   2. filename for the predicted lables.
   3. No.of frames to be considered for predicting
   4. No.of frames to be shifted.
Usage: python cnn_fstat_2output_HT_age_predict sre_test_ht_age.list predicted_lables.out 150 75

These models are trained gender specific, change the trained
models and weight files corresponding to the gender which needs to be predicted 
(Eg: change male from female  and vice-versa)

python GenLabeledData.py
rm -r data
mv result_det data
python AugmentData.py
rm -r data
mv result_aug data
python SplitTrainVal.py
rm -r data
mv result_train_val data

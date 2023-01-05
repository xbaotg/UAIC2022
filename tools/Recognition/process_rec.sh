python GenLabeledData.py
rm -r data
mv result_train data
python GenerateSynthText.py
mv result_synth/images/* data/images/
cat result_synth/labels.txt data/labels.txt > temp.txt
mv temp.txt data/labels.txt
rm -r result_synth
python AugmentData.py
rm -r data
mv result_aug data
python SplitTrainVal.py
rm -r data
mv result_train_val data
